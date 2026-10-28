import torch
import os
import sys
import torch.nn as nn
import soundfile as sf
import numpy as np
import timeit

from torch.autograd.variable import *

import json
import yaml
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import logging
from utils.progressbar import progressbar as pb
from metrics import pesq_scorer, stoi_scorer
import fast_bss_eval
from trainers.modules import multich_stft
from librosa import istft
#from models.gtcrn_paris_binaural import GTCRN_PARIS_Binaural as GTCRN
from models.Paris_enhancement import GTCRN_PARIS
from models.gtcrn_binaural import GTCRN_Binaural
from models.gtcrnXiaobin import GTCRN
from models.gtcrn_paris_stream import StreamGTCRN
import argparse
from speechmos import dnsmos
import onnxruntime
from models.AuxIVA import AuxIVA_nn_stream, AuxIVA_stream_single
import cProfile, pstats, io

config_file = 'config.yaml'
ckpt_path = '/data/hdd0/jizhe.lu/exps/dcgtcrn/checkpoints/model_epoch68.tar'
test_audio_dir = '/data/hdd0/jizhe.lu/Simulated/test_results'
onnxfile = '/home/nis/jizhe.lu/gtcrn/stream/onnx_models/gtcrn_simple.onnx'

parser = argparse.ArgumentParser("MODEL TEST")

# General config
parser.add_argument('--name', type=str, required=True,
                    help='test name')

def IVA_stream_inference(input, out_i,  conv_cache, tra_cache, inter_cache, V_cache, onnx_session):
    '''
    input: numpy array, (1, 2, 257, 1, 2)
    out_i: numpy array, (1, 1, 257, 1, 2)
    conv_cache: numpy array, (2, 1, 32, 16, 65)
    tra_cache: numpy array, (2, 3, 1, 1, 32)
    inter_cache: numpy array, (2, 1, 65, 32)
    onnx_session: onnxruntime session
    '''
    if out_i is None:
        init_input = torch.randn((1, 1, 257, 1, 2))*0.1
    else:
        init_input =torch.from_numpy(out_i)
        # (1, 1, 257, 1)
    #print("init_input", init_input.mean(), init_input.std())
    
    input_ = torch.view_as_complex(torch.from_numpy(input))
    #r_iva = torch.view_as_complex(init_input)
    r_iva = torch.complex(init_input[...,0], torch.zeros_like(init_input[...,0]))  # (1, 1, 257, 1)
    # bss_out = AuxIVA_nn_stream(input_, r_iva)    # (1, 2, 257, 1) complex
    V_cache = torch.from_numpy(V_cache)
    bss_out, V_cache = AuxIVA_stream_single(input_, r_iva, V_cache)    # (1, 2, 257, 1) complex
    bss_out = torch.view_as_real(bss_out).numpy()  # (1, 2, 257, 1, 2)
    # print("bss_out", bss_out.mean(axis=(0,2,3,4)))
    
    out_i,  conv_cache, tra_cache, inter_cache \
                    = onnx_session.run([], {'spec': bss_out,# (1, 2, 257, 1, 2)
                        'conv_cache': conv_cache,
                        'tra_cache': tra_cache,
                        'inter_cache': inter_cache,
                        'init': init_input.numpy()})  
    return out_i,  conv_cache, tra_cache, inter_cache, bss_out, V_cache.numpy()

def normalize(sample):
    sample = sample.astype(np.float32)
    max_abs = np.max(np.abs(sample))
    if max_abs > 1.0:
        sample = sample / max_abs
    return sample

if __name__ == '__main__':
    #args
    arg = parser.parse_args()
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cpu")
    model = GTCRN_Binaural(**config["network"])
    # Model = GTCRN_IVA_PARIS
    # model = Model(**config["network"])
    
    # ckpt_dict = torch.load(ckpt_path, map_location="cpu")
    # model.load_state_dict(ckpt_dict["model"], strict=False)
    # model.cuda()
    #model = GTCRN_PARIS(**config["network"], params=config["network"]).eval()
    ckpt_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt_dict["model"])
    model.cuda()
    session = onnxruntime.InferenceSession(onnxfile, None, providers=['CPUExecutionProvider'])
    
    output_json = os.path.join(test_audio_dir, f"test_results_{os.path.basename(ckpt_path).split('.')[0]}.json")
    output_dir = os.path.join(test_audio_dir, arg.name)
    os.makedirs(output_dir, exist_ok=True)


    test_path = config['test_dataset']['md_file']
    with open(test_path, 'r') as f:
        test_md = json.load(f)

    cur_list = []
    for line in tqdm(test_md, desc="Loading test set"):
        cur_line = dict()
        cur_line['mixture'] = line['noisy']
        cur_line['target'] = line['RTS']
        cur_list.append(cur_line)
    
    #cur_list = cur_list[:500]
    
    ttbar = pb(0, len(cur_list), 20)
    ttbar.start()
    ttime, mtime, cnt = 0., 0., 0
    mix_sisdr_list=[]
    est_sisdr_list=[]
    mix_stoi_list=[]
    est_stoi_list=[]
    mix_pesq_list=[]
    est_pesq_list=[]
    
    est_mos = np.array([0.0, 0.0, 0.0, 0.0])
    mix_mos = np.array([0.0, 0.0, 0.0, 0.0])
    
    results = []
    pesq = pesq_scorer().cuda()
    stoi = stoi_scorer().cuda()
    #model.validate_reset_buffer()

    for test_sample in cur_list:
        cnt += 1
        info = dict()
        start = timeit.default_timer()
        mix_wav_path = test_sample['mixture']
        ref_wav_path = test_sample['target']

        mix,_ = sf.read(mix_wav_path)   # dual channel
        ref,_ = sf.read(ref_wav_path)   # single
        #mix = torch.FloatTensor(mix.astype('float32'))
        #ref = torch.FloatTensor(ref.astype('float32'))
        x = torch.from_numpy(mix)
        x = multich_stft(x.T.unsqueeze(0))
        
        
        
        clean_prior = torch.stft(torch.from_numpy(ref), n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512).pow(0.5), return_complex=True).unsqueeze(0)  # (1, 257, T)
        clean_prior = clean_prior.unsqueeze(1)
        
         # (1, 2, 257, 572)  # B, C, F, T
        #x = torch.view_as_real(x).float()
        # b, c, t
        # nfft = config['STFT']['nfft']
        # hop_size = config['STFT']['frame_hop']
        # win_size = config['STFT']['frame_len']
        # window = config['STFT']['window']
        with torch.no_grad():
            # pr = cProfile.Profile()
            # pr.enable()
            outputs = model(x.cuda())
            # pr.disable()
            # s = io.StringIO()
            # sortby = "cumtime"  # 仅适用于 3.6, 3.7 把这里改成常量了
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats(35)
            # print(s.getvalue())
        # session = onnxruntime.InferenceSession(onnxfile, None, providers=['CPUExecutionProvider'])
        # # conv_cache = np.zeros([2, 1, 32, 16, 65],  dtype="float32")
        # # tra_cache = np.zeros([2, 3, 1, 1, 32],  dtype="float32")
        # # inter_cache = np.zeros([2, 1, 65, 32],  dtype="float32")
        # # V_cache = np.zeros([1, 2, 2, 2, 257, 2], dtype="float32")   # complex

        # T_list = []
        # outputs = []
        # inputs = x.numpy()
        # for i in range(inputs.shape[-2]):    
        #     # if i > 0:
        #     #     prev = out_i  
        #     # else:
        #     #     prev = None
        #     # out_i,  conv_cache, tra_cache, inter_cache, bss_out, V_cache \
        #     #       = IVA_stream_inference(inputs[..., i:i+1, :], prev, conv_cache, tra_cache, inter_cache, V_cache, session)
            
        #     out_i,  conv_cache, tra_cache, inter_cache \
        #         = session.run([], {'mix': inputs[..., i:i+1, :],
        #             'conv_cache': conv_cache,
        #             'tra_cache': tra_cache,
        #             'inter_cache': inter_cache})
            
        #     outputs.append(out_i)
        # outputs = np.concatenate(outputs, axis=2)
        
        outputs = outputs.detach().cpu().numpy()
        enh = istft(outputs, n_fft=512, hop_length=256, win_length=512, window=np.hanning(512)**0.5)

        # with torch.no_grad():
        #     batch_mix = torch.from_numpy(mix).float().cuda()  # t,c
        #     #batch_target = torch.from_numpy(ref).float().cuda()  # 1, t
        #     mix_spec = multich_stft(batch_mix.permute(1, 0).unsqueeze(0))
        #     # b,c,f,t complex
        #     res = []
        #     total_frame = mix_spec.shape[3]
        #     for ii in range(total_frame):
        #         frame_out = model(mix_spec[:, :, :, ii:ii + 1]) # b c f 1
        #         res.append(frame_out)
                
        #     res = torch.cat(res, dim=-1).squeeze(1)  # b c f t
        #     enh = torch.istft(res, nfft, hop_size, win_size, torch.hann_window(win_size).pow(0.5).cuda()).t()
        
        wav_len = enh.shape[-1]
        
        mix = mix[:wav_len, 0:1].T
        est = enh.squeeze(0)  # channel, sample
        ref = np.expand_dims(ref[:wav_len], axis=0)  # channel, sample
        # print("mix", mix.shape, "est", est.shape, "ref", ref.shape)

        #mix_sisdr = fast_bss_eval.si_sdr(ref, mix)
        est_sisdr = fast_bss_eval.si_sdr(ref, est)
            #sisdr
        #mix_sisdr_list.append(mix_sisdr)
        est_sisdr_list.append(est_sisdr)

        #stoi
        #mix_stoi_list.append(stoi(mix,ref))
        est_stoi_list.append(stoi(est,ref))

        #write wav
        save_wav_path = os.path.join(output_dir, f"test_{os.path.basename(mix_wav_path)}")
        sf.write(save_wav_path, est.T, 16000)

        #pesq
        #mix_pesq_list.append(pesq(mix,ref))
        est_pesq_list.append(pesq(est,ref))
        #original_mos = dnsmos.run(normalize(mix.squeeze()), 16000)
        enhanced_mos = dnsmos.run(normalize(est.squeeze()), 16000)
        
        
        est_mos+=np.array([enhanced_mos['sig_mos'], enhanced_mos['bak_mos'], enhanced_mos['ovrl_mos'], enhanced_mos['p808_mos']])
        #mix_mos+=np.array([original_mos['sig_mos'], original_mos['bak_mos'], original_mos['ovrl_mos'], original_mos['p808_mos']])
        
        end = timeit.default_timer()
        curr_time = end - start
        ttime += curr_time
        mtime = ttime / cnt
        ttbar.update_progress(int(cnt), 'test', 'ctime/mtime={:.3f}/{:.3f}'.format(curr_time, mtime))
        info['stoi'] = {'est': est_stoi_list[-1].item()}
        info['pesq'] = {'est': est_pesq_list[-1].item()}
        #info['sisdr'] = {'mix': mix_sisdr.item(), 'est': est_sisdr.item()}
        info['out_path'] = save_wav_path
        info['noisy'] = mix_wav_path
        results.append(info)
        
        
        

    ttbar.finish()
    #del mix_stoi_list,est_stoi_list,mix_pesq_list,est_pesq_list,mix_sisdr_list,est_sisdr_list
    est_mos /= cnt; mix_mos /= cnt
    
    # print('STOI')
    # print(f"{np.mean(mix_stoi_list)}->{np.mean(est_stoi_list)}")
    # print('PESQ')
    # print(f"{np.mean(mix_pesq_list)}->{np.mean(est_pesq_list)}")
    # print('Sisdr')
    # print(f"{np.mean(mix_sisdr_list)}->{np.mean(est_sisdr_list)}")
    # print(f"SIG (语音质量): {mix_mos[0]:.2f}->{est_mos[0]:.2f}")
    # print(f"BAK (背景噪声抑制): {mix_mos[1]:.2f}->{est_mos[1]:.2f}")
    # print(f"OVR (整体质量): {mix_mos[2]:.2f}->{est_mos[2]:.2f}")
    # print(f"P808 (DNSMOS): {mix_mos[3]:.2f}->{est_mos[3]:.2f}")
    
    print('STOI')
    print(f"->{np.mean(est_stoi_list)}")
    print('PESQ')
    print(f"->{np.mean(est_pesq_list)}")
    print('Sisdr')
    print(f"->{np.mean(est_sisdr_list)}")
    print(f"SIG (语音质量): ->{est_mos[0]:.2f}")
    print(f"BAK (背景噪声抑制): ->{est_mos[1]:.2f}")
    print(f"OVR (整体质量): ->{est_mos[2]:.2f}")
    print(f"P808 (DNSMOS): ->{est_mos[3]:.2f}")
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
        f.write(f'STOI'
                f"->{np.mean(est_stoi_list)}"
                f'PESQ'
                f"->{np.mean(est_pesq_list)}"
                f'Sisdr'
                f"->{np.mean(est_sisdr_list)}"  
                f"SIG (语音质量): ->{est_mos[0]:.2f}"
                f"BAK (背景噪声抑制): ->{est_mos[1]:.2f}"
                f"OVR (整体质量): ->{est_mos[2]:.2f}"
                f"P808 (DNSMOS): ->{est_mos[3]:.2f}")

