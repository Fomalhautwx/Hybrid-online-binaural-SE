import os
import torch
import numpy as np
import json
import torch.nn as nn
from einops import rearrange
import yaml
from models.AuxIVA_ICD_RLS_harmonic import AuxIVA_ICD_RLS_harmonic_nn_Frame_Module, AuxIVA_ICD_RLS_harmonic
from models.modules import StreamConv2d, StreamConvTranspose2d, multich_stft, convert_to_stream
import time
import soundfile as sf
from tqdm import tqdm
from models.gtcrn_binaural import GTCRN_Binaural
from models.Paris_enhancement import GTCRN_PARIS
from models.gtcrn_paris_stream import StreamGTCRN
from models.AuxIVA import AuxIVA_nn_stream, AuxIVA_stream_single
from models.modules import Dereverberator
from loss_modules.loss import HybridLoss

from metrics import pesq_scorer, stoi_scorer
import fast_bss_eval


device = torch.device("cpu")
config_file = 'config.yaml'
output_dir = 'sample/inspect'
ckpt_path = '/data/hdd0/jizhe.lu/exps/new_loss/checkpoints/model_epoch34.tar'
os.makedirs(output_dir, exist_ok=True)

def IVA_stream_inference(input, out_i,  conv_cache, tra_cache, inter_cache, V_cache, X_cache, phi_cache, Buffer_size, derev_module, onnx_session):
    '''
    input: numpy array, (1, 2, 257, 1, 2)
    out_i: numpy array, (1, 1, 257, 1, 2)
    conv_cache: numpy array, (2, 1, 32, 16, 65)
    tra_cache: numpy array, (2, 3, 1, 1, 32)
    inter_cache: numpy array, (2, 1, 65, 32)
    derev_module: Dereverberator instance
    onnx_session: onnxruntime session
    '''
    if out_i is None:
        init_input = torch.from_numpy(input[:,1:2,...])  # (1, 1, 257, 1, 2) 直接用输入的第一个通道作为初始输入
    else:
        init_input =torch.from_numpy(out_i)
    r_iva = torch.view_as_complex(init_input)
    mag_prior = torch.abs(r_iva).squeeze(0) ** 2
    input_ = torch.view_as_complex(torch.from_numpy(input)) # 1 2 257 1
    input_ = input_.squeeze(-1).transpose(1,2)  #b F D
    dereved = derev_module.process(input_, mag_prior.to(derev_module.device)) # -> F D complex
    input_ = dereved.transpose(1,2)  # b D F 1 complex
    
    #r_iva = torch.complex(init_input[...,0], torch.zeros_like(init_input[...,0]))  # (1, 1, 257)
    # bss_out = AuxIVA_nn_stream(input_, r_iva)    # (1, 2, 257, 1) complex
    V_cache = torch.from_numpy(V_cache)
    X_cache = torch.from_numpy(X_cache)
    phi_cache = torch.from_numpy(phi_cache)
    bss_out, V_cache, x_cache, phi_cache = \
        AuxIVA_stream_single(input_, r_iva, V_cache, X_cache, phi_cache, n_iter=1, Lb=Buffer_size)    # (1, 2, 257, 1) complex
    bss_out = torch.unsqueeze(bss_out, dim=-1)
    bss_out = torch.view_as_real(bss_out).float().numpy()  # (1, 2, 257, 1, 2)
    der_out = torch.unsqueeze(input_, dim=-1)
    der_out = torch.view_as_real(der_out).float().numpy()
    
    
    init_input = init_input.unsqueeze(-2)
    out,  conv_cache, tra_cache, inter_cache \
                    = onnx_session.run([], {'spec': bss_out,# (1, 2, 257, 1, 2)
                        'conv_cache': conv_cache,
                        'tra_cache': tra_cache,
                        'inter_cache': inter_cache})  
    return out, conv_cache, tra_cache, inter_cache, bss_out, der_out, V_cache.numpy(), x_cache.numpy(), phi_cache.numpy()


def stream_inference(input, out_i,  conv_cache, tra_cache, inter_cache, Buffer_size, derev_module, bss_module, onnx_session):
    '''
    input: numpy array, (1, 2, 257, 1, 2)
    out_i: numpy array, (1, 1, 257, 1, 2)
    conv_cache: numpy array, (2, 1, 32, 16, 65)
    tra_cache: numpy array, (2, 3, 1, 1, 32)
    inter_cache: numpy array, (2, 1, 65, 32)
    derev_module: Dereverberator instance
    bss_module: AuxIVA module
    onnx_session: onnxruntime session
    '''
    if out_i is None:
        init_input = torch.from_numpy(input[:,1:2,...])  # (1, 1, 257, 1, 2) 直接用输入的第一个通道作为初始输入
    else:
        init_input =torch.from_numpy(out_i)
    r_iva = torch.view_as_complex(init_input)
    mag_prior = torch.abs(r_iva) ** 2  # power prior for WPE
    input_ = torch.from_numpy(input)# 1 2 257 1
    input_ = input_.transpose(1,2)  #b F D
    dereved = derev_module.process(input_, mag_prior.to(derev_module.device)) # -> F D complex
    input_ = dereved.transpose(1,2)  # b D F 1 complex
    
    #r_iva = torch.complex(init_input[...,0], torch.zeros_like(init_input[...,0]))  # (1, 1, 257)
    # bss_out, V_cache, x_cache, phi_cache = \
    #     AuxIVA_stream_single(input_, r_iva, V_cache, X_cache, phi_cache, n_iter=1, Lb=Buffer_size)    # (1, 2, 257, 1) complex
    
    bss_out = bss_module(input_, r_iva)
    #bss_out = torch.unsqueeze(bss_out, dim=-1)
    bss_out = torch.view_as_real(bss_out).float().numpy()  # (1, 2, 257, 1, 2)
    der_out = torch.view_as_real(input_).float().numpy()
    
    init_input = init_input.unsqueeze(-2)
    # out,  conv_cache, tra_cache, inter_cache \
    #                 = onnx_session.run([], {'spec': bss_out,# (1, 2, 257, 1, 2)
    #                     'conv_cache': conv_cache,
    #                     'tra_cache': tra_cache,
    #                     'inter_cache': inter_cache})  
    return bss_out[:,0:1,...], conv_cache, tra_cache, inter_cache, bss_out, der_out
    #return out, conv_cache, tra_cache, inter_cache, bss_out, der_out


model = GTCRN_Binaural().to(device).eval()
model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'], strict=False)
stream_model = StreamGTCRN().to(device).eval()
convert_to_stream(stream_model, model)  # 给stream model加载参数

"""Streaming Conversion"""
### offline inference
# Audio input
# x = torch.from_numpy(sf.read('/home/nis/jizhe.lu/Code/2sderev/results/E2EpWPE_HA/for_IVA.wav', dtype='float32')[0])
x = torch.from_numpy(sf.read('/data/ssd0/jizhe.lu/Simulated/TEST/sample_5_clean_fileid_6162_test.wav', dtype='float32')[0])
#x = torch.from_numpy(sf.read('/data/ssd0/jizhe.lu/Simulated/TEST/sample_0_clean_fileid_9027_test.wav', dtype='float32')[0])
#x = x - x.mean(dim=-1, keepdim=True)  # DC offset removal
noisy = x.detach().numpy().T[0:1, :]

x = multich_stft(x.T.unsqueeze(0))
# with torch.no_grad():
#     y = model(x.to(device))
# if y.ndim==4:
#     y = y.squeeze(1)
# y = torch.istft(y, 512, 256, 512, torch.hann_window(512).pow(0.5)).detach().cpu().numpy()
# sf.write('test_wavs/offline.wav', y.squeeze(), 16000)

#print(x.shape)  # (1, 2, 257, 572)  # B, C, F, T
x = torch.view_as_real(x).to(device)  # (1, 2, 257, 572, 2)

## online (streaming) inference
conv_cache = torch.zeros(2, 1, 32, 16, 65).to(device)
tra_cache = torch.zeros(2, 3, 1, 1, 32).to(device)
inter_cache = torch.zeros(2, 1, 65, 32).to(device)
# ys = []
# times = []
# yi = None   # 初始化init变量

# for i in tqdm(range(x.shape[3])):
#     xi = x[:,:,:,i:i+1]
#     tic = time.perf_counter()
#     with torch.no_grad():
#         yi, conv_cache, tra_cache, inter_cache = stream_model(xi, conv_cache, tra_cache, inter_cache, yi)
#     toc = time.perf_counter()
#     times.append((toc-tic)*1000)
#     ys.append(yi)  # (1, 1, 257, 1, 2)虚实stack的
    
# ys = torch.cat(ys, dim=3)  # (1, 1, 257, 572, 2)
# ys = torch.complex(ys[...,0], ys[...,1])  # (1, 1, 257, 572)
# ys = ys.squeeze(1)  # (1, 257, 572)

# ys = torch.istft(ys, 512, 256, 512, torch.hann_window(512).pow(0.5)).detach().cpu().numpy()

# sf.write('test_wavs/enh_stream.wav', ys.squeeze(), 16000)
# print(">>> inference time: mean: {:.1f}ms, max: {:.1f}ms, min: {:.1f}ms".format(sum(times)/len(times), max(times), min(times)))
# print(">>> Streaming error:", np.abs(y-ys).max())

'''# stream_path = '/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/test_wavs/enh_stream.wav'
clean_path = '/data/hdd0/jizhe.lu/Simulated/TEST_TRUTH1/test_speaker-3866_clean.wav'
clean = sf.read(clean_path)[0]
# stream = sf.read(stream_path)[0]
clean = np.expand_dims(clean, axis=0)
# stream = np.expand_dims(stream, axis=0)
stream = ys
print(clean.shape, noisy.shape, y.shape, stream.shape)
mix_sisdr = fast_bss_eval.si_sdr(clean, noisy)
off_sisdr = fast_bss_eval.si_sdr(clean, y)
on_sisdr = fast_bss_eval.si_sdr(clean, stream)

stoi = stoi_scorer()
pesq = pesq_scorer()
print('STOI')
print(f"offline {stoi(noisy, clean)}->{stoi(y, clean)}")
print(f"stream {stoi(noisy, clean)}->{stoi(stream, clean)}")
print('PESQ')
print(f"offline {pesq(noisy, clean)}->{pesq(y, clean)}")
print(f"stream {pesq(noisy, clean)}->{pesq(stream, clean)}")
print('\n')
print('Sisdr')
print(f"offline {mix_sisdr}->{off_sisdr}")
print(f"stream {mix_sisdr}->{on_sisdr}")'''


"""ONNX Conversion"""
import os
import time
import onnx
import onnxruntime
from onnxsim import simplify
from librosa import istft

## convert to onnx
file = 'onnx_models/Paris.onnx'
os.makedirs('onnx_models', exist_ok=True)
if not os.path.exists(file):
    input = torch.randn(1, 2, 257, 1, 2, device=device)
    # init = torch.zeros(1, 1, 257, 1, 2, device=device)
    torch.onnx.export(stream_model,
                    (input, conv_cache, tra_cache, inter_cache),
                    file,
                    input_names = ['spec', 'conv_cache', 'tra_cache', 'inter_cache'],
                    output_names = ['spec_enh', 'conv_cache_out', 'tra_cache_out', 'inter_cache_out'],
                    opset_version=11,
                    dynamic_axes=None,
                    verbose = False)

    onnx_model = onnx.load(file)
    onnx.checker.check_model(onnx_model)
# simplify onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, file.split('.onnx')[0] + '_simple.onnx')

## run onnx model
# session = onnxruntime.InferenceSession(file, None, providers=['CPUExecutionProvider'])
# session = onnxruntime.InferenceSession(file.split('.onnx')[0]+'_simple.onnx', None, providers=['CPUExecutionProvider'])
session = onnxruntime.InferenceSession("/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/onnx_models/Paris_simple.onnx", None, providers=['CPUExecutionProvider'])
buffer_size = 16
conv_cache = np.zeros([2, 1, 32, 16, 65],  dtype="float32")
tra_cache = np.zeros([2, 3, 1, 1, 32],  dtype="float32")
inter_cache = np.zeros([2, 1, 65, 32],  dtype="float32")

with open('/home/nis/jizhe.lu/Code/2sderev/derev_params.json') as f:
    params_dict = json.load(f)
params = params_dict['wpe_ha']
dereverb = Dereverberator(nsrc=2, params=params)   #实例化去混响模块
dereverb.reset_state()
dereverb.to(device)

auxiva_icd = AuxIVA_ICD_RLS_harmonic(alpha2=0.2)  # instantiate iva

V_cache = np.zeros([1, 2, 2, 2, 257, 2], dtype="float32") 
X_cache = np.zeros([1, 2, buffer_size, 257, 2], dtype="float32")   
phi_cache = np.zeros([1,2,buffer_size], dtype="float32")  

T_list = []
outputs = []
bss = []
der = []

groundtruth = "/data/ssd0/jizhe.lu/Simulated/TEST_TRUTH/RTS/test_clean_fileid_6162_RTS.wav"
truth = torch.from_numpy(sf.read(groundtruth)[0])
truth = multich_stft(truth.T.unsqueeze(0))
truth = torch.view_as_real(truth).to(device).float()  # (1, 1, 257, T, 2)

inputs = x.numpy()
prev = None

for i in tqdm(range(-(-inputs.shape[-2]//2))):
    # if i > 0:
    #     prev = out_i  
    # else:
    #     prev = None
    #prev = truth[:,:,:, i].numpy()  # groundtruth作为初始输入，观察模型行为
    tic = time.perf_counter()
    if i < 5: prev = None # 前几帧冷启动
    # out_i, conv_cache, tra_cache, inter_cache, bss_i, der_i, V_cache, X_cache, phi_cache \
    #         = IVA_stream_inference(inputs[..., i, :], prev, conv_cache, tra_cache, inter_cache, V_cache, X_cache, phi_cache, buffer_size, dereverb, session)
    
    out_i, conv_cache, tra_cache, inter_cache, bss_i, der_i \
            = stream_inference(inputs[..., 2*i:2*i+2, :], prev, conv_cache, tra_cache, inter_cache, buffer_size, dereverb, auxiva_icd, session)


    toc = time.perf_counter()
    outputs.append(out_i)
    bss.append(bss_i)
    der.append(der_i)
    T_list.append(toc-tic)
    prev = out_i # 去掉时间维度传给下一帧
    # if i == 1:
    #     exit(0)

outputs = np.concatenate(outputs, axis=3)
bss_res = np.concatenate(bss, axis=3)   #st
der_res = np.concatenate(der, axis=3)

print(der_res.shape)  # (1, 2, 257, T, 2)
enhanced = istft(outputs[...,0] + 1j * outputs[...,1], n_fft=512, hop_length=256, win_length=512, window=np.hanning(512)**0.5)

res = []
for c in range(bss_res.shape[1]):
    chan = bss_res[:, c, ...]
    ch = istft(chan[..., 0] + 1j*chan[..., 1], n_fft=512, hop_length=256, win_length=512, window=np.hanning(512)**0.5)
    res.append(ch)
bss_res = np.stack(res, axis=0)  # C, T
res = []
for c in range(der_res.shape[1]):
    chan = der_res[:, c, ...]
    ch = istft(chan[..., 0] + 1j*chan[..., 1], n_fft=512, hop_length=256, win_length=512, window=np.hanning(512)**0.5)
    res.append(ch)
der_res = np.stack(res, axis=0)  # C, T
print(der_res.shape)

# print(enhanced.shape, bss_res.shape)
sf.write(os.path.join(output_dir, "enhanced.wav"), enhanced.squeeze(), 16000)
sf.write(os.path.join(output_dir, "enh_der_train.wav"), der_res.squeeze().T, 16000)
sf.write(os.path.join(output_dir, "enh_bss_train.wav"), bss_res.squeeze().T, 16000)

# print(">>> Onnx error:", np.abs(y - enhanced).max())
print(">>> inference time: mean: {:.1f}ms, max: {:.1f}ms, min: {:.1f}ms".format(1e3*np.mean(T_list), 1e3*np.max(T_list), 1e3*np.min(T_list)))
print(">>> RTF:", 1e3*np.mean(T_list) / 16)
loss_func = HybridLoss().to(device)

truth = torch.view_as_complex(truth)
x = torch.view_as_complex(x)
outputs = torch.view_as_complex(torch.from_numpy(outputs))
loss_ori = loss_func(x, truth)
loss_enh = loss_func(outputs, truth)
print(f"ONNX Loss: noisy {loss_ori['total']:.4f} -> enhanced {loss_enh['total']:.4f}")
# print(np.sum((y-enhanced)**2))

