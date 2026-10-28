import os
import torch
import numpy as np
import json
import torch.nn as nn
from einops import rearrange
import yaml
from models.AuxIVA_ICD_RLS_harmonic import AuxIVA_ICD_RLS_harmonic_nn_Frame_Module, AuxIVA_ICD_RLS_harmonic
from models.AuxIVA_ICD_analytic import AuxIVA_ICD_RLS_harmonic_nn_Frame_Module as AuxIVA_ICD_RLS_harmonic_analytic
from models.AuxIVA_ISS_harmonic_v2 import AuxIVA_ISS_harmonic_nn_Frame_Module
from models.modules import StreamConv2d, StreamConvTranspose2d, multich_stft, convert_to_stream
import time
import soundfile as sf
from tqdm import tqdm
from models.gtcrn_binaural import GTCRN_Binaural
from models.Paris_enhancement import GTCRN_PARIS
from models.gtcrn_paris_stream import StreamGTCRN, StreamGTCRNDual
from models.AuxIVA import AuxIVA_nn_stream, AuxIVA_stream_single
#from models.Auxiva_april import AuxIVA_stream_single
from models.modules import Dereverberator
from loss_modules.loss import HybridLoss

from metrics import pesq_scorer, stoi_scorer
import fast_bss_eval
from speechmos import dnsmos
from utils.swap_prob_plot import SwapProbRecorder


device = torch.device("cpu")
config_file = 'config.yaml'
output_dir = 'sample/inspect'
ckpt_path = '/data/ssd0/jizhe.lu/exps/dual_mask_gradient_renew/checkpoints/model_epoch114.tar'  # 双掩码 iva_dual ep31
baseline_path = '/data/ssd0/jizhe.lu/exps/dcgtcrn/checkpoints/model_epoch68.tar'
os.makedirs(output_dir, exist_ok=True)

def IVA_stream_inference(input, out_i,  conv_cache, tra_cache, inter_cache, V_cache, X_cache, phi_cache, Buffer_size, derev_module, onnx_session, W_cache=None):
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
        init_input = torch.from_numpy(input[:,1:2,...])  # (1, 1, 257, 2) 直接用输入的第一个通道作为初始输入
    else:
        init_input =torch.from_numpy(out_i)
    r_iva = torch.view_as_complex(init_input)
    mag_prior = r_iva.squeeze(-1)
    input_ = torch.view_as_complex(torch.from_numpy(input)) # 1 2 257 1
    input_ = input_.squeeze(-1).transpose(1,2)  #b F D
    dereved = derev_module.streaming(input_, mag_prior.to(derev_module.device)) # -> F D complex
    input_ = dereved.transpose(1,2)  # b D F 1 complex
    
    #r_iva = torch.complex(init_input[...,0], torch.zeros_like(init_input[...,0]))  # (1, 1, 257)
    # bss_out = AuxIVA_nn_stream(input_, r_iva)    # (1, 2, 257, 1) complex
    V_cache = torch.from_numpy(V_cache)
    X_cache = torch.from_numpy(X_cache)
    phi_cache = torch.from_numpy(phi_cache)
    W_cache = torch.from_numpy(W_cache)
    bss_out, V_cache, x_cache, phi_cache, W_prev, eig_gap = \
        AuxIVA_stream_single(input_, r_iva, V_cache, X_cache, phi_cache, W_cache, n_iter=1, alpha=0.8, Lb=Buffer_size)    # (1, 2, 257, 1) complex
    bss_out = torch.unsqueeze(bss_out, dim=-1)
    bss_out = torch.view_as_real(bss_out).float().numpy()  # (1, 2, 257, 1, 2)
    der_out = torch.unsqueeze(input_, dim=-1)
    der_out = torch.view_as_real(der_out).float().numpy()
    
    
    init_input = init_input.unsqueeze(-2)
    out,  conv_cache, tra_cache, inter_cache \
                    = onnx_session.run([], {'spec': bss_out,# (1, 2, 257, 1, 2)
                        'conv_cache': conv_cache,
                        'tra_cache': tra_cache,
                        'inter_cache': inter_cache,
                        'ref': bss_out[:, 0:1, ...]})  # 方案A：未压缩 IVA 目标路（bp 后）
    return out, conv_cache, tra_cache, inter_cache, bss_out, der_out, V_cache.numpy(), x_cache.numpy(), phi_cache.numpy(), W_prev.numpy(), eig_gap


def stream_inference(input, out_i,  conv_cache, tra_cache, inter_cache, Buffer_size, derev_module, bss_module, onnx_session):
    '''
    input: numpy array, (1, 2, 257, 2)
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
    mag_prior = r_iva.squeeze(-1)
    input_ = torch.view_as_complex(torch.from_numpy(input)) # 1 2 257 1
    input_ = input_.squeeze(-1).transpose(1,2)  #b F D
    dereved = derev_module.streaming(input_, mag_prior.to(derev_module.device)) # -> F D complex
    input_ = dereved.transpose(1,2)  # b D F 1 complex
    # ---- 加这一段 ----
    wpe_peak = input_.abs().max().item()
    # if wpe_peak > 10:
    #     print(f"[WPE spike] frame peak={wpe_peak:.1f}")
    # # ------------------

    
    bss_out = bss_module(input_, r_iva)
    bss_out = torch.unsqueeze(bss_out, dim=-1)
    
    p = 0.3
    mag = bss_out.abs().clamp_min(1e-6)          # 防止 0^(-0.7) = inf
    spec_iva_c = bss_out * mag.pow(p - 1.0)      # 幅度压缩, 相位原样保留
    spec_iva_c = torch.view_as_real(spec_iva_c).float().numpy()  # (B, nsrc, F, T, 2)
    bss_out = torch.view_as_real(bss_out).float().numpy()  # (1, 2, 257, 1, 2)
    
    der_out = torch.unsqueeze(input_, dim=-1)
    der_out = torch.view_as_real(der_out).float().numpy()
    init_input = init_input.unsqueeze(-2)
    out,  conv_cache, tra_cache, inter_cache \
                    = onnx_session.run([], {'spec': spec_iva_c,# (1, 2, 257, 1, 2)
                        'ref': bss_out,# 双掩码：未压缩 IVA 两路（bp 后）
                        'conv_cache': conv_cache,
                        'tra_cache': tra_cache,
                        'inter_cache': inter_cache})
    #return bss_out[:,0:1,...], conv_cache, tra_cache, inter_cache, bss_out, der_out
    return out, conv_cache, tra_cache, inter_cache, bss_out, der_out#, bss_module.swap_prob


def load_stream_from_sd(stream_model, src_sd):
    """把 ckpt 的 state dict（GTCRN_PARIS 子网络键）映射到流式模型。
    规则与 convert_to_stream 一致，但源直接是 dict（双掩码模型无法先载入 GTCRN_Binaural）。"""
    new_sd = stream_model.state_dict()
    for key in list(new_sd.keys()):
        if key in src_sd:
            if ('point_conv' in key) and 'decode' in key and key.endswith('weight'):
                new_sd[key] = src_sd[key].permute([1, 0, 2, 3])
            else:
                new_sd[key] = src_sd[key]
        elif key.replace('Conv1d.', '') in src_sd:
            new_sd[key] = src_sd[key.replace('Conv1d.', '')]
        elif key.replace('Conv2d.', '') in src_sd:
            new_sd[key] = src_sd[key.replace('Conv2d.', '')]
        elif key.replace('ConvTranspose2d.', '') in src_sd:
            k2 = key.replace('ConvTranspose2d.', '')
            if key.endswith('weight'):
                if new_sd[key].shape != src_sd[k2].shape:
                    new_sd[key] = src_sd[k2].permute([1, 0, 2, 3])
                else:
                    new_sd[key] = src_sd[k2]
            else:
                new_sd[key] = src_sd[k2]
        else:
            raise ValueError(f'key 无法映射: {key}')
    stream_model.load_state_dict(new_sd, strict=True)

model = GTCRN_Binaural().to(device).eval()          # 仅用于 warmup 前 20 帧与 dc baseline（稍后载入 dcgtcrn）
stream_model = StreamGTCRNDual().to(device).eval()
load_stream_from_sd(stream_model, torch.load(ckpt_path, map_location=device)['model'])
print(f"stream 双掩码模型已加载 {ckpt_path}\n  mask2_gate = {stream_model.mask2_gate.item():.4f}")
loss_func = HybridLoss().to(device)

"""Streaming Conversion"""
### offline inference
# Audio input
# x = torch.from_numpy(sf.read('/home/nis/jizhe.lu/Code/2sderev/results/E2EpWPE_HA/for_IVA.wav', dtype='float32')[0])
groundtruth = "/data/ssd0/jizhe.lu/Simulated/TEST_TRUTH/RTS/test_clean_fileid_6162_RTS.wav"
x = torch.from_numpy(sf.read('/data/ssd0/jizhe.lu/Simulated/TEST/sample_5_clean_fileid_6162_test.wav', dtype='float32')[0])
#x = torch.from_numpy(sf.read('/data/ssd0/jizhe.lu/Simulated/TEST/sample_0_clean_fileid_9027_test.wav', dtype='float32')[0])
#x = x - x.mean(dim=-1, keepdim=True)  # DC offset removal
noisy = x.detach().numpy().T[0:1, :]

x = multich_stft(x.T.unsqueeze(0))
truth = torch.from_numpy(sf.read(groundtruth)[0])
truth = multich_stft(truth.T.unsqueeze(0))
truth = torch.view_as_real(truth).to(device).float()  # (1, 1, 257, T, 2)

# ---- 对照：权重直接当 DC-GTCRN 用（吃原始混合）----
# 双掩码模型 decoder 输出 4 通道，无法载入单掩码 GTCRN_Binaural，本跑跳过该对照
loss_ivaiss_dc = {'total': float('nan')}
IvaissDC = noisy.copy()   # 占位（不出现在对照表）

# with torch.no_grad():
#     y = model(x.to(device))
# if y.ndim==4:
#     y = y.squeeze(1)
# loss_baseline = loss_func(y.squeeze(0), torch.view_as_complex(truth))
# y = torch.istft(y, 512, 256, 512, torch.hann_window(512).pow(0.5)).detach().cpu().numpy()
# sf.write(os.path.join(output_dir, "baseline.wav"), y.squeeze(), 16000)
dcgtcrn = model.load_state_dict(torch.load(baseline_path, map_location=device)['model'], strict=False)
with torch.no_grad():
    baseline = model(x.to(device))
loss_baseline = loss_func(baseline.squeeze(0), torch.view_as_complex(truth))

#print(x.shape)  # (1, 2, 257, 572)  # B, C, F, T
x = torch.view_as_real(x).to(device)  # (1, 2, 257, 572, 2)

## online (streaming) inference
conv_cache = torch.zeros(2, 1, 32, 16, 65).to(device)
tra_cache = torch.zeros(2, 3, 1, 1, 32).to(device)
inter_cache = torch.zeros(2, 1, 65, 32).to(device)

"""ONNX Conversion"""
import os
import time
import onnx
import onnxruntime
from onnxsim import simplify
from librosa import istft

## convert to onnx
file = 'onnx_models/Ivaiss_dual.onnx'
os.makedirs('onnx_models', exist_ok=True)
if not os.path.exists(file):
    input = torch.randn(1, 2, 257, 1, 2, device=device)
    ref_dummy  = torch.randn(1, 2, 257, 1, 2, device=device)   # 双掩码：ref = 两路 IVA 输出
    # init = torch.zeros(1, 1, 257, 1, 2, device=device)
    torch.onnx.export(stream_model,
                    (input, ref_dummy, conv_cache, tra_cache, inter_cache),
                    file,
                    input_names = ['spec', 'ref', 'conv_cache', 'tra_cache', 'inter_cache'],
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
session = onnxruntime.InferenceSession("/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/onnx_models/Ivaiss_dual_simple.onnx", None, providers=['CPUExecutionProvider'])
buffer_size = 16
conv_cache = np.zeros([2, 1, 32, 16, 65],  dtype="float32")
tra_cache = np.zeros([2, 3, 1, 1, 32],  dtype="float32")
inter_cache = np.zeros([2, 1, 65, 32],  dtype="float32")

with open('/home/nis/jizhe.lu/Code/2sderev/derev_params.json') as f:
    params_dict = json.load(f)
params = params_dict['wpe_ha']
dereverb = Dereverberator(nsrc=2, params=params)   #实例化去混响模块
dereverb.reset_state()
dereverb.wpe.train = False  
dereverb.to(device)

auxiva_icd = AuxIVA_ICD_RLS_harmonic_analytic(alpha=0.99, alpha2=0)  # instantiate iva
auxiva_iss = AuxIVA_ISS_harmonic_nn_Frame_Module(bandsize=257, nsrc=2, alpha=0.9999, alpha2=0, iss_iters=2)  # instantiate iva

V_cache = np.zeros([1, 2, 2, 2, 257, 2], dtype="float32") 
X_cache = np.zeros([1, 2, buffer_size, 257, 2], dtype="float32")   
phi_cache = np.zeros([1,2,buffer_size], dtype="float32")  

T_list = []
outputs = []
bss = []
der = []


inputs = x.numpy()
prev = None
identity = np.eye(2, dtype=np.complex128)
W_cache = np.tile(identity[np.newaxis, np.newaxis, :, :], (1, 257, 1, 1))

for i in tqdm(range(inputs.shape[-2])):
    tic = time.perf_counter()
    if i < 20: 
        frame = torch.from_numpy(inputs[..., i:i+1, :].copy())
        with torch.no_grad():
            out_i = model(torch.view_as_complex(frame))
            out_i = torch.view_as_real(out_i).numpy()
        bss_i = inputs[..., i:i+1, :].copy()
        der_i = inputs[..., i:i+1, :].copy()
    else:
    # out_i, conv_cache, tra_cache, inter_cache, bss_i, der_i, V_cache, X_cache, phi_cache, W_cache, eig_gap\
    #         = IVA_stream_inference(inputs[..., i, :], prev, conv_cache, tra_cache, inter_cache, V_cache, X_cache, phi_cache, buffer_size, dereverb, session, W_cache)  # (1, 1, 257, 1, 2)
    
        out_i, conv_cache, tra_cache, inter_cache, bss_i, der_i \
            = stream_inference(inputs[..., i, :], prev, conv_cache, tra_cache, inter_cache, buffer_size, dereverb, auxiva_iss, session)
    #rec.record(swap)

    toc = time.perf_counter()
    outputs.append(out_i)
    bss.append(bss_i)
    der.append(der_i)
    T_list.append(toc-tic)
    prev = out_i.squeeze(-2) # 去掉时间维度传给下一帧
    # if i == 1:
    #     exit(0)

#rec.save_plot("swap_prob.png")
outputs = np.concatenate(outputs, axis=3)
bss_res = np.concatenate(bss, axis=3)   #st
der_res = np.concatenate(der, axis=3)

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

Baseline = torch.istft(baseline.squeeze(0), 512, 256, 512, torch.hann_window(512).pow(0.5)).detach().cpu().numpy()


sf.write(os.path.join(output_dir, "enhanced.wav"), enhanced.squeeze(), 16000)
sf.write(os.path.join(output_dir, "enh_der_train.wav"), der_res.squeeze().T, 16000)
sf.write(os.path.join(output_dir, "enh_bss_train.wav"), bss_res.squeeze().T, 16000)
sf.write(os.path.join(output_dir, "baseline.wav"), Baseline.squeeze(), 16000)

# print(">>> Onnx error:", np.abs(y - enhanced).max())
print(">>> inference time: mean: {:.1f}ms, max: {:.1f}ms, min: {:.1f}ms".format(1e3*np.mean(T_list), 1e3*np.max(T_list), 1e3*np.min(T_list)))
print(">>> RTF:", 1e3*np.mean(T_list) / 16)


truth = torch.view_as_complex(truth)
x = torch.view_as_complex(x)
outputs = torch.view_as_complex(torch.from_numpy(outputs))
loss_ori = loss_func(x, truth)
loss_enh = loss_func(outputs[..., 50:], truth[..., 50:])
print(f"ONNX Loss: noisy {loss_ori['total']:.4f} -> enhanced {loss_enh['total']:.4f}",
      f"loss baseline:{loss_baseline['total']:.4f}", f"loss ivaiss_as_dc:{loss_ivaiss_dc['total']:.4f}")
# print(np.sum((y-enhanced)**2))

# ============================================================
# Evaluate & Compare
# ============================================================

    
    
_pesq = pesq_scorer()
_stoi = stoi_scorer()


def evaluate_all(signals: dict, ref=None, fs=16000):
    """
    对多条音频统一计算指标并打印对照表。

    Parameters
    ----------
    signals : dict[str, np.ndarray]
        {名称: 音频}，每条音频为 1-D numpy float (1, n_samples)
        例如 {"noisy": noisy, "P0 offline": p0, "P1 stream": p1, "P2 DRISS": p2}

    ref : np.ndarray or None
        干净语音参考，1-D (1, n_samples)。
        为 None 时跳过有参指标（PESQ / STOI / SI-SDR），只算 DNSMOS。

    fs : int
        采样率
    """
    # ---- 列定义 ----
    has_ref = ref is not None
    cols = []
    if has_ref:
        cols += ["PESQ", "STOI", "SI-SDR"]
    cols += ["SIG", "BAK", "OVRL", "P808"]

    # ---- 逐条计算 ----
    rows = {}
    for name, sig in signals.items():
        sig = np.asarray(sig, dtype=np.float64)
                            # 多通道取第一个

        vals = {}
        if has_ref:
            r = np.asarray(ref, dtype=np.float64)
            
            L = min(len(sig), len(r))
            s, r_ = sig[:L], r[:L]

            vals["PESQ"] = float(_pesq(s, r_))
            # except Exception:
            #     vals["PESQ"] = float("nan")

            try:
                vals["STOI"] = float(_stoi(s, r_))
            except Exception:
                vals["STOI"] = float("nan")

            try:
                si = fast_bss_eval.si_sdr(
                    torch.from_numpy(r_).float(),
                    torch.from_numpy(s).float())
                vals["SI-SDR"] = float(si.item())
            except Exception:
                vals["SI-SDR"] = float("nan")

        try:
            s32 = sig.astype(np.float32).squeeze()
            peak = np.max(np.abs(s32)) + 1e-12
            if peak > 1.0:
                s32 = s32 / peak                 # dnsmos 要求 [-1, 1]
            mos = dnsmos.run(s32, sr=fs)
            vals["SIG"]  = mos["sig_mos"]
            vals["BAK"]  = mos["bak_mos"]
            vals["OVRL"] = mos["ovrl_mos"]
            vals["P808"] = mos["p808_mos"]
        except Exception as e:
                for k in ["SIG", "BAK", "OVRL", "P808"]:
                    vals[k] = float("nan")

        rows[name] = vals

    # ---- 打印 ----
    name_w = max(len(n) for n in rows) + 2
    col_w = 8
    header = f"{'':>{name_w}}" + "".join(f"{c:>{col_w}}" for c in cols)
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, vals in rows.items():
        row = f"{name:>{name_w}}"
        for c in cols:
            v = vals.get(c, float("nan"))
            if np.isfinite(v):
                fmt = f"{v:>{col_w}.2f}" if c != "STOI" else f"{v:>{col_w}.4f}"
            else:
                fmt = f"{'N/A':>{col_w}}"
            row += fmt
        print(row)
    print(sep)
    
noisy = noisy
enhanced = enhanced.squeeze(0)
Baseline = Baseline
bss_target = bss_res[0]
bss_interf = bss_res[1]
clean = sf.read(groundtruth)[0].squeeze()
print(">>> Signals shape:", noisy.shape, enhanced.shape, Baseline.shape, bss_target.shape, bss_interf.shape, clean.shape)
signals = {
    "noisy": noisy[:,960:],
    "bss_target": bss_target[:, 960:],
    "bss_interf": bss_interf[:, 960:],
    "enhanced": enhanced[:, 960:],
    "baseline": Baseline[:, 960:],
}
evaluate_all(signals, ref=clean[None,960:], fs=16000)