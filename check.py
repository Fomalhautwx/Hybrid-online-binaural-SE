import os
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from models.modules import StreamConv2d, StreamConvTranspose2d, multich_stft, convert_to_stream
import time
import soundfile as sf
from tqdm import tqdm
from models.gtcrn import GTCRN
from models.gtcrn_paris_stream import StreamGTCRN
from models.AuxIVA import AuxIVA_nn_stream

from metrics import pesq_scorer, stoi_scorer
import fast_bss_eval

device = torch.device("cpu")
os.makedirs('test_wavs', exist_ok=True)

model = GTCRN().to(device).eval()

model.load_state_dict(torch.load('/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/exps/gtcrn_1207/checkpoints/model_epoch150.tar', map_location=device)['model'], strict=False)
stream_model = StreamGTCRN().to(device).eval()
# for name, param in stream_model.encoder.named_parameters():
#     print(name, param.shape)
convert_to_stream(stream_model, model)

"""Streaming Conversion"""
### offline inference
x = torch.from_numpy(sf.read('/data/hdd0/jizhe.lu/Simulated/TEST1/sample_0_speaker-3866_test.wav', dtype='float32')[0])
noisy = x.detach().numpy().T[0:1, :]

x = multich_stft(x.T.unsqueeze(0))
with torch.no_grad():
    y = model(x.to(device))
if y.ndim==4:
    y = y.squeeze(1)
y = torch.istft(y, 512, 256, 512, torch.hann_window(512).pow(0.5)).detach().cpu().numpy()
sf.write('test_wavs/offline.wav', y.squeeze(), 16000)

#print(x.shape)  # (1, 2, 257, 572)  # B, C, F, T
x = torch.view_as_real(x).to(device)  # (1, 2, 257, 572, 2)

## online (streaming) inference
conv_cache = torch.zeros(2, 1, 32, 16, 65).to(device)
tra_cache = torch.zeros(2, 3, 1, 1, 32).to(device)
inter_cache = torch.zeros(2, 1, 65, 32).to(device)
ys = []
times = []
yi = None   # 初始化init变量

for i in tqdm(range(x.shape[3])):
    xi = x[:,:,:,i:i+1]
    tic = time.perf_counter()
    with torch.no_grad():
        yi, conv_cache, tra_cache, inter_cache = stream_model(xi, conv_cache, tra_cache, inter_cache, yi)
    toc = time.perf_counter()
    times.append((toc-tic)*1000)
    ys.append(yi)  # (1, 1, 257, 1, 2)虚实stack的
    
ys = torch.cat(ys, dim=3)  # (1, 1, 257, 572, 2)
ys = torch.complex(ys[...,0], ys[...,1])  # (1, 1, 257, 572)
ys = ys.squeeze(1)  # (1, 257, 572)

ys = torch.istft(ys, 512, 256, 512, torch.hann_window(512).pow(0.5)).detach().cpu().numpy()

sf.write('test_wavs/stream.wav', ys.squeeze(), 16000)
print(">>> inference time: mean: {:.1f}ms, max: {:.1f}ms, min: {:.1f}ms".format(sum(times)/len(times), max(times), min(times)))

print(">>> Streaming error:", np.sum((y-ys)**2)/480000)
print(y.shape, ys.shape)


import matplotlib.pyplot as plt
pl = (y-ys).flatten()[::10]
print(pl.shape)
plt.figure()
plt.plot(pl)
plt.savefig('test_wavs/err.png')
exit()

# stream_path = '/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/test_wavs/enh_stream.wav'
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
print(f"stream {mix_sisdr}->{on_sisdr}")



"""ONNX Conversion"""
import os
import time
import onnx
import onnxruntime
from onnxsim import simplify
from librosa import istft

## convert to onnx
file = 'onnx_models/iva_gtcrn.onnx'
os.makedirs('onnx_models', exist_ok=True)
if not os.path.exists(file):
    input = torch.randn(1, 2, 257, 1, 2, device=device)
    init = torch.zeros(1, 1, 257, 1, 2, device=device)
    torch.onnx.export(stream_model,
                    (input, conv_cache, tra_cache, inter_cache, init),
                    file,
                    input_names = ['spec', 'conv_cache', 'tra_cache', 'inter_cache', 'init'],
                    output_names = ['spec_enh', 'conv_cache_out', 'tra_cache_out', 'inter_cache_out'],
                    opset_version=11,
                    verbose = False)

    onnx_model = onnx.load(file)
    onnx.checker.check_model(onnx_model)

# simplify onnx model
if not os.path.exists(file.split('.onnx')[0]+'_simple.onnx'):
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, file.split('.onnx')[0] + '_simple.onnx')


## run onnx model
# session = onnxruntime.InferenceSession(file, None, providers=['CPUExecutionProvider'])
# session = onnxruntime.InferenceSession(file.split('.onnx')[0]+'_simple.onnx', None, providers=['CPUExecutionProvider'])
session = onnxruntime.InferenceSession("/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/onnx_models/pure_gtcrn_simple.onnx", None, providers=['CPUExecutionProvider'])
conv_cache = np.zeros([2, 1, 32, 16, 65],  dtype="float32")
tra_cache = np.zeros([2, 3, 1, 1, 32],  dtype="float32")
inter_cache = np.zeros([2, 1, 65, 32],  dtype="float32")

T_list = []
outputs = []
bss = []

inputs = x.numpy()
for i in tqdm(range(inputs.shape[-2])):
    tic = time.perf_counter()

    if i > 0:
        prev = out_i  
    else:
        prev = None
    # out_i,  conv_cache, tra_cache, inter_cache, bss_i\
    #         = IVA_stream_inference(inputs[..., i:i+1, :], prev, conv_cache, tra_cache, inter_cache, session)
    
    out_i,  conv_cache, tra_cache, inter_cache \
            = session.run([], {'spec': inputs[..., i:i+1, :],
                'conv_cache': conv_cache,
                'tra_cache': tra_cache,
                'inter_cache': inter_cache,
                'init': out_i if i>0 else np.zeros([1, 1, 257, 1, 2], dtype="float32")})

    toc = time.perf_counter()
    T_list.append(toc-tic)
    outputs.append(out_i)
    # bss.append(bss_i)

outputs = np.concatenate(outputs, axis=3)
# bss_res = np.concatenate(bss, axis=3)   #st
#print(bss_res.mean(axis=(0,2,3,4)))
enhanced = istft(outputs[...,0] + 1j * outputs[...,1], n_fft=512, hop_length=256, win_length=512, window=np.hanning(512)**0.5)
res = []
# for c in range(bss_res.shape[1]):
#     chan = bss_res[:, c, ...]
#     ch = istft(chan[..., 0] + 1j*chan[..., 1], n_fft=512, hop_length=256, win_length=512, window=np.hanning(512)**0.5)
#     res.append(ch)
# bss_res = np.stack(res, axis=0)  # C, T

# print(enhanced.shape, bss_res.shape)
sf.write('test_wavs/enh_onnx.wav', enhanced.squeeze(), 16000)
# sf.write('test_wavs/enh_bss.wav', bss_res.squeeze().T, 16000)

# print(">>> Onnx error:", np.abs(y - enhanced).max())
# print(">>> inference time: mean: {:.1f}ms, max: {:.1f}ms, min: {:.1f}ms".format(1e3*np.mean(T_list), 1e3*np.max(T_list), 1e3*np.min(T_list)))
# print(">>> RTF:", 1e3*np.mean(T_list) / 16)
print(np.sum((y-enhanced)**2))

