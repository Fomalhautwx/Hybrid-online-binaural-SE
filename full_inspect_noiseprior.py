"""full_inspect.py 的噪声先验变体对照（不修改原文件）。

在 sample_5_clean_fileid_6162 一条语音上，流式部署逐帧跑四种噪声先验：
  aligned     —— 存上一帧 X0 作噪声先验（当前行为）：noise_ref = X0(t-1) - Ŝ(t-1)
  misaligned  —— 原来的噪声先验（修复前 bug）：noise_ref = X0(t) - Ŝ(t-1)
  reverse     —— 语音/噪声通道反转：源0=噪声先验、源1=语音先验
  speech      —— 把上一帧 X0 直接喂给噪声行（|X0(t-1)|²，≈语音先验当噪声）

其余（WPE、postfilter ONNX、warmup）完全相同。输出各变体的 enhanced 指标 + bss 两路 SI-SDR。
"""
import os
import time
import json
import numpy as np
import torch
import soundfile as sf
from librosa import istft
import onnxruntime

from models.AuxIVA_ISS_harmonic_v2 import AuxIVA_ISS_harmonic_nn_Frame_Module
from models.modules import multich_stft
from models.gtcrn_binaural import GTCRN_Binaural
from models.gtcrn_paris_stream import StreamGTCRNDual
from models.modules import Dereverberator

from metrics import pesq_scorer, stoi_scorer
import fast_bss_eval
from speechmos import dnsmos

device = torch.device("cpu")
torch.set_num_threads(4)   # 逐帧小矩阵/小 LSTM，多线程调度反噬
ckpt_path = '/data/ssd0/jizhe.lu/exps/dual_mask_gradient_renew/checkpoints/model_epoch114.tar'
baseline_path = '/data/ssd0/jizhe.lu/exps/dcgtcrn/checkpoints/model_epoch68.tar'
onnx_file = '/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/onnx_models/Ivaiss_dual_simple.onnx'
output_dir = 'sample/inspect_noiseprior'
os.makedirs(output_dir, exist_ok=True)

VARIANTS = ['aligned', 'misaligned', 'reverse', 'speech']


class IVAVariant(AuxIVA_ISS_harmonic_nn_Frame_Module):
    """按 variant 改变噪声先验的构造，其余与基类一致。"""

    def __init__(self, variant='aligned', **kw):
        super().__init__(**kw)
        self.variant = variant

    @torch.no_grad()
    def forward(self, X_in, Y_nn):
        # X_in: (B, nch, F) = 当前帧 WPE 输出；Y_nn: (B, 1, F) = 语音先验 r_iva
        if Y_nn.shape[1] != 1:
            return super().forward(X_in, Y_nn)
        X0_prev = self.X0_prev if self.initialized else X_in[:, 0:1, :]
        if self.variant == 'aligned':
            noise = self._build_noise_ref(X0_prev, Y_nn)
            Y2 = torch.cat([Y_nn, noise], dim=1)
        elif self.variant == 'misaligned':
            noise = self._build_noise_ref(X_in[:, 0:1, :], Y_nn)
            Y2 = torch.cat([Y_nn, noise], dim=1)
        elif self.variant == 'reverse':
            noise = self._build_noise_ref(X0_prev, Y_nn)
            Y2 = torch.cat([noise, Y_nn], dim=1)
        elif self.variant == 'speech':
            Y2 = torch.cat([Y_nn, X0_prev], dim=1)   # 噪声行直接吃上一帧 X0
        else:
            raise ValueError(self.variant)
        return super().forward(X_in, Y2)


def load_stream_from_sd(stream_model, src_sd):
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
                new_sd[key] = src_sd[k2] if new_sd[key].shape == src_sd[k2].shape else src_sd[k2].permute([1, 0, 2, 3])
            else:
                new_sd[key] = src_sd[k2]
        else:
            raise ValueError(f'key 无法映射: {key}')
    stream_model.load_state_dict(new_sd, strict=True)


def stream_inference(input, out_i, conv_cache, tra_cache, inter_cache, Buffer_size,
                     derev_module, bss_module, onnx_session, gate0=False):
    if out_i is None:
        init_input = torch.from_numpy(input[:, 1:2, ...])
    else:
        init_input = torch.from_numpy(out_i)
    r_iva = torch.view_as_complex(init_input)
    mag_prior = r_iva.squeeze(-1)
    input_ = torch.view_as_complex(torch.from_numpy(input))
    input_ = input_.squeeze(-1).transpose(1, 2)                     # (B, F, D)
    dereved = derev_module.streaming(input_, mag_prior.to(derev_module.device))
    input_ = dereved.transpose(1, 2)                                 # (B, D, F)

    bss_out = bss_module(input_, r_iva)                              # (B, 2, F)
    bss_out = torch.unsqueeze(bss_out, dim=-1)

    p = 0.3
    mag = bss_out.abs().clamp_min(1e-6)
    spec_iva_c = bss_out * mag.pow(p - 1.0)
    spec_iva_c = torch.view_as_real(spec_iva_c).float().numpy()
    bss_out = torch.view_as_real(bss_out).float().numpy()            # (1,2,257,1,2)

    der_out = torch.unsqueeze(input_, dim=-1)
    der_out = torch.view_as_real(der_out).float().numpy()

    init_input = init_input.unsqueeze(-2)
    ref = bss_out
    if gate0:
        # gate=0 等价于把噪声路载体 Z2 置零：输出 = M1·Z1 + gate·M2·0 = M1·Z1
        ref = bss_out.copy()
        ref[:, 1:2, ...] = 0.0
    out, conv_cache, tra_cache, inter_cache = onnx_session.run(
        [], {'spec': spec_iva_c, 'ref': ref,
             'conv_cache': conv_cache, 'tra_cache': tra_cache, 'inter_cache': inter_cache})
    return out, conv_cache, tra_cache, inter_cache, bss_out, der_out


def run_variant(variant, x, model, dereverb, session, buffer_size=16, gate0=False):
    bss_module = IVAVariant(variant=variant, bandsize=257, nsrc=2, alpha=0.9999,
                            alpha2=0, iss_iters=2).to(device)
    dereverb.reset_state()
    conv_cache = np.zeros([2, 1, 32, 16, 65], dtype="float32")
    tra_cache = np.zeros([2, 3, 1, 1, 32], dtype="float32")
    inter_cache = np.zeros([2, 1, 65, 32], dtype="float32")

    inputs = x.numpy()
    prev = None
    outputs, bss, der = [], [], []
    for i in range(inputs.shape[-2]):
        if i < 20:
            frame = torch.from_numpy(inputs[..., i:i + 1, :].copy())
            with torch.no_grad():
                out_i = model(torch.view_as_complex(frame))
                out_i = torch.view_as_real(out_i).numpy()
            bss_i = inputs[..., i:i + 1, :].copy()
            der_i = inputs[..., i:i + 1, :].copy()
        else:
            out_i, conv_cache, tra_cache, inter_cache, bss_i, der_i = stream_inference(
                inputs[..., i, :], prev, conv_cache, tra_cache, inter_cache,
                buffer_size, dereverb, bss_module, session, gate0=gate0)
        outputs.append(out_i)
        bss.append(bss_i)
        der.append(der_i)
        prev = out_i.squeeze(-2)
        if i % 100 == 0:
            print(f"    [{variant}] frame {i}/{inputs.shape[-2]}", flush=True)

    outputs = np.concatenate(outputs, axis=3)      # (1,1,257,T,2)
    bss_res = np.concatenate(bss, axis=3)          # (1,2,257,T,2)
    der_res = np.concatenate(der, axis=3)

    enhanced = istft(outputs[..., 0] + 1j * outputs[..., 1], n_fft=512, hop_length=256,
                     win_length=512, window=np.hanning(512) ** 0.5)

    bss_wavs = []
    for c in range(bss_res.shape[1]):
        chan = bss_res[:, c, ...]
        ch = istft(chan[..., 0] + 1j * chan[..., 1], n_fft=512, hop_length=256,
                   win_length=512, window=np.hanning(512) ** 0.5)
        bss_wavs.append(ch.squeeze())
    bss_res = np.stack(bss_wavs, axis=0)            # (2, T)

    return enhanced.squeeze(), bss_res


def main():
    # —— 载入模型 ——
    model = GTCRN_Binaural().to(device).eval()
    model.load_state_dict(torch.load(baseline_path, map_location=device)['model'], strict=False)
    stream_model = StreamGTCRNDual().to(device).eval()
    load_stream_from_sd(stream_model, torch.load(ckpt_path, map_location=device)['model'])
    session = onnxruntime.InferenceSession(onnx_file, None, providers=['CPUExecutionProvider'])

    with open('/home/nis/jizhe.lu/Code/2sderev/derev_params.json') as f:
        params = json.load(f)['wpe_ha']
    dereverb = Dereverberator(nsrc=2, params=params)
    dereverb.reset_state()
    dereverb.wpe.train = False
    dereverb.to(device)

    # —— 音频 ——
    groundtruth = "/data/ssd0/jizhe.lu/Simulated/TEST_TRUTH/RTS/test_clean_fileid_6162_RTS.wav"
    noisy = torch.from_numpy(sf.read('/data/ssd0/jizhe.lu/Simulated/TEST/sample_5_clean_fileid_6162_test.wav', dtype='float32')[0])
    x = multich_stft(noisy.T.unsqueeze(0))
    x = torch.view_as_real(x).to(device)             # (1,2,257,T,2)
    clean = sf.read(groundtruth)[0].squeeze()

    # —— 跑各变体 ——
    results = {}
    for var in VARIANTS:
        t0 = time.time()
        enhanced, bss_res = run_variant(var, x, model, dereverb, session)
        results[var] = dict(enhanced=enhanced, bss=bss_res)
        sf.write(os.path.join(output_dir, f"enhanced_{var}.wav"), enhanced, 16000)
        print(f"[{var}] done in {time.time()-t0:.1f}s", flush=True)

    # gate=0：aligned 噪声先验 + 推理时把第二路载体 Z2 置零（= M1·Z1 单掩码）
    t0 = time.time()
    enhanced, bss_res = run_variant('aligned', x, model, dereverb, session, gate0=True)
    results['gate0'] = dict(enhanced=enhanced, bss=bss_res)
    sf.write(os.path.join(output_dir, f"enhanced_gate0.wav"), enhanced, 16000)
    print(f"[gate0] done in {time.time()-t0:.1f}s", flush=True)

    # —— 评测 ——
    _pesq = pesq_scorer()
    _stoi = stoi_scorer()
    ref = clean[None, 960:]

    def metrics(sig):
        sig = np.asarray(sig, dtype=np.float64)            # 1-D
        n = len(ref[0])
        s2 = sig[None, :n]                                  # (1,n) 给 pesq/stoi（需 batch 维）
        r2 = ref                                             # (1,n)
        pesq = float(_pesq(s2, r2))
        stoi_v = float(_stoi(s2, r2))
        sisdr = float(fast_bss_eval.si_sdr(torch.from_numpy(r2).float(),
                                           torch.from_numpy(s2).float()).item())
        s32 = sig[:n].astype(np.float32)
        peak = np.max(np.abs(s32)) + 1e-12
        if peak > 1.0:
            s32 = s32 / peak
        mos = dnsmos.run(s32, sr=16000)
        return dict(PESQ=pesq, STOI=stoi_v, SISDR=sisdr,
                    OVRL=mos['ovrl_mos'], SIG=mos['sig_mos'], BAK=mos['bak_mos'])

    print("\n===== 噪声先验变体对照（sample_5_clean_fileid_6162，截 960 样本）=====")
    hdr = f"{'variant':>12s} {'PESQ':>7s} {'STOI':>8s} {'SI-SDR':>8s} {'OVRL':>6s} {'SIG':>6s} {'BAK':>6s}"
    print(hdr); print('-' * len(hdr))
    for var in list(results.keys()):
        m = metrics(results[var]['enhanced'][960:])
        print(f"{var:>12s} {m['PESQ']:>7.2f} {m['STOI']:>8.4f} {m['SISDR']:>8.2f} "
              f"{m['OVRL']:>6.2f} {m['SIG']:>6.2f} {m['BAK']:>6.2f}")

    print("\n===== IVA bss 两路 SI-SDR（对 RTS，截 960）=====")
    hdr2 = f"{'variant':>12s} {'bss_src0':>9s} {'bss_src1':>9s}"
    print(hdr2); print('-' * len(hdr2))
    for var in list(results.keys()):
        bt = results[var]['bss'][0]; bi = results[var]['bss'][1]
        sf.write(os.path.join(output_dir, f"bss_src0_{var}.wav"), bt, 16000)
        sf.write(os.path.join(output_dir, f"bss_src1_{var}.wav"), bi, 16000)
        n = min(len(bt[960:]), len(ref[0]))
        r2 = ref[:, :n].astype(np.float64)
        st = float(fast_bss_eval.si_sdr(torch.from_numpy(r2).float(),
                                        torch.from_numpy(np.asarray(bt[960:960+n], dtype=np.float64)[None]).float()).item())
        si = float(fast_bss_eval.si_sdr(torch.from_numpy(r2).float(),
                                        torch.from_numpy(np.asarray(bi[960:960+n], dtype=np.float64)[None]).float()).item())
        print(f"{var:>12s} {st:>9.2f} {si:>9.2f}")


if __name__ == '__main__':
    main()
