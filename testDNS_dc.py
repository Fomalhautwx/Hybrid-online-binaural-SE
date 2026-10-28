import torch
import os
import sys
import torch.nn as nn
import soundfile as sf
import numpy as np
import timeit
import gc

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
from models.Paris_enhancement import GTCRN_PARIS
from models.gtcrn_binaural import GTCRN_Binaural
from models.gtcrnXiaobin import GTCRN
from models.gtcrn_paris_stream import StreamGTCRN
from models.AuxIVA_ISS_harmonic_v2 import AuxIVA_ISS_harmonic_nn_Frame_Module
from models.modules import Dereverberator
from trainers.modules import multich_stft
import argparse
import numbers
from speechmos import dnsmos
import onnxruntime
import cProfile, pstats, io

config_file = 'config.yaml'
ckpt_path = '/data/ssd0/jizhe.lu/exps/ivaiss_pow/checkpoints/model_epoch100.tar'
baseline_ckpt_path = '/data/ssd0/jizhe.lu/exps/dcgtcrn/checkpoints/model_epoch70.tar'
test_audio_dir = '/data/ssd0/jizhe.lu/Simulated/test_results'
iva_onnx = "/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/onnx_models/Ivaiss_simple.onnx"

parser = argparse.ArgumentParser("MODEL TEST")

# General config
parser.add_argument('--name', type=str, default='flow',
                    help='test name')
parser.add_argument('--limit', type=int, default=0,
                    help='只测前 N 条 (0=全部)，便于快速验证脚本改动')
parser.add_argument('--no_save_wav', action='store_true', default=False,
                    help='不保存增强音频（跑大规模评测时省磁盘）')
parser.add_argument('--eval_baseline', action='store_true', default=False,
                    help='同一条音频上再跑一遍纯 baseline 作对照（耗时约翻倍）')
parser.add_argument('--warmup_ratio', type=float, default=40.0,
                    help='丢弃前 1/warmup_ratio 的样本作 warmup；0=不丢弃')
parser.add_argument('--no_reset', action='store_true', default=False,
                    help='【对照实验用】不做逐条状态重置，复现原脚本行为')


def stream_inference(input, out_i, conv_cache, tra_cache, inter_cache,
                     derev_module, bss_module, onnx_session):
    '''
    input: numpy array, shape (B, 2, F, 1) complex or (B, 2, F, 1, 2) real/imag
    out_i: numpy array, shape (B, 1, F, 2) real/imag or (B, 1, F, 1, 2)
    conv_cache: numpy array, (2, 1, 32, 16, 65)
    tra_cache: numpy array, (2, 3, 1, 1, 32)
    inter_cache: numpy array, (2, 1, 65, 32)
    derev_module: Dereverberator instance
    bss_module: AuxIVA module
    onnx_session: onnxruntime session

    返回值末尾新增 bss_out —— 用于统计 IVA 前级本身的性能与通道错分情况
    '''
    if out_i is None:
        init_input = torch.from_numpy(input[:, 1:2, ...])  # (B, 1, F, 1) complex or (B,1,F,1,2)
    else:
        init_input = torch.from_numpy(out_i)

    if init_input.is_complex():
        r_iva = init_input
    else:
        r_iva = torch.view_as_complex(init_input.squeeze(-2))

    mag_prior = torch.abs(r_iva) # (B,1, F)

    input_ = torch.from_numpy(input)
    if input_.is_complex():
        input_c = input_
    else:
        input_c = torch.view_as_complex(input_.squeeze(-2))
    input_ = input_.squeeze(-1).transpose(1, 2) if input_.is_complex() else input_c.transpose(1, 2)

    dereved = derev_module.streaming(input_, mag_prior.to(derev_module.device))  # -> F D complex
    input_ = dereved.transpose(1, 2)  # b D F
    
    # ---- 加这一段 ----
    wpe_peak = input_.abs().max().item()
    input_peak = input_.abs().max().item()
    if wpe_peak/input_peak > 10:
        print(f"[WPE spike] frame peak={wpe_peak:.1f}")
    # ------------------
    bss_out = bss_module(input_, r_iva)
    
    # # ---- 和这一段 ----
    # bss_peak = bss_out.abs().max().item()
    # if bss_peak > 10 and wpe_peak < 10:
    #     print(f"[IVA spike] frame peak={bss_peak:.1f}, wpe was clean ({wpe_peak:.2f})")
    # # ------------------
    p = 0.3
    mag = bss_out.abs().clamp_min(1e-6)
    spec_iva_c = bss_out * mag.pow(p - 1.0)
    spec_iva_c = torch.view_as_real(spec_iva_c)
    if spec_iva_c.ndim == 4:
        spec_iva_c = spec_iva_c.unsqueeze(-2)
    spec_iva_c = np.ascontiguousarray(spec_iva_c.float().numpy().astype(np.float32))

    if np.iscomplexobj(input):
        ref_ri = np.stack([input[:, 0:1, ...].real, input[:, 0:1, ...].imag], axis=-1)
    else:
        ref_ri = np.ascontiguousarray(input[:, 0:1, :, None, :].astype(np.float32))

    out, conv_cache, tra_cache, inter_cache = onnx_session.run([], {
        'spec': spec_iva_c,
        'ref': ref_ri,
        'conv_cache': conv_cache,
        'tra_cache': tra_cache,
        'inter_cache': inter_cache})

    return out, conv_cache, tra_cache, inter_cache, bss_out


def normalize(sample):
    sample = sample.astype(np.float32)
    max_abs = np.max(np.abs(sample))
    if max_abs > 1.0:
        sample = sample / max_abs
    return sample


def spec_to_wav(spec_ri):
    """(..., F, T, 2) real/imag -> 波形"""
    return istft(spec_ri[..., 0] + 1j * spec_ri[..., 1], n_fft=512, hop_length=256,
                 win_length=512, window=np.hanning(512) ** 0.5)


def si_sdr_np(ref, est):
    """标量 SI-SDR；ref/est 均为 (1, samples) 的 numpy"""
    try:
        return float(fast_bss_eval.si_sdr(torch.as_tensor(ref).clone().double(),
                                          torch.as_tensor(est).clone().double()).squeeze())
    except Exception:
        return float('nan')


def safe_metric(fn, a, b):
    """调用 pesq/stoi scorer，异常时返回 nan 而不是中断整轮测试

    注意 float(...) 不可省：部分版本的 .item() 返回 np.float32，
    直接塞进 json.dumps 会抛 TypeError。
    """
    try:
        return float(fn(torch.as_tensor(a), torch.as_tensor(b)).item())
    except Exception:
        sig = np.asarray(a).squeeze()
        amax = float(np.abs(sig).max()) if sig.size else 0.0
        rms = float(np.sqrt(np.mean(sig ** 2))) if sig.size else 0.0
        print(f"[metric fail] {type(Exception).__name__}: {Exception} | "
              f"ref len={sig.shape[-1] / 16000:.2f}s max={amax:.2e} rms={rms:.2e}")
        return float('nan')
    #return float(fn(torch.as_tensor(a), torch.as_tensor(b)).item())


def to_py(obj):
    """递归把 numpy 标量/数组转成原生 Python 类型。

    必要性：STOI/PESQ scorer 的 .item() 在部分版本返回 np.float32，
    dnsmos.run() 的返回值同样是 numpy 标量，二者都无法被 json.dumps 序列化
    （TypeError: Object of type float32 is not JSON serializable）。
    统一在写盘前过一遍这个函数，比在每个取值点单独 float() 更不易漏。
    """
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.generic):        # np.float32 / np.float64 / np.int64 ...
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist() if obj.numel() > 1 else obj.item()
    return obj


def running_stats(values):
    """均值之外同时给出中位数/标准差/分位数——均值会被少数崩坏样本主导"""
    v = [x for x in values if not (isinstance(x, float) and np.isnan(x))]
    if len(v) == 0:
        return dict(mean=float('nan'), median=float('nan'), std=float('nan'),
                    p10=float('nan'), p90=float('nan'), n=0)
    a = np.asarray(v, dtype=np.float64)
    return dict(mean=float(a.mean()), median=float(np.median(a)), std=float(a.std()),
                p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)), n=int(a.size))


if __name__ == '__main__':
    arg = parser.parse_args()
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = GTCRN_Binaural(**config["network"]).to(device).eval()
    baseline.load_state_dict(torch.load(baseline_ckpt_path, map_location=device)['model'], strict=False)

    session = onnxruntime.InferenceSession(iva_onnx, None, providers=['CPUExecutionProvider'])
    buffer_size = 16

    with open('/home/nis/jizhe.lu/Code/2sderev/derev_params.json') as f:
        params_dict = json.load(f)
    params = params_dict['wpe_ha']
    dereverb = Dereverberator(nsrc=2, params=params)   # 实例化去混响模块
    dereverb.reset_state()
    dereverb.wpe.train = False
    dereverb.to(device)

    # IVA 构造参数集中管理，便于逐条重新实例化（见下方状态重置）
    IVA_KWARGS = dict(bandsize=257, nsrc=2, alpha=0.9999, alpha2=0, iss_iters=2)
    auxiva_iss = AuxIVA_ISS_harmonic_nn_Frame_Module(**IVA_KWARGS)

    output_dir = os.path.join(test_audio_dir, arg.name)
    os.makedirs(output_dir, exist_ok=True)
    tag = os.path.basename(ckpt_path).split('.')[0]
    output_json = os.path.join(test_audio_dir, f"test_results_{tag}.json")
    # ===== 即时落盘：逐条一行写入 JSONL + CSV，进程被杀也不丢已测结果 =====
    output_jsonl = os.path.join(test_audio_dir, f"test_results_{tag}_{arg.name}.jsonl")
    output_csv = os.path.join(test_audio_dir, f"test_results_{tag}_{arg.name}.csv")
    summary_txt = os.path.join(test_audio_dir, f"test_summary_{tag}_{arg.name}.txt")

    test_path = config['test_dataset']['md_file']
    with open(test_path, 'r') as f:
        test_md = json.load(f)

    cur_list = []
    for line in tqdm(test_md, desc="Loading test set"):
        cur_line = dict()
        cur_line['mixture'] = line['noisy']
        cur_line['target'] = line['RTS']
        cur_list.append(cur_line)

    if arg.limit > 0:
        cur_list = cur_list[:arg.limit]

    ttbar = pb(0, len(cur_list), 20)
    ttbar.start()
    ttime, mtime, cnt = 0., 0., 0

    # 指标容器：noisy / bss 两路 / enhanced / baseline 全都留痕
    keys = ['mix_sisdr', 'mix_stoi', 'mix_pesq',
            'est_sisdr', 'est_stoi', 'est_pesq',
            'base_sisdr', 'base_stoi', 'base_pesq']
    M = {k: [] for k in keys}
    est_mos = np.array([0.0, 0.0, 0.0, 0.0])

    pesq = pesq_scorer().cuda()
    stoi = stoi_scorer().cuda()

    csv_cols = ['idx', 'name', 'mix_sisdr', 'mix_stoi', 'mix_pesq',
                'est_sisdr', 'est_stoi', 'est_pesq',
                'base_sisdr', 'base_stoi', 'base_pesq',
                'sig', 'bak', 'ovrl', 'p808', 'time_s']
    f_jsonl = open(output_jsonl, 'w', buffering=1)   # 行缓冲：写一行刷一次盘
    f_csv = open(output_csv, 'w', buffering=1)
    f_csv.write(','.join(csv_cols) + '\n')

    try:
        for test_sample in cur_list:
            cnt += 1
            start = timeit.default_timer()
            mix_wav_path = test_sample['mixture']
            ref_wav_path = test_sample['target']

            mix, _ = sf.read(mix_wav_path)   # dual channel
            ref, _ = sf.read(ref_wav_path)   # single
            mix = mix.astype('float32')
            ref = ref.astype('float32')
            x = torch.from_numpy(mix)
            x = multich_stft(x.T.unsqueeze(0))
            outputs = baseline(x.cuda()).detach().cpu().numpy()
            #enh = spec_to_wav(outputs)
            enh = istft(outputs, n_fft=512, hop_length=256, win_length=512, window=np.hanning(512) ** 0.5)

            wav_len = enh.shape[-1]
            mix_ref = mix[:wav_len, 0:1].T
            est = enh.squeeze(0)
            ref_1 = np.expand_dims(ref[:wav_len], axis=0)

            # warmup 截断：对所有被比较的信号施加同一截断，保持时间对齐
            cut = int(est.shape[1] / arg.warmup_ratio) if arg.warmup_ratio > 0 else 0
            est_c, ref_c, mix_c = est[:, cut:], ref_1[:, cut:], mix_ref[:, cut:]

            # ---------------- 指标 ----------------
            # (1) noisy 基线：原脚本注释掉了，没有它就无法回答"到底改善了多少"
            mix_sisdr = si_sdr_np(ref_c, mix_c)
            mix_stoi_v = safe_metric(stoi, mix_c, ref_c)
            mix_pesq_v = safe_metric(pesq, mix_c, ref_c)
            # (2) enhanced
            est_sisdr = si_sdr_np(ref_c, est_c)
            est_stoi_v = safe_metric(stoi, est_c, ref_c)
            est_pesq_v = safe_metric(pesq, est_c, ref_c)
            
            # (4) 可选：同一条音频上的纯 baseline 对照
            base_sisdr = base_stoi_v = base_pesq_v = float('nan')
            if arg.eval_baseline:
                with torch.no_grad():
                    base_out = [torch.view_as_real(
                        baseline(torch.from_numpy(x[..., i:i + 1].copy()))).numpy()
                        for i in range(x.shape[-1])]
                base_out = np.concatenate(base_out, axis=3)
                base_wav = spec_to_wav(base_out).squeeze(0)[:, :wav_len][:, cut:]
                base_sisdr = si_sdr_np(ref_c, base_wav)
                base_stoi_v = safe_metric(stoi, base_wav, ref_c)
                base_pesq_v = safe_metric(pesq, base_wav, ref_c)

            try:
                enhanced_mos = dnsmos.run(normalize(est_c.squeeze()), 16000)
            except Exception as e:
                import traceback
                print(f"[FAILED] idx={cnt} {os.path.basename(mix_wav_path)}: {type(e).__name__}: {e}")
                traceback.print_exc()
                f_jsonl.write(json.dumps({'idx': cnt, 'noisy': mix_wav_path,
                                        'error': f'{type(e).__name__}: {e}'}) + '\n')
                continue
            # dnsmos 返回 numpy 标量 -> 统一转 Python float，否则写 json 时报
            # TypeError: Object of type float32 is not JSON serializable
            enhanced_mos = {k: float(v) for k, v in enhanced_mos.items()}
            est_mos += np.array([enhanced_mos['sig_mos'], enhanced_mos['bak_mos'],
                                 enhanced_mos['ovrl_mos'], enhanced_mos['p808_mos']])

            for k, v in zip(keys, [mix_sisdr, mix_stoi_v, mix_pesq_v,
                                   est_sisdr, est_stoi_v, est_pesq_v,
                                   base_sisdr, base_stoi_v, base_pesq_v]):
                M[k].append(v)

            save_wav_path = ""
            if not arg.no_save_wav:
                base = os.path.splitext(os.path.basename(mix_wav_path))[0]   # 去掉扩展名
                ext = ".wav"
                sf.write(os.path.join(output_dir, f"{base}_est{ext}"),
                        est.T, 16000)
                sf.write(os.path.join(output_dir, f"{base}_noisy{ext}"),
                        mix[cut:, 0], 16000)            # 单声道
                sf.write(os.path.join(output_dir, f"{base}_ref{ext}"),
                        ref[cut:], 16000)               # 单声道
            curr_time = timeit.default_timer() - start
            ttime += curr_time
            mtime = ttime / cnt

            # ---------------- 即时落盘 ----------------
            info = {
                'idx': cnt, 'noisy': mix_wav_path, 'target': ref_wav_path,
                'out_path': save_wav_path,
                'sisdr': {'mix': mix_sisdr, 'est': est_sisdr, 'base': base_sisdr},
                'stoi': {'mix': mix_stoi_v, 'est': est_stoi_v, 'base': base_stoi_v},
                'pesq': {'mix': mix_pesq_v, 'est': est_pesq_v, 'base': base_pesq_v},
                'dnsmos': {'sig': enhanced_mos['sig_mos'], 'bak': enhanced_mos['bak_mos'],
                           'ovrl': enhanced_mos['ovrl_mos'], 'p808': enhanced_mos['p808_mos']},
                'time_s': curr_time,
            }
            f_jsonl.write(json.dumps(to_py(info), ensure_ascii=False) + '\n')
            row = [cnt, os.path.basename(mix_wav_path), mix_sisdr, mix_stoi_v, mix_pesq_v,
                   est_sisdr, est_stoi_v, est_pesq_v, base_sisdr, base_stoi_v, base_pesq_v,
                   enhanced_mos['sig_mos'], enhanced_mos['bak_mos'],
                   enhanced_mos['ovrl_mos'], enhanced_mos['p808_mos'], curr_time]
            # 用 numbers.Real 而非 isinstance(v, float)：后者对 np.float32 返回 False，
            # 会让数值被 str() 成 "0.8973000526428223" 之类的长串；整数列保持整数形式
            def _fmt(v):
                if isinstance(v, bool) or not isinstance(v, numbers.Real):
                    return str(v)
                return str(int(v)) if isinstance(v, numbers.Integral) else f'{float(v):.4f}'
            f_csv.write(','.join(_fmt(v) for v in row) + '\n')

            # 进度条实时显示 running mean 与错分率，异常可提早发现并中断
            ttbar.update_progress(int(cnt), 'test',
                                  'ct/mt={:.2f}/{:.2f} SISDR={:.2f} STOI={:.3f}'.format(
                                      curr_time, mtime,
                                      np.nanmean(M['est_sisdr']), np.nanmean(M['est_stoi'])))

            del x, outputs, enh, est, mix, ref
            torch.cuda.empty_cache()
            gc.collect()
    finally:
        f_jsonl.close()
        f_csv.close()

    ttbar.finish()
    est_mos /= max(cnt, 1)

    # ---------------- 汇总：均值之外给出分布 ----------------
    lines = [f"测试集: {len(cur_list)} 条 | ckpt: {tag} | name: {arg.name} | "
             f"逐条重置: {'否(复现原行为)' if arg.no_reset else '是'}",
             f"平均耗时: {mtime:.3f} s/条", ""]
    hdr = f"{'指标':<18}{'mean':>9}{'median':>9}{'std':>8}{'p10':>9}{'p90':>9}{'n':>6}"
    lines += [hdr, '-' * len(hdr)]
    for key, label in [('mix_sisdr', 'noisy SI-SDR'), ('est_sisdr', 'enhanced SI-SDR'),
                       ('base_sisdr', 'baseline SI-SDR'),
                       ('mix_stoi', 'noisy STOI'),
                       ('est_stoi', 'enhanced STOI'), ('base_stoi', 'baseline STOI'),
                       ('mix_pesq', 'noisy PESQ'), 
                       ('est_pesq', 'enhanced PESQ'), ('base_pesq', 'baseline PESQ')]:
        s = running_stats(M[key])
        if s['n'] == 0:
            continue
        lines.append(f"{label:<18}{s['mean']:>9.4f}{s['median']:>9.4f}{s['std']:>8.3f}"
                     f"{s['p10']:>9.4f}{s['p90']:>9.4f}{s['n']:>6d}")
    lines += ["", f"DNSMOS  SIG {est_mos[0]:.2f} | BAK {est_mos[1]:.2f} | "
                  f"OVRL {est_mos[2]:.2f} | P808 {est_mos[3]:.2f}"]

    # 通道错分诊断：直接回答"test 远低于单条测试"的头号嫌疑
    es = np.asarray([v for v in M['est_sisdr'] if not np.isnan(v)])
    if es.size:
        worst = np.argsort(es)[:5] + 1
        lines.append(f"[分布诊断] enhanced SI-SDR < 0 占比 {100.0 * np.mean(es < 0):.1f}%；"
                     f"最差 5 条 idx: {list(worst)}")

    summary = '\n'.join(lines)
    print('\n' + summary)
    with open(summary_txt, 'w') as f:
        f.write(summary + '\n')

    with open(output_jsonl) as f:
        results = [json.loads(l) for l in f if l.strip()]
    with open(output_json, 'w') as f:
        json.dump({'summary': summary, 'per_file': results}, f, indent=4, ensure_ascii=False)

    print(f"\n逐条结果: {output_jsonl}\n           {output_csv}\n汇总: {summary_txt}")
