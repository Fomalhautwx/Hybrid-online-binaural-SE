"""
流式 DC-GTCRN 基线测试
======================
WPE+IVA+GTCRN 级联管线的公平对照：同一 GTCRN 架构（dcgtcrn 权重），
逐帧流式推理（ONNX, 因果 inter-RNN, 每帧只见过往），输入为原始双通道混合。
与 testDNS_dc.py（离线整段推理）的指标协议完全一致，唯一区别是推理方式。

产出与 test_results_model_epoch100_ivaiss_pow.jsonl 逐条对齐，
可直接做 "流式管线 vs 流式纯GTCRN" 的逐样本对比。
"""
import torch
import os
import sys
import torch.nn as nn
import soundfile as sf
import numpy as np
import timeit
import gc

import json
import yaml
import numbers
from tqdm import tqdm
from utils.progressbar import progressbar as pb
from metrics import pesq_scorer, stoi_scorer
import fast_bss_eval
from trainers.modules import multich_stft
from librosa import istft
from models.gtcrn_binaural import GTCRN_Binaural
from models.gtcrn_paris_stream import StreamGTCRN
from models.modules import convert_to_stream
import argparse
from speechmos import dnsmos
import onnxruntime

config_file = 'config.yaml'
ckpt_path = '/data/hdd0/jizhe.lu/exps/dcgtcrn/checkpoints/model_epoch68.tar'
test_audio_dir = '/data/ssd0/jizhe.lu/Simulated/test_results'
onnx_raw_path = 'onnx_models/dcgtcrn_stream.onnx'
onnx_path = 'onnx_models/dcgtcrn_stream_simple.onnx'

parser = argparse.ArgumentParser("STREAMING DC-GTCRN BASELINE TEST")

# General config
parser.add_argument('--name', type=str, default='dc_stream',
                    help='test name')
parser.add_argument('--limit', type=int, default=0,
                    help='只测前 N 条 (0=全部)，便于快速验证脚本改动')
parser.add_argument('--no_save_wav', action='store_true', default=False,
                    help='不保存增强音频（跑大规模评测时省磁盘）')
parser.add_argument('--warmup_ratio', type=float, default=40.0,
                    help='丢弃前 1/warmup_ratio 的样本作 warmup；0=不丢弃')
parser.add_argument('--no_reset', action='store_true', default=False,
                    help='【对照实验用】不做逐条状态重置，状态跨样本继承')


def export_onnx(stream_model):
    """把流式权重导出为 ONNX（已存在则跳过）。返回 session。"""
    os.makedirs('onnx_models', exist_ok=True)
    if not os.path.exists(onnx_path):
        import onnx
        from onnxsim import simplify
        device = next(stream_model.parameters()).device
        spec = torch.randn(1, 2, 257, 1, 2, device=device)
        ref = torch.randn(1, 1, 257, 1, 2, device=device)
        conv_cache = torch.zeros(2, 1, 32, 16, 65, device=device)
        tra_cache = torch.zeros(2, 3, 1, 1, 32, device=device)
        inter_cache = torch.zeros(2, 1, 65, 32, device=device)
        torch.onnx.export(stream_model,
                          (spec, ref, conv_cache, tra_cache, inter_cache),
                          onnx_raw_path,
                          input_names=['spec', 'ref', 'conv_cache', 'tra_cache', 'inter_cache'],
                          output_names=['spec_enh', 'conv_cache_out', 'tra_cache_out', 'inter_cache_out'],
                          opset_version=11,
                          dynamic_axes=None,
                          verbose=False)
        onnx_model = onnx.load(onnx_raw_path)
        onnx.checker.check_model(onnx_model)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, onnx_path)
        print(f"ONNX exported -> {onnx_path}")
    return onnxruntime.InferenceSession(onnx_path, None, providers=['CPUExecutionProvider'])


def check_torch_vs_onnx(stream_model, session, frames=5):
    """torch 流式模型 vs ONNX 数值一致性抽查（抓导出/权重转换 bug）。"""
    device = next(stream_model.parameters()).device
    spec = torch.randn(1, 2, 257, 1, 2, device=device) * 0.1
    ref = torch.randn(1, 1, 257, 1, 2, device=device) * 0.1
    conv_cache = torch.zeros(2, 1, 32, 16, 65, device=device)
    tra_cache = torch.zeros(2, 3, 1, 1, 32, device=device)
    inter_cache = torch.zeros(2, 1, 65, 32, device=device)
    np_cc = np.zeros([2, 1, 32, 16, 65], dtype="float32")
    np_tc = np.zeros([2, 3, 1, 1, 32], dtype="float32")
    np_ic = np.zeros([2, 1, 65, 32], dtype="float32")
    max_diff = 0.0
    with torch.no_grad():
        for _ in range(frames):
            out_t, conv_cache, tra_cache, inter_cache = stream_model(
                spec, ref, conv_cache, tra_cache, inter_cache)
            out_o, np_cc, np_tc, np_ic = session.run([], {
                'spec': np.ascontiguousarray(spec.numpy()),
                'ref': np.ascontiguousarray(ref.numpy()),
                'conv_cache': np_cc, 'tra_cache': np_tc, 'inter_cache': np_ic})
            max_diff = max(max_diff, float(np.abs(out_t.numpy() - out_o).max()))
    print(f"[check] torch vs onnx max abs diff over {frames} frames: {max_diff:.3e}")
    if max_diff > 1e-4:
        print("[check] WARNING: torch/onnx 输出不一致，请检查导出与权重转换")
    return max_diff


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
    try:
        return float(fn(torch.as_tensor(a), torch.as_tensor(b)).item())
    except Exception:
        sig = np.asarray(a).squeeze()
        amax = float(np.abs(sig).max()) if sig.size else 0.0
        rms = float(np.sqrt(np.mean(sig ** 2))) if sig.size else 0.0
        print(f"[metric fail] {type(Exception).__name__}: {Exception} | "
              f"ref len={sig.shape[-1] / 16000:.2f}s max={amax:.2e} rms={rms:.2e}")
        return float('nan')


def to_py(obj):
    """递归把 numpy 标量/数组转成原生 Python 类型"""
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist() if obj.numel() > 1 else obj.item()
    return obj


def running_stats(values):
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

    device = torch.device("cpu")

    # ---- 权重加载 + 流式转换 + ONNX 导出（与 full_inspect.py 同一套流程）----
    model = GTCRN_Binaural().to(device).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'], strict=False)
    stream_model = StreamGTCRN().to(device).eval()
    convert_to_stream(stream_model, model)
    session = export_onnx(stream_model)
    check_torch_vs_onnx(stream_model, session)

    output_dir = os.path.join(test_audio_dir, arg.name)
    os.makedirs(output_dir, exist_ok=True)
    tag = os.path.basename(ckpt_path).split('.')[0]
    output_json = os.path.join(test_audio_dir, f"test_results_{tag}.json")
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

    keys = ['mix_sisdr', 'mix_stoi', 'mix_pesq',
            'est_sisdr', 'est_stoi', 'est_pesq']
    M = {k: [] for k in keys}
    est_mos = np.array([0.0, 0.0, 0.0, 0.0])

    pesq = pesq_scorer().cuda()
    stoi = stoi_scorer().cuda()

    csv_cols = ['idx', 'name', 'mix_sisdr', 'mix_stoi', 'mix_pesq',
                'est_sisdr', 'est_stoi', 'est_pesq',
                'sig', 'bak', 'ovrl', 'p808', 'time_s']
    f_jsonl = open(output_jsonl, 'w', buffering=1)
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

            # ============ 逐条重置全部流式状态（与 testDNS_fixed.py 同协议）============
            if not arg.no_reset:
                conv_cache = np.zeros([2, 1, 32, 16, 65], dtype="float32")
                tra_cache = np.zeros([2, 3, 1, 1, 32], dtype="float32")
                inter_cache = np.zeros([2, 1, 65, 32], dtype="float32")
            elif cnt == 1:
                conv_cache = np.zeros([2, 1, 32, 16, 65], dtype="float32")
                tra_cache = np.zeros([2, 3, 1, 1, 32], dtype="float32")
                inter_cache = np.zeros([2, 1, 65, 32], dtype="float32")
            # =====================================================

            inputs = x.numpy()   # complex (1, 2, 257, T)
            outputs = []
            for i in range(inputs.shape[-1]):
                frame = torch.view_as_real(
                    torch.from_numpy(inputs[..., i:i + 1].copy())).numpy()   # (1,2,257,1,2)
                frame = np.ascontiguousarray(frame.astype(np.float32))
                out_i, conv_cache, tra_cache, inter_cache = session.run([], {
                    'spec': frame,
                    'ref': frame[:, 0:1, ...],     # 原始 ch0，与离线 GTCRN_Binaural 的 ref 一致
                    'conv_cache': conv_cache,
                    'tra_cache': tra_cache,
                    'inter_cache': inter_cache})
                outputs.append(out_i)
            outputs = np.concatenate(outputs, axis=3)   # (1,1,257,T,2)
            enh = spec_to_wav(outputs)

            wav_len = enh.shape[-1]
            mix_ref = mix[:wav_len, 0:1].T
            est = enh.squeeze(0)
            ref_1 = np.expand_dims(ref[:wav_len], axis=0)

            # warmup 截断：与管线/离线 dc 测试同一截断，保持时间对齐
            cut = int(est.shape[1] / arg.warmup_ratio) if arg.warmup_ratio > 0 else 0
            est_c, ref_c, mix_c = est[:, cut:], ref_1[:, cut:], mix_ref[:, cut:]

            # ---------------- 指标 ----------------
            mix_sisdr = si_sdr_np(ref_c, mix_c)
            mix_stoi_v = safe_metric(stoi, mix_c, ref_c)
            mix_pesq_v = safe_metric(pesq, mix_c, ref_c)
            est_sisdr = si_sdr_np(ref_c, est_c)
            est_stoi_v = safe_metric(stoi, est_c, ref_c)
            est_pesq_v = safe_metric(pesq, est_c, ref_c)

            try:
                enhanced_mos = dnsmos.run(normalize(est_c.squeeze()), 16000)
            except Exception as e:
                import traceback
                print(f"[FAILED] idx={cnt} {os.path.basename(mix_wav_path)}: {type(e).__name__}: {e}")
                traceback.print_exc()
                f_jsonl.write(json.dumps({'idx': cnt, 'noisy': mix_wav_path,
                                        'error': f'{type(e).__name__}: {e}'}) + '\n')
                continue
            enhanced_mos = {k: float(v) for k, v in enhanced_mos.items()}
            est_mos += np.array([enhanced_mos['sig_mos'], enhanced_mos['bak_mos'],
                                 enhanced_mos['ovrl_mos'], enhanced_mos['p808_mos']])

            for k, v in zip(keys, [mix_sisdr, mix_stoi_v, mix_pesq_v,
                                   est_sisdr, est_stoi_v, est_pesq_v]):
                M[k].append(v)

            save_wav_path = ""
            if not arg.no_save_wav:
                base = os.path.splitext(os.path.basename(mix_wav_path))[0]
                ext = ".wav"
                sf.write(os.path.join(output_dir, f"{base}_est{ext}"), est.T, 16000)
                sf.write(os.path.join(output_dir, f"{base}_noisy{ext}"),
                        mix[:, 0], 16000)
                sf.write(os.path.join(output_dir, f"{base}_ref{ext}"),
                        ref, 16000)

            curr_time = timeit.default_timer() - start
            ttime += curr_time
            mtime = ttime / cnt

            # ---------------- 即时落盘 ----------------
            info = {
                'idx': cnt, 'noisy': mix_wav_path, 'target': ref_wav_path,
                'out_path': save_wav_path,
                'sisdr': {'mix': mix_sisdr, 'est': est_sisdr},
                'stoi': {'mix': mix_stoi_v, 'est': est_stoi_v},
                'pesq': {'mix': mix_pesq_v, 'est': est_pesq_v},
                'dnsmos': {'sig': enhanced_mos['sig_mos'], 'bak': enhanced_mos['bak_mos'],
                           'ovrl': enhanced_mos['ovrl_mos'], 'p808': enhanced_mos['p808_mos']},
                'time_s': curr_time,
            }
            f_jsonl.write(json.dumps(to_py(info), ensure_ascii=False) + '\n')
            row = [cnt, os.path.basename(mix_wav_path), mix_sisdr, mix_stoi_v, mix_pesq_v,
                   est_sisdr, est_stoi_v, est_pesq_v,
                   enhanced_mos['sig_mos'], enhanced_mos['bak_mos'],
                   enhanced_mos['ovrl_mos'], enhanced_mos['p808_mos'], curr_time]

            def _fmt(v):
                if isinstance(v, bool) or not isinstance(v, numbers.Real):
                    return str(v)
                return str(int(v)) if isinstance(v, numbers.Integral) else f'{float(v):.4f}'
            f_csv.write(','.join(_fmt(v) for v in row) + '\n')

            ttbar.update_progress(int(cnt), 'test',
                                  'ct/mt={:.2f}/{:.2f} SISDR={:.2f} STOI={:.3f}'.format(
                                      curr_time, mtime,
                                      np.nanmean(M['est_sisdr']), np.nanmean(M['est_stoi'])))

            del x, inputs, outputs, enh, est, mix, ref
            torch.cuda.empty_cache()
            gc.collect()
    finally:
        f_jsonl.close()
        f_csv.close()

    ttbar.finish()
    est_mos /= max(cnt, 1)

    lines = [f"测试集: {len(cur_list)} 条 | ckpt: {tag} | name: {arg.name} | "
             f"逐条重置: {'否(复现原行为)' if arg.no_reset else '是'} | "
             f"推理方式: 逐帧流式 ONNX（因果）",
             f"平均耗时: {mtime:.3f} s/条", ""]
    hdr = f"{'指标':<18}{'mean':>9}{'median':>9}{'std':>8}{'p10':>9}{'p90':>9}{'n':>6}"
    lines += [hdr, '-' * len(hdr)]
    for key, label in [('mix_sisdr', 'noisy SI-SDR'), ('est_sisdr', 'enhanced SI-SDR'),
                       ('mix_stoi', 'noisy STOI'), ('est_stoi', 'enhanced STOI'),
                       ('mix_pesq', 'noisy PESQ'), ('est_pesq', 'enhanced PESQ')]:
        s = running_stats(M[key])
        if s['n'] == 0:
            continue
        lines.append(f"{label:<18}{s['mean']:>9.4f}{s['median']:>9.4f}{s['std']:>8.3f}"
                     f"{s['p10']:>9.4f}{s['p90']:>9.4f}{s['n']:>6d}")
    lines += ["", f"DNSMOS  SIG {est_mos[0]:.2f} | BAK {est_mos[1]:.2f} | "
                  f"OVRL {est_mos[2]:.2f} | P808 {est_mos[3]:.2f}"]

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
