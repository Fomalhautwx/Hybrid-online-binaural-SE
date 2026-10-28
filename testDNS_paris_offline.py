"""
离线全管线（训练条件）上界测试
================================
用 GTCRN_PARIS 以训练时的验证条件跑整段推理：
    batch WPE(去混响) + IVA(ChunkForward, teacher 先验) + GTCRN 子网络，整段一次前向。

这是级联管线在"前级非流式、先验来自 teacher GTCRN"下的上界。
与 dc 基线（test_results_model_epoch100_dc.jsonl, SI-SDR 5.68）对比，回答：
    管线在训练条件本身能否超过纯 DC-GTCRN？
    - 能  -> 损失全在流式前级（自身先验 / alpha 0.9999 / OOD warmup），修前级即可
    - 不能 -> 接口架构有损（mask 由 IVA 特征估计却作用于原始混合），需要改接口重训

同时输出 stage1（teacher 先验下的第一遍输出）指标：
    stage1 使用与 stage2 相同的 GTCRN 子网络，但 IVA 先验来自 teacher，
    用于定位"子网络在理想先验下自身的能力上限"。

批处理：按音频时长分组、等长样本组成 batch（WPE 逐帧循环的 GPU 启动开销被摊销）。
尾部不足 batch 的用同组样本重复填充——批内各样本状态相互独立（forward_spec 每次
调用重置 IVA，WPE 按通道独立更新），重复样本的输出直接丢弃，不影响真实样本结果。

指标协议与 testDNS_dc.py / testDNS_fixed.py 完全一致（同一 warmup 截断、同一 scorer）。
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
from collections import defaultdict
from tqdm import tqdm
from utils.progressbar import progressbar as pb
from metrics import pesq_scorer, stoi_scorer
import fast_bss_eval
from trainers.modules import multich_stft
from librosa import istft
from models.Paris_enhancement import GTCRN_PARIS
import argparse
from speechmos import dnsmos

config_file = 'config.yaml'
ckpt_path = '/data/ssd0/jizhe.lu/exps/ivaiss_pow/checkpoints/model_epoch100.tar'
PRIOR_CKPT = '/data/ssd0/jizhe.lu/exps/dcgtcrn/checkpoints/model_epoch68.tar'
test_audio_dir = '/data/ssd0/jizhe.lu/Simulated/test_results'

parser = argparse.ArgumentParser("OFFLINE FULL-PIPELINE (TRAINING CONDITION) TEST")

parser.add_argument('--name', type=str, default='paris_offline',
                    help='test name')
parser.add_argument('--ckpt', type=str, default=ckpt_path,
                    help='被测权重；默认 ivaiss_pow epoch100')
parser.add_argument('--mask_ref', type=str, default='raw', choices=['raw', 'iva', 'iva_dual'],
                    help="mask 作用域：'raw'=旧行为(原始 ch0)，'iva'=方案A(未压缩 IVA 目标路)。"
                         '必须与被测权重的训练配置一致，否则结果无意义')
parser.add_argument('--prior_type', type=str, default='teacher', choices=['teacher', 'mixture_mag'],
                    help="IVA/WPE 先验来源，必须与被测权重的训练配置一致")
parser.add_argument('--prior_ckpt', type=str, default=PRIOR_CKPT,
                    help='teacher 先验网络权重；默认 new_loss ep68。'
                         '注意：pfa_oracle 系训练用 dcgtcrn ep68 当 teacher，评测时必须指到它')
parser.add_argument('--limit', type=int, default=0,
                    help='只测前 N 条 (0=全部)')
parser.add_argument('--batch', type=int, default=8,
                    help='批大小（同长样本分组；尾部重复填充）')
parser.add_argument('--device', type=str, default='cuda',
                    help='推理设备：cuda 或 cpu（GPU 被占时可用 cpu 跑）')
parser.add_argument('--no_save_wav', action='store_true', default=False,
                    help='不保存增强音频')
parser.add_argument('--warmup_ratio', type=float, default=40.0,
                    help='丢弃前 1/warmup_ratio 的样本作 warmup；0=不丢弃')


def normalize(sample):
    sample = sample.astype(np.float32)
    max_abs = np.max(np.abs(sample))
    if max_abs > 1.0:
        sample = sample / max_abs
    return sample


def spec_to_wav(spec_c):
    """(F, T) complex -> 波形"""
    return istft(spec_c, n_fft=512, hop_length=256, win_length=512,
                 window=np.hanning(512) ** 0.5)


def si_sdr_np(ref, est):
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


def build_batches(cur_list, batch_size):
    """按混合音频帧数分组（同组 T 相同），组内按 batch_size 切块，尾部重复填充。"""
    groups = defaultdict(list)
    for s in cur_list:
        groups[sf.info(s['mixture']).frames].append(s)
    batches = []
    for items in groups.values():
        for k in range(0, len(items), batch_size):
            chunk = items[k:k + batch_size]
            n_orig = len(chunk)
            while len(chunk) < batch_size:          # 重复填充（输出会被丢弃）
                chunk.append(chunk[0])
            batches.append((chunk, n_orig))
    return batches


if __name__ == '__main__':
    arg = parser.parse_args()
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # ---- 模型：训练条件（teacher 先验 + batch WPE/IVA），eval 模式 ----
    network_cfg = dict(config['network'])
    network_cfg['prior_ckpt'] = arg.prior_ckpt   # 评测 teacher 必须与被测权重训练时一致
    network_cfg['device'] = arg.device
    network_cfg['mask_ref'] = arg.mask_ref   # 必须与被测权重的训练配置一致
    network_cfg['prior_type'] = arg.prior_type  # 必须与被测权重的训练配置一致
    if arg.mask_ref == 'iva_dual':
        # config.yaml 没有 mask2_gate_init；双掩码模型必须注册 gate 参数才能从 ckpt 载入其值
        network_cfg.setdefault('mask2_gate_init', 0.0)
    model = GTCRN_PARIS(**network_cfg, params=network_cfg).to(arg.device).eval()
    sd = torch.load(arg.ckpt, map_location=arg.device)['model']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {arg.ckpt} (mask_ref={arg.mask_ref})\n"
          f"  missing={len(missing)} unexpected={len(unexpected)}")
    print(f"  missing 样例: {missing[:5]}")

    output_dir = os.path.join(test_audio_dir, arg.name)
    os.makedirs(output_dir, exist_ok=True)
    tag = os.path.basename(arg.ckpt).split('.')[0]
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

    batches = build_batches(cur_list, arg.batch)
    print(f"测试集 {len(cur_list)} 条 -> {len(batches)} 批 (batch={arg.batch})")

    ttbar = pb(0, len(cur_list), 20)
    ttbar.start()
    ttime, mtime, cnt = 0., 0., 0

    keys = ['mix_sisdr', 'mix_stoi', 'mix_pesq',
            'est_sisdr', 'est_stoi', 'est_pesq',
            'st1_sisdr', 'st1_stoi', 'st1_pesq']
    M = {k: [] for k in keys}
    est_mos = np.array([0.0, 0.0, 0.0, 0.0])

    pesq = pesq_scorer().to(arg.device)
    stoi = stoi_scorer().to(arg.device)

    csv_cols = ['idx', 'name', 'mix_sisdr', 'mix_stoi', 'mix_pesq',
                'est_sisdr', 'est_stoi', 'est_pesq',
                'st1_sisdr', 'st1_stoi', 'st1_pesq',
                'sig', 'bak', 'ovrl', 'p808', 'time_s']
    f_jsonl = open(output_jsonl, 'w', buffering=1)
    f_csv = open(output_csv, 'w', buffering=1)
    f_csv.write(','.join(csv_cols) + '\n')

    try:
        for chunk, n_orig in batches:
            start = timeit.default_timer()

            # ---- 加载批内样本并组 batch ----
            mix_specs, truth_specs, mixes, refs = [], [], [], []
            for s in chunk:
                mix, _ = sf.read(s['mixture'])   # dual channel
                ref, _ = sf.read(s['target'])    # single
                mix = mix.astype('float32')
                ref = ref.astype('float32')
                x = multich_stft(torch.from_numpy(mix).T.unsqueeze(0))     # (1,2,257,T)
                truth = multich_stft(torch.from_numpy(ref).unsqueeze(0))   # (1,1,257,Tt)
                T = min(x.shape[-1], truth.shape[-1])
                mixes.append(mix); refs.append(ref)
                mix_specs.append(x[0, :, :, :T])       # (2,257,T) 再去掉 batch 维
                truth_specs.append(truth[0, :, :, :T]) # (1,257,T)

            x_b = torch.stack(mix_specs, 0)       # (B,2,257,T)
            truth_b = torch.stack(truth_specs, 0) # (B,1,257,T)

            try:
                with torch.no_grad():
                    est_b, stage1_b = model(x_b.to(arg.device), truth_b.to(arg.device))
                est_b = est_b.cpu().numpy()
                stage1_b = stage1_b.cpu().numpy()
                enh_list = [spec_to_wav(est_b[j, 0]) for j in range(n_orig)]
                st1_list = [spec_to_wav(stage1_b[j, 0]) for j in range(n_orig)]
            except Exception as e:
                import traceback
                traceback.print_exc()
                for j in range(n_orig):
                    cnt += 1
                    s = chunk[j]
                    print(f"[FAILED] idx={cnt} {os.path.basename(s['mixture'])}: {type(e).__name__}: {e}")
                    f_jsonl.write(json.dumps({'idx': cnt, 'noisy': s['mixture'],
                                            'error': f'{type(e).__name__}: {e}'}) + '\n')
                continue

            batch_time = timeit.default_timer() - start

            # ---- 逐条指标 ----
            for j in range(n_orig):
                cnt += 1
                s = chunk[j]
                mix, ref = mixes[j], refs[j]
                enh, st1 = enh_list[j], st1_list[j]

                wav_len = enh.shape[-1]
                mix_ref = mix[:wav_len, 0:1].T
                ref_1 = np.expand_dims(ref[:wav_len], axis=0)

                cut = int(wav_len / arg.warmup_ratio) if arg.warmup_ratio > 0 else 0
                est_c, st1_c, ref_c, mix_c = enh[cut:], st1[cut:], ref_1[:, cut:], mix_ref[:, cut:]

                mix_sisdr = si_sdr_np(ref_c, mix_c)
                mix_stoi_v = safe_metric(stoi, mix_c, ref_c)
                mix_pesq_v = safe_metric(pesq, mix_c, ref_c)
                est_sisdr = si_sdr_np(ref_c, est_c[None, :])
                est_stoi_v = safe_metric(stoi, est_c[None, :], ref_c)
                est_pesq_v = safe_metric(pesq, est_c[None, :], ref_c)
                st1_sisdr = si_sdr_np(ref_c, st1_c[None, :])
                st1_stoi_v = safe_metric(stoi, st1_c[None, :], ref_c)
                st1_pesq_v = safe_metric(pesq, st1_c[None, :], ref_c)

                try:
                    enhanced_mos = dnsmos.run(normalize(est_c), 16000)
                except Exception as e:
                    import traceback
                    print(f"[FAILED] idx={cnt} {os.path.basename(s['mixture'])}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    f_jsonl.write(json.dumps({'idx': cnt, 'noisy': s['mixture'],
                                            'error': f'{type(e).__name__}: {e}'}) + '\n')
                    continue
                enhanced_mos = {k: float(v) for k, v in enhanced_mos.items()}
                est_mos += np.array([enhanced_mos['sig_mos'], enhanced_mos['bak_mos'],
                                     enhanced_mos['ovrl_mos'], enhanced_mos['p808_mos']])

                for k, v in zip(keys, [mix_sisdr, mix_stoi_v, mix_pesq_v,
                                       est_sisdr, est_stoi_v, est_pesq_v,
                                       st1_sisdr, st1_stoi_v, st1_pesq_v]):
                    M[k].append(v)

                save_wav_path = ""
                if not arg.no_save_wav:
                    base = os.path.splitext(os.path.basename(s['mixture']))[0]
                    sf.write(os.path.join(output_dir, f"{base}_est.wav"), enh, 16000)
                    sf.write(os.path.join(output_dir, f"{base}_st1.wav"), st1, 16000)
                    sf.write(os.path.join(output_dir, f"{base}_noisy.wav"), mix[cut:, 0], 16000)
                    sf.write(os.path.join(output_dir, f"{base}_ref.wav"), ref[cut:], 16000)

                ttime += batch_time / n_orig
                mtime = ttime / cnt

                info = {
                    'idx': cnt, 'noisy': s['mixture'], 'target': s['target'],
                    'out_path': save_wav_path,
                    'sisdr': {'mix': mix_sisdr, 'est': est_sisdr, 'st1': st1_sisdr},
                    'stoi': {'mix': mix_stoi_v, 'est': est_stoi_v, 'st1': st1_stoi_v},
                    'pesq': {'mix': mix_pesq_v, 'est': est_pesq_v, 'st1': st1_pesq_v},
                    'dnsmos': {'sig': enhanced_mos['sig_mos'], 'bak': enhanced_mos['bak_mos'],
                               'ovrl': enhanced_mos['ovrl_mos'], 'p808': enhanced_mos['p808_mos']},
                    'time_s': batch_time / n_orig,
                }
                f_jsonl.write(json.dumps(to_py(info), ensure_ascii=False) + '\n')
                row = [cnt, os.path.basename(s['mixture']), mix_sisdr, mix_stoi_v, mix_pesq_v,
                       est_sisdr, est_stoi_v, est_pesq_v,
                       st1_sisdr, st1_stoi_v, st1_pesq_v,
                       enhanced_mos['sig_mos'], enhanced_mos['bak_mos'],
                       enhanced_mos['ovrl_mos'], enhanced_mos['p808_mos'], batch_time / n_orig]

                def _fmt(v):
                    if isinstance(v, bool) or not isinstance(v, numbers.Real):
                        return str(v)
                    return str(int(v)) if isinstance(v, numbers.Integral) else f'{float(v):.4f}'
                f_csv.write(','.join(_fmt(v) for v in row) + '\n')

                ttbar.update_progress(int(cnt), 'test',
                                      'mt={:.2f} SISDR={:.2f} STOI={:.3f}'.format(
                                          mtime,
                                          np.nanmean(M['est_sisdr']), np.nanmean(M['est_stoi'])))

            del x_b, truth_b, est_b, stage1_b, mix_specs, truth_specs, mixes, refs
            if arg.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    finally:
        f_jsonl.close()
        f_csv.close()

    ttbar.finish()
    est_mos /= max(cnt, 1)

    lines = [f"测试集: {len(cur_list)} 条 | ckpt: {tag} | name: {arg.name} | batch={arg.batch} | "
             f"device={arg.device} | 推理方式: 离线整段（训练条件: teacher 先验 + batch WPE/IVA）",
             f"平均耗时: {mtime:.3f} s/条", ""]
    hdr = f"{'指标':<18}{'mean':>9}{'median':>9}{'std':>8}{'p10':>9}{'p90':>9}{'n':>6}"
    lines += [hdr, '-' * len(hdr)]
    for key, label in [('mix_sisdr', 'noisy SI-SDR'),
                       ('est_sisdr', 'enhanced SI-SDR (stage2)'),
                       ('st1_sisdr', 'stage1 SI-SDR (teacher prior)'),
                       ('mix_stoi', 'noisy STOI'),
                       ('est_stoi', 'enhanced STOI (stage2)'),
                       ('st1_stoi', 'stage1 STOI (teacher prior)'),
                       ('mix_pesq', 'noisy PESQ'),
                       ('est_pesq', 'enhanced PESQ (stage2)'),
                       ('st1_pesq', 'stage1 PESQ (teacher prior)')]:
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
    with open(os.path.join(test_audio_dir, f"test_results_{tag}.json"), 'w') as f:
        json.dump({'summary': summary, 'per_file': results}, f, indent=4, ensure_ascii=False)

    print(f"\n逐条结果: {output_jsonl}\n           {output_csv}\n汇总: {summary_txt}")
