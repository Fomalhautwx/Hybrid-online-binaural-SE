#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
se_eval_suite.py —— 双通道混合增强系统（WPE → IVA → GTCRN）的补充评测

配合 testDNS_fixed.py（混合系统）和 testDNS_dc.py（纯 NN 基线）的输出使用。测试脚本每条样本写一行
jsonl（PESQ / STOI / SI-SDR / DNSMOS），并在结果目录里存（<stem> = 原始 noisy 文件名去掉 .wav）：
  <stem>_est.wav    最终输出，全长 wav_len = 256·(T−1)，从第 0 个样本开始
  <stem>_noisy.wav  输入参考麦（原始 noisy 的 ch0）
  <stem>_ref.wav    RTS 目标
  <stem>_bss.wav    IVA 两路输出，ch0 = json 里的 bss_tgt，ch1 = bss_int（只有混合系统有）
noisy / ref 有两种存法：testDNS_fixed 存 [:wav_len]，与 est 对齐；testDNS_dc 存 [cut:]（从丢弃点开始）。
后一种情况下把 est 的前 cut 个样本裁掉再对齐。

本脚本照搬测试脚本的约定：
  · 前 20 帧走 baseline GTCRN，IVA 从第 20 帧才开始，所以 bss.wav 比其它文件短 20×256 = 5120 个样本，
    且与它们尾部对齐（存的是 bss_wav[:, -L:]）。这里按尾部对齐把 bss 放回原时间轴。
  · json 里 mix / est 的指标在 [wav_len/40 :] 上算；bss 的指标在 IVA 自己的整段上算（没有丢弃 1/40）。
  · SI-SDR 与 fast_bss_eval.si_sdr 默认一致（不去均值）；DNSMOS 用 speechmos，输入先 normalize（峰值 > 1 才缩放）。
  · 音频用 sf.write 默认的 PCM_16 存盘，超出 ±1 的部分会被削掉，重算值可能因此与 json 有出入（第 0 节统计）。

补算：ESTOI、BSS-eval SDR/SIR/SAR、IVA 两路与输入的 DNSMOS、滑窗 SI-SDR（冷启动 / 局部损坏 / IVA 局部
通道颠倒）、电平（过度抑制 / 功率坍缩 / 停顿段降噪）、按输入难度分桶、最差样本清单。
所有整条指标都在同一段上算：从 max(前 1/40, IVA 起点) 到结尾，所以 IVA 和最终输出可以直接比。

compare 子命令：两套系统逐条配对比较（bootstrap 置信区间、Wilcoxon 检验、胜率）；评测目录里有 asr_eval.py
写的 asr_per_utt.csv 时，一并比较 ASR 错误率。

依赖：numpy scipy soundfile pystoi pesq matplotlib speechmos

用法：
  python se_eval_suite.py run --json test_results_xxx_flow.jsonl --wav-dir /path/test_results/flow --out eval_flow
  python se_eval_suite.py compare eval_flow eval_dc --names hybrid dcgtcrn --out cmp_hybrid_dc

结果目录里只有 est.wav、没有存 noisy/ref.wav 时（如纯 NN 基线），加 --test-json 指向测试集 json，
脚本会按 est 文件名（stem）去 json 里找对应的 noisy 与 RTS 参考路径兜底。
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # 多进程时避免 BLAS 线程过量订阅

import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.linalg import cho_factor, cho_solve, toeplitz

# ---- 固定设置（与测试脚本 / 训练配置一致，一般不用改）----
FS = 16000
FILTER_LEN = 512          # BSS-eval 允许失真滤波器长度（32 ms）
WIN, HOP = 1.0, 0.25      # 滑窗长度 / 步长（秒）
COLD = 2.0                # 冷启动段：IVA 起点之后多少秒
ACTIVE_DB = 30.0          # 窗口有效门限：比最大窗口能量低这么多 dB 视为目标静音
FULL_SCALE = 0.9999       # PCM_16 读回后 |x| ≥ 此值视为削波
SAVE_RATIO = 40.0         # testDNS_dc 存 noisy / ref 时丢掉的开头比例（测试脚本的 --warmup_ratio）；
                          # 只用来识别这种存法，与评测时从哪开始（--warmup-ratio）无关

NAN = float("nan")
SIGS = ("mix", "bss", "est")                # bss = IVA 目标通道（bss.wav 的 ch0）
SIG_CN = {"mix": "输入(参考麦)", "bss": "IVA 目标通道", "est": "最终输出"}
JSON_SIG = {"mix": "mix", "bss": "bss_tgt", "bssi": "bss_int", "est": "est"}


# =============================================================================
# 基础工具
# =============================================================================

def finite(v):
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def fget(d, *keys):
    """嵌套取值，取不到或非有限数返回 NaN。"""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return NAN
        d = d[k]
    return float(d) if finite(d) else NAN


def load_records(path):
    recs = []
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[警告] json 第 {ln} 行解析失败，已跳过：{e}")
    return recs


def read_wav(path):
    x, fs = sf.read(str(path), dtype="float64", always_2d=True)
    return x.T, fs                                            # (C, N)


def si_sdr(est, ref, eps=1e-12):
    """与 fast_bss_eval.si_sdr(ref, est) 的默认设置一致：不去均值。"""
    a = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    t = a * ref
    e = est - t
    return float(10 * np.log10((np.dot(t, t) + eps) / (np.dot(e, e) + eps)))


def frames(x, n, h):
    if len(x) < n:
        return np.empty((0, n))
    idx = np.arange(0, len(x) - n + 1, h)
    return np.lib.stride_tricks.sliding_window_view(x, n)[idx]


def win_si_sdr(est, ref, n, h, eps=1e-12):
    E, R = frames(est, n, h), frames(ref, n, h)
    a = (E * R).sum(1) / ((R * R).sum(1) + eps)
    T = a[:, None] * R
    D = E - T
    return 10 * np.log10(((T * T).sum(1) + eps) / ((D * D).sum(1) + eps))


def alpha_db(x, ref):
    """x 在 ref 上的最小二乘系数（dB）：x ≈ α·ref + 与 ref 不相关的部分。"""
    a = np.dot(x, ref) / (np.dot(ref, ref) + 1e-20)
    return float(20 * np.log10(abs(a) + 1e-12))


# =============================================================================
# BSS-eval（时不变滤波器版，与 bss_eval_sources / fast_bss_eval 定义一致，已逐位对照验证）
# 参考源 = [目标, 残差 r = x_ref − 目标]；残差是否乘 α 不影响结果（张成的子空间相同）
# =============================================================================

class BSSEval:
    def __init__(self, refs, L):
        refs = np.asarray(refs, dtype=np.float64)
        self.R, self.N = refs.shape
        self.L = L
        self.nfft = 1 << (self.N + L - 1 - 1).bit_length()
        self.Fr = np.fft.rfft(refs, n=self.nfft, axis=1)
        R, Lf = self.R, L
        G = np.empty((R * Lf, R * Lf))
        for i in range(R):
            for j in range(i, R):
                c = np.fft.irfft(self.Fr[i] * np.conj(self.Fr[j]), n=self.nfft)
                col = np.concatenate([[c[0]], c[-1:-Lf:-1]])    # ρ_ij[0], ρ_ij[-1], …
                blk = toeplitz(col, c[:Lf])                    # blk[k,l] = ρ_ij[l-k]
                G[i * Lf:(i + 1) * Lf, j * Lf:(j + 1) * Lf] = blk
                G[j * Lf:(j + 1) * Lf, i * Lf:(i + 1) * Lf] = blk.T
        load = 1e-10 * np.trace(G) / G.shape[0] + 1e-30
        self.cf_all = cho_factor(G + load * np.eye(G.shape[0]), lower=True)
        self.cf_tgt = cho_factor(G[:Lf, :Lf] + load * np.eye(Lf), lower=True)

    def _proj(self, Fe, cf, nsrc):
        D = np.concatenate([np.fft.irfft(Fe * np.conj(self.Fr[i]), n=self.nfft)[:self.L]
                            for i in range(nsrc)])
        C = cho_solve(cf, D).reshape(nsrc, self.L)
        Fp = (np.fft.rfft(C, n=self.nfft, axis=1) * self.Fr[:nsrc]).sum(0)
        return np.fft.irfft(Fp, n=self.nfft)[:self.N + self.L - 1]

    def __call__(self, est, clamp=100.0):
        est = np.asarray(est, dtype=np.float64)[:self.N]
        Fe = np.fft.rfft(est, n=self.nfft)
        s_t = self._proj(Fe, self.cf_tgt, 1)
        p_all = self._proj(Fe, self.cf_all, self.R)
        e = np.zeros(self.N + self.L - 1)
        e[:self.N] = est
        e_int, e_art = p_all - s_t, e - p_all
        eps = 1e-20
        db = lambda a, b: float(np.clip(10 * np.log10((a + eps) / (b + eps)), -clamp, clamp))
        es = np.dot(s_t, s_t)
        return dict(sdr=db(es, np.sum((e_int + e_art) ** 2)),
                    sir=db(es, np.dot(e_int, e_int)),
                    sar=db(np.sum((s_t + e_int) ** 2), np.dot(e_art, e_art)))


# =============================================================================
# DNSMOS P.835：speechmos，输入处理与测试脚本的 normalize() 一致
# =============================================================================

class DNSMOS:
    def __init__(self):
        from speechmos import dnsmos
        self._run = dnsmos.run

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if len(x) == 0 or not np.all(np.isfinite(x)) or not np.any(x):
            return dict(sig=NAN, bak=NAN, ovrl=NAN, p808=NAN)
        x = x.astype(np.float32)
        m = np.max(np.abs(x))
        if m > 1.0:
            x = x / m
        r = self._run(x, FS)
        return dict(sig=float(r["sig_mos"]), bak=float(r["bak_mos"]),
                    ovrl=float(r["ovrl_mos"]), p808=float(r["p808_mos"]))


# =============================================================================
# 逐条评测
# =============================================================================

_CAT_RE = re.compile(r"sample_\d+_(?:(\w+?)_)?clean_fileid")


def category(stem):
    """语料类别：sample_317_singing_clean_fileid_630_test → singing；没有类别标记 → read。"""
    m = _CAT_RE.search(str(stem))
    if m is None:
        return "other"
    return m.group(1) or "read"


def load_ref_map(path):
    """从测试集 json 建 noisy stem → (noisy, ref) 映射，兜底结果目录里没存 noisy/ref.wav 的情况。

    ref 取 RTS（= jsonl 里的 target）；noisy 取原始输入。key 用 noisy 文件名的 stem，
    与 <stem>_est.wav 的 stem 一致。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    m = {}
    for e in data:
        stem = Path(str(e.get("noisy", ""))).stem
        if stem and stem not in m:
            m[stem] = (e.get("noisy"), e.get("RTS"))
    return m


def locate(rec, wav_dir, ref_map=None):
    stem = Path(str(rec.get("noisy", ""))).stem
    files = {}
    for k in ("est", "bss", "noisy", "ref"):
        p = Path(wav_dir) / f"{stem}_{k}.wav"
        files[k] = p if p.exists() else None
    # 兜底：结果目录里没存 noisy/ref.wav（如纯 NN 基线只存了 est），按 stem 去测试集 json 找
    if ref_map and stem in ref_map:
        n, r = ref_map[stem]
        if files["noisy"] is None:
            files["noisy"] = n
        if files["ref"] is None:
            files["ref"] = r
    return stem, files


def load_utt(files, ratio):
    """读结果目录里的一条样本并对齐到同一时间轴（asr_eval.py 也用这个函数）。
    出错时抛 ValueError（中文原因）。返回 dict：
      est / ref / mix (N,)、bss (2, N) 或 None、nb（bss 通道数）
      head  当前时间轴的 0 在原始音频里的位置（dc 存法下 = cut，否则 0）
      cut   测试脚本的丢弃点 int(wav_len / ratio)（原始时间轴）；c0 = cut − head（当前时间轴）
      off   bss 在当前时间轴上的起点（IVA 起点）；s0 = max(c0, off) 主评测段起点
    """
    miss = [k for k in ("est", "ref", "noisy") if files[k] is None]
    if miss:
        raise ValueError("缺少文件: " + ",".join(miss))
    sig = {}
    for k in ("est", "ref", "noisy"):                         # noisy 只存了参考麦 ch0
        x, fs = read_wav(files[k])
        if fs != FS:
            raise ValueError(f"{k} 采样率 {fs} ≠ {FS}")
        sig[k] = x[0]
    est, ref, mix = sig["est"], sig["ref"], sig["noisy"]
    W = len(est)                                              # wav_len：两个测试脚本都存全长 est
    cut = int(W / ratio) if ratio > 0 else 0                  # 测试脚本的丢弃点：int(wav_len / 40)

    # 对齐以 noisy / ref 为准：testDNS_fixed 存的是 [:wav_len]，与 est 一样长、都从 0 开始；
    # testDNS_dc 存的是 [W/40:]，从测试脚本的丢弃点开始 → 把 est 的前 W/40 个样本裁掉。
    # 这里固定按测试脚本的比例判断，不能用评测的 --warmup-ratio（改了它会认错存法、est 错位）。
    sv = int(W / SAVE_RATIO)
    head = sv if sv > 0 and len(ref) < W - sv // 2 else 0
    est = est[head:]
    N = min(len(est), len(ref), len(mix))
    if N < 2 * FS:
        raise ValueError(f"太短（{N / FS:.2f} s）")
    len_mismatch = int(max(len(est), len(ref), len(mix)) - N)
    est, ref, mix = est[:N], ref[:N], mix[:N]

    # bss.wav：缺前 20 帧，与 est 尾部对齐 → 放回时间轴，前面补零
    bss, nb, off = None, 0, 0
    if files.get("bss") is not None:
        b, fs = read_wav(files["bss"])
        if fs == FS and b.shape[1] > 0:
            Lw = W - head                                     # 裁掉开头后 est 的全长
            Lb = min(b.shape[1], Lw)
            off = Lw - Lb
            nb = min(2, b.shape[0])
            full_b = np.zeros((2, Lw))
            full_b[:nb, off:] = b[:nb, b.shape[1] - Lb:]
            bss = full_b[:, :N]
    c0 = max(cut - head, 0)
    return dict(est=est, ref=ref, mix=mix, bss=bss, nb=nb, head=head, cut=cut, c0=c0,
                off=off, s0=max(c0, off), N=N, len_mismatch=len_mismatch)


def level_stats(ref, y, mix, n_frame=512, n_hop=256, act_db=25.0, pause_db=45.0):
    """语音段电平 y/ref、后 1/3 相对前 1/3 的电平漂移、停顿段降噪量 mix/y（dB）。"""
    Er = (frames(ref, n_frame, n_hop) ** 2).sum(1)
    Ey = (frames(y, n_frame, n_hop) ** 2).sum(1)
    Em = (frames(mix, n_frame, n_hop) ** 2).sum(1)
    if len(Er) == 0 or Er.max() <= 0:
        return NAN, NAN, NAN
    act = Er >= Er.max() * 10 ** (-act_db / 10)
    pause = Er <= Er.max() * 10 ** (-pause_db / 10)
    tiny = 1e-20
    lvl = 10 * np.log10((Ey[act].sum() + tiny) / (Er[act].sum() + tiny))
    third = len(Er) // 3
    a1, a3 = act.copy(), act.copy()
    a1[third:] = False
    a3[:len(Er) - third] = False
    if a1.sum() >= 5 and a3.sum() >= 5:
        l1 = 10 * np.log10((Ey[a1].sum() + tiny) / (Er[a1].sum() + tiny))
        l3 = 10 * np.log10((Ey[a3].sum() + tiny) / (Er[a3].sum() + tiny))
        drift = l3 - l1
    else:
        drift = NAN
    nrp = 10 * np.log10((Em[pause].sum() + tiny) / (Ey[pause].sum() + tiny)) if pause.sum() >= 5 else NAN
    return float(lvl), float(drift), float(nrp)


def process_utt(rec, cfg, mos=None):
    stem, files = locate(rec, cfg["wav_dir"], cfg.get("ref_map"))
    row = {"stem": stem, "idx": rec.get("idx", ""), "error": "", "error_src": ""}
    if rec.get("error"):                                      # 测试脚本里就失败了（没有存音频）
        row["error"], row["error_src"] = str(rec["error"]), "test"
        return row, None
    for k in ("mix", "bss", "bssi", "est"):
        row[f"json_sisdr_{k}"] = fget(rec, "sisdr", JSON_SIG[k])
    for met in ("stoi", "pesq"):
        for k in SIGS:
            row[f"json_{met}_{k}"] = fget(rec, met, JSON_SIG[k])
    for q in ("sig", "bak", "ovrl", "p808"):
        row[f"json_dnsmos_{q}"] = fget(rec, "dnsmos", q)
    row["json_swap_gain"] = fget(rec, "swap_gain")
    row["json_time_s"] = fget(rec, "time_s")

    try:
        u = load_utt(files, cfg["warmup_ratio"])
    except ValueError as e:
        row["error"] = str(e)
        return row, None
    est, ref, mix, bss, nb = u["est"], u["ref"], u["mix"], u["bss"], u["nb"]
    head, cut, c0, off, s0, N = u["head"], u["cut"], u["c0"], u["off"], u["s0"], u["N"]
    row["head"], row["len_mismatch"] = head, u["len_mismatch"]
    row["dur_s"] = (head + N) / FS
    row["bss_off"] = head + off if bss is not None else NAN
    row["cut_s"], row["s0_s"] = cut / FS, (head + s0) / FS

    # —— 削波（PCM_16 存盘时超出 ±1 被截断）——
    for k, x in (("ref", ref), ("mix", mix), ("est", est)):
        row[f"clip_{k}"] = int(np.sum(np.abs(x) >= FULL_SCALE))          # 满幅样本数
    row["clip_bss"] = int(np.sum(np.abs(bss[:nb, off:]) >= FULL_SCALE)) if bss is not None else NAN

    # —— 按测试脚本的口径重算 SI-SDR，只用于第 0 节与 json 核对 ——
    row["chk_sisdr_mix"] = si_sdr(mix[c0:], ref[c0:])
    row["chk_sisdr_est"] = si_sdr(est[c0:], ref[c0:])
    row["chk_sisdr_bss"] = si_sdr(bss[0, off:], ref[off:]) if bss is not None else NAN
    row["chk_sisdr_bssi"] = si_sdr(bss[1, off:], ref[off:]) if nb == 2 else NAN

    # —— 主评测段 [s0, N) ——
    R = ref[s0:]
    S = {"mix": mix[s0:], "est": est[s0:],
         "bss": bss[0, s0:] if bss is not None else None,
         "bssi": bss[1, s0:] if nb == 2 else None}
    for k, x in S.items():
        row[f"sisdr_{k}"] = si_sdr(x, R) if x is not None else NAN
    from pystoi import stoi
    for k in SIGS:
        row[f"estoi_{k}"] = float(stoi(R, S[k], FS, extended=True)) if S[k] is not None else NAN
    if cfg.get("recompute_stoi_pesq"):                         # 从音频重算 STOI / PESQ（主评测段 [s0:]）
        from pesq import pesq as _pesq
        for k in SIGS:
            if S[k] is None:
                row[f"stoi_{k}"] = row[f"pesq_{k}"] = NAN
                continue
            try:
                row[f"stoi_{k}"] = float(stoi(R, S[k], FS, extended=False))
            except Exception:
                row[f"stoi_{k}"] = NAN
            try:
                row[f"pesq_{k}"] = float(_pesq(FS, R, S[k]))
            except Exception:
                row[f"pesq_{k}"] = NAN
    row["alpha_db"] = alpha_db(S["mix"], R)
    be = BSSEval(np.stack([R, S["mix"] - R]), FILTER_LEN)
    for k in SIGS:
        r = be(S[k]) if S[k] is not None else dict(sdr=NAN, sir=NAN, sar=NAN)
        for q in ("sdr", "sir", "sar"):
            row[f"{q}_{k}"] = r[q]

    # —— 滑窗 SI-SDR（时间轴从原始音频开头算；bss 在 IVA 起点之前没有数据，对应窗口记 NaN）——
    n, h = int(WIN * FS), int(HOP * FS)
    Rw = frames(ref, n, h)
    sl = None
    if len(Rw):
        e_r = (Rw ** 2).sum(1)
        active = e_r >= e_r.max() * 10 ** (-ACTIVE_DB / 10)
        starts = np.arange(len(Rw)) * h
        centers = (head + starts + n / 2) / FS
        sl = {"t": centers.astype(np.float32), "active": active,
              "mix": win_si_sdr(mix, ref, n, h).astype(np.float32),
              "est": win_si_sdr(est, ref, n, h).astype(np.float32)}
        if bss is not None:
            cov = starts >= off
            sl["bss"] = np.where(cov, win_si_sdr(bss[0], ref, n, h), np.nan).astype(np.float32)
            if nb == 2:
                sl["bssi"] = np.where(cov, win_si_sdr(bss[1], ref, n, h), np.nan).astype(np.float32)
        ev = active & (starts >= s0)                          # 损坏率 / 最差窗口：只看主评测段
        run = active & (starts >= off)                        # 冷启动：从系统开始工作计时
        t_start = (head + off) / FS if bss is not None else 0.0
        cold = centers < t_start + COLD
        row["win_n"] = int(ev.sum())
        for k in ("bss", "est"):
            if k not in sl:
                for q in ("dmg", "worst_d", "cold_d", "steady_d"):
                    row[f"{q}_{k}"] = NAN
                continue
            dd = (sl[k] - sl["mix"]).astype(float)
            fin = np.isfinite(dd)
            d = dd[ev & fin]
            row[f"dmg_{k}"] = float(np.mean(d < 0)) if len(d) else NAN
            row[f"worst_d_{k}"] = float(d.min()) if len(d) else NAN
            c, s = run & cold & fin, run & ~cold & fin
            row[f"cold_d_{k}"] = float(dd[c].mean()) if c.any() else NAN
            row[f"steady_d_{k}"] = float(dd[s].mean()) if s.any() else NAN
        if "bssi" in sl:
            m = ev & np.isfinite(sl["bss"]) & np.isfinite(sl["bssi"])
            row["flip_bss"] = float(np.mean((sl["bssi"] > sl["bss"])[m])) if m.any() else NAN
        else:
            row["flip_bss"] = NAN
    else:
        row["win_n"] = 0

    # —— 电平 ——
    for k in ("bss", "est"):
        if S[k] is None:
            row[f"lvl_{k}"] = row[f"lvlm_{k}"] = row[f"drift_{k}"] = row[f"nrp_{k}"] = NAN
            continue
        row[f"lvl_{k}"], row[f"drift_{k}"], row[f"nrp_{k}"] = level_stats(R, S[k], S["mix"])
        # 相对“混合里的目标分量” α·ref：ref.wav 与 noisy.wav 电平不一致（α ≠ 1）时看这个
        row[f"lvlm_{k}"] = row[f"lvl_{k}"] - row["alpha_db"] if finite(row[f"lvl_{k}"]) else NAN

    # —— DNSMOS（主评测段；s0 == c0 时 est 的口径与 json 完全相同）——
    row["dnsmos_span_eq"] = int(s0 == c0)
    for k in SIGS:
        for q in ("sig", "bak", "ovrl", "p808"):
            row[f"dnsmos_{k}_{q}"] = NAN
    if mos is not None:
        for k in SIGS:
            if S[k] is not None:
                for q, v in mos(S[k]).items():
                    row[f"dnsmos_{k}_{q}"] = v

    row["time_ratio"] = row["json_time_s"] / row["dur_s"] if finite(row["json_time_s"]) else NAN
    return row, sl


_G = {}


def _init_worker(cfg):
    _G["cfg"] = cfg
    _G["mos"] = DNSMOS() if cfg["dnsmos"] else None


def _work(rec):
    try:
        return process_utt(rec, _G["cfg"], _G["mos"])
    except Exception as e:                                    # 单条失败不影响整体
        stem = Path(str(rec.get("noisy", ""))).stem
        return {"stem": stem, "idx": rec.get("idx", ""), "error": f"{type(e).__name__}: {e}",
                "error_src": "eval"}, None


# =============================================================================
# 统计
# =============================================================================

def col(rows, key):
    return np.array([float(r.get(key, NAN)) if finite(r.get(key, NAN)) else NAN for r in rows])


def boot_ci(v, n_boot=2000, seed=0):
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return NAN, NAN
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def describe(v):
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0, mean=NAN, lo=NAN, hi=NAN, med=NAN, p5=NAN)
    lo, hi = boot_ci(v)
    return dict(n=len(v), mean=float(v.mean()), lo=lo, hi=hi,
                med=float(np.median(v)), p5=float(np.percentile(v, 5)))


def fmt(x, nd=2):
    return "—" if not finite(x) else f"{x:.{nd}f}"


def fmt_ci(s, nd=2):
    if s["n"] == 0:
        return "—"
    return f"{s['mean']:.{nd}f} [{fmt(s['lo'], nd)}, {fmt(s['hi'], nd)}]"


def pct(x):
    return "—" if not finite(x) else f"{100 * x:.1f}%"


# =============================================================================
# 作图（英文标签，避免服务器缺中文字体）
# =============================================================================

C_EST, C_BSS, C_MIX = "#2a78d6", "#eb6834", "#8f8e88"
C_SURF, C_INK, C_INK2, C_GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": C_SURF, "axes.facecolor": C_SURF, "savefig.facecolor": C_SURF,
        "axes.edgecolor": C_GRID, "axes.labelcolor": C_INK2, "axes.titlecolor": C_INK,
        "xtick.color": C_INK2, "ytick.color": C_INK2, "text.color": C_INK,
        "axes.grid": True, "grid.color": C_GRID, "grid.linestyle": "-", "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
        "lines.linewidth": 1.0,
    })
    return plt


def plot_all(rows, slides, out):
    plt = _style()
    has_bss = np.isfinite(col(rows, "sisdr_bss")).any()
    series = [("est", "Final output", C_EST)] + ([("bss", "IVA target ch.", C_BSS)] if has_bss else [])
    off = col(rows, "bss_off")
    t_iva = float(np.nanmedian(off)) / FS if np.isfinite(off).any() else 0.0
    figs = []

    # 1) 冷启动与收敛：窗口 ΔSI-SDR 随时间
    if slides:
        grid = lambda t: np.maximum(np.round((np.asarray(t) - WIN / 2) / HOP).astype(int), 0)
        T = max(int(grid(s["t"]).max()) + 1 for s in slides.values())   # 按原始时间对齐到同一网格
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        for k, lab, c in series:
            M = np.full((len(slides), T), np.nan)
            for i, s in enumerate(slides.values()):
                if k in s:
                    d = (s[k] - s["mix"]).astype(float)
                    d[~s["active"]] = np.nan
                    M[i, grid(s["t"])] = d
            cnt = np.isfinite(M).sum(0)
            keep = cnt >= max(3, 0.3 * len(slides))
            if not keep.any():
                continue
            with np.errstate(all="ignore"):
                med = np.nanmedian(M, 0)
                q1, q3 = np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0)
            tt = np.arange(T) * HOP + WIN / 2
            ax.fill_between(tt[keep], q1[keep], q3[keep], color=c, alpha=0.15, linewidth=0)
            ax.plot(tt[keep], med[keep], color=c, label=f"{lab} (median, IQR band)")
        ax.axhline(0, color=C_INK2, linewidth=0.8)
        y0, y1 = ax.get_ylim()
        marks = (((t_iva, "IVA starts"), (t_iva + COLD, f"IVA +{COLD:.0f} s")) if has_bss
                 else ((COLD, f"cold-start {COLD:.0f} s"),))
        for j, (x, txt) in enumerate(marks):
            if x > 0:
                ax.axvline(x, color=C_INK2, linewidth=0.8, linestyle="--")
                ax.text(x, y1 - j * 0.07 * (y1 - y0), f" {txt}", va="top", ha="left", color=C_INK2, fontsize=8)
        ax.set_xlabel("time since start of utterance (s)")
        ax.set_ylabel("windowed SI-SDR improvement over input (dB)")
        ax.set_title(f"Convergence: {WIN:.1f}-s windows, {len(slides)} utterances")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "fig_convergence.png", dpi=150)
        plt.close(fig)
        figs.append("fig_convergence.png")

    # 2) 改善量分布：整条与窗口两个层面
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, lab, c in series:
        d = col(rows, f"sisdr_{k}") - col(rows, "sisdr_mix")
        d = np.sort(d[np.isfinite(d)])
        if len(d):
            axs[0].plot(d, np.arange(1, len(d) + 1) / len(d), color=c, label=lab, drawstyle="steps-post")
        w = [(s[k] - s["mix"])[s["active"]] for s in slides.values() if k in s]
        if w:
            w = np.concatenate(w).astype(float)
            w = np.sort(w[np.isfinite(w)])
            if len(w):
                axs[1].plot(w, np.arange(1, len(w) + 1) / len(w), color=c, label=lab, drawstyle="steps-post")
    for ax, ttl in zip(axs, ("Per utterance", f"Per {WIN:.1f}-s window")):
        ax.axvline(0, color=C_INK2, linewidth=0.8)
        ax.set_xlabel("SI-SDR improvement over input (dB)")
        ax.set_ylabel("cumulative fraction")
        ax.set_title(ttl)
        ax.legend(loc="lower right")
    fig.suptitle("Left of 0 = worse than the unprocessed input", color=C_INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_delta_cdf.png", dpi=150)
    plt.close(fig)
    figs.append("fig_delta_cdf.png")

    # 3) SIR vs SAR：残差是噪声还是伪影
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for k, lab, c in series:
        x, y = col(rows, f"sir_{k}"), col(rows, f"sar_{k}")
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=18, color=c, alpha=0.8, label=lab,
                   edgecolors=C_SURF, linewidths=1.0)
    ax.set_xlabel("SIR (dB)  — residual noise + late reverb")
    ax.set_ylabel("SAR (dB)  — artifacts")
    ax.set_title("Where the remaining error comes from (per utterance)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "fig_sir_sar.png", dpi=150)
    plt.close(fig)
    figs.append("fig_sir_sar.png")

    # 4) 按输入难度分桶
    mixv = col(rows, "sisdr_mix")
    ok = np.isfinite(mixv)
    if ok.sum() >= 8:
        edges = np.percentile(mixv[ok], [25, 50, 75])
        b = np.digitize(mixv, edges)
        panels = [("ΔSI-SDR (dB)", lambda k: col(rows, f"sisdr_{k}") - mixv),
                  ("SAR (dB)", lambda k: col(rows, f"sar_{k}")),
                  ("local damage rate (%)", lambda k: 100 * col(rows, f"dmg_{k}"))]
        fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
        width = 0.36 if len(series) == 2 else 0.6
        for ax, (ttl, fn) in zip(axs, panels):
            for j, (k, lab, c) in enumerate(series):
                v = fn(k)
                m = [np.nanmean(v[(b == q) & ok]) if np.isfinite(v[(b == q) & ok]).any() else np.nan
                     for q in range(4)]
                xpos = np.arange(4) + (j - (len(series) - 1) / 2) * (width + 0.04)
                ax.bar(xpos, m, width=width, color=c, label=lab, linewidth=0)
            ax.axhline(0, color=C_INK2, linewidth=0.8)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["Q1\nhardest", "Q2", "Q3", "Q4\neasiest"])
            ax.set_title(ttl)
        axs[0].legend(loc="best")
        fig.suptitle("By input difficulty (quartiles of input SI-SDR)", color=C_INK2, fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "fig_buckets.png", dpi=150)
        plt.close(fig)
        figs.append("fig_buckets.png")

    # 5) 最差的 6 条：滑窗曲线
    worst = sorted([r for r in rows if finite(r.get("worst_d_est"))],
                   key=lambda r: r["worst_d_est"])[:6]
    if worst:
        fig, axs = plt.subplots(2, 3, figsize=(11, 5.4), sharey=False)
        for ax, r in zip(axs.flat, worst):
            s = slides.get(r["stem"])
            if s is None:
                continue
            a = s["active"]
            for k, lab, c in [("mix", "Input", C_MIX)] + series:
                if k in s:
                    y = s[k].astype(float).copy()
                    y[~a] = np.nan
                    ax.plot(s["t"], y, color=c, label=lab)
            ax.set_title(r["stem"][-34:], fontsize=8)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("windowed SI-SDR (dB)")
        for ax in list(axs.flat)[len(worst):]:
            ax.set_visible(False)
        axs.flat[0].legend(loc="best", fontsize=7)
        fig.suptitle("Utterances with the worst local damage (gaps = target silent)",
                     color=C_INK2, fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "fig_worst_utts.png", dpi=150)
        plt.close(fig)
        figs.append("fig_worst_utts.png")
    return figs


# =============================================================================
# 报告
# =============================================================================

def write_csv(rows, path):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def counts_txt(v, fmt_v=lambda x: f"{int(x)}", top=3):
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "—"
    c = Counter(v.tolist()).most_common(top)
    return "，".join(f"{fmt_v(a)}：{n} 条" for a, n in c)


def build_report(rows, bad, n_rec, cfg, figs):
    L = []
    P = L.append
    has_bss = np.isfinite(col(rows, "sisdr_bss")).any()
    sigs = [s for s in SIGS if s != "bss" or has_bss]
    recompute = bool(cfg.get("recompute_stoi_pesq"))

    def sp(s, re_key, json_key):
        v = col(rows, re_key.format(s))
        return v if np.isfinite(v).any() else col(rows, json_key.format(s))

    stoi_fn = lambda s: sp(s, "stoi_{}", "json_stoi_{}")       # 重算过就比重算值，否则退回 json
    pesq_fn = lambda s: sp(s, "pesq_{}", "json_pesq_{}")
    stoi_name = "STOI" if recompute else "STOI（json）"
    pesq_name = "PESQ" if recompute else "PESQ（json）"
    dur = col(rows, "dur_s")
    s0, cut = col(rows, "s0_s"), col(rows, "cut_s")
    ratio = cfg["warmup_ratio"]
    P("# 评测报告\n")
    P(f"- 记录 {n_rec} 条，成功评测 {len(rows)} 条，失败 {len(bad)} 条")
    P(f"- 平均时长 {np.nanmean(dur):.1f} s，总时长 {np.nansum(dur) / 3600:.2f} h")
    n_late = int(np.sum(s0 > cut + 1e-9))
    P(f"- 评测段：从 max(前 1/{ratio:g}, IVA 起点) 到结尾，平均丢掉开头 {np.nanmean(s0):.2f} s"
      + (f"；其中 {n_late} 条较短，1/{ratio:g} 比 IVA 起点还早，起点推到了 IVA 起点" if n_late else ""))
    P(f"- BSS-eval 滤波器 {FILTER_LEN} 样本；滑窗 {WIN} s / 步长 {HOP} s；SI-SDR 不去均值（同 fast_bss_eval）\n")

    test_fail = [r for r in bad if r.get("error_src") == "test"]
    other_fail = [r for r in bad if r.get("error_src") != "test"]
    if test_fail:
        P(f"**测试脚本里就失败的 {len(test_fail)} 条**（jsonl 里带 error 字段，没有存音频）：\n")
        for msg, n in Counter(r["error"] for r in test_fail).most_common(3):
            P(f"- {n} 条：`{msg[:160]}`")
        if any("between -1 and 1" in r["error"] for r in test_fail):
            P("- 这个报错来自 speechmos 的输入检查。测试脚本的 normalize() 已经把峰值压到 1 以内，"
              "还报错基本只能是最终输出里有 NaN / Inf，也就是这几条推理发散了，值得单独复现")
        P("- 样本：" + "，".join(f"`{r['stem']}`" for r in test_fail[:10]) + "\n")
    if other_fail:
        P(f"**本脚本评测失败的 {len(other_fail)} 条**（前 10 条）：\n")
        for r in other_fail[:10]:
            P(f"- `{r.get('stem')}`：{r.get('error')}")
        P("")

    # ---------------- 0. 数据核对 ----------------
    P("## 0. 数据核对（先看这一节）\n")
    P(f"按测试脚本的口径重算 SI-SDR，与 json 比较：输入 / 最终输出丢弃前 1/{ratio:g}，IVA 两路用 IVA 自己的整段"
      "（bss.wav 按尾部对齐）。差值 ≈ 0 说明文件对应、对齐、丢弃长度都对。测试脚本默认用 PCM_16 存盘，超出 ±1 被削掉的样本会让重算值偏离 json，所以单独列出“其中有削波”的条数；"
      "DNSMOS 对 16-bit 量化和削波都敏感，OVRL 差零点零几属正常。\n")
    P("| 信号 | 条数 | |差| 中位数 | |差| 最大 | > 0.1 dB 的条数 | 其中有削波 |")
    P("|---|---|---|---|---|---|")
    # 1–3 个满幅样本是峰值归一化留下的，不算削波；成片满幅才会让指标偏离 json
    clip_any = lambda keys: np.any([col(rows, f"clip_{k}") > 3 for k in keys], axis=0)
    for k, cn, ck in [("mix", "输入", ("mix", "ref")), ("bss", "IVA 目标通道", ("bss", "ref")),
                      ("bssi", "IVA 干扰通道", ("bss", "ref")), ("est", "最终输出", ("est", "ref"))]:
        j, r = col(rows, f"json_sisdr_{k}"), col(rows, f"chk_sisdr_{k}")
        ok = np.isfinite(j) & np.isfinite(r)
        if ok.any():
            d = np.abs(j - r)
            big = ok & (d > 0.1)
            P(f"| {cn} SI-SDR | {ok.sum()} | {np.median(d[ok]):.3f} | {d[ok].max():.3f} | {big.sum()} | "
              f"{int(np.sum(big & clip_any(ck)))} |")
    j, r = col(rows, "json_dnsmos_ovrl"), col(rows, "dnsmos_est_ovrl")
    ok = np.isfinite(j) & np.isfinite(r) & (col(rows, "dnsmos_span_eq") == 1)
    if ok.any():
        d = np.abs(j - r)
        big = ok & (d > 0.02)
        P(f"| 最终输出 DNSMOS OVRL | {ok.sum()} | {np.median(d[ok]):.3f} | {d[ok].max():.3f} | "
          f"{big.sum()}（> 0.02） | {int(np.sum(big & clip_any(('est',))))} |")
    P("")
    for k, cn in [("mix", "输入"), ("bss", "IVA 目标通道"), ("est", "最终输出")]:
        cand = []
        for rr in rows:
            jv, rv = rr.get(f"json_sisdr_{k}"), rr.get(f"chk_sisdr_{k}")
            if finite(jv) and finite(rv) and abs(jv - rv) > 0.1:
                cand.append((abs(jv - rv), rr["stem"]))
        if cand:
            cand.sort(reverse=True)
            P(f"- {cn} 差值最大的几条：" + "，".join(f"`{s}` {d:.2f} dB" for d, s in cand[:3]))
    hd = col(rows, "head")
    P(f"- 文件布局：noisy / ref 与 est 等长、都从 0 开始 {int(np.sum(hd == 0))} 条（testDNS_fixed 的存法）；"
      f"noisy / ref 从丢弃点开始存 {int(np.sum(hd > 0))} 条（testDNS_dc 的存法，est 已裁掉前 1/{ratio:g}）")
    lm = col(rows, "len_mismatch")
    P(f"- 对齐后 est / noisy / ref 末尾不等长的条数：{int(np.sum(lm > 0))}，最多差 {np.nanmax(lm):.0f} 个样本（按最短的截齐）")
    if has_bss:
        P(f"- bss.wav 比其它文件短（= IVA 起点）：{counts_txt(col(rows, 'bss_off'), lambda x: f'{int(x)} 样本（{x / 256:g} 帧）')}"
          "。前 20 帧走 baseline GTCRN，应全部是 5120 样本")
    clip_lines = []
    for k, name in (("mix", "noisy"), ("ref", "ref"), ("est", "est"), ("bss", "bss")):
        c = col(rows, f"clip_{k}")
        if np.isfinite(c).any():
            clip_lines.append(f"{name} {int(np.sum(c > 0))} 条" + (f"（单条最多 {int(np.nanmax(c))} 个样本）" if np.nanmax(c) > 0 else ""))
    P("- 削波（读回后 |x| ≥ 0.9999 的样本）：" + "，".join(clip_lines)
      + "。测试脚本 sf.write 时加 subtype='FLOAT' 就不会削")
    for k, name in (("ref", "ref（RTS 目标）"), ("mix", "noisy 参考麦")):
        c = col(rows, f"clip_{k}")
        hit = np.isfinite(c) & (c >= 1) & (c <= 3)
        if np.isfinite(c).any() and hit.mean() > 0.3:
            P(f"  - {name}：{hit.mean() * 100:.0f}% 的条目恰好有 1–3 个满幅样本、没有成片削波，"
              "这是逐条峰值归一化到 1.0 的特征"
              + ("（双通道一起归一化时，峰值约一半落在 ch0）" if k == "mix" else ""))
    a = col(rows, "alpha_db")
    P(f"- ref 与 noisy 的电平关系：noisy 在 ref 上的最小二乘系数 20·log10|α| 中位数 {np.nanmedian(a):+.2f} dB，"
      f"5%–95% 分位 {np.nanpercentile(a, 5):+.2f} ~ {np.nanpercentile(a, 95):+.2f} dB。测试脚本没有对 ref / noisy 做缩放，"
      "所以这是数据集本身的电平关系；α 明显偏离 0 dB 说明 RTS 目标与混合里的目标分量不在同一电平。"
      "SI-SDR / ESTOI / SDR / SIR / SAR 不受影响，只影响第 4 节的电平，那里同时给出相对 α·ref 的电平。")
    P("")

    # ---------------- 1. 总体指标 ----------------
    P("## 1. 总体指标\n")
    sp_note = ("STOI / PESQ 由音频重算（主评测段）。" if recompute
               else "STOI / PESQ 取自 json：输入与最终输出丢弃了前 1/40，IVA 列是 IVA 整段（没丢 1/40）。")
    P(f"格式：均值 [95% bootstrap 置信区间]；第二行是中位数 / 5% 分位。{sp_note}\n")
    dsigs = [s for s in sigs if s != "mix"]
    P("| 指标 | " + " | ".join(SIG_CN[s] for s in sigs) + " | "
      + " | ".join({"bss": "IVA − 输入", "est": "最终 − 输入"}[s] for s in dsigs) + " |")
    P("|" + "---|" * (len(sigs) + len(dsigs) + 1))

    def line(name, key_fn, nd=2, delta=True):
        cells, cells2 = [], []
        for s in sigs:
            d = describe(key_fn(s))
            cells.append(fmt_ci(d, nd))
            cells2.append(f"{fmt(d['med'], nd)} / {fmt(d['p5'], nd)}")
        dcells = [fmt_ci(describe(key_fn(s) - key_fn("mix")), nd) if delta else "—" for s in dsigs]
        P(f"| **{name}** | " + " | ".join(cells) + " | " + " | ".join(dcells) + " |")
        P("| | " + " | ".join(cells2) + " |" + " |" * len(dsigs))

    line("SI-SDR (dB)", lambda s: col(rows, f"sisdr_{s}"))
    line("ESTOI", lambda s: col(rows, f"estoi_{s}"), 3)
    line(stoi_name, stoi_fn, 3)
    line(pesq_name, pesq_fn)
    line("BSS SDR (dB)", lambda s: col(rows, f"sdr_{s}"))
    line("SIR (dB)", lambda s: col(rows, f"sir_{s}"))
    sar = {s: col(rows, f"sar_{s}") for s in sigs}
    sar["mix"] = np.full(len(rows), NAN)                       # 混合本身无伪影，SAR 无意义
    line("SAR (dB)", lambda s: sar[s], delta=False)
    for q, qn in (("sig", "SIG"), ("bak", "BAK"), ("ovrl", "OVRL"), ("p808", "P.808")):
        v = {s: col(rows, f"dnsmos_{s}_{q}") for s in sigs}
        if not np.isfinite(v["est"]).any():
            v["est"] = col(rows, f"json_dnsmos_{q}")           # 没重算时退回 json
        if any(np.isfinite(x).any() for x in v.values()):
            line(f"DNSMOS {qn}", lambda s, v=v: v[s])
    P("")
    P("SIR 里的“干扰”= x_ref − 目标，包含噪声和目标的残余晚期混响；SAR 衡量伪影"
      "（无法写成目标与干扰经短滤波器组合的部分）。伪影比残余噪声更伤 ASR（Iwamoto 等，Interspeech 2022）。\n")

    # ---------------- 2. 损坏与故障 ----------------
    P("## 2. 损坏与故障\n")
    P("| 项目 | " + " | ".join(SIG_CN[s] for s in sigs if s != "mix") + " |")
    P("|---|" + "---|" * (len(sigs) - 1))

    def rate_row(name, fn):
        P(f"| {name} | " + " | ".join(fn(s) for s in sigs if s != "mix") + " |")

    def frac(v):
        v = v[np.isfinite(v)]
        return pct(np.mean(v)) if len(v) else "—"

    def worse(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return pct(np.mean(a[ok] < b[ok])) if ok.any() else "—"

    def positive(v):
        ok = np.isfinite(v)
        return pct(np.mean(v[ok] > 0)) if ok.any() else "—"

    rate_row("整条 SI-SDR 比输入差", lambda s: worse(col(rows, f"sisdr_{s}"), col(rows, "sisdr_mix")))
    rate_row("整条 ESTOI 比输入差", lambda s: worse(col(rows, f"estoi_{s}"), col(rows, "estoi_mix")))
    rate_row(f"整条 {pesq_name} 比输入差", lambda s: worse(pesq_fn(s), pesq_fn("mix")))
    rate_row("局部损坏率：比输入差的窗口比例（均值）", lambda s: frac(col(rows, f"dmg_{s}")))
    rate_row("出现过局部损坏的样本比例", lambda s: positive(col(rows, f"dmg_{s}")))
    rate_row("最差窗口改善量 5% 分位 (dB)", lambda s: fmt(describe(col(rows, f"worst_d_{s}"))["p5"]))
    P("")
    if has_bss:
        P(f"- IVA 整条通道颠倒（json swap_gain > 0）：{positive(col(rows, 'json_swap_gain'))}")
        fl = col(rows, "flip_bss")
        P(f"- IVA 局部通道颠倒（干扰通道的窗口 SI-SDR 高于目标通道）：窗口比例均值 {frac(fl)}，"
          f"出现过的样本 {positive(fl)}")
    P("")

    # ---------------- 3. 冷启动 ----------------
    P("## 3. 冷启动与收敛\n")
    P(f"从 IVA 起点（第 20 帧）开始计时；之前的输出来自 baseline GTCRN，不计入。\n")
    P(f"| 窗口平均改善量 (dB) | IVA 起点后 {COLD:g} s | 之后 | 差值 |")
    P("|---|---|---|---|")
    for s in [x for x in sigs if x != "mix"]:
        c, st = describe(col(rows, f"cold_d_{s}")), describe(col(rows, f"steady_d_{s}"))
        P(f"| {SIG_CN[s]} | {fmt_ci(c)} | {fmt_ci(st)} | {fmt(st['mean'] - c['mean'])} |")
    P("\n曲线见 `fig_convergence.png`。\n")

    # ---------------- 4. 电平 ----------------
    P("## 4. 电平：过度抑制、功率坍缩、停顿段降噪\n")
    P("| 项目 (dB) | " + " | ".join(SIG_CN[s] for s in sigs if s != "mix") + " |")
    P("|---|" + "---|" * (len(sigs) - 1))

    def mean_p5(key):
        return lambda s: f"{fmt(describe(col(rows, key.format(s)))['mean'])} / {fmt(describe(col(rows, key.format(s)))['p5'])}"

    rate_row("语音段电平，相对 ref.wav（均值 / 5% 分位）", mean_p5("lvl_{}"))
    rate_row("语音段电平，相对混合里的目标分量 α·ref（均值 / 5% 分位）", mean_p5("lvlm_{}"))
    rate_row("电平漂移：后 1/3 − 前 1/3（均值 / 5% 分位）", mean_p5("drift_{}"))
    rate_row("停顿段降噪量（均值）", lambda s: fmt(describe(col(rows, f"nrp_{s}"))["mean"]))
    P("\n语音段电平明显为负 = 过度抑制；电平漂移的 5% 分位明显为负 = 少数样本随时间越来越小（功率坍缩）。"
      "IVA 投影回参考麦、掩码作用在 IVA 输出上，输出本该与混合里的目标同电平，所以主要看“相对 α·ref”那行。\n")

    # ---------------- 5. 分组：输入难度 / 语料类别 ----------------
    mixv = col(rows, "sisdr_mix")
    ok = np.isfinite(mixv)
    ovrl = col(rows, "dnsmos_est_ovrl")
    if not np.isfinite(ovrl).any():
        ovrl = col(rows, "json_dnsmos_ovrl")
    gcols = [("输入 SI-SDR", mixv, fmt),
             ("Δ SI-SDR 最终", col(rows, "sisdr_est") - mixv, fmt)]
    if has_bss:
        gcols.append(("Δ SI-SDR IVA", col(rows, "sisdr_bss") - mixv, fmt))
    gcols += [("Δ ESTOI 最终", col(rows, "estoi_est") - col(rows, "estoi_mix"), lambda x: fmt(x, 3)),
              ("Δ PESQ 最终", pesq_fn("est") - pesq_fn("mix"), fmt),
              ("SAR 最终", col(rows, "sar_est"), fmt),
              ("局部损坏率", col(rows, "dmg_est"), pct),
              ("DNSMOS OVRL", ovrl, fmt)]

    def group_table(head, groups):
        P(f"| {head} | 条数 | " + " | ".join(c[0] for c in gcols) + " |")
        P("|---|---|" + "---|" * len(gcols))
        for name, m in groups:
            m = m & ok
            cells = [f(np.nanmean(v[m])) if np.isfinite(v[m]).any() else "—" for _, v, f in gcols]
            P(f"| {name} | {m.sum()} | " + " | ".join(cells) + " |")
        P("")

    if ok.sum() >= 8:
        edges = np.percentile(mixv[ok], [25, 50, 75])
        b = np.digitize(mixv, edges)
        P("## 5. 分组：输入难度与语料类别\n")
        P(f"**按输入 SI-SDR 四分位（Q1 最难）**，分界 {edges[0]:.1f} / {edges[1]:.1f} / {edges[2]:.1f} dB\n")
        group_table("桶", [(f"Q{q + 1}", b == q) for q in range(4)])
        cats = np.array([category(r["stem"]) for r in rows])
        uc = [c for c, _ in Counter(cats.tolist()).most_common()]
        if len(uc) > 1:
            P("**按语料类别**（从文件名 sample_N_<类别>_clean_fileid… 解析，没有类别标记的记为 read）\n")
            group_table("类别", [(c, cats == c) for c in uc])

    # ---------------- 6. 最差样本 ----------------
    P("## 6. 最差样本（建议逐条去听）\n")

    def worst(key, title, label, unit="dB", n=8, reverse=False, only_positive=False):
        cand = [r for r in rows if finite(r.get(key)) and (not only_positive or r[key] > 0)]
        cand.sort(key=lambda r: r[key], reverse=reverse)
        if cand:
            P(f"**{title}**\n")
            for r in cand[:n]:
                v = pct(r[key]) if unit == "%" else f"{r[key]:.2f} {unit}"
                P(f"- `{r['stem']}`：{label} {v}")
            P("")

    worst("worst_d_est", "最差窗口改善量（某一秒比输入差得最多）", "最差窗口 Δ")
    for r in rows:
        if finite(r.get("sisdr_est")) and finite(r.get("sisdr_mix")):
            r["_d_est"] = r["sisdr_est"] - r["sisdr_mix"]
    worst("_d_est", "整条 SI-SDR 改善最小", "ΔSI-SDR")
    worst("sar_est", "伪影最重（SAR 最低）", "SAR")
    if has_bss:
        worst("flip_bss", "IVA 局部颠倒最多", "颠倒窗口比例", unit="%", reverse=True, only_positive=True)
    worst("drift_est", "电平漂移最负（疑似功率坍缩）", "后 1/3 − 前 1/3")

    # ---------------- 7. 耗时 ----------------
    tr = col(rows, "time_ratio")
    if np.isfinite(tr).any():
        P("## 7. 耗时\n")
        P(f"json 的 time_s / 音频时长：均值 {np.nanmean(tr):.3f}，中位数 {np.nanmedian(tr):.3f}，最大 {np.nanmax(tr):.3f}。"
          "注意 time_s 包含测试脚本里算 PESQ / STOI / DNSMOS 和存盘的时间，不是纯推理的实时率。\n")

    P("## 图\n")
    for f in figs:
        P(f"- `{f}`")
    return "\n".join(L) + "\n"


# =============================================================================
# 子命令：run
# =============================================================================

def cmd_run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = load_records(args.json)
    if args.limit:
        recs = recs[:args.limit]
    ref_map = load_ref_map(args.test_json) if args.test_json else None
    cfg = dict(wav_dir=args.wav_dir, warmup_ratio=args.warmup_ratio,
               dnsmos=not args.no_dnsmos, ref_map=ref_map,
               recompute_stoi_pesq=args.recompute_stoi_pesq)
    print(f"共 {len(recs)} 条记录，{args.workers} 个进程，"
          + ("DNSMOS 重算输入 / IVA / 最终输出" if cfg["dnsmos"] else "不重算 DNSMOS（最终输出沿用 json）")
          + ("；STOI / PESQ 从音频重算" if cfg["recompute_stoi_pesq"] else "；STOI / PESQ 沿用 json"))
    t0 = time.time()
    results = []
    if args.workers > 1:
        with Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for i, res in enumerate(pool.imap(_work, recs, chunksize=2), 1):
                results.append(res)
                if i % 20 == 0 or i == len(recs):
                    print(f"  {i}/{len(recs)}  {time.time() - t0:.0f}s", flush=True)
    else:
        _init_worker(cfg)
        for i, rec in enumerate(recs, 1):
            results.append(_work(rec))
            if i % 20 == 0 or i == len(recs):
                print(f"  {i}/{len(recs)}  {time.time() - t0:.0f}s", flush=True)

    rows = [r for r, _ in results if not r.get("error")]
    bad = [r for r, _ in results if r.get("error")]
    slides = {r["stem"]: s for r, s in results if s is not None and not r.get("error")}
    write_csv(rows + bad, out / "per_utt.csv")
    if slides:
        np.savez_compressed(out / "sliding.npz",
                            **{f"{stem}|{k}": v for stem, s in slides.items() for k, v in s.items()})
    if not rows:
        print("没有成功评测的样本，请检查 --wav-dir 与文件命名。失败示例：")
        for r in bad[:5]:
            print("  ", r.get("stem"), r.get("error"))
        return
    figs = plot_all(rows, slides, out)
    report = build_report(rows, bad, len(recs), cfg, figs)
    (out / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"完成，用时 {time.time() - t0:.0f}s。输出目录：{out.resolve()}")


# =============================================================================
# 子命令：compare（两套系统逐条配对）
# =============================================================================

CMP_METRICS = [  # (列名, 显示名, 越大越好?)
    ("sisdr_est", "SI-SDR (dB)", True),
    ("estoi_est", "ESTOI", True),
    ("json_stoi_est", "STOI", True),
    ("json_pesq_est", "PESQ", True),
    ("sir_est", "SIR (dB)", True),
    ("sar_est", "SAR (dB)", True),
    ("dnsmos_est_ovrl", "DNSMOS OVRL", True),
    ("dmg_est", "局部损坏率", False),
    ("worst_d_est", "最差窗口改善量 (dB)", True),
    ("lvlm_est", "语音段电平，相对 α·ref (dB)", None),
    ("asr_rate_est", "ASR 错误率（逐条，%）", False),          # 来自 asr_eval.py 的 asr_per_utt.csv
]


def _read_rows(p):
    out = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            d = {}
            for k, v in r.items():
                try:
                    d[k] = float(v)
                except (TypeError, ValueError):
                    d[k] = v
            # run 加 --recompute-stoi-pesq 时写的是 stoi_/pesq_ 列；compare 统一落到 json_* 列上
            for k in ("stoi", "pesq"):
                if not finite(d.get(f"json_{k}_est")) and finite(d.get(f"{k}_est")):
                    d[f"json_{k}_est"] = d[f"{k}_est"]
            out[r["stem"]] = d
    return out


def read_csv(path):
    """读 run 写的 per_utt.csv（也可以直接给评测输出目录）；同目录有 asr_per_utt.csv 就按样本并进来。"""
    p = Path(path)
    if p.is_dir():
        p = p / "per_utt.csv"
    out = _read_rows(p)
    pa = p.parent / "asr_per_utt.csv"
    if pa.exists():
        for stem, d in _read_rows(pa).items():
            if stem in out:
                out[stem].update({k: v for k, v in d.items() if k.startswith("asr_")})
    return out


def asr_table(A, B, common, na, nb):
    """两套系统的 ASR 结果放进同一张表：只用两边都有结果、且伪标注相同的样本。"""
    g = lambda D, k: np.array([D[s].get(k, NAN) for s in common], dtype=float)
    ta, tb = g(A, "asr_ntok"), g(B, "asr_ntok")
    if not (np.isfinite(ta).any() and np.isfinite(tb).any()):
        return []
    da = {str(A[s].get("asr_dec", "")) for s in common if np.isfinite(A[s].get("asr_ntok", NAN))}
    db = {str(B[s].get("asr_dec", "")) for s in common if np.isfinite(B[s].get("asr_ntok", NAN))}
    if da != db or len(da) > 1:
        fmt_set = lambda d: "、".join(sorted(x or "（旧版脚本，没有记录）" for x in d))
        return ["## ASR 错误率：两边设置不一致，没有出表\n",
                f"- {na}：{fmt_set(da)}", f"- {nb}：{fmt_set(db)}",
                "- 两次运行的 --backend / --beam / 模型不同，伪标注就不是同一份，错误数不能直接比较。"
                "请用相同设置重跑其中一个系统（缓存里已有的伪标注和 ref / noisy 结果会直接复用）。\n"]
    ok = np.isfinite(g(A, "asr_err_est")) & np.isfinite(g(B, "asr_err_est")) & (ta > 0) & (ta == tb)
    n_lab = int(np.sum(np.isfinite(ta) & np.isfinite(tb) & (ta != tb)))
    if ok.sum() < 3:
        return []
    T = ta[ok]
    idx = np.random.default_rng(0).integers(0, ok.sum(), (2000, int(ok.sum())))
    rate = lambda e: 100 * e.sum() / T.sum()
    boot = lambda e: 100 * e[idx].sum(1) / T[idx].sum(1)
    rows = [("干净参考（下限）", g(A, "asr_err_ref")[ok]), ("输入(参考麦)", g(A, "asr_err_mix")[ok])]
    eb_iva = g(A, "asr_err_bss")[ok]
    if np.isfinite(eb_iva).all():
        rows.append((f"{na} IVA 目标通道", eb_iva))
    rows += [(f"{na} 最终输出", g(A, "asr_err_est")[ok]), (f"{nb} 最终输出", g(B, "asr_err_est")[ok])]
    L = [f"## ASR 错误率（{int(ok.sum())} 条共同样本，同一份伪标注）\n",
         "| 信号 | 错误率 [95% CI] | 相对下限多出的错误 |", "|---|---|---|"]
    floor = rows[0][1]
    for name, e in rows:
        b = boot(e)
        extra = "—" if e is floor else f"{rate(e) - rate(floor):.2f}"
        L.append(f"| {name} | {rate(e):.2f} [{np.percentile(b, 2.5):.2f}, {np.percentile(b, 97.5):.2f}] | {extra} |")
    ea, eb = g(A, "asr_err_est")[ok], g(B, "asr_err_est")[ok]
    dd = boot(ea) - boot(eb)
    L.append(f"\n{na} − {nb}：{rate(ea) - rate(eb):+.2f} [{np.percentile(dd, 2.5):+.2f}, {np.percentile(dd, 97.5):+.2f}]"
             "（负 = 前者更好，配对 bootstrap）。")
    # 一致性检查：ref / noisy 与系统无关，两边的识别错误数应逐条相同
    for k, cn in (("ref", "干净参考"), ("mix", "输入")):
        x, y = g(A, f"asr_err_{k}")[ok], g(B, f"asr_err_{k}")[ok]
        nd = int(np.sum(np.isfinite(x) & np.isfinite(y) & (x != y)))
        if nd:
            L.append(f"- 注意：{cn}的识别错误数有 {nd} 条两边不同，说明两次运行的解码设置（--backend / --beam）"
                     f"或 ASR 模型不一致，表里统一用 {na} 那次的结果")
    if n_lab:
        L.append(f"- 有 {n_lab} 条两边伪标注长度不同（评测段不同，通常是较短的样本），已排除")
    mixv = g(A, "sisdr_mix")[ok]
    if np.isfinite(mixv).sum() >= 8:
        edges = np.percentile(mixv[np.isfinite(mixv)], [25, 50, 75])
        bk = np.digitize(mixv, edges)
        L += ["", f"按输入 SI-SDR 四分位（Q1 最难，分界 {edges[0]:.1f} / {edges[1]:.1f} / {edges[2]:.1f} dB），各组错误率（%）：\n",
              "| 桶 | 条数 | " + " | ".join(nm for nm, _ in rows) + " |", "|---|---|" + "---|" * len(rows)]
        for q in range(4):
            m = (bk == q) & np.isfinite(mixv)
            cells = [f"{100 * e[m].sum() / max(T[m].sum(), 1e-9):.2f}" for _, e in rows]
            L.append(f"| Q{q + 1} | {int(m.sum())} | " + " | ".join(cells) + " |")
    L.append("")
    return L


def cmd_compare(args):
    from scipy.stats import wilcoxon
    A, B = read_csv(args.a), read_csv(args.b)
    na, nb = args.names
    common = sorted(set(A) & set(B))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    L = [f"# {na} vs {nb}：逐条配对比较\n",
         f"共同样本 {len(common)} 条（{na} 独有 {len(set(A) - set(B))}，{nb} 独有 {len(set(B) - set(A))}）。"
         f"差值 = {na} − {nb}；胜率 = {na} 更好的样本比例。\n",
         "| 指标 | 条数 | 均值差 [95% CI] | 中位数差 | 胜率 | Wilcoxon p |",
         "|---|---|---|---|---|---|"]
    diffs = {}
    for k, name, higher in CMP_METRICS:
        a = np.array([A[s].get(k, NAN) for s in common], dtype=float)
        b = np.array([B[s].get(k, NAN) for s in common], dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            continue
        d = a[ok] - b[ok]
        diffs[k] = (d, ok)
        lo, hi = boot_ci(d)
        win = "—" if higher is None else pct(np.mean(d > 0) if higher else np.mean(d < 0))
        try:
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = NAN
        p_txt = f"{p:.2g}" if finite(p) else "—"
        L.append(f"| {name} | {ok.sum()} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {np.median(d):+.3f} | {win} | {p_txt} |")
    L.append("\n置信区间不跨 0 且 p < 0.05 才算差异站得住；p 值未做多重比较校正。\n")

    # ASR：在共同样本上出一张表（同一批样本、同一个下限），差值对样本做配对 bootstrap
    asr_rows = asr_table(A, B, common, na, nb)
    if asr_rows:
        L += asr_rows

    mix = np.array([A[s].get("sisdr_mix", NAN) for s in common], dtype=float)
    ok = np.isfinite(mix)
    keys = [k for k in ("sisdr_est", "estoi_est", "json_pesq_est", "sar_est", "dnsmos_est_ovrl") if k in diffs]
    names = dict((m[0], m[1]) for m in CMP_METRICS)

    def diff_table(head, groups):
        L.append(f"| {head} | 条数 | " + " | ".join(names[k] for k in keys) + " |")
        L.append("|---|---|" + "---|" * len(keys))
        for name, g in groups:
            cells = []
            for k in keys:
                d, okk = diffs[k]
                m = g[okk]
                cells.append(f"{d[m].mean():+.3f}" if m.any() else "—")
            L.append(f"| {name} | {int(g.sum())} | " + " | ".join(cells) + " |")
        L.append("")

    if ok.sum() >= 8 and keys:
        edges = np.percentile(mix[ok], [25, 50, 75])
        bk = np.digitize(mix, edges)
        L.append("## 按输入难度分桶的均值差（Q1 最难）\n")
        diff_table("桶", [(f"Q{q + 1}", (bk == q) & ok) for q in range(4)])
    cats = np.array([category(s) for s in common])
    uc = [c for c, _ in Counter(cats.tolist()).most_common()]
    if len(uc) > 1 and keys:
        L.append("## 按语料类别的均值差\n")
        diff_table("类别", [(c, cats == c) for c in uc])

    if "sisdr_est" in diffs:
        plt = _style()
        d, okk = diffs["sisdr_est"]
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
        a = np.array([A[s]["sisdr_est"] for s in common], dtype=float)[okk]
        b = np.array([B[s]["sisdr_est"] for s in common], dtype=float)[okk]
        lim = [np.nanmin([a, b]) - 1, np.nanmax([a, b]) + 1]
        axs[0].plot(lim, lim, color=C_INK2, linewidth=0.8)
        axs[0].scatter(b, a, s=18, color=C_EST, alpha=0.8, edgecolors=C_SURF, linewidths=1.0)
        axs[0].set_xlabel(f"{nb}: SI-SDR (dB)")
        axs[0].set_ylabel(f"{na}: SI-SDR (dB)")
        axs[0].set_title(f"Per utterance (above the line = {na} better)")
        x = mix[okk]
        axs[1].axhline(0, color=C_INK2, linewidth=0.8)
        axs[1].scatter(x, d, s=18, color=C_EST, alpha=0.8, edgecolors=C_SURF, linewidths=1.0)
        axs[1].set_xlabel("input SI-SDR (dB)")
        axs[1].set_ylabel(f"SI-SDR difference {na} − {nb} (dB)")
        axs[1].set_title("Difference vs input difficulty")
        fig.tight_layout()
        fig.savefig(out / "fig_compare.png", dpi=150)
        plt.close(fig)
        L.append("图：`fig_compare.png`\n")
    txt = "\n".join(L) + "\n"
    (out / "compare.md").write_text(txt, encoding="utf-8")
    print(txt)


# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="对一次测试结果做完整评测")
    r.add_argument("--json", required=True, help="测试脚本写的 test_results_*.jsonl")
    r.add_argument("--wav-dir", required=True, help="测试脚本存音频的目录（test_results/<name>）")
    r.add_argument("--out", required=True, help="输出目录")
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    r.add_argument("--warmup-ratio", type=float, default=40.0,
                   help="与测试脚本同名参数一致：丢弃前 1/warmup_ratio；0 = 不丢弃")
    r.add_argument("--no-dnsmos", action="store_true", help="不重算 DNSMOS（最慢的一步）")
    r.add_argument("--recompute-stoi-pesq", action="store_true", help="从音频重算 STOI / PESQ（json 里为 NaN 时有用；PESQ 较慢）")
    r.add_argument("--test-json", default="", help="测试集 json（如 DNS3/test.json）：结果目录里缺 noisy/ref.wav 时按 est 文件名从这里找 ref 与 noisy 路径")
    r.add_argument("--limit", type=int, default=0, help="只评前 N 条（调试用）")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("compare", help="两套系统的 per_utt.csv 逐条配对比较")
    c.add_argument("a", help="系统 A 的评测输出目录（或其中的 per_utt.csv）")
    c.add_argument("b", help="系统 B，同上")
    c.add_argument("--names", nargs=2, default=["A", "B"])
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_compare)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
