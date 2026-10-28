"""
多版本 IVA 对比测试脚本
======================

对比对象（按接口分两类）：
  逐帧流式:
    - ISS      : AuxIVA_ISS_harmonic_nn_Frame_Module   (AuxIVA_ISS_harmonic.py)
    - ICD      : AuxIVA_ICD_RLS_harmonic_nn_Frame_Module (AuxIVA_ICD_RLS_harmonic.py)
    - GEVD-cont: AuxIVA_stream_single (AuxIVA.py, 带 W_prev 连续性绑定 + eig_gap)
    - GEVD-apr : AuxIVA_stream_single (Auxiva_april.py, 按特征值大小绑定)
  整段 chunk:
    - GEVD-batch: AuxIVA_nn_enh_batch (AuxIVA.py)
    - PRA       : pyroomacoustics.bss.auxiva（标准离线 AuxIVA，无先验，作外部基线）

用法：
  # 1) 全合成自测（不需要任何文件）
  python compare_iva.py

  # 2) 用你自己的音频：mix 为双通道混合 wav；clean 为参考干净语音(单通道,与mix对齐)
  python compare_iva.py --mix mix.wav --clean clean.wav

  # 3) 有真实NN先验时（复数STFT, npy, 形状(F,T)）
  python compare_iva.py --mix mix.wav --clean clean.wav --prior prior_stft.npy

先验说明：
  - 给了 --prior 用之；否则若有 clean，用"污染版oracle"先验模拟NN输出
    （幅度误差/相位误差/残噪比例可调）；两者都没有则退化用混合ch0（弱先验）。
  - 所有依赖先验的方法喂同一份先验，保证公平。

指标：
  - SI-SDR（有clean时）：整段 + 1s滑窗收敛曲线（考察冷启动/翻转/稳定性）
  - RTF（逐帧法按逐帧耗时；chunk法按整段耗时折算）
  - GEVD-cont 额外记录 eig_gap（可分性诊断）
  - 频谱图对比

输出：outdir 下各方法增强wav + 三张PNG图。
"""

import argparse
import os
import time
import numpy as np
import soundfile as sf
import librosa
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.AuxIVA_ISS_harmonic import AuxIVA_ISS_harmonic_nn_Frame_Module
from models.AuxIVA_ICD_RLS_harmonic import AuxIVA_ICD_RLS_harmonic_nn_Frame_Module
from models.AuxIVA import AuxIVA_nn_enh_batch

torch.set_grad_enabled(False)

# ─────────────────────────────────────────────────────────────────────────────
# STFT（与你推理管线一致: n_fft=512, hop=256, sqrt-hann）
# ─────────────────────────────────────────────────────────────────────────────
NFFT, HOP = 512, 256
WIN = np.hanning(NFFT) ** 0.5
NBIN = NFFT // 2 + 1


def stft(x):   # x: (C, N) -> (C, F, T) complex
    return np.stack([librosa.stft(c, n_fft=NFFT, hop_length=HOP,
                                  win_length=NFFT, window=WIN) for c in x])


def istft(X, length):   # X: (F, T) -> (N,)
    return librosa.istft(X, n_fft=NFFT, hop_length=HOP, win_length=NFFT,
                         window=WIN, length=length)


# ─────────────────────────────────────────────────────────────────────────────
# 指标
# ─────────────────────────────────────────────────────────────────────────────

def si_sdr(est, ref, eps=1e-12):
    est = est - est.mean(); ref = ref - ref.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    t = a * ref
    return 10 * np.log10((np.dot(t, t) + eps) / (np.dot(est - t, est - t) + eps))


def sliding_si_sdr(est, ref, fs, win_s=1.0, hop_s=0.25):
    w, h = int(win_s * fs), int(hop_s * fs)
    ts, vals = [], []
    for s in range(0, len(ref) - w, h):
        r = ref[s:s + w]
        if np.dot(r, r) < 1e-6:            # 静音段跳过
            continue
        ts.append((s + w / 2) / fs)
        vals.append(si_sdr(est[s:s + w], r))
    return np.array(ts), np.array(vals)


def pick_speech_channel(outs_2ch, ref):
    """outs_2ch: (2, N)。有ref按SI-SDR选，返回(信号, 通道号, 指标)。"""
    scores = [si_sdr(outs_2ch[c], ref) for c in range(2)]
    c = int(np.argmax(scores))
    return outs_2ch[c], c, scores[c]


# ─────────────────────────────────────────────────────────────────────────────
# 合成测试场景（无外部文件时使用）
# ─────────────────────────────────────────────────────────────────────────────

def synth_scene(fs=16000, dur=6.0, snr_db=5.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    t = np.arange(n) / fs

    # "语音"：基频滑动的谐波 + 共振峰 + 音节包络
    f0 = 120 + 60 * np.sin(2 * np.pi * 0.35 * t)
    phase = 2 * np.pi * np.cumsum(f0) / fs
    s = sum((0.7 ** k) * np.sin((k + 1) * phase) for k in range(8))
    from scipy.signal import lfilter, iirpeak
    for fc, q in [(700, 5), (1800, 8), (2600, 10)]:
        b, a = iirpeak(fc / (fs / 2), q)
        s = s + 0.6 * lfilter(b, a, s)
    env = np.zeros(n); i = 0
    while i < n:
        on = rng.integers(int(0.25 * fs), int(0.6 * fs))
        off = rng.integers(int(0.1 * fs), int(0.35 * fs))
        env[i:i + on] = 1.0; i += on + off
    env = np.convolve(env, np.hanning(int(0.03 * fs)) /
                      np.hanning(int(0.03 * fs)).sum(), mode="same")
    s = s * env; s /= np.abs(s).max() + 1e-9

    # 噪声：低通白噪
    v = lfilter([1], [1, -0.6], rng.standard_normal(n))
    v /= np.std(v) + 1e-9

    # 2mic：各源到各mic用随机指数衰减FIR（模拟早期反射）
    def rir(direct_delay):
        h = np.zeros(48); h[direct_delay] = 1.0
        tail = rng.standard_normal(40) * np.exp(-np.arange(40) / 6) * 0.25
        h[direct_delay:direct_delay + 40] += tail[:48 - direct_delay]
        return h
    from scipy.signal import fftconvolve
    s_m0 = fftconvolve(s, rir(0))[:n]; s_m1 = fftconvolve(s, rir(3))[:n]
    v_m0 = fftconvolve(v, rir(2))[:n]; v_m1 = fftconvolve(v, rir(0))[:n]

    # 按段SNR缩放噪声
    g = np.sqrt(np.mean(s_m0 ** 2) / np.mean(v_m0 ** 2) / (10 ** (snr_db / 10)))
    v_m0 *= g; v_m1 *= g
    mix = np.stack([s_m0 + v_m0, s_m1 + v_m1])          # (2, N)
    return mix.astype(np.float64), s_m0.astype(np.float64)


def make_prior_stft(clean_ft, noise_ft_est, mag_err, ph_err, res, seed=1):
    """污染版oracle先验，模拟NN输出。clean_ft:(F,T)"""
    rng = np.random.default_rng(seed)
    me = 1.0 + mag_err * rng.standard_normal(clean_ft.shape)
    pe = ph_err * rng.standard_normal(clean_ft.shape)
    p = (np.abs(clean_ft) * np.clip(me, 0.05, None)
         * np.exp(1j * (np.angle(clean_ft) + pe)))
    if noise_ft_est is not None:
        p = p + res * noise_ft_est
    return p.astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# 各方法包装（统一：输入 X_ft(2,F,T), P_ft(F,T)；输出 (2,F,T), 每帧耗时列表）
# ─────────────────────────────────────────────────────────────────────────────

def run_iss(X_ft, P_ft, alpha2=0.3):
    F, T = P_ft.shape
    m = AuxIVA_ISS_harmonic_nn_Frame_Module(bandsize=F, n_chan=2, nsrc=2,
                                            alpha2=alpha2)
    X = torch.from_numpy(X_ft)[None]                     # (1,2,F,T)
    P = torch.from_numpy(P_ft)[None, None]
    out, tl = [], []
    for t in range(T):
        t0 = time.perf_counter()
        y = m(X[..., t].to(torch.complex128), P[..., t].to(torch.complex128))
        if isinstance(y, tuple):
            y = y[0]
        tl.append(time.perf_counter() - t0)
        out.append(y[0].numpy())
    return np.stack(out, axis=-1), tl                     # (2,F,T)


def run_icd(X_ft, P_ft, alpha2=0.8):
    F, T = P_ft.shape
    m = AuxIVA_ICD_RLS_harmonic_nn_Frame_Module(bandsize=F, alpha2=alpha2, float_dtype=torch.float32, complex_dtype=torch.complex64)
    X = torch.from_numpy(X_ft)[None]
    P = torch.from_numpy(P_ft)[None, None]
    out, tl = [], []
    for t in range(T):
        t0 = time.perf_counter()
        y = m(X[..., t].to(torch.complex64), P[..., t].to(torch.complex64))
        if isinstance(y, tuple):
            y = y[0]
        tl.append(time.perf_counter() - t0)
        out.append(y[0].numpy())
    return np.stack(out, axis=-1), tl


def run_gevd_stream(X_ft, P_ft, Lb=16, continuity=True):
    """AuxIVA.py（continuity=True, 6返回值）或 Auxiva_april.py（False, 4返回值）
    注：continuity 版内部将 W_prev 强转 complex64，故整体用 c64 精度调用。"""
    if continuity:
        from models.AuxIVA import AuxIVA_stream_single as f
    else:
        from models.Auxiva_april import AuxIVA_stream_single as f
    fdt, cdt = torch.float32, torch.complex64
    F, T = P_ft.shape
    V = torch.zeros(1, 2, 2, 2, F, 2, dtype=fdt)
    Xb = torch.zeros(1, 2, Lb, F, 2, dtype=fdt)
    ph = torch.zeros(1, 2, Lb, dtype=fdt)
    Wp = torch.eye(2, dtype=torch.complex64)[None, None].repeat(1, F, 1, 1)
    X = torch.from_numpy(X_ft)[None]
    P = torch.from_numpy(P_ft)[None, None]
    out, tl, gaps = [], [], []
    for t in range(T):
        t0 = time.perf_counter()
        if continuity:
            y, V, Xb, ph, Wp, gap = f(X[..., t].to(cdt), P[..., t].to(cdt),
                                      V, Xb, ph, Wp, Lb=Lb,
                                      float_dtype=fdt, complex_dtype=cdt)
            gaps.append(float(torch.real(gap).median()))
        else:
            y, V, Xb, ph = f(X[..., t].to(cdt), P[..., t].to(cdt),
                             V, Xb, ph, Lb=Lb)
        tl.append(time.perf_counter() - t0)
        out.append(y[0].numpy())
    return np.stack(out, axis=-1), tl, (np.array(gaps) if gaps else None)


def run_gevd_batch(X_ft, P_ft):
    X = torch.from_numpy(X_ft)[None]                    # (1,2,F,T)
    P = torch.from_numpy(P_ft)[None, None]
    t0 = time.perf_counter()
    Y = AuxIVA_nn_enh_batch(X.to(torch.complex64), P.to(torch.complex64))
    dt = time.perf_counter() - t0
    return Y[0].numpy(), dt


def run_pra(X_ft, n_iter=30):
    import pyroomacoustics as pra
    Xp = X_ft.transpose(2, 1, 0)                        # (T,F,C)
    t0 = time.perf_counter()
    Yp = pra.bss.auxiva(Xp.astype(np.complex64), n_iter=n_iter, proj_back=True)
    dt = time.perf_counter() - t0
    return Yp.transpose(2, 1, 0), dt                    # (C,F,T)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default='/home/nis/jizhe.lu/Code/2sderev/results/E2EpWPE_HA/online.wav', help="双通道混合wav")
    ap.add_argument("--clean", default='/data/ssd0/jizhe.lu/Simulated/TEST_TRUTH/RTS/test_clean_fileid_12199_RTS.wav', help="参考干净语音wav(单通道,与mix对齐)")
    ap.add_argument("--prior", default=None, help="真实NN先验STFT npy (F,T) complex")
    ap.add_argument("--outdir", default="iva_compare_out")
    ap.add_argument("--snr", type=float, default=5.0, help="合成场景SNR(dB)")
    ap.add_argument("--Lb", type=int, default=16)
    ap.add_argument("--prior-mag-err", type=float, default=0.5)
    ap.add_argument("--prior-phase-err", type=float, default=0.5)
    ap.add_argument("--prior-res", type=float, default=0.8,
                    help="先验中残留噪声比例")
    ap.add_argument("--prior-fallback-pert", type=float, default=0.15,
                    help="无clean/prior时，对混合ch0施加的扰动强度(0~1)，"
                         "防止 noise_like 恒零导致IVA奇异")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="跳过的方法名, 如 --skip PRA ICD")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    fs = 16000

    # ── 加载/合成音频 ────────────────────────────────────────────────────
    if args.mix:
        mix, fs = sf.read(args.mix)
        mix = mix.T if mix.ndim == 2 else np.stack([mix, mix])
        assert mix.shape[0] == 2, "mix 需为双通道"
        clean = None
        if args.clean:
            clean, _ = sf.read(args.clean)
            clean = clean[:, 0] if clean.ndim == 2 else clean
            L = min(mix.shape[1], len(clean))
            mix, clean = mix[:, :L], clean[:L]
        print(f"已加载 mix: {mix.shape}, clean: {'有' if clean is not None else '无'}")
    else:
        mix, clean = synth_scene(fs=fs, snr_db=args.snr)
        print(f"使用合成场景 (SNR={args.snr}dB), mix: {mix.shape}")

    N = mix.shape[1]
    X_ft = stft(mix).astype(np.complex64)               # (2,F,T)
    F_, T_ = X_ft.shape[1], X_ft.shape[2]
    print(f"STFT: F={F_}, T={T_}")

    # ── 先验 ────────────────────────────────────────────────────────────
    if args.prior:
        prior, _ = sf.read(args.prior)
        prior = prior[:, 0] if prior.ndim == 2 else prior
        L = min(mix.shape[1], len(prior))
        P_ft = stft(prior[:L][None])[0]
        print("使用外部NN先验")
    elif clean is not None:
        C_ft = stft(clean[None])[0]
        Nz_ft = X_ft[0] - C_ft
        P_ft = make_prior_stft(C_ft, Nz_ft, args.prior_mag_err,
                               args.prior_phase_err, args.prior_res)
        print(f"使用污染oracle先验 (幅度误差{args.prior_mag_err}, "
              f"相位误差{args.prior_phase_err}rad, 残噪{args.prior_res})")
    else:
        lvl = args.prior_fallback_pert
        rng = np.random.default_rng(2024)
        ref_mag = np.abs(X_ft[0]).mean(axis=1, keepdims=True) + 1e-8   # (F,1)
        pert = (rng.standard_normal(X_ft[0].shape)
                + 1j * rng.standard_normal(X_ft[0].shape)) / np.sqrt(2)
        P_ft = ((1.0 - lvl) * X_ft[0] + lvl * ref_mag * pert).astype(np.complex64)
        print("⚠ 无clean/prior，退化用混合ch0作先验（各先验法结果仅供参考）")

    # ── 逐个方法运行 ─────────────────────────────────────────────────────
    results = {}          # name -> dict(y2ch=(2,N), frame_ms 或 total_s, extra)
    dur_s = N / fs

    def to_time(Y_ft2):   # (2,F,T) -> (2,N)
        return np.stack([istft(Y_ft2[c], N) for c in range(2)])

    methods = []
    if "ISS" not in args.skip:
        methods.append(("ISS", lambda: run_iss(X_ft, P_ft)))
    if "ICD" not in args.skip:
        methods.append(("ICD", lambda: run_icd(X_ft, P_ft)))
    if "GEVD-cont" not in args.skip:
        methods.append(("GEVD-cont",
                        lambda: run_gevd_stream(X_ft, P_ft, args.Lb, True)))
    if "GEVD-apr" not in args.skip:
        methods.append(("GEVD-apr",
                        lambda: run_gevd_stream(X_ft, P_ft, args.Lb, False)))
    if "GEVD-batch" not in args.skip:
        methods.append(("GEVD-batch", lambda: run_gevd_batch(X_ft, P_ft)))
    if "PRA" not in args.skip:
        methods.append(("PRA", lambda: run_pra(X_ft)))

    for name, fn in methods:
        print(f"\n── 运行 {name} ──")
        try:
            ret = fn()
            extra = None
            if name in ("ISS", "ICD"):
                Y_ft2, tl = ret
                rtf = np.mean(tl) / (HOP / fs)
                results[name] = dict(y=to_time(Y_ft2), rtf=rtf,
                                     frame_ms=1e3 * np.mean(tl))
            elif name.startswith("GEVD-") and name != "GEVD-batch":
                Y_ft2, tl, gaps = ret
                rtf = np.mean(tl) / (HOP / fs)
                results[name] = dict(y=to_time(Y_ft2), rtf=rtf,
                                     frame_ms=1e3 * np.mean(tl), eig_gap=gaps)
            else:
                Y_ft2, dt = ret
                results[name] = dict(y=to_time(Y_ft2), rtf=dt / dur_s,
                                     frame_ms=None)
            print(f"   完成, RTF={results[name]['rtf']:.3f}")
        except Exception as e:
            print(f"   ✗ 失败: {type(e).__name__}: {e}")

    # ── 指标与可视化 ─────────────────────────────────────────────────────
    ref = clean
    mix0 = mix[0]
    summary = []
    sisdr_mix = si_sdr(mix0, ref) if ref is not None else None

    for name, res in results.items():
        if ref is not None:
            y_sp, ch, sc = pick_speech_channel(res["y"], ref)
        else:
            # 无参考：按与先验时域信号的相关选通道
            p_t = istft(P_ft, N)
            corr = [np.abs(np.dot(res["y"][c], p_t)) /
                    (np.linalg.norm(res["y"][c]) * np.linalg.norm(p_t) + 1e-12)
                    for c in range(2)]
            ch = int(np.argmax(corr)); y_sp, sc = res["y"][ch], None
        res["speech"], res["ch"], res["sisdr"] = y_sp, ch, sc
        sf.write(os.path.join(args.outdir, f"enh_{name}.wav"),
                 y_sp / (np.abs(y_sp).max() + 1e-9) * 0.9, fs)
        summary.append((name, sc, res["rtf"], ch))

    # 打印汇总表
    print("\n" + "=" * 62)
    print(f"{'方法':<12}{'SI-SDR(dB)':>12}{'ΔSI-SDR':>10}{'RTF':>8}{'语音通道':>9}")
    if sisdr_mix is not None:
        print(f"{'(mixture)':<12}{sisdr_mix:>12.2f}{'':>10}{'':>8}")
    for name, sc, rtf, ch in summary:
        if sc is not None:
            print(f"{name:<12}{sc:>12.2f}{sc - sisdr_mix:>10.2f}{rtf:>8.3f}{ch:>9}")
        else:
            print(f"{name:<12}{'--':>12}{'--':>10}{rtf:>8.3f}{ch:>9}")

    # ── 图1: 指标条形图 ─────────────────────────────────────────────────
    names = [s[0] for s in summary]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if ref is not None:
        vals = [s[1] for s in summary]
        axes[0].bar(names, vals, color='#4c9be8')
        axes[0].axhline(sisdr_mix, color='r', ls='--', lw=1,
                        label=f'mixture ({sisdr_mix:.1f} dB)')
        axes[0].set_ylabel('SI-SDR (dB)'); axes[0].legend(fontsize=8)
        axes[0].set_title('SI-SDR by method')
    else:
        axes[0].text(0.5, 0.5, 'no clean reference', ha='center')
    axes[1].bar(names, [s[2] for s in summary], color='#e8a54c')
    axes[1].axhline(1.0, color='r', ls='--', lw=1, label='realtime limit')
    axes[1].set_ylabel('RTF'); axes[1].set_yscale('log'); axes[1].legend(fontsize=8)
    axes[1].set_title('Real-Time Factor')
    for ax in axes: ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "iva_compare_metrics.png"), dpi=140)
    plt.close(fig)

    # ── 图2: 滑窗SI-SDR收敛曲线 (+eig_gap诊断) ─────────────────────────
    if ref is not None:
        has_gap = any(r.get("eig_gap") is not None for r in results.values())
        fig, axes = plt.subplots(2 if has_gap else 1, 1,
                                 figsize=(11, 7 if has_gap else 4.5),
                                 sharex=True)
        ax0 = axes[0] if has_gap else axes
        ts, vm = sliding_si_sdr(mix0, ref, fs)
        ax0.plot(ts, vm, 'k--', lw=1, label='mixture')
        for name, res in results.items():
            t_, v_ = sliding_si_sdr(res["speech"], ref, fs)
            ax0.plot(t_, v_, lw=1.4, label=name)
        ax0.set_ylabel('SI-SDR (dB, 1s window)')
        ax0.legend(fontsize=8, ncol=4); ax0.grid(alpha=0.3)
        ax0.set_title('Convergence / stability over time')
        if has_gap:
            for name, res in results.items():
                if res.get("eig_gap") is not None:
                    t_ax = np.arange(len(res["eig_gap"])) * HOP / fs
                    axes[1].plot(t_ax, res["eig_gap"], lw=1, label=f'{name} eig_gap')
            axes[1].set_ylabel('median eig_gap'); axes[1].set_xlabel('time (s)')
            axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        else:
            ax0.set_xlabel('time (s)')
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "iva_compare_convergence.png"), dpi=140)
        plt.close(fig)

    # ── 图3: 频谱图 ────────────────────────────────────────────────────
    n_plots = 1 + (1 if ref is not None else 0) + len(results)
    ncol = 2; nrow = (n_plots + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3 * nrow))
    axes = np.atleast_2d(axes).flatten()

    def show_spec(ax, sig, title):
        sig = np.asarray(sig, dtype=np.float64)
        # 清洗非有限值（NaN / Inf），避免 librosa 报错
        if not np.all(np.isfinite(sig)):
            n_bad = np.count_nonzero(~np.isfinite(sig))
            print(f"[warn] '{title}' 中有 {n_bad} 个非有限值，已替换为 0")
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

        S = librosa.amplitude_to_db(
            np.abs(stft(sig[None])[0]), ref=np.max)
        ax.imshow(S, aspect='auto', origin='lower', cmap='magma',
                extent=[0, N / fs, 0, fs / 2 / 1000], vmin=-70, vmax=0)
        ax.set_title(title, fontsize=9); ax.set_ylabel('kHz')

    k = 0
    show_spec(axes[k], mix0, 'mixture (ch0)'); k += 1
    if ref is not None:
        show_spec(axes[k], ref, 'clean reference'); k += 1
    for name, res in results.items():
        t_ = f'{name}  (SI-SDR {res["sisdr"]:.1f} dB)' if res["sisdr"] is not None else name
        show_spec(axes[k], res["speech"], t_); k += 1
    for j in range(k, len(axes)):
        axes[j].axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "iva_compare_spectrograms.png"), dpi=140)
    plt.close(fig)

    print(f"\n输出目录: {args.outdir}/ (wav + 3张PNG)")


if __name__ == "__main__":
    main()
