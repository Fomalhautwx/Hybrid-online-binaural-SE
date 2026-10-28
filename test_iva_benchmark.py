#!/usr/bin/env python3
"""
IVA BSS Benchmark
=================
对比多种 IVA 实现在仿真混响环境下的盲源分离性能。

Methods:
  ① AuxIVA-ICD-DRISS-IMP (DUT)       — 谐波 clique + 联合去混响
  ② pyroomacoustics AuxIVA (baseline) — 标准 Laplace 先验
  ③ IVA Method A (placeholder)
  ④ IVA Method B (placeholder)

Scenarios (RT60 × angular separation):
  Low  reverb  RT60 = 0.2 s
  Mid  reverb  RT60 = 0.4 s
  High reverb  RT60 = 0.6 s

Metrics:
  SDR / SIR / SAR (mir_eval bss_eval_sources)
  SI-SDR          (scale-invariant SDR)
  SegSDR          (1 s windows, 0.5 s hop)
  PESQ / STOI     (optional)
"""

import sys, os, time, warnings, argparse
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path

import pyroomacoustics as pra
import fast_bss_eval

from scipy.signal.windows import hann
import soundfile as sf
import librosa
import json

# ---- optional perceptual metrics ----
try:
    from pesq import pesq as pesq_fn
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False

try:
    from pystoi import stoi as stoi_fn
    HAS_STOI = True
except ImportError:
    HAS_STOI = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---- DUT ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auxiva_20260726.auxiva_icd_driss_imp import AuxIVA_ICD_DRISS_IMP_harmonic_torch
from models.modules import Dereverberator
from models.AuxIVA_ISS_harmonic_v2 import AuxIVA_ISS_harmonic_nn_Frame_Module
from models.AuxIVA import AuxIVA_nn_stream, AuxIVA_stream_single
from models.AuxIVA_ICD_analytic import AuxIVA_ICD_RLS_harmonic_nn_Frame_Module as AuxIVA_ICD_RLS_harmonic_analytic

# ============================================================
# ---- Placeholder imports for your additional IVA methods ----
# Uncomment and replace with your actual module & function:
#
# from your_module_a import iva_method_a
# from your_module_b import iva_method_b
#
# They must follow the same interface as the DUT:
#   input:  x  — torch.Tensor (n_chan, n_samples)
#   output: y  — torch.Tensor (n_src,  n_samples)
# ============================================================


# ============================================================
# 1. Configuration
# ============================================================
@dataclass
class ScenarioConfig:
    label: str
    room_dim: List[float]
    rt60: float
    fs: int
    n_src: int
    mic_center: List[float]
    mic_spacing: float
    src_angles_deg: List[float]
    src_dist: float
    signal_len_sec: float
    snr_db: float


@dataclass
class STFTConfig:
    nfft: int = 512
    hop: int = 256


@dataclass
class BenchConfig:
    scenarios: List[ScenarioConfig] = field(default_factory=list)
    stft: STFTConfig = field(default_factory=STFTConfig)
    ref_mic: int = 0
    seed: int = 42
    out_dir: str = "DNS3_2Mix_noisy_reverb/iva_compare_out"


def default_bench_config() -> BenchConfig:
    common = dict(
        room_dim=[6.0, 5.0, 3.0],
        fs=16000, n_src=2,
        mic_center=[3.0, 2.5, 1.2],
        mic_spacing=0.05,
        src_angles_deg=[60, 120],
        src_dist=1.8,
        signal_len_sec=5.0,
        snr_db=30.0,
    )
    return BenchConfig(scenarios=[
        ScenarioConfig(label="Low reverb (RT60=0.2s)",  rt60=0.2, **common),
        ScenarioConfig(label="Mid reverb (RT60=0.4s)",  rt60=0.4, **common),
        ScenarioConfig(label="High reverb (RT60=0.6s)", rt60=0.6, **common),
    ])


# ============================================================
# 2. Source Generation & Room Simulation
# ============================================================
def make_sources(n_src: int, n_samples: int, fs: int, seed: int = 0) -> np.ndarray:
    """Generate speech-like harmonic sources with distinct F0 ranges.
    Returns (n_src, n_samples) float64, each row peak-normalised.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float32) / fs
    f0_ranges = [(90, 140), (200, 280), (150, 200), (260, 350)]
    sources = np.zeros((n_src, n_samples))

    for i in range(n_src):
        lo, hi = f0_ranges[i % len(f0_ranges)]
        f0 = lo + (hi - lo) * (0.5 + 0.4 * np.sin(
            2 * np.pi * rng.uniform(0.3, 0.8) * t + rng.uniform(0, 2 * np.pi)))
        phase = np.cumsum(2 * np.pi * f0 / fs)
        n_harm = min(15, int(0.45 * fs / hi))
        sig = sum((1.0 / h) * np.sin(h * phase + rng.uniform(0, 2 * np.pi))
                  for h in range(1, n_harm + 1))
        # speech-like envelope
        env = np.clip(np.sin(2 * np.pi * rng.uniform(3, 5) * t), 0, 1) ** 0.6
        for _ in range(int(n_samples / fs)):
            s = rng.integers(0, max(1, n_samples - fs // 4))
            env[s:s + rng.integers(fs // 20, fs // 4)] *= rng.uniform(0, 0.15)
        sig = sig * env + 0.02 * rng.normal(size=n_samples)
        sources[i] = sig / (np.max(np.abs(sig)) + 1e-12)
    return sources


def simulate_scenario(cfg: ScenarioConfig, stft_cfg: STFTConfig, seed: int = 0):
    """Simulate reverberant room, return mixture + source images at ref mic."""
    n_samples = int(cfg.signal_len_sec * cfg.fs)
    sources = make_sources(cfg.n_src, n_samples, cfg.fs, seed)

    mic_center = np.array(cfg.mic_center)
    offsets = (np.arange(2) - 0.5) * cfg.mic_spacing
    mic_pos = np.column_stack([
        mic_center[0] + offsets,
        np.full(2, mic_center[1]),
        np.full(2, mic_center[2]),
    ]).T  # (3, 2)

    src_positions = []
    for angle_deg in cfg.src_angles_deg[:cfg.n_src]:
        theta = np.deg2rad(angle_deg)
        src_positions.append(
            mic_center + cfg.src_dist * np.array([np.cos(theta), np.sin(theta), 0.0]))

    e_abs, max_order = pra.inverse_sabine(cfg.rt60, cfg.room_dim)
    max_order = min(max_order, 40)

    # simulate each source separately → clean source images
    all_images = []
    for i in range(cfg.n_src):
        room = pra.ShoeBox(cfg.room_dim, fs=cfg.fs,
                           materials=pra.Material(e_abs), max_order=max_order)
        room.add_source(src_positions[i], signal=sources[i])
        room.add_microphone(mic_pos)
        room.simulate()
        img = room.mic_array.signals[:, :n_samples]
        if img.shape[1] < n_samples:
            img = np.pad(img, ((0, 0), (0, n_samples - img.shape[1])))
        all_images.append(img)

    mix = sum(all_images)
    rng = np.random.default_rng(seed + 999)
    noise_power = np.mean(mix ** 2) / (10 ** (cfg.snr_db / 10))
    mix += np.sqrt(noise_power) * rng.normal(size=mix.shape) / (np.std(mix) + 1e-12)

    ref_images = np.stack([img[0] for img in all_images])  # (n_src, n_samples)
    return mix, ref_images, sources, all_images


# ============================================================
# 2b. Real Audio Loading
# ============================================================
def load_audio(path: str, target_fs: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file, resample if needed.  Returns (signal, fs).

    signal shape:
      mono   → (1, n_samples)
      stereo → (2, n_samples)
      N-ch   → (N, n_samples)
    Always float64, peak-normalised to prevent clipping.
    """
    path = str(path)
    data, orig_fs = sf.read(path, dtype="float32", always_2d=True)  # (samples, channels)
    data = data.T                                                    # (channels, samples)

    if orig_fs != target_fs:
        resampled = []
        for ch in range(data.shape[0]):
            resampled.append(librosa.resample(data[ch], orig_sr=orig_fs, target_sr=target_fs))
        data = np.stack(resampled)

    peak = np.max(np.abs(data))
    if peak > 1.0:
        data = data / peak  # prevent downstream clipping, does not affect metrics

    return data, target_fs


def load_audio_scenario(
    mix_path: str,
    ref_paths: Optional[List[str]] = None,
    fs: int = 16000,
    label: Optional[str] = None,
    ref_mic: int = 0,
):
    """Build a scenario from real audio files.

    Parameters
    ----------
    mix_path  : path to the mixture file (must be ≥ 2 channels)
    ref_paths : list of paths to mono clean reference files (one per source).
                If None, metrics cannot be computed — only separation + wav export.
    fs        : target sample rate (files are resampled if needed)
    label     : display label for plots / tables

    Returns
    -------
    mix        : (n_chan, n_samples) np.ndarray
    ref_images : (n_src, n_samples) np.ndarray  OR  None
    cfg        : ScenarioConfig (with fields needed by method wrappers)
    """
    mix, fs = load_audio(mix_path, fs)
    n_chan = mix.shape[0]
    if n_chan < 2:
        raise ValueError(f"Mixture must have ≥2 channels, got {n_chan}.  "
                         f"For a mono file, duplicate it: mix = np.stack([x, x])")

    n_samples = mix.shape[1]

    ref_images = None
    n_src = n_chan  # default: assume as many sources as channels
    if ref_paths:
        refs = []
        r, _ = load_audio(ref_paths, fs)
        if r.shape[0] > 1:
            print(f"  Warning: ref '{ref_paths}' has {r.shape[0]} channels, using ch0 only")
            r = r[:1]
        refs.append(r[0])
        # align lengths
        min_len = min(n_samples, *(r.shape[0] for r in refs))
        mix = mix[:, :min_len]
        n_samples = min_len
        if len(refs) == 1 and n_chan >= 2:
            target = refs[0]                           # 干净语音
            interfere = mix[ref_mic] - target           # 残差 = 噪声 + 其它源
            refs.append(interfere)
        ref_images = np.stack([r[:min_len] for r in refs])
        n_src = len(refs)

    if label is None:
        label = Path(mix_path).stem

    # build a minimal ScenarioConfig that method wrappers need (fs, n_src)
    cfg = ScenarioConfig(
        label=label,
        room_dim=[6, 5, 3], rt60=0.0, fs=fs, n_src=n_src,
        mic_center=[3, 2.5, 1.2], mic_spacing=0.05,
        src_angles_deg=[60, 120], src_dist=1.8,
        signal_len_sec=n_samples / fs, snr_db=99.0,
    )
    return mix, ref_images, cfg


def save_wav(signal: np.ndarray, path: str, fs: int = 16000):
    """Save (n_channels, n_samples) or (n_samples,) array as wav."""
    if signal.ndim == 1:
        signal = signal[None, :]
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / max(peak, 1.0)  # prevent clipping, keep ≤1
    sf.write(path, signal.T, fs, subtype="PCM_16")


# ============================================================
# 3. Metrics
# ============================================================
def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    ref, est = ref - ref.mean(), est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    s_target = alpha * ref
    e_noise = est - s_target
    return 10 * np.log10(np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-12) + 1e-12)


def seg_sdr(est, ref, fs, seg_sec=1.0, hop_sec=0.5):
    seg, hop = int(seg_sec * fs), int(hop_sec * fs)
    n_seg = max(1, (len(est) - seg) // hop + 1)
    sdrs = np.full(n_seg, -np.inf)
    for i in range(n_seg):
        s = i * hop
        r, e = ref[s:s + seg], est[s:s + seg]
        rp, np_ = np.sum(r ** 2), np.sum((e - r) ** 2)
        if rp > 1e-10 and np_ > 1e-10:
            sdrs[i] = 10 * np.log10(rp / np_)
    return sdrs


def _failed_metrics(reason: str) -> Dict[str, float]:
    d = {k: float("-inf") for k in ["SDR", "SIR", "SAR", "SI-SDR", "SegSDR"]}
    d["_reason"] = reason
    d["_seg_sdr_curves"] = []
    d["_perm"] = []
    if HAS_PESQ:  d["PESQ"] = float("nan")
    if HAS_STOI:  d["STOI"] = float("nan")
    return d

def evaluate(est: np.ndarray, ref: np.ndarray, fs: int) -> Dict[str, float]:
    n_src = ref.shape[0]
    L = min(est.shape[1], ref.shape[1])
    est, ref = est[:, :L].copy(), ref[:, :L].copy()

    if not np.all(np.isfinite(est)):
        return _failed_metrics("non-finite output")
    if np.max(np.abs(est)) > 1e6 * (np.max(np.abs(ref)) + 1e-8):
        return _failed_metrics(f"diverged (max|est|={np.max(np.abs(est)):.2e})")

    # ---- fast_bss_eval: same (sdr, sir, sar, perm) interface ----
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(
        ref, est,
        filter_length=512,        # Wiener 滤波器长度（mir_eval 固定 512）
        use_cg_iter=10,           # 用共轭梯度代替 solve，大幅加速
        compute_permutation=True,
        load_diag=1e-5,           # 正则化，防止静音段 Toeplitz 矩阵奇异
    )
    est_perm = est[perm]

    # ---- SI-SDR：fast_bss_eval 内置，不需要手写 ----
    si_sdrs = fast_bss_eval.si_sdr(ref, est_perm, zero_mean=True, clamp_db=80)

    # ---- SegSDR ----
    seg_curves = [seg_sdr(est_perm[n], ref[n], fs) for n in range(n_src)]
    mean_seg = np.mean([np.median(c[np.isfinite(c)]) if np.any(np.isfinite(c)) else -np.inf
                        for c in seg_curves])

    result = {
        "SDR": float(np.mean(sdr)), "SIR": float(np.mean(sir)),
        "SAR": float(np.mean(sar)), "SI-SDR": float(np.mean(si_sdrs)),
        "SegSDR": float(mean_seg),
    }

    if HAS_PESQ and fs == 16000:
        scores = []
        for n in range(n_src):
            try:
                scores.append(pesq_fn(fs, ref[n].astype(np.float32),
                                       est_perm[n].astype(np.float32), "wb"))
            except Exception:
                scores.append(float("nan"))
        result["PESQ"] = float(np.nanmean(scores))

    if HAS_STOI:
        scores = []
        for n in range(n_src):
            try:
                scores.append(stoi_fn(ref[n], est_perm[n], fs, extended=False))
            except Exception:
                scores.append(float("nan"))
        result["STOI"] = float(np.nanmean(scores))

    result["_seg_sdr_curves"] = seg_curves
    result["_perm"] = perm.tolist()
    return result

# ============================================================
# 4. Method Wrappers
# ============================================================
def run_dut(mix, stft_cfg, cfg):
    x = torch.from_numpy(mix).to(torch.float32)
    rms = torch.sqrt(torch.mean(x ** 2))
    y = AuxIVA_ICD_DRISS_IMP_harmonic_torch(
        x / (rms + 1e-12), nfft=stft_cfg.nfft, nol=stft_cfg.hop,
        alpha=0.995, alpha2=0.0, n_iter=2, fs=cfg.fs,
        beta=1.0, gamma=1.0, ref_mic=0,
        tap_len=3, tap_delay=1, warmup=0, verbose=False,
    )
    return (y * rms).numpy()


def run_pra_auxiva(mix, stft_cfg, cfg):
    nfft, hop = stft_cfg.nfft, stft_cfg.hop
    n_samples, n_chan = mix.shape[1], mix.shape[0]
    win = np.sqrt(hann(nfft, sym=False))

    pad_mix = np.pad(mix, ((0, 0), (nfft // 2, nfft // 2)))
    n_frames = (pad_mix.shape[1] - nfft) // hop + 1
    X = np.zeros((n_frames, nfft // 2 + 1, n_chan), dtype=np.complex64)
    for ch in range(n_chan):
        for t in range(n_frames):
            X[t, :, ch] = np.fft.rfft(pad_mix[ch, t * hop: t * hop + nfft] * win)

    Y = pra.bss.auxiva(X, n_iter=30, proj_back=True)

    n_src = Y.shape[2]
    out = np.zeros((n_src, pad_mix.shape[1]))
    for s in range(n_src):
        for t in range(n_frames):
            out[s, t * hop: t * hop + nfft] += np.fft.irfft(Y[t, :, s], n=nfft) * win
    return out[:, nfft // 2: nfft // 2 + n_samples]


# ============================================================
# DNN-WPE + AuxIVA-ISS Pipeline
# ============================================================

_DEREVERB  = None
_IVA_ISS = None
_DEVICE    = "cpu"

class Auxiva_gevd():
    def __init__(self, buffer: int=16):
        self.buffer_size = buffer
        self.V_cache = torch.zeros([1, 2, 2, 2, 257, 2], dtype="float32") 
        self.X_cache = torch.zeros([1, 2, self.buffer_size, 257, 2], dtype="float32")   
        self.phi_cache = torch.zeros([1,2,self.buffer_size], dtype="float32")  
    
    def forward(self, X_frame, Y_prior):
        bss_out, self.V_cache, self.X_cache, self.phi_cache, W_prev, _ = \
        AuxIVA_stream_single(X_frame, Y_prior, self.V_cache, self.X_cache, self.phi_cache, self.W_cache, n_iter=1, alpha=0.8, Lb=self.buffer_size)    # (1, 2, 257, 1) complex
    


def _init_dnn_wpe_iva(params: dict, device: str = "cpu"):
    global _DEREVERB, _IVA_ISS, _DEVICE, _IVA_ICD
    _DEVICE    = device
    _DEREVERB  = Dereverberator(nsrc=2, params=params)
    _DEREVERB.reset_state()
    _DEREVERB.to(device)
    _DEREVERB.wpe.train = False
    _IVA_ISS = AuxIVA_ISS_harmonic_nn_Frame_Module(bandsize=257, nsrc=2, alpha2=0.2, iss_iters=1).to(device)
    _IVA_ICD = AuxIVA_ICD_RLS_harmonic_analytic(alpha=0.99, alpha2=0)


@torch.no_grad()
def _run_pipeline(x: torch.Tensor, nfft: int, hop: int, IVA_MOD, 
                  fs: int = 16000, ref_mic: int = 0) -> torch.Tensor:
    """
    x : (nch, n_samples) float → y : (nsrc, n_samples) float
    """
    device = x.device
    nch, n_samples = x.shape
    nsrc = nch
    win = torch.sqrt(torch.hann_window(nfft, periodic=True, device=device, dtype=x.dtype))

    # ---- 1. STFT ----
    X = torch.stft(x, n_fft=nfft, hop_length=hop, window=win,
                   center=True, pad_mode="constant", return_complex=True)  # (nch, F, T)
    F, T = X.shape[1], X.shape[2]

    # ---- 2. 转成 Dereverberator 要的格式 ----
    X_bfdt = X.unsqueeze(0).permute(0, 2, 1, 3)               # (1, F, nch, T) complex
    X_ri   = torch.stack([X_bfdt.real, X_bfdt.imag], dim=-1)   # (1, F, nch, T, 2)
    
    Y = torch.zeros(1, nsrc, F, T, dtype=X_bfdt.dtype, device=device)
    Y_prev = X_bfdt[:, :, ref_mic:ref_mic + 1, 0].permute(0, 2, 1)  # (1, 1, F) 第 0 帧先验
    for nt in range(T):
        X_frame = X_bfdt[..., nt]
        X_der = _DEREVERB.streaming(X_frame, None)  # B F D complex
        X_iva = IVA_MOD(X_der.transpose(-1,-2), Y_prev)
        Y[:, :, :, nt] = X_iva
        Y_prev = X_iva
        
    # ---- 6. iSTFT ----
    y = torch.istft(Y.squeeze(0), n_fft=nfft, hop_length=hop,
                    window=win, center=True, length=n_samples)  # (nsrc, n_samples)
    return y


def run_auxiva_iss(mix: np.ndarray, stft_cfg, cfg) -> np.ndarray:
    with open('/home/nis/jizhe.lu/Code/2sderev/derev_params.json') as f:
        params_dict = json.load(f)
    params = params_dict['wpe_ha']
    _init_dnn_wpe_iva(params)
    _DEREVERB.reset_state()
    _IVA_ISS.reset_state()
    x   = torch.from_numpy(mix).to(torch.float32).to(_DEVICE)
    rms = torch.sqrt(torch.mean(x ** 2))
    y   = _run_pipeline(x / (rms + 1e-12), nfft=stft_cfg.nfft,
                        hop=stft_cfg.hop, IVA_MOD=_IVA_ISS, fs=cfg.fs)
    return (y * rms).cpu().numpy()

def run_auxiva_GEVD(mix: np.ndarray, stft_cfg, cfg) -> np.ndarray:
    with open('/home/nis/jizhe.lu/Code/2sderev/derev_params.json') as f:
        params_dict = json.load(f)
    params = params_dict['wpe_ha']
    _init_dnn_wpe_iva(params)
    _DEREVERB.reset_state()
    x   = torch.from_numpy(mix).to(torch.float32).to(_DEVICE)
    rms = torch.sqrt(torch.mean(x ** 2))
    y   = _run_pipeline(x / (rms + 1e-12), nfft=stft_cfg.nfft,
                        hop=stft_cfg.hop, IVA_MOD=AuxIVA_stream_single, fs=cfg.fs)
    return (y * rms).cpu().numpy()

def run_auxiva_icd(mix: np.ndarray, stft_cfg, cfg) -> np.ndarray:
    with open('/home/nis/jizhe.lu/Code/2sderev/derev_params.json') as f:
        params_dict = json.load(f)
    params = params_dict['wpe_ha']
    _init_dnn_wpe_iva(params)
    _DEREVERB.reset_state()
    x   = torch.from_numpy(mix).to(torch.float32).to(_DEVICE)
    rms = torch.sqrt(torch.mean(x ** 2))
    y   = _run_pipeline(x / (rms + 1e-12), nfft=stft_cfg.nfft,
                        hop=stft_cfg.hop, IVA_MOD=_IVA_ICD, fs=cfg.fs)
    return (y * rms).cpu().numpy()

# ============================================================
# 5. Benchmark Runner
# ============================================================
@dataclass
class MethodResult:
    name: str
    est: np.ndarray
    elapsed: float
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    label: str
    mix: np.ndarray
    ref_images: np.ndarray
    sources_dry: np.ndarray
    input_metrics: Dict[str, float]
    methods: List[MethodResult] = field(default_factory=list)


def build_method_table():
    methods = [
        ("ICD-DRISS-IMP", run_dut),
        ("pra-AuxIVA",    run_pra_auxiva),
    ]
    methods.append(("AuxIVA ISS",   run_auxiva_iss))
    #methods.append(("AuxIVA GEVD", run_auxiva_GEVD))
    methods.append(("Auxiva ICD",    run_auxiva_icd))
    return methods


def run_benchmark(bench):
    methods = build_method_table()
    results = []

    for si, cfg in enumerate(bench.scenarios):
        print(f"\n{'='*60}")
        print(f"Scenario {si+1}/{len(bench.scenarios)}: {cfg.label}")
        print(f"{'='*60}")

        mix, ref_images, sources_dry, _ = simulate_scenario(
            cfg, bench.stft, seed=bench.seed + si)
        print(f"  mix shape={mix.shape}  max|mix|={np.max(np.abs(mix)):.4f}")

        input_metrics = evaluate(mix[:cfg.n_src], ref_images, cfg.fs)
        print(f"  Input  SDR={input_metrics['SDR']:.2f}  "
              f"SIR={input_metrics['SIR']:.2f}  SAR={input_metrics['SAR']:.2f}")

        sr = ScenarioResult(label=cfg.label, mix=mix, ref_images=ref_images,
                            sources_dry=sources_dry, input_metrics=input_metrics)

        for name, fn in methods:
            print(f"  Running {name} ...", end=" ", flush=True)
            # try:
            t0 = time.perf_counter()
            est = fn(mix, bench.stft, cfg)
            elapsed = time.perf_counter() - t0
            # except Exception as e:
            #     print(f"FAILED ({e})")
            #     sr.methods.append(MethodResult(
            #         name=name, est=mix[:cfg.n_src].copy(), elapsed=0.0,
            #         metrics=_failed_metrics(str(e))))
            #     continue

            metrics = evaluate(est, ref_images, cfg.fs)
            sr.methods.append(MethodResult(name=name, est=est,
                                           elapsed=elapsed, metrics=metrics))
            reason = metrics.get("_reason", "")
            if reason:
                print(f"WARNING: {reason}")
            else:
                print(f"done ({elapsed:.1f}s)  SDR={metrics['SDR']:.2f}  "
                      f"SIR={metrics['SIR']:.2f}  SI-SDR={metrics['SI-SDR']:.2f}")
        results.append(sr)
    return results


def run_audio_benchmark(
    mix_path: str,
    ref_paths: Optional[List[str]] = None,
    stft_cfg: STFTConfig = None,
    fs: int = 16000,
    out_dir: str = "/mnt/user-data/outputs",
) -> ScenarioResult:
    """Run all registered methods on a real audio mixture.

    Saves separated wav files for every method, computes metrics if refs exist,
    and returns a ScenarioResult compatible with the existing plotting functions.
    """
    if stft_cfg is None:
        stft_cfg = STFTConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = build_method_table()

    print(f"\n{'='*60}")
    print(f"  Audio Benchmark:  {mix_path}")
    print(f"{'='*60}")

    mix, ref_images, cfg = load_audio_scenario(mix_path, ref_paths, fs)
    has_refs = ref_images is not None
    print(f"  Loaded  mix {mix.shape}  fs={cfg.fs}  duration={cfg.signal_len_sec:.2f}s")
    if has_refs:
        print(f"  Loaded  {ref_images.shape[0]} reference(s)")
    else:
        print(f"  No references — metrics will be skipped, outputs saved as wav")

    # save mixture
    mix_out = out / f"{cfg.label}_mix.wav"
    save_wav(mix, str(mix_out), fs)
    print(f"  [saved] {mix_out}")

    # input metrics (if refs available)
    if has_refs:
        input_metrics = evaluate(mix[:cfg.n_src], ref_images, cfg.fs)
        print(f"  Input  SDR={input_metrics['SDR']:.2f}  "
              f"SIR={input_metrics['SIR']:.2f}  SAR={input_metrics['SAR']:.2f}")
    else:
        input_metrics = _failed_metrics("no references")

    sr = ScenarioResult(
        label=cfg.label, mix=mix, ref_images=ref_images if has_refs else mix[:cfg.n_src],
        sources_dry=ref_images if has_refs else mix[:cfg.n_src],
        input_metrics=input_metrics,
    )

    for name, fn in methods:
        print(f"  Running {name} ...", end=" ", flush=True)
        #try:
        t0 = time.perf_counter()
        est = fn(mix, stft_cfg, cfg)
        elapsed = time.perf_counter() - t0
        # except Exception as e:
        #     print(f"FAILED ({e})")
        #     sr.methods.append(MethodResult(
        #         name=name, est=mix[:cfg.n_src].copy(), elapsed=0.0,
        #         metrics=_failed_metrics(str(e))))
        #     continue

        # save separated wav — one file per source
        tag = name.replace(" ", "_")
        for s in range(est.shape[0]):
            wav_path = out / f"{cfg.label}_{tag}_src{s}.wav"
            save_wav(est[s], str(wav_path), fs)
        print(f"done ({elapsed:.1f}s)", end="")

        if has_refs:
            metrics = evaluate(est, ref_images, cfg.fs)
            reason = metrics.get("_reason", "")
            if reason:
                print(f"  WARNING: {reason}")
            else:
                print(f"  SDR={metrics['SDR']:.2f}  SIR={metrics['SIR']:.2f}  "
                      f"SI-SDR={metrics['SI-SDR']:.2f}")
        else:
            metrics = _failed_metrics("no references")
            print(f"  (wavs saved)")

        sr.methods.append(MethodResult(name=name, est=est, elapsed=elapsed, metrics=metrics))

    return sr


# ============================================================
# 6. Visualization
# ============================================================
COLORS = {
    "Input": "#9ca3af", "ICD-DRISS-IMP": "#2563eb",
    "pra-AuxIVA": "#d97706", "AuxIVA ISS": "#059669", "Auxiva ICD": "#dc2626",
}

def _c(name):
    return COLORS.get(name, "#6b7280")


def _short_label(label: str) -> str:
    """Extract short tag from label: 'Low reverb (RT60=0.2s)' -> 'RT60=0.2s', else keep as-is."""
    if "(" in label and ")" in label:
        return label.split("(")[1].rstrip(")")
    return label


def plot_metric_bars(results, save_path):
    display = ["SDR", "SIR", "SAR", "SI-SDR", "SegSDR"]
    if HAS_PESQ: display.append("PESQ")
    if HAS_STOI: display.append("STOI")
    n_m = len(display)

    fig, axes = plt.subplots(1, n_m, figsize=(3.8 * n_m, 5), sharey=False)
    if n_m == 1: axes = [axes]

    all_names = ["Input"] + [m.name for m in results[0].methods]
    n_methods = len(all_names)

    for di, metric in enumerate(display):
        ax = axes[di]
        x = np.arange(len(results))
        w = 0.8 / n_methods
        for mi, mname in enumerate(all_names):
            vals = []
            for r in results:
                if mname == "Input":
                    vals.append(r.input_metrics.get(metric, float("nan")))
                else:
                    matched = [m for m in r.methods if m.name == mname]
                    vals.append(matched[0].metrics.get(metric, float("nan"))
                                if matched else float("nan"))
            off = (mi - (n_methods - 1) / 2) * w
            bars = ax.bar(x + off, vals, w * 0.88,
                          label=mname if di == 0 else "", color=_c(mname), alpha=0.85)
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
                            f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels([_short_label(r.label) for r in results], fontsize=8)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_ylabel("dB" if metric not in ("PESQ", "STOI") else "")
        ax.grid(axis="y", alpha=0.3)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("IVA BSS Benchmark — Metric Comparison",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


def plot_delta_bars(results, save_path):
    delta_keys = ["SDR", "SIR", "SAR", "SI-SDR"]
    n_d = len(delta_keys)
    fig, axes = plt.subplots(1, n_d, figsize=(3.8 * n_d, 5))
    if n_d == 1: axes = [axes]
    method_names = [m.name for m in results[0].methods]
    n_methods = len(method_names)

    for di, met in enumerate(delta_keys):
        ax = axes[di]
        x = np.arange(len(results))
        w = 0.8 / n_methods
        for mi, mname in enumerate(method_names):
            deltas = []
            for r in results:
                iv = r.input_metrics.get(met, 0)
                ov = r.methods[mi].metrics.get(met, 0)
                deltas.append(ov - iv if np.isfinite(ov) and np.isfinite(iv) else 0)
            off = (mi - (n_methods - 1) / 2) * w
            bars = ax.bar(x + off, deltas, w * 0.88,
                          label=mname if di == 0 else "", color=_c(mname), alpha=0.85)
            for b, v in zip(bars, deltas):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12,
                        f"{v:+.1f}", ha="center", va="bottom", fontsize=6.5)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([_short_label(r.label) for r in results], fontsize=8)
        ax.set_title(f"Δ{met}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Improvement (dB)")
        ax.grid(axis="y", alpha=0.3)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Improvement over Unprocessed Mixture",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


def plot_seg_sdr(results, save_path, fs=16000):
    n_s = len(results)
    fig, axes = plt.subplots(1, n_s, figsize=(5 * n_s, 3.8), sharey=True)
    if n_s == 1: axes = [axes]
    for si, r in enumerate(results):
        ax = axes[si]
        for m in r.methods:
            curves = m.metrics.get("_seg_sdr_curves", [])
            if not curves: continue
            ml = min(len(c) for c in curves)
            avg = np.mean([c[:ml] for c in curves], axis=0)
            t = np.arange(len(avg)) * 0.5
            ax.plot(t, avg, label=m.name, color=_c(m.name), lw=1.5)
        ax.set_title(_short_label(r.label), fontsize=10)
        ax.set_xlabel("Time (s)")
        if si == 0: ax.set_ylabel("SegSDR (dB)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Segmental SDR over Time", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


def plot_spectrograms(results, save_path, stft_cfg, fs=16000):
    n_s = len(results)
    method_names = [m.name for m in results[0].methods]
    n_rows = 1 + len(method_names)
    fig, axes = plt.subplots(n_rows, n_s, figsize=(4.5 * n_s, 2.2 * n_rows))
    if n_s == 1: axes = axes[:, None]

    nfft, hop = stft_cfg.nfft, stft_cfg.hop
    win = np.sqrt(hann(nfft, sym=False))

    def _spec(sig, ax, title):
        ps = np.pad(sig, nfft // 2)
        nf = (len(ps) - nfft) // hop + 1
        S = np.zeros((nfft // 2 + 1, nf))
        for t in range(nf):
            S[:, t] = np.abs(np.fft.rfft(ps[t * hop: t * hop + nfft] * win))
        Sdb = 20 * np.log10(S + 1e-8)
        vm = np.percentile(Sdb, 99)
        ax.imshow(Sdb, origin="lower", aspect="auto",
                  extent=[0, len(sig) / fs, 0, fs / 2000],
                  vmin=vm - 60, vmax=vm, cmap="magma")
        ax.set_title(title, fontsize=8)
        ax.set_ylabel("kHz", fontsize=7)
        ax.yaxis.set_major_locator(MaxNLocator(4))

    for si, r in enumerate(results):
        tag = _short_label(r.label)
        _spec(r.mix[0], axes[0, si], f"Mix — {tag}")
        for mi, m in enumerate(r.methods):
            perm = m.metrics.get("_perm", list(range(m.est.shape[0])))
            idx = perm[0] if perm else 0
            _spec(m.est[idx], axes[mi + 1, si], f"{m.name} — src0")
        axes[-1, si].set_xlabel("Time (s)", fontsize=8)

    fig.suptitle("Spectrograms (source 0)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


def plot_runtime(results, save_path):
    n_s = len(results)
    method_names = [m.name for m in results[0].methods]
    n_m = len(method_names)
    fig, ax = plt.subplots(figsize=(max(5, 2 * n_s), 3.5))
    x = np.arange(n_s)
    w = 0.8 / n_m
    for mi, mn in enumerate(method_names):
        times = [r.methods[mi].elapsed for r in results]
        off = (mi - (n_m - 1) / 2) * w
        bars = ax.bar(x + off, times, w * 0.88, label=mn, color=_c(mn), alpha=0.85)
        for b, t in zip(bars, times):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{t:.2f}s", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(r.label) for r in results], fontsize=9)
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


def print_summary(results):
    display = ["SDR", "SIR", "SAR", "SI-SDR", "SegSDR"]
    if HAS_PESQ: display.append("PESQ")
    if HAS_STOI: display.append("STOI")

    hdr = f"{'Scenario':<22} {'Method':<18}" + "".join(f" {m:>8}" for m in display) + f" {'Time':>7}"
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in results:
        tag = _short_label(r.label)
        row = f"{tag:<22} {'Input':<18}"
        for m in display:
            v = r.input_metrics.get(m, float("nan"))
            row += f" {v:>8.2f}" if np.isfinite(v) else f" {'N/A':>8}"
        print(row + f" {'—':>7}")
        for mr in r.methods:
            row = f"{'':22} {mr.name:<18}"
            for m in display:
                v = mr.metrics.get(m, float("nan"))
                row += f" {v:>8.2f}" if np.isfinite(v) else f" {'N/A':>8}"
            row += f" {mr.elapsed:>6.2f}s"
            print(row)
        print(sep)


# ============================================================
# 7. Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="IVA BSS Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthetic benchmark (default)
  python test_iva_benchmark.py

  # Real audio with clean references (full metrics)
  python test_iva_benchmark.py --mix mix.wav --refs spk1.wav spk2.wav

  # Real audio without references (separation only, saves wav)
  python test_iva_benchmark.py --mix mix.wav

  # Both synthetic + real audio
  python test_iva_benchmark.py --mix mix.wav --refs spk1.wav spk2.wav --also-synthetic

  # Custom STFT / sample rate
  python test_iva_benchmark.py --mix mix.wav --nfft 2048 --hop 512 --fs 44100
""")
    p.add_argument("--mix", type=str, default='/data/ssd0/jizhe.lu/Simulated/TEST/sample_5_clean_fileid_6162_test.wav',
                   help="Path to multi-channel mixture wav (≥2 ch)")
    p.add_argument("--refs", type=str, default="/data/ssd0/jizhe.lu/Simulated/TEST_TRUTH/RTS/test_clean_fileid_6162_RTS.wav",
                   help="Paths to mono clean reference wav files (one per source)")
    p.add_argument("--fs", type=int, default=16000,
                   help="Target sample rate (resamples if needed)")
    p.add_argument("--nfft", type=int, default=512)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--out-dir", type=str, default="/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/iva_compare_out")
    p.add_argument("--also-synthetic", action="store_true",
                   help="Run synthetic benchmark in addition to real audio")
    p.add_argument("--no-synthetic", action="store_true",
                   help="Skip synthetic benchmark even when no --mix is given")
    return p.parse_args()


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args = parse_args()
    stft_cfg = STFTConfig(nfft=args.nfft, hop=args.hop)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_results = []
    run_synth = (args.mix is None and not args.no_synthetic) or args.also_synthetic

    # ---- Real audio mode ----
    if args.mix is not None:
        sr = run_audio_benchmark(
            mix_path=args.mix,
            ref_paths=args.refs,
            stft_cfg=stft_cfg,
            fs=args.fs,
            out_dir=args.out_dir,
        )
        all_results.append(sr)

    # ---- Synthetic mode ----
    if run_synth:
        bench = default_bench_config()
        bench.stft = stft_cfg
        bench.out_dir = args.out_dir

        print("\n" + "=" * 60)
        print("  Synthetic Benchmark")
        print("=" * 60)
        print(f"  Scenarios : {len(bench.scenarios)}")
        print(f"  Methods   : {', '.join(n for n, _ in build_method_table())}")
        print(f"  STFT      : nfft={stft_cfg.nfft}  hop={stft_cfg.hop}")

        synth_results = run_benchmark(bench)
        all_results.extend(synth_results)

    if not all_results:
        print("Nothing to do. Use --mix or remove --no-synthetic.")
        return

    # ---- Summary table ----
    # only for results that have real metrics
    results_with_metrics = [r for r in all_results
                            if any(np.isfinite(m.metrics.get("SDR", float("-inf")))
                                   for m in r.methods)]
    if results_with_metrics:
        print_summary(results_with_metrics)

        print("\nGenerating plots ...")
        fs = args.fs
        plot_metric_bars(results_with_metrics,    out / "iva_bench_metrics.png")
        plot_delta_bars(results_with_metrics,     out / "iva_bench_delta.png")
        plot_seg_sdr(results_with_metrics,        out / "iva_bench_segsdr.png", fs)
        plot_spectrograms(results_with_metrics,   out / "iva_bench_spectrograms.png",
                          stft_cfg, fs)
        plot_runtime(results_with_metrics,        out / "iva_bench_runtime.png")
    else:
        # no-refs mode: just save spectrograms for visual inspection
        print("\nGenerating spectrograms (no metrics available) ...")
        plot_spectrograms(all_results, out / "iva_bench_spectrograms.png",
                          stft_cfg, args.fs)
        plot_runtime(all_results, out / "iva_bench_runtime.png")

    print(f"\nDone. All outputs saved to {out}")


if __name__ == "__main__":
    main()
