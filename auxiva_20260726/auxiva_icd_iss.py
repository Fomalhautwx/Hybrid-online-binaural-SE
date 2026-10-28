import numpy as np
from scipy.signal.windows import hann
from scipy.linalg import det
from tqdm import tqdm
import librosa
import torch
import torch.nn as nn


def calcu_harmonic_clique(nbands=513, fs=16000):
    F1 = 55
    ncliques = 39
    rf = 10
    F = F1 * 2 ** (np.arange(ncliques) / rf)

    nfft = 2 * (nbands - 1)
    freqs = np.arange(nbands) * fs / nfft

    C_HK = np.zeros((ncliques + 1, nbands), dtype=np.uint8)
    delta = 1 - 2 ** (-1 / 12)

    M = 10
    for h in range(ncliques):
        for k in range(nbands):
            for m in range(1, M + 1):
                if np.abs(freqs[k] - m * F[h]) / (m * F[h]) < delta:
                    C_HK[h, k] = 1
                    break
    C_HK[-1, :] = 1

    return C_HK


@torch.no_grad()
def AuxIVA_ICD_ISS_harmonic_torch(x: torch.Tensor, nfft, nol,
                         alpha=0.995, alpha2=0, n_iter=1, fs=16000,
                         beta=1.0, gamma=1.0, ref_mic=0, cuda=False, verbose=False,
                         float_dtype=torch.float64, complex_dtype=torch.complex128):
    """
        alpha       : 协方差递推的遗忘因子（越接近 1 记忆越长、越平滑）
        alpha2      : ICD 中clique方差 λ 的时间平滑系数
        n_iter      : 每帧的 ISS 迭代扫描次数 (在线场景通常取 1)
        beta, gamma : ICD 广义 pdf 的形状参数（β inter-clique、γ intra-clique)；β=γ=1 即时变高斯
        ref_mic     : 投影回放（最小失真）所参考的麦克风通道
    """
    eps = torch.finfo(float_dtype).eps
    win = np.sqrt(hann(nfft, sym=False))

    device = x.device

    n_chan, nsamples = x.shape

    # (nch, F, T)
    win = torch.from_numpy(win).to(device)
    x = x.to(device)
    X = torch.stft(x, n_fft=nfft, hop_length=nol, window=win, center=True,
                   pad_mode="constant", return_complex=True)   # (nch, F, T)

    # (F, nch, T)
    X = X.permute(1, 0, 2)

    bandsize, n_chan, n_frames = X.shape

    W = torch.eye(n_chan, dtype=complex_dtype, device=device)[None].repeat(bandsize, 1, 1)
    V = torch.zeros((bandsize, n_chan, n_chan, n_chan), dtype=complex_dtype, device=device)
    A = torch.zeros((bandsize, n_chan, n_chan), dtype=complex_dtype, device=device)
    Y = torch.zeros((bandsize, n_chan, n_frames), dtype=complex_dtype, device=device)

    C_HK = calcu_harmonic_clique(nbands=bandsize, fs=fs)
    C_HK = torch.from_numpy(C_HK).to(float_dtype).to(device)

    delta_load = 1e-3
    eye = torch.eye(n_chan, dtype=complex_dtype, device=device)
    V = delta_load * eye[None, None].repeat(bandsize, n_chan, 1, 1)
    A = eye[None].repeat(bandsize, 1, 1)

    frame_iterator = tqdm(range(n_frames)) if verbose else range(n_frames)
    for nf in frame_iterator:
        Y[:, :, nf: nf + 1] = W @ X[:, :, nf: nf + 1]

        PWr = torch.abs(Y[:, :, nf]) ** 2

        # ---------- ICD ----------
        # clique平均功率 ybar2(c) = (1/Nc) * Σ_{k∈c} |y(k)|^2
        Nc = torch.sum(C_HK, dim=-1, keepdim=True)
        ybar2 = C_HK @ PWr / Nc
        ybar2 = torch.clamp(ybar2, min=eps)
        if nf == 0:
            lam = ybar2.clone()
        else:
            lam = alpha2 * lam + (1 - alpha2) * ybar2
        lam = torch.clamp(lam, min=eps)
        # ratio = yhat^2 / λ ；rho^2 = Σ_c ratio^γ
        ratio = ybar2 / lam
        rho2 = torch.clamp(torch.sum(ratio ** gamma, dim=0, keepdim=True), min=eps)  # (1, N)
        # β · rho^(2β-2) · yhat^(2γ-2) / λ^γ
        # = β · (rho2)^(β-1) · ratio^γ / ybar2
        clique_w = beta * (rho2 ** (beta - 1.0)) * (ratio ** gamma) / ybar2 # (H, N)
        phi = C_HK.T @ clique_w # (F, N)
        p = (1 - alpha) * phi   # (F, N)

        CovX = torch.einsum('fi,fj->fij', X[:, :, nf], X[:, :, nf].conj())
        V = alpha * V + p[..., None, None] * CovX[:, None, :, :]                     # (F, N, N, N)

        # ================= ISS =================
        for _ in range(n_iter):
            # 逐个「转向源」k 做秩 1 更新
            for k in range(n_chan):
                # w_k（列向量）= 第 k 行的共轭
                wk = W[:, k, :].conj()                       # (F, N)
                # U_m @ w_k，(F, N, N)
                Uwk = torch.einsum('fmij,fj->fmi', V, wk)
                # num_m = w_m^H U_m w_k（第 m 行即 w_m^H）
                num = torch.einsum('fmi,fmi->fm', W, Uwk)    # (F, N)
                # den_m = w_k^H U_m w_k
                den = torch.einsum('fi,fmi->fm', W[:, k, :], Uwk)   # (F, N)
                den_r = torch.clamp(den.real, min=eps)
                v = num / den                                # (F, N)
                # v_kk = 1 − (w_k^H U_k w_k)^{-1/2}
                v = v.clone()
                v[:, k] = 1.0 - 1.0 / torch.sqrt(den_r[:, k])

                # ---- A = W^{-1} ----
                #   a_k ← a_k + (A v_k) / (1 − v_kk)   (col k)
                #   1 − v_kk = (w_k^H U_k w_k)^{-1/2} > 0
                Avk = torch.einsum('fim,fm->fi', A, v)       # (F, N) = Σ_m v_mk a_m
                one_minus_vkk = (1.0 - v[:, k])[:, None]     # (F, 1)
                A[:, :, k] = A[:, :, k] + Avk / one_minus_vkk

                # ---- W ← W − v_k w_k^H ----
                # w_m^H ← w_m^H − v_mk · w_k^H
                W = W - v[:, :, None] * W[:, None, k, :]

            mask = torch.isnan(W.flatten(1, 2)).any(dim=1)
            W[mask] = torch.eye(n_chan, dtype=complex_dtype, device=device)
            A[mask] = torch.eye(n_chan, dtype=complex_dtype, device=device)

        # ================= 3) project back =================
        Wbp = torch.diag_embed(A[:, ref_mic, :]) @ W          # (F, N, N)
        Y[:, :, nf:nf + 1] = Wbp @ X[:, :, nf:nf + 1]

    # (F, N, T) -> (N, F, T)
    Y = Y.permute(1, 0, 2)
    y = torch.istft(Y, n_fft=nfft, hop_length=nol, window=win, center=True, length=nsamples)
    return y
