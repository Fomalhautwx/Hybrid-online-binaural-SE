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
def AuxIVA_ICD_DRISS_IMP_harmonic_torch(x: torch.Tensor, nfft, nol,
                         alpha=0.995, alpha2=0, n_iter=2, fs=16000,
                         beta=1.0, gamma=1.0, ref_mic=0,
                         tap_len=3, tap_delay=1, warmup=0,
                         cuda=False, verbose=False,
                         float_dtype=torch.float64, complex_dtype=torch.complex128):
    """
    tap_len (L) and tap_delay (D) control the dereverberation filter
    """
    eps = torch.finfo(float_dtype).eps
    win = np.sqrt(hann(nfft, sym=False))

    device = x.device

    n_chan, nsamples = x.shape
    L = tap_len
    D = n_chan * (L + 1)

    # (nch, F, T)
    win = torch.from_numpy(win).to(device)
    x = x.to(device)
    X = torch.stft(x, n_fft=nfft, hop_length=nol, window=win, center=True,
                   pad_mode="constant", return_complex=True)   # (nch, F, T)

    # (F, nch, T)
    X = X.permute(1, 0, 2)

    bandsize, n_chan, n_frames = X.shape

    # unified DR-BSS filter  P = [W | -W Z]  (F, n_chan, D);  init P = [I_N | 0]
    P = torch.zeros((bandsize, n_chan, D), dtype=complex_dtype, device=device)
    P[:, :, :n_chan] = torch.eye(n_chan, dtype=complex_dtype, device=device)[None]
    # per-source augmented weighted covariance V_n  (F, n_chan, D, D)
    V = torch.zeros((bandsize, n_chan, D, D), dtype=complex_dtype, device=device)
    A = torch.zeros((bandsize, n_chan, n_chan), dtype=complex_dtype, device=device)
    # (F, nsrc, T)
    Y = torch.zeros((bandsize, n_chan, n_frames), dtype=complex_dtype, device=device)

    C_HK = calcu_harmonic_clique(nbands=bandsize, fs=fs)
    C_HK = torch.from_numpy(C_HK).to(float_dtype).to(device)

    delta_load = 1e-3
    V = delta_load * torch.eye(D, dtype=complex_dtype, device=device)[None, None].repeat(bandsize, n_chan, 1, 1)
    A = torch.eye(n_chan, dtype=complex_dtype, device=device)[None].repeat(bandsize, 1, 1)

    frame_iterator = tqdm(range(n_frames)) if verbose else range(n_frames)
    for nf in frame_iterator:
        # ---- build augmented (causal) input  x_tilde  (F, D) ----
        parts = [X[:, :, nf]]                                  # current frame x_t
        for l in range(L):
            idx = nf - tap_delay - l
            if idx >= 0:
                parts.append(X[:, :, idx])
            else:
                parts.append(torch.zeros((bandsize, n_chan), dtype=complex_dtype, device=device))
        xt = torch.cat(parts, dim=1)                           # (F, D)

        # ---- output with current filter P ----
        y = torch.einsum('fnd,fd->fn', P, xt)                  # (F, N)
        PWr = torch.abs(y) ** 2                                # (F, N)

        # ---------- ICD ----------
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

        xx = torch.einsum('fi,fj->fij', xt, xt.conj())         # (F, D, D)
        V = alpha * V + p[..., None, None] * xx[:, None, :, :]  # (F, n_chan, D, D)

        # ================= ISS =================
        for _ in range(n_iter):
            # --- separation ISS (updates the W part of P; maintains A = W^-1) ---
            for n in range(n_chan):
                gn = P[:, n, :].conj()                         # (F, D)  g_n = conj(row n of P)
                Vgn = torch.einsum('fmij,fj->fmi', V, gn)      # (F, n_chan, D)  V_m g_n
                num = torch.einsum('fmd,fmd->fm', P, Vgn)      # (F, N)  g_m^H V_m g_n
                den = torch.einsum('fd,fmd->fm', P[:, n, :], Vgn)   # (F, N)  g_n^H V_m g_n
                den_r = torch.clamp(den.real, min=eps)
                v = num / den
                v = v.clone()
                v[:, n] = 1.0 - 1.0 / torch.sqrt(den_r[:, n])  # diagonal (normalization)

                # A = W^-1 steering update (eq. 14) using the N-dim separation v
                Avn = torch.einsum('fim,fm->fi', A, v)         # (F, N)
                A[:, :, n] = A[:, :, n] + Avn / (1.0 - v[:, n])[:, None]

                # G/P row update:  P[m,:] -= v_m * P[n,:]   (m = 1..N)
                P = P - v[:, :, None] * P[:, None, n, :]

            # --- dereverberation ISS-SEQ (inverse-free; updates -W Z columns) ---
            if nf >= warmup:
                for n in range(n_chan, D):
                    # v_mn = (P[m,:] . V_m[:,n]) / V_m[n,n]      (eq. 13),  m = 1..N
                    num = torch.einsum('fmd,fmd->fm', P, V[:, :, :, n])  # (F, N)
                    den_r = torch.clamp(V[:, :, n, n].real, min=eps)     # (F, N)
                    v = num / den_r.to(complex_dtype)
                    # G/P column update:  P[m, n] -= v_m
                    P[:, :, n] = P[:, :, n] - v

            mask = torch.isnan(P.flatten(1, 2)).any(dim=1)
            P[mask] = 0
            P[mask, :, :n_chan] = torch.eye(n_chan, dtype=complex_dtype, device=device)
            A[mask] = torch.eye(n_chan, dtype=complex_dtype, device=device)

        # ================= 3) project back =================
        y = torch.einsum('fnd,fd->fn', P, xt) # (F, N)
        Y[:, :, nf] = A[:, ref_mic, :] * y

    # (F, nsrc, T) -> (nsrc, F, T)
    Y = Y.permute(1, 0, 2)
    y = torch.istft(Y, n_fft=nfft, hop_length=nol, window=win, center=True, length=nsamples)
    return y
