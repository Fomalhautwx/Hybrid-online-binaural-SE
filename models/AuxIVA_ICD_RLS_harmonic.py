import numpy as np
from scipy.signal.windows import hann
from scipy.linalg import det
from tqdm import tqdm
import librosa
import torch
import torch.nn as nn


def calcu_harmonic_clique(nbands=513, fs=16000):
    # generate harmonic clique matrix
    
    # 1. 定义谐波簇的基频参数
    F1 = 55
    ncliques = 39
    rf = 10
    F = F1 * 2 ** (np.arange(ncliques) / rf) # fundamental frequencies
    
    # 2. 计算每个频点对应的实际频率
    nfft = 2 * (nbands-1)
    freqs = np.arange(nbands) * fs / nfft
    
    # 3. 初始化谐波clique矩阵（ncliques+1行：ncliques个谐波簇 + 1行全1兜底）
    C_HK = np.zeros((ncliques+1, nbands), dtype=np.uint8)
    delta = 1 - 2 ** (-1/12)
    
    # 4. 遍历每个谐波簇和频点，判断频点是否属于该谐波簇
    M = 10
    for h in range(ncliques):
        for k in range(nbands):
            for m in range(1, M+1):
                if np.abs(freqs[k] - m * F[h]) / (m * F[h]) < delta:
                    C_HK[h, k] = 1
                    break
    C_HK[-1, :] = 1
        
    return C_HK


def AuxIVA_ICD_RLS_harmonic_numpy(x: np.ndarray, nfft, nol,
                         alpha=0.995, alpha2=0, verbose=False,
                         float_dtype=np.float64, complex_dtype=np.complex128):
    eps = np.finfo(float_dtype).eps
    win = np.sqrt(hann(nfft, sym=False))
    
    n_chan, nsamples = x.shape
    
    # (nch, F, T)
    X = librosa.stft(x, n_fft=nfft, hop_length=nol, window=win, center=True)
    
    # (F, nch, T)
    X = X.transpose(1, 0, 2)
    
    bandsize, n_chan, n_frames = X.shape
    
    invWDE = np.zeros((bandsize, n_chan), dtype=complex_dtype)
    # (F, nsrc, nch)
    W = np.zeros((bandsize, n_chan, n_chan), dtype=complex_dtype) # demixing matrix
    # (F, nsrc, nch, nch)
    V = np.zeros((bandsize, n_chan, n_chan, n_chan), dtype=complex_dtype) # weighted covariance
    U = np.zeros((bandsize, n_chan, n_chan, n_chan), dtype=complex_dtype) # inverse of V
    # (F, nch, nsrc)
    A = np.zeros((bandsize, n_chan, n_chan), dtype=complex_dtype) # inverse of W
    # (F, nsrc, T)
    Y = np.zeros((bandsize, n_chan, n_frames), dtype=complex_dtype)
    
    # initialize
    # (F, nsrc, nch)
    W = np.eye(n_chan, dtype=complex_dtype)[np.newaxis].repeat(bandsize, axis=0)
    # (F, nch)
    w = np.zeros((bandsize, n_chan), dtype=complex_dtype) # updated demixing matrix
    # (F, nsrc, nch)
    dW = np.zeros((bandsize, n_chan, n_chan), dtype=complex_dtype) # the difference in w before and after the update
    
    # clique
    # (H, F)
    C_HK = calcu_harmonic_clique(nbands=bandsize)
    
    frame_iterator = tqdm(range(n_frames)) if verbose else range(n_frames)
    for nf in frame_iterator:
        # (F, nsrc, 1) = (F, nsrc, nch) @ (F, nch, 1)
        Y[:, :, nf: nf+1] = W @ X[:, :, nf: nf+1]
        
        # (F, nsrc)
        PWr = np.abs(Y[:, :, nf]) ** 2
        
        # (H, nsrc)
        if nf == 0:
            r = C_HK @ PWr / np.sum(C_HK, axis=-1, keepdims=True)
        else:
            r = alpha2 * r + (1 - alpha2) * C_HK @ PWr / np.sum(C_HK, axis=-1, keepdims=True)
        r = np.maximum(r, eps)
        
        # (F, nsrc)
        # frequency-varying variance
        phi = C_HK.astype(bool).T @ (1 / r)
        # (F, nsrc)
        p = (1 - alpha) * phi
        
        # (F, nch, nch)
        CovX = np.einsum('fi,fj->fij', X[:, :, nf], X[:, :, nf].conj())
        # (F, nsrc, nch, nch)
        V = alpha * V + p[..., None, None] * CovX[:, None, :, :]
        for ch in range(n_chan):
            if nf == 0:
                # (F, nch, nch)
                invWD = (1 / np.diagonal(W @ V[:, ch, :, :], axis1=1, axis2=2))[..., None] * np.eye(n_chan)
                # (F, nch)
                invWDE = invWD[:, :, ch]
                # (F, nch, nch)
                U[:, ch, :, :] = (1 / np.diagonal(V[:, ch, :, :], axis1=1, axis2=2))[..., None] * np.eye(n_chan)
            else:
                # (F, nch, nch) @ (F, nch, nch) @ (F, nch, nch) -> (F, nch, nch)
                Unumer = p[:, ch, None, None] * U[:, ch, :, :] @ CovX @ U[:, ch, :, :].transpose(0,2,1).conj()
                # (F, 1, 1)
                Udenom = alpha ** 2 + (alpha * p[:, ch, None, None]) * np.real(X[:, None, :, nf].conj() @ U[:, ch, :, :] @ (X[:, :, nf, None]))
                Udenom = np.maximum(Udenom, 1e-6)
                # (F, nch, nch)
                U[:, ch, :, :] = (1/alpha) * U[:, ch, :, :] - Unumer / Udenom
                # (F, nch, nch)
                U[:, ch, :, :] = (U[:, ch, :, :] + U[:, ch, :, :].transpose(0,2,1).conj()) / 2
                # (F, nch)
                invWDE = (U[:, ch, :, :] @ A[:, :, ch, None]).squeeze(-1)
            
            # (F, 1, nch) @ (F, nch, nch) @ (F, mic, 1) -> (F, 1, 1)
            normCoef = np.real(invWDE[:, None, :].conj() @ V[:, ch, :, :] @ invWDE[:, :, None]).squeeze()
            sqnorm = np.maximum(np.sqrt(normCoef), eps)
            # (F, nch)
            w = invWDE / sqnorm[:, None]
            # (F, nch)
            dW[:, ch, :] = w.conj() - W[:, ch, :]
            # (F, nch)
            W[:, ch, :] = w.conj()
            
            if nf == 0:
                A = (1 / np.diagonal(W, axis1=1, axis2=2))[..., None] * np.eye(n_chan)
            else:
                # (F, nch, nch)
                Anumer = A[:, :, ch, None] @ dW[:, None, ch, :] @ A
                # (F, 1, 1)
                Adenom = np.maximum(1 + np.real(dW[:, None, ch, :] @ A[:, :, ch, None]), eps)
                A = A - Anumer / Adenom
        
        # result - back projection with A
        W[np.isnan(W).any(axis=(1,2)), :, :] = np.eye(n_chan)
        # (F, nsrc, nch)
        Wbp = (np.diagonal(A, axis1=1, axis2=2)[..., None] * np.eye(n_chan)) @ W
        # (F, nsrc, nch) @ (F, nch, 1) -> (F, nsrc, 1)
        Y[:, :, nf:nf+1] = Wbp @ X[:, :, nf:nf+1]
            
    # (F, nsrc, T) -> (nsrc, F, T)
    Y = Y.transpose(1, 0, 2)
    
    # (nsrc, wav_len)
    y = librosa.istft(Y, n_fft=nfft, hop_length=nol, window=win, center=True, length=nsamples)
    return y


@torch.no_grad()                    
def AuxIVA_ICD_RLS_harmonic_torch(x: torch.Tensor, nfft, nol,
                         alpha=0.995, alpha2=0, n_iter=1, cuda=False, verbose=False,
                         float_dtype=torch.float64, complex_dtype=torch.complex128):
    eps = torch.finfo(float_dtype).eps
    win = np.sqrt(hann(nfft, sym=False))
    
    device = x.device
    
    n_chan, nsamples = x.shape
    
    # (nch, F, T)
    win = torch.from_numpy(win).to(device)
    x = x.to(device)
    X = torch.stft(x, n_fft=nfft, hop_length=nol, window=win, center=True, 
                   pad_mode="constant", return_complex=True)
    
    # (F, nch, T)
    X = X.permute(1, 0, 2)
    
    bandsize, n_chan, n_frames = X.shape
    
    invWDE = torch.zeros((bandsize, n_chan), dtype=complex_dtype, device=device)
    # (F, nsrc, nch)
    W = torch.zeros((bandsize, n_chan, n_chan), dtype=complex_dtype, device=device) # demixing matrix
    # (F, nsrc, nch, nch)
    V = torch.zeros((bandsize, n_chan, n_chan, n_chan), dtype=complex_dtype, device=device) # weighted covariance
    U = torch.zeros((bandsize, n_chan, n_chan, n_chan), dtype=complex_dtype, device=device) # inverse of V
    # (F, nch, nsrc)
    A = torch.zeros((bandsize, n_chan, n_chan), dtype=complex_dtype, device=device) # inverse of W
    # (F, nsrc, T)
    Y = torch.zeros((bandsize, n_chan, n_frames), dtype=complex_dtype, device=device)
    
    # initialize
    # (F, nsrc, nch)
    W = torch.eye(n_chan, dtype=complex_dtype, device=device)[None].repeat(bandsize, 1, 1)
    # (F, nch)
    w = torch.zeros((bandsize, n_chan), dtype=complex_dtype, device=device) # updated demixing matrix
    # (F, nsrc, nch)
    dW = torch.zeros((bandsize, n_chan, n_chan), dtype=complex_dtype, device=device) # the difference in w before and after the update
    
    # clique
    # (H, F)
    C_HK = calcu_harmonic_clique(nbands=bandsize)
    C_HK = torch.from_numpy(C_HK).to(float_dtype).to(device)
    
    frame_iterator = tqdm(range(n_frames)) if verbose else range(n_frames)
    for nf in frame_iterator:
        for _ in range(n_iter):
            # (F, nsrc, 1) = (F, nsrc, nch) @ (F, nch, 1)
            Y[:, :, nf: nf+1] = W @ X[:, :, nf: nf+1]
            
            # (F, nsrc)
            PWr = torch.abs(Y[:, :, nf]) ** 2
            
            # (H, nsrc)
            if nf == 0:
                r = C_HK @ PWr / torch.sum(C_HK, dim=-1, keepdim=True)
            else:
                r = alpha2 * r + (1 - alpha2) * C_HK @ PWr / torch.sum(C_HK, dim=-1, keepdim=True)
            r = torch.clamp(r, min=eps)
            
            # (F, nsrc)
            # frequency-varying variance
            phi = C_HK.T @ (1 / r)
            # (F, nsrc)
            p = (1 - alpha) * phi
            
            # (F, nch, nch)
            CovX = torch.einsum('fi,fj->fij', X[:, :, nf], X[:, :, nf].conj())
            # (F, nsrc, nch, nch)
            V = alpha * V + p[..., None, None] * CovX[:, None, :, :]
            for ch in range(n_chan):
                if nf == 0:
                    # (F, nch, nch)
                    invWD = (1 / torch.diagonal(W @ V[:, ch, :, :], dim1=1, dim2=2))[..., None] * torch.eye(n_chan, device=device)
                    # (F, nch)
                    invWDE = invWD[:, :, ch]
                    # (F, nch, nch)
                    U[:, ch, :, :] = (1 / torch.diagonal(V[:, ch, :, :], dim1=1, dim2=2))[..., None] * torch.eye(n_chan, device=device)
                else:
                    # (F, nch, nch) @ (F, nch, nch) @ (F, nch, nch) -> (F, nch, nch)
                    Unumer = p[:, ch, None, None] * U[:, ch, :, :] @ CovX @ U[:, ch, :, :].permute(0,2,1).conj()
                    # (F, 1, 1)
                    Udenom = alpha ** 2 + (alpha * p[:, ch, None, None]) * torch.real(X[:, None, :, nf].conj() @ U[:, ch, :, :] @ (X[:, :, nf, None]))
                    Udenom = torch.clamp(Udenom, min=1e-6)
                    # (F, nch, nch)
                    U[:, ch, :, :] = (1/alpha) * U[:, ch, :, :] - Unumer / Udenom
                    # (F, nch, nch)
                    U[:, ch, :, :] = (U[:, ch, :, :] + U[:, ch, :, :].permute(0,2,1).conj()) / 2
                    # (F, nch)
                    invWDE = (U[:, ch, :, :] @ A[:, :, ch, None]).squeeze(-1)
                
                # (F, 1, nch) @ (F, nch, nch) @ (F, mic, 1) -> (F, 1, 1)
                normCoef = torch.real(invWDE[:, None, :].conj() @ V[:, ch, :, :] @ invWDE[:, :, None]).squeeze()
                sqnorm = torch.clamp(torch.sqrt(normCoef), min=eps)
                # (F, nch)
                w = invWDE / sqnorm[:, None]
                # (F, nch)
                dW[:, ch, :] = w.conj() - W[:, ch, :]
                # (F, nch)
                W[:, ch, :] = w.conj()
                
                if nf == 0:
                    # (F, nch, nsrc)
                    A = (1 / torch.diagonal(W, dim1=1, dim2=2))[..., None] * torch.eye(n_chan, device=device)
                else:
                    # (F, nch, nch)
                    Anumer = A[:, :, ch, None] @ dW[:, None, ch, :] @ A
                    # (F, 1, 1)
                    Adenom = torch.clamp(1 + torch.real(dW[:, None, ch, :] @ A[:, :, ch, None]), min=eps)
                    # (F, nch, nsrc)
                    A = A - Anumer / Adenom
            
            # result - back projection with A
            mask = torch.isnan(W.flatten(1, 2)).any(dim=1)
            W[mask] = torch.eye(n_chan, dtype=complex_dtype, device=device)
            
        # (F, nsrc, nch)
        Wbp = torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2)) @ W
        # (F, nsrc, nch) @ (F, nch, 1) -> (F, nsrc, 1)
        Y[:, :, nf:nf+1] = Wbp @ X[:, :, nf:nf+1]
            
    # (F, nsrc, T) -> (nsrc, F, T)
    Y = Y.permute(1, 0, 2)
    
    # (nsrc, wav_len)
    y = torch.istft(Y, n_fft=nfft, hop_length=nol, window=win, center=True, length=nsamples)
    return y                    
                        

@torch.no_grad()
def AuxIVA_ICD_RLS_harmonic_torch_batch(x: torch.Tensor, nfft, nol,
                               alpha=0.995, alpha2=0, cuda=False, verbose=False,
                               float_dtype=torch.float64, complex_dtype=torch.complex128):    
    eps = torch.finfo(float_dtype).eps
    win = np.sqrt(hann(nfft, sym=False))
    
    device = x.device
    
    batch_size, n_chan, nsamples = x.shape

    win = torch.from_numpy(win).to(device)
    # (B, nch, L)
    x = x.to(device)
    # (B*nch, L)
    x = x.flatten(0, 1)
    # (B*nch, F, T) complex
    X = torch.stft(x, n_fft=nfft, hop_length=nol, window=win, center=True, 
                   pad_mode="constant", return_complex=True)
    # (B, nch, F, T) complex
    X = X.reshape(batch_size, -1, *X.shape[1:])
    
    # (B, F, nch, T) complex
    X = X.permute(0, 2, 1, 3)
    
    batch_size, bandsize, n_chan, n_frames = X.shape
    
    invWDE = torch.zeros((batch_size, bandsize, n_chan), dtype=complex_dtype, device=device)
    # (B, F, nsrc, nch)
    W = torch.zeros((batch_size, bandsize, n_chan, n_chan), dtype=complex_dtype, device=device) # demixing matrix
    # (B, F, nsrc, nch, nch)
    V = torch.zeros((batch_size, bandsize, n_chan, n_chan, n_chan), dtype=complex_dtype, device=device) # weighted covariance
    U = torch.zeros((batch_size, bandsize, n_chan, n_chan, n_chan), dtype=complex_dtype, device=device) # inverse of V
    # (B, F, nch, nsrc)
    A = torch.zeros((batch_size, bandsize, n_chan, n_chan), dtype=complex_dtype, device=device) # inverse of W
    # (B, F, nsrc, T)
    Y = torch.zeros((batch_size, bandsize, n_chan, n_frames), dtype=complex_dtype, device=device)
    
    # initialize
    # (B, F, nsrc, nch)
    W = torch.eye(n_chan, dtype=complex_dtype, device=device)[None, None].repeat(batch_size, bandsize, 1, 1)
    # (B, F, nch)
    w = torch.zeros((batch_size, bandsize, n_chan), dtype=complex_dtype, device=device) # updated demixing matrix
    # (B, F, nsrc, nch)
    dW = torch.zeros((batch_size, bandsize, n_chan, n_chan), dtype=complex_dtype, device=device) # the difference in w before and after the update
    
    # clique
    # (H, F)
    C_HK = calcu_harmonic_clique(nbands=bandsize)
    C_HK = torch.from_numpy(C_HK).to(float_dtype).to(device)
    # identity matrix
    I = torch.eye(n_chan, dtype=complex_dtype, device=device)
    
    # covariance
    # (B, F, nch, nch, T)
    CovX = torch.einsum('bfit,bfjt->bfijt', X, X.conj())
    
    frame_iterator = tqdm(range(n_frames)) if verbose else range(n_frames)
    for nf in frame_iterator:
        # (B, F, nsrc, 1) = (B, F, nsrc, nch) @ (B, F, nch, 1)
        Y[:, :, :, nf: nf+1] = W @ X[:, :, :, nf: nf+1]
        
        # (B, F, nsrc)
        PWr = torch.abs(Y[:, :, :, nf]) ** 2
        
        # (B, H, nsrc)
        if nf == 0:
            r = C_HK @ PWr / torch.sum(C_HK, dim=-1, keepdim=True)
        else:
            r = alpha2 * r + (1 - alpha2) * C_HK @ PWr / torch.sum(C_HK, dim=-1, keepdim=True)
        r = torch.clamp(r, min=eps)
        
        # (B, F, nsrc)
        # frequency-varying variance
        phi = C_HK.T @ (1 / r)
        # (B, F, nsrc)
        p = (1 - alpha) * phi
        
        # (B, F, nsrc, nch, nch)
        V = alpha * V + p[..., None, None] * CovX[:, :, None, :, :, nf]
        for ch in range(n_chan):
            if nf == 0:
                # (B, F, nch, nch)
                invWD = (1 / torch.diagonal(W @ V[:, :, ch, :, :], dim1=2, dim2=3))[..., None] * I
                # (B, F, nch)
                invWDE = invWD[:, :, :, ch]
                # (B, F, nch, nch)
                U[:, :, ch, :, :] = (1 / torch.diagonal(V[:, :, ch, :, :], dim1=2, dim2=3))[..., None] * I
            else:
                # (B, F, nch, nch) @ (B, F, nch, nch) @ (B, F, nch, nch) -> (B, F, nch, nch)
                Unumer = p[:, :, ch, None, None] * U[:, :, ch, :, :] @ CovX[..., nf] @ U[:, :, ch, :, :].permute(0,1,3,2).conj()
                # (B, F, 1, 1)
                Udenom = alpha ** 2 + (alpha * p[:, :, ch, None, None]) * torch.real(X[:, :, None, :, nf].conj() @ U[:, :, ch, :, :] @ (X[:, :, :, nf, None]))
                Udenom = torch.clamp(Udenom, min=1e-6)
                # (B, F, nch, nch)
                U[:, :, ch, :, :] = (1/alpha) * U[:, :, ch, :, :] - Unumer / Udenom
                # (B, F, nch, nch)
                U[:, :, ch, :, :] = (U[:, :, ch, :, :] + U[:, :, ch, :, :].permute(0,1,3,2).conj()) / 2
                # (B, F, nch)
                invWDE = (U[:, :, ch, :, :] @ A[:, :, :, ch, None]).squeeze(-1)
            
            # (B, F, 1, nch) @ (B, F, nch, nch) @ (B, F, mic, 1) -> (B, F, 1, 1) -> (B, F)
            normCoef = torch.real(invWDE[:, :, None, :].conj() @ V[:, :, ch, :, :] @ invWDE[:, :, :, None])[..., 0, 0]
            sqnorm = torch.clamp(torch.sqrt(normCoef), min=eps)
            # (B, F, nch)
            w = invWDE / sqnorm[:, :, None]
            # (B, F, nch)
            dW[:, :, ch, :] = w.conj() - W[:, :, ch, :]
            # (B, F, nch)
            W[:, :, ch, :] = w.conj()
            
            if nf == 0:
                # (B, F, nch, nsrc)
                A = (1 / torch.diagonal(W, dim1=2, dim2=3))[..., None] * I
            else:
                # (B, F, nch, nch)
                Anumer = A[:, :, :, ch, None] @ dW[:, :, None, ch, :] @ A
                # (B, F, 1, 1)
                Adenom = torch.clamp(1 + torch.real(dW[:, :, None, ch, :] @ A[:, :, :, ch, None]), min=eps)
                # (B, F, nch, nsrc)
                A = A - Anumer / Adenom
        
        # result - back projection with A
        mask = torch.isnan(W.flatten(2, 3)).any(dim=2)
        W[mask] = I
        # (B, F, nsrc, nch)
        Wbp = torch.diag_embed(torch.diagonal(A, dim1=2, dim2=3)) @ W
        # (B, F, nsrc, nch) @ (B, F, nch, 1) -> (B, F, nsrc, 1)
        Y[:, :, :, nf:nf+1] = Wbp @ X[:, :, :, nf:nf+1]
            
    # (B, F, nsrc, T) -> (B, nsrc, F, T)
    Y = Y.permute(0, 2, 1, 3)
    
    # (B*nsrc, F, T)
    Y = Y.flatten(0, 1)
    
    # (B*nsrc, wav_len)
    y = torch.istft(Y, n_fft=nfft, hop_length=nol, window=win, center=True, length=nsamples)
    
    # (B, nsrc, wav_len)
    y = y.reshape(batch_size, -1, nsamples)
    return y


class AuxIVA_ICD_RLS_harmonic_Module(nn.Module):
    def __init__(self,
                 nfft, nol,
                 alpha=0.995, alpha2=0, cuda=False, verbose=False,
                 float_dtype=torch.float64, complex_dtype=torch.complex128):
        super().__init__()
        
        self.nfft = nfft
        self.nol = nol
        self.alpha = alpha
        self.alpha2 = alpha2
        self.bandsize = self.nfft // 2 + 1
        
        self.eps = torch.finfo(float_dtype).eps
        win = np.sqrt(hann(nfft, sym=False))
        self.register_buffer("win", torch.from_numpy(win))
        
        self.float_dtype = float_dtype
        self.complex_dtype = complex_dtype
        
        # clique
        # (H, F)
        C_HK = calcu_harmonic_clique(nbands=self.bandsize)
        self.register_buffer("C_HK", torch.from_numpy(C_HK).to(float_dtype))
        self.cliquesize, _ = self.C_HK.shape
    
    def forward(self, x: torch.Tensor, verbose=False):
        # (B, nch, L)
        batch_size, n_chan, nsamples = x.shape
        device = x.device
        
        # (B*nch, L)
        x = x.flatten(0, 1)
        # (B*nch, F, T) complex
        X = torch.stft(x, n_fft=self.nfft, hop_length=self.nol, window=self.win, center=True, 
                       pad_mode="constant", return_complex=True)
        
        _, bandsize, n_frames = X.shape
        # (B, nch, F, T) complex
        X = X.reshape(batch_size, -1, bandsize, n_frames)
        # (B, F, nch, T) complex
        X = X.permute(0, 2, 1, 3)
        
        # (B, F, nsrc, nch, nch)
        V = torch.zeros((batch_size, bandsize, n_chan, n_chan, n_chan), dtype=self.complex_dtype, device=device) # weighted covariance
        U = torch.zeros((batch_size, bandsize, n_chan, n_chan, n_chan), dtype=self.complex_dtype, device=device) # inverse of V
        # (F, nch, nsrc)
        A = torch.zeros((bandsize, n_chan, n_chan), dtype=self.complex_dtype, device=device) # inverse of W
        # (B, F, nsrc, T)
        Y = torch.zeros((batch_size, bandsize, n_chan, n_frames), dtype=self.complex_dtype, device=device)
    
        # initialize
        # (B, F, nsrc, nch)
        W = torch.eye(n_chan, dtype=self.complex_dtype, device=device)[None, None].repeat(batch_size, bandsize, 1, 1)
        # (B, F, nsrc, nch)
        dW = torch.zeros((batch_size, bandsize, n_chan, n_chan), dtype=self.complex_dtype, device=device) # the difference in w before and after the update
        
        # identity matrix
        I = torch.eye(n_chan, dtype=self.complex_dtype, device=device)
        r = torch.zeros((batch_size, self.cliquesize, n_chan), dtype=self.float_dtype, device=device)
        
        # covariance
        # (B, F, nch, nch, T)
        CovX = torch.einsum('bfit,bfjt->bfijt', X, X.conj())
        
        frame_iterator = tqdm(range(n_frames)) if verbose else range(n_frames)
        for nf in frame_iterator:
            # (B, F, nsrc, 1) = (B, F, nsrc, nch) @ (B, F, nch, 1)
            Y[:, :, :, nf: nf+1] = W @ X[:, :, :, nf: nf+1]
            
            # (B, F, nsrc)
            PWr = torch.abs(Y[:, :, :, nf]) ** 2
            
            # (B, H, nsrc)
            if nf == 0:
                r = self.C_HK @ PWr / torch.sum(self.C_HK, dim=-1, keepdim=True)
            else:
                r = self.alpha2 * r + (1 - self.alpha2) * self.C_HK @ PWr / torch.sum(self.C_HK, dim=-1, keepdim=True)
            r = torch.clamp(r, min=self.eps)
            
            # (B, F, nsrc)
            # frequency-varying variance
            phi = self.C_HK.T @ (1 / r)
            # (B, F, nsrc)
            p = (1 - self.alpha) * phi
            
            # (B, F, nsrc, nch, nch)
            V = self.alpha * V + p[..., None, None] * CovX[:, :, None, :, :, nf]
            for ch in range(n_chan):
                if nf == 0:
                    # (B, F, nch, nch)
                    invWD = (1 / torch.diagonal(W @ V[:, :, ch, :, :], dim1=2, dim2=3))[..., None] * I
                    # (B, F, nch)
                    invWDE = invWD[:, :, :, ch]
                    # (B, F, nch, nch)
                    U[:, :, ch, :, :] = (1 / torch.diagonal(V[:, :, ch, :, :], dim1=2, dim2=3))[..., None] * I
                else:
                    # (B, F, nch, nch) @ (B, F, nch, nch) @ (B, F, nch, nch) -> (B, F, nch, nch)
                    Unumer = p[:, :, ch, None, None] * U[:, :, ch, :, :] @ CovX[..., nf] @ U[:, :, ch, :, :].permute(0,1,3,2).conj()
                    # (B, F, 1, 1)
                    Udenom = self.alpha ** 2 + (self.alpha * p[:, :, ch, None, None]) * torch.real(X[:, :, None, :, nf].conj() @ U[:, :, ch, :, :] @ (X[:, :, :, nf, None]))
                    Udenom = torch.clamp(Udenom, min=1e-6)
                    # (B, F, nch, nch)
                    U[:, :, ch, :, :] = (1/self.alpha) * U[:, :, ch, :, :] - Unumer / Udenom
                    # (B, F, nch, nch)
                    U[:, :, ch, :, :] = (U[:, :, ch, :, :] + U[:, :, ch, :, :].permute(0,1,3,2).conj()) / 2
                    # (B, F, nch)
                    invWDE = (U[:, :, ch, :, :] @ A[:, :, :, ch, None]).squeeze(-1)
                
                # (B, F, 1, nch) @ (B, F, nch, nch) @ (B, F, mic, 1) -> (B, F, 1, 1) -> (B, F)
                normCoef = torch.real(invWDE[:, :, None, :].conj() @ V[:, :, ch, :, :] @ invWDE[:, :, :, None])[..., 0, 0]
                sqnorm = torch.clamp(torch.sqrt(normCoef), min=self.eps)
                # (B, F, nch)
                w = invWDE / sqnorm[:, :, None]
                # (B, F, nch)
                dW[:, :, ch, :] = w.conj() - W[:, :, ch, :]
                # (B, F, nch)
                W[:, :, ch, :] = w.conj()
                
                if nf == 0:
                    # (B, F, nch, nsrc)
                    A = (1 / torch.diagonal(W, dim1=2, dim2=3))[..., None] * I
                else:
                    # (B, F, nch, nch)
                    Anumer = A[:, :, :, ch, None] @ dW[:, :, None, ch, :] @ A
                    # (B, F, 1, 1)
                    Adenom = torch.clamp(1 + torch.real(dW[:, :, None, ch, :] @ A[:, :, :, ch, None]), min=self.eps)
                    # (B, F, nch, nsrc)
                    A = A - Anumer / Adenom
            
            # result - back projection with A
            mask = torch.isnan(W.flatten(2, 3)).any(dim=2)
            W[mask] = I
            # (B, F, nsrc, nch)
            Wbp = torch.diag_embed(torch.diagonal(A, dim1=2, dim2=3)) @ W
            # (B, F, nsrc, nch) @ (B, F, nch, 1) -> (B, F, nsrc, 1)
            Y[:, :, :, nf:nf+1] = Wbp @ X[:, :, :, nf:nf+1]
                
        # (B, F, nsrc, T) -> (B, nsrc, F, T)
        Y = Y.permute(0, 2, 1, 3)
        
        # (B*nsrc, F, T)
        Y = Y.flatten(0, 1)
        
        # (B*nsrc, wav_len)
        y = torch.istft(Y, n_fft=self.nfft, hop_length=self.nol, window=self.win, center=True, length=nsamples)
        
        # (B, nsrc, wav_len)
        y = y.reshape(batch_size, -1, nsamples)
        return y


@torch.no_grad()
def AuxIVA_ICD_RLS_harmonic_nn_v2(X: torch.Tensor, Y_nn: torch.Tensor, 
                         alpha=0.995, alpha2=0, n_iter=1, verbose=False,
                         float_dtype=torch.float64, complex_dtype=torch.complex128):
    """
    X: (nch, F, T) complex
    Y_nn: (nsrc, F, T) complex
    """
    
    eps = torch.finfo(float_dtype).eps
    device = X.device
    
    # (nsrc, F, T)
    X = X.to(complex_dtype)
    Y_nn = Y_nn.to(complex_dtype)
    
    # (nch, F, T) -> (F, nch, T)
    X = X.permute(1, 0, 2)
    # (nsrc, F, T) -> (F, nsrc, T)
    Y_nn = Y_nn.permute(1, 0, 2)
    
    bandsize, n_chan, n_frames = X.shape
    
    invWDE = torch.zeros((bandsize, n_chan), dtype=complex_dtype, device=device)
    # (F, nsrc, nch)
    W = torch.zeros((bandsize, n_chan, n_chan), dtype=complex_dtype, device=device) # demixing matrix
    # (F, nsrc, nch, nch)
    V = torch.zeros((bandsize, n_chan, n_chan, n_chan), dtype=complex_dtype, device=device) # weighted covariance
    U = torch.zeros((bandsize, n_chan, n_chan, n_chan), dtype=complex_dtype, device=device) # inverse of V
    # (F, nch, nsrc)
    A = torch.zeros((bandsize, n_chan, n_chan), dtype=complex_dtype, device=device) # inverse of W
    # (F, nch, T)
    Y = torch.zeros((bandsize, n_chan, n_frames), dtype=complex_dtype, device=device)
    
    # initialize
    # (F, nsrc, nch)
    W = torch.eye(n_chan, dtype=complex_dtype, device=device)[None].repeat(bandsize, 1, 1)
    # (F, nch)
    w = torch.zeros((bandsize, n_chan), dtype=complex_dtype, device=device) # updated demixing matrix
    # (F, nsrc, nch)
    dW = torch.zeros((bandsize, n_chan, n_chan), dtype=complex_dtype, device=device) # the difference in w before and after the update
    
    # clique
    # (H, F)
    C_HK = calcu_harmonic_clique(nbands=bandsize)
    C_HK = torch.from_numpy(C_HK).to(float_dtype).to(device)

    frame_iterator = tqdm(range(n_frames)) if verbose else range(n_frames)
    for nf in frame_iterator:        
        # (F, nsrc)
        
        Y[:, :, :, nf: nf+1] = W @ X[:, :, :, nf: nf+1]
        #PWr = torch.abs(Y_nn[:, :, nf]) ** 2
        PWr = torch.abs(Y[:, :, nf]) ** 2
        
        # (H, nsrc)
        if nf == 0:
            r = C_HK @ PWr / torch.sum(C_HK, dim=-1, keepdim=True)
        else:
            r = alpha2 * r + (1 - alpha2) * C_HK @ PWr / torch.sum(C_HK, dim=-1, keepdim=True)
        r = torch.clamp(r, min=eps)
        
        # (F, nsrc)
        # frequency-varying variance
        phi = C_HK.T @ (1 / r)
        # (F, nsrc)
        p = (1 - alpha) * phi
        
        # (F, nch, nch)
        CovX = torch.einsum('fi,fj->fij', X[:, :, nf], X[:, :, nf].conj())
        # (F, nsrc, nch, nch)
        V = alpha * V + p[..., None, None] * CovX[:, None, :, :]
        for ch in range(n_chan):
            if nf == 0:
                # (F, nch, nch)
                invWD = (1 / torch.diagonal(W @ V[:, ch, :, :], dim1=1, dim2=2))[..., None] * torch.eye(n_chan, device=device)
                # (F, nch)
                invWDE = invWD[:, :, ch]
                # (F, nch, nch)
                U[:, ch, :, :] = (1 / torch.diagonal(V[:, ch, :, :], dim1=1, dim2=2))[..., None] * torch.eye(n_chan, device=device)
            else:
                # (F, nch, nch) @ (F, nch, nch) @ (F, nch, nch) -> (F, nch, nch)
                Unumer = p[:, ch, None, None] * U[:, ch, :, :] @ CovX @ U[:, ch, :, :].permute(0,2,1).conj()
                # (F, 1, 1)
                Udenom = alpha ** 2 + (alpha * p[:, ch, None, None]) * torch.real(X[:, None, :, nf].conj() @ U[:, ch, :, :] @ (X[:, :, nf, None]))
                Udenom = torch.clamp(Udenom, min=1e-6)
                # (F, nch, nch)
                U[:, ch, :, :] = (1/alpha) * U[:, ch, :, :] - Unumer / Udenom
                # (F, nch, nch)
                U[:, ch, :, :] = (U[:, ch, :, :] + U[:, ch, :, :].permute(0,2,1).conj()) / 2
                # (F, nch)
                invWDE = (U[:, ch, :, :] @ A[:, :, ch, None]).squeeze(-1)
            
            # (F, 1, nch) @ (F, nch, nch) @ (F, mic, 1) -> (F, 1, 1)
            normCoef = torch.real(invWDE[:, None, :].conj() @ V[:, ch, :, :] @ invWDE[:, :, None]).squeeze()
            sqnorm = torch.clamp(torch.sqrt(normCoef), min=eps)
            # (F, nch)
            w = invWDE / sqnorm[:, None]
            # (F, nch)
            dW[:, ch, :] = w.conj() - W[:, ch, :]
            # (F, nch)
            W[:, ch, :] = w.conj()
            
            if nf == 0:
                # (F, nch, nsrc)
                A = (1 / torch.diagonal(W, dim1=1, dim2=2))[..., None] * torch.eye(n_chan, device=device)
            else:
                # (F, nch, nch)
                Anumer = A[:, :, ch, None] @ dW[:, None, ch, :] @ A
                # (F, 1, 1)
                Adenom = torch.clamp(1 + torch.real(dW[:, None, ch, :] @ A[:, :, ch, None]), min=eps)
                # (F, nch, nsrc)
                A = A - Anumer / Adenom
        
        # result - back projection with A
        mask = torch.isnan(W.flatten(1, 2)).any(dim=1)
        W[mask] = torch.eye(n_chan, dtype=complex_dtype, device=device)
        # (F, nsrc, nch)
        Wbp = torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2)) @ W
        # (F, nsrc, nch) @ (F, nch, 1) -> (F, nsrc, 1)
        Y[:, :, nf:nf+1] = Wbp @ X[:, :, nf:nf+1]
            
    # (F, nsrc, T) -> (nsrc, F, T)
    Y = Y.permute(1, 0, 2)
    return Y

@torch.no_grad()
def AuxIVA_ICD_RLS_harmonic_nn_v2_batch(X: torch.Tensor, Y_nn: torch.Tensor, 
                               alpha=0.995, alpha2=0, verbose=False,
                               float_dtype=torch.float64, complex_dtype=torch.complex128):
    """
    X: (B, nch, F, T) complex
    Y_nn: (B, nsrc, F, T) complex
    """
    
    eps = torch.finfo(float_dtype).eps
    device = X.device
    
    # (B, nch, F, T)
    X = X.to(complex_dtype)
    # (B, nsrc, F, T)
    Y_nn = Y_nn.to(complex_dtype)
    
    # (B, nch, F, T) -> (B, F, nch, T)
    X = X.permute(0, 2, 1, 3)
    # (B, nsrc, F, T) -> (B, F, nsrc, T)
    Y_nn = Y_nn.permute(0, 2, 1, 3)
    
    batch_size, bandsize, n_chan, n_frames = X.shape
    
    invWDE = torch.zeros((batch_size, bandsize, n_chan), dtype=complex_dtype, device=device)
    # (B, F, nsrc, nch, nch)
    V = torch.zeros((batch_size, bandsize, n_chan, n_chan, n_chan), dtype=complex_dtype, device=device) # weighted covariance
    U = torch.zeros((batch_size, bandsize, n_chan, n_chan, n_chan), dtype=complex_dtype, device=device) # inverse of V
    # (B, F, nch, nsrc)
    A = torch.zeros((batch_size, bandsize, n_chan, n_chan), dtype=complex_dtype, device=device) # inverse of W
    # (B, F, nsrc, T)
    Y = torch.zeros((batch_size, bandsize, n_chan, n_frames), dtype=complex_dtype, device=device)
    
    # initialize
    # (B, F, nsrc, nch)
    W = torch.eye(n_chan, dtype=complex_dtype, device=device)[None, None].repeat(batch_size, bandsize, 1, 1)
    # (B, F, nch)
    w = torch.zeros((batch_size, bandsize, n_chan), dtype=complex_dtype, device=device) # updated demixing matrix
    # (B, F, nsrc, nch)
    dW = torch.zeros((batch_size, bandsize, n_chan, n_chan), dtype=complex_dtype, device=device) # the difference in w before and after the update
    
    # clique
    # (H, F)
    C_HK = calcu_harmonic_clique(nbands=bandsize)
    C_HK = torch.from_numpy(C_HK).to(device)

    frame_iterator = tqdm(range(n_frames)) if verbose else range(n_frames)
    for nf in frame_iterator:        
        # (B, F, nsrc)
        PWr = torch.abs(Y_nn[:, :, :, nf]) ** 2
        
        # (B, H, nsrc)
        if nf == 0:
            r = C_HK.to(float_dtype) @ PWr / torch.sum(C_HK, dim=-1, keepdim=True)
        else:
            r = alpha2 * r + (1 - alpha2) * C_HK.to(float_dtype) @ PWr / torch.sum(C_HK, dim=-1, keepdim=True)
        r = torch.clamp(r, min=eps)
        
        # (B, F, nsrc)
        # frequency-varying variance
        phi = C_HK.T.to(float_dtype) @ (1 / r)
        # (B, F, nsrc)
        p = (1 - alpha) * phi
        
        # (B, F, nch, nch)
        CovX = torch.einsum('bfi,bfj->bfij', X[:, :, :, nf], X[:, :, :, nf].conj())
        # (B, F, nsrc, nch, nch)
        V = alpha * V + p[..., None, None] * CovX[:, :, None, :, :]
        for ch in range(n_chan):
            if nf == 0:
                # (B, F, nch, nch)
                invWD = (1 / torch.diagonal(W @ V[:, :, ch, :, :], dim1=2, dim2=3))[..., None] * torch.eye(n_chan, device=device)
                # (B, F, nch)
                invWDE = invWD[:, :, :, ch]
                # (B, F, nch, nch)
                U[:, :, ch, :, :] = (1 / torch.diagonal(V[:, :, ch, :, :], dim1=2, dim2=3))[..., None] * torch.eye(n_chan, device=device)
            else:
                # (B, F, nch, nch) @ (B, F, nch, nch) @ (B, F, nch, nch) -> (B, F, nch, nch)
                Unumer = p[:, :, ch, None, None] * U[:, :, ch, :, :] @ CovX @ U[:, :, ch, :, :].permute(0,1,3,2).conj()
                # (B, F, 1, 1)
                Udenom = alpha ** 2 + (alpha * p[:, :, ch, None, None]) * torch.real(X[:, :, None, :, nf].conj() @ U[:, :, ch, :, :] @ (X[:, :, :, nf, None]))
                Udenom = torch.clamp(Udenom, min=1e-6)
                # (B, F, nch, nch)
                U[:, :, ch, :, :] = (1/alpha) * U[:, :, ch, :, :] - Unumer / Udenom
                # (B, F, nch, nch)
                U[:, :, ch, :, :] = (U[:, :, ch, :, :] + U[:, :, ch, :, :].permute(0,1,3,2).conj()) / 2
                # (B, F, nch)
                invWDE = (U[:, :, ch, :, :] @ A[:, :, :, ch, None]).squeeze(-1)
            
            # (B, F, 1, nch) @ (B, F, nch, nch) @ (B, F, mic, 1) -> (B, F, 1, 1) -> (B, F)
            normCoef = torch.real(invWDE[:, :, None, :].conj() @ V[:, :, ch, :, :] @ invWDE[:, :, :, None])[..., 0, 0]
            sqnorm = torch.clamp(torch.sqrt(normCoef), min=eps)
            # (B, F, nch)
            w = invWDE / sqnorm[:, :, None]
            # (B, F, nch)
            dW[:, :, ch, :] = w.conj() - W[:, :, ch, :]
            # (B, F, nch)
            W[:, :, ch, :] = w.conj()
            
            if nf == 0:
                # (B, F, nch, nsrc)
                A = (1 / torch.diagonal(W, dim1=2, dim2=3))[..., None] * torch.eye(n_chan, device=device)
            else:
                # (B, F, nch, nch)
                Anumer = A[:, :, :, ch, None] @ dW[:, :, None, ch, :] @ A
                # (B, F, 1, 1)
                Adenom = torch.clamp(1 + torch.real(dW[:, :, None, ch, :] @ A[:, :, :, ch, None]), min=eps)
                # (B, F, nch, nsrc)
                A = A - Anumer / Adenom
        
        # result - back projection with A
        mask = torch.isnan(W.flatten(2, 3)).any(dim=2)
        W[mask] = torch.eye(n_chan, dtype=complex_dtype, device=device)
        # (B, F, nsrc, nch)
        Wbp = torch.diag_embed(torch.diagonal(A, dim1=2, dim2=3)) @ W
        # (B, F, nsrc, nch) @ (B, F, nch, 1) -> (B, F, nsrc, 1)
        Y[:, :, :, nf:nf+1] = Wbp @ X[:, :, :, nf:nf+1]
            
    # (B, F, nsrc, T) -> (B, nsrc, F, T)
    Y = Y.permute(0, 2, 1, 3)
    return Y


class AuxIVA_ICD_RLS_harmonic_nn_v2_Module(nn.Module):
    def __init__(self,
                 bandsize=257, alpha=0.995, alpha2=0, cuda=False, verbose=False,
                 float_dtype=torch.float64, complex_dtype=torch.complex64):
        super().__init__()
        
        self.alpha = alpha
        self.alpha2 = alpha2
        self.bandsize = bandsize
        
        self.float_dtype = float_dtype
        self.complex_dtype = complex_dtype
        self.eps = torch.finfo(float_dtype).eps
        
        # clique
        # (H, F)
        C_HK = calcu_harmonic_clique(nbands=self.bandsize)
        self.register_buffer("C_HK", torch.from_numpy(C_HK).float())
        self.cliquesize, _ = self.C_HK.shape
        
        self.initialized = False
    
    def forward(self, X: torch.Tensor, Y_nn: torch.Tensor):
        device = X.device
        # if Y_nn.shape[1] == 1:
        #     energy_ratio = Y_nn.abs() / (X[:, 0:1].abs() + 1e-8)
        #     min_mask = 0.03  # 最小保留比例
        #     mask = torch.where(energy_ratio > 0.5, torch.full_like(energy_ratio, min_mask), torch.ones_like(energy_ratio))
        #     interf = (X[:, 0:1] - Y_nn) * mask
        #     Y_nn = torch.cat([interf, Y_nn], dim=1) # (B, 2, F)
            
        if Y_nn.shape[1] == 1:
            # noise_like = (torch.randn_like(Y_nn) + 1j * torch.randn_like(Y_nn))
            noise_like = X[:, 0:1] - Y_nn 
            Y_nn = torch.cat([Y_nn, noise_like], dim=1)     #这个效果怎么又这么好
            
        # (B, nch, F, T)
        X = X.to(self.complex_dtype)
        # (B, nsrc, F, T)
        Y_nn = Y_nn.to(self.complex_dtype)
        
        # (B, nch, F, T) -> (B, F, nch, T)
        X = X.permute(0, 2, 1, 3)
        # (B, nsrc, F, T) -> (B, F, nsrc, T)
        Y_nn = Y_nn.permute(0, 2, 1, 3)
        
        batch_size, bandsize, n_chan, n_frames = X.shape
        
        # (B, F, nsrc, nch, nch)
        V = torch.zeros((batch_size, bandsize, n_chan, n_chan, n_chan), dtype=self.complex_dtype, device=device) # weighted covariance
        U = torch.zeros((batch_size, bandsize, n_chan, n_chan, n_chan), dtype=self.complex_dtype, device=device) # inverse of V
        # (F, nch, nsrc)
        A = torch.zeros((bandsize, n_chan, n_chan), dtype=self.complex_dtype, device=device) # inverse of W
        # (B, F, nsrc, T)
        Y = torch.zeros((batch_size, bandsize, n_chan, n_frames), dtype=self.complex_dtype, device=device)
    
        # initialize
        # (B, F, nsrc, nch)
        W = torch.eye(n_chan, dtype=self.complex_dtype, device=device)[None, None].repeat(batch_size, bandsize, 1, 1)
        # (B, F, nsrc, nch)
        dW = torch.zeros((batch_size, bandsize, n_chan, n_chan), dtype=self.complex_dtype, device=device) # the difference in w before and after the update
        
        # identity matrix
        I = torch.eye(n_chan, dtype=self.complex_dtype, device=device)
        r = torch.zeros((batch_size, self.cliquesize, n_chan), dtype=self.float_dtype, device=device)
        
        # covariance
        # (B, F, nch, nch, T)
        CovX = torch.einsum('bfit,bfjt->bfijt', X, X.conj())
        
        for nf in range(n_frames):
            # (B, F, nsrc, 1) = (B, F, nsrc, nch) @ (B, F, nch, 1)
            Y[:, :, :, nf: nf+1] = W @ X[:, :, :, nf: nf+1]
            
            # (B, F, nsrc)
            PWr = torch.abs(Y_nn[:, :, :, nf]) ** 2
            
            # (B, H, nsrc)
            if nf == 0:
                r = self.C_HK @ PWr / torch.sum(self.C_HK, dim=-1, keepdim=True)
            else:
                r = self.alpha2 * r + (1 - self.alpha2) * self.C_HK @ PWr / torch.sum(self.C_HK, dim=-1, keepdim=True)
            r = torch.clamp(r, min=self.eps)
            
            # (B, F, nsrc)
            # frequency-varying variance
            phi = self.C_HK.T @ (1 / r)
            # (B, F, nsrc)
            p = (1 - self.alpha) * phi
            
            # (B, F, nsrc, nch, nch)
            V = self.alpha * V + p[..., None, None] * CovX[:, :, None, :, :, nf]
            for ch in range(n_chan):
                if nf == 0:
                    # (B, F, nch, nch)
                    invWD = (1 / torch.diagonal(W @ V[:, :, ch, :, :], dim1=2, dim2=3))[..., None] * I
                    # (B, F, nch)
                    invWDE = invWD[:, :, :, ch]
                    # (B, F, nch, nch)
                    U[:, :, ch, :, :] = (1 / torch.diagonal(V[:, :, ch, :, :], dim1=2, dim2=3))[..., None] * I
                else:
                    # (B, F, nch, nch) @ (B, F, nch, nch) @ (B, F, nch, nch) -> (B, F, nch, nch)
                    Unumer = p[:, :, ch, None, None] * U[:, :, ch, :, :] @ CovX[..., nf] @ U[:, :, ch, :, :].permute(0,1,3,2).conj()
                    # (B, F, 1, 1)
                    Udenom = self.alpha ** 2 + (self.alpha * p[:, :, ch, None, None]) * torch.real(X[:, :, None, :, nf].conj() @ U[:, :, ch, :, :] @ (X[:, :, :, nf, None]))
                    Udenom = torch.clamp(Udenom, min=1e-6)
                    # (B, F, nch, nch)
                    U[:, :, ch, :, :] = (1/self.alpha) * U[:, :, ch, :, :] - Unumer / Udenom
                    # (B, F, nch, nch)
                    U[:, :, ch, :, :] = (U[:, :, ch, :, :] + U[:, :, ch, :, :].permute(0,1,3,2).conj()) / 2
                    # (B, F, nch)
                    invWDE = (U[:, :, ch, :, :] @ A[:, :, :, ch, None]).squeeze(-1)
                
                # (B, F, 1, nch) @ (B, F, nch, nch) @ (B, F, mic, 1) -> (B, F, 1, 1) -> (B, F)
                normCoef = torch.real(invWDE[:, :, None, :].conj() @ V[:, :, ch, :, :] @ invWDE[:, :, :, None])[..., 0, 0]
                sqnorm = torch.clamp(torch.sqrt(normCoef), min=self.eps)
                # (B, F, nch)
                w = invWDE / sqnorm[:, :, None]
                # (B, F, nch)
                dW[:, :, ch, :] = w.conj() - W[:, :, ch, :]
                # (B, F, nch)
                W[:, :, ch, :] = w.conj()
                
                if nf == 0:
                    # (B, F, nch, nsrc)
                    A = (1 / torch.diagonal(W, dim1=2, dim2=3))[..., None] * I
                else:
                    # (B, F, nch, nch)
                    Anumer = A[:, :, :, ch, None] @ dW[:, :, None, ch, :] @ A
                    # (B, F, 1, 1)
                    Adenom = torch.clamp(1 + torch.real(dW[:, :, None, ch, :] @ A[:, :, :, ch, None]), min=self.eps)
                    # (B, F, nch, nsrc)
                    A = A - Anumer / Adenom
            A = torch.nan_to_num(A, nan=self.eps) 
            # result - back projection with A
            mask = torch.isnan(W.flatten(2, 3)).any(dim=2)
            W[mask] = I
            # (B, F, nsrc, nch)
            Wbp = torch.diag_embed(torch.diagonal(A, dim1=2, dim2=3)) @ W
            # (B, F, nsrc, nch) @ (B, F, nch, 1) -> (B, F, nsrc, 1)
            Y[:, :, :, nf:nf+1] = Wbp @ X[:, :, :, nf:nf+1]
                
        # (B, F, nsrc, T) -> (B, nsrc, F, T)
        Y = Y.permute(0, 2, 1, 3)
        return Y
    
class AuxIVA_ICD_RLS_harmonic(nn.Module):
    def __init__(self,
                 bandsize=257, alpha=0.995, alpha2=0, cuda=False, verbose=False,
                 float_dtype=torch.float32, complex_dtype=torch.complex64):
        super().__init__()
        
        self.alpha = alpha
        self.alpha2 = alpha2
        self.bandsize = bandsize
        
        self.float_dtype = float_dtype
        self.complex_dtype = complex_dtype
        self.eps = torch.finfo(float_dtype).eps
        
        # clique
        # (H, F)
        C_HK = calcu_harmonic_clique(nbands=self.bandsize)
        self.register_buffer("C_HK", torch.from_numpy(C_HK).float())
        self.cliquesize, _ = self.C_HK.shape
        
        self.initialized = False
    
    def forward(self, X: torch.Tensor, Y_nn: torch.Tensor):
        device = X.device
        
        if Y_nn.shape[1] == 1:
            noise_like = X[:, 0:1] - Y_nn 
            Y_nn = torch.cat([Y_nn, noise_like], dim=1)
            
        # (B, nch, F, T)
        X = X.to(self.complex_dtype)
        # (B, nsrc, F, T)
        Y_nn = Y_nn.to(self.complex_dtype)
        
        # (B, nch, F, T) -> (B, F, nch, T)
        X = X.permute(0, 2, 1, 3)
        # (B, nsrc, F, T) -> (B, F, nsrc, T)
        Y_nn = Y_nn.permute(0, 2, 1, 3)
        
        batch_size, bandsize, n_chan, n_frames = X.shape
        
        # (B, F, nsrc, nch, nch)
        V = torch.zeros((batch_size, bandsize, n_chan, n_chan, n_chan), dtype=self.complex_dtype, device=device) 
        U = torch.zeros((batch_size, bandsize, n_chan, n_chan, n_chan), dtype=self.complex_dtype, device=device) 
        A = torch.zeros((batch_size, bandsize, n_chan, n_chan), dtype=self.complex_dtype, device=device) 
        Y = torch.zeros((batch_size, bandsize, n_chan, n_frames), dtype=self.complex_dtype, device=device)
    
        # initialize
        W = torch.eye(n_chan, dtype=self.complex_dtype, device=device)[None, None].repeat(batch_size, bandsize, 1, 1)
        dW = torch.zeros((batch_size, bandsize, n_chan, n_chan), dtype=self.complex_dtype, device=device) 
        
        I = torch.eye(n_chan, dtype=self.complex_dtype, device=device)
        r = torch.zeros((batch_size, self.cliquesize, n_chan), dtype=self.float_dtype, device=device)
        
        # covariance
        #CovX = torch.einsum('bfit,bfjt->bfijt', X, X.conj())
        
        # (B, F, nsrc, 1:T) = (B, F, nsrc, nch) @ (B, F, nch, 1:T)
        Y = torch.einsum('bfij,bfjk->bfik', W, X)
        
        # (B, F, nsrc, T)
        PWr = (torch.abs(Y_nn) ** 2).permute(0,3,1,2)
        
        # (B, H, nsrc)
        if not self.initialized:
            r = self.C_HK @ PWr / torch.sum(self.C_HK, dim=-1, keepdim=True)
        else:
            r = self.alpha2 * r + (1 - self.alpha2) * self.C_HK @ PWr / torch.sum(self.C_HK, dim=-1, keepdim=True)
        r = torch.clamp(r, min=self.eps).to(self.float_dtype)
        
        # (B, F, nsrc)
        phi = self.C_HK.T @ (1 / r)
        p = (1 - self.alpha) * phi
        p = p.permute(0,2,3,1)
        
        # (B, F, nsrc, nch, nch)
        Xw = p.unsqueeze(-2) * X.unsqueeze(2)
        V = self.alpha * V + torch.einsum('bfsct, bfdct -> bfsdc', Xw, X.conj().unsqueeze(2))/2
        #V = self.alpha * V + p[..., None, None] * CovX[:, :, None, :, :, :].mean(dim=-1, keepdim=True)
        
        for ch in range(n_chan):
            if not self.initialized:
                invWD = (1 / torch.diagonal(W @ V[:, :, ch, :, :], dim1=2, dim2=3))[..., None] * I
                invWDE = invWD[:, :, :, ch]
                U[:, :, ch, :, :] = (1 / torch.diagonal(V[:, :, ch, :, :], dim1=2, dim2=3))[..., None] * I
            else:
                # Unumer = p[:, :, ch, None, None] * U[:, :, ch, :, :] @ CovX.mean(dim=-1) @ U[:, :, ch, :, :].permute(0, 1, 3, 2).conj()
                # Udenom = self.alpha ** 2 + (self.alpha * p[:, :, ch, None, None]) * torch.real(X[:, :, None, :, :].conj() @ U[:, :, ch, :, :] @ X[:, :, :, :, None])
                Unumer = U[:, :, ch, :, :] @ V[:, :, ch, :, :]@ U[:, :, ch, :, :].permute(0, 1, 3, 2).conj()
                Ux = torch.einsum('bfij, bfjt -> bfit', U[:, :, ch, :, :], X)
                quad = torch.einsum('bfit, bfit -> bft', X.conj(), Ux)
                weighted = (p[:, :, ch] * torch.real(quad)).mean(dim=-1)  # (B, F)
                Udenom = self.alpha ** 2 + (self.alpha * weighted[..., None, None])
                Udenom = torch.clamp(Udenom, min=1e-6)
                U[:, :, ch, :, :] = (1 / self.alpha) * U[:, :, ch, :, :] - Unumer / Udenom
                U[:, :, ch, :, :] = (U[:, :, ch, :, :] + U[:, :, ch, :, :].permute(0, 1, 3, 2).conj()) / 2
                invWDE = (U[:, :, ch, :, :] @ A[:, :, :, ch, None]).squeeze(-1)
            
            normCoef = torch.real(invWDE[:, :, None, :].conj() @ V[:, :, ch, :, :] @ invWDE[:, :, :, None])[..., 0, 0]
            sqnorm = torch.clamp(torch.sqrt(normCoef), min=self.eps)
            w = invWDE / sqnorm[:, :, None]
            dW[:, :, ch, :] = w.conj() - W[:, :, ch, :]
            W[:, :, ch, :] = w.conj()
            
            if not self.initialized:
                A = (1 / torch.diagonal(W, dim1=2, dim2=3))[..., None] * I
            else:
                Anumer = A[:, :, :, ch, None] @ dW[:, :, None, ch, :] @ A
                Adenom = torch.clamp(1 + torch.real(dW[:, :, None, ch, :] @ A[:, :, :, ch, None]), min=self.eps)
                A = A - Anumer / Adenom
        
        A = torch.nan_to_num(A, nan=self.eps)
        mask = torch.isnan(W.flatten(2, 3)).any(dim=2)
        W[mask] = I
        Wbp = torch.diag_embed(torch.diagonal(A, dim1=2, dim2=3)) @ W
        Y_out = Wbp @ X
        Y_out = Y_out.permute(0, 2, 1, 3)
        
        self.V = V
        self.U = U
        self.W = W
        self.A = A
        self.r = r
        self.dW = dW
        self.initialized = True
        
        return Y_out
    
class AuxIVA_ICD_RLS_harmonic_nn_Frame_Module(nn.Module):
    def __init__(self,
                 bandsize=257, n_chan=2, nsrc=2,
                 alpha=0.995, alpha2=0, float_dtype=torch.float64, complex_dtype=torch.complex128):
        super().__init__()
        self.alpha = alpha
        self.alpha2 = alpha2
        self.bandsize = bandsize
        self.n_chan = n_chan
        self.nsrc = nsrc
        self.float_dtype = float_dtype
        self.complex_dtype = complex_dtype
        self.eps = torch.finfo(float_dtype).eps

        # clique
        C_HK = calcu_harmonic_clique(nbands=self.bandsize)
        self.register_buffer("C_HK", torch.from_numpy(C_HK).to(float_dtype))
        self.cliquesize, _ = self.C_HK.shape

        # 状态 buffer
        self.register_buffer("V", torch.zeros((1, bandsize, n_chan, n_chan, n_chan), dtype=complex_dtype))
        self.register_buffer("U", torch.zeros((1, bandsize, n_chan, n_chan, n_chan), dtype=complex_dtype))
        self.register_buffer("W", torch.eye(n_chan, dtype=complex_dtype)[None, None].repeat(1, bandsize, 1, 1))
        self.register_buffer("dW", torch.zeros((1, bandsize, n_chan, n_chan), dtype=complex_dtype))
        self.register_buffer("A", torch.zeros((bandsize, n_chan, n_chan), dtype=complex_dtype))
        self.register_buffer("r", torch.zeros((1, self.cliquesize, n_chan), dtype=float_dtype))
        self.initialized = False

    @torch.no_grad()
    def forward(self, X: torch.Tensor, Y_nn: torch.Tensor):
        """
        X: (B, nch, F) complex
        Y_nn: (B, nsrc, F) complex
        """
        B = X.shape[0]
        device = X.device
        target = Y_nn.clone().detach()
        # if Y_nn.shape[1] == 1:
        #     energy_ratio = Y_nn.abs() / (X[:, 0:1].abs() + 1e-8)
        #     min_mask = 0.03  # 最小保留比例
        #     mask = torch.where(energy_ratio > 0.5, torch.full_like(energy_ratio, min_mask), torch.ones_like(energy_ratio))
        #     interf = (X[:, 0:1] - Y_nn) * mask
        #     Y_nn = torch.cat([Y_nn, interf], dim=1) # (B, 2, F)
        if Y_nn.shape[1] == 1:
            # noise_like = (torch.randn_like(Y_nn) + 1j * torch.randn_like(Y_nn))
            noise_like = X[:, 0:1] - Y_nn 
            Y_nn = torch.cat([Y_nn, noise_like], dim=1)     #这个效果怎么又这么好
            
            # speech_mag = torch.abs(Y_nn)
            # mix_mag = torch.abs(X[:, 0:1])
            # # 计算掩模，避免在高频段过度减除
            # mask = torch.clamp(speech_mag / (mix_mag + 1e-8), 0, 1)
            # # 构造噪声先验：如果语音很弱，噪声就接近原信号；如果语音很强，噪声就被抑制
            # noise_prior_mag = mix_mag * (1 - mask) 
            # # 为了保持相位，可以给噪声先验一个随机相位或者指向混音相位
            # Y_nn_noise = noise_prior_mag * torch.exp(1j * torch.angle(X[:, 0:1]))
            # Y_nn = torch.cat([Y_nn, Y_nn_noise], dim=1)

        # (B, nch, F) -> (B, F, nch, 1)
        X = X.to(self.complex_dtype).permute(0, 2, 1).unsqueeze(-1)
        # (B, nsrc, F) -> (B, F, nsrc)
        Y_nn = Y_nn.to(self.complex_dtype).permute(0, 2, 1)

        # (B, F, nsrc, nch, nch)
        V = self.V
        U = self.U
        W = self.W
        dW = self.dW
        A = self.A
        r = self.r

        # (B, F, nsrc, 1) = (B, F, nsrc, nch) @ (B, F, nch, 1)
        Y = W @ X  # (B, F, nsrc, 1)
        # (B, F, nsrc)
        PWr = torch.abs(Y_nn) ** 2

        # (B, H, nsrc)
        C_HK = self.C_HK.to(device)
        if not self.initialized:
            r = C_HK @ PWr / torch.sum(C_HK, dim=-1, keepdim=True)
        else:
            r = self.alpha2 * r + (1 - self.alpha2) * C_HK @ PWr / torch.sum(C_HK, dim=-1, keepdim=True)
        r_floor = 0.01 * r.max(dim=1, keepdim=True).values   # -20 dB 相对地板
        r = torch.clamp(r, min=r_floor.clamp(min=self.eps))
        #r = torch.clamp(r, min=self.eps)

        # (B, F, nsrc)
        #phi = torch.sum(1 / torch.sqrt(r), dim=1, keepdim=True).repeat(1, self.bandsize, 1) 有问题
        phi = torch.einsum('hf,bhn->bfn', C_HK, 1.0/torch.sqrt(r))
        p = (1 - self.alpha) * phi

        # (B, F, nch, nch)
        CovX = torch.einsum('bfi,bfj->bfij', X.squeeze(-1), X.squeeze(-1).conj())
        # (B, F, nsrc, nch, nch)
        V = self.alpha * V + p[..., None, None] * CovX[:, :, None, :, :]
        for ch in range(self.n_chan):
            if not self.initialized:
                invWD = (1 / torch.diagonal(W @ V[:, :, ch, :, :], dim1=2, dim2=3))[..., None] * torch.eye(self.n_chan, device=device)
                invWDE = invWD[:, :, :, ch]
                U[:, :, ch, :, :] = (1 / torch.diagonal(V[:, :, ch, :, :], dim1=2, dim2=3))[..., None] * torch.eye(self.n_chan, device=device)
            else:
                Unumer = p[:, :, ch, None, None] * U[:, :, ch, :, :] @ CovX @ U[:, :, ch, :, :].permute(0,1,3,2).conj()
                Udenom = self.alpha ** 2 + (self.alpha * p[:, :, ch, None, None]) * torch.real(X[:, :, None, :, 0].conj() @ U[:, :, ch, :, :] @ (X[:, :, :, 0, None]))
                Udenom = torch.clamp(Udenom, min=1e-6)
                U[:, :, ch, :, :] = (1/self.alpha) * U[:, :, ch, :, :] - Unumer / Udenom
                U[:, :, ch, :, :] = (U[:, :, ch, :, :] + U[:, :, ch, :, :].permute(0,1,3,2).conj()) / 2
                invWDE = (U[:, :, ch, :, :] @ A[:, :, :, ch, None]).squeeze(-1)

            normCoef = torch.real(invWDE[:, :, None, :] .conj() @ V[:, :, ch, :, :] @ invWDE[:, :, :, None])[..., 0, 0]
            sqnorm = torch.clamp(torch.sqrt(normCoef), min=self.eps)
            w = invWDE / sqnorm[:, :, None]
            #w = 0.2*w.conj()+0.8*W[:, :, ch, :]
            dW[:, :, ch, :] = w.conj() - W[:, :, ch, :]
            W[:, :, ch, :] = w.conj()

            if not self.initialized:
                A = (1 / torch.diagonal(W, dim1=2, dim2=3))[..., None] * torch.eye(self.n_chan, device=device)
            else:
                Anumer = A[:, :, :, ch, None] @ dW[:, :, None, ch, :] @ A
                Adenom = torch.clamp(1 + torch.real(dW[:, :, None, ch, :] @ A[:, :, :, ch, None]), min=self.eps)
                A = A - Anumer / Adenom
        A = torch.nan_to_num(A, nan=self.eps)    #解决nan的问题

        mask = torch.isnan(W.flatten(2, 3)).any(dim=2)
        W[mask] = torch.eye(self.n_chan, dtype=self.complex_dtype, device=device)
       # Wbp = torch.diag_embed(torch.diagonal(A, dim1=2, dim2=3)) @ W
        diag_A = torch.diagonal(A, dim1=2, dim2=3)  # (B, F, nch)
        # 对W的每一行单独缩放
        Wbp = W * diag_A.unsqueeze(-1)  # broadcasting: (B,F,nch,1) * (B,F,nch,nch)
        
        Y_out = Wbp @ X  # (B, F, nsrc, 1)
        Y_out = Y_out.squeeze(-1).permute(0, 2, 1)  # (B, nsrc, F)
        ####
        # ===== Permutation alignment =====
        score0 = torch.abs(torch.sum(Y_out[:, 0, :125] * target[..., :125].conj(), dim=-1))
        score1 = torch.abs(torch.sum(Y_out[:, 1, :125] * target[..., :125].conj(), dim=-1))

        prior_energy = torch.sum(torch.abs(target[..., :]) ** 2, dim=-1)

        swap_mask = (1 < score0/score1) & (prior_energy > 1e-2)
        # if swap_mask.item():
        #     Y_out = Y_out[:, [1, 0], :]
            # print("swapped", prior_energy)
        ####

        # 更新 buffer
        self.V = V
        self.U = U
        self.W = W
        self.A = A
        self.r = r
        self.dW = dW
        self.initialized = True

        return Y_out, score0/score1, prior_energy