import numpy as np
from scipy.signal.windows import hann
from scipy.linalg import det
import librosa
import soundfile as sf
from tqdm import tqdm
import torch
import cProfile, pstats, io


def AuxIVA_numpy(x: np.ndarray, nfft, nol, 
                 n_iter=1, alpha=0.995, beta=None, Lb=1, verbose=False,
                 float_dtype=np.float64, complex_dtype=np.complex128):
    eps = np.finfo(float_dtype).eps
    if beta is None:
        beta = np.array([0.4, 0.4])
    win = hann(nfft, sym=False)
    
    nmic, nn = x.shape
    print(type(x))
    # (nch, F, T)
    X = librosa.stft(x, n_fft=nfft, hop_length=nol, window=win, center=True)
    
    # (nch, T, F)
    X = X.transpose(0, -1, -2)
    
    n_chan,n_frames,n_freq = X.shape
    
    # (nsrc, T, F)
    Y = np.zeros_like(X).astype(complex_dtype)
    # (F, nsrc, nch)
    W = np.eye(n_chan, dtype=complex_dtype)[np.newaxis].repeat(n_freq, axis=0)
    
    # (nch, nch, nsrc, F)
    V = np.zeros((n_chan, n_chan, n_chan, n_freq), dtype=complex_dtype)
    V_pre = V.copy()
    
    # (nsrc, T)
    r = np.zeros((n_chan, n_frames), dtype=float_dtype)
    phi = np.zeros((n_chan, n_frames), dtype=float_dtype)

    frame_iterator = tqdm(range(n_frames)) if verbose else range(n_frames)
    for frame in frame_iterator:
        if np.all(X[:, frame, :] == 0):
            Y[:, frame, :] = X[:, frame, :] # (nsrc, F)
        else:
            for _ in range(n_iter):
                # (nch, nsrc, F) * (nch, 1, F) & sum(0)-> (nsrc, F)
                Y[:, frame, :] = np.sum(W.transpose(2,1,0)*X[:, frame: frame+1], axis=0)         
                # (nsrc,)
                r[:, frame] = np.sum(abs(Y[:, frame, :]) ** 2, axis=-1) ** 0.5
                phi[:, frame] = r[:, frame] ** (2 * beta - 2) * (2 * beta)
                
                for k in range(n_chan):
                    if frame+1 >= Lb:
                        # ((1, Lb, 1) * (nch, Lb, F)) @ (Lb, nch, F) -> (nch, nch, F)
                        cov = np.einsum("mtf,ntf->mnf", phi[k, None, frame - Lb+1: frame+1, None] * X[:, frame - Lb+1: frame+1], X[:, frame - Lb+1: frame+1].conj()) / Lb
                    else:
                        # (scalar * (nch, F)) @ (nch, F) -> (nch, nch, F)
                        cov = np.einsum("mf,nf->mnf", phi[k, frame] * X[:, frame], X[:, frame].conj())
                    # (nch, nch, F)
                    V[:, :, k] = alpha * V_pre[:, :, k] + (1 - alpha) * cov
                
                # (F, nch, nch) @ (F, nch, nch) -> (F, nch, nch)
                H = np.linalg.inv(V[:,:,0].transpose(2, 0, 1) + np.eye(n_chan, dtype=float_dtype)[None] * 1e-8) @ V[:,:,1].transpose(2,0,1)
                # (F,)
                trH = np.trace(H, axis1=1, axis2=2)
                lamd_1 = np.real(trH + np.sqrt(trH ** 2 - 4*det(H))) / 2
                lamd_2 = np.real(trH - np.sqrt(trH ** 2 - 4*det(H))) / 2
                # (F,nch)
                e_1 = np.array([H[:, 1,1]-lamd_1, -H[:, 1,0]]).T
                e_2 = np.array([-H[:, 0,1], H[:, 0,0]-lamd_2[:]]).T
                # (F,nch)
                w1 = e_1 / np.sqrt(np.abs(np.einsum("fm,mnf,fn->f", e_1.conj(), V[:,:,0], e_1))+ 1e-10)[:,None]
                w2 = e_2 / np.sqrt(np.abs(np.einsum("fm,mnf,fn->f", e_2.conj(), V[:,:,1], e_2))+ 1e-10)[:,None]
                # (F,nch)
                W[:, 0] = w1.conj()
                W[:, 1] = w2.conj()
                # (F, nsrc, nch)
                W = np.diagonal(np.linalg.pinv(W + np.eye(n_chan, dtype=float_dtype) * eps), axis1=1, axis2=2)[..., None] * W 
                    
                V_pre = V.copy()
                # (nch, nsrc, F) * (nch, 1, F) & sum(0) -> (nsrc, F)
                Y[:, frame, :] = np.sum(W.transpose(2,1,0)*X[:, frame: frame+1], axis=0)
    
    # (nsrc, F, T)
    Y = Y.transpose(0, -1, -2)
    
    # (nsrc, wav_len)
    y = librosa.istft(Y, n_fft=nfft, hop_length=nol, window=win, center=True, length=nn)
    
    return y


@torch.no_grad()
def AuxIVA_torch(x: torch.Tensor, nfft, nol,
                 n_iter=1, alpha=0.995, beta=None, Lb=1, cuda=False, verbose=False,
                 float_dtype=torch.float64, complex_dtype=torch.complex128):
    eps = torch.finfo(float_dtype).eps
    if beta is None:
        beta = torch.tensor([0.4, 0.4])
    win = hann(nfft, sym=False)
    
    device = x.device
    
    beta = beta.to(device)
    
    nmic, nn = x.shape
    
    win = torch.from_numpy(win).to(device)
    x = x.to(device)
    # (nch, F, T)
    X = torch.stft(x, n_fft=nfft, hop_length=nol, window=win, center=True, 
                   pad_mode="constant", return_complex=True)
    
    # (nch, T, F)
    X = X.permute(0, -1, -2)
    
    n_chan,n_frames,n_freq = X.shape
    
    # (nsrc, T, F)
    Y = torch.zeros_like(X, dtype=complex_dtype, device=device)
    # (F, nsrc, nch)
    W = torch.eye(n_chan, dtype=complex_dtype, device=device).unsqueeze(0).repeat(n_freq, 1, 1)
    
    # (nch, nch, nsrc, F)
    V = torch.zeros((n_chan, n_chan, n_chan, n_freq), dtype=complex_dtype, device=device)
    V_pre = V.clone()
    
    # (nsrc, T)
    r = torch.zeros((n_chan, n_frames), dtype=float_dtype, device=device)
    phi = torch.zeros((n_chan, n_frames), dtype=float_dtype, device=device)

    frame_iterator = tqdm(range(n_frames)) if verbose else range(n_frames)
    for frame in frame_iterator:
        if torch.all(X[:, frame, :] == 0):
            Y[:, frame, :] = X[:, frame, :]  # (nch, F)
        else:
            for _ in range(n_iter):
                # (nch, nsrc, F) * (nch, 1, F) & sum(0)-> (nsrc, F)
                Y[:, frame, :] = torch.sum(W.permute(2, 1, 0) * X[:, frame: frame+1], dim=0)
                # (nsrc,)
                r[:, frame] = torch.sum(torch.abs(Y[:, frame, :]) ** 2, dim=-1) ** 0.5
                phi[:, frame] = r[:, frame] ** (2 * beta - 2) * (2 * beta)
                
                for k in range(n_chan):
                    if frame+1 >= Lb:
                        # ((1, Lb, 1) * (nch, Lb, F)) @ (Lb, nch, F) -> (nch, nch, F)
                        cov = torch.einsum("mtf,ntf->mnf", phi[k, None, frame - Lb + 1: frame + 1, None] * X[:, frame - Lb + 1: frame + 1], X[:, frame - Lb + 1: frame + 1].conj()) / Lb
                    else:
                        # (scalar * (nch, F)) @ (nch, F) -> (nch, nch, F)
                        cov = torch.einsum("mf,nf->mnf", phi[k, frame] * X[:, frame], X[:, frame].conj())
                    # (nch, nch, F)
                    V[:, :, k] = alpha * V_pre[:, :, k] + (1 - alpha) * cov
                
                # (F, nch, nch) @ (F, nch, nch) -> (F, nch, nch)
                H = torch.linalg.inv(V[:, :, 0].permute(2, 0, 1) + torch.eye(n_chan, dtype=float_dtype, device=device)[None] * 1e-8) @ V[:, :, 1].permute(2, 0, 1)
                # (F,)
                trH = torch.linalg.diagonal(H, dim1=1, dim2=2).sum(dim=-1)
                lamd_1 = torch.real(trH + torch.sqrt(trH ** 2 - 4 * torch.linalg.det(H))) / 2
                lamd_2 = torch.real(trH - torch.sqrt(trH ** 2 - 4 * torch.linalg.det(H))) / 2
                # (F,nch)
                e_1 = torch.stack([H[:, 1, 1] - lamd_1, -H[:, 1, 0]], dim=1)
                e_2 = torch.stack([-H[:, 0, 1], H[:, 0, 0] - lamd_2], dim=1)
                # (F,nch)
                w1 = e_1 / torch.sqrt(torch.abs(torch.einsum("fm,mnf,fn->f", e_1.conj(), V[:, :, 0], e_1)) + 1e-10).unsqueeze(1)
                w2 = e_2 / torch.sqrt(torch.abs(torch.einsum("fm,mnf,fn->f", e_2.conj(), V[:, :, 1], e_2)) + 1e-10).unsqueeze(1)
                # (F,nch)
                W[:, 0] = w1.conj()
                W[:, 1] = w2.conj()
                # (F, nsrc, nch)
                W = torch.linalg.diagonal(torch.linalg.pinv(W + torch.eye(n_chan, dtype=float_dtype, device=device) * eps), dim1=1, dim2=2)[..., None] * W
                    
                V_pre = V.clone()
                # (nch, nsrc, F) * (nch, 1, F) -> (nch, nsrc, F) -> (nsrc, F)
                Y[:, frame, :] = torch.sum(W.permute(2,1,0)*X[:, frame: frame+1], dim=0)
    
    # (nch, F, T)
    Y = Y.permute(0, -1, -2)
    
    # (nch, wav_len)
    y = torch.istft(Y, n_fft=nfft, hop_length=nol, window=win, center=True, length=nn)
    
    return y


@torch.no_grad()
def AuxIVA_torch_batch(x: torch.Tensor, nfft, nol,
                       n_iter=1, alpha=0.995, beta=None, Lb=1, cuda=False, verbose=False,
                       float_dtype=torch.float64, complex_dtype=torch.complex128):
    eps = torch.finfo(float_dtype).eps
    if beta is None:
        beta = torch.tensor([0.4, 0.4])
    win = hann(nfft, sym=False)

    device = x.device
    
    beta = beta.to(device)
    
    batch_size, nmic, nn = x.shape
    
    # (B, nch, L)
    x = x.to(device)
    # (B*nch, L)
    x = x.flatten(0, 1)
    # (B*nch, F, T) complex
    win = torch.from_numpy(win).to(device)
    X = torch.stft(x, n_fft=nfft, hop_length=nol, window=win, center=True, 
                   pad_mode="constant", return_complex=True)
    # (B, nch, F, T) complex
    X = X.reshape(batch_size, -1, *X.shape[1:])
        
    # (B, nch, T, F)
    X = X.permute(0, 1, -1, -2)
    
    batch_size,n_chan,n_frames,n_freq = X.shape
    
    # (B, nch, T, F)
    Y = torch.zeros_like(X, dtype=complex_dtype, device=device)
    # (B, F, nsrc, nch)
    W = torch.eye(n_chan, dtype=complex_dtype, device=device)[None, None].repeat(batch_size, n_freq, 1, 1)
    
    # (B, nch, nch, nsrc, F)
    V = torch.zeros((batch_size, n_chan, n_chan, n_chan, n_freq), dtype=complex_dtype, device=device)
    V_pre = V.clone()
    
    # (B, nch, T)
    r = torch.zeros((batch_size, n_chan, n_frames), dtype=float_dtype, device=device)
    phi = torch.zeros((batch_size, n_chan, n_frames), dtype=float_dtype, device=device)

    frame_iterator = tqdm(range(n_frames)) if verbose else range(n_frames)
    for frame in frame_iterator:
        if torch.all(X[:, :, frame, :] == 0): # to be fixed
            Y[:, :, frame, :] = X[:, :, frame, :] # (B, nch, F)
        else:        
            for _ in range(n_iter):
                # (B, nch, nsrc, F) * (B, nch, 1, F) -> (B, nch, nsrc, F) -> (B, nsrc, F)
                Y[:, :, frame, :] = torch.sum(W.permute(0,3,2,1)*X[:, :, frame: frame+1], dim=1)
                # (B, nsrc)
                r[:, :, frame] = torch.sum(torch.abs(Y[:, :, frame, :]) ** 2, dim=-1) ** 0.5
                phi[:, :, frame] = r[:, :, frame] ** (2 * beta - 2) * (2 * beta)
                
                for k in range(n_chan):
                    if frame+1 >= Lb:
                        # ((B, 1, Lb, 1) * (B, nch, Lb, F)) @ (B, Lb, nch, F) -> (B, nch, nch, F)
                        cov = torch.einsum("bmtf,bntf->bmnf", phi[:, k, None, frame - Lb + 1: frame + 1, None] * X[:, :, frame - Lb + 1: frame + 1], X[:, :, frame - Lb + 1: frame + 1].conj()) / Lb
                    else:
                        # (scalar * (B, nch, F)) @ (B, nch, F) -> (B, nch, nch, F)
                        cov = torch.einsum("bmf,bnf->bmnf", phi[:, k, frame] * X[:, :, frame], X[:, :, frame].conj())
                    # (B, nch, nch, F)
                    V[:, :, :, k] = alpha * V_pre[:, :, :, k] + (1 - alpha) * cov
                
                # (B, F, nch, nch) @ (B, F, nch, nch) -> (B, F, nch, nch)
                H = torch.linalg.inv(V[:, :, :, 0].permute(0, 3, 1, 2) + torch.eye(n_chan, dtype=float_dtype, device=device)[None, None] * 1e-8) @ V[:, :, :, 1].permute(0, 3, 1, 2)
                # (B, F)
                trH = torch.linalg.diagonal(H, dim1=2, dim2=3).sum(dim=-1)
                lamd_1 = torch.real(trH + torch.sqrt(trH ** 2 - 4 * torch.linalg.det(H))) / 2
                lamd_2 = torch.real(trH - torch.sqrt(trH ** 2 - 4 * torch.linalg.det(H))) / 2
                # (B, F, nch)
                e_1 = torch.stack([H[:, :, 1, 1] - lamd_1, -H[:, :, 1, 0]], dim=2)
                e_2 = torch.stack([-H[:, :, 0, 1], H[:, :, 0, 0] - lamd_2], dim=2)
                # (B, F, nch)
                w1 = e_1 / torch.sqrt(torch.abs(torch.einsum("bfm,bmnf,bfn->bf", e_1.conj(), V[:, :, :, 0], e_1)) + 1e-10).unsqueeze(2)
                w2 = e_2 / torch.sqrt(torch.abs(torch.einsum("bfm,bmnf,bfn->bf", e_2.conj(), V[:, :, :, 1], e_2)) + 1e-10).unsqueeze(2)
                # (B, F, nch)
                W[:, :, 0] = w1.conj()
                W[:, :, 1] = w2.conj()
                # (B, F, nsrc, nch)
                W = torch.linalg.diagonal(torch.linalg.pinv(W + torch.eye(n_chan, dtype=float_dtype, device=device) * eps), dim1=2, dim2=3)[..., None] * W
                    
                V_pre = V.clone()
                # (B, nch, nsrc, F) * (B, nch, 1, F) -> (B, nch, nsrc, F) -> (B, nsrc, F)
                Y[:, :, frame, :] = torch.sum(W.permute(0,3,2,1)*X[:, :, frame: frame+1], dim=1)
        
    # (B, nch, F, T)
    Y = Y.permute(0, 1, -1, -2)  
    
    # (B*nch, F, T)
    Y = Y.flatten(0, 1)
    
    # (B*nch, wav_len)
    y = torch.istft(Y, n_fft=nfft, hop_length=nol, window=win, center=True, length=nn)
    
    # (B, nch, wav_len)
    y = y.reshape(batch_size, -1, nn)
    
    return y


@torch.no_grad()
def AuxIVA_nn_v2_batch(X: torch.Tensor, Y_nn: torch.Tensor,
                       n_iter=1, alpha=0.995, beta=None, Lb=1, verbose=False,
                       float_dtype=torch.float64, complex_dtype=torch.complex128):
    """
    X: (B, nch, F, T) complex
    Y_nn: (B, nsrc, F, T) complex
    """
    
    eps = torch.finfo(float_dtype).eps
    if beta is None:
        beta = torch.tensor([0.4, 0.4])
    
    device = X.device
    beta = beta.to(device)
    
    # (B, nch, F, T)
    X = X.to(complex_dtype)
    # (B, nsrc, F, T)
    Y_nn = Y_nn.to(complex_dtype)   
    
    # (B, nch, F, T) -> (B, nch, T, F)
    X = X.permute(0, 1, -1, -2)
    # (B, nsrc, F, T) -> (B, nch, T, F)s
    Y_nn = Y_nn.permute(0, 1, -1, -2)
    
    batch_size,n_chan,n_frames,n_freq = X.shape
    
    # (B, nsrc, T, F)
    Y = torch.zeros_like(X, dtype=complex_dtype, device=device)
    # (B, F, nsrc, nch)
    W = torch.eye(n_chan, dtype=complex_dtype, device=device)[None, None].repeat(batch_size, n_freq, 1, 1)
    
    # (B, nch, nch, nsrc, F)
    V = torch.zeros((batch_size, n_chan, n_chan, n_chan, n_freq), dtype=complex_dtype, device=device)
    V_pre = V.clone()
    
    # (B, nsrc, T)
    r = torch.zeros((batch_size, n_chan, n_frames), dtype=float_dtype, device=device)
    phi = torch.zeros((batch_size, n_chan, n_frames), dtype=float_dtype, device=device)

    frame_iterator = tqdm(range(n_frames)) if verbose else range(n_frames)
    for frame in frame_iterator:
        # (B, nsrc, F)
        Y[:, :, frame, :] = Y_nn[:, :, frame, :]
        
        for _ in range(n_iter):
            # (B, nsrc)
            r[:, :, frame] = torch.sum(torch.abs(Y[:, :, frame, :]) ** 2, dim=-1) ** 0.5
            phi[:, :, frame] = r[:, :, frame] ** (2 * beta - 2) * (2 * beta)
            
            for k in range(n_chan):
                if frame+1 >= Lb:
                    # ((B, 1, Lb, 1) * (B, nch, Lb, F)) @ (B, Lb, nch, F) -> (B, nch, nch, F)
                    cov = torch.einsum("bmtf,bntf->bmnf", phi[:, k, None, frame - Lb + 1: frame + 1, None] * X[:, :, frame - Lb + 1: frame + 1], X[:, :, frame - Lb + 1: frame + 1].conj()) / Lb
                else:
                    # (scalar * (B, nch, F)) @ (B, nch, F) -> (B, nch, nch, F)
                    cov = torch.einsum("bmf,bnf->bmnf", phi[:, k, frame] * X[:, :, frame], X[:, :, frame].conj())
                # (B, nch, nch, F)
                V[:, :, :, k] = alpha * V_pre[:, :, :, k] + (1 - alpha) * cov
            
            # (B, F, nch, nch) @ (B, F, nch, nch) -> (B, F, nch, nch)
            H = torch.linalg.inv(V[:, :, :, 0].permute(0, 3, 1, 2) + torch.eye(n_chan, dtype=float_dtype, device=device)[None, None] * 1e-8) @ V[:, :, :, 1].permute(0, 3, 1, 2)
            # (B, F)
            trH = torch.linalg.diagonal(H, dim1=2, dim2=3).sum(dim=-1)
            lamd_1 = torch.real(trH + torch.sqrt(trH ** 2 - 4 * torch.linalg.det(H))) / 2
            lamd_2 = torch.real(trH - torch.sqrt(trH ** 2 - 4 * torch.linalg.det(H))) / 2
            # (B, F, nch)
            e_1 = torch.stack([H[:, :, 1, 1] - lamd_1, -H[:, :, 1, 0]], dim=2)
            e_2 = torch.stack([-H[:, :, 0, 1], H[:, :, 0, 0] - lamd_2], dim=2)
            # (B, F, nch)
            w1 = e_1 / torch.sqrt(torch.abs(torch.einsum("bfm,bmnf,bfn->bf", e_1.conj(), V[:, :, :, 0], e_1)) + 1e-10).unsqueeze(2)
            w2 = e_2 / torch.sqrt(torch.abs(torch.einsum("bfm,bmnf,bfn->bf", e_2.conj(), V[:, :, :, 1], e_2)) + 1e-10).unsqueeze(2)
            W[:, :, 0] = w1.conj()
            W[:, :, 1] = w2.conj()
            # (B, F, nsrc, nch)
            W = torch.linalg.diagonal(torch.linalg.pinv(W + torch.eye(n_chan, dtype=float_dtype, device=device) * eps), dim1=2, dim2=3)[..., None] * W
                
            V_pre = V
            # (B, nch, nsrc, F) * (B, nch, 1, F) -> (B, nch, nsrc, F) -> (B, nsrc, F)
            Y[:, :, frame, :] = torch.sum(W.permute(0,3,2,1)*X[:, :, frame: frame+1], dim=1)
        
    # (B, nsrc, F, T)
    Y = Y.permute(0, 1, -1, -2)  
    
    # (B, nsrc, F, T)
    Y = Y.to(torch.complex64)
    
    return Y


@torch.no_grad()
def AuxIVA_nn_enh_batch(X: torch.Tensor, Y_nn: torch.Tensor,
                       n_iter=1, alpha=0.995, beta=None, Lb=1, verbose=False,
                       float_dtype=torch.float64, complex_dtype=torch.complex128):
    """
    X: (B, nch, F, T) complex
    Y_nn: (B, 1, F, T) complex
    """
    # pr = cProfile.Profile()
    # pr.enable()

    eps = torch.finfo(float_dtype).eps
    #eps = 1e-3
    if beta is None:
        beta = torch.tensor([0.4, 0.4])
    
    device = X.device
    beta = beta.to(device)

    # 仅输入目标信号通道时拟制一个噪声通道
    # if Y_nn.shape[1] == 1:
    #     energy_ratio = Y_nn.abs() / (X[:, 0:1].abs() + 1e-8)
    #     min_mask = 0.03  # 最小保留比例
    #     mask = torch.where(energy_ratio > 0.5, torch.full_like(energy_ratio, min_mask), torch.ones_like(energy_ratio))
    #     interf = (X[:, 0:1] - Y_nn) * mask
    #     Y_nn = torch.cat([interf, Y_nn], dim=1) # (B, 2, F, T)
    if Y_nn.shape[1] == 1:
        # noise_like = (torch.randn_like(Y_nn) + 1j * torch.randn_like(Y_nn))
        noise_like = X[:, 0:1] - Y_nn 
        Y_nn = torch.cat([Y_nn, noise_like], dim=1)     #这个效果怎么又这么好
    
    # (B, nch, F, T)
    X = X.to(complex_dtype)
    # (B, nsrc, F, T)
    Y_nn = Y_nn.to(complex_dtype)   
    
    # (B, nch, F, T) -> (B, nch, T, F)
    X = X.permute(0, 1, -1, -2)
    # (B, nsrc, F, T) -> (B, nch, T, F)
    Y_nn = Y_nn.permute(0, 1, -1, -2)
    
    batch_size,n_chan,n_frames,n_freq = X.shape
    
    # (B, nsrc, T, F)
    Y = torch.zeros_like(X, dtype=complex_dtype, device=device)
    # (B, F, nsrc, nch)
    W = torch.eye(n_chan, dtype=complex_dtype, device=device)[None, None].repeat(batch_size, n_freq, 1, 1)
    
    # (B, nch, nch, nsrc, F)
    V = torch.zeros((batch_size, n_chan, n_chan, n_chan, n_freq), dtype=complex_dtype, device=device)
    V_pre = V.clone()
    
    # (B, nsrc, T)
    r = torch.zeros((batch_size, n_chan, n_frames), dtype=float_dtype, device=device)
    phi = torch.zeros((batch_size, n_chan, n_frames), dtype=float_dtype, device=device)

    frame_iterator = tqdm(range(n_frames)) if verbose else range(n_frames)
    for frame in frame_iterator:
        # (B, nsrc, F)
        Y[:, :, frame, :] = Y_nn[:, :, frame, :]
        
        for _ in range(n_iter):
            # (B, nsrc)
            r[:, :, frame] = torch.sum(torch.abs(Y[:, :, frame, :]) ** 2, dim=-1) ** 0.5
            phi[:, :, frame] = r[:, :, frame] ** (2 * beta - 2) * (2 * beta)
            
            for k in range(n_chan):
                if frame+1 >= Lb:
                    # ((B, 1, Lb, 1) * (B, nch, Lb, F)) @ (B, Lb, nch, F) -> (B, nch, nch, F)
                    cov = torch.einsum("bmtf,bntf->bmnf", phi[:, k, None, frame - Lb + 1: frame + 1, None] * X[:, :, frame - Lb + 1: frame + 1], X[:, :, frame - Lb + 1: frame + 1].conj()) / Lb
                else:
                    # (scalar * (B, nch, F)) @ (B, nch, F) -> (B, nch, nch, F)
                    cov = torch.einsum("bmf,bnf->bmnf", phi[:, k, frame].view(-1,1,1) * X[:, :, frame], X[:, :, frame].conj())
                # (B, nch, nch, F)
                V[:, :, :, k] = alpha * V_pre[:, :, :, k] + (1 - alpha) * cov
            
            # (B, F, nch, nch) @ (B, F, nch, nch) -> (B, F, nch, nch)
            H = torch.linalg.inv(V[:, :, :, 0].permute(0, 3, 1, 2) + torch.eye(n_chan, dtype=float_dtype, device=device)[None, None] * 1e-8) @ V[:, :, :, 1].permute(0, 3, 1, 2)
            # (B, F)
            trH = torch.linalg.diagonal(H, dim1=2, dim2=3).sum(dim=-1)
            lamd_1 = torch.real(trH + torch.sqrt(trH ** 2 - 4 * torch.linalg.det(H))) / 2
            lamd_2 = torch.real(trH - torch.sqrt(trH ** 2 - 4 * torch.linalg.det(H))) / 2
            # (B, F, nch)
            e_1 = torch.stack([H[:, :, 1, 1] - lamd_1, -H[:, :, 1, 0]], dim=2)
            e_2 = torch.stack([-H[:, :, 0, 1], H[:, :, 0, 0] - lamd_2], dim=2)
            # (B, F, nch)
            w1 = e_1 / torch.sqrt(torch.abs(torch.einsum("bfm,bmnf,bfn->bf", e_1.conj(), V[:, :, :, 0], e_1)) + 1e-10).unsqueeze(2)
            w2 = e_2 / torch.sqrt(torch.abs(torch.einsum("bfm,bmnf,bfn->bf", e_2.conj(), V[:, :, :, 1], e_2)) + 1e-10).unsqueeze(2)
            W[:, :, 0] = w1.conj()
            W[:, :, 1] = w2.conj()
            # (B, F, nsrc, nch)
            #W = torch.linalg.diagonal(torch.linalg.pinv(W + torch.eye(n_chan, dtype=float_dtype, device=device) * eps), dim1=2, dim2=3)[..., None] * W
                
            V_pre = V
            # (B, nch, nsrc, F) * (B, nch, 1, F) -> (B, nch, nsrc, F) -> (B, nsrc, F)
            Y[:, :, frame, :] = torch.sum(W.permute(0,3,2,1)*X[:, :, frame: frame+1], dim=1)
        
    def projection_back(reference, Y):
        """
        reference: (B, F, T) complex
        Y: (B, nsrc, F, T) complex
        returns: (B, nsrc, F, T) complex
        """
        # Reference microphone signal: (B, F, T)
        X_ref = reference[:, 0]  # (B, F, T)
        # Compute numerator: <X_ref, Y>
        # (B, C, F)
        numerator = torch.sum(X_ref[:, None, :, :] * torch.conj(Y), dim=-1)
        denominator = torch.sum(torch.abs(Y) ** 2, dim=-1) + eps
        # Scaling factor z: (B, C, F)
        z = numerator / denominator
        # Apply projection back
        Y_pb = z[..., None] * Y  # broadcast over T
        return Y_pb
    # (B, nsrc, F, T)
    Y = projection_back(X.permute(0,1,-1,-2), Y.permute(0, 1, -1, -2))
    Y = Y.to(torch.complex64)
    
    return Y

@torch.no_grad()
def AuxIVA_nn_stream(X: torch.Tensor, Y_nn: torch.Tensor,
                       n_iter=1, alpha=0.995, beta=None, Lb=1, verbose=False,
                       float_dtype=torch.float64, complex_dtype=torch.complex128):
    """
    X: (B, nch, F, T) complex
    Y_nn: (B, 1, F, T) complex, T=1
    returns: (B, 1, F, T) complex
    """

    eps = torch.finfo(float_dtype).eps
    #eps = 1e-3
    if beta is None:
        beta = torch.tensor([0.4, 0.4])
    
    device = X.device
    beta = beta.to(device)
    if Y_nn.shape[1] == 1:
        energy_ratio = Y_nn.abs() / (X[:, 0:1].abs() + 1e-8)
        min_mask = 0.03  # 最小保留比例
        mask = torch.where(energy_ratio > 0.5, torch.full_like(energy_ratio, min_mask), torch.ones_like(energy_ratio))
        interf = (X[:, 0:1] - Y_nn) * mask
        Y_nn = torch.cat([interf, Y_nn], dim=1) # (B, 2, F)

    # 仅输入目标信号通道时拟制一个噪声通道
    '''    if Y_nn.shape[1] == 1:
        # noise_like = (torch.randn_like(Y_nn) + 1j * torch.randn_like(Y_nn))
        noise_like = X[:, 0:1] - Y_nn 
        Y_nn = torch.cat([Y_nn, noise_like], dim=1)  # (B, 2, F, T)'''
        

    # (B, nch, F, T)
    X = X.to(complex_dtype)
    # (B, nsrc, F, T)
    Y_nn = Y_nn.to(complex_dtype)   
    
    # (B, nch, F, T) -> (B, nch, T, F)
    X = X.permute(0, 1, -1, -2)
    # (B, nsrc, F, T) -> (B, nch, T, F)
    Y_nn = Y_nn.permute(0, 1, -1, -2)
    #print(Y_nn.mean(dim=[0,2,3]))
    
    batch_size,n_chan,n_frames,n_freq = X.shape
    
    # (B, nsrc, T, F)
    Y = torch.zeros_like(X, dtype=complex_dtype, device=device)
    # (B, F, nsrc, nch)
    W = torch.eye(n_chan, dtype=complex_dtype, device=device)[None, None].repeat(batch_size, n_freq, 1, 1)
    
    # (B, nch, nch, nsrc, F)
    V = torch.zeros((batch_size, n_chan, n_chan, n_chan, n_freq), dtype=complex_dtype, device=device)
    V_pre = V.clone()
    
    # (B, nsrc, T)
    r = torch.zeros((batch_size, n_chan, n_frames), dtype=float_dtype, device=device)
    phi = torch.zeros((batch_size, n_chan, n_frames), dtype=float_dtype, device=device)

    frame_iterator = tqdm(range(n_frames)) if verbose else range(n_frames)
    for frame in frame_iterator:
        # (B, nsrc, F)
        Y[:, :, frame, :] = Y_nn[:, :, frame, :]
        
        for _ in range(n_iter):
            # (B, nsrc)
            r[:, :, frame] = torch.sum(torch.abs(Y[:, :, frame, :]) ** 2, dim=-1) ** 0.5
            phi[:, :, frame] = r[:, :, frame] ** (2 * beta - 2) * (2 * beta)    # Contrast function
            
            # Streaming IVA
            for k in range(n_chan):
                if frame+1 >= Lb:
                    # ((B, 1, Lb, 1) * (B, nch, Lb, F)) @ (B, Lb, nch, F) -> (B, nch, nch, F)
                    cov = torch.einsum("bmtf,bntf->bmnf", phi[:, k, None, frame - Lb + 1: frame + 1, None] * X[:, :, frame - Lb + 1: frame + 1], X[:, :, frame - Lb + 1: frame + 1].conj()) / Lb
                else:
                    # (scalar * (B, nch, F)) @ (B, nch, F) -> (B, nch, nch, F)
                    cov = torch.einsum("bmf,bnf->bmnf", phi[:, k, frame] * X[:, :, frame], X[:, :, frame].conj())
                print(cov.shape, "cov")
                # (B, nch, nch, F)
                V[:, :, :, k] = alpha * V_pre[:, :, :, k] + (1 - alpha) * cov
                # B, 2, 2, ch, F

            # (B, F, nch, nch) @ (B, F, nch, nch) -> (B, F, nch, nch)
            print(torch.linalg.det(V[:, :, :, 0].permute(0, 3, 1, 2)))
            print(torch.linalg.det(V[:, :, :, 1].permute(0, 3, 1, 2)))
            H = torch.linalg.inv(V[:, :, :, 0].permute(0, 3, 1, 2) + torch.eye(n_chan, dtype=float_dtype, device=device)[None, None] * 1e-8) @ V[:, :, :, 1].permute(0, 3, 1, 2)    # 广义特征值问题的矩阵，据此求特征值
            # (B, F)
            trH = torch.linalg.diagonal(H, dim1=2, dim2=3).sum(dim=-1)
            lamd_1 = torch.real(trH + torch.sqrt(trH ** 2 - 4 * torch.linalg.det(H))) / 2
            lamd_2 = torch.real(trH - torch.sqrt(trH ** 2 - 4 * torch.linalg.det(H))) / 2
            # (B, F, nch) solution for eigen vector
            e_1 = torch.stack([H[:, :, 1, 1] - lamd_1, -H[:, :, 1, 0]], dim=2)
            e_2 = torch.stack([-H[:, :, 0, 1], H[:, :, 0, 0] - lamd_2], dim=2)
            # (B, F, nch)
            w1 = e_1 / torch.sqrt(torch.abs(torch.einsum("bfm,bmnf,bfn->bf", e_1.conj(), V[:, :, :, 0], e_1)) + 1e-10).unsqueeze(2)
            w2 = e_2 / torch.sqrt(torch.abs(torch.einsum("bfm,bmnf,bfn->bf", e_2.conj(), V[:, :, :, 1], e_2)) + 1e-10).unsqueeze(2)
            W[:, :, 0] = w1.conj()
            W[:, :, 1] = w2.conj()

            # print(torch.linalg.det(W[0,128,:,:]))

            # (B, F, nsrc, nch)
            W = torch.linalg.diagonal(torch.linalg.pinv(W + torch.eye(n_chan, dtype=float_dtype, device=device) * eps), dim1=2, dim2=3)[..., None] * W
            # W[:, :, 0] /= torch.linalg.norm(W[:, :, 0], dim=-1, keepdim=True)    
            # print(torch.linalg.det(W[0,128,:,:]))
            V_pre = V   # 做到外面的cache里
            # (B, nch, nsrc, F) * (B, nch, 1, F) -> (B, nch, nsrc, F) -> (B, nsrc, F)
            Y[:, :, frame, :] = torch.sum(W.permute(0,3,2,1)*X[:, :, frame: frame+1], dim=1)
            # 每一帧
            # for k in range(n_chan):
            #     Y_energy = torch.mean(torch.abs(Y[:, k, frame])**2, dim=-1) # value
            #     X_energy = torch.mean(torch.abs(X[:, k, frame])**2, dim=-1)
            #     #print(X_energy/Y_energy)
            #     Y[:, k, frame] *= torch.sqrt(X_energy / (Y_energy))

    # (B, nsrc, F, T)
    Y = Y.permute(0, 1, -1, -2) 
    #print("Y mean:", Y.mean(dim=[0,2,3]))
    
    # (B, nsrc, F, T)
    Y = Y.to(torch.complex64)
    
    
    return Y

# @torch.no_grad()
# def AuxIVA_stream_single(X: torch.Tensor, Y_nn: torch.Tensor, V_pre: torch.Tensor,
#                        n_iter=1, alpha=0.995, beta=None, Lb=1, verbose=False,
#                        float_dtype=torch.float64, complex_dtype=torch.complex128):
#     """
#     X: (B, nch, F, T) complex
#     Y_nn: (B, 1, F, T) complex, T=1
#     returns: (B, 1, F, T) complex
#     """

#     eps = torch.finfo(float_dtype).eps
#     #eps = 1e-3
#     if beta is None:
#         beta = torch.tensor([0.4, 0.4])
    
#     device = X.device
#     beta = beta.to(device)

#     # (B, nch, F, T)
#     X = X.to(complex_dtype)
#     # (B, 1, F, T)
#     Y_nn = Y_nn.to(complex_dtype)   
    
#     # (B, nch, F, T) -> (B, nch, T, F)
#     X = X.permute(0, 1, -1, -2)
#     # (B, nsrc, F, T) -> (B, nch, T, F)
#     Y_nn = Y_nn.permute(0, 1, -1, -2)
#     #print(Y_nn.mean(dim=[0,2,3]))
    
#     batch_size,n_chan,n_frames,n_freq = X.shape
    
#     # (B, nsrc, T, F)
#     Y = torch.zeros_like(X, dtype=complex_dtype, device=device)
#     # (B, F, nsrc, nch)
#     W = torch.eye(n_chan, dtype=complex_dtype, device=device)[None, None].repeat(batch_size, n_freq, 1, 1)
    
#     # (B, nch, nch, nsrc, F)
#     V = torch.zeros((batch_size, n_chan, n_chan, n_chan, n_freq), dtype=complex_dtype, device=device)
#     V_pre = torch.view_as_complex(V_pre.to(device))
    
#     # (B, nsrc, T)
#     r = torch.zeros((batch_size, n_chan, n_frames), dtype=float_dtype, device=device)
#     phi = torch.zeros((batch_size, n_chan, n_frames), dtype=float_dtype, device=device)

#     frame_iterator = tqdm(range(n_frames)) if verbose else range(n_frames)
#     frame = 0
#     # (B, nsrc, F)
#     Y[:, :, frame, :] = Y_nn[:, :, frame, :]
#     #Y[:, :, frame, :] = torch.sum(W.permute(0,3,2,1)*X[:, :, frame: frame+1], dim=1)
#     #Y[:, :, frame, :] = X[:, :, frame, :]  # 初始化为输入信号

#     for _ in range(n_iter):
#         # (B, nsrc)
#         r[:, :, frame] = torch.sum(torch.abs(Y[:, :, frame, :]) ** 2, dim=-1) ** 0.5
#         phi[:, :, frame] = r[:, :, frame] ** (2 * beta - 2) * (2 * beta)
        
#         for k in range(n_chan):
#             if frame+1 >= Lb:
#                 # ((B, 1, Lb, 1) * (B, nch, Lb, F)) @ (B, Lb, nch, F) -> (B, nch, nch, F)
#                 cov = torch.einsum("bmtf,bntf->bmnf", phi[:, k, None, frame - Lb + 1: frame + 1, None] * X[:, :, frame - Lb + 1: frame + 1], X[:, :, frame - Lb + 1: frame + 1].conj()) / Lb
#             else:
#                 # (scalar * (B, nch, F)) @ (B, nch, F) -> (B, nch, nch, F)
#                 cov = torch.einsum("bmf,bnf->bmnf", phi[:, k, frame] * X[:, :, frame], X[:, :, frame].conj())
#             # (B, nch, nch, F)
#             V[:, :, :, k] = alpha * V_pre[:, :, :, k] + (1 - alpha) * cov
        
#         # (B, F, nch, nch) @ (B, F, nch, nch) -> (B, F, nch, nch)
#         H = torch.linalg.inv(V[:, :, :, 0].permute(0, 3, 1, 2) + torch.eye(n_chan, dtype=float_dtype, device=device)[None, None] * 1e-8) @ V[:, :, :, 1].permute(0, 3, 1, 2)
#         # (B, F)
#         trH = torch.linalg.diagonal(H, dim1=2, dim2=3).sum(dim=-1)
#         lamd_1 = torch.real(trH + torch.sqrt(trH ** 2 - 4 * torch.linalg.det(H))) / 2
#         lamd_2 = torch.real(trH - torch.sqrt(trH ** 2 - 4 * torch.linalg.det(H))) / 2
#         # (B, F, nch)
#         e_1 = torch.stack([H[:, :, 1, 1] - lamd_1, -H[:, :, 1, 0]], dim=2)
#         e_2 = torch.stack([-H[:, :, 0, 1], H[:, :, 0, 0] - lamd_2], dim=2)
#         # (B, F, nch)
#         w1 = e_1 / torch.sqrt(torch.abs(torch.einsum("bfm,bmnf,bfn->bf", e_1.conj(), V[:, :, :, 0], e_1)) + 1e-10).unsqueeze(2)
#         w2 = e_2 / torch.sqrt(torch.abs(torch.einsum("bfm,bmnf,bfn->bf", e_2.conj(), V[:, :, :, 1], e_2)) + 1e-10).unsqueeze(2)
#         W[:, :, 0] = w1.conj()
#         W[:, :, 1] = w2.conj()

#         # (B, F, nsrc, nch)
#         W = torch.linalg.diagonal(torch.linalg.pinv(W + torch.eye(n_chan, dtype=float_dtype, device=device) * eps), dim1=2, dim2=3)[..., None] * W
            
#         V_pre = V
#         # (B, nch, nsrc, F) * (B, nch, 1, F) -> (B, nch, nsrc, F) -> (B, nsrc, F)
#         Y[:, :, frame, :] = torch.sum(W.permute(0,3,2,1)*X[:, :, frame: frame+1], dim=1)
#         # 每一帧
#         # for k in range(n_chan):
#         #     Y_energy = torch.mean(torch.abs(Y[:, k, frame])**2, dim=-1) # value
#         #     X_energy = torch.mean(torch.abs(X[:, k, frame])**2, dim=-1)
#         #     # print(X_energy/Y_energy)
#         #     Y[:, k, frame] *= torch.sqrt(X_energy / (Y_energy))

#     # (B, nsrc, F, T)
#     Y = Y.permute(0, 1, -1, -2) 
#     #print("Y mean:", Y.mean(dim=[0,2,3]))
    
#     # (B, nsrc, F, T)
#     Y = Y.to(torch.complex64)
    
#     return Y, torch.view_as_real(V_pre)



@torch.no_grad()
def AuxIVA_stream_single(X_curr, Y_prior, V_prev, X_buffer, phi_buffer,
                         n_iter=1, alpha=0.995, beta=None, Lb=1,
                         float_dtype=torch.float64, complex_dtype=torch.complex128):
    """    
    参数:
    - X_curr: 当前帧输入频谱 (B, nch, F)
    - Y_prior: 当前帧的先验源估计 (B, nsrc, F)，用于初始化权重计算
    - V_prev: 缓存的平滑协方差矩阵 (B, nch, nch, nsrc, F)，实数表示
    - X_buffer: 输入频谱缓存 (B, nch, Lb, F)
    - phi_buffer: 权重因子缓存 (B, nsrc, Lb)
    - n_iter: 迭代次数
    - alpha: 指数平滑系数
    - beta: 源分部先验参数，None 则默认为 [0.4, 0.4]
    - Lb: 协方差估计的窗口长度
    """
    def inv(A, eps=1e-8):
            det = A[...,0,0]*A[...,1,1] - A[...,0,1]*A[...,1,0]
            det = det + eps * (det.abs() < eps)
            return torch.stack([torch.stack([A[...,1,1]/det, -A[...,0,1]/det], -1),
                        torch.stack([-A[...,1,0]/det, A[...,0,0]/det], -1)], -2)
    
    eps = torch.finfo(float_dtype).eps
    if beta is None:
        beta = torch.tensor([0.4, 0.4], device=X_curr.device, dtype=float_dtype)
    
    device = X_curr.device
    batch_size, n_chan, n_freq = X_curr.shape
    n_src = n_chan # 通常 n_src = n_chan = 2
    target = Y_prior.clone().detach()  # (B, nsrc, F) 

    if Y_prior.shape[1] == 1: 
        noise_like = X_curr[:, 0:1] - Y_prior
        Y_prior = torch.cat([Y_prior, noise_like], dim=1)     #这个效果怎么又这么好
        # energy_ratio = Y_prior.abs() / (X_curr[:, 0:1].abs() + 1e-8)
        # min_mask = 0.03  # 最小保留比例
        # mask = torch.where(energy_ratio > 0.5, torch.full_like(energy_ratio, min_mask), torch.ones_like(energy_ratio))
        # interf = (X_curr[:, 0:1] - Y_prior) * mask
        # Y_prior = torch.cat([interf, Y_prior], dim=1) # (B, 2, F, T)

    # 类型转换
    X_curr = X_curr.to(complex_dtype)
    #Y_curr = torch.zeros_like(X_curr, dtype=complex_dtype, device=device)
    V_prev_c = torch.view_as_complex(V_prev.to(device)) # 转换为复数进行计算
    X_buffer = torch.view_as_complex(X_buffer.to(device))

    # 1. 更新 X 缓冲区 (滑动窗口)
    # 将当前帧 X_curr 压入缓存，弹出最老的一帧
    # X_buffer: (B, nch, Lb, F), 
    X_buffer = torch.cat([X_buffer[:, :, 1:], X_curr.unsqueeze(2)], dim=2)

    # 初始化当前帧输出和解混矩阵 (IVA 每一帧通常从单位阵开始或使用上一帧 W)
    W = torch.eye(n_chan, dtype=complex_dtype, device=device)[None, None].repeat(batch_size, n_freq, 1, 1)

    Y_curr = Y_prior.to(complex_dtype)

    for _ in range(n_iter):
        # 2. 计算当前帧的权重 phi
        # r: (B, nsrc) - 跨频率的向量范数
        r_curr = torch.sum(torch.abs(Y_curr) ** 2, dim=-1).sqrt().to(float_dtype)
        # phi_curr: (B, nsrc)
        phi_curr = (r_curr ** (2 * beta - 2)) * (2 * beta)
        
        # 3. 更新 phi 缓冲区 (用于协方差窗口计算)
        temp_phi_buffer = torch.cat([phi_buffer[:, :, 1:], phi_curr.unsqueeze(2)], dim=2)
        
        # 4. 更新协方差矩阵 V
        V_now = torch.zeros_like(V_prev_c)
        for k in range(n_src):
            # 提取第 k 个源在窗口内的权重: (B, 1, Lb, 1)
            phi_k_window = temp_phi_buffer[:, k, None, :, None]
            
            # 计算窗口内的加权协方差: (B, nch, nch, F)
            # 根据公式: V = (1/Lb) * sum( phi * x * x^H )
            cov_window = torch.einsum("bitf,bjtf->bijf", phi_k_window * X_buffer, X_buffer.conj()) / Lb
            
            # 时间平滑更新
            V_now[:, :, :, k, :] = alpha * V_prev_c[:, :, :, k, :] + (1 - alpha) * cov_window
        
        # 5. GEVD 优化目标 (针对 2-channel)
        # 构造 H = V0^-1 * V1
        V0 = V_now[:, :, :, 0, :].permute(0, 3, 1, 2) # (B, F, nch, nch)
        V1 = V_now[:, :, :, 1, :].permute(0, 3, 1, 2)
        
        # 加上微小扰动防止奇异
        eye = torch.eye(n_chan, device=device, dtype=V0.dtype)[None, None]
        V0 = V0 + eye * 1e-9
        # for f in range(V0.shape[0]):
        #     diag0 = V0[0, f, 0, 0].item()
        #     diag1 = V0[0, f, 1, 1].item()
            
        #     if abs(diag0) < 1e-8 or abs(diag1) < 1e-8:
        #         print(f"频率点 {f}: diag0={diag0:.2e}, diag1={diag1:.2e} ⚠️")
        
        #H = torch.linalg.inv(V0) @ V1
        H = inv(V0, 1e-9) @ V1
        
        # 求解 2x2 矩阵的特征值和特征向量 (Closed-form GEVD)
        trH = torch.linalg.diagonal(H, dim1=2, dim2=3).sum(dim=-1)
        detH = torch.linalg.det(H)
        
        # 特征值
        sqrt_disc = torch.sqrt(trH ** 2 - 4 * detH + 1e-12)
        lamd_1 = (trH + sqrt_disc) / 2
        lamd_2 = (trH - sqrt_disc) / 2
        
        # 特征向量 (未归一化)
        e_1 = torch.stack([H[:, :, 0, 1], lamd_1 - H[:, :, 0, 0]], dim=2)
        e_2 = torch.stack([H[:, :, 0, 1], lamd_2 - H[:, :, 0, 0]], dim=2)
        
        # 归一化解混向量 w_m: 使 w^H * V_m * w = 1
        def normalize_w(e, V_m):
            # e: (B, F, nch), V_m: (B, nch, nch, F)
            denom = torch.einsum("bfi,bijf,bfj->bf", e.conj(), V_m, e).abs().sqrt() + 1e-12
            return e / denom.unsqueeze(2)

        def inv(A, eps=1e-8):
            det = A[...,0,0]*A[...,1,1] - A[...,0,1]*A[...,1,0]
            det = det + eps * (det.abs() < eps)
            return torch.stack([torch.stack([A[...,1,1]/det, -A[...,0,1]/det], -1),
                        torch.stack([-A[...,1,0]/det, A[...,0,0]/det], -1)], -2)
            
        w1 = normalize_w(e_1, V_now[:, :, :, 0, :])
        w2 = normalize_w(e_2, V_now[:, :, :, 1, :])
        
        W[:, :, 0, :] = w1.conj()
        W[:, :, 1, :] = w2.conj()

        # 6. Back-projection (投影回参考麦克风，消除幅度歧义)
        # W = diag(inv(W)) * W
        W_inv_diag = torch.linalg.diagonal(torch.linalg.pinv(W + eye * eps), dim1=2, dim2=3)
        W = W_inv_diag[..., None] * W
        # 7. 更新当前帧的输出 Y_curr 用于下一次迭代 (如果有的话)
        # Y = W * X -> (B, F, nsrc, nch) @ (B, F, nch, 1)
        Y_curr = torch.einsum("bfnm,bmf->bnf", W, X_curr)
        
        ####
        # ===== Permutation alignment =====
        # score0 = torch.abs(torch.sum(Y_curr[:, 0, :] * target.conj(), dim=-1))
        # score1 = torch.abs(torch.sum(Y_curr[:, 1, :] * target.conj(), dim=-1))

        # prior_energy = torch.sum(torch.abs(target) ** 2, dim=-1)

        # swap_mask = (score1 > score0) & (prior_energy > 1e-6)
        # if swap_mask.item():
        #     Y_curr = Y_curr[:, [1, 0], :]
        ####
        
        # 更新中间变量
        V_prev_c = V_now
        phi_buffer = temp_phi_buffer
    # 转换回实数格式存储 V
    V_out_real = torch.view_as_real(V_prev_c)
    X_buffer = torch.view_as_real(X_buffer)
    
    return Y_curr, V_out_real, X_buffer, phi_buffer
