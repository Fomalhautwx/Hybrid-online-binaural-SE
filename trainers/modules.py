import numpy as np
import soundfile as sf
import librosa
from scipy.signal.windows import hann
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as torch
import numpy as np
import math


EPS = 1e-10

def win_hanning_period(size):
    a = 2 * np.pi / size
    f = np.arange(size)
    window = 0.5 - 0.5 * np.cos(a * f)
    return window
    

def multich_stft(x, n_fft=512, hop_length=256, win_length=512):
    '''b (c) t-> b c f t complex'''
    if len(x.shape) != 3:
        x=x.unsqueeze(1)
    batch, channels, samples = x.shape
    device = x.device
    
    # 重塑为 (batch*channels, samples) 以便批量处理
    x_reshaped = x.reshape(-1, samples)
    
    # 批量计算STFT
    stft = torch.stft(x_reshaped, 
                     n_fft=n_fft, 
                     hop_length=hop_length, 
                     win_length=win_length, 
                     window=torch.hann_window(win_length).pow(0.5).to(device), 
                     return_complex=True)
    
    # 重塑回 (batch, channels, freq, time)
    stft = stft.reshape(batch, channels, *stft.shape[-2:])
    return stft  # (B,2C,T,F)
    
class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft//2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs-erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs-erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        erb_f = 21.4*np.log10(0.00437*freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10**(erb_f/21.4) - 1)/0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1/nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                                / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2-2):
            erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12)\
                                                    / (bins[i+1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i + 2])  + 1e-12) \
                                                    / (bins[i + 2] - bins[i+1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1]+1] = 1- erb_filters[-2, bins[-2]:bins[-1]+1]
        
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))
    
    def bm(self, x):
        """x: (B,C,T,F)"""
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)
    
    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)
    

class STFT(nn.Module):
    def __init__(self, 
                 frame_len,
                 frame_hop):
        super().__init__()        
        self._fralen = frame_len
        self._fralen_eff = int(frame_len / 2 + 1)
        self.stride = frame_hop
                
        # window
        W = np.sqrt(win_hanning_period(frame_len))
        
        # (2F_eff, N)
        # [0: F_eff] for real part, [F_eff:] for imag part
        stft_coeff = self.get_stft_coeff(frame_len, W)
        # K: (2F_eff, 1, N)
        self.register_buffer("K", torch.tensor(stft_coeff[:, np.newaxis, :], dtype=torch.float32))

    def forward(self, x):
        """
        Accept single channel raw waveform and output real and imag
        args
            x: input signal, N x S or N x C x S
        return
            r: real, N x F x T
            i: imag, N x F x T
        """
        if x.ndim == 2:
            # N x S, reshape N x 1 x S
            x = torch.unsqueeze(x, 1)
            # N x 2F_eff x Tf
            c = F.conv1d(input=x, weight=self.K, stride=self.stride, padding=0)
            # N x F_eff x T
            r, i = torch.chunk(c, 2, dim=1)
        else:
            N, C = x.shape[:2]
            # (N*C, 1, S)
            x = x.flatten(0, 1).unsqueeze(1)
            # (N*C, 2F_eff, Tf)
            c = F.conv1d(input=x, weight=self.K, stride=self.stride, padding=0)
            # (N, C, 2F_eff, Tf)
            c = c.reshape(N, C, *c.shape[1:])
            # (N, C, F_eff, Tf)
            r, i = torch.chunk(c, 2, dim=2)
        
        return r, i

    @staticmethod
    def get_dft_coeff(frame_len):
        N = frame_len
        idx = np.arange(0, N)
        idx_nk = 2.0 * math.pi * idx[:, np.newaxis] * idx[np.newaxis, :] / N
        dft_real_coeff = np.cos(idx_nk)
        dft_img_coeff = -np.sin(idx_nk)
        return dft_real_coeff, dft_img_coeff

    def get_stft_coeff(self, frame_len, win):
        assert len(win) == frame_len
        # (N, N)
        dft_real_coeff, dft_img_coeff = self.get_dft_coeff(frame_len)
        stft_real_coeff = dft_real_coeff * win[np.newaxis, :]
        stft_img_coeff = dft_img_coeff * win[np.newaxis, :]

        return np.concatenate(
            (
                stft_real_coeff[:self._fralen_eff, :], # (F, N)
                stft_img_coeff[:self._fralen_eff, :], # (F, N)
            ),
            axis=0,
        ) # (2F_eff, N)


class ISTFT(nn.Module):
    def __init__(self, 
                 frame_len, 
                 frame_hop):
        super().__init__()        
        self._fralen = frame_len
        self._fralen_eff = int(frame_len / 2 + 1)
        self.stride = frame_hop
        
        # window
        W = np.sqrt(win_hanning_period(frame_len))
        # (2*frame_len, frame_len)
        istft_coeff = self.get_istft_coeff(frame_len, W).T
        # (2*frame_len, 1, frame_len)
        self.register_buffer("K", torch.tensor(istft_coeff[:, np.newaxis, :], dtype=torch.float32))
        # (F, 1, F)
        # self.register_buffer("gain_coeff", torch.tensor(np.diag(win * win)[:, np.newaxis, :]))

    def forward(self, x_real, x_img, length=None):
        """
        Accept phase & magnitude and output raw waveform
        args
            x_real, x_imag: N x F x T or N x C x F x T
        return
            s: N x S
        """
        if x_real.ndim == 3:
            # x_real_padding/x_img_padding: (N, F_eff-2, Tf)
            x_real_padding = torch.flip(x_real[:, 1:-1, :], dims=[1])
            x_img_padding = torch.flip(-x_img[:, 1:-1, :], dims=[1])
            # [N, 2F, Tf]
            x = torch.cat((x_real, x_real_padding, x_img, x_img_padding), dim=1)
            # [N, 1, Tt]
            s = F.conv_transpose1d(input=x, weight=self.K, stride=self.stride, padding=0)        
            # N x S
            s = s.squeeze(1)
            
        else:
            N, C = x_real.shape[:2]
            # (N*C, F_eff, Tf)
            x_real = x_real.flatten(0, 1)
            x_img = x_img.flatten(0, 1)
            # (N*C, F_eff-2, Tf)
            x_real_padding = torch.flip(x_real[:, 1:-1, :], dims=[1])
            x_img_padding = torch.flip(-x_img[:, 1:-1, :], dims=[1])
            # (N*C, 2F, Tf)
            x = torch.cat((x_real, x_real_padding, x_img, x_img_padding), dim=1)
            # (N*C, 1, Tt)
            s = F.conv_transpose1d(input=x, weight=self.K, stride=self.stride, padding=0) 
            # (N, C, S)
            s = s.reshape(N, C, -1)
            
        if length:
            s = F.pad(s, pad=(0, length - s.shape[-1]))

        return s

    @staticmethod
    def get_idft_coeff(frame_len):
        N = frame_len
        idx = np.arange(0, N)
        idx_nk = 2.0 * math.pi * idx[:, np.newaxis] * idx[np.newaxis, :] / N
        # (F, F)
        idft_real_coeff = np.cos(idx_nk) / N
        idft_img_coeff = np.sin(idx_nk) / N
        return idft_real_coeff, idft_img_coeff  # [N,N]

    def get_istft_coeff(self, frame_len, win):
        assert len(win) == frame_len
        # gain = self._hop / np.sum(win * win)
        idft_real_coeff, idft_img_coeff = self.get_idft_coeff(frame_len)
        # (frame_len, frame_len)
        istft_real_coeff = win[:, np.newaxis] * idft_real_coeff
        istft_img_coeff = win[:, np.newaxis] * idft_img_coeff

        # (frame_len, 2*frame_len)
        return np.concatenate((istft_real_coeff, -istft_img_coeff), axis=1)


if __name__ == "__main__":
    # win_len = 512
    # hop_len = 256
    # x = torch.from_numpy(np.random.randn(160000) - 0.5)[None]
    
    # stft = STFT(win_len, hop_len, win_len)
    # X = stft.stft(x)
    # y = stft.istft(X)
    # print(y.shape)
    # print(torch.sum((y - x) **2))
    
    frame_len = 512
    frame_hop = 256
    x = np.random.rand(160000) - 0.5

    stft = STFT(frame_len, frame_hop)
    istft = ISTFT(frame_len, frame_hop)
    
    r, i = stft(torch.tensor(x).float()[None])

    y = istft(r, i).numpy()[0]
    
    print(x.shape, r.shape, i.shape, y.shape)
    
    print(np.sum((x[256:-256]-y[256:-256]) ** 2))

    