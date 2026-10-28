from xml.parsers.expat import model

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
from typing import Union, Tuple
from tqdm import tqdm
from .dnn import backbones
from .dsp import wpe
import cProfile, pstats, io


EPS = 1e-10

def win_hanning_period(size):
    a = 2 * np.pi / size
    f = np.arange(size)
    window = 0.5 - 0.5 * np.cos(a * f)
    return window
    

def multich_stft(x, n_fft=512, hop_length=256, win_length=512):
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


def multich_istft(x, n_fft=512, hop_length=256, win_length=512):
    """
    x: (B, 2C, F, T) -> 还原为 (B, C, T)
    """
    batch, channels, freq, time = x.shape
    stft_complex = x  # (B, C, F, T)

    # 重塑为 (B*C, F, T) 方便批量处理
    stft_complex = stft_complex.reshape(batch * channels, freq, time)

    # 创建与 STFT 相同的窗函数
    window = torch.hann_window(win_length).pow(0.5).to(x.device)

    # 执行 ISTFT
    istft = torch.istft(
        stft_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=None,  # 你可以指定原始长度
        return_complex=False
    )  # -> (B*C, T)

    # 重塑回 (B, C, T)
    samples = istft.shape[-1]
    istft = istft.reshape(batch, channels, samples)
    return istft

def convert_to_stream(stream_model, model):
    # encoder需要加上几处Conv2d deoder需要加上几处ConvTranspose2d
    state_dict = model.state_dict()
    new_state_dict = stream_model.state_dict()

    for key in stream_model.state_dict().keys():
        if key in state_dict.keys():
            if ('point_conv' in key) and 'decode' in key and key.endswith('weight'):
                new_state_dict[key] = state_dict[key].permute([1,0,2,3])
            else:
                new_state_dict[key] = state_dict[key]

        elif key.replace('Conv1d.', '') in state_dict.keys():
            new_state_dict[key] = state_dict[key.replace('Conv1d.', '')]
    
        elif key.replace('Conv2d.', '') in state_dict.keys():
            new_state_dict[key] = state_dict[key.replace('Conv2d.', '')]
        
        ## For StreamConvTranspose2d Version 1:    
        # elif key.replace('ConvTranspose2d.', '') in state_dict.keys():
        #     new_state_dict[key] = state_dict[key.replace('ConvTranspose2d.', '')]
        
        ## For StreamConvTranspose2d Version 2:  
        elif key.replace('ConvTranspose2d.', '') in state_dict.keys():
            if key.endswith('weight'):
                if new_state_dict[key].shape != state_dict[key.replace('ConvTranspose2d.', '')].shape:
                    # new_state_dict[key] = torch.flip(state_dict[key.replace('ConvTranspose2d.', '')].permute([1,0,2,3]), dims=[-2,-1])
                    new_state_dict[key] = state_dict[key.replace('ConvTranspose2d.', '')].permute([1,0,2,3])
                else:
                    # new_state_dict[key] = torch.flip(state_dict[key.replace('ConvTranspose2d.', '')], dims=[-2,-1])
                    new_state_dict[key] = state_dict[key.replace('ConvTranspose2d.', '')]
            
            else:
                new_state_dict[key] = state_dict[key.replace('ConvTranspose2d.', '')]


        else:
            print(key)
            raise(ValueError('key error!'))
        
    stream_model.load_state_dict(new_state_dict)


    
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


class Dereverberator(nn.Module):
    def __init__(self, nsrc, params):
        super().__init__()
        self.channels = nsrc
        self.wpe = getattr(wpe, params["wpe"]["class"])(params, params["wpe"])
        self.dnn_wpe = getattr(backbones, params["dnn_wpe"]["class"])(**params["dnn_wpe"])
        self.dnn_wpe.load_state_dict(torch.load(params["dnn_wpe"]["ckpt"]))
        self.dnn_wpe.load_stats(params["dnn_wpe"]["stats"])
        for param in self.dnn_wpe.parameters():
            param.requires_grad = False

    def to(self, device):
        self.device = device
        self.wpe = self.wpe.to(device)
        self.dnn_wpe = self.dnn_wpe.to(device)
        return self

    def reset_state(self):
        self.dnn_wpe.reset_state()
        self.wpe.reset_state()
        # self.dnn_pf.reset_state()
        # self.pf.reset_state()

    def reset_buffer_val(self, batch_size=None):
        if batch_size is None:
            self.wpe.validation_set_state()
        else:
            self.wpe.validation_set_state(batch_size)
        self.dnn_wpe.reset_state()

    def process(self, Y, clean_mag, **kwargs):
        '''
        clean_mag: F x 1 x 1 or so. magnitude prior every freq bin
        Y: F x D x T × 2
        # (B, F, nch, T), (F,T,1)
        '''
        self.to(Y.device)
        Y = torch.complex(Y[..., 0], Y[..., 1])  # F x D x T 复数化
        B, F, D, T = Y.size()
        assert self.channels == D, f"Mismatch in channels : {self.channels} model vs {D} input"
        # pr = cProfile.Profile()
        # pr.enable()
        X = torch.zeros_like(Y) #[F, D, T]
        with torch.no_grad():
            res = Y.shape[0]
            if res != self.wpe.Y_buffer.shape[0] and self.training:
                Y = self.wpe.padding_batch(Y, res)
                clean_mag = self.wpe.padding_batch(clean_mag, res)
            for _ in range(2): #initialize WPE statistics to get the best performance (optional)
                mag = self.dnn_wpe(Y, return_interference=False)[0]
                for t in range(T):
                    Y_update = Y[..., t] #F,D
                    prior = clean_mag[...,t,:] #b f 1
                    # DNN-WPE
                    # clean_periodogram = mag.squeeze() #F
                    #clean_mag, _ = self.dnn_wpe(Y_update.unsqueeze(-1), return_interference=False)
                    #F,1,1
                    
                    #clean_periodogram = torch.square(clean_mag.squeeze(-1)) #B, F
                    
                    #WPE
                    X_update = self.wpe.step_online(Y_update, prior.squeeze(-1)) #[F, D]
                    # WPE
                    # X_update = self.wpe.step_online(Y_update, mag.squeeze(-1)) #[F, D]
                    X[..., t] = X_update[:res]
        X[:, : 2] = .0 + 0j
        X = torch.complex(X.real.clamp(-1e3, 1e3), X.imag.clamp(-1e3, 1e3))
        n_bad = (~torch.isfinite(X)).sum().item()
        if n_bad > 0:
            print(f"[WPE] {n_bad} non-finite values sanitized")
        # pr.disable()
        # s = io.StringIO()
        # sortby = "cumtime"  # 仅适用于 3.6, 3.7 把这里改成常量了
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())
        # f, d, t
        # self.dnn_wpe.reset_state()
        # complex return
        return X
    
    def streaming(self, Y_frame, mag_prior):
        '''
        Args:
            Y_frame: [F, D] complex
            mag_prior: [B, F, 1] 当前帧的先验幅度之平方 (Magnitude Prior)
        Returns:
            X_frame: [B, F, D] 复数格式的去混响后当前帧
        '''
        # 2. 准备先验功率谱 [B, F]
        clean_mag = self.dnn_wpe(Y_frame.unsqueeze(-1), return_interference=False)[0]
        # b f d t的clean mag
        #clean_periodogram = torch.abs(mag_prior) ** 2
        clean_periodogram = torch.square(clean_mag).squeeze()
        #clean_periodogram = torch.abs(Y_frame[..., 0]) ** 2   # 直接使用输入的第一个通道的幅度平方作为先验功率谱
        # ref_pow = (Y_frame.abs() ** 2).mean().clamp_min(1e-12)
        # clean_periodogram = clean_periodogram.clamp_min(1e-6 * ref_pow)
        
        with torch.no_grad():
            X_frame = self.wpe.step_online(Y_frame, clean_periodogram) #[F, D]
            X_frame[:, : 2] = .0 + 0j
            X = torch.complex(X_frame.real.clamp(-1e3, 1e3), X_frame.imag.clamp(-1e3, 1e3))
        return X
    

class StreamConv2d(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 *args, **kargs):
        super().__init__(*args, **kargs)
        """
        kernel_size = [T_size, F_size] by defalut
        """
        if type(padding) is int:
            self.T_pad = padding
            self.F_pad = padding
        elif type(padding) in [list, tuple]:
            self.T_pad, self.F_pad = padding
        else:
            raise ValueError('Invalid padding size.')
        
        assert self.T_pad == 0, "To meet the demands of causal streaming requirements"
        
        self.Conv2d = nn.Conv2d(in_channels = in_channels, 
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = padding,
                                dilation = dilation,
                                groups = groups,
                                bias = bias)
            
    def forward(self, x, cache):
        """
        x: [bs, C, 1, F]
        cache: [bs, C, T_size-1, F]
        """
        inp = torch.cat([cache, x], dim=2)
        outp = self.Conv2d(inp)
        out_cache = inp[:,:, 1:]
        return outp, out_cache
    

class StreamConvTranspose2d(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 *args, **kargs):
        super().__init__(*args, **kargs)
        """
        kernel_size = [T_size, F_size] by default
        stride = [T_stride, F_stride], and T_stride == 1
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        if type(kernel_size) is int:
            self.T_size = kernel_size
            self.F_size = kernel_size
        elif type(kernel_size) in [list, tuple]:
            self.T_size, self.F_size = kernel_size
        else:
            raise ValueError('Invalid kernel size.')
            
        if type(stride) is int:
            self.T_stride = stride
            self.F_stride = stride
        elif type(stride) in [list, tuple]:
            self.T_stride, self.F_stride = stride
        else:
            raise ValueError('Invalid stride size.')
        
        assert self.T_stride == 1

        if type(padding) is int:
            self.T_pad = padding
            self.F_pad = padding
        elif type(padding) in [list, tuple]:
            self.T_pad, self.F_pad = padding
        else:
            raise ValueError('Invalid padding size.')
        assert(self.T_pad == 0) 

        if type(dilation) is int:
            self.T_dilation = dilation
            self.F_dilation = dilation
        elif type(dilation) in [list, tuple]:
            self.T_dilation, self.F_dilation = dilation
        else:
            raise ValueError('Invalid dilation size.')
        
        # Implementing ConvTranspose2d using Conv2d with weight-time reversal.
        self.ConvTranspose2d = nn.Conv2d(in_channels = in_channels, 
                                        out_channels = out_channels,
                                        kernel_size = kernel_size,
                                        stride = (self.T_stride, 1), # An additional upsampling will be used in forward, if F_stride != 1
                                        padding = (self.T_pad, 0),   # An additional padding will be used in forward, if F_pad != 0
                                        dilation = dilation,
                                        groups = groups,
                                        bias = bias)
        
    def forward(self, x, cache):
        """
        x: [bs,C,1,F]
        cache: [bs,C,T-1,F]
        """
        # [bs,C,T,F]
        inp = torch.cat([cache, x], dim = 2)
        out_cache = inp[:, :, 1:]
        bs, C, T, F = inp.shape
        
        # Upsampling operation
        if self.F_stride > 1: 
            # [bs,C,T,F] -> [bs,C,T,F,1] -> [bs,C,T,F,F_stride] -> [bs,C,T,F_out]
            inp = torch.cat([inp[:,:,:,:,None], torch.zeros([bs,C,T,F,self.F_stride-1])], dim = -1).reshape([bs,C,T,-1])
            left_pad = self.F_stride - 1
            if self.F_size > 1:
                if left_pad <= self.F_size - 1:
                    inp = torch.nn.functional.pad(inp, pad = [(self.F_size - 1)*self.F_dilation-self.F_pad, (self.F_size - 1)*self.F_dilation-self.F_pad - left_pad, 0, 0])
                else:
                    # inp = torch.nn.functional.pad(inp, pad = [self.F_size - 1, 0, 0, 0])[:,:,:,: - (left_pad - self.F_stride + 1)]
                    raise(NotImplementedError)
            else:
                # inp = inp[:,:,:,:-left_pad]
                raise(NotImplementedError)

        else: # F_stride = 1
            inp = torch.nn.functional.pad(inp, pad=[(self.F_size-1)*self.F_dilation-self.F_pad, (self.F_size-1)*self.F_dilation-self.F_pad])
                
        outp = self.ConvTranspose2d(inp)
    
        return outp, out_cache


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

    

    