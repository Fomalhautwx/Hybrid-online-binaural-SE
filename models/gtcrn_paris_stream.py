"""
GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
Ultra tiny, 33.0 MMACs, 23.67 K params
"""
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from .modules import StreamConv2d, StreamConvTranspose2d, multich_stft


class StreamTRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x, h_cache):
        """
        x: (B,C,T,F)
        h_cache: (1,B,C)
        """
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at, h_cache = self.att_gru(zt.transpose(1,2), h_cache)
        at = self.att_fc(at).transpose(1,2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)

        return x * At, h_cache


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class StreamGTConvBlock(nn.Module):
    """Group Temporal Convolution"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        stream_conv_module = StreamConvTranspose2d if use_deconv else StreamConv2d
    
        # self.sfe = SFE(kernel_size=3, stride=1)
        
        self.point_conv1 = conv_module(in_channels//2, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = stream_conv_module(hidden_channels, hidden_channels, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)
        
        self.tra = StreamTRA(in_channels//2)

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])  # (B,2C,T,F)
        return x

    def forward(self, x, conv_cache, tra_cache):
        """
        x: (B, C, T, F)
        conv_cache: (B, C, (kT-1)*dT, F)
        tra_cache: (1, B, C)
        """
        x1, x2 = x[:,:x.shape[1]//2], x[:, x.shape[1]//2:]

        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1, conv_cache = self.depth_conv(h1, conv_cache)
        h1 = self.depth_act(self.depth_bn(h1))
        h1 = self.point_bn2(self.point_conv2(h1))

        h1, tra_cache = self.tra(h1, tra_cache)

        x =  self.shuffle(h1, x2)
        
        return x, conv_cache, tra_cache


class GRNN(nn.Module):
    """Grouped RNN"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (num_layers, B, hidden_size)
        """
        if h== None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers*2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h


class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""
    def __init__(self, input_size, width, hidden_size, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)
    
    def forward(self, x, inter_cache):
        """
        x: (B, C, T, F)
        inter_cache: (1, BF, hidden_size)
        """
        ## Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)      # (B*T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size) # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0,2,1,3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) 
        inter_x, inter_cache = self.inter_rnn(inter_x, inter_cache)     # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)      # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size) # (B,F,T,C)
        inter_x = inter_x.permute(0,2,1,3)   # (B,T,F,C)
        inter_x = self.inter_ln(inter_x) 
        inter_out = torch.add(intra_out, inter_x)
        
        dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)
        
        return dual_out, inter_cache


class StreamEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(4, 32, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            ConvBlock(32, 32, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            StreamGTConvBlock(32, 32, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
            StreamGTConvBlock(32, 32, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            StreamGTConvBlock(32, 32, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False)
        ])

    def forward(self, x, conv_cache, tra_cache):
        """
        x: (B,C,T,F)
        conv_cache: (B,C, (kT-1)*8, F)=1,32,16,65
        tra_cache: (3,1,B,C)
        """
        en_outs = []
        for i in range(2):
            x = self.en_convs[i](x)
            en_outs.append(x)
        x, conv_cache[:,:, :2, :], tra_cache[0] = self.en_convs[2](x, conv_cache[:,:, :2, :], tra_cache[0]); en_outs.append(x)
        x, conv_cache[:,:, 2:6, :], tra_cache[1] = self.en_convs[3](x, conv_cache[:,:, 2:6, :], tra_cache[1]); en_outs.append(x)
        x, conv_cache[:,:, 6:16, :], tra_cache[2] = self.en_convs[4](x, conv_cache[:,:, 6:16, :], tra_cache[2]); en_outs.append(x)
            
        return x, en_outs, conv_cache, tra_cache


class StreamDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            StreamGTConvBlock(32, 32, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=True),
            StreamGTConvBlock(32, 32, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=True),
            StreamGTConvBlock(32, 32, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=True),
            ConvBlock(32, 32, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(32, 32, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=False)
        ])
        self.post_conv = ConvBlock(32, 2, (1,1), stride=(1,1), padding=(0,0), use_deconv=False)

    def forward(self, x, en_outs, conv_cache, tra_cache):
        """
        x: (B,C,T,F)
        conv_cache: (B,C, (kT-1)*8, F)
        tra_cache: (3,1,B,C)
        """
        x, conv_cache[:,:, 6:16, :], tra_cache[0] = self.de_convs[0](x + en_outs[4], conv_cache[:,:, 6:16, :], tra_cache[0])
        x, conv_cache[:,:, 2:6, :], tra_cache[1] = self.de_convs[1](x + en_outs[3], conv_cache[:,:, 2:6, :], tra_cache[1])
        x, conv_cache[:,:, :2, :], tra_cache[2] = self.de_convs[2](x + en_outs[2], conv_cache[:,:, :2, :], tra_cache[2])
        
        for i in range(3, 5):
            x = self.de_convs[i](x + en_outs[4-i])
        x = self.post_conv(x)
        return x, conv_cache, tra_cache
    

class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        s_imag = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


class StreamGTCRN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = StreamEncoder()
        
        self.dpgrnn1 = DPGRNN(32, 65, 32)
        self.dpgrnn2 = DPGRNN(32, 65, 32)
        
        self.decoder = StreamDecoder()

        self.mask = Mask()

    def forward(self, spec, ref, conv_cache, tra_cache, inter_cache):
        """
        spec: (B, F, T, 2) = (1, 257, 1, 2)
        ref: (B, 1, F, T, 2)
        conv_cache: [en_cache, de_cache], (2, B, C, 8(kT-1), F) = (2, 1, 16, 16, 33)
        tra_cache: [en_cache, de_cache], (2, 3, 1, B, C) = (2, 3, 1, 1, 16)
        inter_cache: [cache1, cache2], (2, 1, BF, C) = (2, 1, 33, 16)
        """
        
        # spec_real, spec_imag = torch.real(spec), torch.imag(spec)
        # spec = torch.stack([spec_real, spec_imag], dim=-1)
        #spec_ref = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)  # (B,C,F,T,2)
        
        ref = ref.permute(0, 1, 4, 3, 2).flatten(1, 2)       # (B, 2, T, F)

        # (B, 2*2, T, F)
        feat_mix = spec.permute(0, 1, 4, 3, 2).flatten(1, 2)   # 虚实部叠加到通道维度展开
        # (B, 2, T, F)

        feat, en_outs, conv_cache[0], tra_cache[0] = self.encoder(feat_mix, conv_cache[0], tra_cache[0])
        
        feat, inter_cache[0] = self.dpgrnn1(feat, inter_cache[0]) # (B,16,T,33)
        feat, inter_cache[1] = self.dpgrnn2(feat, inter_cache[1]) # (B,16,T,33)

        m_feat, conv_cache[1], tra_cache[1] = self.decoder(feat, en_outs, conv_cache[1], tra_cache[1])
        
        m = m_feat

        spec_enh = self.mask(m, ref) # (B,2,T,F)
        spec_enh = torch.unsqueeze(spec_enh.permute(0,3,2,1), dim=1)  # (B,C,F,T,2)

        return spec_enh, conv_cache, tra_cache, inter_cache
    
    
class StreamMasked(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = StreamEncoder()
        
        self.dpgrnn1 = DPGRNN(32, 65, 32)
        self.dpgrnn2 = DPGRNN(32, 65, 32)
        
        self.decoder = StreamDecoder()

        self.mask = Mask()

    def forward(self, spec, mask, conv_cache, tra_cache, inter_cache):
        """
        spec: (B, F, T, 2) = (1, 257, 1, 2)
        mask: (B, 2, T, F) 
        conv_cache: [en_cache, de_cache], (2, B, C, 8(kT-1), F) = (2, 1, 16, 16, 33)
        tra_cache: [en_cache, de_cache], (2, 3, 1, B, C) = (2, 3, 1, 1, 16)
        inter_cache: [cache1, cache2], (2, 1, BF, C) = (2, 1, 33, 16)
        """
        
        # spec_real, spec_imag = torch.real(spec), torch.imag(spec)
        # spec = torch.stack([spec_real, spec_imag], dim=-1)
        #spec_ref = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)  # (B,C,F,T,2)
        #ref = spec_ref[:,0:1]# bctf: 1,1,1,572
        
        prev = self.mask(mask, spec[:, 0:1, :, :, :].permute(0, 1, 4, 3, 2).flatten(1, 2))  # (B,2,T,F)  # 上一轮的增强结果作为ref输入encode
        ref = spec[:, 0:1, :, :, :]  # (B, 1, F, T, 2)
        ref = ref.permute(0, 1, 4, 3, 2).flatten(1, 2)  # (B, 2, T, F)
        # (B, 2*2, T, F)
        feat_mix = spec.permute(0, 1, 4, 3, 2).flatten(1, 2)   # 虚实部叠加到通道维度展开
        # (B, 2, T, F)
        feat_mix = torch.cat([prev, feat_mix], dim=1)  # (B, 6, T, F)，把上一轮增强结果和原始混合一起输入encoder

        feat, en_outs, conv_cache[0], tra_cache[0] = self.encoder(feat_mix, conv_cache[0], tra_cache[0])
        
        feat, inter_cache[0] = self.dpgrnn1(feat, inter_cache[0]) # (B,16,T,33)
        feat, inter_cache[1] = self.dpgrnn2(feat, inter_cache[1]) # (B,16,T,33)

        m_feat, conv_cache[1], tra_cache[1] = self.decoder(feat, en_outs, conv_cache[1], tra_cache[1])
        
        m = m_feat

        spec_enh = self.mask(m, ref) # (B,2,T,F)
        spec_enh = torch.unsqueeze(spec_enh.permute(0,3,2,1), dim=1)  # (B,C,F,T,2)

        return spec_enh, conv_cache, tra_cache, inter_cache, m_feat


class StreamDecoderDual(StreamDecoder):
    """双掩码解码器：post_conv 输出 2*n_mask=4 通道，顺序 [M1_re, M1_im, M2_re, M2_im]。
    与 GTCRN_PARIS Decoder(n_mask=2) 对齐（前两通道即旧版单掩码的 M1）。"""
    def __init__(self):
        super().__init__()
        self.post_conv = ConvBlock(32, 4, (1,1), stride=(1,1), padding=(0,0), use_deconv=False)


class StreamGTCRNDual(StreamGTCRN):
    """双掩码流式版（与 GTCRN_PARIS mask_ref='iva_dual' 对齐）：
    输出 = M1 ⊙ Z1 + gate·M2 ⊙ Z2，Z1/Z2 = 未压缩 IVA 两路 back-projection 输出。
    mask2_gate 从 ckpt 键 'mask2_gate' 加载（buffer，ONNX 导出时烘焙为常量）。"""
    def __init__(self):
        super().__init__()
        self.decoder = StreamDecoderDual()
        self.register_buffer('mask2_gate', torch.tensor(0.0))

    def forward(self, spec, ref, conv_cache, tra_cache, inter_cache):
        """
        spec: (B, 2, F, T, 2) 压缩 IVA 特征（两路）
        ref:  (B, 2, F, T, 2) 未压缩 IVA 两路（bp 后）——复数乘法载体
        """
        ref = ref.permute(0, 1, 4, 3, 2)                     # (B, M=2, 2, T, F)
        feat_mix = spec.permute(0, 1, 4, 3, 2).flatten(1, 2)  # (B, 4, T, F)

        feat, en_outs, conv_cache[0], tra_cache[0] = self.encoder(feat_mix, conv_cache[0], tra_cache[0])
        feat, inter_cache[0] = self.dpgrnn1(feat, inter_cache[0])
        feat, inter_cache[1] = self.dpgrnn2(feat, inter_cache[1])

        m_feat, conv_cache[1], tra_cache[1] = self.decoder(feat, en_outs, conv_cache[1], tra_cache[1])

        m = m_feat.view(m_feat.shape[0], 2, 2, m_feat.shape[2], m_feat.shape[3])  # (B, M=2, 2, T, F)
        g = torch.stack([torch.ones_like(self.mask2_gate), self.mask2_gate])       # (2,)  [1, gate]
        m = m * g.view(1, 2, 1, 1, 1)

        s_real = (m[:, :, 0] * ref[:, :, 0] - m[:, :, 1] * ref[:, :, 1]).sum(dim=1)  # (B,T,F)
        s_imag = (m[:, :, 0] * ref[:, :, 1] + m[:, :, 1] * ref[:, :, 0]).sum(dim=1)
        s = torch.stack([s_real, s_imag], dim=1)               # (B,2,T,F)

        spec_enh = torch.unsqueeze(s.permute(0, 3, 2, 1), dim=1)  # (B,1,F,T,2)
        return spec_enh, conv_cache, tra_cache, inter_cache


if __name__ == "__main__":
    import os
    import time
    import soundfile as sf
    from tqdm import tqdm
    from gtcrn_paris_binaural import GTCRN_PARIS_Binaural as GTCRN
    from .modules import convert_to_stream
    
    device = torch.device("cpu")

    model = GTCRN().to(device).eval()
    
    model.load_state_dict(torch.load('/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/exps/gtcrn_1207/checkpoints/model_epoch90.tar', map_location=device)['model'], strict=False)
    stream_model = StreamGTCRN().to(device).eval()
    # for name, param in stream_model.encoder.named_parameters():
    #     print(name, param.shape)
    convert_to_stream(stream_model, model)
    
    """Streaming Conversion"""
    ### offline inference
    x = torch.from_numpy(sf.read('/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/librivox_mixdown.wav', dtype='float32')[0])
    x = multich_stft(x.T.unsqueeze(0))
    print(x.shape)  # (1, 2, 257, 572)  # B, C, F, T
    x = torch.view_as_real(x).to(device)  # (1, 2, 257, 572, 2)

    ### online (streaming) inference
    conv_cache = torch.zeros(2, 1, 32, 16, 65).to(device)
    tra_cache = torch.zeros(2, 3, 1, 1, 32).to(device)
    inter_cache = torch.zeros(2, 1, 65, 32).to(device)
    ys = []
    times = []
    yi = None   # 初始化init变量

    # for i in tqdm(range(x.shape[3])):
    #     xi = x[:,:,:,i:i+1]
    #     tic = time.perf_counter()
    #     with torch.no_grad():
    #         yi, conv_cache, tra_cache, inter_cache = stream_model(xi, conv_cache, tra_cache, inter_cache, yi)
    #     toc = time.perf_counter()
    #     times.append((toc-tic)*1000)
    #     ys.append(yi)  # (1, 1, 257, 1, 2)虚实stack的
        
    # ys = torch.cat(ys, dim=3)  # (1, 1, 257, 572, 2)
    # ys = torch.complex(ys[...,0], ys[...,1])  # (1, 1, 257, 572)
    # ys = ys.squeeze(1)  # (1, 257, 572)

    # ys = torch.istft(ys, 512, 256, 512, torch.hann_window(512).pow(0.5)).detach().cpu().numpy()

    # os.makedirs('test_wavs', exist_ok=True)
    # sf.write('test_wavs/enh_stream.wav', ys.squeeze(), 16000)
    # print(">>> inference time: mean: {:.1f}ms, max: {:.1f}ms, min: {:.1f}ms".format(sum(times)/len(times), max(times), min(times)))
    # # print(">>> Streaming error:", np.abs(y-ys).max())


    """ONNX Conversion"""
    import os
    import time
    import onnx
    import onnxruntime
    from onnxsim import simplify
    from librosa import istft
    
    ## convert to onnx
    file = 'onnx_models/gtcrn.onnx'
    os.makedirs('onnx_models', exist_ok=True)
    #if not os.path.exists(file):
    input = torch.randn(1, 2, 257, 1, 2, device=device)
    init = torch.zeros(1, 1, 257, 1, 2, device=device)
    ref_dummy  = torch.randn(1, 1, 257, 1, 2, device=device) 
    torch.onnx.export(stream_model,
                    (input, conv_cache, tra_cache, inter_cache, init),
                    file,
                    input_names = ['spec', 'conv_cache', 'tra_cache', 'inter_cache', 'init'],
                    output_names = ['spec_enh', 'conv_cache_out', 'tra_cache_out', 'inter_cache_out'],
                    opset_version=11,
                    verbose = False)

    onnx_model = onnx.load(file)
    onnx.checker.check_model(onnx_model)

    # simplify onnx model
    #if not os.path.exists(file.split('.onnx')[0]+'_simple.onnx'):
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, file.split('.onnx')[0] + '_simple.onnx')


    ## run onnx model
    # session = onnxruntime.InferenceSession(file, None, providers=['CPUExecutionProvider'])
    session = onnxruntime.InferenceSession(file.split('.onnx')[0]+'_simple.onnx', None, providers=['CPUExecutionProvider'])
    conv_cache = np.zeros([2, 1, 32, 16, 65],  dtype="float32")
    tra_cache = np.zeros([2, 3, 1, 1, 32],  dtype="float32")
    inter_cache = np.zeros([2, 1, 65, 32],  dtype="float32")

    T_list = []
    outputs = []

    inputs = x.numpy()
    for i in tqdm(range(inputs.shape[-2])):
        tic = time.perf_counter()
        
        out_i,  conv_cache, tra_cache, inter_cache \
                = session.run([], {'spec': inputs[..., i:i+1, :],
                    'conv_cache': conv_cache,
                    'tra_cache': tra_cache,
                    'inter_cache': inter_cache,
                    'init': out_i if i>0 else np.zeros([1, 1, 257, 1, 2], dtype="float32")})

        toc = time.perf_counter()
        T_list.append(toc-tic)
        outputs.append(out_i)

    outputs = np.concatenate(outputs, axis=3)
    print(outputs.shape)  # (1,1,257,572,2)
    enhanced = istft(outputs[...,0] + 1j * outputs[...,1], n_fft=512, hop_length=256, win_length=512, window=np.hanning(512)**0.5)

    sf.write('test_wavs/enh_onnx.wav', enhanced.squeeze(), 16000)
    
    #print(">>> Onnx error:", np.abs(y - enhanced).max())
    print(">>> inference time: mean: {:.1f}ms, max: {:.1f}ms, min: {:.1f}ms".format(1e3*np.mean(T_list), 1e3*np.max(T_list), 1e3*np.min(T_list)))
    print(">>> RTF:", 1e3*np.mean(T_list) / 16)

