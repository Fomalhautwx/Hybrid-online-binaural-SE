import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from models.modules import Dereverberator, multich_istft
from models.AuxIVA import AuxIVA_nn_enh_batch
from models.AuxIVA_ICD_RLS_harmonic import AuxIVA_ICD_RLS_harmonic_nn_v2_Module
from models.AuxIVA_ISS_harmonic_v2 import AuxIVA_ISS_harmonic_nn_Frame_Module
from models.gtcrn_binaural import GTCRN_Binaural

from utils.iva_debug import guard_forward

import soundfile as sf
import time
import os


class TRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1,2))[0]
        at = self.att_fc(at).transpose(1,2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)

        return x * At


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Group Temporal Convolution"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        
        self.point_conv1 = conv_module(in_channels//2, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)
        
        self.tra = TRA(in_channels//2)

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        x = rearrange(x, 'b c g t f -> b (c g) t f')  # (B,2C,T,F)
        return x

    def forward(self, x):
        """x: (B, C, T, F)"""
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))
        h1 = self.tra(h1)
        x = self.shuffle(h1, x2)
        
        return x


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
    
    def forward(self, x):
        """x: (B, C, T, F)"""
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
        inter_x = self.inter_rnn(inter_x)[0]  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)      # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size) # (B,F,T,C)
        inter_x = inter_x.permute(0,2,1,3)   # (B,T,F,C)
        inter_x = self.inter_ln(inter_x) 
        inter_out = torch.add(intra_out, inter_x)
        
        dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)
        
        return dual_out


class Encoder(nn.Module):
    def __init__(self, in_channles, num_channels):
        super().__init__()

        self.en_convs = nn.ModuleList([
            ConvBlock(in_channles, num_channels, (1,5), stride=(1,2), padding=(0,2)),
            ConvBlock(num_channels, num_channels, (1,5), stride=(1,2), padding=(0,2), groups=2),
            GTConvBlock(num_channels, num_channels, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1)),
            GTConvBlock(num_channels, num_channels, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1)),
            GTConvBlock(num_channels, num_channels, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1))
        ])

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self, nsrc, num_channels, mask_type, n_mask=1):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(num_channels, num_channels, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1)),
            GTConvBlock(num_channels, num_channels, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1)),
            GTConvBlock(num_channels, num_channels, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1)),
            ConvBlock(num_channels, num_channels, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True),
            ConvBlock(num_channels, num_channels, (1,5), stride=(1,2), padding=(0,2), use_deconv=True)
        ])
        # 每个复数掩码占 2 个通道（实/虚）。n_mask=1 时与旧版完全一致；
        # n_mask=2 时通道顺序为 [M1_re, M1_im, M2_re, M2_im]，前两个通道就是旧版的 M1。
        self.post_conv = ConvBlock(num_channels, 2 * n_mask, (1,1), stride=(1,1), padding=(0,0), use_deconv=False)
            
        self.nsrc = nsrc

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers-1-i])
        x = self.post_conv(x)
        
        # m_l, m_r = [], []
        # for i in range(self.nsrc):
        #     # (B, 4/2, T, F)
        #     m_l.append(self.out_convs_l[i](x))
        #     m_r.append(self.out_convs_r[i](x))
        
        # return m_l, m_r
        return x    # 改成单通道增强范式
    

class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        # 单掩码（旧行为）: mask (B, 2, T, F),    spec (B, 2, T, F)
        # 多掩码:          mask (B, M, 2, T, F), spec (B, M, 2, T, F)
        # 输出 sum_m M_m * S_m（复数乘法后对 M 求和）: (B, F, T, 2)
        if mask.ndim == 4:
            mask = mask.unsqueeze(1)  # (B, 1, 2, T, F)
        if spec.ndim == 4:
            spec = spec.unsqueeze(1)  # (B, 1, 2, T, F)
        assert mask.shape == spec.shape, f"mask {tuple(mask.shape)} vs spec {tuple(spec.shape)}"
        spec = spec.permute(0, 1, 4, 3, 2)  # (B, M, F, T, 2)
        mask = mask.permute(0, 1, 4, 3, 2)  # (B, M, F, T, 2)
        # (B, F, T)
        s_real = (mask[..., 0]*spec[...,0] - mask[..., -1]*spec[...,-1]).sum(dim=1)
        s_imag = (mask[..., 0]*spec[...,-1] + mask[..., -1]*spec[...,0]).sum(dim=1)

        s = torch.stack([s_real, s_imag], dim=-1)  # (B,F,T,2)
        return s

def save_audio(path, spectrum, sample_rate=16000):
    '''spectrum: (B, C, F, T) complex'''
    waveform = multich_istft(spectrum, n_fft=512, hop_length=256, win_length=512)
    waveform = waveform.squeeze(0).T.cpu().numpy()
    sf.write(path, waveform, sample_rate)


class GTCRN_PARIS(nn.Module):
    def __init__(self,
                 nsrc=2,
                 win_len=512,
                 hop_len=256,
                 num_channels=32,
                 mask_type="dual",
                 bss=None,
                 bss_params=None, 
                 params=None,
                 **kwargs):
        super().__init__()        
        in_channels = 2 * nsrc
        self.derev = Dereverberator(nsrc=nsrc, params=params["dereverberator"])
        self.iva_iss = AuxIVA_ISS_harmonic_nn_Frame_Module(alpha2=0, bp_mode='inv')
        # mask 作用对象：
        #   'iva'      = 方案A postfilter：单个复数掩码作用于未压缩 IVA 目标路（默认）
        #   'iva_dual' = 双掩码：S_hat = M1*Z1 + M2*Z2，Z1/Z2 为 IVA 两路 back-projection 输出
        #                （等价于每个 bin 上学一个 2-tap 波束形成器 = 学出来的 GSC + 后置滤波）
        #   'raw'      = 旧行为（原始 ch0）
        self.mask_ref = (params or {}).get('mask_ref', 'iva')
        assert self.mask_ref in ('iva', 'iva_dual', 'raw'), f"unknown mask_ref: {self.mask_ref}"
        self.n_mask = nsrc if self.mask_ref == 'iva_dual' else 1

        self.encoder = Encoder(in_channels, num_channels)
        self.dpgrnn1 = DPGRNN(num_channels, 65, num_channels)
        self.dpgrnn2 = DPGRNN(num_channels, 65, num_channels)
        self.decoder = Decoder(nsrc, num_channels, mask_type, n_mask=self.n_mask)
        self.mask = Mask()
        self.mask_type = mask_type

        # 可选：M2..M_n 乘一个可学习标量门，初始化为 mask2_gate_init（建议 0.0）。
        # 门=0 时输出与单掩码模型完全一致；配合 warm_start_from_single_mask 可以
        # “从单掩码出发、只在有用时才用第二路”。None = 不加门（默认）。
        gate_init = (params or {}).get('mask2_gate_init', None)
        if self.n_mask > 1 and gate_init is not None:
            self.mask2_gate = nn.Parameter(torch.full((1,), float(gate_init)))
        else:
            self.mask2_gate = None
        self.last_masks = None   # 最近一次 forward 的掩码 (B, M, 2, T, F)，detach，供日志/诊断用

        self.derev.reset_state()
        # 设备可经 config["network"]["device"] 覆盖；默认保持 cuda（训练行为不变）
        device = (params or {}).get('device', 'cuda')
        # IVA/WPE 先验来源：'teacher'=冻结 teacher GTCRN 输出（默认）；'mixture_mag'=混合 ch0 幅度谱
        # （输入派生、可部署，无训练/测试失配；零相位保证 noise_ref = X0 - Y_nn 不退化）
        self.prior_type = (params or {}).get('prior_type', 'teacher')
        self.derev.to(torch.device(device))
        self.cl_factor = None

        self.prior_net = GTCRN_Binaural().to(torch.device(device))
        # 路径可经 config["network"]["prior_ckpt"] 覆盖；默认保持原硬编码路径
        prior_ckpt = (params or {}).get('prior_ckpt',
                                '/data/ssd0/jizhe.lu/exps/dcgtcrn/checkpoints/model_epoch68.tar')
        if self.prior_type == 'teacher':
            sd = torch.load(prior_ckpt, map_location='cpu')
            print(f"Loading prior_net from {prior_ckpt}")
            self.prior_net.load_state_dict(sd['model'], strict=False)
        self.prior_net.eval()
        for p in self.prior_net.parameters():
                p.requires_grad_(False)

    def validate_reset_buffer(self, batch_size=None):
        self.derev.reset_buffer_val(batch_size)
    
    def train_reset_buffer(self):
        self.derev.reset_state()
        
    # Curriculum learning factor setter
    def set_cl_factor(self, cl_factor):
        self.cl_factor = cl_factor
        
    def train(self, mode=True):
        super().train(mode)
        if self.prior_net is not None:
            self.prior_net.eval()                # ★ 永远 eval，别让 BN 统计被污染
        return self

    def state_dict(self, *a, **kw):              # ★ 不把教师权重存进 ckpt
        sd = super().state_dict(*a, **kw)
        return {k: v for k, v in sd.items() if not k.startswith('prior_net.')}

    @torch.no_grad()
    def warm_start_from_single_mask(self, single_sd):
        """用单掩码（mask_ref='iva'）的 checkpoint 初始化双掩码模型。
        除 decoder.post_conv 外所有张量形状相同，直接拷贝；post_conv 的前 2 个输出通道
        （= M1）拷贝旧权重，M2 的通道保持随机初始化。
        若同时设 mask2_gate_init=0.0，初始化后的输出与旧模型逐元素一致。
        single_sd: 旧 ckpt 的 state_dict（例如 torch.load(path)['model']）。
        返回 (copied, partially_copied, skipped) 三个 key 列表，方便核对。
        """
        own = super().state_dict()   # 含 prior_net，但旧 ckpt 里没有，不会被拷
        copied, partial, skipped = [], [], []
        for k, v in single_sd.items():
            if k.startswith('module.'):
                k = k[len('module.'):]   # 兼容 DDP 存下来的 key
            if k not in own:
                skipped.append(k); continue
            dst = own[k]
            if dst.shape == v.shape:
                dst.copy_(v); copied.append(k)
            elif k.startswith('decoder.post_conv.') and v.ndim >= 1 and dst.shape[1:] == v.shape[1:] \
                    and v.shape[0] <= dst.shape[0]:
                dst[:v.shape[0]].copy_(v); partial.append(k)
            else:
                skipped.append(k)
        return copied, partial, skipped

        
    def forward_spec(self, spec, init=None):
        """
        spec: (B, nch, F, T, 2)
        init: (B, 1, F, T, 2)
        2 parts
        输入必然是stacked虚实部
        """
        assert init.ndim == 5, "init should be (B, nsrc, F, T, 2)"
        init = init.detach()
        
        r = torch.roll(init, shifts=1, dims=-2)  # time-shifted init（沿用 init 所在设备）
        
        self.iva_iss.reset_state(batch_size=spec.shape[0])  # 重置 IVA 内部状态
        if self.training:
            self.train_reset_buffer()
        else:
            self.validate_reset_buffer(spec.shape[0])
        
        mag = (r[..., 0]**2 + r[..., 1]**2).mean(dim=1, keepdim=True)  # (b,1,F,T) squared magnitude
        mag = mag.permute(0, 2, 3, 1)  # (b,f,t,1)
        spec_der = self.derev.process(spec.permute(0,2,1,3,4), mag)  # (b, F, nch, T), (F,T,1)->(b,F,nch,T)
        prior = torch.complex(r[..., 0], r[..., 1])  # (B, nsrc, F, T)
        
        spec_iva = self.iva_iss.ChunkForward(spec_der.permute(0,2,1,3), prior)  # (B, nsrc, F, T) time-shifted init as Ynn        
       
        # debug_path = "/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/sample/cl_scratch"
        # save_audio(os.path.join(debug_path, f"spec_derev.wav"), spec_der.permute(0,2,1,3))
        # save_audio(os.path.join(debug_path, f"spec_iva.wav"), spec_iva[0:1])
        p = 0.3
        mag = spec_iva.abs().clamp_min(1e-6)          # 防止 0^(-0.7) = inf
        spec_iva_c = spec_iva * mag.pow(p - 1.0)      # 幅度压缩, 相位原样保留
        spec_iva_c = torch.view_as_real(spec_iva_c)  # (B, nsrc, F, T, 2)
        feat_mix = spec_iva_c.permute(0, 1, -1, 3, 2).flatten(1, 2).float()   # 虚实部叠加到通道维度展开
        # spec_iva = torch.stack([torch.real(spec_iva), torch.imag(spec_iva)], dim=-1)  # (B, nsrc, F, T, 2)
        # feat_mix = spec_iva.permute(0, 1, -1, 3, 2).flatten(1, 2).float()

        assert torch.isfinite(spec_iva).all(), \
    f"IVA 输出非有限! nan={torch.isnan(spec_iva).sum()}, inf={torch.isinf(spec_iva).sum()}"
        if self.mask_ref == 'iva_dual':
            # 双掩码：两路都用未压缩的 back-projection 输出作为载体（与 'iva' 同一个域）
            ref = torch.view_as_real(spec_iva[:, :self.n_mask, ...])  # (B, M, F, T, 2)
            ref = ref.permute(0, 1, -1, 3, 2)                           # (B, M, 2, T, F)
        else:
            if self.mask_ref == 'iva':
                # 方案A（postfilter 范式）：mask 作用于 spec_iva 的目标路。
                # spec_iva = ChunkForward 返回值 = back-projection 之后、未压缩的 IVA 估计（Y_bp），
                # 压缩版是后面两行生成的 spec_iva_c（仅作网络特征 feat_mix，不用于 mask 作用对象）。
                # 输出域=特征域：前端增益 1:1 传导，无需从压缩特征重建原始信号的 mask 分母。
                ref = torch.view_as_real(spec_iva[:, 0:1, ...])  # (B, 1, F, T, 2)
            else:
                ref = spec[:, 0:1, :, :, :]  # (B, 1, F, T, 2) 原始 ch0（旧行为）
            ref = ref.permute(0, 1, -1, 3, 2).flatten(1, 2)  # (B, 2, T, F)

        feat, en_outs = self.encoder(feat_mix)
        feat = self.dpgrnn1(feat)
        feat = self.dpgrnn2(feat)

        m_feat = self.decoder(feat, en_outs) # (B, 2*n_mask, T, F)
        if self.n_mask > 1:
            B_, _, T_, F_ = m_feat.shape
            m_feat = m_feat.view(B_, self.n_mask, 2, T_, F_)          # (B, M, 2, T, F)
            if self.mask2_gate is not None:
                m_feat = torch.cat([m_feat[:, :1], m_feat[:, 1:] * self.mask2_gate], dim=1)
            self.last_masks = m_feat.detach()
        else:
            self.last_masks = m_feat.detach().unsqueeze(1)
        spec_enh = self.mask(m_feat, ref) # (B, f, t, 2)
        spec_enh = torch.unsqueeze(spec_enh, dim=1)  # (B,C,F,T,2)
        return spec_enh # real
    
    @torch.no_grad()
    def set_teacher(self, audio):
        prior = self.prior_net(audio)
        # 调试存档开关：仅 DEBUG_SAVE_PRIOR=1 时写盘（训练时每个 forward 都会调这里
        if os.environ.get('DEBUG_SAVE_PRIOR') == '1':
            # 只存 batch 0：save_audio 的 multich_istft/squeeze(0) 假设 B=1，B>1 时 sf.write 会收到 3 维数组报错
            save_audio(f"/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/sample/noisy{time.strftime('%m%d_%H%M%S')}.wav", audio[:1])
            save_audio(f"/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/sample/prior{time.strftime('%m%d_%H%M%S')}.wav", prior[:1])
        return torch.view_as_real(prior)  # (B, nsrc, F, T, 2)       pass

    @torch.no_grad()
    def get_prior(self, x):
        """返回 IVA/WPE 先验 (B, 1, F, T, 2)（实虚部堆叠）。
        x: (B, C, F, T) complex 输入混合谱。
        teacher 先验：冻结 teacher GTCRN 输出（可部署：teacher 吃混合输入）。
        """
        return self.set_teacher(x)


    def forward(self, x, clean_prior):
        """
        x: (B, C, F, T) complex
        """
        spec_real, spec_imag = torch.real(x), torch.imag(x)
        # (B, C, F, T, 2)
        spec = torch.stack([spec_real, spec_imag], dim=-1)
        clean = torch.stack([torch.real(clean_prior), torch.imag(clean_prior)], dim=-1)
        
        # Stage 1
        spec_enh = self.forward_spec(spec, init=self.get_prior(x))  # (B, 1, F, T, 2)
        init_src = spec_enh.detach()                                  # 从寒假开始这梯度detach就出问题了
        if self.cl_factor is not None and self.training:
            init = self.cl_factor * init_src + (1 - self.cl_factor) * clean
        else:
            init = init_src
        
        spec_enh_enh = self.forward_spec(spec, init)
        # return complex spectrum
        # (B,nsrc,F,T) complex
        return torch.complex(spec_enh_enh[..., 0], spec_enh_enh[..., 1]), torch.complex(spec_enh[..., 0], spec_enh[..., 1])


if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    from torchinfo import summary
    import yaml
    
    with open("/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    model = GTCRN_PARIS(num_channels=32, mask_type="mono", bss=None, params=
                        config["network"])
    model.eval()
    
    x = torch.randn(16, 2, 257, 160, dtype=torch.complex64)
    enh = model(x)
    
    flops, params = get_model_complexity_info(model, (2, 16000), as_strings=True,
                                              print_per_layer_stat=False, verbose=True)
    print(flops, params)
    
    summary(model,
        input_data=[x],
        depth=3,
        verbose=1,
        col_width=16,   
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        row_settings=["var_names"])

    a = torch.randn(1, 2, 16000)
    b = torch.randn(1, 2, 16000)
    c = torch.randn(1, 2, 16000)
    x1 = torch.cat([a, b], dim=2)
    x2 = torch.cat([a, c], dim=2)