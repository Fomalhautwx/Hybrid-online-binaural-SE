import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from .modules import STFT, ISTFT


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
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        self.pad_size = (kernel_size[0]-1) * dilation[0]

        self.point_conv1 = ConvBlock(in_channels//2, hidden_channels, (1, 3), padding=(0, 1))

        self.depth_conv = ConvBlock(hidden_channels, hidden_channels, kernel_size,
                                    stride=stride, padding=padding,
                                    dilation=dilation, groups=hidden_channels)

        self.point_conv2 = ConvBlock(hidden_channels, in_channels//2, 1)
        self.point_conv2.act = nn.Identity()
        
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

        h1 = self.point_conv1(x1)
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_conv(h1)
        h1 = self.point_conv2(h1)
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
    def __init__(self, nsrc, num_channels, mask_type):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(num_channels, num_channels, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1)),
            GTConvBlock(num_channels, num_channels, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1)),
            GTConvBlock(num_channels, num_channels, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1)),
            ConvBlock(num_channels, num_channels, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True),
            ConvBlock(num_channels, num_channels, (1,5), stride=(1,2), padding=(0,2), use_deconv=True)
        ])
        
        self.out_convs = nn.ModuleList()
        
        for _ in range(nsrc):
            if mask_type == "dual":
                self.out_convs.append(
                    nn.Sequential(
                        nn.Conv2d(num_channels, nsrc*2, kernel_size=1),
                        nn.Tanh()
                    ))
            
            elif mask_type == "mono":
                self.out_convs.append(
                    nn.Sequential(
                        nn.Conv2d(num_channels, 2, kernel_size=1),
                        nn.Tanh()
                    ))
               
            else:
                raise NotImplementedError
            
        self.nsrc = nsrc

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers-1-i])
        
        m = []
        for i in range(self.nsrc):
            # (B, 4/2, T, F)
            m.append(self.out_convs[i](x))
        
        return m
    

class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        # mask: (B, 1/M, F, T, 2)
        # spec: (B, F, T, 2) or (B, M, F, T, 2)
        if spec.ndim == 4:
            spec = spec.unsqueeze(1)
        
        # (B, F, T)
        s_real = (mask[..., 0]*spec[...,0] - mask[..., -1]*spec[...,-1]).sum(dim=1)
        s_imag = (mask[..., 0]*spec[...,-1] + mask[..., -1]*spec[...,0]).sum(dim=1)

        s = torch.stack([s_real, s_imag], dim=-1)  # (B,F,T,2)
        return s


class GTCRN_PARIS(nn.Module):
    def __init__(self,
                 nsrc=2,
                 win_len=512,
                 hop_len=256,
                 num_channels=32,
                 mask_type="dual"):
        super().__init__()
        self.stft = STFT(win_len, hop_len)
        self.istft = ISTFT(win_len, hop_len)
        
        in_channels = 3 * nsrc
        self.encoder = Encoder(in_channels, num_channels)
        
        self.dpgrnn1 = DPGRNN(num_channels, 65, num_channels)
        self.dpgrnn2 = DPGRNN(num_channels, 65, num_channels)
        
        self.decoder = Decoder(nsrc, num_channels, mask_type)

        self.mask = Mask()
        self.mask_type = mask_type
    
    def forward_spec(self, spec, init=None):
        """
        spec: (B, nch, F, T, 2)
        init: (B, nsrc, F, T, 2)
        """
        
        if init is None:
            # (B, nsrc, F, T, 2)
            r = torch.zeros_like(spec)
                
        else:
            # (B, nsrc, F, T, 2)
            init = init.detach()
            r = torch.zeros_like(init)
            r[:, :, :, 1:] = init[:, :, :, :-1]
        
        # (B, nsrc, F, T)
        r = torch.norm(r, dim=-1)
        
        # (B, nch*2, T, F)
        feat_mix = spec.permute(0, 1, -1, 3, 2).flatten(1, 2)
        # (B, 1, T, F)
        feat_r1 = r.permute(0, 1, 3, 2)[:, 0:1]
        feat_r2 = r.permute(0, 1, 3, 2)[:, 1:2]
        
        # (B, nch*3, T, F)
        feat = torch.concat([feat_r1, feat_mix, feat_r2], dim=1)
        feat, en_outs = self.encoder(feat)
        
        feat = self.dpgrnn1(feat)
        feat = self.dpgrnn2(feat)

        m_feat = self.decoder(feat, en_outs) # (B, 2/M*2, T, F) * nsrc
        
        spec_srcs = []
        for i in range(len(m_feat)):
            m_real, m_imag = m_feat[i].permute(0, 1, -1, 2).chunk(2, dim=1) # (B, 1/M, F, T)
            m = torch.stack([m_real, m_imag], dim=-1) # (B, 1/M, F, T, 2)
            
            # (B,F,T,2)
            if self.mask_type == "mono":
                spec_src = self.mask(m, spec[:, 0])
                
            else:
                spec_src = self.mask(m, spec)
            
            spec_srcs.append(spec_src)
        
        output = torch.stack(spec_srcs, dim=1) # (B,nsrc,F,T,2)
        
        return output
    
    def forward(self, x, return_wav=True, spec_io=False):
        """
        x: (B, C, L) or (B, C, F, T) complex
        """
        
        if spec_io:
            # (B, C, F, T)
            spec_real, spec_imag = torch.real(x), torch.imag(x)
        else:
            # (B, C, F, T)
            spec_real, spec_imag = self.stft(x)
        # (B, C, F, T, 2)
        spec = torch.stack([spec_real, spec_imag], dim=-1)
        
        # (B, nsrc, F, T, 2)
        r = self.forward_spec(spec, init=None)
        y = self.forward_spec(spec, init=r)
        
        if spec_io:
            # (B,nsrc,F,T) complex
            r = torch.complex(r[..., 0], r[..., 1])
            y = torch.complex(y[..., 0], y[..., 1])
            return r, y

        if return_wav:
            r = self.istft(r[..., 0], r[..., 1], length=x.shape[-1]) # (B,nsrc,L)
            y = self.istft(y[..., 0], y[..., 1], length=x.shape[-1]) # (B,nsrc,L)
        
        return r, y


if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    from torchinfo import summary
    
    model = GTCRN_PARIS(num_channels=32, mask_type="mono")
    model.eval()
    
    x = torch.randn(1, 2, 16000)
    r, y = model(x)
    print(r.shape, r.dtype, y.shape, y.dtype)

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

    r1, y1 = model(x1)
    r2, y2 = model(x2)
    print((r1[..., :16000-384] - r2[..., :16000-384]).abs().max())
    print((r1[..., 16000-384: 16000+128] - r2[..., 16000-384:16000+128]).abs().max())
    
    print((y1[..., :16000-384] - y2[..., :16000-384]).abs().max())
    print((y1[..., 16000-384: 16000+128] - y2[..., 16000-384:16000+128]).abs().max())