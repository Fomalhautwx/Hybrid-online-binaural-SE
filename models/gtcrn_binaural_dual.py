"""双掩码 DC-GTCRN（端到端、无 WPE/IVA 前级）。

与 models/gtcrn_binaural.py 的唯一架构差异：decoder 输出 2 个掩码（4 通道），
分别作用于两路输入麦克风通道后求和：output = M1 ⊙ X_ch0 + M2 ⊙ X_ch1。

这是双掩码级联（M1⊙Z1 + gate·M2⊙Z2，Z 为两路 IVA 输出）的**公平端到端对照**：
同样"两个掩码乘两路信号再求和"，只是载体从"IVA 分离后的两路"换成"原始双麦"。
无 gate（级联里的 gate 是 warm-start 便利，端到端从头训不需要）。
输出形状与单掩码 GTCRN_Binaural 完全一致 (B, 1, F, T) complex，trainer/loss 无需改。
"""
import torch
import torch.nn as nn

from models.gtcrn_binaural import (TRA, ConvBlock, GTConvBlock, GRNN,
                                   DPGRNN, Encoder)


class DecoderDual(nn.Module):
    def __init__(self, nsrc, num_channels):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(num_channels, num_channels, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)),
            GTConvBlock(num_channels, num_channels, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)),
            GTConvBlock(num_channels, num_channels, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)),
            ConvBlock(num_channels, num_channels, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True),
            ConvBlock(num_channels, num_channels, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True)
        ])
        # 2 个掩码 × (实/虚) = 4 通道，顺序 [M1_re, M1_im, M2_re, M2_im]
        self.post_conv = ConvBlock(num_channels, 2 * nsrc, (1, 1), stride=(1, 1), padding=(0, 0), use_deconv=False)
        self.nsrc = nsrc

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers - 1 - i])
        return self.post_conv(x)


class GTCRN_Binaural_Dual(nn.Module):
    def __init__(self, nsrc=2, num_channels=32):
        super().__init__()
        in_channels = 2 * nsrc
        self.encoder = Encoder(in_channels, num_channels)
        self.dpgrnn1 = DPGRNN(num_channels, 65, num_channels)
        self.dpgrnn2 = DPGRNN(num_channels, 65, num_channels)
        self.decoder = DecoderDual(nsrc, num_channels)
        self.n_mask = nsrc

    def forward(self, x):
        """
        x: (B, C=2, F, T) complex
        返回: (B, 1, F, T) complex（与单掩码 GTCRN_Binaural 同形状）
        """
        spec = torch.view_as_real(x).float()                     # (B, C, F, T, 2)
        B, C, F, T, _ = spec.shape

        # ref: 每路 mic 作为一路 mask 载体 (B, M=C, 2, T, F)，2 = (实, 虚)
        ref = spec.permute(0, 1, 4, 3, 2)
        # 特征：两路 mic 的实虚部展开 (B, 2*C, T, F)
        feat = spec.permute(0, 1, 4, 3, 2).flatten(1, 2)

        feat, en_outs = self.encoder(feat)
        feat = self.dpgrnn1(feat)
        feat = self.dpgrnn2(feat)
        m_feat = self.decoder(feat, en_outs)                      # (B, 2*n_mask, T, F)

        m = m_feat.view(B, self.n_mask, 2, T, F)                  # (B, M, 2, T, F)
        # 复数乘法后对 M 求和：output = Σ_m M_m ⊙ ref_m
        s_real = (m[:, :, 0] * ref[:, :, 0] - m[:, :, 1] * ref[:, :, 1]).sum(dim=1)  # (B, T, F)
        s_imag = (m[:, :, 0] * ref[:, :, 1] + m[:, :, 1] * ref[:, :, 0]).sum(dim=1)
        s = torch.stack([s_real, s_imag], dim=-1)                 # (B, T, F, 2)

        spec_enh = torch.unsqueeze(s.permute(0, 2, 1, 3), dim=1)  # (B, 1, F, T, 2)
        return torch.view_as_complex(spec_enh)


if __name__ == "__main__":
    model = GTCRN_Binaural_Dual(nsrc=2, num_channels=32).eval()
    x = torch.randn(1, 2, 257, 100, dtype=torch.complex64)
    with torch.no_grad():
        y = model(x)
    print('input', tuple(x.shape), '-> output', tuple(y.shape), y.dtype)
