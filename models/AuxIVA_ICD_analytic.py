"""
=======================================================
接口与原版完全一致:
    - 类名 / 构造参数:  AuxIVA_ICD_RLS_harmonic_nn_Frame_Module(
          bandsize=257, n_chan=2, nsrc=2, alpha=0.995, alpha2=0,
          float_dtype=torch.float64, complex_dtype=torch.complex128)
    - forward(X, Y_nn) -> (Y_out.flip(dims=[1]), score0/score1, prior_energy)
      X: (B, nch, F) complex;  Y_nn: (B, nsrc, F) complex

新增的可选关键字参数（带默认值, 不影响原调用方式）:
    use_harmonic_phi : bool  是否使用按频点的谐波 clique 权重（默认 True）
    r_floor_ratio    : float clique 能量的相对地板（默认 0.01, 即 -20 dB）
    diag_load        : float 求逆前对 V 的相对对角加载（默认 1e-6）
"""
import numpy as np
import torch
import torch.nn as nn


def calcu_harmonic_clique(nbands=513, fs=16000):
    """生成谐波 clique 隶属矩阵（与原版逐 bit 一致）"""
    F1 = 55
    ncliques = 39
    rf = 10
    F = F1 * 2 ** (np.arange(ncliques) / rf)          # 基频网格
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
    C_HK[-1, :] = 1                                    # 全频带兜底 clique
    return C_HK


def _inv2x2(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """批量 2x2 解析逆, A: (..., 2, 2) complex; 行列式加相对地板防奇异"""
    a, b = A[..., 0, 0], A[..., 0, 1]
    c, d = A[..., 1, 0], A[..., 1, 1]
    det = a * d - b * c
    floor = eps * (A.abs().amax(dim=(-2, -1)) ** 2 + 1.0)
    det = det + floor * (det.abs() < floor)            # |det| 过小时抬升
    inv = torch.stack([torch.stack([d, -b], dim=-1),
                       torch.stack([-c, a], dim=-1)], dim=-2)
    return inv / det[..., None, None]


class AuxIVA_ICD_RLS_harmonic_nn_Frame_Module(nn.Module):
    def __init__(self,
                 bandsize=257, n_chan=2, nsrc=2,
                 alpha=0.995, alpha2=0,
                 float_dtype=torch.float64, complex_dtype=torch.complex128,
                 use_harmonic_phi=True, r_floor_ratio=0.01, diag_load=1e-6):
        super().__init__()
        assert n_chan == 2 and nsrc == 2, "解析逆版为 2x2 特化 (与原部署一致)"
        self.alpha = alpha
        self.alpha2 = alpha2
        self.bandsize = bandsize
        self.n_chan = n_chan
        self.nsrc = nsrc
        self.float_dtype = float_dtype
        self.complex_dtype = complex_dtype
        self.eps = torch.finfo(float_dtype).eps
        self.use_harmonic_phi = use_harmonic_phi
        self.r_floor_ratio = r_floor_ratio
        self.diag_load = diag_load

        C_HK = calcu_harmonic_clique(nbands=self.bandsize)
        self.register_buffer("C_HK", torch.from_numpy(C_HK).to(float_dtype))
        self.cliquesize, _ = self.C_HK.shape

        # 状态: 只剩 V / W / r —— U, A, dW 不再是状态
        eyeC = torch.eye(n_chan, dtype=complex_dtype)
        V0 = 1e-6 * eyeC[None, None, None].repeat(1, bandsize, n_chan, 1, 1)
        self.register_buffer("V", V0.clone())                      # (B,F,src,M,M)
        self.register_buffer("W", eyeC[None, None].repeat(1, bandsize, 1, 1))
        self.register_buffer("r", torch.zeros((1, self.cliquesize, n_chan),
                                              dtype=float_dtype))
        self.initialized = False

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, X: torch.Tensor, Y_nn: torch.Tensor):
        """
        X:    (B, nch, F) complex   当前帧观测
        Y_nn: (B, nsrc, F) complex  NN 先验(可单通道, 自动补噪声路)
        返回: (Y_out.flip(dims=[1]), score0/score1, prior_energy) —— 与原版一致
        """
        device = X.device
        if Y_nn.shape[1] == 1:
            noise_like = X[:, 0:1] - Y_nn
            Y_nn = torch.cat([Y_nn, noise_like], dim=1)
        # capture (and later permute) the possibly-augmented NN prior as target
        target = Y_nn.clone().detach()

        X = X.to(self.complex_dtype).permute(0, 2, 1).unsqueeze(-1)  # (B,F,M,1)
        Y_nn = Y_nn.to(self.complex_dtype).permute(0, 2, 1)          # (B,F,src)
        # make target have shape (B,src,F) to match Y_out for scoring
        target = target.to(self.complex_dtype)                      # (B,src,F)

        V, W, r = self.V, self.W, self.r
        eyeC = torch.eye(self.n_chan, dtype=self.complex_dtype, device=device)

        # ---------- 1. clique 能量 r 与权重 phi ----------
        PWr = torch.abs(Y_nn) ** 2                                   # (B,F,src)
        C_HK = self.C_HK.to(device)
        r_new = C_HK @ PWr / torch.sum(C_HK, dim=-1, keepdim=True)   # (B,H,src)
        if not self.initialized:
            r = r_new
        else:
            r = self.alpha2 * r + (1 - self.alpha2) * r_new
        # 相对地板: 压制静音/失配 clique 的 1/sqrt(r) 爆炸与求和稀释
        r_floor = self.r_floor_ratio * r.amax(dim=1, keepdim=True)
        r = torch.clamp(r, min=torch.clamp(r_floor, min=self.eps))

        inv_sqrt_r = 0.5 / torch.sqrt(r)                             # (B,H,src)
        
        phi = torch.einsum("hf,bhn->bfn", C_HK, inv_sqrt_r)
        p = (1 - self.alpha) * phi                                   # (B,F,src)

        # ---------- 2. 加权协方差 RLS 更新 ----------
        CovX = torch.einsum("bfi,bfj->bfij", X.squeeze(-1), X.squeeze(-1).conj())
        V = self.alpha * V + p[..., None, None] * CovX[:, :, None, :, :]

        # ---------- 3. IP 更新 (逐行, 解析逆, 无递推) ----------
        for ch in range(self.n_chan):
            V_ch = V[:, :, ch]                                       # (B,F,M,M)
            # 相对对角加载, 保证可逆
            load = self.diag_load * torch.diagonal(V_ch, dim1=-2, dim2=-1
                                                   ).real.sum(-1, keepdim=True)
            V_reg = V_ch + load[..., None] * eyeC
            # w = (W V_ch)^{-1} e_ch : 取逆矩阵第 ch 列
            w = _inv2x2(W @ V_reg)[..., :, ch]                       # (B,F,M)
            # 归一化 w^H V w = 1
            normCoef = torch.einsum("bfi,bfij,bfj->bf",
                                    w.conj(), V_ch, w).real
            w = w / torch.sqrt(torch.clamp(normCoef, min=self.eps))[..., None]
            W = W.clone()
            W[:, :, ch, :] = w.conj()

        # ---------- 4. 有限性兜底 (isfinite 覆盖 nan+inf; W/V 同步消毒) ----------
        bad = ~torch.isfinite(torch.view_as_real(W)).flatten(2).all(dim=2)  # (B,F)
        if bad.any():
            W[bad] = eyeC
            Vd = torch.diagonal(V, dim1=-2, dim2=-1).real.clamp(min=1e-8)
            V[bad] = torch.diag_embed(Vd[bad].to(self.complex_dtype))

        # ---------- 5. Back-projection (A = W^{-1} 解析现算, 零漂移) ----------
        A = _inv2x2(W)
        Wbp = torch.diag_embed(torch.diagonal(A, dim1=2, dim2=3)) @ W
        Y_out = (Wbp @ X).squeeze(-1).permute(0, 2, 1)               # (B,src,F)

        # ---------- 6. 置换对齐 (逻辑与原版一致, 但去掉 .item() 支持 batch) ----------
        
        score0 = torch.abs(torch.sum(Y_out[:, 0, :125] * target[:, 0, :125].conj(), dim=-1))
        score1 = torch.abs(torch.sum(Y_out[:, 1, :125] * target[:, 1, :125].conj(), dim=-1))
        prior_energy = torch.sum(torch.abs(target) ** 2, dim=(-1, -2))
        swap = (score0 / torch.clamp(score1, min=self.eps) > 1) & \
               (prior_energy > 1e-2)                                 # (B,) 或 (B,1)
        swap = swap.reshape(-1)
        if swap.any():
            Y_out[swap] = Y_out[swap][:, [1, 0], :]

        # ---------- 7. 提交状态 ----------
        self.V = V
        self.W = W
        self.r = r
        self.initialized = True

        return Y_out.flip(dims=[1]), score0 / torch.clamp(score1, min=self.eps), prior_energy
