"""
AuxIVA-ISS (Iterative Source Steering) with Harmonic Clique Prior
Online / Frame-by-frame version using RLS covariance update.

"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 谐波 clique 矩阵（与原版相同）
# ---------------------------------------------------------------------------

def calcu_harmonic_clique(nbands: int = 513, fs: int = 16000) -> np.ndarray:
    """
    返回形状 (ncliques+1, nbands) 的 uint8 矩阵。
    最后一行全 1，作为"兜底"clique（覆盖所有频点）。
    """
    F1 = 55
    ncliques = 39
    rf = 10
    F = F1 * 2 ** (np.arange(ncliques) / rf)   # 各基频

    nfft = 2 * (nbands - 1)
    freqs = np.arange(nbands) * fs / nfft

    C_HK = np.zeros((ncliques + 1, nbands), dtype=np.uint8)
    delta = 1 - 2 ** (-1 / 12)
    M = 10  # 最高谐波阶数

    for h in range(ncliques):
        for k in range(nbands):
            for m in range(1, M + 1):
                if np.abs(freqs[k] - m * F[h]) / (m * F[h]) < delta:
                    C_HK[h, k] = 1
                    break
    C_HK[-1, :] = 1
    return C_HK


# ---------------------------------------------------------------------------
# AuxIVA-ISS 帧处理模块
# ---------------------------------------------------------------------------

class AuxIVA_ISS_harmonic_nn_Frame_Module(nn.Module):
    """
    在线逐帧 AuxIVA-ISS，带谐波 clique 先验。

    参数
    ----
    bandsize   : STFT 频点数（单边，含 DC 和 Nyquist）
    n_chan     : 麦克风通道数
    nsrc       : 分离源数（通常 == n_chan == 2）
    alpha      : RLS 遗忘因子（协方差 V 的衰减）
    alpha2     : 谐波先验 r 的时间平滑系数（0 = 每帧刷新）
    alpha_perm : 排列概率的时间平滑系数（建议 0.7~0.9）
    iss_iters  : 每帧 ISS 内迭代次数（1 足够在线使用，2-3 收敛更稳）
    """

    def __init__(self,
                 bandsize: int = 257,
                 n_chan: int = 2,
                 nsrc: int = 2,
                 alpha: float = 0.995,
                 alpha2: float = 0.0,
                 alpha_perm: float = 0.8,
                 iss_iters: int = 1,
                 float_dtype=torch.float32,
                 complex_dtype=torch.complex64):
        super().__init__()

        self.alpha      = alpha
        self.alpha2     = alpha2
        self.alpha_perm = alpha_perm
        self.iss_iters  = iss_iters
        self.bandsize   = bandsize
        self.n_chan     = n_chan
        self.nsrc       = nsrc
        self.float_dtype   = float_dtype
        self.complex_dtype = complex_dtype
        self.eps = 1e-8

        # ── 谐波 clique ──────────────────────────────────────────────────
        C_HK = calcu_harmonic_clique(nbands=self.bandsize)
        self.register_buffer("C_HK",
            torch.from_numpy(C_HK).to(float_dtype))          # (H, F)
        self.cliquesize = C_HK.shape[0]

        # ── 状态 buffer ──────────────────────────────────────────────────
        # V[n] = RLS 加权协方差矩阵，对应源 n
        # shape: (1, F, nsrc, nch, nch)
        self.register_buffer("V",
            torch.zeros((1, bandsize, nsrc, n_chan, n_chan),
                        dtype=complex_dtype))

        # 分离矩阵 W: (1, F, nsrc, nch)，初始为单位阵的行向量
        self.register_buffer("W",
            torch.eye(n_chan, dtype=complex_dtype
                      )[None, None, :, :].expand(1, bandsize, -1, -1).clone())

        # 谐波先验的 clique-level 功率估计: (1, H, nsrc)
        self.register_buffer("r",
            torch.zeros((1, self.cliquesize, nsrc), dtype=float_dtype))

        # 逐频点排列翻转概率（用于时间平滑对齐）: (1, F)
        self.register_buffer("swap_prob",
            torch.full((1, bandsize), 0.5, dtype=float_dtype))

        self.initialized = False

    # -----------------------------------------------------------------------
    # 内部工具
    # -----------------------------------------------------------------------
    
    def _iss_update(self,
                    Y: torch.Tensor,
                    V: torch.Tensor,
                    W: torch.Tensor,
                    p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        执行一轮 ISS 更新，遍历所有源。
 
        ISS 核心公式（源 n）：
            V_n  <- alpha * V_n + p_n * (x x^H)     # 由外部已算好，此处接收
            d_n  = V_n y_n / (y_n^H V_n y_n)        # (B, F, nch)
            将 d_n 的第 n 个分量置 0（仅串扰项），
            然后对所有 m≠n：y_m <- y_m - d_n[m] * y_n
            等价地：y_n 本身通过归一化缩放到单位幅度意义上的能量
 
        参数形状
        --------
        Y : (B, F, nsrc)   当前分离信号
        V : (B, F, nsrc, nch, nch)  加权协方差（已由外部更新）
        W : (B, F, nsrc, nch)       分离矩阵行向量
        p : (B, F, nsrc)            ISS 步长权重（= (1-alpha)*phi）
 
        返回
        ----
        Y_new, W_new  (同形状)
        """
        B, F, N = Y.shape
        eps = self.eps
 
        Y_new = Y.clone()
        W_new = W.clone()
 
        # Scheibler ISS（rank-1 更新）。以源 n 为"转向"源，逐行更新：
        #   对 m≠n:  v_mn = (w_m^H V_m w_n) / (w_n^H V_m w_n)   ← 注意用 V_m！
        #            w_m <- w_m - v_mn * w_n
        #   对 m=n:  w_n <- w_n / sqrt(w_n^H V_n w_n)
        #
        # 存储约定：代码里 W 的行 r 满足 y = r @ x（无共轭），
        # 与数学滤波器 w = r̄（y = w^H x）相差一个共轭。
        # 双线性形式 w_a^H V w_b 用存储行计算为
        #     einsum(r_a, V, r_b.conj())
        # 数学更新 w_m <- w_m - v w_n 作用到存储行时系数取共轭：
        #     r_m <- r_m - v̄ * r_n,   y_m <- y_m - v̄ * y_n
 
        for n in range(N):
            y_n = Y_new[:, :, n]           # (B, F)
            r_n = W_new[:, :, n, :]        # (B, F, nch) 存储行
 
            for m in range(N):
                if m == n:
                    continue
                V_m = V[:, :, m, :, :]                     # (B,F,nch,nch)
                r_m = W_new[:, :, m, :]                    # (B,F,nch)
 
                # V_m r̄_n
                Vm_rn = torch.einsum('bfij,bfj->bfi',
                                     V_m, r_n.conj())      # (B,F,nch)
                # 分子 w_m^H V_m w_n = r_m^T V_m r̄_n
                num = torch.einsum('bfi,bfi->bf', r_m, Vm_rn)      # complex
                # 分母 w_n^H V_m w_n = r_n^T V_m r̄_n（Hermitian ⇒ 实数>0）
                den = torch.real(
                    torch.einsum('bfi,bfi->bf', r_n, Vm_rn)
                ).clamp(min=eps)                                    # (B,F)
 
                # 数学最优系数 v* = (w_n^H V_m w_m)/(w_n^H V_m w_n)。
                # 换算到存储行（r = w̄）：c = v̄* = r_m^T V_m r̄_n / den = num/den，
                # 恰好无需额外共轭。
                c = num / den
 
                Y_new[:, :, m]    = Y_new[:, :, m]    - c * y_n
                W_new[:, :, m, :] = W_new[:, :, m, :] - c.unsqueeze(-1) * r_n
 
            # 归一化 w_n（使 w_n^H V_n w_n = 1），注意行 n 用自己的 V_n
            V_n = V[:, :, n, :, :]
            Vn_wn_new = torch.einsum('bfij,bfj->bfi',
                                     V_n, W_new[:, :, n, :].conj())
            scale2 = torch.real(
                torch.einsum('bfi,bfi->bf',
                             W_new[:, :, n, :], Vn_wn_new)
            ).clamp(min=eps)                                           # (B,F)
            scale = torch.sqrt(scale2)                                 # (B,F)
 
            W_new[:, :, n, :] = W_new[:, :, n, :] / scale.unsqueeze(-1)
            Y_new[:, :, n]    = Y_new[:, :, n]    / scale
 
        return Y_new, W_new

    def _back_projection(self,
                         Y: torch.Tensor,
                         W: torch.Tensor,
                         X: torch.Tensor) -> torch.Tensor:
        """
        Back-projection
        对源 n，back-projection 系数为 A[:, n] = inv(W)[:, n]。
        2×2 情形下解析求逆，避免数值误差。

        Y   : (B, F, nsrc)
        W   : (B, F, nsrc, nch)   每行是一个分离滤波器
        X   : (B, F, nch, 1)      原始麦克风信号

        返回: Y_bp (B, F, nsrc)   back-projected 分离信号
        """
        B, F, N = Y.shape

        # 解析 2×2 逆：W = [[a,b],[c,d]], inv = 1/det * [[d,-b],[-c,a]]
        # W shape: (B, F, 2, 2)
        a = W[:, :, 0, 0]; b = W[:, :, 0, 1]
        c = W[:, :, 1, 0]; d = W[:, :, 1, 1]
        det_W = a * d - b * c                    # (B, F)
        det_W = torch.where(
            torch.abs(det_W) < self.eps,
            torch.ones_like(det_W) * self.eps, det_W)

        # A = inv(W), A[:,:,n,n] 是对角元素，用于 back-projection
        A00 =  d / det_W
        A11 =  a / det_W

        # back-projected: y_n_bp = A[n,n] * y_n
        Y_bp = torch.stack([
            A00 * Y[:, :, 0],
            A11 * Y[:, :, 1],
        ], dim=2)                                # (B, F, 2)

        return Y_bp

    # -----------------------------------------------------------------------
    # 逐频点排列对齐（时间平滑）
    # -----------------------------------------------------------------------

    def _permutation_align(self,
                           Y_bp: torch.Tensor,
                           target: torch.Tensor) -> torch.Tensor:
        """
        逐频点判断哪个通道更接近先验 target，做频点级 swap。
        使用时间平滑 swap_prob 防止单帧误判。
        Y_bp   : (B, F, nsrc)
        target : (B, F)         先验幅度（单目标）
        返回   : Y_aligned (B, F, nsrc)，src=0 对应目标
        """
        # 逐频点与先验的幅度乘积相似度
        sim0 = torch.abs(Y_bp[:, :, 0]) * torch.abs(target)   # (B, F)
        sim1 = torch.abs(Y_bp[:, :, 1]) * torch.abs(target)   # (B, F)

        # 1=通道1更像目标（当前需要 swap）
        need_swap_cur = (sim1 > sim0).to(self.float_dtype)     # (B, F)

        # 时间平滑
        self.swap_prob = (self.alpha_perm * self.swap_prob
                          + (1 - self.alpha_perm) * need_swap_cur)
        do_swap = self.swap_prob > 0.5                          # (B, F) bool

        # 逐频点条件交换
        y0 = torch.where(do_swap, Y_bp[:, :, 1], Y_bp[:, :, 0])
        y1 = torch.where(do_swap, Y_bp[:, :, 0], Y_bp[:, :, 1])

        return torch.stack([y0, y1], dim=2)                    # (B, F, 2)
    
    
    def _permutation_align_clique(self, Y_bp, target):
        sim0 = torch.abs(Y_bp[:, :, 0]) * torch.abs(target)  # (B, F)
        sim1 = torch.abs(Y_bp[:, :, 1]) * torch.abs(target)

        # ── 在 clique 粒度上投票 ──────────────────────────────
        # C_HK: (H, F)，把逐频点判断聚合到 clique 级别
        C_HK = self.C_HK  # (H, F)
        C_sum = C_HK.sum(dim=-1, keepdim=True).clamp(min=1)  # (H, 1)

        # clique 内 sim 均值: (B, H)
        clique_sim0 = torch.einsum('hf,bf->bh', C_HK, sim0) / C_sum.squeeze(-1)
        clique_sim1 = torch.einsum('hf,bf->bh', C_HK, sim1) / C_sum.squeeze(-1)

        # clique 级别的 swap 决策: (B, H)
        clique_swap = (clique_sim1 > clique_sim0).double()

        # 映射回频点: (B, F) — 每个频点继承其 clique 的决策
        # 最后一行 C_HK[-1] 全1，取加权平均后用多数票
        clique_vote = torch.einsum('hf,bh->bf', C_HK, clique_swap) / \
                    C_HK.sum(dim=0, keepdim=True).clamp(min=1)  # (B, F)
        need_swap_cur = (clique_vote > 0.5).float()

        # 置信度：clique 内 sim 差值
        clique_conf = ((clique_sim0 - clique_sim1).abs() / 
                    (clique_sim0 + clique_sim1 + self.eps))  # (B, H)
        freq_conf = torch.einsum('hf,bh->bf', C_HK, clique_conf) / \
                    C_HK.sum(dim=0, keepdim=True).clamp(min=1)  # (B, F)
        reliable_mask = freq_conf > 0.25

        # 有置信度门控的时间平滑
        self.swap_prob = torch.where(
            reliable_mask,
            self.alpha_perm * self.swap_prob + (1 - self.alpha_perm) * need_swap_cur,
            self.swap_prob
        )

        do_swap = self.swap_prob > 0.5
        y0 = torch.where(do_swap, Y_bp[:, :, 1], Y_bp[:, :, 0])
        y1 = torch.where(do_swap, Y_bp[:, :, 0], Y_bp[:, :, 1])
        return torch.stack([y0, y1], dim=2)

    # -----------------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def forward(self,
                X_in: torch.Tensor,
                Y_nn: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数
        ----
        X_in : (B, nch, F) complex  — 麦克风 STFT 帧
        Y_nn : (B, 1 或 nsrc, F) complex — 神经网络先验（幅度或复数均可）

        返回
        ----
        Y_out       : (B, nsrc, F) complex  — 分离+back-projected 结果，src=0 为目标                                                                                                                                  
        """
        B = X_in.shape[0]
        device = X_in.device

        # ── 先验准备 ────────────────────────────────────────────────────
        # target: (B, F) 复数，用于排列对齐
        target = Y_nn[:, 0, :].clone().detach().to(self.complex_dtype)

        # 补全噪声通道（若先验只有1路）
        if Y_nn.shape[1] == 1:
            noise_like = X_in[:, 0:1, :] - Y_nn
            Y_nn_full = torch.cat([Y_nn, noise_like], dim=1)   # (B,2,F)
            # p2 = (X_in[:, 0:1].abs()**2 - Y_nn.abs()**2).clamp(min=0)
            # floor = (0.1 * X_in[:, 0:1].abs())**2          # 10%谱底,防止全零
            # noise_like = torch.sqrt(torch.maximum(p2, floor)).to(Y_nn.dtype)  # 相位可省
            # Y_nn_full = torch.cat([Y_nn, noise_like], dim=1) 
        else:
            Y_nn_full = Y_nn

        # ── 张量重排 ────────────────────────────────────────────────────
        # X : (B, F, nch, 1)
        X = X_in.to(self.complex_dtype).permute(0, 2, 1).unsqueeze(-1)
        # Y_nn_f : (B, F, nsrc)
        Y_nn_f = Y_nn_full.to(self.complex_dtype).permute(0, 2, 1)

        V  = self.V
        W  = self.W
        r  = self.r
        C_HK = self.C_HK.to(device)                           # (H, F)

        # ── 当前分离输出（用上帧 W）────────────────────────────────────
        # Y : (B, F, nsrc)
        Y = torch.einsum('bfnc,bfc->bfn', W, X.squeeze(-1))

        # ── 谐波先验 r 更新 ────────────────────────────────────────────
        # PWr : (B, F, nsrc)  先验功率
        PWr = torch.abs(Y_nn_f) ** 2

        # C_HK : (H, F)  ->  (H, F) @ (B, F, nsrc) 需要 einsum
        # r_new : (B, H, nsrc)  = sum_k C_HK[h,k] * PWr[b,k,n] / sum_k C_HK[h,k]
        C_sum = C_HK.sum(dim=-1, keepdim=True)                # (H, 1)
        r_new = torch.einsum('hf,bfn->bhn', C_HK, PWr) / C_sum  # (B,H,nsrc)

        if not self.initialized:
            r = r_new
        else:
            r = self.alpha2 * r + (1 - self.alpha2) * r_new
        r = torch.clamp(r, min=self.eps)

        # ── ISS 权重 phi ────────────────────────────────────────────────
        # phi : (B, F, nsrc)  = sum_h 1/sqrt(r[b,h,n]) 映射回频点
        phi = torch.einsum('hf,bhn->bfn',
                           C_HK / C_sum,
                           1.0 / torch.sqrt(r))                # (B,F,nsrc)
        p = (1.0 - self.alpha) * phi                           # ISS 步长

        # ── RLS 协方差 V 更新 ───────────────────────────────────────────
        # CovX : (B, F, nch, nch)
        CovX = torch.einsum('bfic,bfjc->bfij',
                            X, X.conj()).squeeze(-1).squeeze(-1)
        # 修正：X shape=(B,F,nch,1), einsum 直接用 squeeze
        x_vec = X.squeeze(-1)                                  # (B, F, nch)
        CovX = torch.einsum('bfi,bfj->bfij',
                            x_vec, x_vec.conj())               # (B,F,nch,nch)

        # V : (1, F, nsrc, nch, nch)
        V = (self.alpha * V
             + p[:, :, :, None, None] * CovX[:, :, None, :, :])

        # ── ISS 迭代更新 ────────────────────────────────────────────────
        for _ in range(self.iss_iters):
            Y, W = self._iss_update(Y, V, W, p)

        # NaN 保护：若 W 出现 NaN，将对应频点重置为单位阵
        nan_mask = torch.isnan(W).any(dim=-1).any(dim=-1)     # (B, F)
        if nan_mask.any():
            eye = torch.eye(self.n_chan, dtype=self.complex_dtype,
                            device=device
                            ).unsqueeze(0).unsqueeze(0)        # (1,1,N,N)
            W = torch.where(
                nan_mask.unsqueeze(-1).unsqueeze(-1).expand_as(W),
                eye.expand_as(W), W)
            # 对应 Y 也重置为 X 的第一通道
            Y = torch.where(
                nan_mask.unsqueeze(-1).expand_as(Y),
                x_vec[:, :, :self.nsrc], Y)

        # ── Back-projection ─────────────────────────────────────────────
        Y_bp = self._back_projection(Y, W, X)                  # (B, F, nsrc)

        # ── 逐频点排列对齐（时间平滑）──────────────────────────────────
        #Y_aligned = self._permutation_align_clique(Y_bp, target)      # (B, F, nsrc)
        Y_aligned = Y_bp

        # ── 更新 buffer ─────────────────────────────────────────────────
        self.V = V
        self.W = W
        self.r = r
        self.initialized = True

        # 输出 (B, nsrc, F)，src=0 为目标
        Y_out = Y_aligned.permute(0, 2, 1)                     # (B, nsrc, F)

        return Y_out
    
    
    @torch.no_grad()
    def ChunkForward(self,
                X_in: torch.Tensor,
                Y_nn: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        X_in : (B, nch, F, T) complex — 整段含噪多通道 STFT
        Y_nn : (B, 1 或 nsrc, F, T) complex — 整段先验
 
        返回
        ----
        Y_out : (B, nsrc, F, T) complex — 分离结果，通道顺序与先验通道顺序一致
        """
        B, C, F, T = X_in.shape
        device = X_in.device
        assert C == self.n_chan and F == self.bandsize, \
            f"输入形状 (B,{C},{F},T) 与模块配置 (nch={self.n_chan}, F={self.bandsize}) 不符"
 
        # ── 先验补全噪声通道（整段一次完成，与单帧逐帧结果相同）─────────
        if Y_nn.shape[1] == 1:
            noise_like = X_in[:, 0:1] - Y_nn
            Y_nn_full = torch.cat([Y_nn, noise_like], dim=1)   # (B,2,F,T)
        else:
            Y_nn_full = Y_nn
 
        X_all = X_in.to(self.complex_dtype)
        P_all = Y_nn_full.to(self.complex_dtype)
 
        C_HK = self.C_HK.to(device)                            # (H, F)
        C_sum = C_HK.sum(dim=-1, keepdim=True)                 # (H, 1)
 
        # ── 每段语音独立初始化的状态（对应单帧版的 buffer）──────────────
        V = torch.zeros((B, F, self.nsrc, C, C),
                        dtype=self.complex_dtype, device=device)
        W = torch.eye(C, dtype=self.complex_dtype, device=device
                      )[None, None].expand(B, F, -1, -1).clone()  # (B,F,N,C)
        r = None                                                # 首帧直接赋值
 
        outs = []
        eye = torch.eye(C, dtype=self.complex_dtype,
                        device=device)[None, None]               # (1,1,N,C)
 
        for nf in range(T):
            # 当前帧张量重排：(B, F, nch) / (B, F, nsrc)
            x_vec = X_all[:, :, :, nf].permute(0, 2, 1)         # (B,F,nch)
            Y_nn_f = P_all[:, :, :, nf].permute(0, 2, 1)        # (B,F,nsrc)
 
            # ── 用上帧 W 得到当前分离输出 ─────────────────────────────
            Y = torch.einsum('bfnc,bfc->bfn', W, x_vec)
 
            # ── 谐波先验 r 更新 ──────────────────────────────────────
            PWr = torch.abs(Y_nn_f) ** 2                        # (B,F,nsrc)
            r_new = torch.einsum('hf,bfn->bhn', C_HK, PWr) / C_sum
            if r is None:
                r = r_new
            else:
                r = self.alpha2 * r + (1 - self.alpha2) * r_new
            r = torch.clamp(r, min=self.eps)
 
            # ── ISS 权重 phi 与步长 p ───────────────────────────────
            phi = torch.einsum('hf,bhn->bfn',
                               C_HK / C_sum, 1.0 / torch.sqrt(r))
            p = (1.0 - self.alpha) * phi                        # (B,F,nsrc)
 
            # ── RLS 协方差更新 ──────────────────────────────────────
            CovX = torch.einsum('bfi,bfj->bfij', x_vec, x_vec.conj())
            V = self.alpha * V + p[:, :, :, None, None] * CovX[:, :, None]
 
            # ── ISS 迭代 ────────────────────────────────────────────
            for _ in range(self.iss_iters):
                Y, W = self._iss_update(Y, V, W, p)
 
            # ── NaN 保护（与单帧版一致）─────────────────────────────
            nan_mask = torch.isnan(W).any(dim=-1).any(dim=-1)   # (B,F)
            W = torch.where(
                nan_mask.unsqueeze(-1).unsqueeze(-1).expand_as(W),
                eye.expand_as(W), W)
            Y = torch.where(
                    nan_mask.unsqueeze(-1).expand_as(Y),
                    x_vec[:, :, :self.nsrc], Y)
 
            # ── Back-projection ─────────────────────────────────────
            Y_bp = self._back_projection(Y, W, X_in)                  # (B,F,nsrc)
            outs.append(Y_bp.permute(0, 2, 1))                  # (B,nsrc,F)
 
        return torch.stack(outs, dim=-1)                        # (B,nsrc,F,T)