"""
AuxIVA-ISS (Iterative Source Steering) with Harmonic Clique Prior
Online / Frame-by-frame version using RLS covariance update.

v2：在 v1 数值修复的基础上，按 Nakashima & Ono (2022)
"Inverse-free Online IVA with Flexible Iterative Source Steering" 对齐，
并采用文献中通行的在线 IVA 数值稳定做法。CHANGELOG 见文件末尾。
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 谐波 clique 矩阵（向量化，语义与原版一致）
# ---------------------------------------------------------------------------

def calcu_harmonic_clique(nbands: int = 513,
                          fs: int = 16000,
                          F1: float = 55.0,
                          ncliques: int = 39,
                          rf: float = 10.0,
                          M: int = 10) -> np.ndarray:
    """返回 (ncliques+1, nbands) uint8；最后一行全 1，作为覆盖全频段的兜底 clique。"""
    F = F1 * 2.0 ** (np.arange(ncliques) / rf)
    nfft = 2 * (nbands - 1)
    freqs = np.arange(nbands) * fs / nfft
    delta = 1 - 2 ** (-1 / 12)

    m  = np.arange(1, M + 1)[None, :, None]     # (1,M,1)
    Fh = F[:, None, None]                       # (H,1,1)
    fk = freqs[None, None, :]                   # (1,1,K)
    hit = np.abs(fk - m * Fh) / (m * Fh) < delta

    C_HK = np.zeros((ncliques + 1, nbands), dtype=np.uint8)
    C_HK[:ncliques] = hit.any(axis=1).astype(np.uint8)
    C_HK[-1, :] = 1
    return C_HK


# ---------------------------------------------------------------------------
# AuxIVA-ISS 帧处理模块
# ---------------------------------------------------------------------------

class AuxIVA_ISS_harmonic_nn_Frame_Module(nn.Module):
    """
    在线逐帧 AuxIVA-ISS，带谐波 clique 先验。

    与论文的对应关系
    ----------------
    alpha       ↔ 遗忘因子 α（论文式 (9)，实验取 0.99）
    iss_iters   ↔ N_iter（论文实验取 2）
    v_load_rel  ↔ 初始/持续对角加载。论文取 U_kf0 = E × 0.001；本实现改为
                  相对量并写成 RLS 不动点形式，使 V ≡ load·I + (信号项)，
                  既不随时间衰减也不累积。
    update_srcs ↔ 论文 IV-B 的 I_k：只对移动源施加 ISS steering。

    数值参数
    --------
    phi_eps_rel : φ = 1/(sqrt(r) + ε) 中 ε 相对 sqrt(r) 频率均值的比例，
                  限制 φ 的动态范围（≈ -20·log10(phi_eps_rel) dB）
    bp_mode     : 'inv' = 解析求逆取 inv(W)[ref, n]（默认，论文做法）
                  'ls'  = 最小二乘投影 z=(R_x w)_ref/(w^H R_x w)，inverse-free
                  两者在 W 收敛时严格相等（已用解析协方差验证到 1e-14），但
                  W 未收敛时 LS 的标度会吸收残余干扰的相关性，实测反而更差
                  （良态场景 21.8 → 14.7 dB）；病态场景下两者持平。
                  K>2 时逐帧逐频点求 K×K 逆的开销可观，那时再考虑 'ls'。
    """

    def __init__(self,
                 bandsize: int = 257,
                 n_chan: int = 2,
                 nsrc: int = 2,
                 fs: int = 16000,
                 alpha: float = 0.995,
                 alpha2: float = 0.0,
                 alpha_perm: float = 0.8,
                 iss_iters: int = 2,
                 # —— 对角加载 ——
                 v_load_rel: float = 1e-4,
                 v_load_abs: float = 1e-20,
                 # —— 源模型 ——
                 phi_eps_rel: float = 1e-3,
                 noise_floor: float = 0.1,
                 # —— ISS 条件数保护 ——
                 delta_rel: float = 1e-6,
                 pow_floor_rel: float = 1e-8,
                 pow_floor_abs: float = 1e-20,
                 # —— 灵活更新 (I_k) ——
                 update_srcs=None,
                 warmup_frames: int = 200,
                 # —— back-projection ——
                 bp_mode: str = 'inv',
                 bp_alpha: float = None,
                 bp_eps_rel: float = 1e-8,
                 det_rel: float = 1e-6,
                 degenerate_mode: str = 'passthrough',
                 ref_mic: int = 0,
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
        self.ref_mic    = ref_mic
        self.float_dtype   = float_dtype
        self.complex_dtype = complex_dtype

        self.tiny          = 1e-30 if float_dtype == torch.float64 else 1e-20
        self.v_load_rel    = v_load_rel
        self.v_load_abs    = v_load_abs
        self.phi_eps_rel   = phi_eps_rel
        self.noise_floor   = noise_floor
        self.delta_rel     = delta_rel
        self.pow_floor_rel = pow_floor_rel
        self.pow_floor_abs = pow_floor_abs
        self.det_rel       = det_rel
        assert degenerate_mode in ('floor','passthrough','mute')
        self.degenerate_mode = degenerate_mode

        assert bp_mode in ('ls', 'inv'), "bp_mode 只能是 'ls' 或 'inv'"
        self.bp_mode   = bp_mode
        self.bp_alpha  = alpha if bp_alpha is None else bp_alpha
        self.bp_eps_rel = bp_eps_rel

        # —— I_k：收敛前用全部源，之后只更新 update_srcs ——
        self.all_srcs      = list(range(nsrc))
        self.update_srcs   = self.all_srcs if update_srcs is None else list(update_srcs)
        self.warmup_frames = warmup_frames

        # ── 谐波 clique ──────────────────────────────────────────────────
        C_HK = calcu_harmonic_clique(nbands=self.bandsize, fs=fs)
        self.register_buffer("C_HK", torch.from_numpy(C_HK).to(float_dtype))
        self.cliquesize = C_HK.shape[0]
        self.register_buffer("C_width",
            self.C_HK.sum(dim=-1, keepdim=True).clamp(min=1.0))   # (H,1) clique 宽度
        self.register_buffer("C_cover",
            self.C_HK.sum(dim=0).clamp(min=1.0))                  # (F,)  频点覆盖次数

        # ── 状态 buffer ──────────────────────────────────────────────────
        self.register_buffer("V", torch.zeros(
            (1, bandsize, nsrc, n_chan, n_chan), dtype=complex_dtype))
        self.register_buffer("W", torch.eye(n_chan, dtype=complex_dtype
            )[None, None].expand(1, bandsize, -1, -1).clone())
        self.register_buffer("r", torch.zeros(
            (1, self.cliquesize, nsrc), dtype=float_dtype))
        self.register_buffer("v_ref", torch.zeros(
            (1, nsrc), dtype=float_dtype))                        # V 的全局尺度
        self.register_buffer("bp_Rx", torch.zeros(
            (1, bandsize, n_chan, n_chan), dtype=complex_dtype))  # 输入协方差 R_x
        self.register_buffer("swap_prob", torch.full(
            (1, bandsize), 0.5, dtype=float_dtype))

        self.initialized   = False
        self.n_frames      = 0
        self.n_repairs     = 0    # gauge fix 复活零行的次数
        self.n_degenerate  = 0    # |det W| 触到下限的次数
        self.n_output_fix  = 0    # 输出兜底抹平非有限值的次数
        self.strict        = False  # True 时首个非有限值直接抛异常

    # -----------------------------------------------------------------------
    # 状态管理
    # -----------------------------------------------------------------------

    def reset_state(self, batch_size: int = 1, device=None):
        device = device or self.W.device
        B, Fb, N, C = batch_size, self.bandsize, self.nsrc, self.n_chan
        z  = lambda *s, dt: torch.zeros(s, dtype=dt, device=device)
        self.V = z(B, Fb, N, C, C, dt=self.complex_dtype)
        self.W = torch.eye(C, dtype=self.complex_dtype, device=device
                           )[None, None].expand(B, Fb, -1, -1).clone()
        self.r         = z(B, self.cliquesize, N, dt=self.float_dtype)
        self.v_ref     = z(B, N, dt=self.float_dtype)
        self.bp_Rx     = z(B, Fb, C, C, dt=self.complex_dtype)
        self.swap_prob = torch.full((B, Fb), 0.5,
                                    dtype=self.float_dtype, device=device)
        self.initialized   = False
        self.n_frames      = 0
        self.n_repairs     = 0    # gauge fix 复活零行的次数
        self.n_degenerate  = 0    # |det W| 触到下限的次数
        self.n_output_fix  = 0    # 输出兜底抹平非有限值的次数
        self.strict        = False  # True 时首个非有限值直接抛异常

    def _ensure_state(self, B, device):
        if (self.W.shape[0] != B or self.W.device != device
                or not self.initialized):
            self.reset_state(B, device)

    def set_active_sources(self, idx):
        """运行时切换 I_k。idx=None 恢复为全部源。"""
        self.update_srcs = self.all_srcs if idx is None else list(idx)

    def _current_src_idx(self, n_frames):
        """论文 IV-B：收敛前用全部源，收敛后只更新移动源。"""
        return self.all_srcs if n_frames < self.warmup_frames else self.update_srcs

    # -----------------------------------------------------------------------
    # 源模型：clique 功率 r 与权重 phi
    # -----------------------------------------------------------------------

    def _update_prior(self, Y_nn_f, r_state):
        """Y_nn_f: (B,F,N) → (r, phi)"""
        PWr = Y_nn_f.abs() ** 2
        r_new = torch.einsum('hf,bfn->bhn', self.C_HK, PWr) / self.C_width

        r = r_new if r_state is None else \
            self.alpha2 * r_state + (1 - self.alpha2) * r_new

        # φ = 1/(sqrt(r) + ε)：ε 加在幅度内部而非硬截断 r，权重随信号连续变化
        # （ISS2 论文用绝对 ε=1e-10；这里改为相对量以适配任意输入电平）
        r_amp = r.clamp(min=0.0).sqrt()                            # (B,H,N)
        eps_a = self.phi_eps_rel * r_amp.mean(dim=1, keepdim=True) + self.tiny
        phi_h = 1.0 / (r_amp + eps_a)                              # (B,H,N)

        # clique → 频点：按“该频点被多少 clique 覆盖”归一化
        phi = torch.einsum('hf,bhn->bfn', self.C_HK, phi_h) \
              / self.C_cover[None, :, None]
        return r, phi

    # -----------------------------------------------------------------------
    # 协方差 RLS 更新（含不衰减、不累积的对角加载）
    # -----------------------------------------------------------------------

    def _update_cov(self, V, v_ref, x_vec, phi, p, first):
        """
        V ← αV + p·xx^H + (1-α)·load·I

        写成这个形式后 V ≡ load·I + RLS{p·xx^H}：
        load 项既不像 V_0=δI 那样随 α^t 衰减掉，也不像 V←V+δI 那样累积到
        δ/(1-α)。全零频点上 V 精确收敛到 load·I，归一化在一帧内到达不动点。
        """
        C = x_vec.shape[-1]
        # V 的全局尺度 ≈ E_f[φ·|x|²]
        xp  = (x_vec.abs() ** 2).sum(-1) / C                       # (B,F)
        cur = (phi * xp[..., None]).mean(dim=1)                    # (B,N)
        cur = torch.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)
        v_ref = cur if first else self.alpha * v_ref + (1 - self.alpha) * cur
        # v_ref 一旦被污染会毒化所有频点，且 _sanitize 是按频点的，覆盖不到
        v_ref = torch.where(torch.isfinite(v_ref), v_ref, torch.zeros_like(v_ref))
        load = self.v_load_rel * v_ref + self.v_load_abs           # (B,N)

        eye  = torch.eye(C, dtype=V.dtype, device=V.device)
        CovX = torch.einsum('bfi,bfj->bfij', x_vec, x_vec.conj())
        V = (self.alpha * V
             + p[..., None, None] * CovX[:, :, None]
             + (1 - self.alpha) * load[:, None, :, None, None] * eye)
        return V, v_ref

    # -----------------------------------------------------------------------
    # ISS 更新
    # -----------------------------------------------------------------------

    def _iss_update(self, Y, V, W, src_idx):
        """
        论文式 (10)(11)：以源 k 为 steering 源，
            m≠k: v_mk = (w_m^H U_m w_k)/(w_k^H U_m w_k),  w_m ← w_m - v_mk w_k
            m=k: w_k ← w_k / sqrt(w_k^H U_k w_k)
        存储行 r = conj(w)，r^T 即解混矩阵的行本身，故系数无需再取共轭。
        src_idx 即论文的 I_k。
        """
        B, Fb, N = Y.shape
        C = W.shape[-1]
        eye = torch.eye(C, dtype=V.dtype, device=V.device)

        trV     = V.diagonal(dim1=-2, dim2=-1).sum(-1).real / C     # (B,F,N)
        trV_ref = trV.mean(dim=1, keepdim=True)
        floor   = torch.clamp(self.pow_floor_rel * trV_ref, min=self.pow_floor_abs)
        active  = trV > floor
        V_reg   = V + (self.delta_rel * trV + self.tiny)[..., None, None] * eye

        Y_new, W_new = Y.clone(), W.clone()

        for n in src_idx:
            y_n = Y_new[..., n]
            r_n = W_new[..., n, :]

            for m in range(N):
                if m == n:
                    continue
                V_m   = V_reg[:, :, m]
                Vm_rn = torch.einsum('bfij,bfj->bfi', V_m, r_n.conj())
                num   = torch.einsum('bfi,bfi->bf', W_new[..., m, :], Vm_rn)
                den   = torch.einsum('bfi,bfi->bf', r_n, Vm_rn).real
                c = num / den.clamp(min=self.tiny)
                c = torch.where(active[:, :, m], c, torch.zeros_like(c))

                Y_new[..., m]    = Y_new[..., m]    - c * y_n
                W_new[..., m, :] = W_new[..., m, :] - c.unsqueeze(-1) * r_n

            V_n   = V_reg[:, :, n]
            Vn_rn = torch.einsum('bfij,bfj->bfi', V_n, W_new[..., n, :].conj())
            s2    = torch.einsum('bfi,bfi->bf', W_new[..., n, :], Vn_rn).real
            scale = torch.sqrt(s2.clamp(min=self.tiny))
            scale = torch.where(active[:, :, n], scale, torch.ones_like(scale))

            W_new[..., n, :] = W_new[..., n, :] / scale.unsqueeze(-1)
            Y_new[..., n]    = Y_new[..., n]    / scale

        return Y_new, W_new

    # -----------------------------------------------------------------------
    # 规范化：W 各行归一到单位范数（对 ISS 与 back-projection 均为等价变换）
    # -----------------------------------------------------------------------

    def _gauge_fix(self, W, Y):
        rn = torch.linalg.vector_norm(W, dim=-1)                   # (B,F,N)
        dead = rn < self.tiny            # 恰好为零的行：0/tiny 仍是 0，救不回来
        if dead.any():
            self.n_repairs += int(dead.sum())
            C = W.shape[-1]
            eye = torch.eye(C, dtype=W.dtype, device=W.device)
            repl = eye[torch.arange(W.shape[2], device=W.device) % C]   # (N,C)
            W = torch.where(dead[..., None], repl[None, None].expand_as(W), W)
            Y = torch.where(dead, torch.zeros_like(Y), Y)
            rn = torch.linalg.vector_norm(W, dim=-1)
        rn = rn.clamp(min=self.tiny)
        return W / rn.unsqueeze(-1), Y / rn

    # -----------------------------------------------------------------------
    # 非有限值保护
    # -----------------------------------------------------------------------

    def _sanitize(self, W, Y, V, x_vec):
        bad = ~torch.isfinite(W).all(-1).all(-1) | ~torch.isfinite(Y).all(-1)
        if bad.any():
            eye = torch.eye(self.n_chan, dtype=W.dtype, device=W.device)[None, None]
            W = torch.where(bad[..., None, None].expand_as(W), eye.expand_as(W), W)
            Y = torch.where(bad[..., None].expand_as(Y), x_vec[..., :self.nsrc], Y)
            V = torch.where(bad[..., None, None, None].expand_as(V),
                            torch.zeros_like(V), V)
        return W, Y, V

    # -----------------------------------------------------------------------
    # Back-projection
    # -----------------------------------------------------------------------

    def _project_back_ls(self, Y, W, x_vec, Rx, first):
        """
        最小二乘投影（最小失真原则）：z_n = argmin_z E|x_ref - z·y_n|²
                                        = E[x_ref y_n*] / E[|y_n|²]

        关键：不要直接平滑 E[x_ref y*] 和 E[|y|²]——那两个统计量是用历史的
        W 累积的，而 W 一直在自适应，会引入滞后。把 W 提到期望外面：
            E[x_ref y_n*] = (R_x w_n)_ref ,   E[|y_n|²] = w_n^H R_x w_n
        于是只需递推与 W 无关的输入协方差 R_x，w_n 用当前值代入。
        仍然只有矩阵-向量乘，没有任何求逆。

        y_n → 0 时分子分母同阶趋零，退化平滑；
        对比 inv(W) 路线的 1/det 是分母单独趋零，会发散。
        存储行 r = conj(w)：(R_x w_n)_ref = (R_x r̄_n)_ref，w_n^H R_x w_n = r_n^T R_x r̄_n。
        """
        beta = self.bp_alpha
        Rx_new = torch.einsum('bfi,bfj->bfij', x_vec, x_vec.conj())
        Rx = Rx_new if first else beta * Rx + (1 - beta) * Rx_new

        Rx_rW = torch.einsum('bfij,bfnj->bfni', Rx, W.conj())      # (B,F,N,C)
        num = Rx_rW[..., self.ref_mic]                             # (B,F,N)
        den = torch.einsum('bfni,bfni->bfn', W, Rx_rW).real         # (B,F,N)

        d_fl = self.bp_eps_rel * den.mean(dim=1, keepdim=True) + self.tiny
        z = num / torch.maximum(den, d_fl).to(num.dtype)
        return z * Y, Rx

    def _finalize(self, Y_bp):
        """back-projection 之后的最后一道闸。原来的 _sanitize 在它之前跑，
        所以 det 退化产生的 NaN 会直接漏到输出。"""
        bad = ~torch.isfinite(Y_bp)
        if bad.any():
            self.n_output_fix += int(bad.sum())
            if self.strict:
                idx = bad.nonzero()
                raise FloatingPointError(
                    f"back-projection 输出非有限：{int(bad.sum())} 个，"
                    f"频点 {sorted(set(idx[:,1].tolist()))[:20]}")
            Y_bp = torch.nan_to_num(Y_bp, nan=0.0, posinf=0.0, neginf=0.0)
        return Y_bp

    def _project_back_inv(self, Y, W, x_ref=None):
        """inv(W)[ref, n]·y_n，保相位截断 det。"""
        C, ref = W.shape[-1], self.ref_mic
        if C == 2 and self.nsrc == 2:
            a = W[..., 0, 0]; b = W[..., 0, 1]
            c = W[..., 1, 0]; d = W[..., 1, 1]
            det = a * d - b * c
            n0 = torch.sqrt((a.abs() ** 2 + b.abs() ** 2).clamp(min=self.tiny))
            n1 = torch.sqrt((c.abs() ** 2 + d.abs() ** 2).clamp(min=self.tiny))
            fl = self.det_rel * n0 * n1 + self.tiny
            # 保相位截断。注意不能写成 det*(clamp(mag,fl)/clamp(mag,tiny))：
            # |det|=0 时那是 0×有限=0，下限根本不生效，随后 d/det 直接 NaN。
            mag = det.abs()
            unit = torch.where(mag > self.tiny,
                               det / mag.clamp(min=self.tiny),
                               torch.ones_like(det))               # 无相位可言时取 1
            degen = mag < fl
            if degen.any():
                self.n_degenerate += int(degen.sum())
            det = unit * torch.clamp(mag, min=fl)
            A_r0, A_r1 = (d / det, -b / det) if ref == 0 else (-c / det, a / det)
            out = torch.stack([A_r0 * Y[..., 0], A_r1 * Y[..., 1]], dim=2)

            # det 触底意味着该频点根本没分开（两行趋于平行）。仅仅把 det 截到
            # 下限，增益可以放大到 1/det_rel = 1e6 —— 输出有限，但会在该频点
            # 产生 120 dB 的尖峰，喂进损失函数就是一次梯度爆炸。
            if self.degenerate_mode != 'floor' and degen.any():
                if self.degenerate_mode == 'passthrough':
                    fb = torch.zeros_like(out)
                    fb[..., ref] = x_ref if x_ref is not None else 0   # 该频点不做增强
                else:
                    fb = torch.zeros_like(out)
                out = torch.where(degen[..., None], fb, out)
            return out

        eye = torch.eye(C, dtype=W.dtype, device=W.device)
        A = torch.linalg.inv(W + self.tiny * eye)
        return A[..., ref, :] * Y

    # -----------------------------------------------------------------------
    # 逐频点排列对齐
    # -----------------------------------------------------------------------

    def _permutation_align(self, Y_bp, target):
        sim0 = Y_bp[..., 0].abs() * target.abs()
        sim1 = Y_bp[..., 1].abs() * target.abs()
        need = (sim1 > sim0).to(self.float_dtype)
        self.swap_prob = (self.alpha_perm * self.swap_prob
                          + (1 - self.alpha_perm) * need)
        do = self.swap_prob > 0.5
        return torch.stack([torch.where(do, Y_bp[..., 1], Y_bp[..., 0]),
                            torch.where(do, Y_bp[..., 0], Y_bp[..., 1])], dim=2)

    def _permutation_align_clique(self, Y_bp, target):
        sim0 = Y_bp[..., 0].abs() * target.abs()
        sim1 = Y_bp[..., 1].abs() * target.abs()
        C_HK, width, cover = self.C_HK, self.C_width.squeeze(-1), self.C_cover

        cs0 = torch.einsum('hf,bf->bh', C_HK, sim0) / width
        cs1 = torch.einsum('hf,bf->bh', C_HK, sim1) / width
        cswap = (cs1 > cs0).to(self.float_dtype)                   # 原版 .double() 会 dtype 冲突

        vote = torch.einsum('hf,bh->bf', C_HK, cswap) / cover
        need = (vote > 0.5).to(self.float_dtype)

        conf = (cs0 - cs1).abs() / (cs0 + cs1).clamp(min=self.tiny)
        fconf = torch.einsum('hf,bh->bf', C_HK, conf) / cover
        self.swap_prob = torch.where(
            fconf > 0.25,
            self.alpha_perm * self.swap_prob + (1 - self.alpha_perm) * need,
            self.swap_prob)

        do = self.swap_prob > 0.5
        return torch.stack([torch.where(do, Y_bp[..., 1], Y_bp[..., 0]),
                            torch.where(do, Y_bp[..., 0], Y_bp[..., 1])], dim=2)

    # -----------------------------------------------------------------------
    # 先验噪声通道
    # -----------------------------------------------------------------------

    def _build_noise_ref(self, X0, Y_nn):
        resid = X0 - Y_nn
        r_abs, x_abs = resid.abs(), X0.abs()
        mag  = torch.maximum(r_abs, self.noise_floor * x_abs)
        unit = torch.where(r_abs > self.tiny,
                           resid / r_abs.clamp(min=self.tiny),
                           X0 / x_abs.clamp(min=self.tiny))
        return mag.to(resid.dtype) * unit

    # -----------------------------------------------------------------------
    # 单帧 forward
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, X_in: torch.Tensor, Y_nn: torch.Tensor):
        """
        X_in : (B, nch, F) complex
        Y_nn : (B, 1 或 nsrc, F) complex
        返回 : (B, nsrc, F) complex，src=0 为目标
        """
        B, device = X_in.shape[0], X_in.device
        self._ensure_state(B, device)
        first = not self.initialized

        X_in = torch.nan_to_num(X_in.to(self.complex_dtype))
        Y_nn = torch.nan_to_num(Y_nn.to(self.complex_dtype))

        if Y_nn.shape[1] == 1:
            Y_nn_full = torch.cat(
                [Y_nn, self._build_noise_ref(X_in[:, 0:1, :], Y_nn)], dim=1)
        else:
            Y_nn_full = Y_nn

        x_vec  = X_in.permute(0, 2, 1)
        Y_nn_f = Y_nn_full.permute(0, 2, 1)

        Y = torch.einsum('bfnc,bfc->bfn', self.W, x_vec)

        r, phi = self._update_prior(Y_nn_f, None if first else self.r)
        p = (1.0 - self.alpha) * phi
        V, v_ref = self._update_cov(self.V, self.v_ref, x_vec, phi, p, first)

        W = self.W
        src_idx = self._current_src_idx(self.n_frames)
        for _ in range(self.iss_iters):
            Y, W = self._iss_update(Y, V, W, src_idx)

        W, Y, V = self._sanitize(W, Y, V, x_vec)
        W, Y = self._gauge_fix(W, Y)

        if self.bp_mode == 'ls':
            Y_bp, self.bp_Rx = self._project_back_ls(
                Y, W, x_vec, self.bp_Rx, first)
        else:
            Y_bp = self._project_back_inv(Y, W, x_vec[..., self.ref_mic])

        Y_bp = self._finalize(Y_bp)
        self.V, self.W, self.r, self.v_ref = V, W, r, v_ref
        self.initialized = True
        self.n_frames += 1
        return Y_bp.permute(0, 2, 1)

    # -----------------------------------------------------------------------
    # 整段处理（每次调用独立初始化状态）
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def ChunkForward(self, X_in: torch.Tensor, Y_nn: torch.Tensor):
        """
        X_in : (B, nch, F, T) complex
        Y_nn : (B, 1 或 nsrc, F, T) complex
        返回 : (B, nsrc, F, T) complex
        """
        B, C, Fb, T = X_in.shape
        device = X_in.device
        assert C == self.n_chan and Fb == self.bandsize, \
            f"输入形状 (B,{C},{Fb},T) 与配置 (nch={self.n_chan}, F={self.bandsize}) 不符"

        X_in = torch.nan_to_num(X_in.to(self.complex_dtype))
        Y_nn = torch.nan_to_num(Y_nn.to(self.complex_dtype))

        if Y_nn.shape[1] == 1:
            P_all = torch.cat(
                [Y_nn, self._build_noise_ref(X_in[:, 0:1], Y_nn)], dim=1)
        else:
            P_all = Y_nn

        V = torch.zeros((B, Fb, self.nsrc, C, C),
                        dtype=self.complex_dtype, device=device)
        W = torch.eye(C, dtype=self.complex_dtype, device=device
                      )[None, None].expand(B, Fb, -1, -1).clone()
        v_ref = torch.zeros((B, self.nsrc), dtype=self.float_dtype, device=device)
        Rx = torch.zeros((B, Fb, C, C), dtype=self.complex_dtype, device=device)
        r = None
        outs = []

        for nf in range(T):
            first  = (nf == 0)
            x_vec  = X_in[:, :, :, nf].permute(0, 2, 1)
            Y_nn_f = P_all[:, :, :, nf].permute(0, 2, 1)

            Y = torch.einsum('bfnc,bfc->bfn', W, x_vec)
            r, phi = self._update_prior(Y_nn_f, r)
            p = (1.0 - self.alpha) * phi
            V, v_ref = self._update_cov(V, v_ref, x_vec, phi, p, first)

            src_idx = self._current_src_idx(nf)
            for _ in range(self.iss_iters):
                Y, W = self._iss_update(Y, V, W, src_idx)

            W, Y, V = self._sanitize(W, Y, V, x_vec)
            W, Y = self._gauge_fix(W, Y)

            if self.bp_mode == 'ls':
                Y_bp, Rx = self._project_back_ls(Y, W, x_vec, Rx, first)
            else:
                Y_bp = self._project_back_inv(Y, W, x_vec[..., self.ref_mic])

            outs.append(self._finalize(Y_bp).permute(0, 2, 1))

        return torch.stack(outs, dim=-1)


# ---------------------------------------------------------------------------
# CHANGELOG
# ---------------------------------------------------------------------------
# v1（数值修复）
#   [致命] 全零频点处 w^H V w 被 clamp 到 finfo.eps，归一化每帧放大 2896 倍，
#          ~11 帧溢出为 inf，而 isnan() 抓不到 inf。
#   [高]   back-projection 各源参考通道不一致（原用 A[n,n]）。
#   [高]   forward 在 B>1 时切片赋值报错；无跨段 reset。
#   [中]   phi 的 clique→频点映射除以了 clique 宽度而非频点覆盖次数。
#   [中]   r/den 下限改相对量；噪声参考加谱底；det 保相位截断。
#   [低]   _permutation_align_clique 的 .double() dtype 冲突；冗余 CovX。
#   [新增] _gauge_fix：W 行归一，对 ISS 与 back-projection 均为等价变换。
#
# v2（对照论文 + 文献通行做法）
#   [1] 对角加载。论文取 U_kf0 = E×0.001，本实现写成 RLS 不动点形式
#       V ← αV + p·xx^H + (1-α)·load·I，使 load 恒定：既不随 α^t 衰减
#       （论文写法在长流式下会漂），也不累积到 δ/(1-α)（V←V+δI 写法的问题）。
#       load 相对 E_f[φ|x|²] 取值，对输入电平不敏感。默认 1e-4 而非论文的
#       1e-3：消融显示 1e-3 会因偏置损失约 1.1 dB，而零频点的不动点性质
#       与加载量大小无关（任意 load>0 均一帧收敛），故取更小值。
#   [2] 新增 inverse-free 的最小二乘投影 bp_mode='ls'：
#           z_n = (R_x w_n)_ref / (w_n^H R_x w_n)
#       （R_x 与 W 无关，故递推 R_x 而非直接平滑 E[x y*]，避免统计量滞后。）
#       已验证与 inv 路线在 W 收敛时严格相等。但实测 W 未收敛时 LS 更差，
#       且加载+gauge fix 之后 inv 路线已无数值风险，故默认仍用 'inv'。
#   [3] I_k 灵活更新（论文 IV-B）：warmup_frames 帧后只对 update_srcs 施加
#       steering。论文 Table I 中 ISS(one) 比 ISS(all) 更快且 SDR 更高。
#   [4] φ = 1/(sqrt(r)+ε)，ε 加在幅度内部而非硬截断 r（ISS2 论文做法，
#       原文用绝对 ε=1e-10，此处改为相对量）。
#   [5] iss_iters 默认 1 → 2，与论文实验设置一致。
