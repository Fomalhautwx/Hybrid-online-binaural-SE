"""
iva_debug.py —— AuxIVA-ISS 前向 NaN 定位工具

用法分两步：

  1) 训练循环里挂一个廉价守卫，抓到坏样本就落盘（几乎不影响速度）

        from iva_debug import guard_forward
        y = guard_forward(iva, X, P, sample_id=batch["utt_id"], dump_dir="nan_dumps")

  2) 离线对落盘的样本跑探针，定位首个非有限值

        from iva_debug import NaNProbe
        d = torch.load("nan_dumps/xxx.pt")
        NaNProbe(iva).run(d["X"], d["P"]).print()

设计要点：前向是确定性的，NaN 一定有唯一的「最早出现点」，后面全是污染。
探针在每帧每个阶段之后做有限性检查，一旦发现就停下并把上下文全部打印出来。
"""

import os, math, torch


# ---------------------------------------------------------------------------
# 阶段 1：训练循环里的廉价守卫
# ---------------------------------------------------------------------------

@torch.no_grad()
def guard_forward(model, X, P, sample_id=None, dump_dir="nan_dumps", raise_on_bad=False):
    """
    正常调用 model.ChunkForward，输出含非有限值时把输入落盘。
    只检查输出，开销可以忽略。
    """
    Y = model.ChunkForward(X, P)
    bad = ~torch.isfinite(Y)
    if not bad.any():
        return Y

    idx = bad.nonzero()
    b0 = idx[:, 0].min().item()
    t0 = idx[idx[:, 0] == b0][:, 3].min().item()
    sel = idx[(idx[:, 0] == b0) & (idx[:, 3] == t0)]
    fbins = sorted(set(sel[:, 2].tolist()))

    name = str(sample_id) if sample_id is not None else "unknown"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:80]
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"{safe}_b{b0}_t{t0}.pt")
    torch.save({"X": X[b0:b0 + 1].cpu(), "P": P[b0:b0 + 1].cpu(),
                "sample_id": name, "batch_idx": b0,
                "first_frame": t0, "first_bins": fbins,
                "n_bad_total": int(bad.sum())}, path)

    msg = (f"[iva_debug] 输出非有限：sample={name} batch_idx={b0} "
           f"首帧 t={t0} 频点数={len(fbins)} 频点范围=[{fbins[0]},{fbins[-1]}] "
           f"全序列非有限总数={int(bad.sum())} → 已存 {path}")
    if raise_on_bad:
        raise FloatingPointError(msg)
    print(msg, flush=True)
    return Y


# ---------------------------------------------------------------------------
# 阶段 2：逐帧逐阶段探针
# ---------------------------------------------------------------------------

class _Rec:
    def __init__(self):
        self.first = None          # (frame, stage, bins)
        self.frames = []           # 每帧的风险指标
        self.stage_hits = {}       # stage -> 首次出现的帧


class NaNProbe:
    """
    复刻 ChunkForward 的每一步，但调用的是 model 自己的方法，
    因此不会与实现漂移。每步之后做有限性检查。
    """

    STAGES = ["x_in", "prior_r", "prior_phi", "v_ref", "V_cov",
              "Y_iss", "W_iss", "post_sanitize", "post_gauge", "det_W", "Y_bp"]

    def __init__(self, model, stop_at_first=True):
        self.m = model
        self.stop = stop_at_first

    # ---- 工具 -------------------------------------------------------------

    @staticmethod
    def _bad_bins(t, fdim):
        """返回非有限值所在的频点下标；fdim 是频率所在的维度。"""
        bad = ~torch.isfinite(t)
        if not bad.any():
            return None
        dims = [d for d in range(t.dim()) if d != fdim]
        m = bad.any(dim=dims[0], keepdim=True) if dims else bad
        for d in dims[1:]:
            m = m.any(dim=d, keepdim=True)
        return sorted(set(m.reshape(-1).nonzero().reshape(-1).tolist()))

    def _chk(self, rec, nf, stage, tensor, fdim):
        bins = self._bad_bins(tensor, fdim)
        if bins is None:
            return False
        rec.stage_hits.setdefault(stage, nf)
        if rec.first is None:
            rec.first = (nf, stage, bins)
        return True

    # ---- 主流程 -----------------------------------------------------------

    @torch.no_grad()
    def run(self, X_in, Y_nn):
        m = self.m
        B, C, Fb, T = X_in.shape
        dev = X_in.device
        rec = _Rec()

        X_raw, P_raw = X_in, Y_nn
        X_in = torch.nan_to_num(X_in.to(m.complex_dtype))
        Y_nn = torch.nan_to_num(Y_nn.to(m.complex_dtype))
        rec.input_had_nonfinite = bool((~torch.isfinite(X_raw)).any() or
                                       (~torch.isfinite(P_raw)).any())

        if Y_nn.shape[1] == 1:
            P_all = torch.cat([Y_nn, m._build_noise_ref(X_in[:, 0:1], Y_nn)], dim=1)
        else:
            P_all = Y_nn

        V = torch.zeros((B, Fb, m.nsrc, C, C), dtype=m.complex_dtype, device=dev)
        W = torch.eye(C, dtype=m.complex_dtype, device=dev
                      )[None, None].expand(B, Fb, -1, -1).clone()
        v_ref = torch.zeros((B, m.nsrc), dtype=m.float_dtype, device=dev)
        Rx = torch.zeros((B, Fb, C, C), dtype=m.complex_dtype, device=dev)
        r = None

        for nf in range(T):
            first = (nf == 0)
            x = X_in[:, :, :, nf].permute(0, 2, 1)
            yn = P_all[:, :, :, nf].permute(0, 2, 1)
            stop = False

            stop |= self._chk(rec, nf, "x_in", x, 1)

            Y = torch.einsum('bfnc,bfc->bfn', W, x)
            r, phi = m._update_prior(yn, r)
            stop |= self._chk(rec, nf, "prior_r", r, 1)      # r 的第 1 维是 clique
            stop |= self._chk(rec, nf, "prior_phi", phi, 1)

            p = (1.0 - m.alpha) * phi
            V, v_ref = m._update_cov(V, v_ref, x, phi, p, first)
            stop |= self._chk(rec, nf, "v_ref", v_ref.unsqueeze(1), 1)
            stop |= self._chk(rec, nf, "V_cov", V, 1)

            for _ in range(m.iss_iters):
                Y, W = m._iss_update(Y, V, W, m._current_src_idx(nf))
            stop |= self._chk(rec, nf, "Y_iss", Y, 1)
            stop |= self._chk(rec, nf, "W_iss", W, 1)

            W, Y, V = m._sanitize(W, Y, V, x)
            stop |= self._chk(rec, nf, "post_sanitize", W, 1)
            W, Y = m._gauge_fix(W, Y)
            stop |= self._chk(rec, nf, "post_gauge", W, 1)

            # —— 风险指标（即使没 NaN 也记录，用来看「快出事」的趋势）——
            rn = torch.linalg.vector_norm(W, dim=-1)                  # (B,F,N)
            det = (W[..., 0, 0] * W[..., 1, 1] - W[..., 0, 1] * W[..., 1, 0]
                   ).abs() if (C == 2 and m.nsrc == 2) else torch.ones_like(rn[..., 0])
            trV = V.diagonal(dim1=-2, dim2=-1).sum(-1).real / C
            fin = lambda t: t[torch.isfinite(t)]
            rec.frames.append(dict(
                t=nf,
                det_min=float(fin(det).min()) if fin(det).numel() else float('nan'),
                det_argmin=int(det.argmin() % Fb),
                rown_min=float(fin(rn).min()) if fin(rn).numel() else float('nan'),
                phi_max=float(fin(phi).max()) if fin(phi).numel() else float('nan'),
                trV_max=float(fin(trV).max()) if fin(trV).numel() else float('nan'),
                trV_min=float(fin(trV).min()) if fin(trV).numel() else float('nan'),
                x_max=float(x.abs().max()),
            ))

            if m.bp_mode == 'ls':
                Y_bp, Rx = m._project_back_ls(Y, W, x, Rx, first)
            else:
                stop |= self._chk(rec, nf, "det_W", det, 1)
                Y_bp = m._project_back_inv(Y, W)
            stop |= self._chk(rec, nf, "Y_bp", Y_bp, 1)

            if stop and self.stop:
                rec.n_frames_run = nf + 1
                self.rec = rec
                self._context(rec, x, yn, r, phi, V, W, det)
                return self

        rec.n_frames_run = T
        self.rec = rec
        return self

    # ---- 首次故障处的上下文 ----------------------------------------------

    def _context(self, rec, x, yn, r, phi, V, W, det):
        nf, stage, bins = rec.first
        b = bins[:12]
        C = W.shape[-1]
        ctx = {}
        idx = torch.tensor(b, device=x.device)
        ctx["|x| 该频点"] = x[0].index_select(0, idx).abs().tolist()
        ctx["|先验| 该频点"] = yn[0].index_select(0, idx).abs().tolist()
        ctx["phi 该频点"] = phi[0].index_select(0, idx).tolist()
        trV = (V.diagonal(dim1=-2, dim2=-1).sum(-1).real / C)[0].index_select(0, idx)
        ctx["trace(V)/C 该频点"] = trV.tolist()
        ctx["W 行范数 该频点"] = torch.linalg.vector_norm(
            W, dim=-1)[0].index_select(0, idx).tolist()
        ctx["|det W| 该频点"] = det[0].index_select(0, idx).tolist()
        rec.context = ctx

    # ---- 报告 -------------------------------------------------------------

    def print(self, n_trend=8):
        rec = self.rec
        L = []
        L.append("=" * 72)
        L.append("AuxIVA-ISS 前向探针报告")
        L.append("=" * 72)
        if rec.input_had_nonfinite:
            L.append("⚠ 输入 X 或先验本身就含非有限值（已被 nan_to_num 抹平，但源头在上游）")

        if rec.first is None:
            L.append(f"跑完 {rec.n_frames_run} 帧，未发现非有限值。")
        else:
            nf, stage, bins = rec.first
            L.append(f"首个非有限值：第 {nf} 帧 · 阶段 [{stage}] · "
                     f"{len(bins)} 个频点，范围 [{bins[0]}, {bins[-1]}]")
            L.append(f"频点列表（前 20 个）：{bins[:20]}")
            L.append("")
            L.append("该帧、该频点上的状态：")
            for k, v in rec.context.items():
                if isinstance(v, list):
                    s = ", ".join(f"{x:.3e}" if isinstance(x, float) else str(x)
                                  for x in (v[:6] if not isinstance(v[0], list) else v[:3]))
                    L.append(f"  {k:<22} {s}")
            L.append("")
            L.append("各阶段首次出现非有限值的帧号（判断污染传播方向）：")
            for s in self.STAGES:
                if s in rec.stage_hits:
                    L.append(f"  {s:<16} t={rec.stage_hits[s]}")

        L.append("")
        L.append(f"最后 {n_trend} 帧的风险指标（看是否在缓慢劣化）：")
        L.append(f"  {'帧':>5} {'|det|min':>11} {'行范数min':>11} {'phi max':>11} "
                 f"{'trV max':>11} {'trV min':>11} {'|x|max':>10}")
        for f in rec.frames[-n_trend:]:
            L.append(f"  {f['t']:>5} {f['det_min']:>11.3e} {f['rown_min']:>11.3e} "
                     f"{f['phi_max']:>11.3e} {f['trV_max']:>11.3e} "
                     f"{f['trV_min']:>11.3e} {f['x_max']:>10.3e}")
        L.append("")
        L.append(self._diagnose())
        L.append("=" * 72)
        print("\n".join(L))
        return self

    def _diagnose(self):
        """按首个失效阶段给出最可能的原因。"""
        rec = self.rec
        if rec.first is None:
            return "结论：该样本正常。若训练中仍报错，换一个落盘样本再试。"
        stage = rec.first[1]
        last = rec.frames[-1] if rec.frames else {}
        tips = {
            "x_in":        "输入本身非有限 → 问题在 STFT / 数据加载，不在 IVA。",
            "prior_r":     "先验功率非有限 → 检查 NN 输出是否溢出。",
            "prior_phi":   "phi 溢出 → 先验在该频点整体为零，相对 eps 退化成 tiny，"
                           "1/(0+1e-20)=1e20。修法：eps 改为对全频段取参考，而非按源取。",
            "v_ref":       "全局尺度 v_ref 被污染 → 这是 per-source 标量，一旦坏掉会毒化"
                           "所有频点且 _sanitize 覆盖不到，表现为整帧大面积 NaN。",
            "V_cov":       "协方差溢出 → 检查 phi_max 与 |x|max，多半是 phi 巨大叠加大幅度输入。",
            "Y_iss":       "ISS 中输出发散 → 看 det 与行范数是否已趋零。",
            "W_iss":       "ISS 中解混矩阵发散 → den 或 s2 触到下限，检查 trV min。",
            "post_gauge":  "gauge fix 后出现 NaN → 存在恰好为零的 W 行（0/tiny 仍是 0）。",
            "det_W":       "det(W) 非有限。",
            "Y_bp":        "back-projection 出 NaN，而之前各阶段都正常 → "
                           "几乎可以确定是 det(W) 恰好为 0（两行平行或某行全零）。"
                           "保相位截断 det*(clamp(|det|,fl)/clamp(|det|,tiny)) 在 |det|=0 时"
                           "等于 0×有限=0，下限不生效。且该 NaN 在 _sanitize 之后产生，"
                           "内部状态看起来完全正常。",
        }
        head = "结论推断：" + tips.get(stage, "见上方阶段表。")
        extra = []
        if last.get("det_min", 1) < 1e-5:
            extra.append(f"|det| 已降到 {last['det_min']:.2e}，解混矩阵接近奇异。")
        if last.get("rown_min", 1) < 1e-12:
            extra.append(f"存在近乎为零的 W 行（范数 {last['rown_min']:.2e}）。")
        if last.get("phi_max", 0) > 1e12:
            extra.append(f"phi 最大值 {last['phi_max']:.2e}，先验在某些频点接近全零。")
        return head + ("\n           " + " ".join(extra) if extra else "")


# ---------------------------------------------------------------------------
# 自检：注入已知故障，确认探针能定位
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from ..models.AuxIVA_ISS_harmonic_v2 import AuxIVA_ISS_harmonic_nn_Frame_Module as V2
    torch.manual_seed(0)
    Fb, T = 64, 40
    X = ((torch.randn(1, 2, Fb, T) + 1j * torch.randn(1, 2, Fb, T)) * .05).to(torch.complex64)
    P = X[:, 0:1] * .6

    print(">>> 正常样本")
    NaNProbe(V2(bandsize=Fb)).run(X, P).print(n_trend=3)

    print("\n>>> 注入故障：第 20 帧起把 f=30 的 W 行置零（模拟 det=0）")
    m = V2(bandsize=Fb)
    orig = m._gauge_fix
    def broken(W, Y):
        W, Y = orig(W, Y)
        if getattr(m, "_tick", 0) >= 20:
            W = W.clone(); W[:, 30, 1, :] = 0
        m._tick = getattr(m, "_tick", 0) + 1
        return W, Y
    m._gauge_fix = broken
    NaNProbe(m).run(X, P).print(n_trend=3)
