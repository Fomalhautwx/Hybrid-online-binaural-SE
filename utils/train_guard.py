"""
train_guard.py —— 训练循环 NaN 定位

核心思想：loss 变 NaN 并「持续为 NaN」，几乎必然是参数已被污染。
参数一旦是 NaN，之后每个 batch 都是 NaN，与输入无关。
所以要做的不是排查第 45 个 batch，而是**在参数被写坏之前拦下那一步**。

拦截点有三个，按时间顺序：
    loss 非有限  →  梯度非有限  →  参数非有限
只要在 optimizer.step() 之前查前两个，参数就永远不会坏，
坏 batch 也能原样落盘，供离线用 anomaly detection 复现。

用法：

    guard = TrainGuard(model, optimizer, dump_dir="train_dumps")

    for step, batch in enumerate(loader):
        loss = criterion(model(batch["x"]), batch["y"])
        if not guard.step(loss, batch=batch, step=step):
            continue          # 已跳过这一步，参数未被污染
        # guard.step 内部已完成 backward / clip / optimizer.step()

    guard.summary()
"""

import os, math, statistics, torch


# ---------------------------------------------------------------------------

def first_nonfinite_param(model):
    """返回首个含非有限值的参数名，没有则返回 None。"""
    for n, p in model.named_parameters():
        if p is not None and not torch.isfinite(p).all():
            return n
    for n, b in model.named_buffers():
        if b is not None and b.is_floating_point() and not torch.isfinite(b).all():
            return f"(buffer) {n}"
    return None


def first_nonfinite_grad(model):
    """返回首个含非有限梯度的参数名及其梯度范数。"""
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            g = p.grad
            n_nan = int(torch.isnan(g).sum())
            n_inf = int(torch.isinf(g).sum())
            return n, n_nan, n_inf
    return None


def grad_norm_table(model, top=8):
    """按梯度范数排序，看是哪一层在爆。"""
    rows = []
    for n, p in model.named_parameters():
        if p.grad is not None:
            rows.append((float(p.grad.detach().norm()), n))
    rows.sort(reverse=True)
    return rows[:top]


# ---------------------------------------------------------------------------

class TrainGuard:
    def __init__(self, model, optimizer, dump_dir="train_dumps",
                 max_grad_norm=5.0, spike_ratio=50.0, window=100,
                 skip_bad_step=True, verbose=True):
        """
        max_grad_norm : 梯度裁剪阈值；None 表示不裁剪
        spike_ratio   : loss 超过近期中位数的多少倍算「尖峰」（尖峰虽有限，
                        但往往是梯度爆炸的前兆，值得单独记录）
        skip_bad_step : True 时遇到非有限 loss/梯度就跳过该步（参数不受污染）
        """
        self.model, self.opt = model, optimizer
        self.dump_dir = dump_dir
        self.max_grad_norm = max_grad_norm
        self.spike_ratio = spike_ratio
        self.window = window
        self.skip = skip_bad_step
        self.verbose = verbose

        self.hist = []
        self.n_bad_loss = 0
        self.n_bad_grad = 0
        self.n_spike = 0
        self.n_ok = 0
        self.events = []

    # ----------------------------------------------------------------------

    def _dump(self, tag, step, batch, extra=None):
        os.makedirs(self.dump_dir, exist_ok=True)
        path = os.path.join(self.dump_dir, f"step{step:06d}_{tag}.pt")
        blob = {"tag": tag, "step": step, "extra": extra or {}}
        if batch is not None:
            try:
                blob["batch"] = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                                 for k, v in batch.items()} \
                    if isinstance(batch, dict) else batch
            except Exception as e:
                blob["batch_error"] = repr(e)
        torch.save(blob, path)
        return path

    def _say(self, msg):
        if self.verbose:
            print(f"[train_guard] {msg}", flush=True)

    # ----------------------------------------------------------------------

    def step(self, loss, batch=None, step=-1):
        """
        返回 True 表示这一步正常完成（已 backward + step）；
        返回 False 表示被拦下，参数未被修改，调用方应 continue。
        """
        lv = float(loss.detach()) if loss.numel() == 1 else float('nan')

        # —— 拦截点 0：参数是否已经坏了（说明之前某步漏网） ——
        bad_p = first_nonfinite_param(self.model)
        if bad_p is not None:
            self.events.append((step, "params_already_bad", bad_p))
            self._say(f"step {step}: 参数已含非有限值，首个是 [{bad_p}]。"
                      f"损坏发生在更早的步骤——请检查是否有绕过 guard 的 optimizer.step()，"
                      f"或从最近的 checkpoint 重启并全程挂上 guard。")
            return False

        # —— 拦截点 1：loss ——
        if not math.isfinite(lv):
            self.n_bad_loss += 1
            p = self._dump("bad_loss", step, batch, {"loss": lv})
            self.events.append((step, "bad_loss", lv))
            self._say(f"step {step}: loss={lv} 非有限。前向出问题，梯度还没算。"
                      f"→ 已存 {p}。下一步用 replay_with_anomaly() 复现该 batch。")
            if self.skip:
                self.opt.zero_grad(set_to_none=True)
                return False

        # loss 尖峰（有限但异常大）
        if self.hist and math.isfinite(lv):
            med = statistics.median(self.hist[-self.window:])
            if med > 0 and lv > self.spike_ratio * med:
                self.n_spike += 1
                self.events.append((step, "loss_spike", lv))
                self._say(f"step {step}: loss 尖峰 {lv:.4g}（近期中位数 {med:.4g}，"
                          f"{lv/med:.0f}×）。梯度爆炸常从这里开始。")
                self._dump("loss_spike", step, batch, {"loss": lv, "median": med})

        # —— backward ——
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        # —— 拦截点 2：梯度 ——
        bad_g = first_nonfinite_grad(self.model)
        if bad_g is not None:
            name, n_nan, n_inf = bad_g
            self.n_bad_grad += 1
            tbl = grad_norm_table(self.model)
            p = self._dump("bad_grad", step, batch,
                           {"loss": lv, "param": name, "n_nan": n_nan,
                            "n_inf": n_inf, "grad_norms": tbl})
            self.events.append((step, "bad_grad", name))
            self._say(f"step {step}: 梯度非有限。首个是 [{name}] "
                      f"(nan={n_nan}, inf={n_inf})，而 loss={lv:.6g} 是有限的 → "
                      f"损失函数里存在前向有限、反向发散的点（幂次<1 的开方、log、除法）。"
                      f"→ 已存 {p}")
            if self.verbose:
                self._say("  梯度范数最大的几层：" +
                          ", ".join(f"{n}={g:.3g}" for g, n in tbl[:4]))
            if self.skip:
                self.opt.zero_grad(set_to_none=True)
                return False

        # —— 裁剪 ——
        if self.max_grad_norm is not None:
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                self.max_grad_norm)
            # 注意：total_norm 为 nan 时 clip_coef 也是 nan，裁剪救不了，会把梯度全变 nan
            if not torch.isfinite(gn):
                self.n_bad_grad += 1
                self.events.append((step, "bad_grad_norm", float(gn)))
                self._say(f"step {step}: 梯度总范数 {float(gn)} 非有限，裁剪无效，跳过。")
                self.opt.zero_grad(set_to_none=True)
                return False

        self.opt.step()

        # —— 拦截点 3：参数（兜底，正常不该触发） ——
        bad_p = first_nonfinite_param(self.model)
        if bad_p is not None:
            self.events.append((step, "params_broken_by_step", bad_p))
            self._say(f"step {step}: optimizer.step() 之后参数变成非有限 [{bad_p}]。"
                      f"梯度检查通过却仍损坏，通常是优化器内部状态（Adam 的 "
                      f"exp_avg_sq）已被更早的一次爆炸污染 → 需要重建 optimizer 或从 "
                      f"checkpoint 恢复。")
            return False

        self.n_ok += 1
        if math.isfinite(lv):
            self.hist.append(lv)
        return True

    # ----------------------------------------------------------------------

    def summary(self):
        print("=" * 66)
        print("train_guard 统计")
        print(f"  正常步数        {self.n_ok}")
        print(f"  loss 非有限     {self.n_bad_loss}")
        print(f"  梯度非有限      {self.n_bad_grad}")
        print(f"  loss 尖峰       {self.n_spike}")
        if self.events:
            print("  事件（前 20 条）：")
            for e in self.events[:20]:
                print(f"    step {e[0]:>7}  {e[1]:<24} {e[2]}")
        print("=" * 66)


# ---------------------------------------------------------------------------

def replay_with_anomaly(closure):
    """
    用 anomaly detection 复现被 guard 抓下的那个 batch，
    直接报出「哪一个前向算子」产生了非有限梯度。慢，但只跑一次。

        d = torch.load("train_dumps/step000044_bad_grad.pt")
        replay_with_anomaly(lambda: criterion(model(d["batch"]["x"]),
                                              d["batch"]["y"]))
    """
    with torch.autograd.set_detect_anomaly(True):
        loss = closure()
        loss.backward()
    return loss


# ---------------------------------------------------------------------------
# 常见「前向有限、反向发散」写法的安全替代
# ---------------------------------------------------------------------------

EPS = 1e-8

def safe_abs(z, eps=EPS):
    """复数取模。直接 z.abs() 在 z=0 处梯度不稳，且 abs()**p (p<1) 必然 inf。"""
    if torch.is_complex(z):
        return (z.real ** 2 + z.imag ** 2 + eps).sqrt()
    return (z ** 2 + eps).sqrt()

def safe_pow(x, p, eps=EPS):
    """幂律压缩。x**0.3 在 x=0 处导数是 inf —— 语音增强压缩谱损失的头号坑。"""
    return (x.clamp(min=0) + eps) ** p

def safe_log(x, eps=EPS):
    return torch.log(x.clamp(min=0) + eps)

def compressed_spectral_loss(est, ref, p=0.3, eps=EPS, w_phase=0.3):
    """
    压缩谱 MSE 的安全版本（幅度项 + 复数项）。
    est / ref: 复数谱，任意形状。
    """
    ea, ra = safe_abs(est, eps), safe_abs(ref, eps)
    ec, rc = safe_pow(ea, p, 0.0), safe_pow(ra, p, 0.0)
    mag = ((ec - rc) ** 2).mean()
    # 复数项：用压缩幅度乘单位相位，单位相位也要防 0
    eu = est / ea.clamp(min=eps)
    ru = ref / ra.clamp(min=eps)
    cpx = ((ec * eu - rc * ru).abs() ** 2).mean()
    return mag + w_phase * cpx


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # 复刻真实场景：掩蔽型网络 est = mask * X。
    # 输入频谱含精确为零的频点（静音段 / 带限 / padding）→ est 恰好为 0
    # → est.abs()**0.3 的导数是 0.3*x^(-0.7) → inf
    class MaskNet(torch.nn.Module):
        def __init__(s_):
            super().__init__()
            s_.f = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(),
                                       torch.nn.Linear(64, 32), torch.nn.Sigmoid())
        def forward(s_, X):
            m = s_.f(X.abs())
            return m * X

    def make_batch(zero_bins):
        X = (torch.randn(4, 32) + 1j * torch.randn(4, 32)).to(torch.complex64)
        if zero_bins:
            X[:, 20:26] = 0                      # 该频段精确为零
        ref = (torch.randn(4, 32) + 1j * torch.randn(4, 32)).to(torch.complex64)
        return X, ref

    print(">>> 不安全写法：est.abs()**0.3，第 5 步喂入含全零频点的样本")
    net = MaskNet(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    guard = TrainGuard(net, opt, dump_dir="/tmp/tg", max_grad_norm=5.0)
    for s_i in range(9):
        X, ref = make_batch(zero_bins=(s_i == 5))
        est = net(X)
        loss = ((est.abs() ** 0.3 - ref.abs() ** 0.3) ** 2).mean()
        ok = guard.step(loss, batch={"X": X, "ref": ref}, step=s_i)
        if not ok:
            print(f"    → 已拦下，参数是否仍然干净: "
                  f"{first_nonfinite_param(net) is None}")
    guard.summary()

    print("\n>>> 若不拦截，看参数会怎样（同样的第 5 步）")
    net2 = MaskNet(); opt2 = torch.optim.Adam(net2.parameters(), 1e-3)
    for s_i in range(9):
        X, ref = make_batch(zero_bins=(s_i == 5))
        est = net2(X)
        loss = ((est.abs() ** 0.3 - ref.abs() ** 0.3) ** 2).mean()
        opt2.zero_grad(); loss.backward(); opt2.step()
        bad = first_nonfinite_param(net2)
        print(f"    step {s_i}  loss={float(loss):>10.4g}  首个坏参数={bad}")

    print("\n>>> 安全写法：safe_pow(safe_abs(...), 0.3)")
    net3 = MaskNet(); g3 = TrainGuard(net3, torch.optim.Adam(net3.parameters(), 1e-3),
                                      dump_dir="/tmp/tg3", verbose=False)
    for s_i in range(9):
        X, ref = make_batch(zero_bins=(s_i == 5))
        loss = compressed_spectral_loss(net3(X), ref)
        g3.step(loss, step=s_i)
    g3.summary()
