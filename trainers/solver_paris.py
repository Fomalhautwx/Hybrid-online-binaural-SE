"""
Solver-style trainer for GTCRN_PARIS (WPE+IVA 前端 / GTCRN 后端 / PARIS 两阶段训练)

结构参考原 solver.py:
  - checkpoint 保存/恢复(含训练状态)
  - 验证集 plateau -> 学习率减半(回滚到最优权重) -> 早停
  - 线性 warmup / 梯度累积 / 梯度裁剪
  - TensorBoard 记录

在此基础上为 PARIS 增加:
  - PARIS 两阶段编排直接在 solver 内实现(_paris_forward), 模型只做单次前传:
      Stage 1: clean 先验(teacher forcing, 可按概率置零模拟冷启动)
      Stage 2: 训练时课程混合 cl*est1.detach() + (1-cl)*clean; 验证时纯自估计先验
  - 课程学习因子 cl_factor 的逐 epoch 线性调度 (0=纯 clean 先验 -> 1=纯自估计先验)
  - 双 loss 加权: loss = w1 * loss(stage1) + w2 * loss(stage2)
  - DDP 安全的 NaN/Inf 跳批(所有 rank 同步跳过, 避免 allreduce 挂死)
  - STFT/iSTFT 在 solver 内完成, 数据集只需产出时域波形

DDP 注意事项:
  两个 stage 都通过 self.model(...) 调用(即 DDP wrapper 的 forward), 绝不直接调
  model.module.xxx, 否则梯度同步 hooks 不会触发。"两次前传 + 单次 backward"
  要求 DDP 以 find_unused_parameters=False 构建(train_paris.py 已如此设置)。

config.yaml 需要在 trainer 段补充的键(均有默认值, 不加也能跑):
  trainer:
    max_epoch: 100
    clip_grad_norm: 5.0
    accu_grad: false
    effec_batch_size: 16
    batch_size: 8
    lr_warmup: true
    warmup_steps: 15000
    halve_patience: 5          # 验证集多少个 epoch 无提升后减半 LR
    early_stop_patience: 10    # 多少个 epoch 无提升后早停
    stage1_loss_weight: 0.25   # w1; 为 0 时 stage-1 自动在 no_grad 下运行(省显存)
    stage2_loss_weight: 0.75   # w2
    cl_start: 0.0              # 课程学习起始因子
    cl_end: 1.0                # 课程学习终止因子
    cl_start_epoch: 1          # 从第几个 epoch 开始爬坡
    cl_ramp_epochs: 30         # 爬坡持续多少个 epoch
    p_zero_prior: 0.0          # 训练时以该概率将 stage-1 的 clean 先验置零(冷启动鲁棒性)
    reset_derev_per_batch: true
"""

import os
import time

import torch
import torch.distributed as dist


class Solver(object):
    def __init__(self, config, model, optimizer, loss_func,
                 train_data, validation_data, test_data=None,
                 writer=None, rank=0, world_size=1):
        self.config = config
        self.tr_cfg = config.get("trainer", {})
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.writer = writer
        self.rank = rank
        self.world_size = world_size
        self.distributed = world_size > 1
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.print = (rank == 0)

        # ---- STFT 参数: 与模型/推理保持一致 ----
        net_cfg = config.get("network", {})
        self.n_fft = net_cfg.get("win_len", 512)
        self.hop = net_cfg.get("hop_len", 256)
        self._window = torch.hann_window(self.n_fft).to(self.device)

        # ---- 训练超参 ----
        g = self.tr_cfg.get
        self.max_epoch = g("max_epoch", 100)
        self.clip_grad_norm = g("clip_grad_norm", 5.0)
        self.accu_grad = g("accu_grad", False)
        self.effec_batch_size = g("effec_batch_size", 1)
        self.batch_size = g("batch_size", 1)
        self.lr_warmup = g("lr_warmup", False)
        self.warmup_steps = g("warmup_steps", 15000)
        self.halve_patience = g("halve_patience", 5)
        self.early_stop_patience = g("early_stop_patience", 10)
        self.w1 = g("stage1_loss_weight", 0.25)
        self.w2 = g("stage2_loss_weight", 0.75)
        self.cl_start = g("cl_start", 0.0)
        self.cl_end = g("cl_end", 1.0)
        self.cl_start_epoch = g("cl_start_epoch", 1)
        self.cl_ramp_epochs = g("cl_ramp_epochs", 30)
        self.p_zero_prior = g("p_zero_prior", 0.0)
        self.cl = self.cl_start  # 当前课程学习因子, 每个 epoch 由 train() 更新
        self.reset_derev_per_batch = g("reset_derev_per_batch", True)
        self.checkpoint_dir = self.tr_cfg.get("exp_path", "exps/paris")
        os.makedirs(f"{self.checkpoint_dir}/checkpoints", exist_ok=True)

        self.base_lr = config.get("optimizer", {}).get("lr", 1e-3)

        self._init()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _m(self):
        """取出真正的 nn.Module (兼容 DDP 包裹与否)"""
        return self.model.module if hasattr(self.model, "module") else self.model

    def _stft_mc(self, wav):
        """多通道 STFT. wav: (B, C, L) 或 (B, L) -> (B, C, F, T) complex"""
        if wav.ndim == 2:
            wav = wav.unsqueeze(0)
        B, C, L = wav.shape
        spec = torch.stft(wav.reshape(B * C, L), n_fft=self.n_fft,
                          hop_length=self.hop, win_length=self.n_fft,
                          window=self._window, center=True,
                          return_complex=True)
        F_, T_ = spec.shape[-2], spec.shape[-1]
        return spec.reshape(B, C, F_, T_)

    def _istft(self, spec, length):
        """spec: (B, 1, F, T) complex -> (B, length)"""
        spec = spec.squeeze(1)
        return torch.istft(spec, n_fft=self.n_fft, hop_length=self.hop,
                           win_length=self.n_fft, window=self._window,
                           center=True, length=length)

    def _reduce_tensor(self, tensor):
        if not self.distributed:
            return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt

    # ------------------------------------------------------------------ #
    # init / checkpoint
    # ------------------------------------------------------------------ #
    def _init(self):
        self.halving = False
        self.step_num = 1
        self.best_val_loss = float("inf")
        self.val_no_impv = 0
        self.start_epoch = 1
        self.epoch = 0
        self.nan_skips = 0

        if self.tr_cfg.get("resume", False):
            ckpt = f"{self.checkpoint_dir}/checkpoints/{self.tr_cfg.get('ckpt', 'last_checkpoint.pt')}"
            self._load_model(ckpt, load_training_stat=True)
        elif self.print:
            print("Start new training from scratch (或已由 train.py 加载 pretrained)")
        self._save_model(f"{self.checkpoint_dir}/checkpoints/last_checkpoint.pt")

    def _load_model(self, path, load_optimizer=False, load_training_stat=False):
        checkpoint = torch.load(path, map_location="cpu")
        pretrained_model = checkpoint["model"]
        state = self.model.state_dict()
        for key in state.keys():
            if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
                state[key] = pretrained_model[key]
            elif f"module.{key}" in pretrained_model and \
                    state[key].shape == pretrained_model[f"module.{key}"].shape:
                state[key] = pretrained_model[f"module.{key}"]
            elif self.print:
                print(f"{key} not loaded")
        self.model.load_state_dict(state)

        if load_optimizer or load_training_stat:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if load_training_stat:
            self.step_num = checkpoint["step_num"]
            self.best_val_loss = checkpoint["best_val_loss"]
            self.val_no_impv = checkpoint["val_no_impv"]
            self.start_epoch = checkpoint["epoch"]
            self.epoch = self.start_epoch - 1
            if self.print:
                print(f"Resume training from epoch: {self.start_epoch}")

    def _save_model(self, path):
        if not self.print:
            return
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch + 1,
            "step_num": self.step_num,
            "best_val_loss": self.best_val_loss,
            "val_no_impv": self.val_no_impv,
        }
        torch.save(checkpoint, path)

    # ------------------------------------------------------------------ #
    # schedules
    # ------------------------------------------------------------------ #
    def _adjust_lr_warmup(self):
        """线性 warmup 到 base_lr(比原 solver 的 noam 变体公式更直观)"""
        if self.step_num < self.warmup_steps:
            lr = self.base_lr * self.step_num / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        self.step_num += 1

    def _current_cl_factor(self):
        """课程学习因子: cl_start_epoch 前恒为 cl_start, 之后线性爬坡到 cl_end"""
        if self.epoch < self.cl_start_epoch:
            return self.cl_start
        ramp = min(1.0, (self.epoch - self.cl_start_epoch) / max(1, self.cl_ramp_epochs))
        return self.cl_start + ramp * (self.cl_end - self.cl_start)

    # ------------------------------------------------------------------ #
    # PARIS two-pass orchestration
    # ------------------------------------------------------------------ #
    def _paris_forward(self, spec_mix, spec_tgt, state):
        """PARIS 两阶段前传(从模型 forward 移入 solver)

        Stage 1: clean 先验(teacher forcing); 训练时可按 p_zero_prior 概率
                 将先验置零, 让网络见过"无先验"的推理冷启动工况
        Stage 2: 训练时课程混合 init2 = cl * est1.detach() + (1-cl) * clean;
                 验证时直接用自估计先验(与原模型 forward 的行为一致)

        梯度策略:
          - 先验路径始终截断: est1 进入 stage-2 前 detach
            (模型 forward_spec 内部对 init 也有一道 detach 兜底)
          - w1 > 0 时 stage-1 保留计算图, 使 w1*loss1 能训练"教师先验工况";
            w1 == 0 时 stage-1 在 no_grad 下运行, 省一半激活显存

        DDP: 两个 stage 都必须通过 self.model(...) 调用(经过 DDP wrapper),
        两次前传 + 单次 backward 需 find_unused_parameters=False。

        spec_mix: (B, C, F, T) complex
        spec_tgt: (B, 1, F, T) complex
        返回: (est2, est1), 均为 (B, 1, F, T) complex
        """
        # ---- Stage 1 先验: clean (可随机置零) ----
        init1 = spec_tgt
        if state == "train" and self.p_zero_prior > 0:
            drop = (torch.rand(init1.shape[0], 1, 1, 1, device=init1.device)
                    < self.p_zero_prior)
            init1 = torch.where(drop, torch.zeros_like(init1), init1)

        # ---- Stage 1 ----
        s1_needs_grad = (self.w1 > 0) and (state == "train") and torch.is_grad_enabled()
        if s1_needs_grad:
            est1 = self.model(spec_mix, init1)
            s1_prior = est1.detach()   # 先验路径截断
        else:
            with torch.no_grad():
                est1 = self.model(spec_mix, init1)
            s1_prior = est1

        # ---- Stage 2 先验: 课程混合(训练) / 纯自估计(验证) ----
        if state == "train":
            init2 = self.cl * s1_prior + (1.0 - self.cl) * spec_tgt
        else:
            init2 = s1_prior

        est2 = self.model(spec_mix, init2)
        return est2, est1

    # ------------------------------------------------------------------ #
    # train loop
    # ------------------------------------------------------------------ #
    def train(self):
        for self.epoch in range(self.start_epoch, self.max_epoch + 1):
            if self.distributed and hasattr(self.train_data, "sampler") and \
                    hasattr(self.train_data.sampler, "set_epoch"):
                self.train_data.sampler.set_epoch(self.epoch)

            # 设定本 epoch 的课程学习因子(由 _paris_forward 使用)
            self.cl = self._current_cl_factor()
            if self.print:
                print(f"Epoch {self.epoch} | cl_factor = {self.cl:.3f}")

            # ---- Train ----
            self.model.train()
            start = time.time()
            tr_loss, tr_l1, tr_l2 = self._run_one_epoch(self.train_data, state="train")
            tr_loss = self._reduce_tensor(tr_loss)
            if self.print:
                print(f"Train Summary | Epoch {self.epoch} | {time.time()-start:.1f}s | "
                      f"Loss {tr_loss:.4f} (s1 {tr_l1:.4f} / s2 {tr_l2:.4f}) | "
                      f"nan_skips {self.nan_skips}")

            # ---- Validation ----
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss, val_l1, val_l2 = self._run_one_epoch(self.validation_data, state="val")
            val_loss = self._reduce_tensor(val_loss)
            val_l2 = self._reduce_tensor(val_l2)
            if self.print:
                print(f"Valid Summary | Epoch {self.epoch} | {time.time()-start:.1f}s | "
                      f"Loss {val_loss:.4f} (s1 {val_l1:.4f} / s2 {val_l2:.4f})")

            # ---- 早停/减半以 stage-2 验证 loss 为准(最终关心的输出) ----
            monitored = float(val_l2)
            find_best = False
            if monitored >= self.best_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv == self.halve_patience:
                    self.halving = True
                elif self.val_no_impv >= self.early_stop_patience:
                    if self.print:
                        print(f"No improvement for {self.early_stop_patience} epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0
                self.best_val_loss = monitored
                find_best = True

            if self.halving:
                self.halving = False
                self._load_model(f"{self.checkpoint_dir}/checkpoints/last_best_checkpoint.pt",
                                 load_optimizer=True)
                if self.print:
                    print("Reload weights/optimizer from last best checkpoint")
                for pg in self.optimizer.param_groups:
                    pg["lr"] *= 0.5
                if self.print:
                    print(f"Learning rate adjusted to: {self.optimizer.param_groups[0]['lr']:.6f}")

            # ---- Logging & checkpoint ----
            if self.print and self.writer is not None:
                self.writer.add_scalar("Train/loss", tr_loss, self.epoch)
                self.writer.add_scalar("Train/loss_stage1", tr_l1, self.epoch)
                self.writer.add_scalar("Train/loss_stage2", tr_l2, self.epoch)
                self.writer.add_scalar("Val/loss", val_loss, self.epoch)
                self.writer.add_scalar("Val/loss_stage2", val_l2, self.epoch)
                self.writer.add_scalar("Schedule/cl_factor", self.cl, self.epoch)
                self.writer.add_scalar("Schedule/lr", self.optimizer.param_groups[0]["lr"], self.epoch)

            self._save_model(f"{self.checkpoint_dir}/checkpoints/last_checkpoint.pt")
            if find_best:
                self._save_model(f"{self.checkpoint_dir}/checkpoints/last_best_checkpoint.pt")
                if self.print:
                    print("Found new best model, dict saved")

    # ------------------------------------------------------------------ #
    # one epoch
    # ------------------------------------------------------------------ #
    def _run_one_epoch(self, data_loader, state="train"):
        total, total1, total2 = 0.0, 0.0, 0.0
        accu_count = 0
        n_batches = 0
        self.optimizer.zero_grad()

        for i, batch in enumerate(data_loader):
            # ======== 适配点 1: 数据格式 ========
            # 假设 dataset 产出 (a_mix, a_tgt[, ...]):
            #   a_mix: (B, C, L)  多通道混合波形
            #   a_tgt: (B, L) 或 (B, 1, L)  参考通道目标干净波形
            # 若你的 datasets_new.Dataloader 输出不同, 只需改这几行
            a_mix, a_tgt = batch[0], batch[1].unsqueeze(0)  # (B, C, L), (B, L)
            a_mix = a_mix.to(self.device, non_blocking=True)
            a_tgt = a_tgt.to(self.device, non_blocking=True)
            if a_tgt.ndim == 3:
                a_tgt = a_tgt.squeeze(1)
            L = a_tgt.shape[-1]

            spec_mix = self._stft_mc(a_mix)                 # (B, C, F, T) complex
            spec_tgt = self._stft_mc(a_tgt)                 # (B, 1, F, T) complex

            # derev 内部状态按 batch 重置(样本间相互独立)
            if self.reset_derev_per_batch:
                if state == "train":
                    self._m().train_reset_buffer()
                else:
                    self._m().validate_reset_buffer()

            est2_spec, est1_spec = self._paris_forward(spec_mix, spec_tgt, state)

            # ======== 适配点 2: loss 域 ========
            # 这里假设 HybridLoss 接收 (est_wav, tgt_wav);
            # 若它接收复数谱, 直接把 est*_spec / spec_tgt 传进去并删掉 iSTFT
            est2 = self._istft(est2_spec, L)
            loss2 = self.loss_func(est2, a_tgt)
            if self.w1 > 0:
                est1 = self._istft(est1_spec, L)
                loss1 = self.loss_func(est1, a_tgt)
            else:
                loss1 = torch.zeros((), device=self.device)
            loss = self.w1 * loss1 + self.w2 * loss2

            # ---- DDP 安全的 NaN/Inf 跳批: 所有 rank 一致决策, 避免 allreduce 挂死 ----
            finite = torch.isfinite(loss.detach()).float()
            if self.distributed:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if finite.item() < 0.5:
                self.nan_skips += 1
                self.optimizer.zero_grad()
                continue

            if state == "train":
                if self.accu_grad:
                    accu_count += 1
                    scale = self.effec_batch_size / self.batch_size
                    (loss / scale).backward()
                    if accu_count == int(scale):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                        if self.lr_warmup:
                            self._adjust_lr_warmup()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        accu_count = 0
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    if self.lr_warmup:
                        self._adjust_lr_warmup()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total += loss.detach()
            total1 += loss1.detach()
            total2 += loss2.detach()
            n_batches += 1

        n = max(1, n_batches)
        return total / n, total1 / n, total2 / n

    # ------------------------------------------------------------------ #
    # evaluate
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, data_loader=None):
        """加载最优权重, 在测试集上报告 SI-SNRi (可选 PESQ/STOI)"""
        data_loader = data_loader or self.test_data or self.validation_data
        self._load_model(f"{self.checkpoint_dir}/checkpoints/last_best_checkpoint.pt")
        self.model.eval()

        try:
            from pesq import pesq as pesq_fn
        except ImportError:
            pesq_fn = None
        try:
            from pystoi import stoi as stoi_fn
        except ImportError:
            stoi_fn = None
        sr = self.config.get("audio_sr", 16000)

        def si_snr(est, ref, eps=1e-8):
            est = est - est.mean(dim=-1, keepdim=True)
            ref = ref - ref.mean(dim=-1, keepdim=True)
            proj = (torch.sum(est * ref, -1, keepdim=True) /
                    (torch.sum(ref ** 2, -1, keepdim=True) + eps)) * ref
            noise = est - proj
            return 10 * torch.log10((proj.pow(2).sum(-1) + eps) /
                                    (noise.pow(2).sum(-1) + eps))

        avg_sisnri, avg_pesqi, avg_stoii, count = 0.0, 0.0, 0.0, 0
        for batch in data_loader:
            a_mix, a_tgt = batch[0].to(self.device), batch[1].to(self.device)
            if a_tgt.ndim == 3:
                a_tgt = a_tgt.squeeze(1)
            L = a_tgt.shape[-1]
            spec_mix = self._stft_mc(a_mix)
            spec_tgt = self._stft_mc(a_tgt)
            self._m().validate_reset_buffer()
            est2_spec, _ = self._paris_forward(spec_mix, spec_tgt, state="val")
            est = self._istft(est2_spec, L)

            mix_ref = a_mix[:, 0] if a_mix.ndim == 3 else a_mix
            sisnri = (si_snr(est, a_tgt) - si_snr(mix_ref, a_tgt)).mean()
            avg_sisnri += float(sisnri)

            if pesq_fn is not None or stoi_fn is not None:
                e = est[0].cpu().numpy()
                t = a_tgt[0].cpu().numpy()
                m = mix_ref[0].cpu().numpy()
                e = e / (abs(e).max() + 1e-8)
                if pesq_fn is not None:
                    avg_pesqi += pesq_fn(sr, t, e, "wb") - pesq_fn(sr, t, m, "wb")
                if stoi_fn is not None:
                    avg_stoii += stoi_fn(t, e, sr) - stoi_fn(t, m, sr)
            count += 1

        print(f"Avg SI-SNRi: {avg_sisnri / count:.3f} dB")
        if pesq_fn is not None:
            print(f"Avg PESQi:  {avg_pesqi / count:.3f}")
        if stoi_fn is not None:
            print(f"Avg STOIi:  {avg_stoii / count:.3f}")
