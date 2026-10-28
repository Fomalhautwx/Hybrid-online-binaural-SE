"""
swap_prob 推理记录 + 结束后出图工具

用法（三行接入现有推理循环）：

    from swap_prob_plot import SwapProbRecorder

    rec = SwapProbRecorder(bandsize=257, fs=16000)

    for X, Y_nn in frame_stream:
        Y_out, score, energy = model(X, Y_nn)
        rec.record(model.swap_prob)              # 每帧一行，仅拷贝一个(F,)向量

    rec.save_plot("swap_prob_result.png")        # 推理结束后输出图片
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # 无显示环境也能出图
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager


def _has_cjk_font() -> bool:
    """检测系统是否有可用的中文字体，没有则回退到英文标签。"""
    candidates = ["Noto Sans CJK SC", "SimHei", "Microsoft YaHei",
                  "PingFang SC", "WenQuanYi Zen Hei", "Source Han Sans SC"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


_CJK = _has_cjk_font()

# 中英标签表：无中文字体时自动回退英文，避免乱码/警告
_L = {
    "title_heat":  ("swap_prob 时频热力图（绿=保持原通道 红=稳定翻转 黄=不稳定）",
                    "swap_prob heatmap (green=keep, red=swap, yellow=unstable)"),
    "frame":       ("帧", "frame"),
    "freq":        ("频率 (Hz)", "frequency (Hz)"),
    "title_band":  ("各频段 swap_prob 均值", "band-wise mean swap_prob"),
    "title_hist":  ("swap_prob 分布直方图", "swap_prob histogram"),
    "count":       ("计数", "count"),
    "stable":      ("稳定占比", "stable ratio"),
    "unstable":    ("不稳定频点", "unstable bins"),
    "mean":        ("全局均值", "global mean"),
    "total":       ("总帧数", "frames"),
    "band_names":  (["<500Hz", "0.5~1k", "1~2k", "2~4k", ">4k"],) * 2,
    "threshold":   ("0.5 判决阈值", "decision threshold 0.5"),
}


def _t(key: str) -> str:
    pair = _L[key]
    return pair[0] if _CJK else pair[1]


class SwapProbRecorder:
    """推理中逐帧记录 swap_prob，推理结束后调用 save_plot() 输出 PNG。"""

    def __init__(self, bandsize: int = 257, fs: int = 16000):
        self.bandsize = bandsize
        self.fs = fs
        nfft = 2 * (bandsize - 1)
        self.freqs = np.arange(bandsize) * fs / nfft
        self._rows: list[np.ndarray] = []

    # ── 推理循环中每帧调用 ────────────────────────────────────────────

    def record(self, swap_prob: torch.Tensor):
        """swap_prob: model.swap_prob，shape (1, F) 或 (F,)。"""
        sp = swap_prob.detach().cpu().float().squeeze().numpy().copy()
        if sp.shape[-1] != self.bandsize:
            raise ValueError(
                f"swap_prob 维度 {sp.shape} 与 bandsize={self.bandsize} 不符")
        self._rows.append(sp)


    def log(self, number):
        num = number.detach().cpu().float().squeeze().numpy().copy()
        self._rows.append(num)
        
        
    def reset(self):
        self._rows.clear()

    @property
    def num_frames(self) -> int:
        return len(self._rows)

    # ── 推理结束后调用 ────────────────────────────────────────────────

    def save_plot(self,
                  save_path: str,
                  figsize: tuple = (14, 9),
                  dpi: int = 150) -> str:
        """
        绘制三张子图并保存：
          1. 时频热力图（横轴帧、纵轴频率、颜色=swap_prob）
          2. 各频段均值随时间变化曲线
          3. 全局 swap_prob 直方图（衡量对齐收敛程度）
        返回保存路径。
        """
        if not self._rows:
            raise ValueError("没有记录任何帧，请先在推理循环中调用 record()")

        mat = np.stack(self._rows, axis=0)          # (T, F)
        T = mat.shape[0]
        t_axis = np.arange(T)

        fig = plt.figure(figsize=figsize, facecolor="#1e1e2e")
        gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.30,
                               left=0.07, right=0.97, top=0.90, bottom=0.07)
        ax_heat = fig.add_subplot(gs[0, :])
        ax_band = fig.add_subplot(gs[1, 0])
        ax_hist = fig.add_subplot(gs[1, 1])

        for ax in (ax_heat, ax_band, ax_hist):
            ax.set_facecolor("#13131f")
            ax.tick_params(colors="#cccccc", labelsize=8)
            for s in ax.spines.values():
                s.set_edgecolor("#444466")

        # ── 1. 热力图 ────────────────────────────────────────────────
        im = ax_heat.imshow(
            mat.T, aspect="auto", origin="lower",
            vmin=0.0, vmax=1.0, cmap="RdYlGn_r",
            extent=[0, T, self.freqs[0], self.freqs[-1]],
        )
        cbar = fig.colorbar(im, ax=ax_heat, pad=0.01)
        cbar.ax.tick_params(colors="#cccccc", labelsize=7)
        cbar.set_label("swap_prob", color="#cccccc", fontsize=8)
        ax_heat.set_title(_t("title_heat"), color="white", fontsize=9)
        ax_heat.set_xlabel(_t("frame"), color="#aaaacc", fontsize=8)
        ax_heat.set_ylabel(_t("freq"), color="#aaaacc", fontsize=8)

        # ── 2. 各频段均值曲线 ───────────────────────────────────────
        band_edges = [(0, 500), (500, 1000), (1000, 2000),
                      (2000, 4000), (4000, self.fs // 2)]
        band_names = _L["band_names"][0]
        colors = ["#ff6b6b", "#ffa94d", "#51cf66", "#339af0", "#cc5de8"]
        for (flo, fhi), name, c in zip(band_edges, band_names, colors):
            idx = np.where((self.freqs >= flo) & (self.freqs < fhi))[0]
            if idx.size == 0:
                continue
            ax_band.plot(t_axis, mat[:, idx].mean(axis=1),
                         color=c, lw=1.2, label=name)
        ax_band.axhline(0.5, color="#ff6666", lw=0.8, ls="--",
                        label=_t("threshold"))
        ax_band.fill_between(t_axis, 0.2, 0.8, color="#ffa94d", alpha=0.08)
        ax_band.set_ylim(0, 1)
        ax_band.set_xlim(0, max(T - 1, 1))
        ax_band.set_title(_t("title_band"), color="white", fontsize=8)
        ax_band.set_xlabel(_t("frame"), color="#aaaacc", fontsize=7)
        ax_band.set_ylabel("swap_prob", color="#aaaacc", fontsize=7)
        ax_band.legend(fontsize=6, loc="upper right",
                       facecolor="#13131f", labelcolor="white",
                       framealpha=0.6)

        # ── 3. 直方图 ───────────────────────────────────────────────
        flat = mat.flatten()
        stable_ratio = float(((flat < 0.2) | (flat > 0.8)).mean()) * 100
        ax_hist.hist(flat, bins=50, range=(0, 1),
                     color="#7ec8e3", edgecolor="#1e1e2e", alpha=0.85)
        ax_hist.axvline(0.2, color="#51cf66", lw=1.0, ls="--")
        ax_hist.axvline(0.8, color="#51cf66", lw=1.0, ls="--")
        ax_hist.axvline(0.5, color="#ff6666", lw=1.0, ls=":")
        ax_hist.set_title(
            f"{_t('title_hist')} ({_t('stable')} {stable_ratio:.1f}%)",
            color="white", fontsize=8)
        ax_hist.set_xlabel("swap_prob", color="#aaaacc", fontsize=7)
        ax_hist.set_ylabel(_t("count"), color="#aaaacc", fontsize=7)

        # ── 总标题 ──────────────────────────────────────────────────
        mean_per_freq = mat.mean(axis=0)
        n_unstable = int(((mean_per_freq > 0.2) &
                          (mean_per_freq < 0.8)).sum())
        fig.suptitle(
            f"swap_prob | {_t('total')}: {T} | "
            f"{_t('unstable')}: {n_unstable}/{self.bandsize} | "
            f"{_t('mean')}: {mat.mean():.3f}",
            color="white", fontsize=10, y=0.96)

        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor="#1e1e2e")
        plt.close(fig)
        return save_path
    
    def curve(self, save_path: str,
                  figsize: tuple = (14, 9),
                  dpi: int = 150):
        if not self._rows:
            raise ValueError("没有记录任何帧，请先在推理循环中调用 record()")

        mat = np.stack(self._rows, axis=0).mean(axis=1)       # (T, F)
        
        T = mat.shape[0]
        t_axis = np.arange(T)

        plt.plot(t_axis, mat)
        plt.savefig(save_path)
