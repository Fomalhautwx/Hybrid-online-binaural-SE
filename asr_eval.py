#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asr_eval.py —— 用 Whisper 测识别错误率。没有人工文本标注时，先用强模型识别干净的 ref 得到伪标注。

流程：
  1. 伪标注：标注模型（默认 large-v3）识别 ref 的评测段，得到文本和语种。结果写进 --cache，
     不同系统共用：同一条样本、同一段 ref 只识别一次。
  2. 评测：评测模型（默认 small）按伪标注的语种强制解码 ref / noisy / IVA 目标通道 / 最终输出，
     与伪标注逐字词对齐，统计替换 / 删除 / 插入。ref 这一行是下限：评测模型在干净语音上本身的错误。
  · 评测段与 se_eval_suite.py 完全相同（共用 load_utt）：从 max(前 1/40, IVA 起点) 到结尾。
  · 中文 / 粤语 / 日文按字，其它语种按词；中英混说时中文按字、英文按词（MER）。繁体统一转简体。
  · 歌声样本默认跳过（伪标注不可靠），--keep-singing 保留并单独列出。
  · 解码用 Whisper 默认的温度回退（先 temperature 0，出现循环或置信度过低再升温重解码；默认贪心，--beam 5 用束搜索），
    不以前文为条件，固定随机种子，结果可复现。结尾的 “Thank you.” 一类幻觉短语在伪标注和识别结果里都去掉。
  · 识别结果里出现连续重复 ≥ 4 次的片段记为循环幻觉，单独统计（这类错误主要反映 ASR 被噪声带偏）。
  · 每个信号先统一到 −25 dBFS 的 RMS 再送进 ASR，排除电平差异的影响。

输出（--out 用 se_eval_suite.py run 的同一个输出目录）：
  asr_per_utt.csv   逐条：语种、标注字词数、各信号的错误数 / 替换 / 删除 / 插入 / 错误率
  asr_hyps.jsonl    逐条的伪标注与各信号识别文本（排查用）
  asr_summary.md    语料级错误率（bootstrap 置信区间）、按语种 / 输入难度 / 类别分组、退步最多的样本
se_eval_suite.py compare 会自动读同一目录下的 asr_per_utt.csv。

依赖：pip install openai-whisper opencc-python-reimplemented（--backend faster 另需 pip install faster-whisper）
用法（放在 se_eval_suite.py 同目录）：
  python asr_eval.py --json <flow 的 jsonl> --wav-dir <flow 结果目录> --out eval_flow
  python asr_eval.py --json <dc 的 jsonl>   --wav-dir <dc 结果目录>   --out eval_dc
  几套系统要同时跑时，先单独生成一次伪标注，再并行启动（否则各进程会各自识别一遍、伪标注不一致）：
  python asr_eval.py --json <flow 的 jsonl> --wav-dir <flow 结果目录> --out eval_flow --labels-only
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from se_eval_suite import (FS, NAN, boot_ci, category, col, finite, fmt, load_records, load_utt,
                           locate, si_sdr, write_csv)

SIGS = ("ref", "mix", "bss", "est")
SIG_CN = {"ref": "干净参考（下限）", "mix": "输入(参考麦)", "bss": "IVA 目标通道", "est": "最终输出"}
CHAR_LANGS = {"zh", "yue", "ja"}                       # 按字计错的语种
TARGET_DBFS = -25.0
# 先 temperature 0 + beam 5；输出重复度过高（gzip 压缩比 > 2.4，典型的循环幻觉）或平均对数概率过低时，
# 依次升温重新解码。这是 Whisper 官方的默认策略，能打断大部分 “mass mass mass …” 式的循环。
TEMPS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# Whisper 在结尾静音 / 噪声上常凭空加的短语，伪标注和识别结果都去掉（对称处理，不偏向任何系统）
_TAIL = {"en": re.compile(r"(?:[\s.,!?]*\b(?:thank you(?: (?:very much|for watching))?|thanks for watching)\b[.!?]*"
                         r"|(?:^|(?<=[.!?]))\s*you[.!?]*)+\s*$", re.I),     # 单独成句的 “You” 才去
         "zh": re.compile(r"(?:[\s，。！]*(?:中文字幕提供|字幕by\S*|字幕由\S*提供|谢谢观看|请不吝点赞.*|明镜与点点栏目))+\s*$")}


def strip_tail(text, lang):
    pat = _TAIL.get("zh" if lang in CHAR_LANGS else lang)
    return pat.sub("", text).strip() if pat is not None else text.strip()


def has_loop(tok, n_max=8, rep=4):
    """连续重复 ≥ rep 次的 n-gram（n ≤ n_max）：Whisper 的循环幻觉。"""
    L = len(tok)
    for n in range(1, n_max + 1):
        for i in range(0, L - n * rep + 1):
            g = tok[i:i + n]
            if all(tok[i + j * n:i + (j + 1) * n] == g for j in range(1, rep)):
                return True
    return False


# =============================================================================
# 文本规范化与对齐
# =============================================================================

class TextNorm:
    """英文用 Whisper 的 EnglishTextNormalizer（数字、缩写、英美拼写统一）；其它语种用 BasicTextNormalizer
    （小写、去标点）；中文先繁转简，再按字切分，夹杂的英文按词。"""
    _CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿]|[a-z0-9']+")

    def __init__(self):
        from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
        self.en, self.basic = EnglishTextNormalizer(), BasicTextNormalizer()
        self.t2s = None
        try:
            import opencc
            for cfg in ("t2s", "t2s.json"):
                try:
                    self.t2s = opencc.OpenCC(cfg).convert
                    break
                except Exception:
                    continue
        except ImportError:
            pass
        if self.t2s is None:
            print("[警告] 没装 opencc，中文繁简不会统一（pip install opencc-python-reimplemented）")

    def __call__(self, text, lang):
        if lang in CHAR_LANGS:
            if self.t2s is not None:
                text = self.t2s(text)
            return self._CJK.findall(self.basic(text).lower())
        return (self.en(text) if lang == "en" else self.basic(text)).split()


def edit_counts(ref, hyp):
    """Levenshtein 对齐，返回 (替换, 删除, 插入)。"""
    n, m = len(ref), len(hyp)
    if n == 0 or m == 0:
        return 0, n, m
    vocab = {}
    r = np.array([vocab.setdefault(t, len(vocab)) for t in ref])
    h = np.array([vocab.setdefault(t, len(vocab)) for t in hyp])
    D = np.empty((n + 1, m + 1), dtype=np.int32)
    D[0] = np.arange(m + 1)
    ar = np.arange(m + 1)
    tmp = np.empty(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        tmp[0] = i
        tmp[1:] = np.minimum(D[i - 1, :-1] + (h != r[i - 1]), D[i - 1, 1:] + 1)
        D[i] = np.minimum.accumulate(tmp - ar) + ar             # 插入沿行累积
    i, j, S, Dl, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            S += int(r[i - 1] != h[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            Dl, i = Dl + 1, i - 1
        else:
            I, j = I + 1, j - 1
    return S, Dl, I


# =============================================================================
# ASR
# =============================================================================

class ASR:
    """两种后端：openai（openai-whisper，读 .pt）/ faster（faster-whisper，CTranslate2，快 3–4 倍）。"""

    def __init__(self, name, device, backend="openai", beam=1):
        import torch
        self._torch, self.backend, self.beam = torch, backend, beam
        cuda = device.startswith("cuda")
        if backend == "faster":
            from faster_whisper import WhisperModel
            idx = int(device.split(":")[1]) if ":" in device else 0
            self.model = WhisperModel(name, device="cuda" if cuda else "cpu", device_index=idx,
                                      compute_type="float16" if cuda else "int8")
        else:
            import whisper
            self.fp16 = cuda
            self.model = whisper.load_model(name, device=device)

    def __call__(self, x, lang=None):
        x = np.asarray(x, dtype=np.float64)
        rms = np.sqrt(np.mean(x ** 2))
        if not np.isfinite(rms) or rms < 1e-8:
            return "", lang or ""
        x = (x * (10 ** (TARGET_DBFS / 20) / rms)).astype(np.float32)
        common = dict(language=lang, task="transcribe", best_of=5, compression_ratio_threshold=2.4,
                      no_speech_threshold=0.6, condition_on_previous_text=False)
        if self.backend == "faster":
            import ctranslate2
            ctranslate2.set_random_seed(0)                    # 温度回退时会采样，固定种子保证可复现
            segs, info = self.model.transcribe(x, beam_size=self.beam, temperature=list(TEMPS),
                                               log_prob_threshold=-1.0, **common)
            return "".join(sg.text for sg in segs).strip(), info.language or lang or ""
        self._torch.manual_seed(0)
        r = self.model.transcribe(x, temperature=TEMPS, beam_size=self.beam if self.beam > 1 else None,
                                  logprob_threshold=-1.0, fp16=self.fp16, verbose=None, **common)
        return r["text"].strip(), r.get("language", lang or "")

    def free(self):
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


class Cache:
    """识别结果缓存（jsonl，逐行追加）：中断后重跑会跳过已识别的，伪标注和 ref / noisy 的结果跨系统复用。
    同一个键有多条时（例如两个进程同时从头跑、各识别了一遍）只认最早写入的那条，保证所有系统用同一份伪标注。"""

    def __init__(self, path):
        self.path, self.d = Path(path), {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(line)
                    self.d.setdefault(e["key"], e)
                except (json.JSONDecodeError, KeyError):
                    continue
        self.f = open(self.path, "a", encoding="utf-8", buffering=1)

    def get(self, key):
        return self.d.get(key)

    def put(self, key, text, lang):
        e = {"key": key, "text": text, "lang": lang}
        self.d.setdefault(key, e)
        self.f.write(json.dumps(e, ensure_ascii=False) + "\n")


# =============================================================================
# 主流程
# =============================================================================

def iter_utts(recs, wav_dir, ratio, keep_singing):
    """逐条读入并对齐；产出 (rec, stem, u 或 None, 跳过原因)。"""
    for rec in recs:
        stem, files = locate(rec, wav_dir)
        if rec.get("error"):
            yield rec, stem, None, "测试脚本失败"
            continue
        if not keep_singing and category(stem) == "singing":
            yield rec, stem, None, "歌声"
            continue
        try:
            u = load_utt(files, ratio)
        except ValueError as e:
            yield rec, stem, None, f"读文件失败：{e}"
            continue
        yield rec, stem, u, ""


def span_key(stem, u):
    a, b = u["head"] + u["s0"], u["head"] + u["N"]               # 原始时间轴上的评测段
    return f"{stem}|{a}-{b}"


def corpus_rate(err, ntok):
    ok = np.isfinite(err) & np.isfinite(ntok) & (ntok > 0)
    return 100 * err[ok].sum() / max(ntok[ok].sum(), 1e-9) if ok.any() else NAN


def corpus_ci(err, ntok, n_boot=2000, seed=0):
    ok = np.isfinite(err) & np.isfinite(ntok) & (ntok > 0)
    e, t = err[ok], ntok[ok]
    if len(e) < 2:
        return NAN, NAN
    idx = np.random.default_rng(seed).integers(0, len(e), (n_boot, len(e)))
    r = 100 * e[idx].sum(1) / t[idx].sum(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def build_summary(rows, skipped, n_rec, args, has_bss):
    L = []
    P = L.append
    sigs = [s for s in SIGS if s != "bss" or has_bss]
    P("# ASR 评测\n")
    P(f"- 标注模型 Whisper {args.label_model}（识别 ref 得到伪标注），评测模型 Whisper {args.eval_model}；"
      f"{args.backend}-whisper，温度回退解码（temperature 0 起、beam {args.beam}）、不以前文为条件、去掉结尾幻觉短语；"
      "送入 ASR 前统一到 −25 dBFS")
    P(f"- 评测段与 se_eval_suite.py 相同：从 max(前 1/{args.warmup_ratio:g}, IVA 起点) 到结尾，"
      f"平均从 {np.nanmean(col(rows, 'asr_span_s')):.2f} s 开始")
    P(f"- 记录 {n_rec} 条，参与统计 {len(rows)} 条"
      + ("；跳过：" + "，".join(f"{k} {v} 条" for k, v in Counter(skipped).most_common()) if skipped else ""))
    langs = Counter(r["asr_lang"] for r in rows)
    P("- 伪标注语种：" + "，".join(f"{k} {v} 条" for k, v in langs.most_common()) + "\n")

    nt = col(rows, "asr_ntok")
    lang = np.array([r["asr_lang"] for r in rows])
    groups = [("中文等按字（CER）", np.isin(lang, list(CHAR_LANGS))), ("英文（WER）", lang == "en"),
              ("其它语种（WER）", ~np.isin(lang, list(CHAR_LANGS) + ["en"]))]
    groups = [(g, m) for g, m in groups if m.any()]

    P("## 1. 语料级错误率\n")
    P("错误率 = 总错误数 / 伪标注总字词数（%），方括号是对样本做 bootstrap 的 95% 置信区间。"
      "ref 这一行是评测模型在干净语音上的错误，是其它几行的下限。\n")
    P("| 信号 | 错误率 | 替换 | 删除 | 插入 | " + " | ".join(g for g, _ in groups) + " | 循环幻觉条数 | 去掉任一路有循环的样本后 |")
    P("|---|---|---|---|---|" + "---|" * len(groups) + "---|---|")
    tot = np.nansum(nt)
    anyloop = np.zeros(len(rows), bool)
    for s in sigs:
        anyloop |= col(rows, f"asr_loop_{s}") > 0
    for s in sigs:
        err = col(rows, f"asr_err_{s}")
        lo, hi = corpus_ci(err, nt)
        parts = [f"{100 * np.nansum(col(rows, f'asr_{q}_{s}')) / max(tot, 1e-9):.2f}" for q in ("sub", "del", "ins")]
        gcells = [fmt(corpus_rate(err[m], nt[m])) for _, m in groups]
        nloop = int(np.nansum(col(rows, f"asr_loop_{s}")))
        P(f"| {SIG_CN[s]} | {fmt(corpus_rate(err, nt))} [{fmt(lo)}, {fmt(hi)}] | " + " | ".join(parts)
          + " | " + " | ".join(gcells) + f" | {nloop} | {fmt(corpus_rate(err[~anyloop], nt[~anyloop]))} |")
    P(f"\n替换 / 删除 / 插入同样以伪标注总字词数为分母。删除多 = 过度抑制或伪影把字吞掉了；插入多 = 残余噪声上的幻觉。"
      f"循环幻觉 = 识别结果里有连续重复 ≥ 4 次的片段，一条能贡献上百个插入错误，最后一列去掉了"
      f"任一路出现循环的 {int(anyloop.sum())} 条，所有信号用同一批样本。\n")

    def group_rows(head, groups_):
        P(f"| {head} | 条数 | 输入 SI-SDR | " + " | ".join(SIG_CN[s] for s in sigs) + " |")
        P("|---|---|---|" + "---|" * len(sigs))
        for name, m in groups_:
            cells = [fmt(corpus_rate(col(rows, f"asr_err_{s}")[m], nt[m])) for s in sigs]
            P(f"| {name} | {int(m.sum())} | {fmt(np.nanmean(col(rows, 'asr_in_sisdr')[m]))} | " + " | ".join(cells) + " |")
        P("")

    mixv = col(rows, "asr_in_sisdr")
    ok = np.isfinite(mixv)
    if ok.sum() >= 8:
        edges = np.percentile(mixv[ok], [25, 50, 75])
        b = np.digitize(mixv, edges)
        P("## 2. 按输入难度分桶（输入 SI-SDR 四分位，Q1 最难）\n")
        P(f"分界 {edges[0]:.1f} / {edges[1]:.1f} / {edges[2]:.1f} dB；表中是各信号的语料级错误率（%）。\n")
        group_rows("桶", [(f"Q{q + 1}", (b == q) & ok) for q in range(4)])
    cats = np.array([category(r["stem"]) for r in rows])
    uc = [c for c, _ in Counter(cats.tolist()).most_common()]
    if len(uc) > 1:
        P("## 3. 按语料类别\n")
        group_rows("类别", [(c, cats == c) for c in uc])

    P("## 4. 最终输出比输入退步最多的样本\n")
    cand = [r for r in rows if finite(r.get("asr_err_est")) and finite(r.get("asr_err_mix")) and r["asr_ntok"] > 0]
    cand.sort(key=lambda r: r["asr_err_mix"] - r["asr_err_est"])
    for r in cand[:8]:
        if r["asr_err_est"] <= r["asr_err_mix"]:
            break
        P(f"- `{r['stem']}`（{r['asr_lang']}，{int(r['asr_ntok'])} 字词）：输入 {r['asr_rate_mix']:.1f}% → "
          f"最终 {r['asr_rate_est']:.1f}%（删除 {int(r['asr_del_est'])}，插入 {int(r['asr_ins_est'])}）")
    P("\n识别文本见 `asr_hyps.jsonl`。")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, help="测试脚本写的 test_results_*.jsonl")
    ap.add_argument("--wav-dir", required=True, help="测试脚本存音频的目录")
    ap.add_argument("--out", required=True, help="输出目录（建议用 se_eval_suite.py run 的同一个目录）")
    ap.add_argument("--label-model", default="large-v3", help="生成伪标注的 Whisper 模型（名字或 .pt 路径）")
    ap.add_argument("--eval-model", default="small", help="评测用的 Whisper 模型")
    ap.add_argument("--cache", default="asr_cache.jsonl", help="识别结果缓存，几套系统共用同一个文件")
    ap.add_argument("--device", default=None, help="默认有 GPU 用 cuda；多卡可写 cuda:1")
    ap.add_argument("--backend", choices=["openai", "faster"], default="openai",
                    help="openai = openai-whisper（.pt 权重）；faster = faster-whisper（快 3–4 倍，需 pip install faster-whisper）")
    ap.add_argument("--beam", type=int, default=5, help="束宽：1 = 贪心（默认，快）；5 = Whisper 论文设置（约慢 2–3 倍）")
    ap.add_argument("--warmup-ratio", type=float, default=6.0, help="与测试脚本一致：丢弃前 1/warmup_ratio")
    ap.add_argument("--keep-singing", action="store_true", help="保留歌声样本（默认跳过）")
    ap.add_argument("--langs", default="", help="只统计这些语种的样本，逗号分隔，如 en 或 en,zh（默认全部）")
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 条（调试用）")
    ap.add_argument("--labels-only", action="store_true",
                    help="只生成伪标注就退出。几套系统要并行跑时，先用它把伪标注生成好，再同时启动")
    args = ap.parse_args()

    import os
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":                                       # se_eval_suite 为多进程把线程数设成了 1
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = load_records(args.json)
    if args.limit:
        recs = recs[:args.limit]
    wav_key = str(Path(args.wav_dir).resolve())
    cache = Cache(args.cache)
    langs = {x.strip() for x in args.langs.split(",") if x.strip()}
    # 缓存标签：解码设置不同的结果不混用（openai + beam 5 沿用之前的 d2，已跑过的不用重算）
    dec = "d2" if (args.backend, args.beam) == ("openai", 5) else f"d2-{args.backend}-b{args.beam}"
    dec_desc = f"{args.backend}-whisper beam{args.beam} 标注{args.label_model} 评测{args.eval_model}"
    print(f"设备 {device}；Whisper 解码约 1 s / 条（GPU）是正常的，明显更慢请检查是否真在用 GPU")
    norm = TextNorm()
    t0 = time.time()

    # —— 第 1 遍：伪标注（缓存里没有的才识别；第一次需要时才加载大模型）——
    asr, n_new = None, 0
    bar = tqdm(iter_utts(recs, args.wav_dir, args.warmup_ratio, args.keep_singing), total=len(recs),
               desc=f"伪标注 {args.label_model}", unit="条", dynamic_ncols=True)
    for _, stem, u, why in bar:
        key = f"label|{dec}|{args.label_model}|{span_key(stem, u)}" if u is not None else None
        if key is None or cache.get(key) is not None:
            continue
        if asr is None:
            bar.write(f"加载 Whisper {args.label_model}（{device}）")
            asr = ASR(args.label_model, device, args.backend, args.beam)
        text, lang = asr(u["ref"][u["s0"]:])
        cache.put(key, text, lang)
        n_new += 1
        bar.set_postfix(新识别=n_new, refresh=False)
    bar.close()
    if asr is not None:
        asr.free()
    if args.labels_only:
        print(f"伪标注已写入 {args.cache}（本次新识别 {n_new} 条）")
        return

    # —— 第 2 遍：评测模型识别各信号 ——
    asr = None
    rows, skipped, hyps = [], [], []
    bar = tqdm(iter_utts(recs, args.wav_dir, args.warmup_ratio, args.keep_singing), total=len(recs),
               desc=f"评测 {args.eval_model}", unit="条", dynamic_ncols=True)
    for rec, stem, u, why in bar:
        if u is None:
            skipped.append(why.split("：")[0])
            continue
        lab = cache.get(f"label|{dec}|{args.label_model}|{span_key(stem, u)}")
        lang = lab["lang"] or "en"
        if langs and lang not in langs:
            skipped.append(f"语种 {lang}")
            continue
        ref_tok = norm(strip_tail(lab["text"], lang), lang)
        if not ref_tok:
            skipped.append("伪标注为空")
            continue
        s0 = u["s0"]
        sig = {"ref": u["ref"][s0:], "mix": u["mix"][s0:], "est": u["est"][s0:],
               "bss": u["bss"][0, s0:] if u["bss"] is not None else None}
        row = {"stem": stem, "idx": rec.get("idx", ""), "asr_dec": dec_desc, "asr_lang": lang, "asr_ntok": len(ref_tok),
               "asr_span_s": (u["head"] + s0) / FS, "asr_in_sisdr": si_sdr(sig["mix"], sig["ref"])}
        hyp_rec = {"stem": stem, "lang": lang, "label": lab["text"]}
        for k in SIGS:
            if sig[k] is None:
                continue
            shared = k in ("ref", "mix")                             # 与系统无关，可跨系统复用
            key = f"eval|{dec}|{args.eval_model}|{k}|{'' if shared else wav_key}|{span_key(stem, u)}|{lang}"
            hit = cache.get(key)
            if hit is None:
                if asr is None:
                    bar.write(f"加载 Whisper {args.eval_model}（{device}）")
                    asr = ASR(args.eval_model, device, args.backend, args.beam)
                text, _ = asr(sig[k], lang)
                cache.put(key, text, lang)
            else:
                text = hit["text"]
            hyp_tok = norm(strip_tail(text, lang), lang)
            S, Dl, I = edit_counts(ref_tok, hyp_tok)
            row.update({f"asr_loop_{k}": int(has_loop(hyp_tok)),
                        f"asr_sub_{k}": S, f"asr_del_{k}": Dl, f"asr_ins_{k}": I,
                        f"asr_err_{k}": S + Dl + I, f"asr_rate_{k}": 100 * (S + Dl + I) / len(ref_tok)})
            hyp_rec[k] = text
        rows.append(row)
        hyps.append(hyp_rec)
        bar.set_postfix(参与=len(rows), 跳过=len(skipped), refresh=False)
    bar.close()

    if not rows:
        print("没有可统计的样本。跳过原因：", Counter(skipped).most_common())
        return
    write_csv(rows, out / "asr_per_utt.csv")
    with open(out / "asr_hyps.jsonl", "w", encoding="utf-8") as f:
        for h in hyps:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    has_bss = np.isfinite(col(rows, "asr_err_bss")).any()
    report = build_summary(rows, skipped, len(recs), args, has_bss)
    (out / "asr_summary.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"完成，用时 {time.time() - t0:.0f}s。输出：{out.resolve()}")


if __name__ == "__main__":
    main()
