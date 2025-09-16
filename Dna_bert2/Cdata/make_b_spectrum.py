#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_b_spectrum.py
从序列与标签生成“CRE-specific feature spectrum”（论文B图）：
1) 用 DNABERT-2 编码得到 token-level embeddings
2) MiniBatchKMeans 聚成 K 个 codebook center
3) 对每条序列做软分配 -> 得到 K 维码本特征分布
4) 按类别求平均谱，并与总体均值做 log2 富集
5) 统一特征顺序绘制分面条形图

输出：
- out_dir/feature_spectrum_b.png
- out_dir/feature_spectra_values.csv  （每类在统一排序下的谱值）
- out_dir/feature_order.csv           （横轴顺序与原 center 索引）
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt


# -----------------------------
# 数据加载与检查
# -----------------------------
def load_data(csv_path, seq_col='sequence', label_col='label',
              split_col=None, keep_split=None):
    p = Path(csv_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"CSV 不存在：{p}")

    df = pd.read_csv(p)
    for col, name in [(seq_col, 'seq-col'), (label_col, 'label-col')]:
        if col not in df.columns:
            raise KeyError(f"CSV 中找不到列 `{col}`（参数 --{name} ）；"
                           f"实际列有：{list(df.columns)}")

    if split_col:
        if split_col not in df.columns:
            raise KeyError(f"CSV 中找不到分割列 `{split_col}`（参数 --split-col ）；"
                           f"实际列有：{list(df.columns)}")
        if keep_split:
            df = df[df[split_col].isin(keep_split)]
            if len(df) == 0:
                raise ValueError(f"筛选 split={keep_split} 后无样本。")

    # 去掉空序列
    df = df.dropna(subset=[seq_col, label_col])
    seqs = df[seq_col].astype(str).tolist()
    labels = df[label_col].astype(str).to_numpy()

    if len(seqs) == 0:
        raise ValueError("读取到的序列数量为 0。请检查 CSV 与筛选条件。")

    return seqs, labels


# -----------------------------
# 模型编码
# -----------------------------
@torch.no_grad()
def encode_tokens(seqs, model_name_or_path, batch_size=8, device='cuda', max_len=1024):
    """
    返回：list(np.ndarray)，每条为 [Ti, H] 的 token embedding
    - 强制 return_dict=True；若底层仍返回 tuple，兜底取 out[0]
    - 显式启用 truncation 与 max_length，避免长度不定
    """
    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    mdl.config.return_dict = True
    mdl.to(device).eval()

    all_token_embeds = []
    for i in tqdm(range(0, len(seqs), batch_size), desc="Encoding"):
        batch = seqs[i:i+batch_size]
        enc = tok(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_len
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = mdl(**enc, output_hidden_states=False, return_dict=True)
        X = out[0] if isinstance(out, tuple) else out.last_hidden_state  # [B, T, H]

        mask = enc['attention_mask'].bool()  # [B, T]
        B, T, H = X.shape
        for b in range(B):
            valid = X[b][mask[b]]  # [Tv, H]
            all_token_embeds.append(valid.detach().cpu().numpy().astype(np.float32))
    return all_token_embeds  # list of (Ti, H)


# -----------------------------
# 聚类与软分配
# -----------------------------
def sample_tokens_for_kmeans(token_lists, max_tokens=200_000, seed=0):
    """
    从所有 token 向量中按序列长度占比抽样，避免爆内存
    """
    rng = np.random.default_rng(seed)
    total = int(sum(x.shape[0] for x in token_lists))
    take = min(int(max_tokens), total)
    if take <= 0:
        raise ValueError("用于 KMeans 的样本数为 0。")

    out = []
    acc = 0
    for arr in token_lists:
        if acc >= take:
            break
        n = arr.shape[0]
        m = max(1, int(round(take * n / total)))
        m = min(m, n)
        idx = rng.choice(n, size=m, replace=False)
        out.append(arr[idx])
        acc += m
    return np.concatenate(out, axis=0)


def soft_assign(seq_tokens, centers, tau=0.5):
    """
    对单条序列的 token 向量 [T, H] 相对 K 个中心 [K, H] 做软分配
    使用 -||x-c||^2 / tau 的 softmax（逐 token），再对 token 求和并归一化
    返回： [K]，和为 1
    """
    # dist^2 = x^2 + c^2 - 2 x·c
    xc = seq_tokens @ centers.T                    # [T, K]
    c2 = (centers ** 2).sum(axis=1, keepdims=True).T  # [1, K]
    x2 = (seq_tokens ** 2).sum(axis=1, keepdims=True) # [T, 1]
    dist2 = x2 + c2 - 2.0 * xc                     # [T, K]

    logits = -dist2 / max(1e-6, float(tau))
    logits = logits - logits.max(axis=1, keepdims=True)  # 数值稳定
    p = np.exp(logits)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)       # 每个 token 的 K 维
    s = p.sum(axis=0)                                     # 聚合到序列级
    s = s / (s.sum() + 1e-12)
    return s.astype(np.float32)


# -----------------------------
# 计算谱与绘图
# -----------------------------
def compute_spectra(P, labels, classes, enrichment=True, eps=1e-9, order_by=None):
    """
    P: [N, K] 各样本码本分布，labels: [N] 字符串标签
    classes: 要绘制的类别列表（会按照此顺序绘制）
    enrichment: True 则返回 log2((S_c+eps)/(S_all+eps))
    order_by: 用哪个类别的谱确定横轴顺序；None 则用 classes[0]
    """
    K = P.shape[1]
    spectra = {}
    for c in classes:
        idx = (labels == c)
        spectra[c] = P[idx].mean(axis=0) if idx.any() else np.zeros(K, dtype=np.float32)

    if enrichment:
        pall = P.mean(axis=0)
        for c in classes:
            spectra[c] = np.log2((spectra[c] + eps) / (pall + eps))

    ref = order_by if (order_by in classes) else classes[0]
    order = np.argsort(spectra[ref])[::-1]  # 大到小
    for c in classes:
        spectra[c] = spectra[c][order]

    return spectra, order


def plot_spectra(spectra, classes, order, outfile,
                 dashed=('25%', '75%'), ylabel='Enrichment (log2)'):
    """
    统一 y 轴范围，分面条形图
    """
    K = len(order)
    x = np.arange(K)
    rows = len(classes)

    # 统一 y 轴范围
    all_vals = np.concatenate([spectra[c] for c in classes], axis=0)
    y_min, y_max = float(all_vals.min()), float(all_vals.max())
    pad = 0.05 * max(1e-6, (y_max - y_min))
    y_lim = (y_min - pad, y_max + pad)

    fig, axes = plt.subplots(rows, 1, figsize=(10, 0.9 + 1.8 * rows), sharex=True)
    if rows == 1:
        axes = [axes]

    for ax, c in zip(axes, classes):
        vals = spectra[c]
        ax.bar(x, vals, width=0.9, linewidth=0)
        ax.set_ylim(*y_lim)
        ax.axhline(0.0, lw=0.6, color='black')

        # 竖虚线（支持百分位或整数索引）
        for pos in dashed or []:
            if isinstance(pos, str) and pos.endswith('%'):
                q = float(pos[:-1]) / 100.0
                xi = int(round(q * (K - 1)))
            else:
                xi = int(pos)
            xi = max(0, min(K - 1, xi))
            ax.axvline(xi, ls='--', lw=0.8, color='gray', alpha=0.9)

        ax.text(1.01, 0.5, c, transform=ax.transAxes, va='center', ha='left')
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel('')

    axes[-1].set_xlabel('Codebook feature (ordered)')
    plt.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=600, bbox_inches='tight')
    plt.close(fig)


# -----------------------------
# 主流程
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='包含 sequence/label[/split] 的 CSV 路径')
    ap.add_argument('--seq-col', default='sequence')
    ap.add_argument('--label-col', default='label')
    ap.add_argument('--split-col', default=None)
    ap.add_argument('--keep-split', default=None, nargs='*', help="如: --keep-split test")

    ap.add_argument('--model-path', required=True, help='DNABERT-2 本地路径或 HF 名称')
    ap.add_argument('--k', type=int, default=256)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--max-len', type=int, default=1024, help='tokenizer 截断长度，默认 1024')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--tokens-for-kmeans', type=int, default=200000)
    ap.add_argument('--order-by', default=None, help='按哪个类别的谱排序（默认取第一个类别）')
    ap.add_argument('--out-dir', default='b_spectrum_out')
    args = ap.parse_args()

    # 1) 读数据
    seqs, labels = load_data(
        args.csv, args.seq_col, args.label_col,
        args.split_col, args.keep_split
    )

    # 自动按 CSV 实际标签（升序）作图；你也可自定义排序：
    classes = sorted(pd.unique(labels).tolist())
    if len(classes) < 2:
        raise ValueError(f"检测到的类别数为 {len(classes)}，不足以绘制多面板（需要>=2）。"
                         f"实际类别：{classes}")

    print(f"[INFO] 样本数: {len(labels)} | 类别: {classes}")

    # 2) 编码
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    token_lists = encode_tokens(
        seqs,
        args.model_path,
        batch_size=args.batch_size,
        device=device,
        max_len=args.max_len
    )

    # 3) KMeans（抽样以节省内存/时间）
    sample = sample_tokens_for_kmeans(token_lists, max_tokens=args.tokens_for_kmeans, seed=0)
    print(f"[INFO] 用于KMeans的样本 token 数: {sample.shape[0]} ; 维度: {sample.shape[1]}")
    kmeans = MiniBatchKMeans(
        n_clusters=args.k,
        random_state=0,
        batch_size=10000,
        n_init='auto'
    )
    kmeans.fit(sample)
    centers = kmeans.cluster_centers_.astype(np.float32)

    # 4) 对每条序列做软分配 -> P
    N = len(token_lists)
    P = np.zeros((N, args.k), dtype=np.float32)
    for i, tok in enumerate(tqdm(token_lists, desc="Soft assignment")):
        P[i] = soft_assign(tok, centers, tau=args.tau)

    # 5) 计算谱与排序
    spectra, order = compute_spectra(
        P, labels, classes, enrichment=True, order_by=args.order_by
    )

    # 6) 保存数值与顺序
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存谱值（行=class，列=feat_rank_x）
    df_vals = pd.DataFrame({'class': classes})
    for rank, idx in enumerate(order):
        col = f'feat_rank_{rank:04d}'
        df_vals[col] = [spectra[c][rank] for c in classes]
    df_vals.to_csv(out_dir / 'feature_spectra_values.csv', index=False)

    # 保存横轴顺序对应的 center 索引
    df_order = pd.DataFrame({
        'rank': np.arange(len(order), dtype=int),
        'center_index': order.astype(int)
    })
    df_order.to_csv(out_dir / 'feature_order.csv', index=False)

    # 7) 绘图
    plot_spectra(
        spectra, classes, order,
        outfile=str(out_dir / 'feature_spectrum_b.png'),
        dashed=('25%', '75%'),
        ylabel='Enrichment (log2)'
    )

    print(f"[Done] 图已保存: {out_dir / 'feature_spectrum_b.png'}")
    print(f"[Done] 数值表:   {out_dir / 'feature_spectra_values.csv'}")
    print(f"[Done] 顺序表:   {out_dir / 'feature_order.csv'}")


if __name__ == '__main__':
    main()
