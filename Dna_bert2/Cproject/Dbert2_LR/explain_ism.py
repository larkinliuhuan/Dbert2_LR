import os
import sys
import json
import math
import argparse
import hashlib
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

try:
    import requests  # 可选：当使用 --service_url 时需要
except Exception:  # noqa: E722 - 仅在未安装 requests 时容忍
    requests = None

from transformers import AutoTokenizer, AutoModel

# 可选：图片绘制依赖
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
except Exception:  # noqa: E722 - 允许无图环境
    plt = None
    mpl = None


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _sha1_short(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _parse_fasta(path: str) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"输入文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        seq_id = None
        seq_chunks: List[str] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    entries.append((seq_id, "".join(seq_chunks)))
                seq_id = line[1:].strip() or f"seq_{len(entries)+1}"
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if seq_id is not None:
            entries.append((seq_id, "".join(seq_chunks)))
    return entries


def _read_sequences(
    input_file: Optional[str],
    input_column: str,
    sequences: Optional[List[str]],
) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    if input_file:
        lower = input_file.lower()
        if lower.endswith(".fa") or lower.endswith(".fasta"):
            results.extend(_parse_fasta(input_file))
        elif lower.endswith(".csv"):
            df = pd.read_csv(input_file)
            if input_column not in df.columns:
                raise KeyError(
                    f"CSV中未找到列: {input_column}. 可通过 --input_column 指定; 当前可选列: {list(df.columns)}"
                )
            for i, row in df.iterrows():
                sid = str(row.get("id", f"row_{i+1}"))
                seq = str(row[input_column])
                results.append((sid, seq))
        elif lower.endswith(".txt"):
            with open(input_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    results.append((f"line_{i+1}", line))
        else:
            raise ValueError("不支持的输入文件类型，仅支持 .fa/.fasta/.csv/.txt")

    if sequences:
        for s in sequences:
            sid = f"seq_{_sha1_short(s)}"
            results.append((sid, s))

    if not results:
        raise ValueError("未提供任何序列。请使用 --input_file 或 --seq 传入。")

    # 规范化序列
    normed: List[Tuple[str, str]] = []
    for sid, s in results:
        s2 = s.strip().upper().replace("U", "T")
        normed.append((sid, s2))
    return normed


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


class _LocalHybridModel:
    def __init__(
        self,
        artifacts_dir: str,
        model_module_path: str,
        base_model_name: str,
        device: torch.device,
        local_files_only: bool,
        max_length: int,
    ) -> None:
        self.device = device
        self.max_length = max_length

        # 动态导入 hybrid 模型定义
        if not os.path.exists(model_module_path):
            raise FileNotFoundError(
                f"未找到模型脚本: {model_module_path}. 请传入 Cproject/Dbert2_LR/hybrid_dnabert2_rnn_lstm.py 的实际路径"
            )
        import importlib.util

        spec = importlib.util.spec_from_file_location("hybrid_mod", model_module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模型脚本: {model_module_path}")
        hybrid_mod = importlib.util.module_from_spec(spec)
        sys.modules["hybrid_mod"] = hybrid_mod
        spec.loader.exec_module(hybrid_mod)  # type: ignore[attr-defined]

        if not hasattr(hybrid_mod, "HybridDNABERT2"):
            raise AttributeError("模型脚本中未找到类 HybridDNABERT2")
        HybridDNABERT2 = getattr(hybrid_mod, "HybridDNABERT2")

        # 读取标签映射（兼容多种结构）
        idx_to_label: Dict[int, str] = {}
        label_map_path = os.path.join(artifacts_dir, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, "r", encoding="utf-8") as f:
                lm = json.load(f)
            if isinstance(lm, dict) and isinstance(lm.get("id2label"), dict):
                for k, v in lm["id2label"].items():
                    try:
                        idx_to_label[int(k)] = str(v)
                    except Exception:
                        pass
            elif isinstance(lm, dict) and isinstance(lm.get("label2id"), dict):
                for lbl, idx in lm["label2id"].items():
                    try:
                        idx_to_label[int(idx)] = str(lbl)
                    except Exception:
                        pass
            elif isinstance(lm, dict):
                for k, v in lm.items():
                    try:
                        idx_to_label[int(k)] = str(v)
                    except Exception:
                        pass
        if not idx_to_label:
            idx_to_label = {0: "class_0", 1: "class_1"}
        self.idx_to_label = idx_to_label
        self.num_classes = 1 + max(self.idx_to_label.keys())

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        # 加载 DNABERT-2 基座并实例化 HybridDNABERT2(base_model, hidden_size, num_labels)
        base_model = AutoModel.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        # 禁用可能存在的 flash attention
        if hasattr(base_model.config, "use_flash_attn"):
            try:
                base_model.config.use_flash_attn = False
            except Exception:
                pass

        hidden_size = getattr(base_model.config, "hidden_size", None)
        if hidden_size is None:
            if hasattr(base_model.config, "d_model"):
                hidden_size = int(getattr(base_model.config, "d_model"))
            else:
                raise AttributeError("无法从 base_model.config 获取 hidden_size/d_model")

        model = HybridDNABERT2(base_model, hidden_size=int(hidden_size), num_labels=self.num_classes)  # type: ignore[arg-type]

        # 加载权重
        ckpt_candidates = [
            os.path.join(artifacts_dir, "best_model.pt"),
            os.path.join(artifacts_dir, "model.pt"),
            os.path.join(artifacts_dir, "checkpoint.pt"),
        ]
        ckpt_path = next((p for p in ckpt_candidates if os.path.exists(p)), None)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"未找到模型权重(best_model.pt / model.pt / checkpoint.pt) 于 {artifacts_dir}"
            )
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict):
            for key in ["model_state_dict", "state_dict", "model", "module"]:
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        if not isinstance(state, dict):
            raise ValueError("权重文件格式不正确，未获得 state_dict")

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            # 打印提示但不中断
            print(f"[warning] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, seqs: List[str], batch_size: int) -> np.ndarray:
        probs_list: List[np.ndarray] = []
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc)  # 期望输出 [B, C]
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            probs_list.append(prob)
        return np.concatenate(probs_list, axis=0)


class _ServiceModel:
    def __init__(self, service_url: str, timeout: float = 30.0) -> None:
        if requests is None:
            raise ImportError("未安装 requests，无法使用 --service_url 模式。请 pip install requests")
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout

    def predict_proba(self, seqs: List[str], batch_size: int) -> np.ndarray:
        # 尝试 /predict_batch，否则逐条 /predict
        probs: List[List[float]] = []
        batch_endpoint = f"{self.service_url}/predict_batch"
        single_endpoint = f"{self.service_url}/predict"
        use_batch = True
        # 先试探 batch 接口
        try:
            resp = requests.post(
                batch_endpoint,
                json={"sequences": seqs},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                # 期望格式: {"probs": [[...], ...]}
                if isinstance(data, dict) and "probs" in data:
                    return np.array(data["probs"], dtype=np.float32)
        except Exception:
            use_batch = False

        if not use_batch:
            for s in seqs:
                resp = requests.post(single_endpoint, json={"sequence": s}, timeout=self.timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"服务调用失败: HTTP {resp.status_code} {resp.text}")
                data = resp.json()
                # 期望格式: {"probs": [...]} 或 {"probabilities": [...]}
                arr = data.get("probs") or data.get("probabilities")
                if not isinstance(arr, list):
                    raise ValueError("服务返回格式错误，未找到 probs/probabilities")
                probs.append([float(x) for x in arr])
        return np.array(probs, dtype=np.float32)


def _resolve_target_index(
    ref_probs: np.ndarray,
    target_label: Optional[str],
    idx_to_label: Optional[Dict[int, str]],
) -> int:
    if target_label is None:
        return int(ref_probs.argmax())
    # 支持直接给 index
    if target_label.isdigit():
        return int(target_label)
    # 支持按标签名匹配
    if idx_to_label:
        for i, name in idx_to_label.items():
            if str(name) == target_label:
                return int(i)
    raise ValueError(f"无法解析 target_label: {target_label}")


def _generate_mutants(seq: str) -> Tuple[List[str], List[Tuple[int, str]]]:
    bases = ["A", "C", "G", "T"]
    seq_list = list(seq)
    mutants: List[str] = []
    meta: List[Tuple[int, str]] = []  # (position, alt)
    for i, b in enumerate(seq_list):
        if b not in bases:
            continue
        for alt in bases:
            if alt == b:
                continue
            seq_list[i] = alt
            mutants.append("".join(seq_list))
            meta.append((i, alt))
            seq_list[i] = b
    return mutants, meta


def ism_explain_for_sequence(
    seq_id: str,
    seq: str,
    predictor,
    batch_size: int,
    max_length: int,
    target_label: Optional[str],
    idx_to_label: Optional[Dict[int, str]],
) -> pd.DataFrame:
    # 参考概率
    ref_prob = predictor.predict_proba([seq], batch_size=batch_size)[0]
    target_idx = _resolve_target_index(ref_prob, target_label, idx_to_label)

    # 全量突变
    mutants, meta = _generate_mutants(seq)
    if not mutants:
        raise ValueError("序列中不存在可替换的 A/C/G/T 碱基，无法进行ISM")

    # 批量推理突变体
    mut_probs = predictor.predict_proba(mutants, batch_size=batch_size)
    mut_scores = mut_probs[:, target_idx] - ref_prob[target_idx]

    # 聚合到每个位点：记录三种替换的分数
    bases = ["A", "C", "G", "T"]
    seq_list = list(seq)
    L = len(seq_list)

    per_pos: Dict[int, Dict[str, float]] = {}
    for (pos, alt), delta in zip(meta, mut_scores):
        d = per_pos.setdefault(pos, {})
        d[alt] = float(delta)

    records: List[Dict[str, object]] = []
    for pos in range(L):
        ref = seq_list[pos]
        deltas = {b: (per_pos.get(pos, {}).get(b) if b != ref else np.nan) for b in bases}
        # 选择绝对值最大的替换作为该位重要性（带符号）
        best_alt = None
        best_val = 0.0
        for b in bases:
            if math.isnan(deltas[b]):
                continue
            v = float(deltas[b])
            if abs(v) > abs(best_val):
                best_val = v
                best_alt = b
        p_ref = float(ref_prob[target_idx])
        p_best = float(p_ref + (best_val if best_alt is not None else 0.0))
        records.append({
            "seq_id": seq_id,
            "position": pos + 1,  # 1-based
            "ref_base": ref,
            "best_alt": best_alt if best_alt is not None else "",
            "ism_score": float(best_val),
            "p_ref": p_ref,
            "p_best": p_best,
            "delta_A": float(deltas["A"]) if not math.isnan(deltas["A"]) else np.nan,
            "delta_C": float(deltas["C"]) if not math.isnan(deltas["C"]) else np.nan,
            "delta_G": float(deltas["G"]) if not math.isnan(deltas["G"]) else np.nan,
            "delta_T": float(deltas["T"]) if not math.isnan(deltas["T"]) else np.nan,
        })
    return pd.DataFrame.from_records(records)


def _sanitize_filename(name: str) -> str:
    bad = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', ' ', '\t', '\n', '\r']
    out = name
    for b in bad:
        out = out.replace(b, '_')
    return out


def _save_plot_ism(
    seq_id: str,
    seq: str,
    df: pd.DataFrame,
    out_dir: str,
    fmt: str = "png",
    width: float = 12.0,
    height: float = 3.0,
) -> Optional[str]:
    if plt is None:
        print("[warn] 未安装 matplotlib，跳过图片生成。可 pip install matplotlib")
        return None

    positions = df["position"].to_numpy()
    scores = df["ism_score"].to_numpy()
    L = len(scores)
    if L == 0:
        return None

    vmax = float(np.max(np.abs(scores)))
    if vmax <= 0:
        vmax = 1e-9
    norm = scores / (vmax + 1e-9)  # [-1,1]
    cmap = plt.get_cmap("bwr")
    colors = cmap((norm + 1.0) / 2.0)

    fig, axes = plt.subplots(2, 1, figsize=(width, height * 2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax0, ax1 = axes

    # 折线图
    ax0.plot(positions, scores, color="#333333", linewidth=1.5)
    ax0.axhline(0.0, color="#999999", linewidth=1.0)
    ax0.set_ylabel("ISM ΔP(target)")
    ax0.grid(True, color="#eeeeee", linestyle="--", linewidth=0.8, alpha=0.8)
    ax0.set_title(f"ISM importance for {seq_id}")

    # 颜色条（柱形），直观显示正负贡献
    ax1.bar(positions, scores, color=colors, width=1.0, align="center")
    ax1.axhline(0.0, color="#999999", linewidth=1.0)
    ax1.set_xlabel("Position (1-based)")
    ax1.set_ylabel("ΔP")

    # 可选：短序列显示碱基字符
    if L <= 200:
        ticks = positions
        labels = list(seq[:L])
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(labels, fontsize=6, rotation=90)

    # 颜色条图例
    if mpl is not None:
        sm = mpl.cm.ScalarMappable(cmap="bwr", norm=mpl.colors.Normalize(vmin=-vmax, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.025, pad=0.02)
        cbar.set_label("ΔP scale")

    fig.tight_layout()
    fname = f"ism_{_sanitize_filename(seq_id)[:40]}_{_sha1_short(seq)}.{fmt}"
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return fpath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="仅基于 ISM（原位突变）的解释脚本：对每个碱基尝试三种替换，计算指定类别概率的变化"
    )
    parser.add_argument("--artifacts_dir", type=str, default="Cproject/Dbert2_LR/artifacts",
                        help="分类模型工件目录（包含 best_model.pt / label_map.json）")
    parser.add_argument("--model_module_path", type=str, default="Cproject/Dbert2_LR/hybrid_dnabert2_rnn_lstm.py",
                        help="hybrid 模型定义脚本路径")
    parser.add_argument("--model_name", type=str, default="./dnabert2_117m",
                        help="DNABERT-2 基座（本地目录或HF名称）")
    parser.add_argument("--local_files_only", action="store_true", help="仅离线加载模型与分词器")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="推理设备")
    parser.add_argument("--max_length", type=int, default=512, help="token 最大长度（将截断）")
    parser.add_argument("--batch_size", type=int, default=48, help="突变体批量大小")

    parser.add_argument("--input_file", type=str, default=None,
                        help="输入序列文件(.fa/.fasta/.csv/.txt)")
    parser.add_argument("--input_column", type=str, default="sequence",
                        help="当输入为CSV时，序列所在列名")
    parser.add_argument("--seq", type=str, nargs="*", default=None,
                        help="直接通过命令行传入的序列，可多条")

    parser.add_argument("--target_label", type=str, default=None,
                        help="解释的目标类别；可为索引(如 '1')或标签名(如 'Enhancer')；缺省取模型预测类别")

    parser.add_argument("--output_dir", type=str, default="Cproject/Dbert2_LR/artifacts/explanations",
                        help="输出目录，将生成每条序列的 CSV 结果")

    parser.add_argument("--service_url", type=str, default=None,
                        help="可选：若提供，则通过HTTP服务进行推理（用于与现有Web服务对接）")

    # 图片导出
    parser.add_argument("--save_plot", action="store_true", help="是否同时导出每条序列的ISM图片")
    parser.add_argument("--plot_format", type=str, default="png", choices=["png", "svg", "pdf"], help="图片格式")
    parser.add_argument("--plot_width", type=float, default=12.0, help="图片宽度英寸")
    parser.add_argument("--plot_height", type=float, default=3.0, help="图片高度英寸(单子图)")

    args = parser.parse_args()

    device = _device_from_arg(args.device)
    _ensure_dir(args.output_dir)

    # 读取序列
    items = _read_sequences(args.input_file, args.input_column, args.seq)

    # 构造预测器
    if args.service_url:
        predictor = _ServiceModel(args.service_url)
        idx_to_label: Optional[Dict[int, str]] = None
    else:
        predictor = _LocalHybridModel(
            artifacts_dir=args.artifacts_dir,
            model_module_path=args.model_module_path,
            base_model_name=args.model_name,
            device=device,
            local_files_only=args.local_files_only,
            max_length=args.max_length,
        )
        idx_to_label = predictor.idx_to_label  # type: ignore[attr-defined]

    # 逐条序列进行 ISM，并写出 CSV
    all_frames: List[pd.DataFrame] = []
    for sid, seq in tqdm(items, desc="ISM explaining"):
        try:
            df = ism_explain_for_sequence(
                seq_id=sid,
                seq=seq,
                predictor=predictor,
                batch_size=args.batch_size,
                max_length=args.max_length,
                target_label=args.target_label,
                idx_to_label=idx_to_label,
            )
        except Exception as e:  # noqa: E722
            print(f"[error] {sid}: {e}")
            continue

        out_name = f"ism_{sid[:40]}_{_sha1_short(seq)}.csv"
        out_path = os.path.join(args.output_dir, out_name)
        df.to_csv(out_path, index=False)
        print(f"[ok] 写出: {out_path}")
        all_frames.append(df)

        if args.save_plot:
            img_path = _save_plot_ism(
                seq_id=sid,
                seq=seq,
                df=df,
                out_dir=args.output_dir,
                fmt=args.plot_format,
                width=args.plot_width,
                height=args.plot_height,
            )
            if img_path:
                print(f"[ok] 图片: {img_path}")

    # 汇总写一个 all.csv 方便一次性查看
    if all_frames:
        combined = pd.concat(all_frames, axis=0, ignore_index=True)
        combined.to_csv(os.path.join(args.output_dir, "all_ism_results.csv"), index=False)
        print(f"[ok] 汇总写出: {os.path.join(args.output_dir, 'all_ism_results.csv')}")


if __name__ == "__main__":
    main()


