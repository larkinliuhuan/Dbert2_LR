import os, json, argparse, random
from typing import List, Tuple, Dict, Optional

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 可选：图片绘制依赖
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
except Exception:  # 允许无图环境
    plt = None
    mpl = None

import math
import hashlib
import re

# 本脚本重写为精简且稳定的 ISM 解释脚本，统一使用 4 空格缩进，避免缩进错误。
# 仅依赖本仓库已有的 HybridDNABERT2 定义与已训练权重。
from hybrid_dnabert2_rnn_lstm import HybridDNABERT2


HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _sha1_short(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _resolve_path_maybe_relative(p: Optional[str]) -> Optional[str]:
    """把可能的相对路径解析为可用的绝对路径：按 CWD、脚本目录、项目根依次尝试。"""
    if not p:
        return p
    if os.path.exists(p):
        return p
    candidates = [
        os.path.abspath(os.path.join(os.getcwd(), p)),
        os.path.abspath(os.path.join(HERE, p)),
        os.path.abspath(os.path.join(PROJECT_ROOT, p)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return p


def _default_artifact_candidates() -> List[str]:
    return [
        os.path.join(HERE, "artifacts", "artifacts"),
        os.path.join(PROJECT_ROOT, "Nproject", "artifacts"),
        os.path.join(PROJECT_ROOT, "Cproject", "Dert2", "artifacts"),
    ]


def _find_artifacts_dir(user_path: Optional[str]) -> str:
    paths = []
    if user_path:
        paths.append(_resolve_path_maybe_relative(user_path))
    paths.extend(_default_artifact_candidates())

    tried = []
    for p in paths:
        if not p:
            continue
        lm = os.path.join(p, "label_map.json")
        bm = os.path.join(p, "best_model.pt")
        tried.append(p)
        if os.path.exists(lm) and os.path.exists(bm):
            return p

    raise FileNotFoundError(
        "未找到可用的 artifacts_dir。请通过 --artifacts_dir 指定，或将 best_model.pt 与 label_map.json 放在以下任一路径下。\n" +
        "尝试过: " + ", ".join(tried)
    )

# 新增：选择空闲显存最多的GPU
def _pick_best_cuda_device() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return torch.device("cpu")
    best_idx = 0
    best_free = -1
    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)  # bytes
        except Exception:
            free, total = (0, 0)
        if free > best_free:
            best_free = free
            best_idx = i
    return torch.device(f"cuda:{best_idx}")


def _resolve_model_dir(model_name: str) -> str:
    """当使用本地目录作为模型名时，允许相对脚本或项目根的路径。"""
    if os.path.isdir(model_name):
        return model_name
    # 尝试项目根
    cand = os.path.join(PROJECT_ROOT, model_name)
    if os.path.isdir(cand):
        return cand
    # 尝试脚本目录
    cand = os.path.join(HERE, model_name)
    if os.path.isdir(cand):
        return cand
    return model_name


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
    csv_fallback: Optional[str] = None,
    csv_sequence_column: str = "sequence",
    csv_sample_n: Optional[int] = None,
    csv_seed: int = 42,
    excel_sheet: Optional[str] = None,
    id_column: str = "id",
    id_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """返回 (seq_id, sequence) 列表。优先使用 input_file/seq；若两者都未提供且给出 csv_fallback 则从 CSV 读取。支持 .xlsx/.xls。"""
    
    def _guess_sequence_from_text(text: object) -> Optional[str]:
        if text is None:
            return None
        s = str(text).strip()
        if not s:
            return None
        # 拆分常见分隔符，并尝试在每段中寻找最长的 A/C/G/T/U 片段
        parts = re.split(r"[\s,;\|/\t]", s) + [s]
        best = ""
        for p in parts:
            for m in re.finditer(r"[ACGTUacgtu]+", str(p)):
                cand = m.group(0).upper().replace("U", "T")
                if len(cand) > len(best):
                    best = cand
        if len(best) >= 8:  # 最少长度阈值
            return best
        return None

    def _extract_sequences_from_dataframe(df: pd.DataFrame, prefer_col: str) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        if df is None or len(df) == 0:
            return pairs
        # 候选列名：优先用户指定，其次常见同义词
        synonyms = [prefer_col, "sequence", "Sequence", "seq", "Seq", "dna", "promoter", "Promoter", "序列"]
        # 去重同时保持顺序
        seen = set()
        cand_cols = [c for c in synonyms if c not in seen and not seen.add(c) and (isinstance(df.columns, pd.Index) and c in df.columns)]
        # 如果没有匹配列，尝试自动选择包含最多 DNA 片段的列
        if not cand_cols and hasattr(df, "columns"):
            best_col = None
            best_count = 0
            for c in df.columns:
                try:
                    series = df[c]
                    count = 0
                    for v in series:
                        if _guess_sequence_from_text(v):
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_col = c
                except Exception:
                    pass
            if best_col is not None and best_count > 0:
                cand_cols = [best_col]
        # 从候选列抽取
        for col in cand_cols:
            for i, row in df.iterrows():
                raw = row[col]
                seq = _guess_sequence_from_text(raw)
                if seq:
                    sid = str(row.get("id", row.get("ID", row.get("name", row.get("Name", f"row_{i+1}")))))
                    pairs.append((sid, seq))
            if pairs:
                return pairs
        # header=None 情况：取第一列尝试
        if not pairs and df.shape[1] >= 1:
            col0 = df.columns[0]
            for i, row in df.iterrows():
                seq = _guess_sequence_from_text(row[col0])
                if seq:
                    sid = f"row_{i+1}"
                    pairs.append((sid, seq))
        return pairs

    results: List[Tuple[str, str]] = []

    # 优先：通用输入文件
    if input_file:
        input_file = _resolve_path_maybe_relative(input_file)
        lower = input_file.lower()
        if lower.endswith(".fa") or lower.endswith(".fasta"):
            results.extend(_parse_fasta(input_file))
        elif lower.endswith(".csv"):
            df = pd.read_csv(input_file)
            # 基于ID过滤（如指定）
            if id_filter is not None and len(id_filter) > 0:
                if id_column not in df.columns:
                    raise KeyError(f"CSV中未找到ID列: {id_column}")
                df = df[df[id_column].astype(str).isin(set(id_filter))].reset_index(drop=True)
            pairs = _extract_sequences_from_dataframe(df, input_column)
            if not pairs:
                # 尝试无表头读取
                df2 = pd.read_csv(input_file, header=None)
                if id_filter is not None and len(id_filter) > 0:
                    # 无表头时无法按列名过滤，直接走启发式列提取
                    pass
                pairs = _extract_sequences_from_dataframe(df2, input_column)
            if not pairs:
                raise KeyError(
                    f"CSV中未找到列: {input_column}，且启发式解析失败。当前列: {list(df.columns)}"
                )
            results.extend(pairs)
        elif lower.endswith(".txt"):
            with open(input_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    results.append((f"line_{i+1}", line))
        elif lower.endswith(".xlsx") or lower.endswith(".xls"):
            # 先按提供/默认工作表读取
            try:
                df = pd.read_excel(input_file, sheet_name=excel_sheet if excel_sheet else 0)
                if id_filter is not None and len(id_filter) > 0 and id_column in getattr(df, 'columns', []):
                    df = df[df[id_column].astype(str).isin(set(id_filter))].reset_index(drop=True)
                pairs = _extract_sequences_from_dataframe(df, input_column)
            except Exception:
                pairs = []
            # 若失败，遍历所有工作表与 header=None 的组合进行启发式解析
            if not pairs:
                xls = pd.ExcelFile(input_file)
                for sh in xls.sheet_names:
                    df_try = xls.parse(sheet_name=sh)
                    if id_filter is not None and len(id_filter) > 0 and id_column in getattr(df_try, 'columns', []):
                        df_try = df_try[df_try[id_column].astype(str).isin(set(id_filter))].reset_index(drop=True)
                    pairs = _extract_sequences_from_dataframe(df_try, input_column)
                    if pairs:
                        break
                    df_try2 = xls.parse(sheet_name=sh, header=None)
                    # 无表头无法按列名过滤，这里仅进行启发式提取
                    pairs = _extract_sequences_from_dataframe(df_try2, input_column)
                    if pairs:
                        break
            if not pairs:
                raise KeyError(
                    f"Excel中未找到列: {input_column}，且在所有工作表上启发式解析失败。请用 --input_sheet 或 --input_column 指定。"
                )
            results.extend(pairs)
        else:
            raise ValueError("不支持的输入文件类型，仅支持 .fa/.fasta/.csv/.txt/.xlsx/.xls")

    # 其次：命令行序列
    if sequences:
        for s in sequences:
            sid = f"seq_{_sha1_short(s)}"
            results.append((sid, s))

    # 兜底：兼容旧的 csv_path+sequence_column
    if (not results) and csv_fallback:
        csv_fallback = _resolve_path_maybe_relative(csv_fallback)
        df = pd.read_csv(csv_fallback)
        if id_filter is not None and len(id_filter) > 0:
            if id_column not in df.columns:
                raise KeyError(f"CSV中未找到ID列: {id_column}")
            df = df[df[id_column].astype(str).isin(set(id_filter))].reset_index(drop=True)
        if csv_sequence_column not in df.columns:
            raise KeyError(f"CSV中未找到列: {csv_sequence_column}")
        if csv_sample_n is not None and len(df) > csv_sample_n:
            df = df.sample(n=csv_sample_n, random_state=csv_seed).reset_index(drop=True)
        for i, row in df.iterrows():
            sid = str(row.get("id", f"row_{i+1}"))
            seq = str(row[csv_sequence_column])
            results.append((sid, seq))

    if id_filter is not None and len(id_filter) > 0 and not results:
        raise ValueError("按给定ID未匹配到任何序列，请检查 --id_column 与 --ids 是否正确。")

    if not results:
        raise ValueError("未提供任何序列。请使用 --input_file 或 --seq，或提供 --csv_path 作为兜底。")

    # 规范化
    normed: List[Tuple[str, str]] = []
    for sid, s in results:
        s2 = s.strip().upper().replace("U", "T")
        normed.append((sid, s2))
    return normed


class LocalPredictor:
    def __init__(self, artifacts_dir: str, model_name: str, device: str = "auto", local_files_only: bool = True, max_length: int = 512):
        self.artifacts_dir = artifacts_dir
        self.max_length = int(max_length)
        if device in ("auto", "auto_free"):
            if torch.cuda.is_available():
                self.device = _pick_best_cuda_device()
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # label map
        lm_path = os.path.join(artifacts_dir, "label_map.json")
        with open(lm_path, "r", encoding="utf-8") as f:
            lm = json.load(f)
        self.label2id: Dict[str, int] = lm.get("label2id", {})
        self.id2label: Dict[int, str] = {int(k): v for k, v in lm.get("id2label", {}).items()}

        # base model + hybrid head
        model_name_resolved = _resolve_model_dir(model_name)
        base = AutoModel.from_pretrained(model_name_resolved, local_files_only=local_files_only, trust_remote_code=True)
        if hasattr(base.config, "use_flash_attn"):
            base.config.use_flash_attn = False
        hidden = base.config.hidden_size
        self.model = HybridDNABERT2(base, hidden_size=hidden, num_labels=len(self.label2id))
        
        # 兼容多种 checkpoint 保存格式（state_dict 直存 或 包含 model_state_dict 的字典）
        ckpt_path = os.path.join(artifacts_dir, "best_model.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        # 宽松加载，忽略与当前定义不相关的键（如 config/label2id/id2label 等元信息）
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

        # 记录所选设备与显存情况（可选）
        try:
            if self.device.type == "cuda":
                free, total = torch.cuda.mem_get_info(self.device)
                print(f"[INFO] 使用设备 {self.device}; 显存 可用/总计={free/(1024**3):.1f}/{total/(1024**3):.1f} GiB")
            else:
                print("[INFO] 使用CPU运行")
        except Exception:
            pass

        self.tok = AutoTokenizer.from_pretrained(model_name_resolved, local_files_only=local_files_only, trust_remote_code=True)

    @torch.no_grad()
    def predict_proba(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        probs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits, _ = self.model(**enc, labels=None)
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            probs.append(p)
        return np.concatenate(probs, axis=0) if probs else np.zeros((0, len(self.id2label)))


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


def _sanitize_filename(name: str) -> str:
    bad = ['\\\n', '\\', '/', ':', '*', '?', '"', '<', '>', '|', ' ', '\t', '\n', '\r']
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


def ism_explain_for_sequence(
    seq_id: str,
    seq: str,
    predictor: LocalPredictor,
    batch_size: int,
    target_label: Optional[str],
    idx_to_label: Optional[Dict[int, str]],
) -> pd.DataFrame:
    # 参考概率
    ref_prob = predictor.predict_proba([seq], batch_size=batch_size)[0]

    # 解析目标索引
    if target_label is None:
        target_idx = int(np.argmax(ref_prob))
    else:
        # 支持索引与名称
        if target_label.isdigit():
            target_idx = int(target_label)
        else:
            # 名称 -> 索引
            if idx_to_label is not None:
                name_to_idx = {v: k for k, v in idx_to_label.items()}
                if target_label not in name_to_idx:
                    raise ValueError(f"target_label 不在 label_map 中: {target_label}")
                target_idx = int(name_to_idx[target_label])
            else:
                raise ValueError("未提供 label_map 信息，无法根据名称解析 target_label")

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
            if (b == ref) or (b not in deltas) or (deltas[b] is None) or (isinstance(deltas[b], float) and math.isnan(deltas[b])):
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
            "delta_A": float(deltas.get("A", np.nan)) if not (ref == "A") else np.nan,
            "delta_C": float(deltas.get("C", np.nan)) if not (ref == "C") else np.nan,
            "delta_G": float(deltas.get("G", np.nan)) if not (ref == "G") else np.nan,
            "delta_T": float(deltas.get("T", np.nan)) if not (ref == "T") else np.nan,
        })
    return pd.DataFrame.from_records(records)


def parse_args():
    ap = argparse.ArgumentParser("ISM for Arabidopsis CRE (enhanced, with plotting & flexible input)")
    # 模型与设备
    ap.add_argument("--artifacts_dir", default=None, help="包含 best_model.pt 与 label_map.json 的目录；留空将自动探测")
    ap.add_argument("--model_name", default="dnabert2_117m")
    ap.add_argument("--device", default="auto", help="设备选择：'cpu'/'cuda:0'等；'auto'会自动选择可用显存最多的GPU")
    ap.add_argument("--local_files_only", action="store_true", default=True)
    ap.add_argument("--max_length", type=int, default=512)

    # ISM 推理批量
    ap.add_argument("--batch_size", type=int, default=48)

    # 输入：默认使用 Ndata/arabidopsis_cre_all.csv，并按 id 列筛选两条示例
    ap.add_argument("--input_file", type=str, default=os.path.join("Ndata", "arabidopsis_cre_all.csv"), help="输入序列文件(.fa/.fasta/.csv/.txt/.xlsx/.xls)")
    ap.add_argument("--input_column", type=str, default="sequence", help="当输入为CSV/Excel时，序列所在列名")
    ap.add_argument("--input_sheet", type=str, default=None, help="当输入为Excel时的表名（留空取首个工作表）")
    ap.add_argument("--seq", type=str, nargs="*", default=None, help="直接通过命令行传入的序列，可多条")

    ap.add_argument("--csv_path", default=os.path.join("Ndata", "arabidopsis_cre_all.csv"))
    ap.add_argument("--sequence_column", default="sequence", help="CSV中序列所在列名")
    ap.add_argument("--sample_ep20", action="store_true", help="随机抽样总计 20 条用于解释（仅 csv_path 兜底时生效）")
    ap.add_argument("--seed", type=int, default=42)

    # 基于ID的筛选
    ap.add_argument("--id_column", type=str, default="id", help="用于按ID筛选的列名")
    ap.add_argument(
        "--ids", type=str, nargs="*",
        default=[
            # 第一组（10 条）
            "AT4G17520_1;TSS=9771332;strand=+",
            "AT5G12910_1;TSS=4077187;strand=+",
            "AT1G17050_1;TSS=5829100;strand=+",
            "AT5G62560_1;TSS=25110007;strand=+",
            "AT1G14455_1;TSS=4948345;strand=-",
            "AT2G27860_1;TSS=11866901;strand=-",
            "AT1G28520_1;TSS=10029193;strand=+",
            "AT3G21220_1;TSS=7445729;strand=+",
            "AT1G67775_1;TSS=25411736;strand=-",
            "AT4G11580_1;TSS=7007786;strand=-",
            # 第二组（13 条）
            "AT5G67110_1;TSS=26786376;strand=-",
            "AT2G30460_1;TSS=12978885;strand=-",
            "AT3G21970_1;TSS=7742433;strand=+",
            "AT5G67320_1;TSS=26857155;strand=+",
            "AT3G28450_1;TSS=10667292;strand=+",
            "AT1G28305_1;TSS=9905516;strand=-",
            "AT2G34980_1;TSS=14748889;strand=+",
            "AT1G18410_1;TSS=6342472;strand=-",
            "AT4G20410_1;TSS=11016556;strand=-",
            "AT3G54826_1;TSS=20310398;strand=+",
            "AT1G75100_1;TSS=28193782;strand=-",
            "AT5G14660_1;TSS=4728698;strand=-",
            "AT1G23720_1;TSS=8388522;strand=+",
        ],
        help="仅处理给定ID列表；默认包含 23 条ID。若要处理全部行，请使用 --no_id_filter"
    )
    ap.add_argument("--no_id_filter", action="store_true", help="忽略 --ids，处理所有行")

    # 目标类别
    ap.add_argument("--target_label", default=None, help="目标类别（名称或索引）；留空则取模型预测 argmax")

    # 输出与绘图
    ap.add_argument("--output_dir", default=os.path.join("Dbert2_LR", "artifacts", "ism_outputs"))
    ap.add_argument("--save_plot", dest="save_plot", action="store_true", default=True, help="是否同时导出每条序列的ISM图片（默认开启，可用 --no_save_plot 关闭）")
    ap.add_argument("--no_save_plot", dest="save_plot", action="store_false", help="关闭图片生成")
    ap.add_argument("--plot_format", type=str, default="png", choices=["png", "svg", "pdf"], help="图片格式")
    ap.add_argument("--plot_width", type=float, default=12.0, help="图片宽度英寸")
    ap.add_argument("--plot_height", type=float, default=3.0, help="图片高度英寸(单子图)")

    return ap.parse_args()


def main():
    args = parse_args()

    # 自动解析 artifacts 目录
    artifacts_dir = _find_artifacts_dir(args.artifacts_dir)

    _ensure_dir(_resolve_path_maybe_relative(args.output_dir))

    # 构造预测器（本地）
    predictor = LocalPredictor(
        artifacts_dir=artifacts_dir,
        model_name=args.model_name,
        device=args.device,
        local_files_only=args.local_files_only,
        max_length=args.max_length,
    )

    # 读取序列：优先 input_file/seq；否则使用 csv_path 兜底
    sample_n = 20 if args.sample_ep20 else None
    items = _read_sequences(
        input_file=args.input_file,
        input_column=args.input_column,
        sequences=args.seq,
        csv_fallback=args.csv_path,
        csv_sequence_column=args.sequence_column,
        csv_sample_n=sample_n,
        csv_seed=args.seed,
        excel_sheet=args.input_sheet,
        id_column=args.id_column,
        id_filter=None if args.no_id_filter else args.ids,
    )

    print(f"[INFO] 共解析到 {len(items)} 条序列。示例ID: {[sid for sid,_ in items[:5]]}")

    out_dir_resolved = _resolve_path_maybe_relative(args.output_dir)

    # 逐条序列进行 ISM，并写出 CSV/图片
    all_frames: List[pd.DataFrame] = []
    for sid, seq in items:
        df = ism_explain_for_sequence(
            seq_id=sid,
            seq=seq,
            predictor=predictor,
            batch_size=args.batch_size,
            target_label=args.target_label,
            idx_to_label=predictor.id2label,
        )
        # 确保按位置升序输出，便于查看
        df = df.sort_values(["position"], ascending=True).reset_index(drop=True)
        out_name = f"ism_{_sanitize_filename(sid)[:40]}_{_sha1_short(seq)}.csv"
        out_path = os.path.join(out_dir_resolved, out_name)
        df.to_csv(out_path, index=False)
        print(f"[OK] 写出: {out_path}")
        all_frames.append(df)

        if args.save_plot:
            try:
                img_path = _save_plot_ism(
                    seq_id=sid,
                    seq=seq,
                    df=df,
                    out_dir=out_dir_resolved,
                    fmt=args.plot_format,
                    width=args.plot_width,
                    height=args.plot_height,
                )
                if img_path:
                    print(f"[OK] 图片: {img_path}")
                else:
                    print(f"[WARN] 跳过图片（无数据或未安装matplotlib）: {sid}")
            except Exception as e:
                print(f"[ERROR] Plotting failed for seq_id '{sid}': {e}")

    # 汇总写一个 all.csv 方便一次性查看
    if all_frames:
        combined = pd.concat(all_frames, axis=0, ignore_index=True)
        combined = combined.sort_values(["seq_id", "position"], ascending=[True, True]).reset_index(drop=True)
        combined.to_csv(os.path.join(out_dir_resolved, "all_ism_results.csv"), index=False)
        print(f"[OK] 汇总写出: {os.path.join(out_dir_resolved, 'all_ism_results.csv')}")


if __name__ == "__main__":
    main()