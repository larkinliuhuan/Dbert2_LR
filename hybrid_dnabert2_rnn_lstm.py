import os
import csv
import json
import random
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


# -----------------------
# Utils
# -----------------------

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_label(s: str) -> str:
    s = (s or "").strip()
    low = ''.join(ch for ch in s.lower() if ch.isalnum())
    if low in {"promoter", "promoters"}:
        return "Promoter"
    if low in {"othercre", "other_cres", "othercres", "other", "creother"}:
        return "Other CRE"
    if low in {"noncre", "non_cre", "noncrest", "negative", "non"}:
        return "Non-CRE"
    return s


@dataclass
class TrainArgs:
    csv_path: str = os.path.join("Ndata", "arabidopsis_cre_all.csv")
    output_dir: str = os.path.join("Cbert2_Lr", "artifacts")
    model_name: str = "dnabert2_117m"
    local_files_only: bool = True
    batch_size: int = 8
    val_batch_size: int = 16
    epochs: int = 5
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 512
    seed: int = 42
    use_class_weights: bool = True
    trust_remote_code: bool = True
    freeze_base: bool = False  # 是否冻结DNABERT-2主干


# -----------------------
# Data
# -----------------------
class SeqClsDataset(Dataset):
    def __init__(self, rows: List[Dict], label2id: Dict[str, int]):
        self.rows = rows
        self.label2id = label2id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        seq = r["sequence"].upper()
        label = self.label2id[r["label"]]
        return {"sequence": seq, "label": label, "id": r.get("id", str(idx))}


def read_split_csv(csv_path: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
    train, val, test = [], [], []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = (row.get('sequence') or '').strip()
            if not seq:
                continue
            label = normalize_label(row.get('label', ''))
            split = (row.get('split', 'train') or 'train').strip().lower()
            item = {"id": row.get('id', ''), "sequence": seq, "label": label, "split": split}
            if split in {"val", "valid", "validation"}:
                val.append(item)
            elif split in {"test", "testing"}:
                test.append(item)
            else:
                train.append(item)
    return train, val, test


def build_label_map(train_rows: List[Dict], val_rows: List[Dict], test_rows: List[Dict]) -> Dict[str, int]:
    labels = [r["label"] for r in (train_rows + val_rows + test_rows)]
    uniq = list(dict.fromkeys(labels))
    canonical = ["Promoter", "Other CRE", "Non-CRE"]
    ordered = canonical if all(l in uniq for l in canonical) else uniq
    return {l: i for i, l in enumerate(ordered)}


def collate_fn(batch, tokenizer, max_length: int):
    texts = [b["sequence"] for b in batch]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    ids = [b["id"] for b in batch]
    return enc, labels, ids


# try to import AUROC computation from sklearn; if unavailable, we'll skip AUROC
try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

def compute_macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    eps = 1e-8
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        f1s.append(f1)
    return float(sum(f1s) / len(f1s))


def compute_macro_prf1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Tuple[float, float, float]:
    eps = 1e-8
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    p_macro = float(sum(precisions) / len(precisions)) if precisions else 0.0
    r_macro = float(sum(recalls) / len(recalls)) if recalls else 0.0
    f1_macro = float(sum(f1s) / len(f1s)) if f1s else 0.0
    return p_macro, r_macro, f1_macro


# -----------------------
# Hybrid Model: DNABERT-2 encoder + RNN + LSTM heads
# -----------------------
class HybridDNABERT2(nn.Module):
    def __init__(self, base_model, hidden_size: int, num_labels: int, rnn_hidden: int = 384, lstm_hidden: int = 384, dropout: float = 0.1):
        super().__init__()
        self.base = base_model
        self.rnn = nn.RNN(input_size=hidden_size, hidden_size=rnn_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        merged_dim = (rnn_hidden * 2) + (lstm_hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(merged_dim, merged_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(merged_dim // 2, num_labels),
        )

    @staticmethod
    def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B,S,H), mask: (B,S)
        mask = mask.float().unsqueeze(-1)  # (B,S,1)
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1e-6)  # (B,1)
        return x.sum(dim=1) / denom

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # Some DNABERT-2 builds may ignore return_dict and always return a tuple.
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False,
            return_dict=True,
        )
        # Robustly extract last hidden state from different return types
        if isinstance(outputs, dict):
            seq = outputs.get('last_hidden_state', None)
            if seq is None and len(outputs) > 0:
                # fallback to first value in dict values
                seq = list(outputs.values())[0]
        elif hasattr(outputs, 'last_hidden_state'):
            seq = outputs.last_hidden_state
        else:
            # tuple/list fallback
            seq = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        # RNN branch
        rnn_out, _ = self.rnn(seq)  # (B,S,2*rnn_hidden)
        # LSTM branch
        lstm_out, _ = self.lstm(seq)  # (B,S,2*lstm_hidden)
        # Masked mean pooling for both branches
        pooled_rnn = self.masked_mean_pool(rnn_out, attention_mask)
        pooled_lstm = self.masked_mean_pool(lstm_out, attention_mask)
        feat = torch.cat([pooled_rnn, pooled_lstm], dim=-1)
        feat = self.dropout(feat)
        logits = self.classifier(feat)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return logits, loss


# -----------------------
# Main Train/Eval
# -----------------------

def main(args: TrainArgs):
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # Resolve CSV path
    if not os.path.isabs(args.csv_path):
        cwd_candidate = os.path.abspath(args.csv_path)
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        proj_candidate = os.path.abspath(os.path.join(proj_root, args.csv_path))
        if os.path.exists(cwd_candidate):
            args.csv_path = cwd_candidate
        elif os.path.exists(proj_candidate):
            args.csv_path = proj_candidate

    print(f"Loading CSV: {args.csv_path}")
    train_rows, val_rows, test_rows = read_split_csv(args.csv_path)
    print(f"Rows -> train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")

    # Label map
    label2id = build_label_map(train_rows, val_rows, test_rows)
    id2label = {v: k for k, v in label2id.items()}
    def filt(rows):
        return [r for r in rows if r["label"] in label2id]
    train_rows, val_rows, test_rows = filt(train_rows), filt(val_rows), filt(test_rows)

    with open(os.path.join(args.output_dir, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    # Resolve model dir
    if not os.path.isabs(args.model_name):
        cwd_candidate = os.path.abspath(args.model_name)
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        proj_candidate = os.path.abspath(os.path.join(proj_root, args.model_name))
        if os.path.isdir(cwd_candidate):
            args.model_name = cwd_candidate
        elif os.path.isdir(proj_candidate):
            args.model_name = proj_candidate
    print(f"Loading base model from: {args.model_name} (local_only={args.local_files_only})")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    base_model = AutoModel.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    # Disable flash attention to avoid environment issues if present
    if hasattr(base_model.config, 'use_flash_attn'):
        base_model.config.use_flash_attn = False

    hidden_size = base_model.config.hidden_size

    model = HybridDNABERT2(base_model, hidden_size=hidden_size, num_labels=len(label2id), dropout=0.1)

    if args.freeze_base:
        for p in model.base.parameters():
            p.requires_grad = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_ds = SeqClsDataset(train_rows, label2id)
    val_ds = SeqClsDataset(val_rows, label2id)
    test_ds = SeqClsDataset(test_rows, label2id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length))
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length))
    test_loader = DataLoader(test_ds, batch_size=args.val_batch_size, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length))

    # Class weights
    weight = None
    if args.use_class_weights:
        counts = [0] * len(label2id)
        for r in train_rows:
            counts[label2id[r['label']]] += 1
        total = sum(counts)
        if total > 0:
            inv = [total / (c if c > 0 else 1) for c in counts]
            s = sum(inv)
            norm = [x / s * len(inv) for x in inv]
            weight = torch.tensor(norm, dtype=torch.float, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs if len(train_loader) > 0 else 0
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps) if total_steps > 0 else None

    best_f1 = -1.0
    best_path = os.path.join(args.output_dir, 'best_model.pt')
    metrics_path = os.path.join(args.output_dir, 'metrics.csv')

    with open(metrics_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["epoch", "split", "loss", "accuracy", "precision_macro", "recall_macro", "f1_macro", "auroc_ovr"]) 

    def run_eval(loader):
        total_loss = 0.0
        all_preds, all_labels = [], []
        all_probs = []
        model.eval()
        with torch.no_grad():
            for enc, labels, _ids in loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                labels = labels.to(device)
                logits, loss = model(**enc, labels=labels)
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                all_probs.append(logits.softmax(dim=-1).cpu())
        if all_preds:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            acc = (all_preds == all_labels).float().mean().item()
            p_macro, r_macro, f1_macro = compute_macro_prf1(all_preds, all_labels, num_classes=len(label2id))
            # AUROC one-vs-rest
            auroc = float('nan')
            if roc_auc_score is not None:
                try:
                    probs = torch.cat(all_probs).numpy()
                    y_true = all_labels.numpy()
                    if probs.shape[1] == len(label2id):
                        auroc = float(roc_auc_score(y_true, probs, multi_class='ovr'))
                except Exception:
                    auroc = float('nan')
            avg_loss = total_loss / max(1, len(loader))
            return {
                'loss': avg_loss,
                'accuracy': acc,
                'precision_macro': p_macro,
                'recall_macro': r_macro,
                'f1_macro': f1_macro,
                'auroc_ovr': auroc,
            }
        else:
            return {
                'loss': total_loss / max(1, len(loader)),
                'accuracy': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                'f1_macro': 0.0,
                'auroc_ovr': float('nan'),
            }

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for enc, labels, _ids in train_loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            logits, loss = model(**enc, labels=labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running += loss.item()
        train_loss = running / max(1, len(train_loader))

        # Evaluate on train/val for metrics
        train_metrics = run_eval(train_loader)
        val_metrics = run_eval(val_loader)

        # Overwrite train_metrics loss with running average from training loop for clarity
        train_metrics['loss'] = train_loss

        with open(metrics_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([epoch, 'train', f"{train_metrics['loss']:.6f}", f"{train_metrics['accuracy']:.6f}", f"{train_metrics['precision_macro']:.6f}", f"{train_metrics['recall_macro']:.6f}", f"{train_metrics['f1_macro']:.6f}", f"{train_metrics['auroc_ovr']:.6f}" if not math.isnan(train_metrics['auroc_ovr']) else 'nan'])
            w.writerow([epoch, 'val', f"{val_metrics['loss']:.6f}", f"{val_metrics['accuracy']:.6f}", f"{val_metrics['precision_macro']:.6f}", f"{val_metrics['recall_macro']:.6f}", f"{val_metrics['f1_macro']:.6f}", f"{val_metrics['auroc_ovr']:.6f}" if not math.isnan(val_metrics['auroc_ovr']) else 'nan'])

        auroc_train_str = 'nan' if math.isnan(train_metrics['auroc_ovr']) else f"{train_metrics['auroc_ovr']:.4f}"
        auroc_val_str = 'nan' if math.isnan(val_metrics['auroc_ovr']) else f"{val_metrics['auroc_ovr']:.4f}"
        print(f"Epoch {epoch}/{args.epochs} TRAIN | loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} P_macro={train_metrics['precision_macro']:.4f} R_macro={train_metrics['recall_macro']:.4f} F1_macro={train_metrics['f1_macro']:.4f} AUROC_OVR={auroc_train_str}", flush=True)
        print(f"Epoch {epoch}/{args.epochs}   VAL | loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} P_macro={val_metrics['precision_macro']:.4f} R_macro={val_metrics['recall_macro']:.4f} F1_macro={val_metrics['f1_macro']:.4f} AUROC_OVR={auroc_val_str}", flush=True)

        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': base_model.config.to_dict(),
                'label2id': label2id,
                'id2label': id2label,
            }, best_path)

    # Test evaluation with best
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Use the same eval to compute full test metrics
    test_metrics = run_eval(test_loader)

    # Export y_true and y_score (per-class probabilities) for test set to AUROC.csv
    def _safe_col(s: str) -> str:
        return ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)

    num_classes = len(label2id)
    probs_cols = [f"score_{_safe_col(id2label[i])}" for i in range(num_classes)]
    auroc_csv_path = os.path.join(args.output_dir, 'AUROC_LR.csv')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(auroc_csv_path, 'w', newline='', encoding='utf-8') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['id', 'y_true', 'y_true_label'] + probs_cols)
        model.eval()
        with torch.no_grad():
            for enc, labels, ids in test_loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                logits, _ = model(**enc, labels=None)
                probs = logits.softmax(dim=-1).cpu().numpy()
                labels_np = labels.numpy()
                for j in range(len(ids)):
                    y_true_idx = int(labels_np[j])
                    row = [ids[j], y_true_idx, id2label[y_true_idx]] + [float(probs[j, k]) for k in range(num_classes)]
                    writer.writerow(row)

    with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({
            "test_loss": test_metrics['loss'],
            "test_acc": test_metrics['accuracy'],
            "test_precision_macro": test_metrics['precision_macro'],
            "test_recall_macro": test_metrics['recall_macro'],
            "test_macro_f1": test_metrics['f1_macro'],
            "test_auroc_ovr": None if math.isnan(test_metrics['auroc_ovr']) else test_metrics['auroc_ovr'],
        }, f, ensure_ascii=False, indent=2)

    with open(metrics_path, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow(['final', 'test', f"{test_metrics['loss']:.6f}", f"{test_metrics['accuracy']:.6f}", f"{test_metrics['precision_macro']:.6f}", f"{test_metrics['recall_macro']:.6f}", f"{test_metrics['f1_macro']:.6f}", f"{test_metrics['auroc_ovr']:.6f}" if not math.isnan(test_metrics['auroc_ovr']) else 'nan'])

    test_auroc_str = 'nan' if math.isnan(test_metrics['auroc_ovr']) else f"{test_metrics['auroc_ovr']:.4f}"
    print(f"Done. Best val_macro_f1={best_f1:.4f}. Test: loss={test_metrics['loss']:.4f}, acc={test_metrics['accuracy']:.4f}, macroF1={test_metrics['f1_macro']:.4f}, AUROC_OVR={test_auroc_str}")


if __name__ == "__main__":
    args = TrainArgs()
    # Env overrides
    args.csv_path = os.getenv('NP_CSV', args.csv_path)
    args.output_dir = os.getenv('NP_OUT', args.output_dir)
    args.model_name = os.getenv('NP_MODEL', args.model_name)
    args.batch_size = int(os.getenv('NP_BS', args.batch_size))
    args.val_batch_size = int(os.getenv('NP_VBS', args.val_batch_size))
    args.epochs = int(os.getenv('NP_EPOCHS', args.epochs))
    args.lr = float(os.getenv('NP_LR', args.lr))
    args.max_length = int(os.getenv('NP_MAXLEN', args.max_length))
    args.local_files_only = os.getenv('NP_LOCAL_ONLY', '1') not in {'0', 'false', 'False'}
    args.trust_remote_code = os.getenv('NP_TRUST_REMOTE', '1') not in {'0', 'false', 'False'}
    args.freeze_base = os.getenv('NP_FREEZE_BASE', '0') in {'1', 'true', 'True'}

    main(args)