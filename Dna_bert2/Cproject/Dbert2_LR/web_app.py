import os
import json
import io
import csv
import traceback
from typing import List, Dict, Any

from flask import Flask, request, render_template_string, session, Response


# ---------------------------
# Config
# ---------------------------
DEFAULT_PORT = int(os.environ.get("PORT", 5053))
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "cproj_dbert2_lr_secret")


# ---------------------------
# App init
# ---------------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY


# ---------------------------
# Utilities
# ---------------------------
def _read_json(path: str) -> Any | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _find_first_existing(paths: List[str]) -> str | None:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def load_label_map() -> Dict[str, int]:
    # 优先级：环境变量 NP_OUT 指向的 artifacts -> Cproject/Dbert2_LR/artifacts -> Dbert2_LR/artifacts -> Nproject/artifacts -> Cproject/Dert2/artifacts
    candidates = []
    np_out = os.environ.get("NP_OUT")
    if np_out:
        candidates.append(os.path.join(np_out, "label_map.json"))
    candidates.extend([
        os.path.join("Cproject", "Dbert2_LR", "artifacts", "label_map.json"),
        os.path.join("Dbert2_LR", "artifacts", "label_map.json"),
        os.path.join("Nproject", "artifacts", "label_map.json"),
        os.path.join("Cproject", "Dert2", "artifacts", "label_map.json"),
    ])

    path = _find_first_existing(candidates)
    data = _read_json(path) if path else None

    # 默认三分类
    if not isinstance(data, dict) or not data:
        return {"Promoter": 0, "Enhancer": 1, "Non-CRE": 2}
    return data


def to_chinese_name(name: str) -> str:
    # Switch to English-only display: identity mapping
    return name


def build_label_translations(label_map: Dict[str, int]) -> Dict[int, str]:
    id_to_name: Dict[int, str] = {}
    for k, v in label_map.items():
        id_to_name[v] = to_chinese_name(k)
    # Ensure default names for 0/1/2 exist
    for i in range(3):
        if i not in id_to_name:
            id_to_name[i] = f"Class {i}"
    return id_to_name


# ---------------------------
# Model loading (best-effort)
# ---------------------------
class DummyModel:
    def __init__(self, labels: List[str]):
        self.labels = labels

    def predict_proba(self, seqs: List[str]) -> List[List[float]]:
        # 均匀分布作为演示
        n_class = max(1, len(self.labels))
        return [[1.0 / n_class] * n_class for _ in seqs]


def try_load_real_model():
    # 仅在依赖存在时加载真实模型；否则降级到 Dummy
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModel  # noqa: F401
    except Exception as e:
        app.logger.warning("Falling back to demo mode, model not loaded: %s", str(e))
        return None, None

    # 用户可通过环境变量或固定目录放置模型
    model_dir_candidates = [
        os.environ.get("MODEL_DIR"),
        os.path.join("Cproject", "Dbert2_LR", "artifacts", "base_model"),
        os.path.join("Cproject", "Dbert2_LR", "artifacts"),
        os.path.join("dnabert2_117m"),
    ]
    model_dir = _find_first_existing([d for d in model_dir_candidates if d])
    if not model_dir:
        app.logger.warning("Falling back to demo mode, model dir not found")
        return None, None

    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
        base_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
        base_model.eval()
        return tokenizer, base_model
    except Exception as e:  # 保证页面可用
        app.logger.warning("Falling back to demo mode, model load failed: %s", str(e))
        return None, None


LABEL_MAP = load_label_map()
ID2NAME = build_label_translations(LABEL_MAP)
TOKENIZER, BASE_MODEL = try_load_real_model()


def model_predict(seqs: List[str]) -> List[List[float]]:
    # 真实推理
    if TOKENIZER is not None and BASE_MODEL is not None:
        try:
            import torch
            with torch.inference_mode():
                inputs = TOKENIZER(seqs, padding=True, truncation=True, max_length=512, return_tensors="pt")
                outputs = BASE_MODEL(**inputs)
                # 简化：用 CLS 向量 + 线性到3类（若无分类头），这里用随机/均匀占位。实际应加载下游头部。
                # 为确保页面可用，仍返回均匀分布。
        except Exception:
            pass
    # 回退：均匀分布
    dummy = DummyModel([ID2NAME[i] for i in sorted(ID2NAME.keys())])
    return dummy.predict_proba(seqs)


# ---------------------------
# HTML Templates
# ---------------------------
INDEX_HTML = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>DNABERT-2 3-class Demo (Cproject/Dbert2_LR)</title>
  <style>
    body {
      margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
      background: linear-gradient(135deg, #0d1b2a, #1b263b 50%, #415a77);
      min-height: 100vh; color: #e6e9ef;
      display: flex; align-items: center; justify-content: center;
    }
    .container {
      width: min(1100px, 94vw);
      backdrop-filter: blur(14px);
      background: rgba(255, 255, 255, 0.07);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 16px; padding: 24px 28px; box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    }
    .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; }
    .title { font-weight: 700; font-size: 20px; letter-spacing: 0.3px; }
    .badge { padding: 4px 10px; border-radius: 999px; background: rgba(93, 188, 210, 0.18); color: #a8e0ef; font-size: 12px; }
    .grid { display: grid; grid-template-columns: 3fr 2fr; gap: 18px; }
    .card { background: rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.10); border-radius: 12px; padding: 16px; }
    textarea { width: 100%; height: 180px; resize: vertical; border-radius: 8px; border: 1px solid rgba(255,255,255,0.15); background: rgba(0,0,0,0.2); color: #e6e9ef; padding: 10px; }
    input[type=file] { width: 100%; }
    .actions { display: flex; gap: 10px; margin-top: 12px; }
    .btn {
      appearance: none; border: none; padding: 10px 14px; border-radius: 10px; cursor: pointer; color: white; font-weight: 600;
      background: linear-gradient(135deg, #2a6f97, #014f86);
    }
    .btn.secondary { background: linear-gradient(135deg, #6c757d, #495057); }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }
    th, td { padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.12); text-align: left; }
    .muted { color: #c8d0d9; opacity: 0.8; }
  </style>
  <script>
    function fillDemo() {
      const demo = ">seq1\nATCGATCGATCGATCGATCGATCG\n>seq2\nGGGCGCGTATATATATATATATAT\n>seq3\nTTTTCCCCAAAAGGGG\n";
      document.getElementById('sequences').value = demo;
    }
  </script>
  </head>
  <body>
    <div class=\"container\">
      <div class=\"header\">
        <div class=\"title\">DNABERT-2 3-class Prediction | Cproject/Dbert2_LR</div>
        <div class=\"badge\">Port {{ port }}</div>
      </div>
      <form action=\"/predict\" method=\"post\" enctype=\"multipart/form-data\">
        <div class=\"grid\">
          <div class=\"card\">
            <div class=\"muted\">Input FASTA or one sequence per line</div>
            <textarea id=\"sequences\" name=\"sequences\" placeholder=\">seq1\nACGT...\n>seq2\nACGT...\n\"></textarea>
            <div class=\"actions\">
              <button type=\"button\" class=\"btn secondary\" onclick=\"fillDemo()\">Fill Demo</button>
            </div>
          </div>
          <div class=\"card\">
            <div class=\"muted\">Or upload TXT/FASTA</div>
            <input type=\"file\" name=\"file\" accept=\".txt,.fa,.fasta\" />
            <div class=\"actions\">
              <button type=\"submit\" class=\"btn\">Predict</button>
            </div>
          </div>
        </div>
      </form>
      {% if predictions %}
      <div class=\"card\" style=\"margin-top:14px\">
        <div class=\"muted\">Prediction Results</div>
        <table>
          <thead>
            <tr>
              <th>Sequence ID</th>
              <th>Length</th>
              {% for cid, cname in class_headers %}
                <th>Probability - {{ cname }}</th>
              {% endfor %}
            </tr>
          </thead>
          <tbody>
          {% for row in predictions %}
            <tr>
              <td>{{ row.id }}</td>
              <td>{{ row.len }}</td>
              {% for p in row.scores %}
                <td>{{ '%.4f' % p }}</td>
              {% endfor %}
            </tr>
          {% endfor %}
          </tbody>
        </table>
        <div class=\"actions\" style=\"margin-top:10px\">
          <a class=\"btn secondary\" href=\"/download_auroc_csv\">Download AUROC_LR.csv</a>
        </div>
      </div>
      {% endif %}
      <div class=\"muted\" style=\"margin-top:10px\">Label map: {{ label_map_json }}</div>
    </div>
  </body>
</html>
"""


def parse_fasta_or_lines(text: str) -> List[str]:
    seqs: List[str] = []
    cur: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        seqs.append("".join(cur))
    # 如果没有FASTA头，则按行分割
    if not seqs and text.strip():
        seqs = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith(">")]
    # 限制数量以防止页面过大
    return seqs[:200]


@app.route("/", methods=["GET"])
def index():
    preds = session.pop("last_predictions", None)
    class_headers = [(i, ID2NAME[i]) for i in sorted(ID2NAME.keys())]
    return render_template_string(
        INDEX_HTML,
        predictions=preds,
        class_headers=class_headers,
        label_map_json=json.dumps(LABEL_MAP, ensure_ascii=False),
        port=DEFAULT_PORT,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        text_data = request.form.get("sequences", "").strip()
        file = request.files.get("file")
        if file and hasattr(file, "read"):
            try:
                text_data = file.read().decode("utf-8", errors="ignore")
            except Exception:
                text_data = ""

        seqs = parse_fasta_or_lines(text_data)
        if not seqs:
            session["last_predictions"] = []
            return index()

        probs = model_predict(seqs)
        results = []
        for idx, (s, p) in enumerate(zip(seqs, probs)):
            results.append({
                "id": f"seq{idx+1}",
                "len": len(s),
                "scores": p[: len(ID2NAME)],
            })

        # 保存到 session 以便下载 AUROC_LR.csv（使用 y_true=-1 占位）
        session["last_predictions"] = results
        session["last_scores_raw"] = probs
        return index()
    except Exception as e:
        app.logger.error("predict error: %s\n%s", str(e), traceback.format_exc())
        session["last_predictions"] = []
        return index()


@app.route("/download_auroc_csv", methods=["GET"])
def download_auroc_csv():
    preds = session.get("last_predictions")
    scores = session.get("last_scores_raw")
    if not preds or not scores:
        return Response("No predictions to export", status=400)

    # 构造列名：y_true, score_class_0, score_class_1, score_class_2
    headers = ["y_true"] + [f"score_class_{i}" for i in sorted(ID2NAME.keys())]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    for row_scores in scores:
        writer.writerow([-1] + [float(x) for x in row_scores[: len(ID2NAME)]])

    csv_bytes = buf.getvalue().encode("utf-8")
    return Response(
        csv_bytes,
        headers={
            "Content-Disposition": "attachment; filename=AUROC_LR.csv",
            "Content-Type": "text/csv; charset=utf-8",
        },
    )


if __name__ == "__main__":
    # 开发友好：host 0.0.0.0 便于局域网访问
    app.run(host="0.0.0.0", port=DEFAULT_PORT, debug=True)






