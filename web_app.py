# -*- coding: utf-8 -*-
from flask import Flask, render_template_string, request, session, Response
import os
import json
import importlib

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key')

# ---------------- Inference ----------------
_tokenizer = None
_model = None
_id2label = None
_device = None

# Display mapping: English-only (identity)
LABEL_DISPLAY = {}

def _disp_name(lbl: str) -> str:
    return lbl


def _try_read_label_map(paths):
    """Try multiple label_map.json locations and return id2label dict or None."""
    for p in paths:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                mp = json.load(f)
            id2 = mp.get('id2label') or {}
            if id2:
                # keys may be str
                return {int(k): v for k, v in id2.items()} if isinstance(list(id2.keys())[0], str) else id2
        except Exception:
            continue
    return None


def _discover_id2label():
    """Discover class names from artifacts; fallback to sensible defaults."""
    here = os.path.dirname(__file__)
    candidates = []
    # 1) NP_OUT env (preferred)
    np_out = os.environ.get('NP_OUT', os.path.join('Cbert2_Lr', 'artifacts'))
    candidates.append(os.path.join(_resolve_path(np_out), 'label_map.json'))
    # 2) Local common locations
    candidates.extend([
        os.path.join(here, 'artifacts', 'label_map.json'),
        _resolve_path(os.path.join('Nproject', 'artifacts', 'label_map.json')),
        _resolve_path(os.path.join('Cproject', 'Cbert2_Lr', 'artifacts', 'label_map.json')),
        _resolve_path(os.path.join('Cproject', 'Dert2', 'artifacts', 'label_map.json')),
    ])
    id2 = _try_read_label_map(candidates)
    if id2:
        return id2
    # Fallback to common CRE 3-class labels
    env_labels = os.environ.get('NP_FALLBACK_LABELS', 'Promoter,Enhancer,Non-CRE').split(',')
    env_labels = [s.strip() for s in env_labels if s.strip()]
    return {i: env_labels[i] for i in range(len(env_labels))}


def _resolve_path(p: str):
    if os.path.isabs(p):
        return p
    here = os.path.dirname(__file__)
    c = os.path.abspath(os.path.join(os.getcwd(), p))
    d = os.path.abspath(os.path.join(here, p))
    return c if os.path.exists(c) else d


def load_model():
    """Try to load real model. Return True if success, else False (fallback mode)."""
    global _tokenizer, _model, _id2label, _device
    if _model is not None and _tokenizer is not None and _id2label is not None:
        return True
    try:
        import torch  # local import
        from transformers import AutoTokenizer, AutoModel  # local import
        mod = importlib.import_module('hybrid_dnabert2_rnn_lstm')
        HybridDNABERT2 = getattr(mod, 'HybridDNABERT2')

        base_model_dir = _resolve_path(os.environ.get('NP_MODEL', 'dnabert2_117m'))
        artifacts_dir = _resolve_path(os.environ.get('NP_OUT', os.path.join('Cbert2_Lr', 'artifacts')))
        label_map_path = os.path.join(artifacts_dir, 'label_map.json')
        best_path = os.path.join(artifacts_dir, 'best_model.pt')

        # id2label discovery (tolerant if label_map.json missing here)
        _id2 = _discover_id2label()

        tokenizer = AutoTokenizer.from_pretrained(base_model_dir, local_files_only=True, trust_remote_code=True)
        base = AutoModel.from_pretrained(base_model_dir, local_files_only=True, trust_remote_code=True)
        if hasattr(base.config, 'use_flash_attn'):
            base.config.use_flash_attn = False

        num_labels = len(_id2)
        model = HybridDNABERT2(base, hidden_size=base.config.hidden_size, num_labels=num_labels, dropout=0.1)

        state = torch.load(best_path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if isinstance(state, dict):
            try:
                model.load_state_dict(state, strict=False)
            except Exception:
                fixed = {k.replace('module.', ''): v for k, v in state.items()}
                model.load_state_dict(fixed, strict=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        _tokenizer = tokenizer
        _model = model
        _id2label = _id2
        _device = device
        app.logger.info('Model loaded successfully with %d labels', num_labels)
        return True
    except Exception as e:
        # Fallback to demo mode
        _tokenizer = None
        _model = None
        _id2label = None
        _device = None
        app.logger.warning('Falling back to demo mode, model not loaded: %s', str(e))
        return False


def _fallback_predict(entries):
    # Prefer discovered labels from artifacts; else env; else 2-class placeholder
    id2 = _discover_id2label()
    labels = [id2[i] for i in sorted(id2.keys())]
    if not labels:
        labels = os.environ.get('NP_FALLBACK_LABELS', 'Promoter,Enhancer,Non-CRE').split(',')
    results = []
    for i, (sid, seq) in enumerate(entries):
        sid = sid or f'SEQ_{i+1:03d}'
        s = (seq or '').upper()
        total = max(1, len(s))
        gc = (s.count('G') + s.count('C')) / total
        at = 1.0 - gc
        # simple heuristic for demo: Promoter ~ TATA motif, Enhancer ~ GC, Non-CRE ~ remainder
        tata = 1.0 if 'TATA' in s else 0.0
        raw = []
        if len(labels) == 2:
            raw = [gc, at]
        elif len(labels) >= 3:
            raw = [tata * 0.6 + at * 0.4, gc, max(1e-6, 1.0 - (tata * 0.6 + at * 0.4 + gc))]
            # if more than 3, pad evenly
            if len(labels) > 3:
                extra = len(labels) - 3
                remain = max(1e-6, 1.0 - sum(raw))
                raw.extend([remain / extra] * extra)
        else:
            raw = [1.0]
        sm = sum(raw) or 1.0
        probs = [p / sm for p in raw]
        sm = sum(probs) or 1.0
        probs = [p / sm for p in probs]
        best_idx = int(max(range(len(probs)), key=lambda j: probs[j]))
        row = {'id': sid, 'pred': labels[best_idx]}
        for j, lb in enumerate(labels):
            row[f'score_{lb}'] = round(float(probs[j]), 6)
        results.append(row)
    return results


def predict_sequences(entries):
    if not entries:
        return []
    ok = load_model()
    if not ok:
        return _fallback_predict(entries)

    # Real inference path
    import torch  # safe now
    ids = [sid or f'SEQ_{i+1:03d}' for i, (sid, _s) in enumerate(entries)]
    seqs = [s for _sid, s in entries]
    enc = _tokenizer(seqs, padding=True, truncation=True, max_length=int(os.environ.get('NP_MAXLEN', 512)), return_tensors='pt')
    enc = {k: v.to(_device) for k, v in enc.items()}
    with torch.no_grad():
        logits, _ = _model(**enc, labels=None)
        probs = logits.softmax(dim=-1).cpu().tolist()
    results = []
    for i, pr in enumerate(probs):
        best_idx = int(max(range(len(pr)), key=lambda j: pr[j]))
        row = {'id': ids[i], 'pred': _id2label[best_idx]}
        for j, p in enumerate(pr):
            row[f'score_{_id2label[j]}'] = round(float(p), 6)
        results.append(row)
    return results

# ---------------- Utils ----------------

def parse_fasta(text: str):
    entries = []
    if not text:
        return entries
    curr_id, buf = None, []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith('>'):
            if curr_id is not None:
                entries.append((curr_id, ''.join(buf).upper()))
            curr_id = s[1:].strip()
            buf = []
        else:
            buf.append(s)
    if curr_id is not None:
        entries.append((curr_id, ''.join(buf).upper()))
    return entries


def results_to_csv(results):
    if not results:
        return 'id,prediction\n'
    # dynamic header
    cols = ['id'] + [c for c in results[0].keys() if c.startswith('score_')] + ['prediction']
    lines = [','.join(cols)]
    for r in results:
        row = [str(r.get('id',''))] + [str(r[k]) for k in cols if k.startswith('score_')] + [str(r.get('pred',''))]
        lines.append(','.join(row))
    return '\n'.join(lines)

# ---------------- Templates (compact) ----------------
INDEX_HTML = '''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A_cre</title>
<style>
:root{
  --bg:#0b0f1a;--panel:#0f172a;--glass:rgba(255,255,255,.06);--glass2:rgba(255,255,255,.12);
  --fg:#e5e7eb;--muted:#94a3b8;--brand:#7c3aed;--accent:#06b6d4;
  --pink:#ec4899;--cyan:#06b6d4;--sky:#0ea5e9;--amber:#f59e0b;--emerald:#10b981;--rose:#f43f5e;
  --grad-a:#7c3aed;--grad-b:#06b6d4;--grad-c:#f59e0b;--grad-d:#8b5cf6;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;font-family:Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(124,58,237,.32), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(14,165,233,.22), transparent 55%),
    radial-gradient(1000px 700px at 0% 100%, rgba(245,158,11,.20), transparent 60%),
    linear-gradient(150deg,#070b13 0%, #0d1424 45%, #0a0f1c 100%);
}
.hidden{display:none !important}
.bg-orbs{position:fixed;inset:0;pointer-events:none;z-index:0}
.orb{position:absolute;filter:blur(60px);opacity:.35;border-radius:50%;animation:float 12s ease-in-out infinite}
.o1{width:280px;height:280px;left:8%;top:6%;background:radial-gradient(circle at 30% 30%,rgba(124,58,237,.5),transparent 60%)}
.o2{width:300px;height:300px;right:6%;top:10%;background:radial-gradient(circle at 30% 30%,rgba(6,182,212,.45),transparent 60%);animation-delay:1.2s}
.o3{width:320px;height:320px;left:10%;bottom:6%;background:radial-gradient(circle at 30% 30%,rgba(245,158,11,.45),transparent 60%);animation-delay:.6s}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(18px)}}
.container{position:relative;z-index:1;max-width:1080px;margin:0 auto;padding:22px 16px}
.nav{position:sticky;top:0;z-index:50;background:linear-gradient(135deg,rgba(124,58,237,.24),rgba(6,182,212,.24));
  backdrop-filter:blur(10px);border-bottom:1px solid rgba(255,255,255,.12)}
.navin{display:flex;align-items:center;gap:12px;padding:12px 10px 12px 18px}
    .navin .left-group{display:flex;align-items:center;gap:10px}
    .navin .links{margin-left:auto;display:flex;align-items:center}
    .dna-sep{height:20px;flex:1 1 auto;display:block;align-self:center;margin:0 12px;min-width:180px;filter:drop-shadow(0 2px 6px rgba(0,0,0,.25));pointer-events:none}
    .logo{font-weight:900;letter-spacing:.3px;font-size:24px;background:linear-gradient(90deg,var(--grad-a),var(--grad-b),var(--grad-c),var(--grad-d));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .nav a{color:var(--muted);text-decoration:none;margin:0 10px}
    @media(max-width:820px){.dna-sep{display:none}}
.card{position:relative;border-radius:18px;padding:20px;margin:18px 0;
  border:1px solid transparent;
  background:
    linear-gradient(var(--panel), var(--panel)) padding-box,
    linear-gradient(120deg,var(--grad-a),var(--grad-b),var(--grad-c),var(--grad-d)) border-box;
  box-shadow:0 10px 30px rgba(0,0,0,.25)}
.title{font-size:2rem;margin:6px 0 14px;background:linear-gradient(120deg,var(--grad-a),var(--grad-b),var(--grad-c),var(--grad-d));
   -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-size:300% 300%;animation:shine 4s linear infinite}
 #hero .title{text-align:center}
 @keyframes shine{0%{background-position:0% 50%}100%{background-position:100% 50%}}
.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:3fr 2fr;gap:12px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
textarea,input[type=file]{width:100%;border-radius:12px;border:1px solid var(--glass2);
  background:rgba(255,255,255,.04);color:var(--fg);padding:12px;outline:none;transition:border .2s, box-shadow .2s}
textarea:focus,input[type=file]:focus{border-color:rgba(124,58,237,.6);box-shadow:0 0 0 3px rgba(124,58,237,.18)}
textarea{min-height:220px;font-family:Consolas,monospace}
/* custom file uploader - enhanced dropzone */
.filebox{position:relative;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;min-height:220px;border:1.5px dashed rgba(138,180,248,.38);background:linear-gradient(180deg,rgba(255,255,255,.035),rgba(255,255,255,.02));border-radius:14px;cursor:pointer;transition:border-color .2s, box-shadow .2s, transform .15s}
.filebox:hover{border-color:#8ab4f8;box-shadow:0 4px 24px rgba(138,180,248,.15) inset, 0 8px 26px rgba(0,0,0,.18)}
.filebox.dragover{border-color:#8ab4f8;background:rgba(255,255,255,.05);box-shadow:0 0 0 3px rgba(138,180,248,.15)}
.filebox::before{content:'';position:absolute;inset:0;border-radius:14px;background:radial-gradient(120px 60px at 20% 20%,rgba(124,58,237,.18),transparent 60%), radial-gradient(160px 80px at 80% 80%,rgba(6,182,212,.14),transparent 60%);pointer-events:none;opacity:.7}
.file-ico{display:inline-flex;width:52px;height:52px;border-radius:14px;align-items:center;justify-content:center;background:linear-gradient(135deg,rgba(124,58,237,.25),rgba(6,182,212,.25));box-shadow:0 6px 14px rgba(0,0,0,.18)}
.file-ico svg{filter:drop-shadow(0 2px 4px rgba(0,0,0,.25))}
.file-desc{text-align:center}
.filename{color:var(--muted)}
.filename.sub{font-size:.9rem;opacity:.6}
.btn{position:relative;overflow:hidden;display:inline-flex;align-items:center;gap:8px;padding:11px 16px;border-radius:12px;color:#fff;border:none;cursor:pointer;text-decoration:none;
  transition:transform .15s ease, filter .15s ease, box-shadow .2s}
.btn:active{transform:translateY(1px)}
.btn-emerald{background:linear-gradient(135deg,var(--emerald),#34d399);box-shadow:0 10px 24px rgba(16,185,129,.25)}
.btn-amber{background:linear-gradient(135deg,var(--amber),#fbbf24);box-shadow:0 10px 24px rgba(245,158,11,.25)}
.btn-cyan{background:linear-gradient(135deg,var(--cyan),var(--sky));box-shadow:0 10px 24px rgba(14,165,233,.25)}
.btn-pink{background:linear-gradient(135deg,var(--pink),#fb7185);box-shadow:0 10px 24px rgba(236,72,153,.25)}
.btn:hover{filter:saturate(1.08)}
.small{font-size:.92rem}
.helper{font-size:.88rem;color:var(--muted);margin-top:6px}
.hr{height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.14),transparent);margin:8px 0}
  /* image gallery */
  .img-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
  @media(max-width:900px){.img-grid{grid-template-columns:1fr}}
  .img-card{border-radius:14px;padding:14px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.10);
    display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:180px;box-shadow:0 6px 20px rgba(0,0,0,.18)}
  .img-card .caption{margin-top:10px;color:var(--muted);font-size:.92rem;text-align:center}
</style></head>
<body>
  <div class="nav"><div class="container navin">
    <div class="left-group">
        <div class="logo">A_cre</div>
        <svg class="dna-sep" viewBox="0 0 420 20" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <defs>
            <linearGradient id="dnaSepGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="var(--grad-a)"/>
              <stop offset="50%" stop-color="var(--grad-b)"/>
              <stop offset="100%" stop-color="var(--grad-d)"/>
            </linearGradient>
          </defs>
          <!-- 中线基准 -->
          <line x1="0" y1="10" x2="420" y2="10" stroke="rgba(255,255,255,.12)" stroke-width="1"/>
          <!-- DNA双链（两股） -->
          <path d="M0 6 C 40 2, 80 14, 120 10 S 200 2, 240 10 S 320 18, 360 10 S 420 2, 420 2" fill="none" stroke="url(#dnaSepGrad)" stroke-width="1.6" stroke-linecap="round" opacity="0.9"/>
          <path d="M0 14 C 40 18, 80 6, 120 10 S 200 18, 240 10 S 320 2, 360 10 S 420 18, 420 18" fill="none" stroke="url(#dnaSepGrad)" stroke-width="1.6" stroke-linecap="round" opacity="0.9"/>
          <!-- 横档（梯级） -->
          <g stroke="rgba(255,255,255,.55)" stroke-width="1">
            <line x1="20" y1="8.8" x2="20" y2="11.2"/>
            <line x1="60" y1="8.2" x2="60" y2="11.8"/>
            <line x1="100" y1="8.8" x2="100" y2="11.2"/>
            <line x1="140" y1="9.2" x2="140" y2="10.8"/>
            <line x1="180" y1="8.6" x2="180" y2="11.4"/>
            <line x1="220" y1="8.8" x2="220" y2="11.2"/>
            <line x1="260" y1="9.0" x2="260" y2="11.0"/>
            <line x1="300" y1="9.2" x2="300" y2="10.8"/>
            <line x1="340" y1="8.8" x2="340" y2="11.2"/>
            <line x1="380" y1="8.6" x2="380" y2="11.4"/>
          </g>
        </svg>
      </div>
      <div class="links">
      <a href="#intro">Introduction</a>
      <a href="#sub">Submission</a>
      <a href="#cite">Citation</a>
      <a href="#contact">Contact</a>
      <a href="/download" target="_blank">Download</a>
    </div>
  </div></div>
  <div class="bg-orbs"><span class="orb o1"></span><span class="orb o2"></span><span class="orb o3"></span></div>

  <div class="container">
    <div class="card" id="hero">
        <div class="title">A_cre: DNABERT2-based CRE classifier (Dbert2_LR)</div>
      </div>

    <div class="card hidden" id="intro">
      <div class="title" style="font-size:1.35rem">Introduction</div>
      <p class="muted">Paste FASTA text or upload files to obtain per-class probabilities and the final predicted label. You can download the results as CSV.</p>
      <div class="hr"></div>
      <ul class="muted" style="padding-left:18px;margin:8px 0 0">
        <li>Input: .fa/.fasta/.txt/.csv; textarea and file upload are supported</li>
        <li>Output: per-sequence class probabilities and predicted label (Promoter / Other CRE / Non-CRE)</li>
      </ul>
    </div>

    <div class="card" id="sub">
      <form method="POST" action="/predict" enctype="multipart/form-data">
        <div class="grid">
          <div>
            <label class="muted">FASTA text</label>
            <textarea id="fasta_text" name="fasta_text" placeholder=">Seq1\nATCG...\n>Seq2\nGGCA..."></textarea>
            <div class="helper">Recommended: each sequence starts with ">" as ID; DNA alphabet A/C/G/T/N.</div>
          </div>
          <div>
            <label class="muted">Upload files</label>
            <div class="filebox" id="filebox" tabindex="0" role="button" aria-label="file selection area">
              <input id="file_input" type="file" name="file" accept=".fa,.fasta,.txt,.csv" multiple hidden>
              <div class="file-desc">
                <div id="file_name" class="filename">Drag files here, or click to select</div>
                <div class="filename sub">Supports .fa / .fasta / .txt / .csv (multiple)</div>
              </div>
            </div>
          </div>
        </div>
        <div style="margin-top:14px;display:flex;gap:10px;flex-wrap:wrap">
          <button class="btn btn-emerald" type="submit">Submit</button>
          <a class="btn btn-amber" href="/download" target="_blank">Download Results</a>
          <button class="btn btn-cyan small" type="button" onclick="fillDemo()">Fill Demo</button>
          <button class="btn btn-pink small" type="button" onclick="clearAll()">Clear</button>
        </div>
      </form>
    </div>

    <div class="card" id="cite">
        <div class="title" style="font-size:1.35rem">Citation</div>
        <p class="muted">If you use this tool in your research, please follow the usage guidelines.</p>
      </div>

    <!-- 已移除：Image Gallery / Illustrations -->

    <div class="card" id="contact">
        <div class="title" style="font-size:1.35rem">Contact</div>
        <p class="muted">For questions and suggestions, please contact the authors via email.</p>
      </div>
  </div>

<script>
// Ripple effect for buttons (ES5 compatible)
(function(){
  function ripple(e){
    var btn = e.currentTarget;
    var circle = document.createElement('span');
    var d = Math.max(btn.clientWidth, btn.clientHeight);
    circle.style.width = circle.style.height = d + 'px';
    var rect = btn.getBoundingClientRect();
    circle.style.left = (e.clientX - rect.left - d/2) + 'px';
    circle.style.top  = (e.clientY - rect.top  - d/2) + 'px';
    circle.style.position='absolute';
    circle.style.background='rgba(255,255,255,.25)';
    circle.style.borderRadius='50%';
    circle.style.transform='scale(0)';
    circle.style.animation='ripple .6s ease-out';
    circle.style.pointerEvents='none';
    btn.appendChild(circle);
    setTimeout(function(){ if(circle && circle.parentNode){ circle.parentNode.removeChild(circle); } }, 600);
  }
  var style=document.createElement('style');
  style.textContent='@keyframes ripple{to{transform:scale(2.5);opacity:0}}';
  document.head.appendChild(style);
  var btns = document.querySelectorAll('.btn');
  for(var i=0;i<btns.length;i++){ btns[i].addEventListener('click', ripple); }
})();

// 已移除浅色主题注入，保留深色渐变主题

// file input filename update + dropzone interactions (ES5)
(function(){
  var box = document.getElementById('filebox');
  var fi = document.getElementById('file_input');
  var fn = document.getElementById('file_name');

  function updateName(fs){
    if(!fn) return;
    if(fs && fs.length){
      if(fs.length===1){ fn.textContent=fs[0].name; fn.title=fs[0].name; }
      else{
        var names=[]; for(var i=0;i<fs.length;i++){ names.push(fs[i].name); }
        fn.textContent=fs.length+' files'; fn.title=names.join('\\n');
      }
    }else{
      fn.textContent='Drag files here, or click to select'; fn.removeAttribute('title');
    }
  }

  if(fi){ fi.addEventListener('change', function(){ updateName(this.files); }); }
  if(box){
    box.addEventListener('click', function(){ if(fi){ fi.click(); } });
    box.addEventListener('keydown', function(e){ if(e.key==='Enter' || e.key===' '){ e.preventDefault(); if(fi){ fi.click(); } }});
    ['dragenter','dragover'].forEach(function(t){ box.addEventListener(t, function(e){ e.preventDefault(); e.stopPropagation(); box.classList.add('dragover'); }); });
    ['dragleave','drop'].forEach(function(t){ box.addEventListener(t, function(e){ e.preventDefault(); e.stopPropagation(); box.classList.remove('dragover'); }); });
    box.addEventListener('drop', function(e){
      if(!fi) return; var dt = e.dataTransfer; if(!dt || !dt.files || !dt.files.length) return;
      try{ fi.files = dt.files; }catch(err){}
      updateName(dt.files);
    });
  }

  window.fillDemo = function(){
    var demo = ">Seq1\\nATGCGTACGTAGCTAGCTAGCTAGCTAG\\n>Seq2\\nGGGCGCGTATATATATATATATATA\\n>Seq3\\nTTTTNNNACGTACGTACGTACGTACGTA";
    var ta = document.getElementById('fasta_text'); if(ta) ta.value = demo;
  };
  window.clearAll = function(){
    var ta = document.getElementById('fasta_text'); if(ta) ta.value='';
    if(fi){ fi.value=''; }
    if(fn){ fn.textContent='Drag files here, or click to select'; fn.removeAttribute('title'); }
  };
})();

// Show the Introduction panel only when the nav link is clicked
(function(){
  function showIntro(){ var intro=document.getElementById('intro'); if(intro){ intro.classList.remove('hidden'); intro.scrollIntoView({behavior:'smooth',block:'start'}); } }
  var as=document.querySelectorAll('a[href="#intro"]');
  for(var i=0;i<as.length;i++){
    as[i].addEventListener('click', function(e){ e.preventDefault(); showIntro(); });
  }
})();
</script>
</body></html>'''

RESULT_HTML = '''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Prediction Results - A_cre</title>
<style>
:root{--fg:#e5e7eb;--muted:#94a3b8;--amber:#f59e0b;--emerald:#10b981;--cyan:#06b6d4;--sky:#0ea5e9}
body{margin:0;font-family:Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);
  background:linear-gradient(150deg,#070b13 0%, #0d1424 45%, #0a0f1c 100%)}
.wrap{max-width:1100px;margin:0 auto;padding:26px 16px}
.card{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);backdrop-filter:blur(16px);border-radius:18px;padding:18px}
.btn{position:relative;overflow:hidden;display:inline-flex;align-items:center;gap:8px;padding:8px 14px;border-radius:10px;color:#fff;text-decoration:none;border:none;cursor:pointer;margin-right:8px}
.btn-amber{background:linear-gradient(135deg,var(--amber),#fbbf24)}
.btn-cyan{background:linear-gradient(135deg,var(--cyan),var(--sky))}
.table{overflow:auto}
table{width:100%;border-collapse:collapse}
th,td{padding:10px;border-bottom:1px solid rgba(255,255,255,.12);text-align:left;vertical-align:middle}
th{position:sticky;top:0;background:rgba(255,255,255,.08)}
tbody tr:nth-child(odd){background:rgba(255,255,255,.025)}
tbody tr:hover{background:rgba(255,255,255,.05)}
.pill{display:inline-block;padding:4px 10px;border-radius:999px;background:linear-gradient(135deg,#22c55e,#34d399);color:#052e16;font-weight:600}
.idcell{word-break:break-all;max-width:360px}
</style></head>
<body>
  <div class="wrap">
    <h2 style="margin:0 0 14px">Prediction Results</h2>
    <div class="card">
      <div class="table">
        <table>
          <thead>
            <tr>
              <th>Sequence ID</th>
              {% for lb, dp in label_pairs %}<th>score_{{ dp }}</th>{% endfor %}
              <th>Prediction</th>
            </tr>
          </thead>
          <tbody>
            {% for r in results %}
            <tr>
              <td class="idcell">{{ r.id }}</td>
              {% for lb, dp in label_pairs %}<td>{{ '%.6f'|format(r['score_'+lb]) }}</td>{% endfor %}
              <td><span class="pill">{{ label_display.get(r.pred, r.pred) }}</span></td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
    <div style="margin-top:12px">
      <a class="btn btn-amber" href="/download" target="_blank">Download CSV</a>
      <a class="btn btn-cyan" href="/">Back to Home</a>
    </div>
  </div>
<script>
// 保留深色主题，移除浅色覆盖
</script>
</body></html>'''

# ---------------- Routes ----------------
@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/predict', methods=['POST'])
def predict():
    text = ''
    if request.form.get('fasta_text'):
        text += request.form['fasta_text'] + '\n'
    # 支持多文件上传
    files = request.files.getlist('file') if 'file' in request.files else []
    for f in files:
        if f and getattr(f, 'filename', ''):
            try:
                text += f.read().decode('utf-8', errors='ignore') + '\n'
            except Exception:
                pass
    entries = parse_fasta(text)
    res = predict_sequences(entries)
    session['results'] = res
    csv = results_to_csv(res)
    session['results_csv'] = csv
    labels = []
    if res:
        labels = [c[len('score_'):] for c in res[0].keys() if c.startswith('score_')]
    label_pairs = [(lb, _disp_name(lb)) for lb in labels]
    label_display = {lb: _disp_name(lb) for lb in labels}
    # Also translate prediction pill via label_display in template
    return render_template_string(RESULT_HTML, results=res, labels=labels, label_pairs=label_pairs, label_display=label_display)

@app.route('/download')
def download():
    csv = session.get('results_csv', 'id,prediction\n')
    return Response(csv, mimetype='text/csv; charset=utf-8', headers={'Content-Disposition': 'attachment; filename="predictions_lr.csv"'})

if __name__ == '__main__':
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', '5052'))
    app.run(debug=True, host=host, port=port)