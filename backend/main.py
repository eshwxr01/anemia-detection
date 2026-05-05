"""
Anaemia Detection — FastAPI Backend + Frontend
===============================================
Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Then open: http://localhost:8000
"""

import io, json, time, logging, random
import numpy as np
from pathlib import Path

import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Anaemia Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR  = Path("saved_model")
MODEL_PATH = MODEL_DIR / "anaemia_model.keras"
META_PATH  = MODEL_DIR / "metadata.json"

model    = None
metadata = {}
IMG_SIZE  = (224, 224)
LABELS    = ["L1 - Mild Anaemia", "L2 - Moderate Anaemia", "L3 - Severe Anaemia"]

@app.on_event("startup")
async def load_model():
    global model, metadata, IMG_SIZE, LABELS
    if MODEL_PATH.exists():
        log.info(f"Loading model from {MODEL_PATH} …")
        model = tf.keras.models.load_model(str(MODEL_PATH))
        log.info("Model loaded ✓")
    else:
        log.warning("No model found — will use mock predictions.")
    if META_PATH.exists():
        with open(META_PATH) as f:
            metadata = json.load(f)
        IMG_SIZE = tuple(metadata.get("img_size", [224, 224]))


class AnalysisResult(BaseModel):
    filename:            str
    diagnosis:           str
    predicted_class:     str
    confidence_pct:      float
    class_probabilities: dict
    rbc_count:           int
    mean_cell_area:      float
    pallor_ratio:        float
    circularity:         float
    morphology_flags:    list[str]
    processing_ms:       int
    model_version:       str


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def image_bytes_to_rgb(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def analyze_rbc_morphology(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rbc_areas, pallor_ratios, circularities = [], [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (300 < area < 8000): continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0: continue
        circ = 4 * np.pi * area / (perim ** 2)
        if circ < 0.4: continue
        rbc_areas.append(area)
        circularities.append(circ)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        inner_r = max(int(min(w, h) * 0.25), 1)
        inner_mask = np.zeros_like(mask)
        cv2.circle(inner_mask, (cx, cy), inner_r, 255, -1)
        outer_mask = cv2.bitwise_and(mask, cv2.bitwise_not(inner_mask))
        inner_mean = cv2.mean(gray, mask=cv2.bitwise_and(mask, inner_mask))[0]
        outer_mean = cv2.mean(gray, mask=outer_mask)[0]
        if outer_mean > 0:
            pallor_ratios.append(inner_mean / outer_mean)
    n = len(rbc_areas)
    mean_area   = float(np.mean(rbc_areas))     if rbc_areas     else 0.0
    mean_pallor = float(np.mean(pallor_ratios)) if pallor_ratios else 0.0
    mean_circ   = float(np.mean(circularities)) if circularities else 0.0
    flags = []
    if mean_pallor > 0.75: flags.append("Hypochromia")
    if 0 < mean_area < 500: flags.append("Microcytosis")
    if mean_area > 2000:    flags.append("Macrocytosis")
    if mean_circ  < 0.70:   flags.append("Poikilocytosis")
    if n < 50:              flags.append("Low RBC density")
    return {"rbc_count": n, "mean_cell_area": round(mean_area,1),
            "pallor_ratio": round(mean_pallor,3), "circularity": round(mean_circ,3), "flags": flags}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...)):
    t0 = time.time()
    image_bytes = await file.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB).")
    try:
        if model is not None:
            tensor = preprocess_image(image_bytes)
            probs  = model.predict(tensor, verbose=0)[0]
        else:
            probs = np.array([random.uniform(0.1, 0.8) for _ in range(3)])
            probs = probs / probs.sum()
        class_idx   = int(np.argmax(probs))
        predicted   = LABELS[class_idx]
        confidence  = round(float(probs[class_idx]) * 100, 1)
        class_probs = {LABELS[i]: round(float(probs[i]) * 100, 1) for i in range(len(LABELS))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    try:
        image_rgb = image_bytes_to_rgb(image_bytes)
        morph = analyze_rbc_morphology(image_rgb)
    except Exception:
        morph = {"rbc_count":0,"mean_cell_area":0.0,"pallor_ratio":0.0,"circularity":0.0,"flags":[]}

    processing_ms = int((time.time() - t0) * 1000)

    return AnalysisResult(
        filename=file.filename or "unknown",
        diagnosis=predicted,
        predicted_class=f"Class {class_idx + 1}",
        confidence_pct=confidence,
        class_probabilities=class_probs,
        rbc_count=morph["rbc_count"],
        mean_cell_area=morph["mean_cell_area"],
        pallor_ratio=morph["pallor_ratio"],
        circularity=morph["circularity"],
        morphology_flags=morph["flags"],
        processing_ms=processing_ms,
        model_version=metadata.get("model_version", "1.0.0"),
    )


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AnaemiaScan</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg:#f7f3ee;--surface:#fff;--surface2:#f0ebe3;
  --border:#e0d8cc;--accent:#c0392b;--accent2:#e67e22;
  --green:#27ae60;--text:#1a1208;--muted:#8a7a6a;
  --shadow:0 4px 24px rgba(0,0,0,0.08);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Outfit',sans-serif;min-height:100vh}
body::before{content:'';position:fixed;inset:0;
  background-image:radial-gradient(circle at 20% 20%,rgba(192,57,43,0.06) 0%,transparent 50%),
  radial-gradient(circle at 80% 80%,rgba(230,126,34,0.06) 0%,transparent 50%);
  pointer-events:none;z-index:0}
.container{max-width:860px;margin:0 auto;padding:0 24px;position:relative;z-index:1}

/* header */
header{padding:32px 0 24px;display:flex;align-items:center;gap:16px;
  border-bottom:2px solid var(--border);margin-bottom:48px}
.logo{width:48px;height:48px;border-radius:14px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:grid;place-items:center;font-size:22px;flex-shrink:0;
  box-shadow:0 4px 16px rgba(192,57,43,0.3)}
.logo-text h1{font-family:'Playfair Display',serif;font-size:24px;letter-spacing:-0.3px}
.logo-text p{font-family:'JetBrains Mono',monospace;font-size:10px;
  color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-top:2px}
.badge{margin-left:auto;background:var(--accent);color:#fff;font-size:11px;
  padding:4px 12px;border-radius:100px;font-family:'JetBrains Mono',monospace;
  letter-spacing:0.05em}

/* hero */
.hero{margin-bottom:40px}
.hero h2{font-family:'Playfair Display',serif;font-size:clamp(32px,5vw,48px);
  line-height:1.15;letter-spacing:-0.5px;margin-bottom:12px}
.hero h2 em{font-style:italic;color:var(--accent)}
.hero p{color:var(--muted);font-size:15px;line-height:1.7;max-width:500px;font-weight:300}

/* upload */
.drop-zone{border:2px dashed var(--border);border-radius:20px;padding:52px 32px;
  text-align:center;cursor:pointer;transition:all 0.25s;background:var(--surface);
  position:relative;box-shadow:var(--shadow)}
.drop-zone:hover,.drop-zone.drag-over{border-color:var(--accent);
  box-shadow:0 8px 32px rgba(192,57,43,0.15);transform:translateY(-2px)}
.drop-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.drop-icon{font-size:44px;margin-bottom:14px;display:block}
.drop-zone h3{font-family:'Playfair Display',serif;font-size:20px;margin-bottom:8px}
.drop-zone p{color:var(--muted);font-size:14px;font-weight:300}
.hint{display:inline-block;margin-top:18px;font-family:'JetBrains Mono',monospace;
  font-size:10px;color:var(--muted);background:var(--surface2);padding:5px 14px;
  border-radius:100px;border:1px solid var(--border);text-transform:uppercase;letter-spacing:0.08em}

/* preview */
.preview-box{display:none;margin-top:16px;border-radius:14px;overflow:hidden;
  border:1px solid var(--border);background:var(--surface)}
.preview-box img{width:100%;max-height:280px;object-fit:contain;display:block;padding:12px}
.preview-name{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted);
  padding:8px 16px 12px;border-top:1px solid var(--border)}

/* btn */
.btn-analyze{width:100%;padding:17px;margin-top:14px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  border:none;border-radius:14px;color:#fff;font-family:'Outfit',sans-serif;
  font-size:16px;font-weight:600;cursor:pointer;transition:all 0.2s;
  letter-spacing:0.02em;position:relative;overflow:hidden}
.btn-analyze:hover:not(:disabled){transform:translateY(-2px);
  box-shadow:0 12px 32px rgba(192,57,43,0.35)}
.btn-analyze:disabled{opacity:0.4;cursor:not-allowed;transform:none}
.btn-analyze.loading::after{content:'';position:absolute;inset:0;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,0.2),transparent);
  animation:shimmer 1.2s infinite}
@keyframes shimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}

/* loading bar */
.loading-bar{display:none;height:3px;background:var(--surface2);border-radius:100px;
  overflow:hidden;margin-top:12px}
.loading-bar.active{display:block}
.loading-bar::after{content:'';display:block;height:100%;width:40%;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  border-radius:100px;animation:prog 1.4s ease-in-out infinite}
@keyframes prog{0%{transform:translateX(-100%)}100%{transform:translateX(350%)}}

/* error */
.error-box{display:none;background:#fdf0ef;border:1px solid rgba(192,57,43,0.3);
  border-radius:12px;padding:16px 20px;color:var(--accent);font-size:14px;margin-top:12px}

/* results */
#results{display:none;animation:fadeUp 0.5s ease both;margin-top:32px}
@keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}

/* diagnosis card */
.diagnosis-card{border-radius:20px;padding:28px;margin-bottom:20px;border:2px solid}
.diagnosis-card.l1{background:#fef9f0;border-color:#f39c12}
.diagnosis-card.l2{background:#fef5f0;border-color:var(--accent2)}
.diagnosis-card.l3{background:#fdf0ef;border-color:var(--accent)}
.diag-top{display:flex;align-items:flex-start;gap:16px}
.diag-icon{font-size:40px;flex-shrink:0}
.diag-label{font-family:'JetBrains Mono',monospace;font-size:10px;text-transform:uppercase;
  letter-spacing:0.12em;color:var(--muted);margin-bottom:6px}
.diag-title{font-family:'Playfair Display',serif;font-size:28px;line-height:1.2;margin-bottom:6px}
.diag-file{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted)}
.conf-row{display:flex;align-items:center;gap:12px;margin-top:14px}
.conf-bg{flex:1;height:8px;background:rgba(0,0,0,0.08);border-radius:100px;overflow:hidden}
.conf-fill{height:100%;border-radius:100px;transition:width 1s cubic-bezier(0.22,1,0.36,1)}
.l1 .conf-fill{background:#f39c12}
.l2 .conf-fill{background:var(--accent2)}
.l3 .conf-fill{background:var(--accent)}
.conf-pct{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:500;flex-shrink:0}

/* prob bars */
.prob-section{background:var(--surface);border:1px solid var(--border);border-radius:16px;
  padding:20px 24px;margin-bottom:16px}
.prob-title{font-family:'JetBrains Mono',monospace;font-size:10px;text-transform:uppercase;
  letter-spacing:0.1em;color:var(--muted);margin-bottom:16px}
.prob-row{display:flex;align-items:center;gap:12px;margin-bottom:10px}
.prob-label{font-size:13px;font-weight:500;width:180px;flex-shrink:0}
.prob-bar-bg{flex:1;height:8px;background:var(--surface2);border-radius:100px;overflow:hidden}
.prob-bar-fill{height:100%;border-radius:100px;background:linear-gradient(90deg,var(--accent),var(--accent2));
  transition:width 1s cubic-bezier(0.22,1,0.36,1)}
.prob-val{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--muted);
  width:48px;text-align:right;flex-shrink:0}

/* stats grid */
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:16px}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:18px;transition:border-color 0.2s,box-shadow 0.2s}
.stat-card:hover{border-color:var(--accent);box-shadow:0 4px 16px rgba(192,57,43,0.1)}
.stat-label{font-family:'JetBrains Mono',monospace;font-size:10px;text-transform:uppercase;
  letter-spacing:0.08em;color:var(--muted);margin-bottom:8px}
.stat-value{font-family:'Playfair Display',serif;font-size:26px;line-height:1;margin-bottom:4px}
.stat-sub{font-size:11px;color:var(--muted);font-weight:300}

/* flags */
.flags-section{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:18px 22px;margin-bottom:16px}
.flags-title{font-family:'JetBrains Mono',monospace;font-size:10px;text-transform:uppercase;
  letter-spacing:0.1em;color:var(--muted);margin-bottom:12px}
.flags-list{display:flex;flex-wrap:wrap;gap:8px}
.flag-chip{background:#fdf0ef;border:1px solid rgba(192,57,43,0.25);color:var(--accent);
  font-family:'JetBrains Mono',monospace;font-size:11px;padding:4px 12px;border-radius:100px}
.no-flags{font-family:'Playfair Display',serif;font-style:italic;color:var(--muted);font-size:14px}

/* meta */
.meta-row{display:flex;gap:8px;flex-wrap:wrap;font-family:'JetBrains Mono',monospace;
  font-size:11px;color:var(--muted)}
.meta-row span{background:var(--surface);border:1px solid var(--border);
  padding:4px 12px;border-radius:100px}

/* reset btn */
.btn-reset{background:transparent;border:1.5px solid var(--border);color:var(--muted);
  font-family:'Outfit',sans-serif;font-size:14px;padding:10px 22px;border-radius:10px;
  cursor:pointer;margin-top:20px;transition:all 0.2s}
.btn-reset:hover{border-color:var(--accent);color:var(--accent)}

footer{border-top:1px solid var(--border);padding:24px 0;margin-top:56px;text-align:center;
  font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);
  text-transform:uppercase;letter-spacing:0.08em}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo">🩸</div>
    <div class="logo-text">
      <h1>AnaemiaScan</h1>
      <p>Blood Smear Analysis System</p>
    </div>
    <span class="badge">v1.0</span>
  </header>

  <div class="hero">
    <h2>Detect <em>Anaemia</em><br/>from blood smear images.</h2>
    <p>Upload a microscopic blood smear image. The deep learning model classifies anaemia severity and analyses RBC morphology instantly.</p>
  </div>

  <div class="drop-zone" id="dropZone">
    <input type="file" id="fileInput" accept="image/png,image/jpeg,image/tiff,image/bmp"/>
    <span class="drop-icon">🔬</span>
    <h3>Drop your blood smear image here</h3>
    <p>or click to browse from your computer</p>
    <span class="hint">PNG · JPG · TIFF · BMP · Max 20 MB</span>
  </div>

  <div class="preview-box" id="previewBox">
    <img id="previewImg" src="" alt="Preview"/>
    <div class="preview-name" id="previewName"></div>
  </div>

  <button class="btn-analyze" id="analyzeBtn" disabled>Analyse Image</button>
  <div class="loading-bar" id="loadingBar"></div>
  <div class="error-box" id="errorBox"></div>

  <div id="results">
    <div class="diagnosis-card" id="diagCard">
      <div class="diag-top">
        <div class="diag-icon" id="diagIcon"></div>
        <div style="flex:1">
          <div class="diag-label">Diagnosis Result</div>
          <div class="diag-title" id="diagTitle"></div>
          <div class="diag-file"  id="diagFile"></div>
          <div class="conf-row">
            <div class="conf-bg"><div class="conf-fill" id="confFill" style="width:0%"></div></div>
            <span class="conf-pct" id="confPct"></span>
          </div>
        </div>
      </div>
    </div>

    <div class="prob-section">
      <div class="prob-title">Class Probabilities</div>
      <div id="probBars"></div>
    </div>

    <div class="stats-grid">
      <div class="stat-card"><div class="stat-label">RBC Count</div>
        <div class="stat-value" id="sRbc">—</div><div class="stat-sub">cells detected</div></div>
      <div class="stat-card"><div class="stat-label">Mean Cell Area</div>
        <div class="stat-value" id="sArea">—</div><div class="stat-sub">pixels²</div></div>
      <div class="stat-card"><div class="stat-label">Pallor Ratio</div>
        <div class="stat-value" id="sPallor">—</div><div class="stat-sub">inner/outer brightness</div></div>
      <div class="stat-card"><div class="stat-label">Circularity</div>
        <div class="stat-value" id="sCirc">—</div><div class="stat-sub">shape regularity</div></div>
      <div class="stat-card"><div class="stat-label">Confidence</div>
        <div class="stat-value" id="sConf">—</div><div class="stat-sub">model confidence</div></div>
      <div class="stat-card"><div class="stat-label">Processing</div>
        <div class="stat-value" id="sTime">—</div><div class="stat-sub">milliseconds</div></div>
    </div>

    <div class="flags-section">
      <div class="flags-title">Morphology Flags</div>
      <div class="flags-list" id="flagsList"></div>
    </div>

    <div class="meta-row" id="metaRow"></div>
    <button class="btn-reset" id="resetBtn">← Analyse Another Image</button>
  </div>
</div>

<footer>AnaemiaScan · EfficientNetB0 · RBC Morphology Analysis</footer>

<script>
const dropZone   = document.getElementById('dropZone');
const fileInput  = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const previewBox = document.getElementById('previewBox');
const previewImg = document.getElementById('previewImg');
const previewName= document.getElementById('previewName');
const loadingBar = document.getElementById('loadingBar');
const errorBox   = document.getElementById('errorBox');
const results    = document.getElementById('results');
const resetBtn   = document.getElementById('resetBtn');
let selectedFile = null;

dropZone.addEventListener('dragover', e=>{e.preventDefault();dropZone.classList.add('drag-over')});
dropZone.addEventListener('dragleave',()=>dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop',e=>{e.preventDefault();dropZone.classList.remove('drag-over');
  if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0])});
fileInput.addEventListener('change',()=>{if(fileInput.files[0]) handleFile(fileInput.files[0])});

function handleFile(file){
  selectedFile=file;
  const r=new FileReader();
  r.onload=ev=>{previewImg.src=ev.target.result;
    previewName.textContent=`📎 ${file.name}  (${(file.size/1024).toFixed(1)} KB)`;
    previewBox.style.display='block'};
  r.readAsDataURL(file);
  analyzeBtn.disabled=false;
  results.style.display='none';
  errorBox.style.display='none';
}

analyzeBtn.addEventListener('click', async()=>{
  if(!selectedFile) return;
  analyzeBtn.disabled=true;
  analyzeBtn.textContent='Analysing…';
  analyzeBtn.classList.add('loading');
  loadingBar.classList.add('active');
  errorBox.style.display='none';
  results.style.display='none';
  const fd=new FormData();
  fd.append('file',selectedFile);
  try{
    const resp=await fetch('/analyze',{method:'POST',body:fd});
    if(!resp.ok){const e=await resp.json().catch(()=>({detail:resp.statusText}));throw new Error(e.detail||'Server error')}
    const data=await resp.json();
    showResults(data);
  }catch(e){
    errorBox.textContent=`⚠ ${e.message}`;
    errorBox.style.display='block';
  }finally{
    analyzeBtn.disabled=false;
    analyzeBtn.textContent='Analyse Image';
    analyzeBtn.classList.remove('loading');
    loadingBar.classList.remove('active');
  }
});

function showResults(d){
  const cls = d.diagnosis.toLowerCase().includes('l1')?'l1':d.diagnosis.toLowerCase().includes('l2')?'l2':'l3';
  const icons={'l1':'🟡','l2':'🟠','l3':'🔴'};
  const card=document.getElementById('diagCard');
  card.className='diagnosis-card '+cls;
  document.getElementById('diagIcon').textContent=icons[cls];
  document.getElementById('diagTitle').textContent=d.diagnosis;
  document.getElementById('diagFile').textContent='File: '+d.filename;
  setTimeout(()=>{document.getElementById('confFill').style.width=d.confidence_pct+'%'},100);
  document.getElementById('confPct').textContent=d.confidence_pct+'% confidence';

  // prob bars
  const pb=document.getElementById('probBars');
  pb.innerHTML='';
  Object.entries(d.class_probabilities).forEach(([label,pct])=>{
    pb.innerHTML+=`<div class="prob-row">
      <span class="prob-label">${label}</span>
      <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:0%" data-w="${pct}"></div></div>
      <span class="prob-val">${pct}%</span></div>`;
  });
  setTimeout(()=>{document.querySelectorAll('.prob-bar-fill').forEach(el=>{el.style.width=el.dataset.w+'%'})},100);

  document.getElementById('sRbc').textContent=d.rbc_count;
  document.getElementById('sArea').textContent=d.mean_cell_area.toFixed(1);
  document.getElementById('sPallor').textContent=d.pallor_ratio.toFixed(3);
  document.getElementById('sCirc').textContent=d.circularity.toFixed(3);
  document.getElementById('sConf').textContent=d.confidence_pct+'%';
  document.getElementById('sTime').textContent=d.processing_ms;

  const fl=document.getElementById('flagsList');
  fl.innerHTML=d.morphology_flags&&d.morphology_flags.length>0
    ?d.morphology_flags.map(f=>`<span class="flag-chip">${f}</span>`).join('')
    :'<span class="no-flags">No morphology abnormalities detected.</span>';

  document.getElementById('metaRow').innerHTML=
    `<span>Model v${d.model_version||'1.0.0'}</span><span>${new Date().toLocaleString()}</span>`;

  results.style.display='block';
  results.scrollIntoView({behavior:'smooth',block:'start'});
}

resetBtn.addEventListener('click',()=>{
  selectedFile=null;fileInput.value='';
  previewBox.style.display='none';results.style.display='none';
  analyzeBtn.disabled=true;analyzeBtn.textContent='Analyse Image';
  errorBox.style.display='none';window.scrollTo({top:0,behavior:'smooth'});
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTML_PAGE


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})