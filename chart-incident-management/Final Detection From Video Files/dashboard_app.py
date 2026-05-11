import json
import os
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List

from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for
from werkzeug.utils import secure_filename

import dashboard_backend as backend

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, "dashboard_uploads")
RUNS_DIR = os.path.join(APP_DIR, "dashboard_runs")
PROJECT_LOGO_PATH = os.path.join(APP_DIR, "final logo png.png")
UMD_LOGO_PATH = os.path.join(APP_DIR, "umd_logo.png")
SELECTED_CAMERAS_PATH = os.path.join(APP_DIR, "SelectedCameras.json")
WORKER_SCRIPT_PATH = os.path.join(APP_DIR, "dashboard_worker.py")
PROJECT_TITLE = "UMD CHART Incident Detection and Reporting Dashboard"
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "m4v", "wmv", "webm"}
MAX_LOG_LINES = 1200
JOB_POLL_MS = 250

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024 * 2


def now_ts() -> float:
    return time.time()


def allowed_video_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def load_selected_cameras() -> List[Dict[str, str]]:
    if not os.path.exists(SELECTED_CAMERAS_PATH):
        return []

    try:
        with open(SELECTED_CAMERAS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []

        cleaned = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            url = str(item.get("URL", "")).strip()
            cam_id = str(item.get("id", "")).strip()
            if name and url:
                cleaned.append({
                    "name": name,
                    "URL": url,
                    "id": cam_id,
                })
        return cleaned
    except Exception:
        return []

def run_dir_for(job_id: str) -> str:
    return os.path.join(RUNS_DIR, f"job_{job_id}")


def state_path_for(job_id: str) -> str:
    return os.path.join(run_dir_for(job_id), "job_state.json")


def log_path_for(job_id: str) -> str:
    return os.path.join(run_dir_for(job_id), "job_logs.txt")


def preview_path_for(job_id: str) -> str:
    return os.path.join(run_dir_for(job_id), "preview_latest.jpg")


def create_job(input_type: str, source_value: str, recorded_lookup_url: str = "") -> Dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    run_dir = run_dir_for(job_id)
    os.makedirs(run_dir, exist_ok=True)

    state = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "step": "Queued",
        "input_type": input_type,
        "source_value": source_value,
        "recorded_lookup_url": recorded_lookup_url,
        "run_dir": run_dir,
        "preview_path": "",
        "preview_version": 0,
        "result": None,
        "error": "",
        "created_at": now_ts(),
        "updated_at": now_ts(),
    }
    write_state(job_id, state)
    return state


def write_state(job_id: str, state: Dict[str, Any]) -> None:
    path = state_path_for(job_id)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)



def load_state(job_id: str) -> Dict[str, Any]:
    path = state_path_for(job_id)
    if not os.path.exists(path):
        abort(404)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_state(job_id: str, **kwargs) -> Dict[str, Any]:
    state = load_state(job_id)
    state.update(kwargs)
    state["updated_at"] = now_ts()
    write_state(job_id, state)
    return state


def append_log(job_id: str, message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    with open(log_path_for(job_id), "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def read_tail_lines(path: str, max_lines: int = MAX_LOG_LINES) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    return lines[-max_lines:]


def load_job_view(job_id: str) -> Dict[str, Any]:
    state = load_state(job_id)
    state["logs"] = read_tail_lines(log_path_for(job_id))

    preview_path = preview_path_for(job_id)
    if os.path.exists(preview_path):
        state["preview_path"] = preview_path
    return state


def spawn_worker(job_id: str) -> None:
    state = load_state(job_id)
    log_path = log_path_for(job_id)
    worker_cmd = [
        sys.executable,
        WORKER_SCRIPT_PATH,
        "--run-dir", state["run_dir"],
        "--job-id", state["job_id"],
        "--input-type", state["input_type"],
        "--source-value", state["source_value"],
        "--recorded-lookup-url", state.get("recorded_lookup_url", ""),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""

    with open(log_path, "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            worker_cmd,
            cwd=APP_DIR,
            stdout=log_file,
            stderr=log_file,
            env=env,
            start_new_session=True,
        )

    update_state(
        job_id,
        status="starting",
        step="Launching worker process",
        progress=max(1, int(state.get("progress", 0))),
        worker_pid=process.pid,
    )
    append_log(job_id, f"Spawned worker process PID {process.pid}")


HOME_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{ title }}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Inter:wght@400;500;600;700&display=swap');

    body { font-family: 'Inter', Arial, sans-serif; background: #f7f8fb; margin: 0; color: #1f2937; }

    .hero {
      background: linear-gradient(135deg, #eaf0f8 0%, #dfe7f2 100%);
      padding: 28px 24px 12px 24px;
      border-bottom: 1px solid #d5deea;
    }

    .hero-inner { max-width: 1200px; margin: 0 auto; }

    .hero-top {
      display: grid;
      grid-template-columns: 120px 1fr 120px;
      align-items: center;
      gap: 18px;
    }

    .hero-logo {
      display: flex;
      justify-content: center;
      align-items: center;
    }

    .hero-logo img {
      max-width: 110px;
      max-height: 110px;
      object-fit: contain;
      display: block;
    }

    .hero-center {
      text-align: center;
    }

    .hero-project {
      margin: 0;
      font-family: 'Orbitron', Arial, sans-serif;
      font-size: 30px;
      font-weight: 700;
      letter-spacing: 0.08em;
      color: #0f172a;
    }

    .hero-main-title {
      margin: 8px 0 0 0;
      font-family: 'Orbitron', Arial, sans-serif;
      font-size: 26px;
      font-weight: 500;
      letter-spacing: 0.03em;
      color: #0f172a;
      line-height: 1.25;
    }

    .hero-dashboard {
      margin: 18px 0 4px 0;
      text-align: center;
      font-size: 22px;
      font-weight: 600;
      color: #111827;
    }

    .hero-subtext {
      margin: 0 auto;
      max-width: 980px;
      text-align: left;
      color: #475569;
      font-size: 17px;
      line-height: 1.45;
    }

    .page { max-width: 1200px; margin: 24px auto; padding: 0 18px 40px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .card { background: white; border-radius: 18px; padding: 22px; box-shadow: 0 10px 24px rgba(15,23,42,0.06); border: 1px solid #e5e7eb; }
    .card h2 { margin-top: 0; font-size: 22px; }
    label { display: block; font-weight: bold; margin: 12px 0 6px; }
    input[type=text], input[type=file] { width: 100%; box-sizing: border-box; padding: 12px; border: 1px solid #cbd5e1; border-radius: 10px; }
    .help { color: #64748b; font-size: 13px; margin-top: 4px; }
    button { margin-top: 16px; padding: 12px 18px; border: 0; border-radius: 10px; background: #991b1b; color: white; font-size: 15px; cursor: pointer; }
    button:hover { background: #7f1d1d; }
    .note { margin-top: 24px; background: #fff8e1; border: 1px solid #facc15; padding: 14px 16px; border-radius: 12px; }

    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
      .hero-top {
        grid-template-columns: 1fr;
        text-align: center;
      }
      .hero-logo img {
        max-width: 90px;
        max-height: 90px;
      }
      .hero-main-title {
        font-size: 21px;
      }
    }
  </style>
</head>
<body>
  <div class="hero">
    <div class="hero-inner">
      <div class="hero-top">
        <div class="hero-logo">
          {% if umd_logo_exists %}
            <img src="{{ url_for('umd_logo') }}" alt="UMD Logo">
          {% else %}
            <strong>UMD</strong>
          {% endif %}
        </div>

        <div class="hero-center">
          <h2 class="hero-project">Providence</h2>
          <h1 class="hero-main-title">AI-Powered Traffic Incident Management Assistant</h1>
        </div>

        <div class="hero-logo">
          {% if project_logo_exists %}
            <img src="{{ url_for('project_logo') }}" alt="Project Logo">
          {% else %}
            <strong>Project Logo</strong>
          {% endif %}
        </div>
      </div>

      <div class="hero-dashboard">Dashboard</div>
      <p class="hero-subtext">
        Upload a recorded crash video or analyze a CHART live stream URL. The dashboard will run detection, verification, metadata lookup, and PDF report generation.
      </p>
    </div>
  </div>

  <div class="page">
    <div class="grid">
      <div class="card">
        <h2>Recorded Video Analysis</h2>
        <form action="{{ url_for('start_recorded_job') }}" method="post" enctype="multipart/form-data">
          <label>Upload video file</label>
          <input type="file" name="video_file" accept=".mp4,.mov,.avi,.mkv,.m4v,.wmv,.webm" required>
          <div class="help">Common video files are supported.</div>

          <label>Optional CHART camera / stream URL</label>
          <input type="text" name="camera_lookup_url" placeholder="Paste matching CHART .m3u8 or public URL if you have it">
          <div class="help">This helps the dashboard fill camera metadata for recorded videos.</div>

          <button type="submit">Start Recorded Video Analysis</button>
        </form>
      </div>

      <div class="card">
        <h2>Live Stream Analysis</h2>
        <form action="{{ url_for('start_live_job') }}" method="post">
          <label>Select a camera</label>
          <select name="selected_camera_url" required style="width: 100%; box-sizing: border-box; padding: 12px; border: 1px solid #cbd5e1; border-radius: 10px;">
            <option value="">Choose one</option>
            {% for cam in selected_cameras %}
              <option value="{{ cam.URL }}">{{ cam.name }}</option>
            {% endfor %}
          </select>
          <div class="help">These are the selected high-quality cameras from SelectedCameras.json.</div>
          <label style="margin-top:12px;">Or paste a CHART .m3u8 stream URL manually</label>
          <input type="text" name="stream_url" placeholder="https://.../playlist.m3u8">
          <div class="help">If both are filled, the manually pasted URL will be used.</div>
          <button type="submit">Start Live Stream Analysis</button>
        </form>
      </div>
    

    <div class="note">
      <strong>How it works:</strong> once a job starts, a monitoring page opens with a progress bar, terminal-style log panel, and preview screen showing the latest detection frame.
    </div>
  </div>
</body>
</html>
"""


JOB_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Job {{ job_id }}</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f7f8fb; margin: 0; color: #111827; }
    .wrap { max-width: 1380px; margin: 0 auto; padding: 22px 18px 32px; }
    .topbar { display:flex; justify-content:space-between; gap:20px; align-items:center; flex-wrap:wrap; }
    .pill { padding: 8px 12px; border-radius: 999px; background:#e5e7eb; font-weight:bold; }
    .card { background:white; border:1px solid #e5e7eb; border-radius:18px; box-shadow:0 10px 24px rgba(15,23,42,0.05); padding:18px; }
    .progress-shell { height: 26px; background:#e5e7eb; border-radius:999px; overflow:hidden; }
    .progress-bar { height: 26px; width:{{ job.progress or 0 }}%; background:linear-gradient(90deg,#991b1b,#dc2626); transition: width 0.35s ease; }
    .step { margin-top:12px; font-size:18px; color:#334155; }
    .actions { display:flex; gap:12px; margin-top:16px; flex-wrap:wrap; }
    a.button { text-decoration:none; background:#b91c1c; color:white; padding:12px 16px; border-radius:10px; font-weight:bold; }
    .error { color:#b91c1c; font-weight:bold; margin-top:12px; }
    .stack { display:grid; grid-template-columns: 1fr; gap:20px; margin-top:18px; }
    .preview-box { background:#0b1736; border-radius:18px; min-height:520px; max-height:760px; display:flex; align-items:center; justify-content:center; overflow:hidden; }
    .preview-box img { width:100%; display:block; }
    .preview-placeholder { color:white; font-size:18px; padding:24px; text-align:center; }
    .terminal { background:#020617; color:#d1fae5; border-radius:18px; padding:16px; min-height:180px; max-height:260px; overflow:auto; font-family: Consolas, monospace; white-space:pre-wrap; }
    @media (max-width: 900px) {
      .preview-box { min-height:340px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div>
        <h1 style="margin:0 0 6px 0; font-size:30px;">Processing Job {{ job_id }}</h1>
        <div style="font-size:18px; color:#475569;">Input type: <strong>{{ job.input_type }}</strong></div>
      </div>
      <div class="pill" id="status-pill">{{ job.status|upper }}</div>
    </div>

    <div class="card" style="margin-top:18px;">
      <div class="progress-shell"><div class="progress-bar" id="progress-bar"></div></div>
      <div class="step" id="step-text">{{ job.step or '' }}</div>
      <div class="actions">
        <a class="button" href="{{ url_for('home') }}">Back Home</a>
        <a class="button" id="results-link" href="{{ url_for('job_result_page', job_id=job_id) }}" {% if job.status != 'done' %}style="display:none;"{% endif %}>Open Results Page</a>
      </div>
      <div class="error" id="error-text">{{ job.error }}</div>
    </div>

    <div class="stack">
      <div class="card">
        <h2 style="margin-top:0;">Detection Preview</h2>
        <div class="preview-box" id="preview-box">
          {% if has_preview %}
            <img id="preview-image" src="{{ url_for('api_preview', job_id=job_id) }}?v={{ preview_bust }}" alt="Preview">
          {% else %}
            <div class="preview-placeholder" id="preview-placeholder">Waiting for detection preview...</div>
            <img id="preview-image" src="" alt="Preview" style="display:none;">
          {% endif %}
        </div>
      </div>

      <div class="card">
        <h2 style="margin-top:0;">Terminal Output</h2>
        <div class="terminal" id="terminal-box">{{ logs_text }}</div>
      </div>
    </div>
  </div>

<script>
(() => {
  const jobId = {{ job_id|tojson }};
  const pollMs = {{ poll_ms }};
  let lastPreviewVersion = {{ preview_bust|int }};
  let stopped = false;

  const progressBar = document.getElementById('progress-bar');
  const stepText = document.getElementById('step-text');
  const statusPill = document.getElementById('status-pill');
  const errorText = document.getElementById('error-text');
  const terminalBox = document.getElementById('terminal-box');
  const previewImg = document.getElementById('preview-image');
  const previewPlaceholder = document.getElementById('preview-placeholder');
  const resultsLink = document.getElementById('results-link');

  async function poll() {
    if (stopped) return;
    try {
      const res = await fetch(`/jobs/${jobId}/state.json?ts=${Date.now()}`, { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      progressBar.style.width = `${data.progress || 0}%`;
      stepText.textContent = data.step || '';
      statusPill.textContent = (data.status || '').toUpperCase();
      errorText.textContent = data.error || '';
      terminalBox.textContent = data.logs_text || '';
      terminalBox.scrollTop = terminalBox.scrollHeight;

      if (data.has_preview) {
        if (previewPlaceholder) previewPlaceholder.style.display = 'none';
        previewImg.style.display = 'block';
        if ((data.preview_version || 0) !== lastPreviewVersion) {
          lastPreviewVersion = data.preview_version || 0;
          previewImg.src = `/jobs/${jobId}/preview?v=${lastPreviewVersion}`;
        }
      }

      if (data.status === 'done') {
        resultsLink.style.display = 'inline-block';
        stopped = true;
      } else if (data.status === 'error') {
        stopped = true;
      }
    } catch (err) {
      console.error(err);
    }
  }

  setInterval(poll, pollMs);
})();
</script>
</body>
</html>
"""

RESULT_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Results {{ job_id }}</title>
  <style>
    body { font-family: Arial, sans-serif; background:#f8fafc; margin:0; color:#111827; }
    .wrap { max-width:1200px; margin:0 auto; padding:24px 18px 36px; }
    .top { display:flex; justify-content:space-between; gap:20px; align-items:center; flex-wrap:wrap; }
    .buttons { display:flex; gap:12px; flex-wrap:wrap; }
    a.button { text-decoration:none; background:#991b1b; color:white; padding:12px 16px; border-radius:10px; font-weight:bold; }
    .grid { display:grid; grid-template-columns: repeat(5, 1fr); gap:16px; margin-top:18px; }
    .card { background:white; border:1px solid #e5e7eb; border-radius:18px; padding:18px; box-shadow:0 10px 24px rgba(15,23,42,0.05); }
    .card h3 { margin:0 0 8px 0; font-size:16px; color:#475569; }
    .value { font-size:20px; font-weight:bold; }
    .sub { margin-top:6px; color:#475569; line-height:1.4; }
    .full { margin-top:20px; display:grid; grid-template-columns: 1.3fr 0.7fr; gap:20px; }
    .severity-shell { height:18px; background:#e5e7eb; border-radius:999px; overflow:hidden; }
    .severity-bar { height:18px; width: {{ severity_percent }}%; background: linear-gradient(90deg, #16a34a, #f59e0b, #dc2626); }
    .image-box img { width:100%; border-radius:14px; border:1px solid #e5e7eb; }
    .desc { white-space:pre-wrap; line-height:1.5; }
    @media (max-width: 1100px) { .grid { grid-template-columns: 1fr 1fr; } .full { grid-template-columns: 1fr; } }
    @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div>
      <h1 style="margin:0 0 8px 0;">Incident Results</h1>
      <div style="color:#475569;">Job {{ job_id }}</div>
    </div>
    <div class="buttons">
      <a class="button" href="{{ url_for('job_page', job_id=job_id) }}">Back to Monitoring Page</a>
      <a class="button" href="{{ url_for('download_pdf', job_id=job_id) }}">Download Full PDF Report</a>
      <a class="button" href="{{ url_for('download_json', job_id=job_id) }}">Download JSON</a>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h3>Accident Verified</h3>
      <div class="value">{{ 'Yes' if summary.verified else 'No' }}</div>
      <div class="sub">{{ summary.verified_status }}</div>
    </div>
    <div class="card">
      <h3>Location</h3>
      <div class="value">{{ summary.city or 'Unknown' }}</div>
      <div class="sub">Lat {{ summary.latitude }} | Lon {{ summary.longitude }}</div>
    </div>
    <div class="card">
      <h3>Severity</h3>
      <div class="value">{{ summary.severity_category }}</div>
      <div class="sub">Score: {{ summary.severity_score }}</div>
      <div class="severity-shell" style="margin-top:10px;"><div class="severity-bar"></div></div>
    </div>
    <div class="card">
      <h3>Camera Name</h3>
      <div class="value" style="font-size:18px;">{{ summary.camera_name or 'Unknown' }}</div>
      <div class="sub">Primary lookup camera</div>
    </div>
    <div class="card">
      <h3>Incident Types</h3>
      <div class="value" style="font-size:18px;">{{ summary.incident_types|join(', ') if summary.incident_types else 'Unknown' }}</div>
      <div class="sub">Model-reported incident classes</div>
    </div>
  </div>

  <div class="full">
    <div class="card">
      <h2 style="margin-top:0;">Brief Accident Description</h2>
      <div class="desc">{{ summary.description }}</div>
    </div>
    <div class="card image-box">
      <h2 style="margin-top:0;">Best Detection Frame</h2>
      {% if has_best_frame %}
        <img src="{{ url_for('job_best_frame', job_id=job_id) }}" alt="Best frame">
      {% else %}
        <div class="sub">No best frame available.</div>
      {% endif %}
    </div>
  </div>
</div>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(
        HOME_TEMPLATE,
        title=PROJECT_TITLE,
        project_logo_exists=os.path.exists(backend.core.LOGO_PATH),
        umd_logo_exists=os.path.exists(UMD_LOGO_PATH),
        selected_cameras=load_selected_cameras(),
    )


@app.route("/assets/project-logo")
def project_logo():
    if not os.path.exists(backend.core.LOGO_PATH):
        abort(404)
    return send_file(backend.core.LOGO_PATH)


@app.route("/assets/umd-logo")
def umd_logo():
    if not os.path.exists(UMD_LOGO_PATH):
        abort(404)
    return send_file(UMD_LOGO_PATH)


@app.route("/start-recorded", methods=["POST"])
def start_recorded_job():
    file = request.files.get("video_file")
    lookup_url = request.form.get("camera_lookup_url", "").strip()

    if file is None or file.filename == "":
        return "No video file selected.", 400
    if not allowed_video_file(file.filename):
        return "Unsupported video file type.", 400

    filename = secure_filename(file.filename)
    state = create_job("recorded", source_value="pending", recorded_lookup_url=lookup_url)
    job_id = state["job_id"]
    saved_path = os.path.join(state["run_dir"], filename)
    file.save(saved_path)
    update_state(job_id, source_value=saved_path)
    append_log(job_id, f"Video file uploaded successfully: {saved_path}")
    spawn_worker(job_id)
    return redirect(url_for("job_page", job_id=job_id))


@app.route("/start-live", methods=["POST"])
def start_live_job():
    manual_stream_url = request.form.get("stream_url", "").strip()
    selected_camera_url = request.form.get("selected_camera_url", "").strip()

    stream_url = manual_stream_url if manual_stream_url else selected_camera_url

    if not stream_url:
        return "Please select a camera or enter a stream URL.", 400

    state = create_job("live", source_value=stream_url)
    job_id = state["job_id"]
    append_log(job_id, f"Live stream URL received: {stream_url}")
    spawn_worker(job_id)
    return redirect(url_for("job_page", job_id=job_id))


@app.route("/jobs/<job_id>")
def job_page(job_id):
    job = load_job_view(job_id)
    has_preview = bool(job.get("preview_path") and os.path.exists(job.get("preview_path")))
    logs_text = "\n".join(job.get("logs", []))
    return render_template_string(
        JOB_TEMPLATE,
        job_id=job_id,
        job=job,
        has_preview=has_preview,
        preview_bust=int(job.get("preview_version", 0)),
        logs_text=logs_text,
        poll_ms=JOB_POLL_MS,
    )


@app.route("/jobs/<job_id>/state.json")
def job_state_json(job_id):
    job = load_job_view(job_id)
    has_preview = bool(job.get("preview_path") and os.path.exists(job.get("preview_path")))
    payload = {
        "job_id": job_id,
        "status": job.get("status", ""),
        "progress": job.get("progress", 0),
        "step": job.get("step", ""),
        "error": job.get("error", ""),
        "preview_version": int(job.get("preview_version", 0) or 0),
        "has_preview": has_preview,
        "logs_text": "\n".join(job.get("logs", [])),
    }
    return app.response_class(response=json.dumps(payload), mimetype="application/json")


@app.route("/jobs/<job_id>/result")
def job_result_page(job_id):
    job = load_job_view(job_id)
    if job.get("status") != "done" or not job.get("result"):
        return redirect(url_for("job_page", job_id=job_id))
    summary = job["result"]["summary"]
    severity_score = summary.get("severity_score")
    severity_percent = 0 if isinstance(severity_score, str) else max(0, min(100, int(severity_score)))
    best_frame_path = summary.get("best_frame_path", "")
    return render_template_string(
        RESULT_TEMPLATE,
        job_id=job_id,
        summary=summary,
        severity_percent=severity_percent,
        has_best_frame=bool(best_frame_path and os.path.exists(best_frame_path)),
    )


@app.route("/jobs/<job_id>/preview")
def api_preview(job_id):
    job = load_job_view(job_id)
    preview_path = job.get("preview_path", "")
    if not preview_path or not os.path.exists(preview_path):
        abort(404)
    return send_file(preview_path, mimetype="image/jpeg")


@app.route("/jobs/<job_id>/best-frame")
def job_best_frame(job_id):
    job = load_job_view(job_id)
    if not job.get("result"):
        abort(404)
    path = job["result"]["summary"].get("best_frame_path", "")
    if not path or not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype="image/jpeg")


@app.route("/jobs/<job_id>/download/pdf")
def download_pdf(job_id):
    job = load_job_view(job_id)
    if not job.get("result"):
        abort(404)
    pdf_path = job["result"]["outputs"].get("pdf_path", "")
    if not pdf_path or not os.path.exists(pdf_path):
        abort(404)
    return send_file(pdf_path, as_attachment=True, download_name=os.path.basename(pdf_path))


@app.route("/jobs/<job_id>/download/json")
def download_json(job_id):
    job = load_job_view(job_id)
    if not job.get("result"):
        abort(404)
    json_path = job["result"]["outputs"].get("json_path", "")
    if not json_path or not os.path.exists(json_path):
        abort(404)
    return send_file(json_path, as_attachment=True, download_name=os.path.basename(json_path))


if __name__ == "__main__":
    print("Starting dashboard on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)
