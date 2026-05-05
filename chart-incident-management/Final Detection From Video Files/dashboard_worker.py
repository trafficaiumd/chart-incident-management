'''
import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import dashboard_backend as backend


class FileStateStore:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.state_path = os.path.join(run_dir, "job_state.json")
        self.log_path = os.path.join(run_dir, "job_logs.txt")
        self.preview_path = os.path.join(run_dir, "preview_latest.jpg")

    def now(self) -> float:
        return time.time()

    def load_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.state_path):
            return {}
        with open(self.state_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_state(self, state: Dict[str, Any]) -> None:
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.state_path)

    def update_state(self, **kwargs) -> None:
        state = self.load_state()
        state.update(kwargs)
        state["updated_at"] = self.now()
        self.write_state(state)

    def append_log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {message}\n")
            f.flush()

    def save_preview(self, frame_bgr) -> None:
        ok, encoded = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return
        with open(self.preview_path, "wb") as f:
            f.write(encoded.tobytes())
        state = self.load_state()
        state["preview_path"] = self.preview_path
        state["preview_version"] = int(state.get("preview_version", 0)) + 1
        state["updated_at"] = self.now()
        self.write_state(state)


class FileCallbacks(backend.DashboardCallbacks):
    def __init__(self, store: FileStateStore):
        self.store = store
        super().__init__(
            log_fn=self.log,
            progress_fn=self.progress,
            preview_fn=self.preview,
        )

    def log(self, message: str) -> None:
        self.store.append_log(str(message))

    def progress(self, value: int, step: str) -> None:
        self.store.update_state(progress=max(0, min(100, int(value))), step=str(step))

    def preview(self, frame_bgr) -> None:
        if frame_bgr is not None:
            self.store.save_preview(frame_bgr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--input-type", choices=["recorded", "live"], required=True)
    parser.add_argument("--source-value", required=True)
    parser.add_argument("--recorded-lookup-url", default="")
    args = parser.parse_args()

    store = FileStateStore(args.run_dir)
    state = store.load_state()
    state.update(
        {
            "job_id": args.job_id,
            "status": "running",
            "progress": int(state.get("progress", 1) or 1),
            "step": state.get("step", "Starting"),
            "input_type": args.input_type,
            "source_value": args.source_value,
            "recorded_lookup_url": args.recorded_lookup_url,
            "run_dir": args.run_dir,
            "worker_pid": os.getpid(),
            "updated_at": time.time(),
        }
    )
    store.write_state(state)
    callbacks = FileCallbacks(store)
    callbacks.log("Dashboard worker process started.")

    try:
        result = backend.run_dashboard_pipeline(
            input_type=args.input_type,
            source_value=args.source_value,
            recorded_lookup_url=args.recorded_lookup_url,
            run_dir=args.run_dir,
            callbacks=callbacks,
        )
        store.update_state(status="done", progress=100, step="Completed", result=result, error="")
        callbacks.log("Job completed successfully.")
    except Exception as exc:
        tb = traceback.format_exc()
        callbacks.log("ERROR: " + str(exc))
        callbacks.log(tb)
        store.update_state(status="error", step="Failed", error=str(exc), traceback=tb)
        raise


if __name__ == "__main__":
    main()
'''

import argparse
import json
import os
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"        # <--- Add this
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"    # <--- Add this
import sys
import time
import traceback
from typing import Any, Dict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import dashboard_backend as backend


class FileStateStore:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.state_path = os.path.join(run_dir, "job_state.json")
        self.log_path = os.path.join(run_dir, "job_logs.txt")
        self.preview_path = os.path.join(run_dir, "preview_latest.jpg")
        self.last_preview_time = 0 # Added for rate limiting

    def now(self) -> float:
        return time.time()

    def load_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.state_path):
            return {}
        with open(self.state_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_state(self, state: Dict[str, Any]) -> None:
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.state_path)

    def update_state(self, **kwargs) -> None:
        state = self.load_state()
        state.update(kwargs)
        state["updated_at"] = self.now()
        self.write_state(state)

    def append_log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {message}\n")
            f.flush()

    def save_preview(self, frame_bgr) -> None:
        # Optimization: Rate limit the preview updates to ~7 FPS
        current_time = self.now()
        if current_time - self.last_preview_time < 0.15:
            return
        
        # Optimization: Use lower JPEG quality for speed
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if not ok:
            return
            
        with open(self.preview_path, "wb") as f:
            f.write(encoded.tobytes())
            
        self.last_preview_time = current_time
        
        # Optimization: Only update state if necessary to avoid extra JSON I/O
        state = self.load_state()
        state["preview_path"] = self.preview_path
        state["preview_version"] = int(state.get("preview_version", 0)) + 1
        state["updated_at"] = self.now()
        self.write_state(state)


class FileCallbacks(backend.DashboardCallbacks):
    def __init__(self, store: FileStateStore):
        self.store = store
        super().__init__(
            log_fn=self.log,
            progress_fn=self.progress,
            preview_fn=self.preview,
        )

    def log(self, message: str) -> None:
        self.store.append_log(str(message))

    def progress(self, value: int, step: str) -> None:
        self.store.update_state(progress=max(0, min(100, int(value))), step=str(step))

    def preview(self, frame_bgr) -> None:
        if frame_bgr is not None:
            self.store.save_preview(frame_bgr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--input-type", choices=["recorded", "live"], required=True)
    parser.add_argument("--source-value", required=True)
    parser.add_argument("--recorded-lookup-url", default="")
    args = parser.parse_args()

    store = FileStateStore(args.run_dir)
    state = store.load_state()
    state.update(
        {
            "job_id": args.job_id,
            "status": "running",
            "progress": int(state.get("progress", 1) or 1),
            "step": state.get("step", "Starting"),
            "input_type": args.input_type,
            "source_value": args.source_value,
            "recorded_lookup_url": args.recorded_lookup_url,
            "run_dir": args.run_dir,
            "worker_pid": os.getpid(),
            "updated_at": time.time(),
        }
    )
    store.write_state(state)
    callbacks = FileCallbacks(store)
    callbacks.log("Dashboard worker process started.")

    try:
        result = backend.run_dashboard_pipeline(
            input_type=args.input_type,
            source_value=args.source_value,
            recorded_lookup_url=args.recorded_lookup_url,
            run_dir=args.run_dir,
            callbacks=callbacks,
        )
        store.update_state(status="done", progress=100, step="Completed", result=result, error="")
        callbacks.log("Job completed successfully.")
    except Exception as exc:
        tb = traceback.format_exc()
        callbacks.log("ERROR: " + str(exc))
        callbacks.log(tb)
        store.update_state(status="error", step="Failed", error=str(exc), traceback=tb)
        raise


if __name__ == "__main__":
    main()