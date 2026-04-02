

# radar sensors -> abrupt change in distance -> possible collision scored 
# -> triggers ai processing of camera feed recording -> model assesses if collision is likely 
# -> if yes, alert CHART and EMS/Police and save video clip of incident for review

import json
import cv2
import threading
import time
from collections import deque
from scipy.spatial import KDTree

# =========================
# FILE PATHS
# =========================

CAMERA_FILE = "GetCameras.json"
SENSOR_FILE = "GetSensors.json"

# =========================
# LOADERS
# =========================

def load_cameras(path):
    with open(path) as f:
        return json.load(f)

def load_sensors(path):
    with open(path) as f:
        return json.load(f)

# =========================
# KD TREE
# =========================

def build_kdtree(cameras):
    points = [(cam["lat"], cam["lon"]) for cam in cameras]
    return KDTree(points)

def get_nearest_cameras(lat, lon, tree, cameras, k=3):
    distances, indices = tree.query((lat, lon), k=k)

    if k == 1:
        indices = [indices]

    return [cameras[i] for i in indices]

# =========================
# CAMERA STREAM
# =========================

class CameraStream:
    def __init__(self, url, buffer_size=150):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.buffer = deque(maxlen=buffer_size)

        if not self.cap.isOpened():
            print(f"⚠️ Failed to open stream: {url}")

    def read_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            self.buffer.append(frame)
            time.sleep(0.03)  # ~30 FPS

    def get_buffer(self):
        return list(self.buffer)

# =========================
# START STREAMS
# =========================

def start_camera_streams(camera_list):
    streams = {}

    for cam in camera_list:
        stream = CameraStream(cam["url"])
        streams[cam["id"]] = stream

        threading.Thread(target=stream.read_loop, daemon=True).start()

    return streams

# =========================
# AI MODEL (PLACEHOLDER)
# =========================

def call_ai_model(frames):
    print(f"🤖 AI analyzing {len(frames)} frames...")
    return {"accident": True}

# =========================
# SAVE CLIP
# =========================

def save_clip(frames, filename):
    if not frames:
        return

    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))

    for f in frames:
        out.write(f)

    out.release()

# =========================
# RADAR HANDLER
# =========================

last_trigger_time = {}
COOLDOWN = 30

def handle_radar_event(sensor_id, sensors, sensor_to_cameras, streams):
    if sensor_id not in sensors:
        print("Unknown sensor")
        return

    now = time.time()

    if sensor_id in last_trigger_time and now - last_trigger_time[sensor_id] < COOLDOWN:
        print("⏳ Cooldown active")
        return

    last_trigger_time[sensor_id] = now

    nearest_cams = sensor_to_cameras[sensor_id]

    print(f"\n📡 Sensor {sensor_id} triggered")
    print("📍 Closest Cameras:")

    for cam in nearest_cams:
        print(f" - {cam['id']}")

    for cam in nearest_cams:
        stream = streams.get(cam["id"])

        if not stream:
            continue

        frames = stream.get_buffer()

        if len(frames) < 10:
            continue

        result = call_ai_model(frames)

        if result["accident"]:
            filename = f"incident_{sensor_id}_{cam['id']}.mp4"
            save_clip(frames, filename)
            print(f"🚨 Accident detected! Saved: {filename}")
            return

    print("✅ No accident detected")

# =========================
# MAIN
# =========================

if __name__ == "__camerafeed__":
    # Load data
    cameras = load_cameras(CAMERA_FILE)
    sensors_list = load_sensors(SENSOR_FILE)

    # Convert sensors to dict for fast lookup
    sensors = {s["id"]: s for s in sensors_list}

    # Build KDTree
    tree = build_kdtree(cameras)

    # Precompute nearest 3 cameras per sensor
    sensor_to_cameras = {
        s_id: get_nearest_cameras(s["lat"], s["lon"], tree, cameras, k=3)
        for s_id, s in sensors.items()
    }

    # ⚠️ Limit streams for testing (IMPORTANT)
    streams = start_camera_streams(cameras[:10])

    print("⏳ Warming up camera buffers...")
    time.sleep(5)

    # Simulated radar trigger
    handle_radar_event("sensor_1", sensors, sensor_to_cameras, streams)