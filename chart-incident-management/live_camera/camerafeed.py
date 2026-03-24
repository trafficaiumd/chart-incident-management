

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
# 1. LOAD CAMERAS JSON
# =========================

def load_cameras(path):
    with open(path) as f:
        data = json.load(f)

    cameras = []

    for cam in data:
        cameras.append({
            "id": cam["id"],
            "url": cam["url"],
            "lat": cam["lat"],
            "lon": cam["lon"]
        })

    return cameras


# =========================
# 2. LOAD SENSORS JSON
# =========================

def load_sensors(path):
    with open(path) as f:
        data = json.load(f)

    sensors = {}

    for s in data:
        sensors[s["id"]] = {
            "lat": s["lat"],
            "lon": s["lon"]
        }

    return sensors


# =========================
# 3. BUILD KD TREE (CAMERAS)
# =========================

def build_kdtree(cameras):
    points = [(cam["lat"], cam["lon"]) for cam in cameras]
    tree = KDTree(points)
    return tree


# =========================
# 4. FIND CLOSEST CAMERAS
# =========================

def get_nearest_cameras(lat, lon, tree, cameras, k=3):
    distances, indices = tree.query((lat, lon), k=k)

    if k == 1:
        indices = [indices]

    return [cameras[i] for i in indices]


# =========================
# 5. CAMERA STREAM CLASS
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
            time.sleep(0.03)

    def get_buffer(self):
        return list(self.buffer)


# =========================
# 6. START CAMERA STREAMS
# =========================

def start_camera_streams(camera_list):
    streams = {}

    for cam in camera_list:
        stream = CameraStream(cam["url"])
        streams[cam["id"]] = stream

        threading.Thread(target=stream.read_loop, daemon=True).start()

    return streams


# =========================
# 7. AI MODEL PLACEHOLDER
# =========================

def call_ai_model(frames):
    print(f"🤖 AI analyzing {len(frames)} frames...")
    return {"accident": True}


# =========================
# 8. SAVE VIDEO CLIP
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
# 9. RADAR EVENT HANDLER
# =========================

last_trigger_time = {}
COOLDOWN = 30

def handle_radar_event(sensor_id, sensors, tree, cameras, streams):
    if sensor_id not in sensors:
        print("Unknown sensor")
        return

    lat = sensors[sensor_id]["lat"]
    lon = sensors[sensor_id]["lon"]

    now = time.time()

    if sensor_id in last_trigger_time and now - last_trigger_time[sensor_id] < COOLDOWN:
        print("⏳ Cooldown active")
        return

    last_trigger_time[sensor_id] = now

    nearest_cams = get_nearest_cameras(lat, lon, tree, cameras, k=3)

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
# 10. MAIN
# =========================

if __name__ == "__main__":
    cameras = load_cameras("cameras.json")
    sensors = load_sensors("sensors.json")

    tree = build_kdtree(cameras)

    # ⚠️ limit for testing
    streams = start_camera_streams(cameras[:10])

    # Simulate radar trigger
    time.sleep(5)

    handle_radar_event("sensor_1", sensors, tree, cameras, streams)