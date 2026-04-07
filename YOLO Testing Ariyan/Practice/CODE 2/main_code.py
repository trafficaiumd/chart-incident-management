'''
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "tls_verify=0"
import cv2
'''
from ultralytics import YOLO

model = YOLO("/home/group1/chart-incident-management/YOLO Testing Ariyan/Practice/CODE 2/weights/epoch14.pt")

stream_url = "https://strmr10.sha.maryland.gov/rtplive/b300be2300a4005c0055fa36c4235c0a/chunklist_w399738125.m3u8"

results = model(
    source=stream_url,
    stream=True,
    show=False,
    conf=0.6,
    device=0
)

for r in results:
    print(r.boxes)
'''

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "tls_verify=0"

import gradio as gr
from ultralytics import YOLO
import cv2
import time

# Load model
model = YOLO("/home/group1/chart-incident-management/YOLO Testing Ariyan/Practice/CODE 2/weights/epoch14.pt")

def detect_stream(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    last_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Skip frames (fast!)
        if time.time() - last_time < 0.9:
            continue

        # Resize (huge speed boost)
        frame = cv2.resize(frame, (640, 360))

        results = model(frame, conf=0.7, imgsz=416, device="cpu")

        img = results[0].plot()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        last_time = time.time()

        yield img # 👈 this streams frames to Gradio

        time.sleep(1)  # reduce load

app = gr.Interface(
    fn=detect_stream,
    inputs=gr.Textbox(label="Enter CHART Camera .m3u8 URL"),
    outputs=gr.Image(label="Live Detection"),
    title="Live Traffic Accident Detection",
    description="Paste a CHART camera stream URL to detect accidents in real time"
)

app.launch()
'''