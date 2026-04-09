'''
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "tls_verify=0"
import cv2

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

import sys
import cv2
import time
import torch
import gradio as gr
from ultralytics import YOLO

print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("GPU ready:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# Good speed setting on RTX 4090
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model = YOLO("/home/group1/chart-incident-management/YOLO Testing Ariyan/Practice/CODE 2/weights/epoch14.pt")

def detect_stream(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise ValueError("Could not open the stream URL.")

    # Try to keep stream buffer small
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    last_emit = 0
    target_delay = 0.40   # lower = more updates, higher = less load

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        now = time.time()
        if now - last_emit < target_delay:
            continue

        frame = cv2.resize(frame, (640, 360))

        results = model(
            frame,
            conf=0.6,
            imgsz=320,      # smaller = faster
            device=0,       # GPU
            half=True,      # FP16 on CUDA for speed
            verbose=False
        )

        plotted = results[0].plot()
        plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

        last_emit = now
        yield plotted

app = gr.Interface(
    fn=detect_stream,
    inputs=gr.Textbox(label="Enter CHART Camera .m3u8 URL"),
    outputs=gr.Image(label="Live Detection"),
    title="Live Traffic Accident Detection",
    description="Paste a CHART camera stream URL to detect accidents and vehicles."
)

app.launch(server_name="0.0.0.0", server_port=7860, share=False)
#'''