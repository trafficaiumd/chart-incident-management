'''
from ultralytics import YOLO

# Load the fine-tuned model
model = YOLO('path/to/best.pt')

# Run inference on an image
results = model('path/to/image.jpg')

# Visualize the results
results.show()
'''

#Image Detection
import gradio as gr
from ultralytics import YOLO
import cv2

# Load model (change path if needed)
model = YOLO("/home/group1/chart-incident-management/YOLO Testing Ariyan/Practice/CODE 2/weights/epoch14.pt")

def detect(file):
    results = model(file, conf=0.7)
    img = results[0].plot()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

app = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="filepath"),
    outputs="image",
    title="Traffic Accident Detection",
    description="Upload an image and detect accidents"
)

app.launch()

'''
#Video Detection
import gradio as gr
from ultralytics import YOLO
import cv2
import tempfile

# Load your model
model = YOLO("C:/Users/Ariyan/College/UMD/ENCE 465/Practice/Code 3/weights/epoch14.pt")

def process_video(video_file):
    cap = cv2.VideoCapture(video_file)

    # Create a temporary file (auto-deletes later)
    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = temp_output.name

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.7)
        annotated = results[0].plot()

        out.write(annotated)

    cap.release()
    out.release()

    return output_path

app = gr.Interface(
    fn=process_video,
    inputs=gr.Video(),
    outputs=gr.Video(),
    title="Traffic Accident Detection (Video)",
    description="Upload a video and see detection results"
)

app.launch()
'''