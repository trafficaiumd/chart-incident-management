import os
import cv2
from ultralytics import YOLO

# -----------------------------
# Paths. Change these based on your setup
# -----------------------------
model_path = "C:/Users/Ariyan/College/UMD/ENCE 465/Practice/Code 3/weights/epoch14.pt"
video_path = "C:/Users/Ariyan/College/UMD/ENCE 465/Practice/Code 1/static/files/istockphoto.mp4"
output_dir = "C:/Users/Ariyan/College/UMD/ENCE 465/Practice/Code 1/static/files/output"
logo_path = "C:/Users/Ariyan/College/UMD/ENCE 465/final logo png.png"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Settings
# -----------------------------
CONF_THRESHOLD = 0.75
VID_STRIDE = 20
FRAMES_BEFORE = 55
FRAMES_AFTER = 65
SHOW_WINDOW = True
ACCIDENT_CLASS_NAME = "accident"

# Logo settings
LOGO_SCALE = 0.1
LOGO_MARGIN = 20

# -----------------------------
# Logo helpers
# -----------------------------
def load_logo_rgba(path):
    logo = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        raise RuntimeError(f"Could not load logo: {path}")

    if len(logo.shape) == 2:
        logo = cv2.cvtColor(logo, cv2.COLOR_GRAY2BGRA)
    elif logo.shape[2] == 3:
        b, g, r = cv2.split(logo)
        alpha = 255 * (b > -1).astype("uint8")
        logo = cv2.merge((b, g, r, alpha))

    return logo

def resize_logo_for_frame(logo_rgba, frame_width):
    target_w = max(1, int(frame_width * LOGO_SCALE))
    h, w = logo_rgba.shape[:2]
    aspect = h / w
    target_h = max(1, int(target_w * aspect))
    return cv2.resize(logo_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)

def overlay_logo_top_right(frame_bgr, logo_rgba, margin=15):
    fh, fw = frame_bgr.shape[:2]
    lh, lw = logo_rgba.shape[:2]

    if lw >= fw or lh >= fh:
        return frame_bgr

    x = fw - lw - margin
    y = margin

    if x < 0 or y < 0:
        return frame_bgr

    output = frame_bgr.copy()

    logo_bgr = logo_rgba[:, :, :3]
    alpha = logo_rgba[:, :, 3] / 255.0
    alpha = alpha[:, :, None]

    roi = output[y:y + lh, x:x + lw]
    blended = (alpha * logo_bgr + (1 - alpha) * roi).astype("uint8")
    output[y:y + lh, x:x + lw] = blended

    return output

# -----------------------------
# Load model
# -----------------------------
model = YOLO(model_path)
class_names = model.names

# Load logo once
logo_rgba_original = load_logo_rgba(logo_path)

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize logo for this video
logo_rgba = resize_logo_for_frame(logo_rgba_original, width)

# -----------------------------
# Track best accident detection
# -----------------------------
best_conf = -1.0
best_frame_idx = None
best_raw_frame = None
best_annotated_frame = None

frame_idx = 0
last_display_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % VID_STRIDE == 0:
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]
        annotated = result.plot()

        # Add logo to annotated frame for display and saved annotated image
        annotated_with_logo = overlay_logo_top_right(annotated, logo_rgba, LOGO_MARGIN)

        # Save this annotated frame for display stability
        last_display_frame = annotated_with_logo.copy()
        display_frame = last_display_frame

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = str(class_names[cls_id]).lower()

                if cls_name == ACCIDENT_CLASS_NAME and conf > best_conf:
                    best_conf = conf
                    best_frame_idx = frame_idx
                    best_raw_frame = frame.copy()
                    best_annotated_frame = annotated_with_logo.copy()
    else:
        # Instead of showing raw frame, keep showing last annotated frame
        if last_display_frame is not None:
            display_frame = last_display_frame
        else:
            display_frame = overlay_logo_top_right(frame, logo_rgba, LOGO_MARGIN)

    if SHOW_WINDOW:
        cv2.imshow("YOLO Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# If no accident found
# -----------------------------
if best_frame_idx is None:
    print("No accident detected in the video.")
else:
    print(f"Best accident detection:")
    print(f"  Confidence: {best_conf:.4f}")
    print(f"  Frame index: {best_frame_idx}")

    best_raw_frame_path = os.path.join(output_dir, "best_accident_frame_raw.jpg")
    best_annotated_frame_path = os.path.join(output_dir, "best_accident_frame_annotated.jpg")

    cv2.imwrite(best_raw_frame_path, best_raw_frame)
    cv2.imwrite(best_annotated_frame_path, best_annotated_frame)

    print(f"Saved raw best frame to: {best_raw_frame_path}")
    print(f"Saved annotated best frame to: {best_annotated_frame_path}")

    start_frame = max(0, best_frame_idx - FRAMES_BEFORE)
    end_frame = min(frame_count - 1, best_frame_idx + FRAMES_AFTER)

    clip_path = os.path.join(output_dir, "accident_context_clip_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_idx = start_frame

    while current_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        frame_with_logo = overlay_logo_top_right(frame, logo_rgba, LOGO_MARGIN)
        writer.write(frame_with_logo)
        current_idx += 1

    cap.release()
    writer.release()

    print(f"Saved raw accident clip to: {clip_path}")