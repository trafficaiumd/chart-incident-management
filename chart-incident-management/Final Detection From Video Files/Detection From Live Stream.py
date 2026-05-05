import os
import cv2
import time
from datetime import datetime
from collections import deque
from ultralytics import YOLO

# =========================================================
# CONFIG
# =========================================================
model_path = "C:/Users/Ariyan/College/UMD/ENCE 465/Practice/Code 3/weights/epoch14.pt"
stream_url = "https://strmr10.sha.maryland.gov/rtplive/b300be2300a4005c0055fa36c4235c0a/chunklist_w399738125.m3u8"
output_dir = "C:/Users/Ariyan/College/UMD/ENCE 465/Practice/Code 1/static/files/output"
logo_path = "C:/Users/Ariyan/College/UMD/ENCE 465/final logo png.png"
os.makedirs(output_dir, exist_ok=True)

# Detection thresholds
ACCIDENT_CONF_THRESHOLD = 0.91 #Used to be 0.9
VEHICLE_CONF_THRESHOLD = 0.5 #Used to be 0.5

REQUIRED_DETECTIONS = 5
DETECTION_WINDOW_SEC = 5
VID_STRIDE = 4
SHOW_WINDOW = True

PRE_EVENT_SEC = 5
POST_EVENT_SEC = 5
BUFFER_SEC = PRE_EVENT_SEC + 2
COOLDOWN_SEC = 15

ACCIDENT_CLASS_NAME = "accident"
VEHICLE_CLASS_NAME = "vehicle"

MIN_DETECTION_GAP_SEC = 0.5

# Logo settings
LOGO_SCALE = 0.1          # fraction of frame width
LOGO_MARGIN = 20           # pixels from edges

# =========================================================
# LOAD MODEL
# =========================================================
model = YOLO(model_path)
class_names = model.names
print("Model classes:", class_names)

# =========================================================
# LOGO HELPERS
# =========================================================
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

    overlay = frame_bgr.copy()

    logo_bgr = logo_rgba[:, :, :3]
    alpha = logo_rgba[:, :, 3] / 255.0
    alpha = alpha[:, :, None]

    roi = overlay[y:y + lh, x:x + lw]
    blended = (alpha * logo_bgr + (1 - alpha) * roi).astype("uint8")
    overlay[y:y + lh, x:x + lw] = blended
    return overlay

# Load logo once
logo_rgba_original = load_logo_rgba(logo_path)

# =========================================================
# OPEN STREAM
# =========================================================
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    raise RuntimeError(f"Could not open stream: {stream_url}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps > 120:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width <= 0 or height <= 0:
    width, height = 1280, 720

print(f"Using FPS={fps}, size=({width}, {height})")

# Prepare resized logo for this stream size
logo_rgba = resize_logo_for_frame(logo_rgba_original, width)

# =========================================================
# STATE
# =========================================================
max_buffer_frames = int(BUFFER_SEC * fps) + 5
frame_buffer = deque(maxlen=max_buffer_frames)     # stores (timestamp, frame)
recent_detections = deque()                        # stores (timestamp, confidence)

frame_idx = 0
last_display_frame = None
last_saved_time = 0
last_detection_time = 0
event_counter = 0

# =========================================================
# HELPERS
# =========================================================
def format_ts(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def prune_old_detections(now_ts):
    while recent_detections and (now_ts - recent_detections[0][0] > DETECTION_WINDOW_SEC):
        recent_detections.popleft()

def save_event_clip(pre_frames, cap_obj, fps_value, out_w, out_h, save_dir, event_id, post_sec, logo_rgba):
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    clip_path = os.path.join(save_dir, f"accident_event_{event_id}_{timestamp_str}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(clip_path, fourcc, fps_value, (out_w, out_h))

    # Write buffered frames first, with logo added
    for _, buffered_frame in pre_frames:
        frame_out = overlay_logo_top_right(buffered_frame.copy(), logo_rgba, LOGO_MARGIN)
        writer.write(frame_out)

    # Then capture post-event live frames, with logo added
    post_frames_to_capture = int(post_sec * fps_value)
    captured = 0

    while captured < post_frames_to_capture:
        ret, frame = cap_obj.read()
        if not ret:
            print("Warning: stream ended or frame read failed while capturing post-event clip.")
            break

        if frame.shape[1] != out_w or frame.shape[0] != out_h:
            frame = cv2.resize(frame, (out_w, out_h))

        frame_out = overlay_logo_top_right(frame.copy(), logo_rgba, LOGO_MARGIN)
        writer.write(frame_out)
        captured += 1

    writer.release()
    return clip_path

# =========================================================
# MAIN LOOP
# =========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed. Waiting briefly and trying again...")
        time.sleep(1)
        cap.release()
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print("Reconnection failed. Retrying...")
            continue
        else:
            print("Reconnected to stream.")
            continue

    now_ts = time.time()

    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height))

    # Store RAW frame in rolling buffer (without logo)
    frame_buffer.append((now_ts, frame.copy()))

    display_frame = frame

    if frame_idx % VID_STRIDE == 0:
        results = model(frame, conf=min(ACCIDENT_CONF_THRESHOLD, VEHICLE_CONF_THRESHOLD), verbose=False)
        result = results[0]

        annotated = frame.copy()

        found_accident_this_frame = False
        best_accident_conf_this_frame = -1.0

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                cls_name = str(class_names[cls_id]).lower()

                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

                draw_this_box = False
                label = None
                color = (0, 255, 0)

                if cls_name == VEHICLE_CLASS_NAME and conf >= VEHICLE_CONF_THRESHOLD:
                    draw_this_box = True
                    label = f"{cls_name} {conf:.2f}"
                    color = (255, 90, 0) 

                elif cls_name == ACCIDENT_CLASS_NAME and conf >= ACCIDENT_CONF_THRESHOLD:
                    draw_this_box = True
                    label = f"{cls_name} {conf:.2f}"
                    color = (0, 30, 255)

                    found_accident_this_frame = True
                    best_accident_conf_this_frame = max(best_accident_conf_this_frame, conf)

                if draw_this_box:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.6,
                        color,
                        1
                    )

        annotated = overlay_logo_top_right(annotated, logo_rgba, LOGO_MARGIN)

        last_display_frame = annotated.copy()
        display_frame = last_display_frame

        if found_accident_this_frame:
            if (now_ts - last_detection_time) >= MIN_DETECTION_GAP_SEC:
                recent_detections.append((now_ts, best_accident_conf_this_frame))
                last_detection_time = now_ts
                print(
                    f"[DETECTION] accident conf={best_accident_conf_this_frame:.3f} "
                    f"| at {format_ts(now_ts)} "
                    f"| detections in window={len(recent_detections)}"
                )

        prune_old_detections(now_ts)

        in_cooldown = (now_ts - last_saved_time) < COOLDOWN_SEC

        if not in_cooldown and len(recent_detections) >= REQUIRED_DETECTIONS:
            event_counter += 1
            confirmation_ts = time.time()

            pre_frames = [(ts, fr) for ts, fr in frame_buffer if confirmation_ts - ts <= PRE_EVENT_SEC]

            print("\n" + "=" * 70)
            print(f"[EVENT CONFIRMED] Event #{event_counter}")
            print(f"Confirmation time: {format_ts(confirmation_ts)}")
            print(f"Rule met: {REQUIRED_DETECTIONS} accident detections within {DETECTION_WINDOW_SEC} seconds")
            print("Triggering detections:")

            for det_num, (det_ts, det_conf) in enumerate(recent_detections, start=1):
                print(f"  {det_num}. time={format_ts(det_ts)} | conf={det_conf:.3f}")

            clip_path = save_event_clip(
                pre_frames=pre_frames,
                cap_obj=cap,
                fps_value=fps,
                out_w=width,
                out_h=height,
                save_dir=output_dir,
                event_id=event_counter,
                post_sec=POST_EVENT_SEC,
                logo_rgba=logo_rgba
            )

            print(f"Saved clip: {clip_path}")
            print("=" * 70 + "\n")

            last_saved_time = time.time()
            recent_detections.clear()
            frame_buffer.clear()
            last_display_frame = None

    else:
        if last_display_frame is not None:
            display_frame = last_display_frame
        else:
            display_frame = overlay_logo_top_right(display_frame, logo_rgba, LOGO_MARGIN)

    if SHOW_WINDOW:
        cv2.imshow("Live YOLO Stream", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()