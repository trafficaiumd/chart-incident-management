import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"    
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8" 

import json
import time
import tempfile
import importlib.util
from datetime import datetime
from collections import deque
from typing import Any, Dict, Optional

import cv2
from google.genai import types
from google.genai import errors as genai_errors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_SCRIPT_FILENAME = "Final_Integration.py" # Adjusted to match your uploaded filename
CORE_SCRIPT_PATH = os.path.join(SCRIPT_DIR, CORE_SCRIPT_FILENAME)


class DashboardCallbacks:
    def __init__(self, log_fn=None, progress_fn=None, preview_fn=None):
        self.log_fn = log_fn
        self.progress_fn = progress_fn
        self.preview_fn = preview_fn

    def log(self, message: str) -> None:
        if self.log_fn:
            self.log_fn(str(message))

    def progress(self, value: int, step: str) -> None:
        if self.progress_fn:
            self.progress_fn(int(value), str(step))

    def preview(self, frame_bgr) -> None:
        if self.preview_fn is not None and frame_bgr is not None:
            self.preview_fn(frame_bgr)


def load_core_module():
    if not os.path.exists(CORE_SCRIPT_PATH):
        raise FileNotFoundError(
            f"Core pipeline script not found: {CORE_SCRIPT_PATH}. "
            f"Place a copy of your final pipeline script next to this file or update CORE_SCRIPT_FILENAME."
        )

    spec = importlib.util.spec_from_file_location("dashboard_pipeline_core", CORE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load core pipeline module from {CORE_SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load_core_module()


def format_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def generate_content_with_retry_dashboard(model: str, contents, config=None, callbacks: Optional[DashboardCallbacks] = None, max_retries: int = 5, delay: int = 5):
    for attempt in range(max_retries):
        try:
            return core.client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except genai_errors.ServerError as e:
            if callbacks:
                callbacks.log(f"Gemini temporary server error ({e}). Retrying in {delay} seconds...")
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)


class JobLogger:
    def __init__(self, callbacks: Optional[DashboardCallbacks]):
        self.callbacks = callbacks or DashboardCallbacks()

    def log(self, message: str) -> None:
        self.callbacks.log(message)

    def progress(self, value: int, step: str) -> None:
        self.callbacks.progress(value, step)

    def preview(self, frame_bgr) -> None:
        self.callbacks.preview(frame_bgr)


def ensure_core_files() -> None:
    core.file_exists_or_raise(core.YOLO_MODEL_PATH, "YOLO model")
    core.file_exists_or_raise(core.LOGO_PATH, "Logo")
    core.file_exists_or_raise(core.PROMPT_TXT_PATH, "Prompt file")
    # core.file_exists_or_raise(core.GEOJSON_PATH, "GeoJSON file") # Added/Checked based on core
    # core.file_exists_or_raise(core.GETCAMERAS_PATH, "GetCameras file")


def _prepare_model() -> Any:
    model = core.YOLO(core.YOLO_MODEL_PATH)
    try:
        model.to("cpu")
    except Exception:
        pass
    return model


def detect_from_recorded_video_dashboard(video_path: str, run_dir: str, callbacks: Optional[DashboardCallbacks] = None) -> Dict[str, str]:
    logger = JobLogger(callbacks)
    logger.log("=" * 80)
    logger.log("RUNNING RECORDED-VIDEO DETECTION")
    logger.log("=" * 80)
    logger.progress(10, "Recorded video received")

    CONF_THRESHOLD = 0.75
    VID_STRIDE = 8 #This was 20 before
    FRAMES_BEFORE = 55
    FRAMES_AFTER = 65
    ACCIDENT_CLASS_NAME = "accident"
    LOGO_SCALE = 0.1
    LOGO_MARGIN = 20

    best_raw_frame_path = os.path.join(run_dir, "best_accident_frame_raw.jpg")
    best_annotated_frame_path = os.path.join(run_dir, "best_accident_frame_annotated.jpg")
    clip_path = os.path.join(run_dir, "accident_context_clip_raw.mp4")

    logger.log("Loading YOLO model on CPU...")
    model = _prepare_model()
    class_names = model.names
    logo_rgba_original = core.load_logo_rgba(core.LOGO_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.log(f"Video opened successfully | FPS={fps:.2f} | Frames={frame_count} | Size=({width}, {height})")
    logo_rgba = core.resize_logo_for_frame(logo_rgba_original, width, LOGO_SCALE)

    best_conf = -1.0
    best_frame_idx = None
    best_raw_frame = None
    best_annotated_frame = None

    frame_idx = 0
    last_display_frame = None
    last_progress_emit = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count > 0:
            progress = 10 + int(40 * (frame_idx / max(frame_count, 1)))
            if progress != last_progress_emit:
                logger.progress(progress, "Scanning recorded video for accident detections")
                last_progress_emit = progress

        # PERFORMANCE FIX: Only update the dashboard preview when YOLO analyzes a frame
        if frame_idx % VID_STRIDE == 0:
            results = model(frame, conf=CONF_THRESHOLD, verbose=False, device="cpu")
            result = results[0]
            annotated = result.plot()
            annotated_with_logo = core.overlay_logo_top_right(annotated, logo_rgba, LOGO_MARGIN)

            last_display_frame = annotated_with_logo.copy()
            logger.preview(last_display_frame) # Update dashboard preview

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
                        logger.log(f"[DETECTION] New best accident frame | confidence={best_conf:.4f} | frame={best_frame_idx}")
        # REMOVED redundancy: The dashboard no longer attempts to preview every skipped frame.

        frame_idx += 1

    cap.release()

    if best_frame_idx is None:
        raise RuntimeError("No accident detected in the recorded video.")

    logger.log("Best accident detection:")
    logger.log(f"  Confidence: {best_conf:.4f}")
    logger.log(f"  Frame index: {best_frame_idx}")

    cv2.imwrite(best_raw_frame_path, best_raw_frame)
    cv2.imwrite(best_annotated_frame_path, best_annotated_frame)

    logger.log(f"Saved raw best frame to: {best_raw_frame_path}")
    logger.log(f"Saved annotated best frame to: {best_annotated_frame_path}")
    logger.preview(best_annotated_frame)
    logger.progress(55, "Best recorded-video frame found")

    start_frame = max(0, best_frame_idx - FRAMES_BEFORE)
    end_frame = min(frame_count - 1, best_frame_idx + FRAMES_AFTER)

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

        frame_with_logo = core.overlay_logo_top_right(frame, logo_rgba, LOGO_MARGIN)
        writer.write(frame_with_logo)
        current_idx += 1

    cap.release()
    writer.release()

    logger.log(f"Saved raw accident clip to: {clip_path}")
    logger.progress(60, "Recorded-video detection complete")

    return {
        "raw_image_path": best_raw_frame_path,
        "annotated_image_path": best_annotated_frame_path,
        "clip_video_path": clip_path,
    }


def detect_from_live_stream_dashboard(stream_url: str, run_dir: str, callbacks: Optional[DashboardCallbacks] = None) -> Dict[str, str]:
    logger = JobLogger(callbacks)
    logger.log("=" * 80)
    logger.log("RUNNING LIVE-STREAM DETECTION")
    logger.log("=" * 80)
    logger.progress(10, "Connecting to live stream")

    ACCIDENT_CONF_THRESHOLD = 0.98 #Used to be 0.9
    VEHICLE_CONF_THRESHOLD = 0.45 #Used to be 0.65
    REQUIRED_DETECTIONS = 6 #This was 5 before
    DETECTION_WINDOW_SEC = 4 #This was 5 before
    VID_STRIDE = 4 
    PRE_EVENT_SEC = 5
    POST_EVENT_SEC = 5
    BUFFER_SEC = PRE_EVENT_SEC + 2
    COOLDOWN_SEC = 15
    ACCIDENT_CLASS_NAME = "accident"
    VEHICLE_CLASS_NAME = "vehicle"
    MIN_DETECTION_GAP_SEC = 0.5
    LOGO_SCALE = 0.1
    LOGO_MARGIN = 20

    best_raw_frame_path = os.path.join(run_dir, "best_accident_frame_raw.jpg")
    best_annotated_frame_path = os.path.join(run_dir, "best_accident_frame_annotated.jpg")
    clip_path = os.path.join(run_dir, "accident_context_clip_raw.mp4")

    def prune_old_detections(recent_detections, now_ts):
        while recent_detections and (now_ts - recent_detections[0][0] > DETECTION_WINDOW_SEC):
            recent_detections.popleft()

    def save_event_clip(pre_frames, cap_obj, fps_value, out_w, out_h, save_path, post_sec, logo_rgba):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps_value, (out_w, out_h))

        for _, buffered_frame in pre_frames:
            frame_out = core.overlay_logo_top_right(buffered_frame.copy(), logo_rgba, LOGO_MARGIN)
            writer.write(frame_out)

        post_frames_to_capture = int(post_sec * fps_value)
        captured = 0

        while captured < post_frames_to_capture:
            ret, frame = cap_obj.read()
            if not ret:
                logger.log("Warning: stream ended or frame read failed while capturing post-event clip.")
                break

            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h))

            frame_out = core.overlay_logo_top_right(frame.copy(), logo_rgba, LOGO_MARGIN)
            writer.write(frame_out)
            captured += 1

        writer.release()

    logger.log("Loading YOLO model on CPU...")
    model = _prepare_model()
    class_names = model.names
    logger.log(f"Model classes: {class_names}")

    logo_rgba_original = core.load_logo_rgba(core.LOGO_PATH)

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

    logger.log(f"Using FPS={fps}, size=({width}, {height})")
    logger.progress(15, "Live stream connected")

    logo_rgba = core.resize_logo_for_frame(logo_rgba_original, width, LOGO_SCALE)

    max_buffer_frames = int(BUFFER_SEC * fps) + 5
    frame_buffer = deque(maxlen=max_buffer_frames)
    recent_detections = deque()

    frame_idx = 0
    last_display_frame = None
    last_saved_time = 0
    last_detection_time = 0

    best_conf = -1.0
    best_raw_frame = None
    best_annotated_frame = None
    progress_phase = 20

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.log("Frame read failed. Waiting briefly and trying again...")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                logger.log("Reconnection failed. Retrying...")
                continue
            logger.log("Reconnected to stream.")
            continue

        now_ts = time.time()

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        frame_buffer.append((now_ts, frame.copy()))

        # PERFORMANCE FIX: Only update the dashboard preview when YOLO analyzes a frame
        if frame_idx % VID_STRIDE == 0:
            results = model(frame, conf=min(ACCIDENT_CONF_THRESHOLD, VEHICLE_CONF_THRESHOLD), verbose=False, device="cpu")
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
                            1,
                        )

            annotated = core.overlay_logo_top_right(annotated, logo_rgba, LOGO_MARGIN)
            last_display_frame = annotated.copy()
            logger.preview(last_display_frame) # Update dashboard preview

            if found_accident_this_frame and (now_ts - last_detection_time) >= MIN_DETECTION_GAP_SEC:
                recent_detections.append((now_ts, best_accident_conf_this_frame))
                last_detection_time = now_ts

                logger.log(
                    f"[DETECTION] accident conf={best_accident_conf_this_frame:.3f} "
                    f"| at {format_ts(now_ts)} "
                    f"| detections in window={len(recent_detections)}"
                )

                if best_accident_conf_this_frame > best_conf:
                    best_conf = best_accident_conf_this_frame
                    best_raw_frame = frame.copy()
                    best_annotated_frame = annotated.copy()

                progress_phase = min(55, progress_phase + 3)
                logger.progress(progress_phase, "Monitoring live detections")

            prune_old_detections(recent_detections, now_ts)
            in_cooldown = (now_ts - last_saved_time) < COOLDOWN_SEC

            if not in_cooldown and len(recent_detections) >= REQUIRED_DETECTIONS:
                confirmation_ts = time.time()
                pre_frames = [(ts, fr) for ts, fr in frame_buffer if confirmation_ts - ts <= PRE_EVENT_SEC]

                logger.log("\n" + "=" * 70)
                logger.log("[EVENT CONFIRMED]")
                logger.log(f"Confirmation time: {format_ts(confirmation_ts)}")
                logger.log(f"Rule met: {REQUIRED_DETECTIONS} accident detections within {DETECTION_WINDOW_SEC} seconds")
                logger.log("Triggering detections:")
                for det_num, (det_ts, det_conf) in enumerate(recent_detections, start=1):
                    logger.log(f"  {det_num}. time={format_ts(det_ts)} | conf={det_conf:.3f}")

                save_event_clip(
                    pre_frames=pre_frames,
                    cap_obj=cap,
                    fps_value=fps,
                    out_w=width,
                    out_h=height,
                    save_path=clip_path,
                    post_sec=POST_EVENT_SEC,
                    logo_rgba=logo_rgba,
                )

                if best_raw_frame is None:
                    best_raw_frame = pre_frames[-1][1].copy() if pre_frames else frame.copy()
                if best_annotated_frame is None:
                    best_annotated_frame = last_display_frame.copy() if last_display_frame is not None else frame.copy()

                cv2.imwrite(best_raw_frame_path, best_raw_frame)
                cv2.imwrite(best_annotated_frame_path, best_annotated_frame)

                logger.log(f"Saved raw best frame to: {best_raw_frame_path}")
                logger.log(f"Saved annotated best frame to: {best_annotated_frame_path}")
                logger.log(f"Saved clip: {clip_path}")
                logger.log("=" * 70 + "\n")
                logger.preview(best_annotated_frame)
                logger.progress(60, "Live event clip captured")

                cap.release()
                return {
                    "raw_image_path": best_raw_frame_path,
                    "annotated_image_path": best_annotated_frame_path,
                    "clip_video_path": clip_path,
                }
        # REMOVED redundancy: Avoids unnecessary I/O during non-analyzed frames.

        frame_idx += 1


def analyze_image_accident_verification_dashboard(raw_image_path: str, callbacks: Optional[DashboardCallbacks] = None) -> Dict[str, Any]:
    image = core.PILImage.open(raw_image_path)
    response = generate_content_with_retry_dashboard(
        model=core.MODEL_NAME,
        contents=[core.image_verification_prompt(), image],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
        callbacks=callbacks,
    )
    text = core.clean_response(response.text)
    data = json.loads(text)
    return {
        "confirmed_accident": data.get("confirmed_accident", "UNKNOWN"),
        "note": data.get("note", ""),
    }


def analyze_video_accident_verification_dashboard(video_path: str, callbacks: Optional[DashboardCallbacks] = None) -> Dict[str, Any]:
    video_file = core.upload_video_and_wait(video_path)
    response = generate_content_with_retry_dashboard(
        model=core.MODEL_NAME,
        contents=[video_file, core.video_verification_prompt()],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
        callbacks=callbacks,
    )
    text = core.clean_response(response.text)
    data = json.loads(text)
    return {
        "confirmed_accident": data.get("confirmed_accident", "UNKNOWN"),
        "note": data.get("note", ""),
    }


def analyze_main_incident_dashboard(raw_image_path: str, camera_info: Dict[str, Any], prompt_template: str, source_video_url: str, annotated_image_path: str, clip_video_path: str, callbacks: Optional[DashboardCallbacks] = None) -> Dict[str, Any]:
    prompt = core.build_main_prompt(
        prompt_template=prompt_template,
        camera_info=camera_info,
        source_video_url=source_video_url,
        raw_image_path=raw_image_path,
        annotated_image_path=annotated_image_path,
        clip_video_path=clip_video_path,
    )

    raw_image = core.PILImage.open(raw_image_path)
    response = generate_content_with_retry_dashboard(
        model=core.MODEL_NAME,
        contents=[prompt, raw_image],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
        callbacks=callbacks,
    )
    text = core.clean_response(response.text)
    output = json.loads(text)
    return core.normalize_output(
        model_output=output,
        camera_info=camera_info,
        raw_image_path=raw_image_path,
        annotated_image_path=annotated_image_path,
        clip_video_path=clip_video_path,
        source_video_url=source_video_url,
    )


def run_report_pipeline_dashboard(raw_image_path: str, annotated_image_path: str, clip_video_path: str, source_video_url: str, output_json_path: str, output_pdf_path: str, callbacks: Optional[DashboardCallbacks] = None) -> Dict[str, Any]:
    logger = JobLogger(callbacks)
    logger.log("=" * 80)
    logger.log("RUNNING AI IMAGE + VIDEO ASSESSMENT WITH GEOJSON LOOKUP")
    logger.log("=" * 80)
    logger.progress(62, "Preparing report pipeline")

    core.file_exists_or_raise(core.PROMPT_TXT_PATH, "Prompt file")
    core.file_exists_or_raise(raw_image_path, "Unannotated image")
    core.file_exists_or_raise(annotated_image_path, "Annotated image")
    core.file_exists_or_raise(clip_video_path, "Clip video")
    core.file_exists_or_raise(core.GEOJSON_PATH, "GeoJSON file")
    core.file_exists_or_raise(core.GETCAMERAS_PATH, "GetCameras file")

    logger.log("\n1. Loading prompt...")
    prompt_template = core.load_text(core.PROMPT_TXT_PATH)
    logger.log("   ✓ Prompt loaded")
    logger.progress(65, "Prompt loaded")

    logger.log("\n2. Loading GeoJSON...")
    geojson_data = core.load_json(core.GEOJSON_PATH)
    logger.log("   ✓ GeoJSON loaded")

    logger.log("\n2b. Loading GetCameras.json...")
    getcameras_data = core.load_json(core.GETCAMERAS_PATH)
    logger.log("   ✓ GetCameras.json loaded")
    logger.progress(68, "Camera metadata loaded")

    logger.log("\n3. Looking up camera...")
    camera_id = core.extract_camera_id_from_m3u8_url(source_video_url)
    if camera_id:
        logger.log(f"   Extracted camera id from stream URL: {camera_id}")
        camera_info = core.find_camera_in_geojson_by_id(geojson_data, camera_id)
    else:
        logger.log("   No camera id found in stream URL, falling back to URL lookup...")
        camera_info = core.find_camera_in_geojson_by_url(geojson_data, source_video_url)

    logger.log("   ✓ Camera lookup complete")
    logger.log(f"   Matched camera id: {camera_info.get('id', '')}")
    logger.log(f"   Matched camera name: {camera_info.get('name', '')}")

    extra_camera_info = core.find_camera_in_getcameras_by_id(getcameras_data, camera_info.get("id", ""))
    camera_info["routePrefix"] = extra_camera_info.get("routePrefix", "")
    camera_info["routeNumber"] = extra_camera_info.get("routeNumber", "")
    camera_info["routeSuffix"] = extra_camera_info.get("routeSuffix", "")
    camera_info["milePost"] = extra_camera_info.get("milePost", "")
    camera_info["opStatus"] = extra_camera_info.get("opStatus", "")
    camera_info["cctvIp"] = extra_camera_info.get("cctvIp", "")
    camera_info["cameraCategories"] = extra_camera_info.get("cameraCategories", [])
    camera_info["known_lane_count"] = extra_camera_info.get("known_lane_count", "")
    camera_info["direction_of_travel_interpreted"] = extra_camera_info.get("direction_of_travel_interpreted", "")
    camera_info["confidence_direction"] = extra_camera_info.get("confidence_direction", "")
    camera_info["city"] = core.get_city_from_lat_lon(camera_info.get("lat"), camera_info.get("lon"))
    camera_info["weather_live"] = core.get_weather_condition(camera_info.get("lat"), camera_info.get("lon"))
    camera_info["publicVideoURL"] = source_video_url
    logger.progress(72, "Camera, city, and weather resolved")

    logger.log("\n4. Calling Gemini on unannotated image for accident verification...")
    image_verification = analyze_image_accident_verification_dashboard(raw_image_path, callbacks)
    logger.log("   ✓ Image verification complete")
    logger.progress(78, "Image verification complete")

    logger.log("\n5. Calling Gemini on video clip for accident verification...")
    video_verification = analyze_video_accident_verification_dashboard(clip_video_path, callbacks)
    logger.log("   ✓ Video verification complete")
    logger.progress(84, "Video verification complete")

    logger.log("\n6. Running main incident analysis on unannotated image...")
    model_output = analyze_main_incident_dashboard(
        raw_image_path=raw_image_path,
        camera_info=camera_info,
        prompt_template=prompt_template,
        source_video_url=source_video_url,
        annotated_image_path=annotated_image_path,
        clip_video_path=clip_video_path,
        callbacks=callbacks,
    )
    logger.log("   ✓ Main Gemini analysis complete")
    logger.progress(90, "Main analysis complete")

    verification_summary = core.combine_accident_verification(
        image_result=image_verification,
        video_result=video_verification,
    )
    model_output["incident"]["actual_accident_verified_by_gemini"] = verification_summary["actual_accident_verified_by_gemini"]
    model_output["incident"]["actual_accident_verification_status"] = verification_summary["actual_accident_verification_status"]
    model_output["incident"]["actual_accident_verification_note"] = verification_summary["actual_accident_verification_note"]
    model_output["verification_details"] = {
        "image_verification": image_verification,
        "video_verification": video_verification,
    }

    logger.log("\n7. Saving JSON...")
    core.save_json(model_output, output_json_path)
    logger.log(f"   ✓ JSON saved: {output_json_path}")
    logger.progress(95, "JSON report saved")

    logger.log("\n8. Generating PDF...")
    core.generate_pdf(
        image_path=raw_image_path,
        annotated_image_path=annotated_image_path,
        model_output=model_output,
        output_path=output_pdf_path,
    )
    logger.log(f"   ✓ PDF generated: {output_pdf_path}")
    logger.progress(100, "PDF report generated")

    return model_output


def extract_summary(model_output: Dict[str, Any], detection_outputs: Dict[str, str]) -> Dict[str, Any]:
    incident = model_output.get("incident", {})
    camera = model_output.get("camera", {})
    severity_info = model_output.get("severity_info", {})
    score, category, _notes = core.calculate_severity(severity_info)

    description = incident.get("why") or "No summary description was generated."
    verified = incident.get("actual_accident_verified_by_gemini", False)
    verified_status = incident.get("actual_accident_verification_status", "")

    return {
        "description": description,
        "verified": bool(verified),
        "verified_status": verified_status,
        "camera_name": camera.get("name", ""),
        "city": camera.get("city", ""),
        "latitude": camera.get("lat", ""),
        "longitude": camera.get("lon", ""),
        "severity_score": score,
        "severity_category": category,
        "incident_types": incident.get("incident_types", []),
        "best_frame_path": detection_outputs.get("annotated_image_path", ""),
    }


def run_dashboard_pipeline(input_type: str, source_value: str, recorded_lookup_url: str = "", run_dir: Optional[str] = None, callbacks: Optional[DashboardCallbacks] = None) -> Dict[str, Any]:
    ensure_core_files()
    callbacks = callbacks or DashboardCallbacks()
    logger = JobLogger(callbacks)

    if run_dir is None:
        run_dir = tempfile.mkdtemp(prefix="dashboard_run_", dir=SCRIPT_DIR)
    os.makedirs(run_dir, exist_ok=True)

    if input_type not in {"recorded", "live"}:
        raise ValueError("input_type must be 'recorded' or 'live'")

    logger.log("Starting dashboard pipeline...")
    logger.log(f"Working folder: {run_dir}")

    if input_type == "recorded":
        source_video_url = recorded_lookup_url.strip()
        detection_outputs = detect_from_recorded_video_dashboard(source_value, run_dir, callbacks)
    else:
        source_video_url = source_value.strip()
        detection_outputs = detect_from_live_stream_dashboard(source_value, run_dir, callbacks)

    output_json_path = os.path.join(run_dir, "incident_report.json")
    output_pdf_path = os.path.join(run_dir, "incident_report.pdf")

    model_output = run_report_pipeline_dashboard(
        raw_image_path=detection_outputs["raw_image_path"],
        annotated_image_path=detection_outputs["annotated_image_path"],
        clip_video_path=detection_outputs["clip_video_path"],
        source_video_url=source_video_url,
        output_json_path=output_json_path,
        output_pdf_path=output_pdf_path,
        callbacks=callbacks,
    )

    summary = extract_summary(model_output, detection_outputs)
    logger.log("Dashboard pipeline complete.")

    return {
        "run_dir": run_dir,
        "input_type": input_type,
        "source_value": source_value,
        "source_video_url": source_video_url,
        "summary": summary,
        "outputs": {
            **detection_outputs,
            "json_path": output_json_path,
            "pdf_path": output_pdf_path,
        },
        "model_output": model_output,
    }