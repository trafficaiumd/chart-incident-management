import os
import cv2
from collections import deque
from datetime import datetime
from ultralytics import YOLO


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "YOLO-Weights", "yolov8m-5000-300.pt")
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMAGE_SIZE = 640
DEFAULT_OUTPUT_VIDEO = os.path.join(BASE_DIR, "output.mp4")


class AccidentDetector:
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        conf_threshold=DEFAULT_CONFIDENCE,
        img_size=DEFAULT_IMAGE_SIZE,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.img_size = img_size

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = YOLO(self.model_path)
        self.class_names = self.model.names if hasattr(self.model, "names") else {0: "Accident"}

    def _parse_result(self, result):
        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = self.class_names.get(cls, str(cls))

            detections.append(
                {
                    "class_id": cls,
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )

        return detections

    def _annotate_frame(self, frame, detections, verified=False):
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            label = f'{det["class_name"]} {conf:.2f}'

            color = (0, 204, 255) if not verified else (0, 0, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_bg_x2 = x1 + text_size[0] + 8
            text_bg_y1 = max(y1 - text_size[1] - 10, 0)
            text_bg_y2 = max(y1, text_size[1] + 10)

            cv2.rectangle(
                annotated,
                (x1, text_bg_y1),
                (text_bg_x2, text_bg_y2),
                color,
                -1,
            )

            cv2.putText(
                annotated,
                label,
                (x1 + 4, max(y1 - 5, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        status_text = "VERIFIED ACCIDENT" if verified else "Monitoring"
        status_color = (0, 0, 255) if verified else (255, 255, 0)

        cv2.putText(
            annotated,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            3,
            cv2.LINE_AA,
        )

        return annotated

    def predict_frame(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=self.img_size,
            verbose=False,
        )

        result = results[0]
        detections = self._parse_result(result)

        frame_has_detection = len(detections) > 0
        max_conf = max((d["confidence"] for d in detections), default=0.0)

        return {
            "detections": detections,
            "frame_has_detection": frame_has_detection,
            "max_confidence": round(max_conf, 4),
            "raw_result": result,
        }

    def process_image(self, file_path, save_dir=None):
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(BASE_DIR, "runs", "detect", f"predict_{timestamp}")

        os.makedirs(save_dir, exist_ok=True)

        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not read image: {file_path}")

        prediction = self.predict_frame(image)
        annotated = self._annotate_frame(
            image,
            prediction["detections"],
            verified=prediction["frame_has_detection"],
        )

        output_filename = os.path.basename(file_path)
        output_path = os.path.join(save_dir, output_filename)
        cv2.imwrite(output_path, annotated)

        return {
            "input_path": file_path,
            "output_path": output_path,
            "detections": prediction["detections"],
            "accident_detected": prediction["frame_has_detection"],
            "verified": prediction["frame_has_detection"],
            "max_confidence": prediction["max_confidence"],
            "source_type": "image",
        }

    def process_webcam_frame(self, frame):
        prediction = self.predict_frame(frame)
        annotated = self._annotate_frame(
            frame,
            prediction["detections"],
            verified=prediction["frame_has_detection"],
        )

        return annotated, {
            "detections": prediction["detections"],
            "accident_detected": prediction["frame_has_detection"],
            "verified": prediction["frame_has_detection"],
            "max_confidence": prediction["max_confidence"],
            "source_type": "webcam",
        }

    def video_stream_generator(
        self,
        video_path,
        output_path=DEFAULT_OUTPUT_VIDEO,
        verification_window=5,
        min_verified_hits=3,
    ):
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 10.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        hit_history = deque(maxlen=verification_window)
        frame_index = 0

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_index += 1
                prediction = self.predict_frame(frame)

                frame_hit = 1 if prediction["frame_has_detection"] else 0
                hit_history.append(frame_hit)

                verified = sum(hit_history) >= min_verified_hits

                annotated = self._annotate_frame(
                    frame,
                    prediction["detections"],
                    verified=verified,
                )

                out.write(annotated)

                frame_result = {
                    "frame_number": frame_index,
                    "detections": prediction["detections"],
                    "accident_detected": prediction["frame_has_detection"],
                    "verified": verified,
                    "max_confidence": prediction["max_confidence"],
                    "source_type": "video",
                }

                yield annotated, frame_result

        finally:
            cap.release()
            out.release()


# Load the model only once
detector = AccidentDetector()


def process_image(file_path):
    return detector.process_image(file_path)


def webcam_detection(frame):
    return detector.process_webcam_frame(frame)


def video_detection(path_x):
    yield from detector.video_stream_generator(path_x)