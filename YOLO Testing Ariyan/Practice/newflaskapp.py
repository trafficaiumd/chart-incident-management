import os
import cv2
from flask import (
    Flask,
    render_template,
    request,
    session,
    Response,
    send_file,
    url_for,
    abort,
)
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired

from New_YOLO_Video import process_image, video_detection, webcam_detection


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "files")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "output.mp4")

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = "roihansori"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")


def get_extension(filename):
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def is_image_file(filename):
    return get_extension(filename) in IMAGE_EXTENSIONS


def is_video_file(filename):
    return get_extension(filename) in VIDEO_EXTENSIONS


def save_uploaded_file(file_storage):
    filename = secure_filename(file_storage.filename)
    if not filename:
        raise ValueError("Invalid filename.")

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file_storage.save(save_path)
    return filename, save_path


@app.route("/", methods=["GET"])
@app.route("/home", methods=["GET"])
def home():
    session.clear()
    return render_template("index.html")


@app.route("/image_upload", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("invalid.html", error="No file was uploaded.")

        file = request.files["file"]

        if not file or file.filename == "":
            return render_template("invalid.html", error="No file selected.")

        if not is_image_file(file.filename):
            return render_template("invalid.html", error="Unsupported image format.")

        try:
            original_filename, saved_path = save_uploaded_file(file)
            result = process_image(saved_path)
            session["last_image_output"] = result["output_path"]

            relative_output_path = os.path.relpath(result["output_path"], BASE_DIR)
            image_path = "/" + relative_output_path.replace("\\", "/")

            return render_template(
                "image.html",
                image_path=image_path,
                detection_result=result,
                original_filename=original_filename,
            )
        except Exception as e:
            return render_template("invalid.html", error=str(e))

    return render_template("image.html")


@app.route("/image_upload_page", methods=["GET"])
def image():
    return render_template("image.html")


@app.route("/<path:filename>")
def display(filename):
    full_path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(full_path):
        abort(404)

    return send_file(full_path)


@app.route("/download_image", methods=["GET"])
def download_image():
    output_path = session.get("last_image_output")

    if not output_path or not os.path.exists(output_path):
        return render_template("invalid.html", error="No processed image available for download.")

    return send_file(output_path, as_attachment=True)


@app.route("/video_detection", methods=["GET", "POST"])
def front():
    form = UploadFileForm()

    if form.validate_on_submit():
        file = form.file.data

        if not is_video_file(file.filename):
            return render_template("invalid.html", error="Unsupported video format.", form=form)

        try:
            filename, saved_path = save_uploaded_file(file)
            session["video_path"] = saved_path
            session["video_filename"] = filename
        except Exception as e:
            return render_template("invalid.html", error=str(e), form=form)

    return render_template("video.html", form=form)


def generate_video_frames(video_path):
    for annotated_frame, frame_result in video_detection(video_path):
        success, buffer = cv2.imencode(".jpg", annotated_frame)
        if not success:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n"
        )


@app.route("/video", methods=["GET"])
def video():
    video_path = session.get("video_path")

    if not video_path or not os.path.exists(video_path):
        return render_template("invalid.html", error="No uploaded video found.")

    return Response(
        generate_video_frames(video_path),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/download_video", methods=["GET"])
def download_video():
    if not os.path.exists(OUTPUT_VIDEO_PATH):
        return render_template("invalid.html", error="No processed video available for download.")

    return send_file(OUTPUT_VIDEO_PATH, as_attachment=True)


def generate_webcam_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            annotated_frame, result = webcam_detection(frame)

            ok, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ok:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n"
            )
    finally:
        cap.release()


@app.route("/webcam_detection", methods=["GET", "POST"])
def webcam():
    session.clear()
    return render_template("webcam.html")


@app.route("/video_feed", methods=["GET"])
def video_feed():
    return Response(
        generate_webcam_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/requests", methods=["GET", "POST"])
def tasks():
    return render_template("webcam.html")


if __name__ == "__main__":
    app.run(debug=True)