import atexit
import os

import cv2
from flask import Flask, Response, request, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(BASE, "static", "uploads")
os.makedirs(UPLOADS, exist_ok=True)

app = Flask(__name__, static_folder=os.path.join(BASE, "static"))
model = None
camera = None


def get_model():
    global model
    if model is None:
        model = YOLO("yolov8s.pt")
    return model


def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera


def blank_frame(text):
    frame = cv2.zeros((480, 640, 3), dtype="uint8")
    cv2.putText(frame, text, (24, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
    return frame


def generate_frames():
    cam = get_camera()
    if not cam.isOpened():
        ok, buffer = cv2.imencode(".jpg", blank_frame("No webcam on this machine"))
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        return

    detector = get_model()
    while True:
        success, frame = cam.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        annotated = frame
        for result in detector(frame, stream=True, conf=0.25):
            annotated = result.plot()
        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"


def save_upload(file):
    name = secure_filename(file.filename or "")
    if not name:
        return None
    path = os.path.join(UPLOADS, name)
    file.save(path)
    return path


@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.route("/bg.jpg")
def background():
    return send_from_directory(BASE, "bg.jpg")


@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/upload_image", methods=["POST"])
def upload_image():
    file = request.files.get("file")
    if file is None:
        return "No file uploaded.", 400
    path = save_upload(file)
    if path is None:
        return "Choose an image file.", 400
    output = os.path.join(UPLOADS, f"detected_{os.path.basename(path)}")
    results = get_model()(path, conf=0.25)
    results[0].save(filename=output)
    return f"Saved {os.path.relpath(output, BASE)}"


@app.route("/upload_video", methods=["POST"])
def upload_video():
    file = request.files.get("file")
    if file is None:
        return "No file uploaded.", 400
    path = save_upload(file)
    if path is None:
        return "Choose a video file.", 400

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return "Could not read that video.", 400

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = capture.get(cv2.CAP_PROP_FPS) or 20
    output = os.path.join(UPLOADS, f"detected_{os.path.splitext(os.path.basename(path))[0]}.mp4")
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    detector = get_model()
    frames = 0
    while frames < 300:
        success, frame = capture.read()
        if not success:
            break
        annotated = detector(frame, conf=0.25)[0].plot()
        writer.write(annotated)
        frames += 1
    capture.release()
    writer.release()
    if frames == 0:
        return "The video had no readable frames.", 400
    return f"Saved {os.path.relpath(output, BASE)} ({frames} frames)"


@app.route("/health")
def health():
    return {"ok": True}


def release_camera():
    global camera
    if camera is not None and camera.isOpened():
        camera.release()


atexit.register(release_camera)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
