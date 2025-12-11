import os
import cv2
from flask import Flask, render_template, Response, request
from ultralytics import YOLO
import atexit

# ✅ Show which HTML file Flask is loading
print("✅ Using index.html from:", os.path.abspath('templates/index.html'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['TEMPLATES_AUTO_RELOAD'] = True  # 🔁 Auto reload templates during development
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ✅ Load YOLOv8 model (use yolov8m.pt for higher accuracy)
model = YOLO("yolov8s.pt")
print("✅ YOLOv8 model loaded successfully with classes:", model.names)

# ✅ Open webcam
camera = cv2.VideoCapture(0)

def generate_frames():
    """Stream live webcam video with YOLOv8 detections."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 🔁 Flip frame for natural webcam view
            frame = cv2.flip(frame, 1)

            # ✅ Run YOLOv8 detection (lower conf for smaller object detection)
            results = model(frame, stream=True, conf=0.25)

            for r in results:
                annotated_frame = r.plot()

            # Encode and stream frame
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    """Render the main webpage."""
    return render_template('index.html')


@app.route('/video')
def video():
    """Return the video feed to the webpage."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload and YOLOv8 detection."""
    file = request.files['file']
    if not file:
        return "❌ No file uploaded.", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Run YOLOv8 inference
    results = model(filepath, conf=0.25)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"detected_{file.filename}")
    results[0].save(filename=output_path)

    print(f"✅ Processed image saved at: {output_path}")
    return f"✅ Image '{file.filename}' processed! Check 'static/uploads' for result."


@atexit.register
def release_camera():
    """Release camera when app stops."""
    if camera.isOpened():
        camera.release()
    print("📷 Camera released successfully.")


if __name__ == "__main__":
    # Disable caching in debug mode
    @app.after_request
    def add_header(response):
        response.headers['Cache-Control'] = 'no-store'
        return response

    app.run(host='0.0.0.0', port=5000, debug=True)
