# Real-Time Object Detection

Live site: [https://aaradhya0408.github.io/Mini_Project/](https://aaradhya0408.github.io/Mini_Project/)

The page opens a camera, or takes a photo or video, and draws boxes in the browser. That demo uses COCO-SSD so it can run on GitHub Pages without a Python server.

The Python files are the local versions:

- `real_time_object_detection_with_yolov5_using_python.py` loads YOLOv5 and reads the webcam. Press `q` to close it.
- `app.py` is a Flask app that runs YOLOv8 on a webcam stream or an uploaded file.
- `Real_Time_Object_Detection_with_YOLOv5_using_Python.ipynb` is the notebook write-up.

## Run the local YOLOv8 app

```bash
pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The first run downloads `yolov8s.pt`. Uploaded results are written to `static/uploads/`.

`/video` is the YOLOv8 webcam stream. It stays quiet until something requests it, and it shows a message frame when this machine has no camera.

## Run the YOLOv5 script

```bash
python real_time_object_detection_with_yolov5_using_python.py
```
