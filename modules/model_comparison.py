from ultralytics import YOLO
import cv2
import numpy as np
import base64

# Load models once
model_v8 = YOLO("yolov8s.pt")
model_v11 = YOLO("yolo11n.pt")  # if available in your ultralytics

# YOLOv5 using ultralytics fallback
model_v5 = YOLO("yolov5s.pt")


def run_model(model, img):
    results = model(img)

    for r in results:
        plotted = r.plot()  # bounding boxes drawn
        return plotted


def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def compare_models(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    out_v5 = run_model(model_v5, img)
    out_v8 = run_model(model_v8, img)
    out_v11 = run_model(model_v11, img)

    return {
        "v5": encode_image(out_v5),
        "v8": encode_image(out_v8),
        "v11": encode_image(out_v11),
    }