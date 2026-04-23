from ultralytics import YOLO
import cv2
import numpy as np
import base64

model = YOLO("models/best.pt")

# ✅ For LIVE (boxes + labels)
def detect_currency_live(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(img, conf=0.5)

    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            label = model.names[cls]

            detections.append({
                "box": [x1, y1, x2, y2],
                "label": label
            })

    return detections


def detect_currency_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(img, conf=0.5)

    labels = []

    for r in results:
        plotted = r.plot()

        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            labels.append(label)

    _, buffer = cv2.imencode(".jpg", plotted)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return img_base64, list(set(labels))