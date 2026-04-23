from ultralytics import YOLO
import cv2
import numpy as np

# Use slightly better model
model = YOLO("yolov8s.pt")  # better than v8n

def detect_objects(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(img, conf=0.5)  # confidence threshold

    detections = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "box": [x1, y1, x2, y2]
            })

    return detections