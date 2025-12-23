# This server rungs alongside confidence_test.py
# confidence_test.py will send the frames to this server and this server will run the YOLO inference
# How to run this server (example): uvicorn confidenceserver:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI, File, UploadFile, Request
import cv2
from ultralytics import YOLO
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import csv
import os
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

TARGET_CLASS   = 'mug'
CONF_THRESHOLD = 0.3
CSV_FILE       = "confidence_logs.csv"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading YOLO model...")
model = YOLO("models/yolo11n.pt") # Choose the path of the model you want to test
print("Model loaded.")
print(model.names)

@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    image_data = await file.read() # Receives a JPG frame from confidence_test.py
    image = Image.open(BytesIO(image_data)).convert("RGB")
    frame = np.array(image)

    results = model(frame, conf=CONF_THRESHOLD)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        if label == TARGET_CLASS:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": label,
                "conf": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })

    return {"detections": detections}

@app.get("/target/")
def get_target_label():
    return {"target": "mug"}

@app.post("/log_confidences/")
async def log_confidences(req: Request):
    data = await req.json()
    log = data.get("log", [])

    if not log:
        return {"status": "no data"}

    clean_log = [entry for entry in log if entry.get("confidence") not in [None, "", "null"]]

    filename = f"confidence_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "confidence"])
        for entry in clean_log:
            writer.writerow([entry["timestamp"], entry["confidence"]])

    confidences = [entry["confidence"] for entry in clean_log]
    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    return {
        "status": f"logged to {filename}",
        "average_confidence": avg_conf
    }

