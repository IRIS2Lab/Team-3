# This script runs real-time confidence testing for a specific detection model
# It combines RGB and thermal frames, sends them to a detection server, and logs the detection confidence values.
# This code should run simultaneously with confidenceserver.py
import cv2
import threading
import time
import requests
import numpy as np
from DRV8825 import DRV8825
from datetime import datetime

FUSION_ALPHA = 0 # 0 is RGB Only whereas 1 is thermal only
FLIR_SCALE   = 0.25
OFFSET_X     = 60
OFFSET_Y     = 135

PAN_DELAY        = 0.002
ERROR_SCALE_PAN  = 0.5
MAX_STEPS_PAN    = 30
START_DEADBAND_P =  80

TILT_DELAY       = 0.0001
START_DEADBAND_T = 20
SMOOTH_ALPHA     = 0.1

TARGET_CLASS     = 'mug'
SERVER_URL       = "http://<SERVER_IP>:8000" # Change this to your actual server IP address

pan_motor = DRV8825(dir_pin=13, step_pin=19, enable_pin=12, mode_pins=(16, 17, 20))
pan_motor.SetMicroStep('software', 'step1/32')
tilt_motor = DRV8825(dir_pin=24, step_pin=18, enable_pin=4, mode_pins=(21, 22, 27))
tilt_motor.SetMicroStep('software', 'step1/32')

def move_pan(direction, steps):
    if steps > 0:
        pan_motor.TurnStep(Dir=direction, steps=steps, stepdelay=PAN_DELAY)

def step_tilt(direction):
    tilt_motor.TurnStep(Dir=direction, steps=1, stepdelay=TILT_DELAY)

def send_frame_to_server(frame): # This will send the fused frame to the detection server and returns the JSON detections
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
    try:
        response = requests.post(f"{SERVER_URL}/detect/", files=files, timeout=2)
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None

def send_confidences_to_server(conf_log): # It will send the collected confidence log back to the server
    try:
        response = requests.post(f"{SERVER_URL}/log_confidences/", json={"log": conf_log})
        print("Logged:", response.text)
    except Exception as e:
        print("Failed to send log:", e)

def main():
    cap_rgb = cv2.VideoCapture(2)
    cap_thermal = cv2.VideoCapture(0)
    if not cap_rgb.isOpened() or not cap_thermal.isOpened():
        print("ERROR: Could not open cameras")
        return

    smoothed_err = 0.0
    ret_rgb, frame_rgb = cap_rgb.read()
    H, W = frame_rgb.shape[:2]
    CX, CY = W // 2, H // 2

    collecting = False
    conf_log = []
    collect_start_time = None

    print("Tracking. Press 's' to start 10s logging...")

    try:
        while True:
            ret_rgb, rgb_frame = cap_rgb.read()
            ret_th, thermal_frame = cap_thermal.read()
            if not ret_rgb or not ret_th:
                continue

            thermal_resized = cv2.resize(thermal_frame, None, fx=FLIR_SCALE, fy=FLIR_SCALE)
            canvas = np.zeros_like(rgb_frame)
            x_start = max(0, OFFSET_X)
            y_start = max(0, OFFSET_Y)
            x_end = min(W, x_start + thermal_resized.shape[1])
            y_end = min(H, y_start + thermal_resized.shape[0])
            x_src = max(0, -OFFSET_X)
            y_src = max(0, -OFFSET_Y)
            canvas[y_start:y_end, x_start:x_end] = thermal_resized[y_src:y_src + (y_end - y_start),
                                                                    x_src:x_src + (x_end - x_start)]
            fused = cv2.addWeighted(rgb_frame, 1 - FUSION_ALPHA, canvas, FUSION_ALPHA, 0)

            data = send_frame_to_server(fused)
            detections = data.get("detections", []) if data else []

            best_box = None
            best_area = 0
            conf = None

            for det in detections:
                if det['label'] != TARGET_CLASS:
                    continue
                x1, y1, x2, y2 = det['bbox']
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_box = (x1, y1, x2 - x1, y2 - y1)
                    conf = det['conf']

            if best_box:
                x, y, w, h = best_box
                cx = x + w // 2
                cy = y + h // 2

                err_p = cx - CX
                if abs(err_p) > START_DEADBAND_P:
                    dir_p = 'forward' if err_p > 0 else 'backward'
                    steps_p = min(int(abs(err_p) * ERROR_SCALE_PAN), MAX_STEPS_PAN)
                    threading.Thread(target=move_pan, args=(dir_p, steps_p), daemon=True).start()

                err_t = cy - CY
                smoothed_err = SMOOTH_ALPHA * smoothed_err + (1 - SMOOTH_ALPHA) * err_t
                if abs(smoothed_err) > START_DEADBAND_T:
                    dir_t = 'forward' if smoothed_err > 0 else 'backward'
                    step_tilt(dir_t)

                cv2.rectangle(fused, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(fused, f"Conf: {conf:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if collecting:
                now = time.time()
                conf_val = conf if conf is not None else "null"
                conf_log.append({"timestamp": now, "confidence": conf_val})
                if now - collect_start_time >= 10:
                    collecting = False
                    print("Sending log...")
                    send_confidences_to_server(conf_log)
                    conf_log.clear()

            cv2.imshow("Fused View", fused)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and not collecting:
                print("Started logging for 10s...")
                collect_start_time = time.time()
                collecting = True
                conf_log.clear()

    except KeyboardInterrupt:
        pass

    cap_rgb.release()
    cap_thermal.release()
    cv2.destroyAllWindows()
    pan_motor.Stop()
    tilt_motor.Stop()

if __name__ == "__main__":
    main()

