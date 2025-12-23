# This code performs real-time tracking using a CSRT tracker
# A user will select an ROI in the camera feed, and the turret will continuously move to keep the tracked object centered in the frame
import cv2
import threading
import RPi.GPIO as GPIO
from DRV8825 import DRV8825

pan_motor = DRV8825(dir_pin=13, step_pin=19, enable_pin=12, mode_pins=(16,17,20))
pan_motor.SetMicroStep('software', 'step1/32')

tilt_motor = DRV8825(dir_pin=24, step_pin=18, enable_pin=4, mode_pins=(21,22,27))
tilt_motor.SetMicroStep('software', 'step1/32')

PAN_DELAY        = 0.002
ERROR_SCALE_PAN  = 0.5
MAX_STEPS_PAN    = 30
START_DEADBAND_P = 80
STOP_DEADBAND_P  = 20

TILT_DELAY       = 0.0001
START_DEADBAND_T = 60
STOP_DEADBAND_T  = 30
SMOOTH_ALPHA     = 0.1

CAMERA_INDEX     = 2

def make_csrt(): # Creates the CSRT tracker
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()
    return cv2.TrackerCSRT_create()

def move_pan(direction, steps):
    if steps > 0:
        pan_motor.TurnStep(Dir=direction, steps=steps, stepdelay=PAN_DELAY)

def step_tilt(direction):
    tilt_motor.TurnStep(Dir=direction, steps=1, stepdelay=TILT_DELAY)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera #{CAMERA_INDEX}")
ret, tmp = cap.read()
if not ret:
    raise RuntimeError("Cannot read initial frame")
H, W = tmp.shape[:2]
CX = W // 2
CY = H // 2

tracker = None
smoothed_err = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    key = cv2.waitKey(1) & 0xFF
    if tracker is None:
        cv2.putText(frame, "Press 's' to select ROI", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Pan-Tilt Tracking", frame)
        if key == ord('s'):
            roi = cv2.selectROI("Pan-Tilt Tracking", frame, False, False) # User can manually select the object to track
            x,y,w,h = map(int, roi)
            if w and h:
                tracker = make_csrt()
                tracker.init(frame, (x,y,w,h))
                smoothed_err = 0.0
        elif key == ord('q'):
            break
    else:
        ok, bbox = tracker.update(frame)
        if not ok:
            tracker = None
        else:
            x,y,w,h = map(int, bbox)
            cx = x + w//2
            cy = y + h//2

            err_t = cy - CY
            smoothed_err = SMOOTH_ALPHA*smoothed_err + (1-SMOOTH_ALPHA)*err_t

            if abs(cx - CX) > START_DEADBAND_P: # Horizontal pixel error controls the pan motion
                dir_p = 'forward' if (cx - CX)>0 else 'backward'
                steps_p = min(int(abs(cx - CX)*ERROR_SCALE_PAN), MAX_STEPS_PAN)
                threading.Thread(target=move_pan, args=(dir_p, steps_p), daemon=True).start()

            if abs(smoothed_err) > START_DEADBAND_T: # Vertical pixel error controls the tilt motion
                dir_t = 'forward' if smoothed_err>0 else 'backward'
                step_tilt(dir_t)

            cv2.imshow("Pan-Tilt Tracking", frame)
            if key == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
pan_motor.Stop()
tilt_motor.Stop()
GPIO.cleanup()


