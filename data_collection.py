# This script was used for data collection
# It records synchronized RGB and LWIR video streams while also logging lux using a VEML7700 sensor
# The script saves RGB/LWIR video clips and a CSV file of lux readings for later dataset processing and labeling

import os
import time
import cv2
import board
import busio
import serial
import csv
from DRV8825 import DRV8825
import adafruit_veml7700

i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_veml7700.VEML7700(i2c)
tilt_motor = DRV8825(dir_pin=24, step_pin=18, enable_pin=4, mode_pins=(21, 22, 27))
tilt_motor.SetMicroStep('software', 'step1/32')
tilt_delay = 0.01

# camera_port_finder.py was used to find these device indices
cap_rgb = cv2.VideoCapture(0)
cap_lwir = cv2.VideoCapture(2)

fps = 5

w_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
h_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
w_lwir = int(cap_lwir.get(cv2.CAP_PROP_FRAME_WIDTH))
h_lwir = int(cap_lwir.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
os.makedirs('output', exist_ok=True)
out_rgb = cv2.VideoWriter('output/rgb.avi', fourcc, fps, (w_rgb, h_rgb))
out_lwir = cv2.VideoWriter('output/lwir.avi', fourcc, fps, (w_lwir, h_lwir))

if not out_rgb.isOpened() or not out_lwir.isOpened():
    print("Error: VideoWriter failed to open")
    exit(1)

print("Press 's' to start") # press 's' to start recording
while True:
    r1, f1 = cap_rgb.read()
    r2, f2 = cap_lwir.read()
    if not r1 or not r2:
        continue
    cv2.imshow("RGB", f1)
    cv2.imshow("LWIR", f2)
    if cv2.waitKey(1) == ord('s'):
        break

tilt_motor.TurnStep(Dir='backward', steps=50, stepdelay=tilt_delay) # tilts the turret upward; increase/decrease 'steps' to change the tilt angle


from rangefinder_test import get_latest_reading
SERIAL_PORT = '/dev/ttyAMA0'
BAUD_RATE = 115200
try:
    ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE,
                        parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                        bytesize=serial.EIGHTBITS, timeout=1)
    ser.reset_input_buffer()
except serial.SerialException:
    exit(1)
    
rgb_frames = 0
lwir_frames = 0
lux_readings = []

start = time.time()
while time.time() - start < 3: # allows 3 seconds to warm-up cameras
    r1, f1 = cap_rgb.read()
    r2, f2 = cap_lwir.read()
    if not r1 or not r2:
        continue
    out_rgb.write(f1)
    out_lwir.write(f2)
    rgb_frames += 1
    lwir_frames += 1
    cv2.imshow("RGB", f1)
    cv2.imshow("LWIR", f2)
    if cv2.waitKey(1) == ord('q'):
        break

rec_start = time.time()
while time.time() - rec_start < 8: # records RGB/LWIR frames and logs lux readings per iteration
    r1, f1 = cap_rgb.read()
    r2, f2 = cap_lwir.read()
    if not r1 or not r2:
        continue
    out_rgb.write(f1)
    out_lwir.write(f2)
    lux_readings.append(sensor.lux)
    rgb_frames += 1
    lwir_frames += 1
    cv2.imshow("RGB", f1)
    cv2.imshow("LWIR", f2)
    if cv2.waitKey(1) == ord('q'):
        break

tilt_motor.Stop()
cap_rgb.release()
cap_lwir.release()
out_rgb.release()
out_lwir.release()
cv2.destroyAllWindows()

with open('output/lux.csv', 'w', newline='') as f: # save lux readings to csv
    wtr = csv.writer(f)
    wtr.writerow(['frame', 'lux'])
    for i, l in enumerate(lux_readings):
        wtr.writerow([i, l])

print(f"Total RGB frames: {rgb_frames}")
print(f"Total LWIR frames: {lwir_frames}")
print(f"Approx RGB FPS: {rgb_frames / 11:.2f}")
print(f"Approx LWIR FPS: {lwir_frames / 11:.2f}")
print(f"Lux readings recorded: {len(lux_readings)}") # expected number of lux readings is approximately 40 (FPS * 8 seconds)
