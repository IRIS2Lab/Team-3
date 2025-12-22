# This code lists all connected video devices and helps identify which /dev/video index corresponds to the RGB and LWIR cameras

import os
import subprocess
import cv2

def list_video_devices():
    base_path = "/sys/class/video4linux"
    print("Connected Video Devices:")
    for dev in os.listdir(base_path):
        device_path = os.path.join(base_path, dev)
        name_file = os.path.join(device_path, "name")
        if os.path.exists(name_file):
            with open(name_file, 'r') as f:
                name = f.read().strip()
            print(f"  - /dev/{dev} {name}")

def test_camera(index):  # Tries to open a camera index and grab one frame to verify it works
    print(f"\nTesting /dev/video{index} ...")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Could not open /dev/video{index}")
        return
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f" /dev/video{index} opened but did not return a frame")
    else:
        h, w = frame.shape[:2]
        print(f"Success: Got frame of size {w}x{h}")
        cv2.imshow(f"video{index}", frame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    list_video_devices()
    for i in range(6): # Test a few common video indices (increase range if needed)
        test_camera(i)


