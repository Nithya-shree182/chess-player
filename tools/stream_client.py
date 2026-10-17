"""Read the MJPEG stream on the robot laptop and show it.

Run: python stream_client.py http://<webcam-laptop-ip>:8080/stream
Press q to quit. In your own code, use the same cv2.VideoCapture(url) and .read().
"""
import sys

import cv2

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080/stream"
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    raise SystemExit(f"Cannot open {url}")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("remote webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
