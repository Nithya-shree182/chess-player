"""Serve a webcam as an MJPEG stream over HTTP (stdlib + OpenCV only).

Run:   python stream_server.py --camera 0 --port 8080
View:  http://<this-laptop-ip>:8080/        (browser)
Read:  cv2.VideoCapture("http://<this-laptop-ip>:8080/stream")
"""
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

latest_jpeg = None
lock = threading.Lock()


def grab_loop(cam_index, width, height):
    global latest_jpeg
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # CAP_DSHOW is faster to open on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {cam_index}; try --camera 1")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with lock:
                latest_jpeg = buf.tobytes()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body = b'<html><body style="margin:0;background:#000"><img src="/stream" style="width:100%"></body></html>'
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/snapshot":
            with lock:
                jpg = latest_jpeg
            if jpg is None:
                self.send_error(503, "no frame yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with lock:
                        jpg = latest_jpeg
                    if jpg is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg + b"\r\n")
                    time.sleep(1 / 30)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    threading.Thread(target=grab_loop, args=(a.camera, a.width, a.height), daemon=True).start()
    print(f"Serving on http://0.0.0.0:{a.port}/  (stream: /stream, single frame: /snapshot)")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()
