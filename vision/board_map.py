"""Manual board calibration + pixel <-> square mapping (replaces ArUco while the board and camera are fixed).

Calibrate:  python board_map.py calibrate [snapshot/stream URL | image.jpg] [out.json]   (one json per camera)
  Click the 4 outer corners of the 8x8 playing area in THIS order:  a1, h1, h8, a8
  (a1 = white's bottom-left corner as seen from white's side; clicking in this order fixes the board orientation).
  Press s to save, r to redo, q to quit.  Writes board_calib.json next to this file.

Use in code:
  from board_map import BoardMap
  bm = BoardMap(); bm.pixel_to_square(x, y); bm.detections_to_squares([(cls, x1, y1, x2, y2), ...])
"""
import json
import os
import sys
import urllib.request

import cv2
import numpy as np

CALIB = os.path.join(os.path.dirname(__file__), "board_calib.json")
ORDER = ["a1", "h1", "h8", "a8"]


class BoardMap:
    def __init__(self, path=CALIB, corners=None):
        if corners is None:
            corners = json.load(open(path))["corners_px"]
        src = np.array(corners, np.float32)                       # pixels of a1, h1, h8, a8
        dst = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], np.float32)   # board units: x = file 0..8, y = rank 0..8
        self.H = cv2.getPerspectiveTransform(src, dst)
        self.Hinv = np.linalg.inv(self.H)

    def pixel_to_board(self, x, y):
        v = self.H @ np.array([x, y, 1.0])
        return v[:2] / v[2]

    def pixel_to_square(self, x, y):
        f, r = self.pixel_to_board(x, y)
        if not (0 <= f < 8 and 0 <= r < 8):
            return None
        return "abcdefgh"[int(f)] + str(int(r) + 1)

    def square_to_pixel(self, sq, frac=(0.5, 0.5)):
        f, r = ord(sq[0]) - 97 + frac[0], int(sq[1]) - 1 + frac[1]
        v = self.Hinv @ np.array([f, r, 1.0])
        return v[:2] / v[2]

    def detections_to_squares(self, dets, base_frac=0.12):
        """dets: [(label, x1, y1, x2, y2)] pixel boxes. A piece's square is where its BASE is, so use the bottom-centre of
        the box (a little above the lower edge), not the box centre: tall pieces lean toward the camera."""
        out = {}
        for label, x1, y1, x2, y2 in dets:
            sq = self.pixel_to_square((x1 + x2) / 2, y2 - base_frac * (y2 - y1))
            if sq:
                out[sq] = label
        return out

    def overlay(self, img):
        img = img.copy()
        for i in range(9):
            for a, b in [((i, 0), (i, 8)), ((0, i), (8, i))]:
                p = [tuple(int(v) for v in self.Hinv @ np.array([*q, 1.0]) / (self.Hinv @ np.array([*q, 1.0]))[2]) for q in (a, b)]
                cv2.line(img, p[0][:2], p[1][:2], (0, 255, 0), 1)
        for f in range(8):
            for r in range(8):
                sq = "abcdefgh"[f] + str(r + 1)
                x, y = self.square_to_pixel(sq)
                cv2.putText(img, sq, (int(x) - 10, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        return img


def load_image(src):
    if src.startswith("http") and src.rstrip("/").endswith("stream"):      # MJPEG stream without a /snapshot endpoint
        cap = cv2.VideoCapture(src)
        for _ in range(5):                                                # skip a few frames so the buffer is fresh
            ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"no frame from {src}")
        return frame
    if src.startswith("http"):
        data = np.frombuffer(urllib.request.urlopen(src, timeout=5).read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.imread(src)


def calibrate(src, out=CALIB):
    img = load_image(src)
    pts = []

    def redraw():
        vis = img.copy()
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 5, (0, 0, 255), -1)
            cv2.putText(vis, ORDER[i], (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(pts) == 4:
            vis = BoardMap(corners=pts).overlay(vis)
        else:
            cv2.putText(vis, f"click corner {ORDER[len(pts)]}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("calibrate", vis)

    def on_click(e, x, y, *_):
        if e == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y)); redraw()

    cv2.namedWindow("calibrate"); cv2.setMouseCallback("calibrate", on_click); redraw()
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == ord("r"):
            pts.clear(); redraw()
        elif k == ord("s") and len(pts) == 4:
            json.dump({"corners_px": pts, "order": ORDER, "image_size": img.shape[1::-1]}, open(out, "w"), indent=1)
            print("saved", out); break
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        calibrate(sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8080/snapshot",
                  sys.argv[3] if len(sys.argv) > 3 else CALIB)
