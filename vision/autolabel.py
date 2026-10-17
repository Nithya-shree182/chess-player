"""Turn captured (image, FEN) pairs into a YOLO dataset using each camera's clicked-corner calibration.

For every occupied square a box is built from the projected square geometry: base at the square centre, width ~ the
square's pixel width, height = HS[piece] x that width (HS = how tall the piece looks in square-widths: measure it once).
Always run with --preview first and check the boxes by eye; adjust HS and re-run.

  python autolabel.py --cam third --calib board_calib.json --preview 6
  python autolabel.py --cam third --calib board_calib.json --cam wrist --calib board_calib_wrist.json     (build the dataset)
"""
import argparse
import glob
import os
import random
import shutil
import sys

import chess
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from board_map import BoardMap

CLASSES = [f"{c}_{n}" for c in ("white", "black") for n in ("pawn", "knight", "bishop", "rook", "queen", "king")]
HS = dict(p=0.80, n=1.10, b=1.20, r=0.90, q=1.35, k=1.40)   # apparent piece height in square-widths: MEASURE on your pieces
W_FRAC = 0.80                                                 # box width as a fraction of the square's pixel width
ROOT = os.path.join(os.path.dirname(__file__), "..", "dataset")


def boxes_for(bm: BoardMap, fen: str, shape):
    h_img, w_img = shape[:2]
    out = []
    for sq, p in chess.Board(fen).piece_map().items():
        name = chess.square_name(sq)
        corners = np.array([bm.square_to_pixel(name, (u, v)) for u in (0, 1) for v in (0, 1)])
        cx, cy = bm.square_to_pixel(name)
        sq_w, sq_h = np.ptp(corners[:, 0]), np.ptp(corners[:, 1])
        base_y = cy + 0.2 * sq_h
        w = W_FRAC * sq_w
        ph = HS[p.symbol().lower()] * sq_w
        x1, x2, y1, y2 = cx - w / 2, cx + w / 2, base_y - ph, base_y
        x1, x2, y1, y2 = max(0, x1), min(w_img, x2), max(0, y1), min(h_img, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        cls = ("white_" if p.color else "black_") + chess.piece_name(p.piece_type)
        out.append((CLASSES.index(cls), x1, y1, x2, y2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", action="append", required=True)
    ap.add_argument("--calib", action="append", required=True)
    ap.add_argument("--preview", type=int, default=0, help="write N overlay images to dataset/preview and stop")
    ap.add_argument("--val", type=float, default=0.2, help="fraction of positions (from the END of each game) used for validation")
    a = ap.parse_args()
    here = os.path.dirname(__file__)
    items = []
    for cam, cal in zip(a.cam, a.calib):
        bm = BoardMap(os.path.join(here, cal))
        for jpg in sorted(glob.glob(os.path.join(ROOT, "raw", cam, "*.jpg"))):
            items.append((cam, bm, jpg, open(jpg[:-4] + ".fen").read().strip()))
    if a.preview:
        os.makedirs(os.path.join(ROOT, "preview"), exist_ok=True)
        for cam, bm, jpg, fen in random.Random(0).sample(items, min(a.preview, len(items))):
            img = cv2.imread(jpg)
            for c, x1, y1, x2, y2 in boxes_for(bm, fen, img.shape):
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, CLASSES[c], (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imwrite(os.path.join(ROOT, "preview", f"{cam}_{os.path.basename(jpg)}"), img)
        print("wrote previews to", os.path.join(ROOT, "preview"))
        sys.exit()
    # split by (session, ply): the last `val` fraction of each session's plies goes to validation, never mixing a position
    plies = {}
    for cam, bm, jpg, fen in items:
        sess, ply = os.path.basename(jpg).split("_")[:2]
        plies.setdefault(sess, set()).add(int(ply))
    cut = {s: sorted(p)[int(len(p) * (1 - a.val))] for s, p in plies.items()}
    for sub in ("train", "val"):
        for kind in ("images", "labels"):
            shutil.rmtree(os.path.join(ROOT, "yolo", kind, sub), ignore_errors=True)
            os.makedirs(os.path.join(ROOT, "yolo", kind, sub))
    n = dict(train=0, val=0)
    for cam, bm, jpg, fen in items:
        sess, ply = os.path.basename(jpg).split("_")[:2]
        sub = "val" if int(ply) >= cut[sess] else "train"
        img = cv2.imread(jpg)
        stem = f"{cam}_{os.path.basename(jpg)[:-4]}"
        shutil.copy(jpg, os.path.join(ROOT, "yolo", "images", sub, stem + ".jpg"))
        h, w = img.shape[:2]
        with open(os.path.join(ROOT, "yolo", "labels", sub, stem + ".txt"), "w") as fh:
            for c, x1, y1, x2, y2 in boxes_for(bm, fen, img.shape):
                fh.write(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
        n[sub] += 1
    yaml = ("path: " + os.path.abspath(os.path.join(ROOT, "yolo")).replace("\\", "/") + "\ntrain: images/train\nval: images/val\nnames:\n"
            + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    open(os.path.join(ROOT, "yolo", "chess.yaml"), "w").write(yaml)
    print("images:", n, "| wrote", os.path.join(ROOT, "yolo", "chess.yaml"))
