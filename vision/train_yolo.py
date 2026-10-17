"""Fine-tune YOLO11l on the auto-labelled chess dataset.   pip install ultralytics   (needs an NVIDIA GPU + CUDA PyTorch)

  python train_yolo.py
  python train_yolo.py --imgsz 960 --batch 8
"""
import argparse
import os

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "dataset", "yolo", "chess.yaml"))
ap.add_argument("--model", default="yolo11l.pt")
ap.add_argument("--epochs", type=int, default=80)
ap.add_argument("--imgsz", type=int, default=960)    # pieces are small in a 1280x720 frame; 960 keeps detail
ap.add_argument("--batch", type=int, default=-1)     # -1 = let Ultralytics pick a size that fits the GPU
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--name", default="chess_yolo11l")
a = ap.parse_args()

YOLO(a.model).train(
    data=os.path.abspath(a.data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=20, device=0, workers=a.workers,
    fliplr=0.0, flipud=0.0,   # a mirrored board is a different position for our purposes: keep orientation
    degrees=3, translate=0.05, scale=0.2, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,   # small shifts + lighting variation
    project=os.path.join(os.path.dirname(__file__), "..", "runs"), name=a.name)
