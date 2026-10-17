"""Capture a labelled dataset by playing a random legal game on the real board.

The script invents a random legal game. For each move it tells you what to do on the board (including removing captured
pieces), you press Enter, and it grabs frames from every camera and stores them with the exact FEN. A one-move change per
shot gives ~40-80 realistic positions without typing a single FEN.

  python capture_dataset.py --cams third=http://localhost:8080/snapshot wrist=http://<other-laptop>:8080/snapshot --plies 60 --shots 3 --seed 1

Between the --shots frames of one position, nudge the lighting / pieces slightly for variety. Run it again with a different
--seed and --session for another game. Output: dataset/raw/<cam>/<session>_<ply>_<shot>.jpg, with a .fen file next to each.
"""
import argparse
import os
import random
import sys

import chess
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from board_map import load_image

ap = argparse.ArgumentParser()
ap.add_argument("--cams", nargs="+", required=True, help="name=url_or_path (snapshot URLs)")
ap.add_argument("--plies", type=int, default=60)
ap.add_argument("--shots", type=int, default=3)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--session", default="g1")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "dataset", "raw"))
a = ap.parse_args()
cams = dict(c.split("=", 1) for c in a.cams)
for n in cams:
    os.makedirs(os.path.join(a.out, n), exist_ok=True)

rng = random.Random(a.seed)
board = chess.Board()


def save(ply):
    for shot in range(a.shots):
        if shot:
            input(f"   shot {shot + 1}/{a.shots}: nudge light/pieces slightly, Enter... ")
        for n, url in cams.items():
            img = load_image(url)
            stem = os.path.join(a.out, n, f"{a.session}_{ply:03d}_{shot}")
            cv2.imwrite(stem + ".jpg", img)
            open(stem + ".fen", "w").write(board.fen())
    print(f"   saved {a.shots} shot(s) x {len(cams)} cam(s), position {board.board_fen()}")


input("Set up the START position, press Enter to capture it... ")
save(0)
for ply in range(1, a.plies + 1):
    if board.is_game_over():
        break
    caps = [m for m in board.legal_moves if board.is_capture(m)]
    mv = rng.choice(caps) if caps and rng.random() < 0.35 else rng.choice(list(board.legal_moves))
    note = ""
    if board.is_capture(mv):
        note = "  (CAPTURE: remove the captured piece from the board)"
    if board.is_castling(mv):
        note = "  (CASTLING: move the rook too)"
    if mv.promotion:
        note = "  (PROMOTION: swap the pawn for the new piece)"
    print(f"[{ply}] make this move: {board.san(mv)}  = {mv.uci()}{note}")
    board.push(mv)
    input("   done? Enter to capture... ")
    save(ply)
print("finished")
