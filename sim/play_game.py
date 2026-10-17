"""Interactive game loop: human move in (typed now, from vision later) -> Stockfish reply -> physical plan.

Usage:  python play_game.py [--robot white|black] [--think 1.0] [--fen "<FEN>"]
Type moves in UCI (e2e4) or SAN (Nf3). 'quit' to stop. The robot's plan per move is printed and saved to out/last_plan.json.
"""
import argparse
import json
import os

import chess

import chess_move
import engine

ap = argparse.ArgumentParser()
ap.add_argument("--robot", default="black", choices=["white", "black"])
ap.add_argument("--think", type=float, default=1.0)
ap.add_argument("--fen", default=None)
a = ap.parse_args()
board = chess.Board(a.fen) if a.fen else chess.Board()
robot = chess.WHITE if a.robot == "white" else chess.BLACK


def robot_turn():
    mv = engine.engine_move(board, a.think)
    plan = chess_move.plan(board.fen(), mv.uci())
    os.makedirs("out", exist_ok=True)
    json.dump(plan, open(os.path.join("out", "last_plan.json"), "w"), indent=1)
    print(f"ROBOT plays {board.san(mv)} ({mv.uci()}); physical plan ok={plan['physical_ok']}")
    for s in plan["steps"]:
        print(f"   {s['step']:8s} {s['note']:28s} {'ok' if s['ok'] else 'FAIL'}  collisions={s['collisions'] or 'none'}")
    board.push(mv)                       # real robot: execute the plan, re-look with the camera, then push


while not board.is_game_over():
    if board.turn == robot:
        robot_turn()
        continue
    txt = input(f"{'white' if board.turn else 'black'} (human) move> ").strip()
    if txt in ("quit", "q"):
        break
    try:
        mv = board.parse_uci(txt) if len(txt) in (4, 5) and txt[0] in "abcdefgh" and txt[1].isdigit() else board.parse_san(txt)
    except ValueError:
        print("  illegal or unreadable move, legal:", " ".join(board.san(m) for m in board.legal_moves)); continue
    board.push(mv)
print("game over:", board.result() if board.is_game_over() else "stopped", "| FEN:", board.fen())
