"""Stockfish wiring: game state (python-chess) -> engine move -> physical plan (chess_move.plan).

Set STOCKFISH_PATH to the stockfish binary (or put it on PATH). On the Ubuntu robot laptop: `sudo apt install stockfish`
gives /usr/games/stockfish.

Usage:  python engine.py "<FEN or 'start'>" [human_uci]
  - if human_uci is given it is validated and applied first (the human's move seen by vision)
  - Stockfish then replies, and the reply is expanded into transfers and checked by the planner
"""
import os
import shutil
import sys

import chess
import chess.engine

import chess_move


def find_stockfish():
    local = os.path.join(os.path.dirname(__file__), "..", "engines", "stockfish_dl", "stockfish", "stockfish-windows-x86-64-universal.exe")
    path = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish") or shutil.which("stockfish.exe") or local
    if not path or not os.path.exists(path):
        raise SystemExit("Stockfish not found. Install it and set STOCKFISH_PATH=<path to binary>.")
    return path


def engine_move(board: chess.Board, think_s=1.0):
    with chess.engine.SimpleEngine.popen_uci(find_stockfish()) as eng:
        return eng.play(board, chess.engine.Limit(time=think_s)).move


def play_turn(fen, human_uci=None, think_s=1.0):
    board = chess.Board() if fen == "start" else chess.Board(fen)
    if human_uci:
        mv = chess.Move.from_uci(human_uci)
        if mv not in board.legal_moves:
            return dict(error=f"human move {human_uci} is illegal here")
        board.push(mv)
    if board.is_game_over():
        return dict(game_over=board.result(), fen=board.fen())
    mv = engine_move(board, think_s)
    plan = chess_move.plan(board.fen(), mv.uci())          # validate again + expand into physical transfers + IK/collision check
    board.push(mv)
    return dict(engine_move=mv.uci(), plan=plan, fen_after=board.fen(), game_over=board.result() if board.is_game_over() else None)


if __name__ == "__main__":
    res = play_turn(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print({k: v for k, v in res.items() if k != "plan"})
    if "plan" in res:
        for s in res["plan"].get("steps", []):
            print(" ", s["step"], s["note"], "ok" if s["ok"] else "FAIL", s["collisions"] or "")
