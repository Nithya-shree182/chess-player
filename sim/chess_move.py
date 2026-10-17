"""Validate a chess move with python-chess, expand it into physical piece transfers, and plan/check each transfer.

Usage:  python chess_move.py "<FEN or 'start'>" <uci move, e.g. e1e2>
Physical transfers (what the arm actually does):
  normal move        : src -> dst
  capture            : captured piece -> graveyard slot first, then src -> dst
  en passant         : captured pawn (on the pawn's square, not dst) -> graveyard, then src -> dst
  castling           : king src -> dst, then rook -> its new square
  promotion          : pawn (after capture handling) -> graveyard, spare piece S0 -> dst
"""
import json
import os
import sys

import chess
import numpy as np

import kin
import planner as P


def board_dict(b: chess.Board):
    return {chess.square_name(sq): p.symbol().upper() for sq, p in b.piece_map().items()}


def expand(b: chess.Board, mv: chess.Move):
    """Move -> ordered list of (src, dst, note). Graveyard/spare slots are named G<n>/S<n>."""
    src, dst = chess.square_name(mv.from_square), chess.square_name(mv.to_square)
    ts, g = [], 0
    if b.is_en_passant(mv):
        cap = chess.square_name(chess.square(chess.square_file(mv.to_square), chess.square_rank(mv.from_square)))
        ts.append((cap, f"G{g}", "en passant capture"))
    elif b.is_capture(mv):
        ts.append((dst, f"G{g}", "capture"))
    if b.is_castling(mv):
        r = chess.square_rank(mv.from_square)
        kingside = chess.square_file(mv.to_square) > chess.square_file(mv.from_square)
        rook_src = chess.square_name(chess.square(7 if kingside else 0, r))
        rook_dst = chess.square_name(chess.square(5 if kingside else 3, r))
        ts += [(src, dst, "castling: king"), (rook_src, rook_dst, "castling: rook")]
    elif mv.promotion:
        ts += [(src, f"G{g + 1}", "promotion: remove pawn"), ("S0", dst, "promotion: place spare piece")]
    else:
        ts.append((src, dst, "move"))
    return ts


def check_transfer(bd, src, dst):
    """Plan one transfer on the dict board `bd`, check IK + collisions. Returns (ok, info) and updates bd."""
    wps = P.plan_move(src, dst, bd)
    pts, holding = [], []
    for (_, a, _), (lab, b, grip) in zip(wps, wps[1:]):
        seg, _ = P.segment(a, b)
        pts += list(seg)
        holding += [grip == "closed" and lab != "grasp"] * len(seg)
    pts, holding = np.array(pts), np.array(holding)
    Q, err, tilt = P.solve_path(pts[::3])
    hb = P.GRIP_FRAC * P.PIECE_H[bd[src]]
    hits = P.collisions([(p, h) for p, h in zip(pts, holding)], bd, skip=src, held_below=hb)
    info = dict(piece=bd[src], z_safe_cm=round(float(max(w[1][2] for w in wps)) * 100, 1), seconds=round(len(pts) / 30, 1),
                max_ik_err_mm=round(float(err.max()) * 1000, 2), max_tilt_deg=round(float(tilt.max()), 1),
                collisions=sorted(set(s for _, s in hits)),
                waypoints=[(l, [round(float(v), 4) for v in xyz], g) for l, xyz, g in wps])
    ok = err.max() < 0.002 and tilt.max() < 20 and not hits
    bd[dst] = bd.pop(src)
    return ok, info


def plan(fen, uci):
    b = chess.Board() if fen == "start" else chess.Board(fen)
    mv = chess.Move.from_uci(uci)
    if mv not in b.legal_moves:
        legal = sorted(m.uci() for m in b.legal_moves if m.from_square == mv.from_square)
        return dict(valid=False, reason=f"{uci} is not legal here", legal_moves_from_that_square=legal)
    bd, steps, ok_all = board_dict(b), [], True
    for src, dst, note in expand(b, mv):
        if src == "S0":
            bd["S0"] = "Q"                       # a spare queen waits in slot S0
        ok, info = check_transfer(bd, src, dst)
        ok_all &= ok
        steps.append(dict(step=f"{src}->{dst}", note=note, ok=bool(ok), **info))
    return dict(valid=True, uci=uci, san=b.san(mv), physical_ok=bool(ok_all), steps=steps)


if __name__ == "__main__":
    fen, uci = sys.argv[1], sys.argv[2]
    res = plan(fen, uci)
    out = os.path.join(os.path.dirname(__file__), "out", f"plan_{uci}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    if not res["valid"]:
        print("REJECTED:", res["reason"], "| legal from that square:", res["legal_moves_from_that_square"])
    else:
        print(f"VALID {res['san']} ({uci}) -> physically feasible: {res['physical_ok']}")
        for s in res["steps"]:
            print(f"  {s['step']:8s} {s['note']:28s} piece={s['piece']} z_safe={s['z_safe_cm']}cm t={s['seconds']}s "
                  f"ik_err={s['max_ik_err_mm']}mm tilt={s['max_tilt_deg']} collisions={s['collisions'] or 'none'} ok={s['ok']}")
        print("saved", out)
