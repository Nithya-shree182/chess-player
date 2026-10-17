"""Sweep board size / position and report whether every square is reachable at both grasp and hover height."""
import numpy as np, kin, planner as P
zs = [0.012, 0.075]
rows = []
for S in [0.024, 0.026, 0.028, 0.030]:
    for near in [0.11, 0.12, 0.13]:
        P.S, P.BOARD_NEAR_X = S, near
        q, bad = P.HOME_Q, []
        for f in "abcdefgh":
            for r in "18":                      # corners/edges are the worst cases: ranks 1 and 8
                for z in zs:
                    x, y = P.sq_xy(f + r)
                    q, e, t = kin.ik([x, y, z], q)
                    if e > 0.002 or t > 20: bad.append(f"{f}{r}@{int(z*1000)}")
        print(f"S={S*100:.1f}cm near={near*100:.0f}cm  bad={len(bad)} {bad[:6]}", flush=True)
