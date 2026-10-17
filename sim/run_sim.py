"""Simulate moving one piece (default e2 -> e4) with the planner, check reach / collisions, and render a GIF + PNGs.

Run:  python run_sim.py [src dst]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import kin
import planner as P

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
src, dst = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("e2", "e4")
board = dict(P.START)
hb = P.GRIP_FRAC * P.PIECE_H[board[src]]    # how far the carried piece hangs below the tip

# 1. reachability of all 64 squares at grasp height ---------------------------------------------------------------
rm = P.reach_map()
bad = [s for s, (ok, e, t) in rm.items() if not ok]
print(f"Board S={P.S*100:.1f} cm, near edge x={P.BOARD_NEAR_X*100:.1f} cm -> unreachable squares: {bad or 'none'}")
fig, ax = plt.subplots(figsize=(5, 5))
for s, (ok, e, t) in rm.items():
    x, y = P.sq_xy(s)
    ax.add_patch(plt.Rectangle((y - P.S / 2, x - P.S / 2), P.S, P.S, color="#9be59b" if ok else "#ff8c8c", ec="k", lw=.4))
    ax.text(y, x, s, ha="center", va="center", fontsize=6)
ax.plot(0, 0, "ko"); ax.text(0, 0.008, "robot base", ha="center", fontsize=7)
ax.set_aspect("equal"); ax.set_xlabel("y [m]"); ax.set_ylabel("x (away from base) [m]"); ax.set_title("Reachability (green = ok)")
ax.autoscale(); fig.savefig(os.path.join(OUT, "reach_map.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

# 2. plan + trajectory --------------------------------------------------------------------------------------------
wps = P.plan_move(src, dst, board)
pts, holding, labels = [], [], []
for (_, a, _), (lab, b, grip) in zip(wps, wps[1:]):
    seg, _ = P.segment(a, b)
    pts += list(seg)
    labels += [lab] * len(seg)
    holding += [grip == "closed" and lab != "grasp"] * len(seg)    # piece is in the gripper after 'grasp' until 'release'
pts, holding = np.array(pts), np.array(holding)
first_hold = int(np.argmax(holding))

Q, errs, tilts = P.solve_path(pts)
print(f"Plan: {len(wps)} waypoints, {len(pts)} control steps (~{len(pts)/30:.1f} s at 30 Hz)")
print(f"IK: max position error {errs.max()*1000:.2f} mm, max tool tilt {tilts.max():.1f} deg")
dq = np.degrees(np.abs(np.diff(Q, axis=0))).max(axis=0)
print("Max joint step per tick [deg]:", dict(zip(kin.JOINTS, dq.round(2))), "(>10 would hint at an IK flip)")

# 3. collision check: planned (hover pattern) vs naive straight line at grasp height ------------------------------
hits = P.collisions([(p, h) for p, h in zip(pts, holding)], board, skip=src, held_below=hb)
print(f"Planned path collisions: {sorted(set(s for _, s in hits)) or 'none'}")
xs, ys = P.sq_xy(src); xd, yd = P.sq_xy(dst)
zg = P.GRIP_FRAC * P.PIECE_H[board[src]]
naive, _ = P.segment((xs, ys, zg), (xd, yd, zg))
nh = P.collisions([(p, True) for p in naive], board, skip=src, held_below=hb)
print(f"Naive straight-line drag at grasp height: {sorted(set(s for _, s in nh)) or 'none'}")

# 4. render -------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(11, 5))
a3 = fig.add_subplot(121, projection="3d"); a2 = fig.add_subplot(122)


def draw_board(ax):
    for f in range(8):
        for r in range(8):
            x, y = P.sq_xy("abcdefgh"[f] + str(r + 1))
            ax.add_patch(plt.Rectangle((y - P.S / 2, x - P.S / 2), P.S, P.S, color="#d9b27c" if (f + r) % 2 else "#7a4b1e", alpha=.8))


def frame(i):
    a3.cla(); a2.cla()
    chain = np.array([[0, 0, 0]] + [t[:3, 3] for t in kin.fk_all(Q[i])])
    tip = pts[i]
    # moving piece: on src before the grasp, in the gripper while held, on dst afterwards
    if holding[i]:
        mx, my, mz = tip[0], tip[1], tip[2] - hb
    elif i < first_hold:
        (mx, my), mz = (xs, ys), 0.0
    else:
        (mx, my), mz = (xd, yd), 0.0
    for sq, p in board.items():
        if sq == src:
            continue
        x, y = P.sq_xy(sq)
        a3.plot([y, y], [x, x], [0, P.PIECE_H[p]], color="#eeeeee" if sq[1] in "12" else "#222222", lw=4, solid_capstyle="round")
    a3.plot([my, my], [mx, mx], [mz, mz + P.PIECE_H[board[src]]], color="red", lw=5, solid_capstyle="round")
    a3.plot(chain[:, 1], chain[:, 0], chain[:, 2], "-o", color="#ff7f0e", lw=3, ms=4)
    a3.plot(pts[:, 1], pts[:, 0], pts[:, 2], color="g", lw=1, alpha=.6)
    a3.set_xlim(-0.2, 0.2); a3.set_ylim(0, 0.4); a3.set_zlim(0, 0.3)
    a3.set_xlabel("y"); a3.set_ylabel("x"); a3.set_zlabel("z"); a3.view_init(25, -60)
    a3.set_title(f"{src}->{dst}: {labels[min(i, len(labels) - 1)]}")
    draw_board(a2)
    for sq, p in board.items():
        if sq == src:
            continue
        x, y = P.sq_xy(sq)
        a2.add_patch(plt.Circle((y, x), P.PIECE_R, color="w" if sq[1] in "12" else "k", ec="gray"))
    a2.add_patch(plt.Circle((my, mx), P.PIECE_R, color="red"))
    a2.plot(pts[:, 1], pts[:, 0], "g", lw=1)
    a2.plot(chain[:, 1], chain[:, 0], "-", color="#ff7f0e", lw=3)
    a2.set_xlim(-0.2, 0.2); a2.set_ylim(0, 0.4); a2.set_aspect("equal"); a2.set_title("top view (green = path)")


step = max(1, len(pts) // 90)
idx = list(range(0, len(pts), step))
ani = FuncAnimation(fig, lambda k: frame(idx[k]), frames=len(idx))
ani.save(os.path.join(OUT, f"{src}{dst}.gif"), writer=PillowWriter(fps=15))
for name, i in [("start", 0), ("carry", int(np.where(holding)[0][len(np.where(holding)[0]) // 2])), ("end", len(pts) - 1)]:
    frame(i); fig.savefig(os.path.join(OUT, f"{src}{dst}_{name}.png"), dpi=110)
print("Wrote", OUT)
