"""Board -> robot calibration from the arm touching the board.

Procedure (leader arm or manual jog): put the gripper TIP on the centre of the four corner squares, a1, h1, h8, a8,
(tip just touching the board surface, jaws closed), and record the follower's 5 joint angles at each.
Write them to a JSON file in DEGREES:
    {"a1": [pan, lift, elbow, wrist_flex, wrist_roll], "h1": [...], "h8": [...], "a8": [...]}
Then:   python board_robot_calib.py fit corners.json      -> writes board_robot_calib.json
        python board_robot_calib.py selftest                -> synthetic round-trip check

What it does: forward kinematics (URDF) turns each reading into a tip xyz in the robot frame; the 4 centres define the board
(position, rotation, square size, tilt of the surface). planner.load_calibration() then maps every square by bilinear
interpolation, so a rotated board is handled. It reports quality checks you should read before trusting it.
"""
import json
import os
import sys

import numpy as np

import kin
import planner as P

OUT = os.path.join(os.path.dirname(__file__), "board_robot_calib.json")


def fit(corners_deg, radians=False):
    xyz = {}
    for k in ("a1", "h1", "h8", "a8"):
        q = np.asarray(corners_deg[k], float)
        q = q if radians else np.radians(q)
        xyz[k] = kin.fk(q)[:3, 3]
    a1, h1, h8, a8 = (xyz[k] for k in ("a1", "h1", "h8", "a8"))
    side = [np.linalg.norm(h1 - a1), np.linalg.norm(h8 - h1), np.linalg.norm(a8 - h8), np.linalg.norm(a1 - a8)]
    S = float(np.mean(side) / 7)
    rep = dict(
        square_size_cm=round(S * 100, 2),
        side_lengths_cm=[round(s * 100, 2) for s in side],
        side_mismatch_mm=round((max(side) - min(side)) * 1000, 1),                                # ~0 for a square board
        parallelogram_error_mm=round(float(np.linalg.norm((a1 + h8) / 2 - (h1 + a8) / 2)) * 1000, 1),   # ~0 if the 4 touches are consistent
        surface_height_spread_mm=round((max(p[2] for p in xyz.values()) - min(p[2] for p in xyz.values())) * 1000, 1),
        yaw_deg=round(float(np.degrees(np.arctan2(*(h1 - a1)[:2][::-1]))), 1),
    )
    warn = []
    if rep["parallelogram_error_mm"] > 4: warn.append("corner touches are inconsistent (>4 mm): redo the readings")
    if rep["side_mismatch_mm"] > 8: warn.append("sides differ by >8 mm: board not square or a corner is mis-touched")
    if rep["surface_height_spread_mm"] > 5: warn.append("tip height differs >5 mm between corners: board tilted or tip not touching")
    cal = dict(centers_xy={k: [float(v[0]), float(v[1])] for k, v in xyz.items()}, z_table=float(np.mean([p[2] for p in xyz.values()])),
               square_size=S, report=rep, warnings=warn)
    return cal


def selftest():
    """Place a rotated, shifted board, make the arm 'touch' its corners via IK, fit, and compare every square."""
    rng = np.random.default_rng(0)
    yaw, S, origin = np.radians(7), 0.026, np.array([0.14, -0.09])
    Rm = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    truth = lambda f, r: origin + Rm @ np.array([r * S, f * S])          # files along +y, ranks along +x, then rotated
    q, deg = P.HOME_Q, {}
    for k, (f, r) in dict(a1=(0, 0), h1=(7, 0), h8=(7, 7), a8=(0, 7)).items():
        x, y = truth(f, r)
        q, e, t = kin.ik([x, y, 0.005], q)
        deg[k] = (np.degrees(q) + rng.normal(0, 0.15, 5)).tolist()        # 0.15 deg of reading noise
    cal = fit(deg)
    json.dump(cal, open(OUT, "w"), indent=1)
    P.load_calibration(OUT)
    err = max(np.linalg.norm(P.sq_xy("abcdefgh"[f] + str(r + 1)) - truth(f, r)) for f in range(8) for r in range(8))
    print("report:", cal["report"], "warnings:", cal["warnings"])
    print(f"max square position error vs truth over all 64 squares: {err * 1000:.2f} mm (tolerance is ~{S * 1000 / 2:.0f} mm)")
    os.remove(OUT)


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    elif sys.argv[1] == "fit":
        cal = fit(json.load(open(sys.argv[2])))
        json.dump(cal, open(OUT, "w"), indent=1)
        print(json.dumps(cal["report"], indent=1)); print("WARNINGS:", cal["warnings"] or "none"); print("saved", OUT)
