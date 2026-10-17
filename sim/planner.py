"""Board geometry, piece model, waypoint planner and trajectory generator (pure numpy, shared with the real robot)."""
import numpy as np

import kin

# ---- configuration (measure these on the real board / pieces) -------------------------------------------------
S = 0.026                 # square size [m] (the arm's pitch-down reach limits this, see sweep.py)
BOARD_NEAR_X = 0.120      # distance from robot base to the near edge of the board along x [m]
BOARD_Y0 = 0.0            # lateral offset of board centre from the robot axis [m]
PIECE_H = dict(P=0.028, N=0.036, B=0.040, R=0.032, Q=0.048, K=0.056)   # heights [m]
PIECE_R = 0.0075          # base radius used for collision checks [m]
GRIP_FRAC = 0.55          # grasp at this fraction of piece height
CLEAR = 0.010             # extra clearance over the tallest piece [m]
TOOL_LEN = 0.060          # length of gripper body above the tip that must stay clear of pieces
TOOL_R = 0.012           # half-width of the jaws + margin
HOME_Q = np.array([0.0, -1.0, 1.0, 0.8, 0.0])


GRAVE_DX = 0.020                       # spacing of graveyard slots G0.. (right of the h-file) and spare slots S0.. (left of the a-file)
Z_TABLE = 0.0                          # board surface height in the robot frame (set by calibration)
_CORNERS = None                        # xy of the centres of a1, h1, h8, a8 in the robot frame (set by calibration)


def default_corners():
    """Ideal board: files along robot y, ranks along robot x, near edge at BOARD_NEAR_X."""
    c = lambda f, r: [BOARD_NEAR_X + (r + 0.5) * S, BOARD_Y0 + (f - 3.5) * S]
    return np.array([c(0, 0), c(7, 0), c(7, 7), c(0, 7)])


def frac_xy(f, r):
    """Robot-frame xy of board position (file f, rank r) in square units, centres at 0..7. Bilinear in the 4 corner centres,
    so a rotated / slightly skewed board is handled; extrapolates just outside the board for graveyard and spare slots."""
    A, B, C, D = _CORNERS if _CORNERS is not None else default_corners()      # a1, h1, h8, a8
    u, v = f / 7.0, r / 7.0
    return (1 - u) * (1 - v) * A + u * (1 - v) * B + u * v * C + (1 - u) * v * D


def sq_xy(sq):
    """'e2' -> robot-frame (x, y) of the square centre. 'G3' = graveyard slot, 'S0' = spare-piece slot."""
    if sq[0] == "G":
        return frac_xy(7 + 1.6, int(sq[1:]) * GRAVE_DX / S)
    if sq[0] == "S":
        return frac_xy(-1.6, int(sq[1:]) * GRAVE_DX / S)
    return frac_xy(ord(sq[0]) - ord("a"), int(sq[1]) - 1)


def load_calibration(path):
    """Apply a calibration written by board_robot_calib.py."""
    global _CORNERS, Z_TABLE, S
    import json
    c = json.load(open(path))
    _CORNERS = np.array([c["centers_xy"][k] for k in ("a1", "h1", "h8", "a8")])
    Z_TABLE, S = c["z_table"], c["square_size"]


START = {}
for i, p in enumerate("RNBQKBNR"):
    f = "abcdefgh"[i]
    START[f + "1"], START[f + "8"] = p, p
    START[f + "2"], START[f + "7"] = "P", "P"


def solve_path(points, q0=HOME_Q):
    """IK along a list of xyz points, seeding each solve with the previous one. Returns (Q, errs, tilts)."""
    Q, errs, tilts, q = [], [], [], q0
    for p in points:
        q, e, t = kin.ik([p[0], p[1], p[2] + Z_TABLE], q)
        Q.append(q); errs.append(e); tilts.append(t)
    return np.array(Q), np.array(errs), np.array(tilts)


def reach_map(grasp_z=0.015, tol=0.002, max_tilt=15.0):
    """Check every square at the lowest (grasp/place) height. Returns dict sq -> (ok, err_mm, tilt_deg)."""
    out, q = {}, HOME_Q
    for f in "abcdefgh":
        for r in "12345678":
            x, y = sq_xy(f + r)
            q, e, t = kin.ik([x, y, grasp_z], q)
            out[f + r] = (e < tol and t < max_tilt, e * 1000, t)
    return out


def min_jerk(n):
    s = np.linspace(0, 1, n)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def segment(a, b, speed=0.10, hz=30):
    """Straight Cartesian segment a->b with min-jerk timing at ~`speed` m/s average. Returns (pts, duration)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    dur = max(0.4, np.linalg.norm(b - a) / speed)
    n = max(2, int(dur * hz))
    return a + np.outer(min_jerk(n), b - a), dur


def corridor_height(src, dst, board):
    """Tallest piece (other than the mover) near the straight line src->dst, so travel height is only as high as needed."""
    a, b = sq_xy(src), sq_xy(dst)
    tall = 0.0
    for sq, p in board.items():
        if sq == src:
            continue
        c = sq_xy(sq)
        t = np.clip(np.dot(c - a, b - a) / (np.dot(b - a, b - a) + 1e-12), 0, 1)
        if np.linalg.norm(c - (a + t * (b - a))) < 2 * PIECE_R + 0.002:   # only the carried piece must clear (lateral overlap)
            tall = max(tall, PIECE_H[p])
    return tall


def plan_move(src, dst, board, capture=False):
    """Waypoint list for moving the piece on src to dst. Returns list of (label, xyz, gripper_state)."""
    piece = board[src]
    h = PIECE_H[piece]
    z_grasp = GRIP_FRAC * h
    z_place = 0.004 + GRIP_FRAC * h + 0.003          # set down just above the surface
    held_below = GRIP_FRAC * h                       # carried piece hangs this far below the tip
    z_safe = corridor_height(src, dst, board) + held_below + CLEAR
    (xs, ys), (xd, yd) = sq_xy(src), sq_xy(dst)
    return [
        ("hover src", (xs, ys, z_safe), "open"),
        ("descend", (xs, ys, z_grasp), "open"),
        ("grasp", (xs, ys, z_grasp), "closed"),
        ("lift", (xs, ys, z_safe), "closed"),
        ("travel", (xd, yd, z_safe), "closed"),
        ("descend", (xd, yd, z_place), "closed"),
        ("release", (xd, yd, z_place), "open"),
        ("retreat", (xd, yd, z_safe), "open"),
    ]


def collisions(tip_traj, board, skip, held_below=0.0):
    """Check the swept tool volume (and the held piece) against every piece except `skip`.

    The tool is modelled as a vertical capsule from the tip upwards (TOOL_LEN) with radius TOOL_R. The held piece is a
    cylinder hanging below the tip. Returns list of (step, square) hits.
    """
    hits = []
    cyl = [(sq, sq_xy(sq), PIECE_H[p]) for sq, p in board.items() if sq != skip]
    for i, (pt, hold) in enumerate(tip_traj):
        for sq, xy, h in cyl:
            d = np.linalg.norm(pt[:2] - xy)
            # tool body vs piece
            if d < TOOL_R + PIECE_R and pt[2] < h and pt[2] + TOOL_LEN > 0:
                hits.append((i, sq)); continue
            # held piece vs piece
            if hold and d < 2 * PIECE_R and pt[2] - held_below < h:
                hits.append((i, sq))
    return hits
