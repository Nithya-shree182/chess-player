"""Forward / inverse kinematics for the SO-101 straight from the official URDF (numpy + scipy only).

Chain: base_link -> shoulder_pan -> shoulder_lift -> elbow_flex -> wrist_flex -> wrist_roll -> gripper_frame_link.
Angles are radians here. Robot frame: origin at base_link, z up.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares

URDF = os.path.join(os.path.dirname(__file__), "assets", "so101_new_calib.urdf")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TIP_LINK = "gripper_frame_link"


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def _tf(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = _rpy(*rpy)
    T[:3, 3] = xyz
    return T


def _load():
    root = ET.parse(URDF).getroot()
    by_child = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in o.get("xyz").split()]
        rpy = [float(v) for v in o.get("rpy").split()]
        ax = j.find("axis")
        axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3)
        lim = j.find("limit")
        lo, hi = (float(lim.get("lower")), float(lim.get("upper"))) if lim is not None else (0, 0)
        by_child[j.find("child").get("link")] = dict(
            name=j.get("name"), parent=j.find("parent").get("link"),
            T=_tf(xyz, rpy), axis=axis, lo=lo, hi=hi)
    chain, link = [], TIP_LINK
    while link in by_child:
        chain.append(by_child[link])
        link = by_child[link]["parent"]
    return chain[::-1]


CHAIN = _load()
LIMITS = np.array([[j["lo"], j["hi"]] for j in CHAIN if j["name"] in JOINTS])


def _rot_axis(axis, q):
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K
    T = np.eye(4)
    T[:3, :3] = R
    return T


def fk_all(q):
    """Return list of 4x4 frames after each chain element (base -> tip)."""
    T, out, i = np.eye(4), [], 0
    for j in CHAIN:
        T = T @ j["T"]
        if j["name"] in JOINTS:
            T = T @ _rot_axis(j["axis"], q[i])
            i += 1
        out.append(T.copy())
    return out


def fk(q):
    return fk_all(q)[-1]


def ik(target_xyz, q0, pitch_down=True, w_orient=0.15, lock_roll=True):
    """Solve for 5 joint angles so the gripper tip reaches target_xyz (metres).

    The SO-101 has 5 DOF, so full orientation can't be imposed. We ask for the gripper's approach axis
    (tip frame z) to point down with a low weight. Returns (q, position_error_m, tilt_deg).
    """
    target = np.asarray(target_xyz, float)

    def resid(q):
        T = fk(np.r_[q[:4], 0.0 if lock_roll else q[4]] if lock_roll else q)
        r = list(T[:3, 3] - target)
        if pitch_down:
            r.append(w_orient * (np.dot(T[:3, 2], [0, 0, -1]) - 1.0))
        return r

    n = 4 if lock_roll else 5
    res = least_squares(lambda x: resid(np.r_[x, 0.0] if lock_roll else x), np.asarray(q0, float)[:n],
                        bounds=(LIMITS[:n, 0], LIMITS[:n, 1]), xtol=1e-10, ftol=1e-10)
    q = np.r_[res.x, 0.0] if lock_roll else res.x
    T = fk(q)
    err = np.linalg.norm(T[:3, 3] - target)
    tilt = np.degrees(np.arccos(np.clip(np.dot(T[:3, 2], [0, 0, -1]), -1, 1)))
    return q, err, tilt
