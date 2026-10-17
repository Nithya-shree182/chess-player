# Autonomous Chess Player: SO-101 + Wrist Camera + YOLOv8 + Stockfish

Hand-off document for the RTX laptop (the machine that runs everything). Hackathon budget: ~13 hours.
Items marked **[VERIFY]** come from general knowledge, not from reading the installed library. Check them against the installed versions before relying on them.

---

## 1. Goal

A robot arm plays a full chess game against a human, autonomously:
1. Look at the board with the wrist camera.
2. Work out the current position and the human's move.
3. Let Stockfish choose the reply.
4. Physically pick up and place the piece (handling captures, castling, en passant, promotion) without knocking other pieces over.
5. Check the result visually and retry if wrong.

An LLM agent (Claude) sits on top as orchestrator and commentator. It does **not** choose chess moves.

## 2. Hardware

- LeRobot SO-101 follower arm (5 DOF + gripper, Feetech servos, position control). Reach is roughly 35 cm **[VERIFY with a tape measure]**.
- SO-101 leader arm (used for teaching calibration poses, not for training).
- Wrist ("eye-in-hand") RGB camera. **No depth.**
- RTX laptop: runs vision inference, IK, game logic and the robot loop.
- Board and pieces: small board (~28-32 cm, squares ~3.5-4 cm). Pieces with wide base and graspable neck. Four ArUco markers around the board corners (outside the playing area).
- Optional: one fixed overhead webcam. Decide this first (see section 10).

## 3. Architecture: who computes what

| Component | Runs on | What it does |
|---|---|---|
| ArUco + homography (OpenCV) | laptop CPU | pixel to board mm, every capture |
| YOLOv8 detector (Ultralytics) | laptop GPU | boxes + class for the 12 piece types |
| Square assignment | laptop | box base point through homography to one of 64 squares |
| Game state (`python-chess`) | laptop | tracks position, validates the human's move, legal moves |
| Stockfish | laptop | picks the robot's move |
| **Motion planner (our own code)** | laptop | **this is the thing that "calculates the movement plan"** (section 6) |
| Servo control | laptop to USB to Feetech bus | receives joint angle targets |
| Claude agent | API call from laptop | orchestration, error recovery, commentary |

**The movement plan is computed by our own deterministic Python code:** a waypoint generator (hover/descend/grasp/lift/travel/place), an inverse-kinematics solver that turns each waypoint into joint angles, and a trajectory interpolator that streams intermediate joint targets at ~30 Hz. There is no neural network, no MoveIt and no simulation in the loop. Stockfish only supplies a chess move such as `e2e4`; it knows nothing about the arm.

What the robot actually receives: a dict of **joint angle targets**, one per joint, sent each tick:

```python
robot.send_action({"shoulder_pan.pos": ..., "shoulder_lift.pos": ..., "elbow_flex.pos": ...,
                   "wrist_flex.pos": ..., "wrist_roll.pos": ..., "gripper.pos": ...})
```
Units are degrees or normalized values depending on config. **[VERIFY key names and units in the installed LeRobot version.]**

## 4. Why no training for motion, and no sim / MoveIt / ROS2

- Squares are at fixed, known positions once the board is calibrated. IK + waypoints reaches them more precisely than a learned policy.
- The vertical hover/descend/lift pattern avoids other pieces by construction, so collision planning is unnecessary.
- MoveIt/ROS2 setup would cost hours. Simulation adds a sim-to-real gap.
- Imitation learning (ACT, SmolVLA) with the leader arm is a stretch goal only. Do not start it unless everything else works.
- The leader arm is used for **teaching calibration poses** and tuning the gripper.

The only thing we train is **YOLOv8** for piece detection.

## 5. Vision pipeline (per capture)

1. Arm moves to a fixed **look pose** (high enough that the whole board is in frame; check this early).
2. Capture frame. Detect ArUco markers (`cv2.aruco`, e.g. `DICT_4X4_50`). Compute homography pixel to board mm. Redo it every capture to absorb small pose differences.
3. Run YOLOv8 on the raw frame (not a warped one, which smears tall pieces). Output: boxes, one of 12 classes, confidence.
4. For each box take the **bottom-center point** (where the base meets the board). Push it through the homography to get board mm, then to a square index 0-63.
5. Build the 64-square state.
6. Validate with `python-chess`: the new state must equal the previous position plus exactly one legal move. If not, re-look, and if it still fails hand it to the agent / ask the human to confirm.
7. After each robot move, repeat to confirm the board matches the expected position.

YOLO is treated as a sensor. `python-chess` is the source of truth. We start from the known starting position, so the model only has to answer "what changed?". White vs black can be double-checked with a brightness test on the square.

No depth is needed: the board is a known plane (from the ArUco markers) and the piece height comes from a lookup table by class.

## 6. Motion planning (our code)

### 6.1 Coordinates
- `square_to_board_mm(sq)`: center of the cell, `(col*S + S/2, row*S + S/2)`.
- `board_to_robot(x, y)`: rigid transform (rotation + translation) found in calibration (section 8).
- Z values: `z_table` (board surface in robot frame) plus offsets.

### 6.2 Waypoint recipe for one piece transfer `from_sq to to_sq`
1. `hover(from)` at `z_safe` (above the tallest piece, with margin).
2. Descend to `z_grasp(from_piece_class)` (about 50-60% of piece height).
3. Close gripper to the width for that class. Check the servo position (section 6.5).
4. Lift to `z_safe`.
5. Travel to `hover(to)`.
6. Descend to `z_place` (slightly above the surface).
7. Open gripper. Retreat to `z_safe`.
8. Return to the look pose.

Special cases:
- **Capture:** transfer the captured piece to a graveyard slot first.
- **Castling:** two transfers (king, then rook).
- **En passant:** capture on the pawn's actual square, not the destination.
- **Promotion:** remove the pawn, place a spare piece (keep spare queens at a known spot).

### 6.3 Inverse kinematics
- LeRobot ships a URDF-based kinematics helper (built on `placo`) and the SO-ARM100 repo ships the URDF **[VERIFY module path and API in the installed version]**. Input: current joint angles and a target pose. Output: joint angles.
- The SO-101 has 5 DOF, so a full 6-DOF pose cannot be imposed. Ask for "gripper pointing roughly straight down" with a low orientation weight so the solver compromises instead of failing.
- Seed each solve with the previous solution to avoid elbow flips.
- Reject any target where IK error is above a few mm. Run a **reachability sweep over all 64 squares at both heights during setup** to find squares that are out of reach.

### 6.4 Trajectory execution
- Never send the final angles in one step. Interpolate over 0.5-1.5 s with a smooth (min-jerk / cosine) profile, streamed at ~30 Hz.
- Vertical moves: interpolate in joint space. Long horizontal travel at `z_safe`: interpolate in Cartesian space and solve IK at each step so the path stays level.
- Limit speed and acceleration; the servos are weak and fast moves topple pieces.
- Close the gripper gradually and add a short dwell before lifting.

### 6.5 Grasp verification
After closing, read the gripper position and compare it to the expected width for that class:
- fully closed: missed, retry
- near expected: holding
- far wider than expected: misgrip, release and retry

Grasp XY is the square center. Optionally correct by the offset between the YOLO box base point and the square center (only if it proves reliable).

## 7. Code layout (suggested)

```
d:\chess-player\
  PLAN.md                  this file
  config.py                board size, square size, marker size/ids, piece heights, gripper widths, z levels
  calib\
    camera_intrinsics.py   checkerboard calibration of the wrist cam (optional but recommended)
    board_to_robot.py      teach corners with leader, solve rigid transform
    reachability_sweep.py  IK on all 64 squares
  vision\
    aruco_homography.py
    capture_dataset.py     look pose + capture + auto-label from a known FEN
    train_yolo.py / chess.yaml
    detect_board.py        frame -> 64-square state
  motion\
    ik.py
    trajectory.py          interpolation + send_action loop
    pick_place.py          waypoint recipes, captures, castling, promotion
  game\
    state.py               python-chess wrapper, move inference from board diff
    engine.py              Stockfish wrapper
  agent\
    tools.py               look(), get_board(), engine_move(), execute_move(...)
    orchestrator.py        Claude loop, recovery, commentary
  main.py
```

## 8. Calibration (manual, one-time, then re-check before the demo)

1. **Robot calibration:** run the standard LeRobot calibration for follower and leader.
2. **Camera intrinsics (recommended):** photograph a printed checkerboard ~15-20 times, run `cv2.calibrateCamera`. Cuts homography error from lens distortion.
3. **Board to robot transform:** with the leader arm, move the follower so the gripper tip touches known points (a1 corner, h1 corner, one more). Record the joint angles, run forward kinematics, solve the rigid transform. Alternative: teach hover poses over the four corner squares and interpolate; this tolerates URDF error.
4. **Piece table:** measure the height of each of the 6 piece types, and find the gripper closing value that holds each one firmly without crushing it.
5. **Look pose:** choose it, confirm all four markers and the full board are visible.
6. **Sanity test:** command the tip to hover at 2 mm above a1, h1, a8 and h8 and check by eye.

## 9. YOLO training

### Data
- ~20-30 distinct positions (opening, midgame, endgame). Include several positions with kings and queens so every class has at least 100-150 instances.
- 8-12 photos per position, varying lighting and look-pose jitter. Total ~250-400 images.
- Shoot from the **same wrist camera at the same look pose** as the demo. Do not rely on online datasets as the main source.

### Auto-labeling (avoids hours of hand-labeling)
1. Set up a known position and record its FEN.
2. Capture from the look pose and detect the ArUco markers.
3. Project each occupied square back into the image with the inverse homography.
4. Draw a box around that point. Box size per class is calibrated once (taller for king/queen).
5. Write YOLO-format labels with the class from the FEN.
6. Spot-check 30-50 images by overlaying the boxes.

### Training
```
pip install ultralytics
yolo detect train data=chess.yaml model=yolov8s.pt epochs=60 imgsz=640 batch=16
```
- Split train/val by **position**, not by photo.
- Disable flip augmentations if they could confuse orientation.
- Expect roughly 15-40 minutes on the RTX GPU.
- Review the confusion matrix. Add images for the weak classes (usually bishop/pawn, king/queen) and retrain once.

## 10. Decisions to make first (they change the design)

1. **Fixed overhead camera or wrist-only?** The wrist cam has to rise to a look pose to see the whole board. A cheap fixed cam makes vision much simpler. Reference projects used 1-3 cameras.
2. **Board and piece size/design.** The 35 cm reach limit means far corners of a standard board are unreachable. Pieces need wide bases and necks the gripper can hold.
3. **Gripper pads.** Grip tape or rubber on the fingertips. Grasping is the highest-risk component.

## 11. Manual work list (honest estimate)

| Task | Who | Time |
|---|---|---|
| Print/prepare board, ArUco markers, pieces, gripper pads | hands-on | 1 hr |
| Robot + leader calibration, check teleop, camera works | hands-on | 1 hr |
| Wrist-cam intrinsics (photograph checkerboard) | hands-on | 20 min |
| Board-to-robot transform with leader arm | hands-on | 30 min |
| Measure piece heights and gripper widths | hands-on | 30 min |
| Reachability check and fix board position | hands-on | 30 min |
| Arrange positions + shoot dataset (largest manual cost) | hands-on | 2-3 hr |
| Spot-check auto-labels, review training, retrain | light | 1 hr |
| Rehearse full games, tune speeds and grasp | hands-on | 1-2 hr |
| Record a backup demo video | hands-on | 20 min |

Everything else (code) is written and run on the laptop.

## 12. Schedule (13 h)

| Hours | Work |
|---|---|
| 0-2 | Hardware setup, LeRobot install, robot/camera check, print board and pieces, choose look pose |
| 2-5 | ArUco/homography, board-to-robot transform, IK, reachability on all 64 squares, hover/pick/place on one square then all |
| 5-8 | Piece table, captures/castling/graveyard, Stockfish + `python-chess` loop running through the robot with human move typed in |
| in parallel from ~3 | Vision person: capture script, auto-label, dataset session, YOLO training |
| 8-10 | Connect vision move detection, test lighting |
| 10-12 | Claude agent layer, retry logic, failure recovery, full-game rehearsal |
| 12-13 | Demo script, backup video, buffer |

Split the team: one person on robot motion (5-8), one on vision and data. Merge at hour ~8.

## 13. Fallbacks

- Vision fails: the human enters moves on screen; the robot still plays.
- Wrist cam can't see the board: add a fixed overhead webcam.
- Grasp unreliable: fix the pieces or gripper pads before touching software.
- Reach fails on far squares: move or shrink the board.
- Always keep a recorded clean run as insurance.

## 14. Reference projects (found by web search; none is a ready SO-101 chess repo)

- Hack Club "Robby Fischer" (SO-ARM101, 3 cameras, Stockfish, imitation learning): https://archive.hackclub.com/archive/xf2Ix/mhtml/html. No public repo confirmed.
- Iowa State senior design (webcam, homography, Stockfish, UR arm): https://seniord.cs.iastate.edu/2025-May-17/
- CSUB poster (OpenCV warp + YOLO, 4-DOF arm): https://cs.csub.edu/seniorexpo/2025/ECE/Automated%20Chess%20Player%20Using%20a%204-DOF%20Robotic%20Arm/poster.pdf
- YOLOv8 + Stockfish + QArm paper (87% first-try, 97% by second try): https://www.mdpi.com/2079-9292/8/10/1154
- LeRobot SO-101 docs: https://huggingface.co/docs/lerobot/en/so101
- GitHub topic so-101: https://github.com/topics/so-101

## 15. Open questions for the laptop session to resolve early

1. Which LeRobot version is installed, and what are the exact `send_action` keys, units and kinematics module API?
2. Is the URDF available locally, and does the IK solver reach a pitch-down orientation on this arm?
3. At what look-pose height does the wrist cam see the whole board and all 4 markers?
4. What is the real reach, and which squares are reachable at the chosen board position?

---

## 16. Motion planning: design and simulation results

Code in `sim/` (kinematic only, numpy + scipy, uses the official SO-101 URDF in `sim/assets/`):
`kin.py` (FK/IK from URDF), `planner.py` (board geometry, waypoints, trajectory, collision check),
`run_sim.py <src> <dst>` (reach map, plan, collision check, GIF/PNGs in `sim/out/`), `sweep.py` / `ceiling.py` (reach studies).

Planner pipeline for one piece: waypoints (hover, descend, grasp, lift, travel, descend, release, retreat) ->
straight Cartesian segments with min-jerk timing at ~0.1 m/s -> IK per control tick (seeded from the previous
solution) -> joint targets at 30 Hz. Travel height is chosen per move from the tallest piece near the path plus the
carried piece's hang below the tip plus 1 cm.

Simulation findings (sim, not hardware):
- e2->e4 and d1->h5 plan with 0 mm IK error, no collisions, max joint step ~3 deg/tick, ~3-4 s per move.
- A naive straight-line drag at grasp height hits d2, e1, e2, f2 for d1->h5. The hover pattern avoids all of them.
- **Reach limit:** with a pitch-down grasp the arm covers roughly x = 12-33 cm from the base. Every square reachable at
  both grasp and hover height only for squares of about 2.4-2.6 cm with the near edge ~12 cm from the base (2.8 cm loses
  the far corners, 3.0 cm loses a lot). Hover height must stay under ~7.5-9 cm near the base. A standard 4-5 cm square
  board will NOT fit with vertical grasps. Options: small board and pieces, tilted grasps (needs sim + hardware testing),
  or restrict play to a sub-region.
- Not modelled: servo backlash, gripper finger geometry, real piece shapes, deflection under load.

## 17. Current state (wired so far)
- `sim/engine.py`: python-chess + Stockfish 19 (binary in `engines/stockfish_dl/`, or set STOCKFISH_PATH) -> move -> `chess_move.plan`.
- `sim/play_game.py`: interactive loop, human moves typed now (replace with vision later).
- `vision/board_map.py`: manual 4-corner calibration (click a1, h1, h8, a8 in the camera image) replaces ArUco while board
  and camera stay fixed; `BoardMap.detections_to_squares()` maps YOLO boxes (base point) to squares.
- Still TODO: real robot driver (send joint targets), board->robot calibration with the leader arm, YOLO training, vision loop.
