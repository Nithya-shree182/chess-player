import numpy as np, kin, planner as P
P.S, P.BOARD_NEAR_X = 0.030, 0.13
def ceiling(sq, tilt_max, zs=np.arange(0.02, 0.14, 0.01)):
    x, y = P.sq_xy(sq); q = P.HOME_Q; best = 0
    for z in zs:
        q, e, t = kin.ik([x, y, z], q)
        if e < 0.002 and t < tilt_max: best = z
    return best
for sq in ["a1","d1","h1","a4","d4","a8","d8","h8","d6"]:
    print(sq, "ceiling z @tilt<20: %.2f  @tilt<35: %.2f" % (ceiling(sq,20), ceiling(sq,35)), flush=True)
