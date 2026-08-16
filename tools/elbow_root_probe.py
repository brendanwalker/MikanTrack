import json, math, pathlib, sys
import numpy as np

# Per-model-frame view of the elbow root choice: the two bone-circle candidates,
# how far each sits from the elbow ray, and which one a dump took.
REC = pathlib.Path(sys.argv[1])
DUMP = pathlib.Path(sys.argv[2])
SIDE = sys.argv[3] if len(sys.argv) > 3 else "l"
FIRST = int(sys.argv[4]) if len(sys.argv) > 4 else 0
LAST = int(sys.argv[5]) if len(sys.argv) > 5 else 10 ** 9

lines = REC.read_text(encoding="utf-8").splitlines()
hdr = json.loads(lines[0])
cams = hdr["appConfig"]["cameras"]
camIndex = next(i for i, c in enumerate(cams) if c.get("bodyPose", {}).get("enabled"))
cam = cams[camIndex]
M = np.array(cam["extrinsics"]["markerFromCamera"]).reshape(4, 4).T
R, camPos = M[:3, :3], M[:3, 3]
K = np.array(cam["intrinsics"]["undistortedCameraMatrix"]).reshape(3, 3).T
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
body = hdr["appConfig"]["body"]
FOREARM = body["forearmLengthMeters"]
UPPER = (body["shoulderWidthMeters"] * body["upperArmPerShoulderWidth"]
         if body["deriveUpperArmFromShoulderWidth"] else body["upperArmLengthMeters"])
ELBOW = 13 if SIDE == "l" else 14

recs = [r for r in (json.loads(l) for l in lines) if r.get("type") == "frame"]
dump = {f["frame"]: f for f in json.load(open(DUMP))["frames"]}


def quat_rot(q, v):
    x, y, z, w = q
    t = 2 * np.cross([x, y, z], v)
    return v + w * t + np.cross([x, y, z], t)


def ray(px, py):
    d = R @ np.array([(px - cx) / fx, (py - cy) / fy, 1.0])
    return d / np.linalg.norm(d)


def circle_points(S, W):
    s2w = W - S
    span = np.linalg.norm(s2w)
    axis = s2w / span
    if span >= UPPER + FOREARM or span <= abs(UPPER - FOREARM):
        return None
    a = (span * span + UPPER * UPPER - FOREARM * FOREARM) / (2 * span)
    r2 = UPPER * UPPER - a * a
    if r2 <= 1e-8:
        return None
    center, r = S + axis * a, math.sqrt(r2)
    ref = np.array([0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
    pxv = np.cross(ref, axis)
    pxv /= np.linalg.norm(pxv)
    pyv = np.cross(axis, pxv)
    ang = np.linspace(0, 2 * math.pi, 1441)[:-1]
    return center + (np.cos(ang)[:, None] * pxv + np.sin(ang)[:, None] * pyv) * r


seen = set()
for i in sorted(dump):
    if i < FIRST or i > LAST or i >= len(recs):
        continue
    rec = recs[i]
    src = [c for c in rec["fresh"] if c["cam"] == camIndex]
    b = src[0].get("body") if src else None
    if not b or not b.get("valid") or b["modelFrameIndex"] in seen:
        continue
    seen.add(b["modelFrameIndex"])
    fr = dump[i]
    s = [x for x in fr["replayedBody"] if x["side"].lower().startswith(SIDE)][0]
    if not s.get("hasForearmPose") or not s.get("hasShoulder"):
        continue
    out = [o for o in fr["replayedOut"] if o["side"] == (0 if SIDE == "l" else 1)]
    if not out:
        continue
    o = out[0]
    S = np.array(s["shoulderPositionWorld"])
    E = np.array(s["elbowPositionWorld"])
    W = np.array(o["palmPosW"]) + quat_rot(np.array(o["palmQuatW"]),
                                           np.array([-o["skeleton"]["baseInPalm"][2][0], 0, 0]))
    pts = circle_points(S, W)
    vis = b["visibility"][ELBOW]
    if pts is None:
        print(f"{i:4d} vis{vis:.2f} straight-arm")
        continue
    ip = b["imagePoints"][ELBOW]
    d = ray(ip[0], ip[1])
    rel = pts - camPos
    dd = np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)
    n = len(dd)
    mins = sorted([k for k in range(n) if dd[k] < dd[k - 1] and dd[k] <= dd[(k + 1) % n]],
                  key=lambda k: dd[k])[:2]
    cs = [(pts[k], dd[k]) for k in mins]
    chosen = int(np.argmin([np.linalg.norm(P - E) for P, _ in cs]))
    txt = " ".join(f"{'*' if j == chosen else ' '}[dzS{P[2]-S[2]:+.3f} rd{r*1000:5.1f}]"
                   for j, (P, r) in enumerate(cs))
    margin = (cs[1 - chosen][1] - cs[chosen][1]) * 1000 if len(cs) == 2 else float("nan")
    print(f"{i:4d} vis{vis:.2f} sep{(np.linalg.norm(cs[0][0]-cs[1][0])*1000 if len(cs)>1 else 0):5.0f}mm "
          f"margin{margin:+7.1f}mm | {txt}")
