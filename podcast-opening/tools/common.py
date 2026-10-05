"""Shared helpers: frame I/O, alignment, and flow-based retiming."""
import cv2
import numpy as np
import subprocess

W, H, FPS = 3840, 2160, 30


def read_frames(path, start, end):
    """Return frames [start, end] inclusive (BGR uint8)."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out, i = [], 0
    while i <= end:
        ok, f = cap.read()
        if not ok:
            break
        if i >= start:
            out.append(f)
        i += 1
    cap.release()
    return out


def align_affine(ref, img, mask=None, scale=0.25):
    """Partial-affine transform mapping ref -> img using ORB features inside mask."""
    r = cv2.resize(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
    m = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
    mk = None if mask is None else cv2.resize(mask, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_NEAREST)
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(r, mk)
    k2, d2 = orb.detectAndCompute(m, mk)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ms = bf.match(d1, d2)
    p1 = np.float32([k1[x.queryIdx].pt for x in ms]) / scale
    p2 = np.float32([k2[x.trainIdx].pt for x in ms]) / scale
    A, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    return A


def warp(img, A):
    return cv2.warpAffine(img, A, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


_dis = None


def interp(a, b, alpha, scale=0.5):
    """Optical-flow interpolated frame between a and b at fraction alpha."""
    global _dis
    alpha = float(alpha)
    if alpha <= 1e-3:
        return a
    if alpha >= 1 - 1e-3:
        return b
    if _dis is None:
        _dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
    fab = _dis.calc(ga, gb, None)
    fba = _dis.calc(gb, ga, None)
    h, w = a.shape[:2]
    fab = cv2.resize(fab, (w, h)) / scale
    fba = cv2.resize(fba, (w, h)) / scale
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    # backward-warp both endpoints toward the intermediate time
    wa = cv2.remap(a, gx - alpha * fab[..., 0], gy - alpha * fab[..., 1], cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    wb = cv2.remap(b, gx - (1 - alpha) * fba[..., 0], gy - (1 - alpha) * fba[..., 1], cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(wa, 1 - alpha, wb, alpha, 0)


class Writer:
    """Pipe BGR frames into ffmpeg (lossless FFV1 intermediate by default)."""

    def __init__(self, path, w=W, h=H, fps=FPS, codec=("-c:v", "ffv1", "-level", "3")):
        self.p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", *codec, path], stdin=subprocess.PIPE)

    def write(self, f):
        self.p.stdin.write(np.ascontiguousarray(f).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def laplace_fill(img, hole, wx=1.0, wy=1.0):
    """Harmonic (membrane) fill of hole pixels from their boundary values.

    wx/wy weight horizontal/vertical neighbours; wy >> wx keeps vertical wall panelling straight.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    h, w = hole.shape
    ys, xs = np.nonzero(hole)
    idx = -np.ones((h, w), np.int64)
    idx[ys, xs] = np.arange(len(ys))
    n = len(ys)
    rows, cols, vals = [], [], []
    b = np.zeros((n, img.shape[2]))
    for dy, dx, wt in ((-1, 0, wy), (1, 0, wy), (0, -1, wx), (0, 1, wx)):
        ny, nx = np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)
        inside = idx[ny, nx] >= 0
        rows.append(np.arange(n)[inside]); cols.append(idx[ny, nx][inside]); vals.append(np.full(inside.sum(), -wt))
        b[~inside] += wt * img[ny[~inside], nx[~inside]]
    A = sp.csr_matrix((np.concatenate(vals + [np.full(n, 2 * (wx + wy))]),
                       (np.concatenate(rows + [np.arange(n)]), np.concatenate(cols + [np.arange(n)]))), (n, n))
    out = img.astype(np.float64).copy()
    for c in range(img.shape[2]):
        out[ys, xs, c] = spsolve(A, b[:, c])
    return out


def smooth_affines(As, keep=None, degree=3, t=None, sigma=None):
    """Fit each affine parameter with a low-order polynomial over time.

    Per-frame estimates are noisy; the real camera move is smooth, so smoothing the path
    removes jitter in the patched area. keep marks frames used for the fit (e.g. skip
    duplicated 24p->30p frames, which would otherwise bias it). t is the time axis (defaults to
    frame index; pass the unique-frame index so duplicates don't count as camera time).
    """
    As = np.array(As, np.float64).reshape(len(As), 6)
    t = np.arange(len(As), dtype=np.float64) if t is None else np.asarray(t, np.float64)
    m = np.ones(len(As), bool) if keep is None else np.asarray(keep, bool)
    out = np.empty_like(As)
    for j in range(6):
        if sigma is None:   # global polynomial: for steady moves (drone push)
            c = np.polyfit(t[m], As[m, j], degree)
            out[:, j] = np.polyval(c, t)
        else:               # local Gaussian: follows real accelerations, removes frame-to-frame noise
            w = np.exp(-0.5 * ((t[:, None] - t[m][None, :]) / sigma) ** 2)
            out[:, j] = (w * As[m, j][None, :]).sum(1) / w.sum(1)
    return out.reshape(-1, 2, 3).astype(np.float32)
