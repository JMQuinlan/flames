"""
Helpers to digitize 1-bit scanned line plots (used for Kang et al. 1999 and Stojkovic et al. 2002).

  ink = load(png)                         boolean array [row, col], True = black
  box = frame(ink, rows=(r0, r1))         (left, right, top, bottom) pixel of the axis frame in a row band
  xt, yt = ticks(ink, box)                pixel positions of the tick marks on the bottom / left axis
  markers(ink, box)                       list of dicts: centre (col, row), bbox, area, fill ratio, hollow?, shape guess
  Axis(p0, v0, p1, v1)                    linear pixel -> value map

Markers are found by filling the holes of the image and removing everything thinner than the
marker (binary opening): what survives are the marker bodies; lines and text strokes vanish.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def load(png):
    a = np.asarray(Image.open(png).convert("L"))
    return a < 128


def _runs(v):
    """(start, length) of the True runs of a 1D boolean array"""
    d = np.diff(np.concatenate([[0], v.astype(np.int8), [0]]))
    s = np.where(d == 1)[0]; e = np.where(d == -1)[0]
    return s, e - s


def frame(ink, rows=None, cols=None, minfrac=0.6):
    r0, r1 = rows if rows else (0, ink.shape[0]); c0, c1 = cols if cols else (0, ink.shape[1])
    sub = ink[r0:r1, c0:c1]
    hl = [(r, *max(zip(*_runs(sub[r])[::-1]), default=(0, 0))) for r in range(sub.shape[0])]      # (row, len, start)
    vl = [(c, *max(zip(*_runs(sub[:, c])[::-1]), default=(0, 0))) for c in range(sub.shape[1])]
    H = [h for h in hl if h[1] > minfrac * sub.shape[1]]; V = [v for v in vl if v[1] > minfrac * sub.shape[0]]
    top, bottom = H[0][0], H[-1][0]; left, right = V[0][0], V[-1][0]
    return left + c0, right + c0, top + r0, bottom + r0


def ticks(ink, box, depth=14, minlen=5):
    """tick marks pointing INTO the frame from the bottom and left axes: centre pixel of each"""
    L, R, T, B = box
    # bottom axis: for each column, length of the ink run going up from the frame line
    up = np.array([(np.argmin(ink[B - depth:B + 1, c][::-1]) if not ink[B - depth:B + 1, c].all() else depth + 1) for c in range(L, R + 1)])
    base = np.median(up); s, n = _runs(up >= base + minlen)
    xt = [L + a + (b - 1) / 2 for a, b in zip(s, n) if b <= 8]
    rt = np.array([(np.argmin(ink[r, L:L + depth + 1]) if not ink[r, L:L + depth + 1].all() else depth + 1) for r in range(T, B + 1)])
    base = np.median(rt); s, n = _runs(rt >= base + minlen)
    yt = [T + a + (b - 1) / 2 for a, b in zip(s, n) if b <= 8]
    return np.array(xt), np.array(yt)


class Axis:
    def __init__(self, p0, v0, p1, v1): self.p0, self.v0, self.k = p0, v0, (v1 - v0) / (p1 - p0)
    def __call__(self, p): return self.v0 + (np.asarray(p) - self.p0) * self.k
    def pix(self, v): return self.p0 + (np.asarray(v) - self.v0) / self.k

    @staticmethod
    def fit(pix, vals):
        k, b = np.polyfit(pix, vals, 1); a = Axis(0.0, b, 1.0, b + k); a.rms = float(np.sqrt(np.mean((np.polyval([k, b], pix) - vals)**2))); return a


def markers(ink, box, open_r=3, amin=40, amax=600, pad=3):
    L, R, T, B = box
    T0, L0 = max(T - pad, 0), max(L - pad, 0)
    sub = ink[T0:B + pad + 1, L0:R + pad + 1].copy()
    # remove the frame lines so that markers on the axes are not glued to them
    fr = np.zeros_like(sub)
    for r in range(sub.shape[0]):
        s, n = _runs(sub[r]);
        for a, b in zip(s, n):
            if b > 0.5 * sub.shape[1]: fr[r, a:a + b] = True
    for c in range(sub.shape[1]):
        s, n = _runs(sub[:, c])
        for a, b in zip(s, n):
            if b > 0.5 * sub.shape[0]: fr[a:a + b, c] = True
    work = sub & ~fr
    filled = ndi.binary_fill_holes(work)
    yy, xx = np.ogrid[-open_r:open_r + 1, -open_r:open_r + 1]; se = xx * xx + yy * yy <= open_r * open_r
    body = ndi.binary_opening(filled, structure=se)
    lab, n = ndi.label(body); out = []
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        m = lab[sl] == k; area = int(m.sum())
        if not (amin <= area <= amax): continue
        h, w = m.shape; cy, cx = ndi.center_of_mass(m)
        inkfrac = float((work[sl] & m).sum()) / area             # ~1 solid marker, small for hollow
        fill = area / (h * w)
        # vertical asymmetry: triangle up has its mass low (large row), triangle down high
        asym = (cy - (h - 1) / 2) / h
        prof = m.sum(axis=1); top_w = prof[: max(1, h // 4)].mean() / w; bot_w = prof[-max(1, h // 4):].mean() / w
        if fill > 0.9: shape = "square"
        elif top_w < 0.45 and bot_w > 0.7: shape = "tri_up"
        elif bot_w < 0.45 and top_w > 0.7: shape = "tri_down"
        elif top_w < 0.5 and bot_w < 0.5: shape = "diamond" if fill < 0.62 else "circle"
        else: shape = "circle"
        out.append(dict(col=L0 + sl[1].start + cx, row=T0 + sl[0].start + cy, w=w, h=h, area=area, fill=round(fill, 2),
                        solid=inkfrac > 0.8, shape=shape, top_w=round(float(top_w), 2), bot_w=round(float(bot_w), 2)))
    return out


def regular(t, tol=0.25):
    """keep the ticks that lie on the dominant regular spacing"""
    t = np.sort(np.asarray(t, float)); d = np.diff(t); sp = np.median(d[d > 0.5 * np.median(d)])
    best = []
    for t0 in t:
        k = np.round((t - t0) / sp); ok = np.abs(t - t0 - k * sp) < tol * sp
        if ok.sum() > len(best): best = t[ok]
    return np.array(best), sp


def axis_from_ticks(t, anchor_pix, anchor_val, step):
    """regular ticks t (pixels); the tick nearest anchor_pix has value anchor_val; consecutive ticks differ by step
    (signed: for a y axis, value per tick going DOWN the image)"""
    t, sp = regular(t); i0 = int(np.argmin(np.abs(t - anchor_pix)))
    k = np.round((t - t[i0]) / sp); ax = Axis.fit(t, anchor_val + k * step); ax.n = len(t); return ax


def edge_markers(ink, box, side="left", width=7, minh=7):
    """markers sitting ON an axis (half hidden by the frame): row centres (side = left/right) or column centres
    (bottom/top) of the ink clusters in a strip just inside the frame; ticks (thinner than minh) are ignored"""
    L, R, T, B = box
    if side in ("left", "right"):
        strip = ink[T + 2:B - 1, L + 2:L + 2 + width] if side == "left" else ink[T + 2:B - 1, R - 1 - width:R - 1]
        v = strip.any(axis=1); s, n = _runs(v); return [T + 2 + a + (b - 1) / 2 for a, b in zip(s, n) if b >= minh]
    strip = ink[B - 1 - width:B - 1, L + 2:R - 1] if side == "bottom" else ink[T + 2:T + 2 + width, L + 2:R - 1]
    v = strip.any(axis=0); s, n = _runs(v); return [L + 2 + a + (b - 1) / 2 for a, b in zip(s, n) if b >= minh]


def template(ink, col, row, half=9):
    """patch of the image around a clean, isolated marker (centre col, row), to be used as a matched filter"""
    c, r = int(round(col)), int(round(row)); return ink[r - half:r + half + 1, c - half:c + half + 1].astype(float)


def synth(shape, size=13, lw=1.6, half=9):
    """synthetic hollow marker outline (circle, square, tri_up, tri_down, diamond) of the given full size [pixels]"""
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float); r = size / 2.0
    if shape == "circle": d = np.abs(np.hypot(x, y) - r)
    elif shape == "square": d = np.abs(np.maximum(np.abs(x), np.abs(y)) - r)
    elif shape == "diamond": d = np.abs((np.abs(x) + np.abs(y)) - r) / np.sqrt(2)
    else:
        sgn = 1.0 if shape == "tri_up" else -1.0; yy = sgn * y        # apex at yy = -h_a, base at yy = +h_b
        hb = r * 0.75; ha = r * 1.1; w = r * 1.15
        # distance to the three edges of the triangle (apex (0,-ha), base corners (+-w, hb))
        def seg(px, py, ax, ay, bx, by):
            vx, vy = bx - ax, by - ay; t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy), 0, 1)
            return np.hypot(px - ax - t * vx, py - ay - t * vy)
        d = np.minimum(np.minimum(seg(x, yy, 0, -ha, w, hb), seg(x, yy, 0, -ha, -w, hb)), seg(x, yy, -w, hb, w, hb))
    return (d <= lw / 2 + 0.25).astype(float)


def match(ink, tmpl, col, rows, dcol=2):
    """normalised cross-correlation of the template with the image, centred on column col (+-dcol), for every row
    in rows = (r0, r1).  Returns (rows, best score over the column offsets, best offset)"""
    h = tmpl.shape[0] // 2; t = tmpl - tmpl.mean(); tn = np.sqrt((t * t).sum())
    r0, r1 = rows; rr = np.arange(max(r0, h), min(r1, ink.shape[0] - h - 1) + 1); best = np.full(len(rr), -1.0); off = np.zeros(len(rr), int)
    c0 = int(round(col))
    for dc in range(-dcol, dcol + 1):
        c = c0 + dc
        if c - h < 0 or c + h + 1 > ink.shape[1]: continue
        for n, r in enumerate(rr):
            p = ink[r - h:r + h + 1, c - h:c + h + 1].astype(float); p = p - p.mean(); pn = np.sqrt((p * p).sum())
            sc = (p * t).sum() / (pn * tn) if pn > 0 else -1.0
            if sc > best[n]: best[n] = sc; off[n] = dc
    return rr, best, off


def peaks(rr, sc, thresh=0.5, sep=5):
    """local maxima of a score profile above thresh, at least sep rows apart: list of (row, score)"""
    out = []
    for n in np.argsort(sc)[::-1]:
        if sc[n] < thresh: break
        if all(abs(rr[n] - r) >= sep for r, _ in out): out.append((float(rr[n]), float(sc[n])))
    return sorted(out)


def coverage(ink, tmpl, col, rows, dcol=3, tol=1):
    """fraction of the template's outline pixels that fall on ink (ink dilated by tol): robust to extra ink
    (lines, overlapping markers).  Scanned over the rows (r0, r1) and column offsets +-dcol around col."""
    h = tmpl.shape[0] // 2; t = tmpl > 0; n = t.sum(); fat = ndi.binary_dilation(ink, iterations=tol) if tol else ink
    r0, r1 = rows; rr = np.arange(max(r0, h), min(r1, ink.shape[0] - h - 1) + 1); best = np.zeros(len(rr)); off = np.zeros(len(rr), int)
    c0 = int(round(col))
    for dc in range(-dcol, dcol + 1):
        c = c0 + dc
        if c - h < 0 or c + h + 1 > ink.shape[1]: continue
        for k, r in enumerate(rr):
            sc = (fat[r - h:r + h + 1, c - h:c + h + 1] & t).sum() / n
            if sc > best[k]: best[k] = sc; off[k] = dc
    return rr, best, off


# ---------------------------------------------------------------- large markers: templates cut from the legend
def legend_template(ink, row, col, half=18, open_r=4, solid=False):
    """ring / interior (hollow marker) or body / halo (solid marker) masks of an isolated legend marker at (row, col);
    the legend's line through the marker is removed by the opening."""
    r, c = int(round(row)), int(round(col)); p = ink[r - half:r + half + 1, c - half:c + half + 1]
    yy, xx = np.ogrid[-open_r:open_r + 1, -open_r:open_r + 1]; se = xx * xx + yy * yy <= open_r * open_r
    body = ndi.binary_opening(ndi.binary_fill_holes(p), structure=se)
    lab, n = ndi.label(body)
    if n > 1: body = lab == lab[half, half] if lab[half, half] else lab == (1 + int(np.argmax(ndi.sum(body, lab, range(1, n + 1)))))
    if solid:
        return dict(solid=True, pos=body, neg=ndi.binary_dilation(body, iterations=4) & ~ndi.binary_dilation(body, iterations=1))
    ring = p & ndi.binary_dilation(body, iterations=1) & ~ndi.binary_erosion(body, iterations=3)
    return dict(solid=False, pos=ring, neg=ndi.binary_erosion(body, iterations=4))


def score_map(ink, t, cols, rows, tol=1, wneg=0.6):
    """score = fraction of the template's positive pixels on ink - wneg * fraction of its negative pixels on ink,
    for marker centres in cols = (c0, c1), rows = (r0, r1).  Returns (score array, r0, c0)."""
    h = t["pos"].shape[0] // 2; P = np.pad(ink, h + 2); fat = ndi.binary_dilation(P, iterations=tol) if tol and not t["solid"] else P
    pos, neg = t["pos"], t["neg"]; npos, nneg = max(pos.sum(), 1), max(neg.sum(), 1)
    c0, c1 = cols; r0, r1 = rows; out = np.zeros((r1 - r0 + 1, c1 - c0 + 1))
    for i, r in enumerate(range(r0, r1 + 1)):
        for j, c in enumerate(range(c0, c1 + 1)):
            a = fat[r + 2:r + 2 * h + 3, c + 2:c + 2 * h + 3]; b = P[r + 2:r + 2 * h + 3, c + 2:c + 2 * h + 3]
            out[i, j] = (a & pos).sum() / npos - wneg * (b & neg).sum() / nneg
    return out, r0, c0


def best_peaks(sc, r0, c0, n=3, sep=6):
    """the n best local maxima of a score map: list of (score, row, col)"""
    s = sc.copy(); out = []
    for _ in range(n):
        i, j = np.unravel_index(np.argmax(s), s.shape); out.append((float(s[i, j]), r0 + i, c0 + j))
        s[max(i - sep, 0):i + sep + 1, max(j - sep, 0):j + sep + 1] = -9
    return out


def zoom(ink, X, Y, xr, yr, out, scale=6, ystep=1.0, xstep=None, ysub=None):
    """enlarged crop of the data window xr x yr (data units) with labelled grid lines every ystep (and xstep):
    for reading values by eye where markers overlap too much for the detectors."""
    from PIL import Image, ImageDraw
    c0, c1 = sorted(int(round(float(X.pix(v)))) for v in xr); r0, r1 = sorted(int(round(float(Y.pix(v)))) for v in yr)
    c0, r0 = max(c0, 0), max(r0, 0); p = ink[r0:r1 + 1, c0:c1 + 1]
    im = Image.fromarray(np.where(p, 0, 255).astype(np.uint8)).resize((p.shape[1] * scale, p.shape[0] * scale), Image.NEAREST).convert("RGB")
    d = ImageDraw.Draw(im)
    def hl(v, col, lab=True):
        y = (float(Y.pix(v)) - r0 + 0.5) * scale; d.line([(0, y), (im.width, y)], fill=col, width=1)
        if lab: d.text((2, y - 11), f"{v:g}", fill=col)
    lo, hi = sorted(yr)
    if ysub:
        v = np.ceil(lo / ysub) * ysub
        while v <= hi: hl(v, (255, 200, 200), False); v += ysub
    v = np.ceil(lo / ystep) * ystep
    while v <= hi + 1e-9: hl(v, (255, 0, 0)); v += ystep
    if xstep:
        lo, hi = sorted(xr); v = np.ceil(lo / xstep) * xstep
        while v <= hi + 1e-9:
            x = (float(X.pix(v)) - c0 + 0.5) * scale; d.line([(x, 0), (x, im.height)], fill=(0, 0, 255), width=1); d.text((x + 2, 2), f"{v:g}", fill=(0, 0, 255)); v += xstep
    im.save(out); return out


def refine(ink, t, col, row, dcol=4, drow=8):
    """best position of template t within +-dcol, +-drow of an expected marker centre: (score, row, col)"""
    sc, r0, c0 = score_map(ink, t, (int(round(col)) - dcol, int(round(col)) + dcol), (int(round(row)) - drow, int(round(row)) + drow))
    i, j = np.unravel_index(np.argmax(sc), sc.shape)
    # average over the plateau of near-best positions (flat maxima on straight outlines)
    ii, jj = np.where(sc >= sc[i, j] - 0.02)
    return float(sc[i, j]), r0 + float(ii.mean()), c0 + float(jj.mean())
