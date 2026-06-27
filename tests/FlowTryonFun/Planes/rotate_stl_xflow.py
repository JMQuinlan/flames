#!/usr/bin/env python3
# Rotate a binary STL so the aircraft fuselage (long axis) aligns with +x and the
# NOSE points toward -x (into a freestream flowing from -x to +x).  AMReX's
# read_stl_file only scales+translates, so any reorientation must be baked into
# the mesh here.  Writes <name>_xflow.STL and prints the new bbox + the scale/
# center that map the model to its real length, origin-centered.
import struct, sys, numpy as np

def read_binary_stl(fn):
    with open(fn, 'rb') as f:
        data = f.read()
    n = struct.unpack('<I', data[80:84])[0]
    tris = np.zeros((n, 4, 3))   # [normal, v1, v2, v3]
    off = 84
    for i in range(n):
        vals = struct.unpack('<12fH', data[off:off+50]); off += 50
        tris[i, 0] = vals[0:3]    # normal
        tris[i, 1] = vals[3:6]    # v1
        tris[i, 2] = vals[6:9]    # v2
        tris[i, 3] = vals[9:12]   # v3
    return tris

def write_binary_stl(fn, tris):
    n = tris.shape[0]
    with open(fn, 'wb') as f:
        f.write(b'rotated for +x flow'.ljust(80, b'\0'))
        f.write(struct.pack('<I', n))
        for i in range(n):
            f.write(struct.pack('<12fH', *tris[i].flatten(), 0))

def main(fn, out, real_length):
    tris = read_binary_stl(fn)
    V = tris[:, 1:].reshape(-1, 3)          # all vertices
    ext = V.max(0) - V.min(0)
    print(f'{fn}: bbox ext = {ext.round(2)}  (assume longest=fuselage, mid=wingspan, short=height)')

    # (1) sort axes by extent so the plane flies LEVEL:
    #     x <- longest (fuselage),  y <- 2nd longest (wingspan, horizontal),
    #     z <- shortest (height/thickness, vertical).  Keep it a proper rotation.
    order = list(np.argsort(ext)[::-1])     # [longest, mid, short]
    R = np.zeros((3, 3))
    for i, src in enumerate(order):
        R[i, src] = 1.0
    if np.linalg.det(R) < 0:
        R[1] *= -1.0                        # left-right mirror (symmetric a/c) -> det +1, keeps up/down
    Vr = V @ R.T

    # (2) nose detection: tip end has the smaller cross-section in y-z.
    x = Vr[:, 0]; xlo, xhi = x.min(), x.max()
    slab = 0.08 * (xhi - xlo)
    def crossspan(mask):
        yz = Vr[mask][:, 1:]
        return np.hypot(*(yz - yz.mean(0)).T).mean() if mask.any() else 1e9
    span_lo = crossspan(x < xlo + slab)
    span_hi = crossspan(x > xhi - slab)
    nose_at = 'xlo(-x)' if span_lo < span_hi else 'xhi(+x)'
    print(f'  cross-span: -x end={span_lo:.2f}  +x end={span_hi:.2f}  -> nose at {nose_at}')

    # want nose at -x; if it is at +x, rotate 180 deg about z
    if span_hi < span_lo:
        R = np.array([[-1,0,0],[0,-1,0],[0,0,1]], float) @ R
        print('  -> flipping 180 about z so nose faces -x')

    # apply final rotation to vertices AND normals
    tris[:, 1:] = (tris[:, 1:].reshape(-1, 3) @ R.T).reshape(-1, 3, 3)
    tris[:, 0]  = tris[:, 0] @ R.T
    write_binary_stl(out, tris)

    Vf = tris[:, 1:].reshape(-1, 3)
    lo, hi = Vf.min(0), Vf.max(0); ext = hi - lo; ctr = (lo + hi) / 2
    s = real_length / ext[0]
    center = -s * ctr
    print(f'  WROTE {out}')
    print(f'  new bbox: x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] z[{lo[2]:.2f},{hi[2]:.2f}]')
    print(f'  scale  = {s:.6f}    (real length {real_length} m / {ext[0]:.2f})')
    print(f'  center = {center[0]:.4f} {center[1]:.4f} {center[2]:.4f}')
    print(f'  real size: len(x)={s*ext[0]:.2f} span(y)={s*ext[1]:.2f} height(z)={s*ext[2]:.2f}')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
