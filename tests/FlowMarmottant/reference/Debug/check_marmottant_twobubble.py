"""Two-bubble Marmottant check (LOCAL sigma(R) discriminator).

Two gas bubbles of different radii sit in one domain (A at x<0, B at x>0), each in
its OWN Laplace balance with its OWN Marmottant sigma(R_i). A *global* averaged radius
gives a single sigma -> at least one bubble is mis-tensioned -> it drifts. A *local*
sigma(R(x)) tensions each correctly -> both stay put. We split the domain at x=0 and
track each bubble's radius from its gas volume: 2D R_i = sqrt(V_g,i/pi).

Usage: python check_marmottant_twobubble.py <output_dir> [Ra] [Rb]
"""
import glob, os, sys
import numpy as np
import yt
yt.funcs.mylog.setLevel(40)

Ra = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
Rb = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25

def cells(d):
    return sorted(glob.glob(os.path.join(d, "*cell")))

def radii(pf):
    ds = yt.load(pf)
    ad = ds.all_data()
    x   = np.array(ad["x"])
    ag  = np.clip(1.0 - np.array(ad["eta"]), 0.0, 1.0)   # gas fraction
    try:    vol = np.array(ad["index", "cell_volume"])
    except Exception: vol = np.array(ad["cell_volume"])
    VgA = float(np.sum((ag * vol)[x < 0.0]))             # bubble A (left)
    VgB = float(np.sum((ag * vol)[x > 0.0]))             # bubble B (right)
    RA = np.sqrt(VgA / np.pi)
    RB = np.sqrt(VgB / np.pi)
    return float(ds.current_time), float(RA), float(RB)

def main():
    d = sys.argv[1]
    cs = cells(d)
    if not cs:
        print("NO OUTPUT in", d); sys.exit(1)
    print("=" * 64)
    print("MARMOTTANT TWO-BUBBLE CHECK :", os.path.basename(d))
    print("=" * 64)
    print(f"  nominal Ra={Ra}, Rb={Rb}")
    print(f"  {'t':>11} {'RA':>9} {'RA/Ra':>8} {'RB':>9} {'RB/Rb':>8}")
    RAs, RBs = [], []
    for c in cs:
        t, RA, RB = radii(c)
        RAs.append(RA); RBs.append(RB)
        print(f"  {t:>11.4e} {RA:>9.5f} {RA/Ra:>8.4f} {RB:>9.5f} {RB/Rb:>8.4f}")
    RAs, RBs = np.array(RAs), np.array(RBs)
    dA = 100.0 * np.max(np.abs(RAs - RAs[0]) / RAs[0])
    dB = 100.0 * np.max(np.abs(RBs - RBs[0]) / RBs[0])
    print("\n" + "=" * 64)
    print(f"  bubble A: RA(end)/Ra={RAs[-1]/Ra:.4f}  max drift={dA:.2f}%")
    print(f"  bubble B: RB(end)/Rb={RBs[-1]/Rb:.4f}  max drift={dB:.2f}%")
    ok = (dA < 10.0) and (dB < 10.0)
    print(f"  BOTH STABLE (<10% drift each): {'PASS' if ok else 'FAIL'}")
    print("=" * 64)

if __name__ == "__main__":
    main()
