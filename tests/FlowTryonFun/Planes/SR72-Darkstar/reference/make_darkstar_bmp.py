#!/usr/bin/env python3
# ============================================================================
#  make_darkstar_bmp.py
#  ---------------------------------------------------------------------------
#  Convert a top-view aircraft image (the user's 199w2i-TopView.jpg) into a
#  greyscale embedded-solid phi BMP for Hydro2:
#     airframe = BLACK (0)  -> phi 0 (solid)
#     air      = WHITE (255)-> phi 1 (fluid)
#
#  Pipeline: greyscale -> threshold off the white background -> fill interior
#  holes (cockpit/panel markings) so the whole planform is solid -> orient so
#  the nose points -x (into the +x freestream) -> scale/place in the domain
#  frame -> Gaussian-blur the edge (diffuse phi).
#
#  Output: ../darkstar.bmp   (maps 1:1 onto the domain via fit=coord)
# ============================================================================
import os
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

# ----------------------------- CONFIG --------------------------------------
SRC_IMG   = "../199w2i-TopView.jpg"   # source aircraft image
OUT_BMP   = "../darkstar.bmp"
IMG_W, IMG_H = 1600, 800              # frame px (2:1, matches the domain aspect)
DOM_LO = (-3.0, -3.0)                 # domain box the frame maps onto
DOM_HI = ( 9.0,  3.0)

THRESH      = 232                     # pixel < THRESH => airframe; white bg => air
ROTATE_CCW  = 90                      # rotate source CCW so the nose points -x (90: nose-up -> nose-left)
FLIP_LR     = False                   # mirror left-right if needed
FILL_HOLES  = False                   # fill enclosed white markings (thin lines blur away anyway)

TARGET_LENGTH = 4.3                   # airframe nose->tail length in DOMAIN units
NOSE_X        = 0.3                   # domain x of the nose tip
CENTER_Y      = 0.0                   # domain y of the airframe centerline
BLUR_PX       = 5.0                   # Gaussian blur radius (px) -> diffuse phi edge
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def _abs(p): return p if os.path.isabs(p) else os.path.normpath(os.path.join(SCRIPT_DIR, p))


def fill_holes(mask):
    """Fill holes in a boolean solid mask (regions not connected to the border)."""
    try:
        from scipy.ndimage import binary_fill_holes
        return binary_fill_holes(mask)
    except Exception:
        # PIL flood-fill fallback: flood the OUTER background from the corner;
        # interior holes (not reached) become solid.
        inv = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), "L")  # solid=0, bg/holes=255
        ImageDraw.floodfill(inv, (0, 0), 0)                    # outer bg 255 -> 0
        holes = np.asarray(inv) == 255                         # enclosed interior holes
        return mask | holes


def main():
    src = _abs(SRC_IMG)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"source image not found: {src}")
    g = np.asarray(Image.open(src).convert("L"), dtype=np.uint8)

    # --- segment airframe off the white background ---
    solid = g < THRESH
    if FILL_HOLES:
        solid = fill_holes(solid)

    # airframe -> black (0), air -> white (255)
    im = Image.fromarray(np.where(solid, 0, 255).astype(np.uint8), mode="L")

    # --- orient: nose -x ---
    if ROTATE_CCW:
        im = im.rotate(ROTATE_CCW, expand=True, fillcolor=255)
    if FLIP_LR:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)

    # --- crop to the airframe bbox ---
    arr = np.asarray(im)
    ys, xs = np.where(arr < 128)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    plane = im.crop((x0, y0, x1 + 1, y1 + 1))
    pw, ph = plane.size                     # nose->tail (px), span (px)

    # --- scale so the airframe length = TARGET_LENGTH domain units ---
    dom_w = DOM_HI[0] - DOM_LO[0]
    dom_h = DOM_HI[1] - DOM_LO[1]
    px_per_x = IMG_W / dom_w
    px_per_y = IMG_H / dom_h
    new_w = max(1, int(round(TARGET_LENGTH * px_per_x)))
    new_h = max(1, int(round(new_w * (ph / pw) * (px_per_x / px_per_y))))  # keep true aspect in domain units
    plane = plane.resize((new_w, new_h), Image.LANCZOS)

    # --- paste into the white (fluid) frame at the requested domain position ---
    frame = Image.new("L", (IMG_W, IMG_H), 255)
    nose_px = int(round((NOSE_X - DOM_LO[0]) * px_per_x))
    cy_px   = int(round((DOM_HI[1] - CENTER_Y) * px_per_y))   # image y is top-down
    frame.paste(plane, (nose_px, cy_px - new_h // 2))

    # --- diffuse edge + save ---
    frame = frame.filter(ImageFilter.GaussianBlur(BLUR_PX))
    out = _abs(OUT_BMP)
    frame.convert("RGB").save(out, format="BMP")

    a = np.asarray(frame, dtype=float) / 255.0
    print(f"wrote {out}  ({IMG_W}x{IMG_H}, blur={BLUR_PX}px)")
    print(f"  airframe length = {TARGET_LENGTH} units (nose x={NOSE_X}); span = "
          f"{new_h/px_per_y:.2f} units ; solid(phi<0.5) = {(a<0.5).mean()*100:.1f}%")
    print(f"  set in input:  solid.phi.ic.bmp.coord.lo = {DOM_LO[0]} {DOM_LO[1]}")
    print(f"                 solid.phi.ic.bmp.coord.hi = {DOM_HI[0]} {DOM_HI[1]}")


if __name__ == "__main__":
    main()
