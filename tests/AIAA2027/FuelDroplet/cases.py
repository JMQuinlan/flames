# Case list for make_fuel_decks.py (exec'd there): (fluid, D [m], Ms, max_level, vaporization).
# D = 10 um.  Mach sweep as the water column, each with and without vaporization at D/dx = 100 (max_level 2);
# D/dx = 200 (max_level 3) variants for INCLINE.  n-dodecane first.
D_FUEL = 10e-6
for _fl in ("dodecane", "pentane"):
    for _M in (1.47, 2.0, 3.0, 5.0, 7.0, 10.0):
        CASES += [(_fl, D_FUEL, _M, 2, 0), (_fl, D_FUEL, _M, 2, 1), (_fl, D_FUEL, _M, 3, 0), (_fl, D_FUEL, _M, 3, 1)]

# Long-domain fragment-tracking runs (2026-11-12): n-pentane only (dodecane evaporates ~1e-6 of its mass during breakup),
# Ms 1.47, 2, 3 only (post-shock p 2.4-10.5 bar, below the n-pentane critical pressure 33.7 bar where the film model
# holds; Ms 5 is at 29 bar, near-critical).  Finest D/dx = 100 (base 12.5, max_level 3), t* = 8.
for _M in (1.47, 2.0, 3.0):
    CASES += [("pentane", D_FUEL, _M, 3, 0, LONG), ("pentane", D_FUEL, _M, 3, 1, LONG)]

# Tandem pair (2026-11-12): Ms 3 on two n-pentane columns, D = 10 um each (same as the single-column runs), 10 um
# surface-to-surface gap (PAIR_GAP) along the shock direction; LONG layout.
CASES += [("pentane", D_FUEL, 3.0, 3, 0, PAIR), ("pentane", D_FUEL, 3.0, 3, 1, PAIR)]
