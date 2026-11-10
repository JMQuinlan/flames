# usage: python3 make_cattaneo_envelope.py OUT.txt [UltrasoundPulse.dat]
# Envelope of the measured Cattaneo et al. 2025 pulse, following ExperimentalPressureEnvelope.m:
#   P = lowpass(P, 2 MHz, fs, 'ImpulseResponse','iir')   -> zero-phase elliptic low-pass
#   [up,lo] = envelope(P, 5000, 'peak')                  -> spline through peaks >= 5000 samples apart
#   start where 0.5*(up-lo) > 5e-5; scale = 1/max(up); up *= scale; lo = -scale*lo
# Output: t[s] up lo on [0, NCP/f] (burst of NCP cycles, as in BubbleRadiusDisplacementImpulseStress.m).
import sys, numpy as np
from scipy.signal import ellip, sosfiltfilt, find_peaks
from scipy.interpolate import CubicSpline
fs = 1e10
D = np.loadtxt(sys.argv[2] if len(sys.argv) > 2 else 'UltrasoundPulse.dat'); P = D[:, 1]
# MATLAB lowpass(x, fpass, fs, 'ImpulseResponse','iir') defaults: Steepness 0.85, passband ripple 0.1 dB,
# stopband attenuation 60 dB -> stopband edge fpass + (1-0.85)*(fs/2 - fpass); minimum-order elliptic, zero-phase.
from scipy.signal import ellipord
fpass = 2e6; fstop = fpass + (1 - 0.85) * (fs / 2 - fpass)
N, Wn = ellipord(fpass, fstop, 0.1, 60, fs=fs)
sos = ellip(N, 0.1, 60, Wn, btype='low', fs=fs, output='sos')
print('elliptic order', N, 'stopband', fstop)
P = sosfiltfilt(sos, P)
n = np.arange(len(P))
def env(x):
    # MATLAB envelope(x, np, 'peak'): spline through local maxima >= np apart (first/last samples only
    # added when fewer than 2 peaks); interp1(...,'spline') extrapolates beyond the end peaks.
    pk, _ = find_peaks(x, distance=5000)
    idx = pk if len(pk) >= 2 else np.r_[0, pk, len(x) - 1]
    return CubicSpline(idx, x[idx], bc_type='not-a-knot', extrapolate=True)(n)
up = env(P); lo = -env(-P)
e = 0.5 * (up - lo); i0 = np.argmax(e > 5e-5)
t = np.arange(len(P) - i0) / fs
sc = 1.0 / up.max(); UP = sc * up[i0:]; LO = -sc * lo[i0:]
f, NCP = 1e6, 40
tt = np.arange(0.0, NCP / f + 1e-15, 5e-9)
out = np.c_[tt, np.interp(tt, t, UP, right=0.0), np.interp(tt, t, LO, right=0.0)]
np.savetxt(sys.argv[1], out, fmt='%.9e %.6e %.6e',
           header='Cattaneo et al. 2025 (Nat Phys 21:590) measured 1 MHz pulse envelope; t[s] up lo; burst 0..40 us (NCP=40)')
print('rows', len(out), 'up max', out[:, 1].max(), 'at', out[out[:, 1].argmax(), 0] * 1e6, 'us; up at 2,5,10,20,30 us:',
      [round(float(np.interp(x * 1e-6, tt, out[:, 1])), 3) for x in (2, 5, 10, 20, 30)])
