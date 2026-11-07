import os
import sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reference"))
from marmottant_rpe_km import Shell, Liquid, Gas, solve_km, natural_frequency
R0=2e-6; sh=Shell(chi=1.0,R_buck=1.96e-6,sigma_break=0.073,sigma_water=0.0728,kappa_s=7.2e-9)
c_l=np.sqrt(2.35*(1e5+1e9)/1000); p0=1e5; p_eq=p0+2*sh.sigma(R0)/R0
L=Liquid(rho=1000,mu=1e-3,c=c_l,p_inf=p0); w0=natural_frequency(R0,sh,L,Gas(p_g0=p_eq))
wd=w0; T=2*np.pi/wd; N=40; t=np.linspace(0,N*T,N*400+1)
_,R,_=solve_km(R0,0,(0,N*T),sh,L,Gas(p_g0=p_eq),drive=(4e5,wd),t_eval=t); R/=R0
mins=[R[k*400:(k+1)*400].min() for k in range(N)]
print("per-cycle Rmin:", " ".join(f"{m:.3f}" for m in mins))
# linear f0/pi: settle time
wl=w0/np.pi; Tl=2*np.pi/wl; t=np.linspace(0,10*Tl,10*400+1)
_,R,_=solve_km(R0,0,(0,10*Tl),sh,L,Gas(p_g0=p_eq),drive=(1e4,wl),t_eval=t); R/=R0
print("linear per-cycle amp:", " ".join(f"{(R[k*400:(k+1)*400].max()-R[k*400:(k+1)*400].min())/2:.5f}" for k in range(10)))
print(f"T0 {2*np.pi/w0*1e9:.1f} ns, T_lin {Tl*1e9:.1f} ns")
