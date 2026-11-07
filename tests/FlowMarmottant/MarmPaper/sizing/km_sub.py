import os
import sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reference"))
from marmottant_rpe_km import Shell, Liquid, Gas, solve_km, natural_frequency
from multiprocessing import Pool
R0=2e-6; sh=Shell(chi=1.0,R_buck=1.96e-6,sigma_break=0.073,sigma_water=0.0728,kappa_s=7.2e-9)
c_l=np.sqrt(2.35*(1e5+1e9)/1000); p0=1e5; p_eq=p0+2*sh.sigma(R0)/R0
L=Liquid(rho=1000,mu=1e-3,c=c_l,p_inf=p0); w0=natural_frequency(R0,sh,L,Gas(p_g0=p_eq))
def one(a):
    fr,A=a; wd=fr*w0; T=2*np.pi/wd; N=60
    t=np.linspace(0,N*T,N*400+1)
    try: tt,R,_=solve_km(R0,0,(0,N*T),sh,L,Gas(p_g0=p_eq),drive=(A,wd),t_eval=t); R=R/R0
    except Exception as e: return (fr,A,None)
    if len(R)<len(t): return (fr,A,None)
    m=t>30*T; x=R[m]-R[m].mean(); F=np.abs(np.fft.rfft(x*np.hanning(len(x)))); fq=np.fft.rfftfreq(len(x),t[1]-t[0]); fd=wd/2/np.pi
    am=lambda f:F[np.argmin(np.abs(fq-f))]
    # period-doubling check: compare R at successive drive periods (late)
    k=np.arange(30,N)*400; pk=R[k]; pd=np.abs(pk[1:]-pk[:-1]).max()
    return (fr,A,(R.min(),R.max(),am(fd/2)/am(fd),am(2*fd)/am(fd),pd))
if __name__=='__main__':
  args=[(fr,A) for fr in [0.9,0.95,1.0,1.05,1.1] for A in [3.3e5,3.6e5,4.0e5,4.5e5]]
  with Pool(12) as p:
    for fr,A,r in p.map(one,args):
        if r is None: print(f"f/f0 {fr} A {A:.1e}: failed"); continue
        print(f"f/f0 {fr:.1f} A {A:.1e}: Rmin {r[0]:.3f} Rmax {r[1]:.3f}  f/2 {r[2]:.3f}  2f {r[3]:.3f}  period-jitter {r[4]:.4f}")
