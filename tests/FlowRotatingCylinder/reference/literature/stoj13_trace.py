import sys; sys.path.insert(0,'.'); import digitize_tools as D, numpy as np
def trace(ink, box, X, Y, xticks, yticks, blank=None, gap=3, from_top=False, tick=12, extra=4, mincols=3):
    """markers (x crosses) on the curve nearest the bottom of the frame: list of (x value, y value)"""
    L,R,T,B=box; c0,r0=L+3,T+3; sub=ink[r0:B-2, c0:R-2].copy()
    if blank: sub[blank[0]:blank[1], blank[2]:blank[3]]=False
    for xt in xticks:
        j=int(round(xt-c0)); sub[-tick:, max(j-3,0):j+4]=False; sub[:tick, max(j-3,0):j+4]=False
    for yt in yticks:
        i=int(round(yt-r0)); sub[max(i-3,0):i+4, :tick]=False; sub[max(i-3,0):i+4, -tick:]=False
    span=np.zeros(sub.shape[1]); mid=np.full(sub.shape[1],np.nan)
    for j in range(sub.shape[1]):
        r=np.where(sub[:,j])[0]
        if not len(r): continue
        cl=np.split(r,np.where(np.diff(r)>gap)[0]+1); cc=cl[0] if from_top else cl[-1]
        span[j]=cc.max()-cc.min()+1; mid[j]=0.5*(cc.max()+cc.min())
    base=np.median(span[span>0]); s,nn=D._runs(span>=base+extra); out=[]
    for a,b in zip(s,nn):
        if b<mincols: continue
        jc=a+(b-1)/2; j=int(round(jc)); out.append((float(X(c0+jc)), float(Y(r0+mid[j])), int(b), float(span[j])))
    return out, span, mid, (c0,r0), base
