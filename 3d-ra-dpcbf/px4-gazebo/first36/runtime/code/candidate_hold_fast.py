"""Allocation-free interval evaluation of the exact Park gradient and CBF."""
import numpy as np
from numba import njit
@njit(inline="always")
def ia(a,b):return np.nextafter(a[0]+b[0],-np.inf),np.nextafter(a[1]+b[1],np.inf)
@njit(inline="always")
def ine(a):return -a[1],-a[0]
@njit(inline="always")
def isu(a,b):return ia(a,ine(b))
@njit(inline="always")
def im(a,b):
    p=a[0]*b[0];q=a[0]*b[1];r=a[1]*b[0];s=a[1]*b[1]
    return np.nextafter(min(p,q,r,s),-np.inf),np.nextafter(max(p,q,r,s),np.inf)
@njit(inline="always")
def isc(a,s):return im(a,(s,s))
@njit(inline="always")
def iq(a):
    low=0. if a[0]<=0.<=a[1] else min(a[0]*a[0],a[1]*a[1])
    return max(0.,np.nextafter(low,-np.inf)),np.nextafter(max(a[0]*a[0],a[1]*a[1]),np.inf)
@njit(inline="always")
def ir(a):
    if a[0]<=0.<=a[1]:return -np.inf,np.inf
    return np.nextafter(1./a[1],-np.inf),np.nextafter(1./a[0],np.inf)
@njit(inline="always")
def isq(a):
    if a[0]<=0:return np.nan,np.nan
    return np.nextafter(np.sqrt(a[0]),-np.inf),np.nextafter(np.sqrt(a[1]),np.inf)
@njit(cache=True)
def evaluate(rr,ww,ar,R0,kh,delta):
    d2=ia(ia(iq(rr[0]),iq(rr[1])),iq(rr[2]))
    s2=ia(ia(ia(iq(ww[0]),iq(ww[1])),iq(ww[2])),(1e-12,1e-12))
    D=isq(d2);L=isq(isu(d2,(R0*R0,R0*R0)));S=isq(s2)
    if not np.isfinite(L[0]) or not np.isfinite(S[0]):return -np.inf
    iD=ir(D);iL=ir(L);iS=ir(S);iS3=im(iq(iS),iS)
    e=(im(rr[0],iD),im(rr[1],iD),im(rr[2],iD))
    q=ia(ia(im(e[0],ww[0]),im(e[1],ww[1])),im(e[2],ww[2]))
    z=(isu(ww[0],im(q,e[0])),isu(ww[1],im(q,e[1])),isu(ww[2],im(q,e[2])))
    Z=ia(ia(iq(z[0]),iq(z[1])),iq(z[2]))
    lam=isc(im(L,iS),.1);h=ia(ia(q,im(lam,Z)),isc(L,.5))
    common=im(ia(isc(im(Z,iS),.1),(.5,.5)),im(D,iL))
    phi=(0.,0.);bn=(0.,0.)
    for l in range(3):
        gr=isu(ia(im(z[l],iD),im(common,e[l])),isc(im(im(im(lam,q),z[l]),iD),2.))
        gw=isu(ia(e[l],isc(im(lam,z[l]),2.)),isc(im(im(im(L,Z),ww[l]),iS3),.1))
        phi=ia(phi,ia(im(gr,ww[l]),im(gw,ar[l])))
        v=max(abs(gw[0]),abs(gw[1]));bn=ia(bn,iq((v,v)))
    phi=ia(phi,isc(h,kh))
    return isu(phi,isc((np.nextafter(np.sqrt(bn[1]),-np.inf),np.nextafter(np.sqrt(bn[1]),np.inf)),delta))[0]
@njit(cache=True)
def candidate_lower(x,obs,par,u,dt,delta,segments):
    out=np.full(len(obs),-np.inf)
    for i in range(len(obs)):
        minimum=np.inf
        for j in range(segments):
            a=dt*j/segments;b=np.nextafter(dt*(j+1)/segments,np.inf)
            T=(a,b);half=isc(iq(T),.5)
            rr=np.empty((3,2));ww=np.empty((3,2));ar=np.empty((3,2))
            for l in range(3):
                r=obs[i,l]-x[l];w=obs[i,l+3]-x[l+3];acc=obs[i,l+6]-u[l]
                re=np.nextafter(1.1*b**3/6.+delta*b*b/2.,np.inf)
                we=np.nextafter(1.1*b*b/2.+delta*b,np.inf)
                rr[l]=ia(ia(ia((r,r),isc(T,w)),isc(half,acc)),(-re,re))
                ww[l]=ia(ia((w,w),isc(T,acc)),(-we,we))
                ar[l]=ia((acc,acc),(-1.1*b,1.1*b))
            lower=evaluate(rr,ww,ar,par[21]+par[22]+obs[i,9],par[3],delta)
            if not np.isfinite(lower):minimum=-np.inf;break
            minimum=min(minimum,lower)
        out[i]=minimum
    return out
