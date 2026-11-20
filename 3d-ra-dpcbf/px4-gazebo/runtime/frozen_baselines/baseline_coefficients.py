"""Explicit 3D adaptations, separate from the unchanged proposed controller.

VO: union of four tangent halfspaces per spherical obstacle (two perpendicular
planes); exhaustive, unbudgeted branch QPs. Not the original unicycle MIQP.
C3BF: r.w + sqrt(D^2-R0^2)*||w||, including obstacle acceleration.
Park-inspired: q + kl*sqrt(D^2-R0^2)*Z/||w|| + km*sqrt(D^2-R0^2).
Full chain derivatives of kl*d/speed are included; no frozen coefficients.
"""
import numpy as np
from numba import njit
from dynamic_study.kernel import constraints as main_constraints,project_qp

KL=.1
KM=.5

@njit(cache=True)
def coefficients(x,obs,R,V,ref,par,method):
    if method<8:
        return main_constraints(x,obs,R,V,ref,par,method)
    B,C,log=main_constraints(x,obs,R,V,ref,par,3)
    for i in range(len(obs)):
        r=obs[i,:3]-x[:3];w=obs[i,3:6]-x[3:];D=np.linalg.norm(r)
        e=r/D;q=e@w;z=w-q*e;Z=z@z
        rad=par[21]+par[22]+obs[i,9]
        d=np.sqrt(max(D*D-rad*rad,1e-12));speed=np.sqrt(w@w+1e-12)
        lam=KL*d/speed;mu=KM*d
        h=q+lam*Z+mu
        grad_r=z/D+(KL*Z/speed+KM)*(D/d)*e-2*lam*q*z/D
        grad_w=e+2*lam*z-KL*d*Z*w/speed**3
        B[i]=-grad_w;C[i]=grad_r@w+grad_w@obs[i,6:9]+par[3]*h
        log[i,7]=h;log[i,8]=C[i];log[i,9:12]=B[i]
    return B,C,log

@njit(cache=True)
def vo_planes(x,obs,par):
    nobs=len(obs);B=np.zeros((nobs,4,3));C=np.zeros((nobs,4));h=np.zeros((nobs,4))
    for i in range(nobs):
        r=obs[i,:3]-x[:3];w=obs[i,3:6]-x[3:];D=np.linalg.norm(r)
        e=r/D;q=e@w;z=w-q*e;P=np.eye(3)-np.outer(e,e)
        rad=par[21]+par[22]+obs[i,9];a=rad/D;b=np.sqrt(max(1-a*a,1e-12))
        for plane in range(2):
            axis=np.zeros(3);axis[2 if plane==0 else 1]=1.
            raw=P@axis
            if np.linalg.norm(raw)<.05:axis=np.array([1.,0.,0.]);raw=P@axis
            length=np.linalg.norm(raw);normal=raw/length
            Jraw=-(e@axis)*P/D-np.outer(e,axis)@P/D
            Jn=(np.eye(3)-np.outer(normal,normal))@Jraw/length
            for side in range(2):
                sgn=1. if side==0 else -1.;n=sgn*normal;Jnside=sgn*Jn;j=2*plane+side
                hv=a*q+b*(n@w)
                gr=a*z/D+(-a*q/D+rad*rad/(b*D**3)*(n@w))*e+b*Jnside.T@w
                gw=a*e+b*n
                B[i,j]=-gw;C[i,j]=gr@w+gw@obs[i,6:9]+par[3]*hv;h[i,j]=hv
    return B,C,h

@njit(cache=True)
def vo_hold_margins(x,obs,par,dt):
    """Future-free bound for f=hdot+kh*h, h=g(r).w, ||g||=1.

    With relative acceleration A and obstacle jerk J, |f'| is bounded by
    H_g W^3 + 3 J_g W A + J + kh*(J_g W^2 + A).
    The projected-axis normalization satisfies ||Dn||<=2/(D*l) and
    ||D2n||<=8/(D^2*l)+12/(D^2*l^2). Radial derivatives of a=R/D,
    b=sqrt(1-a^2) give the conservative J_g,H_g bounds below.
    Assumes the selected projection axis does not switch within a hold interval;
    this scene's axis normalizations are independently audited away from .05.
    """
    result=np.zeros((len(obs),4));A=np.sqrt(12.)+.8
    for i in range(len(obs)):
        r=obs[i,:3]-x[:3];w=obs[i,3:6]-x[3:];D=np.linalg.norm(r);e=r/D
        W=np.linalg.norm(w)+dt*A;lo=D-dt*W;rad=par[21]+par[22]+obs[i,9]
        if lo<=rad:
            result[i,:]=np.inf;continue
        a=rad/lo;b=np.sqrt(1-a*a)
        for plane in range(2):
            axis=np.zeros(3);axis[2 if plane==0 else 1]=1.
            length=np.sqrt(max(0.,1-(axis@e)**2))
            if length<.05:
                axis=np.array([1.,0.,0.]);length=np.sqrt(max(0.,1-(axis@e)**2))
            l=length-2*dt*W/lo
            if l<=0:
                result[i,2*plane:2*plane+2]=np.inf;continue
            Jg=(2*a+a*a/b+2/l)/lo
            Hg=(7*a+3*a*a/b+a**4/b**3+8/l+12/l**2+4*a*a/(b*l))/lo**2
            L=Hg*W**3+3*Jg*W*A+1.1+par[3]*(Jg*W*W+A)
            result[i,2*plane:2*plane+2]=dt*L+2e-7
    return result

@njit(cache=True)
def choose_vo(target,BB,CC,hh):
    """Exact exhaustive QP choice over the declared 4^N branch combinations.
    Switch only to branches whose current h is nonnegative, preserving the
    selected branch's initial condition; no arbitrary computation timeout.
    """
    n=len(BB);best=np.zeros(3);bestcost=1e100;bestids=np.zeros(n,np.int64);calls=0
    B=np.zeros((n,3));C=np.zeros(n);ids=np.zeros(n,np.int64)
    for combination in range(4**n):
        number=combination;valid=True
        for i in range(n):
            j=number%4;number//=4;ids[i]=j
            if hh[i,j]<-2e-7:valid=False;break
            B[i]=BB[i,j];C[i]=CC[i,j]
        if not valid:continue
        calls+=1;u,ok,it=project_qp(target,B,C)
        cost=np.sum((u-target)**2)
        if ok and cost<bestcost-1e-12:best=u;bestcost=cost;bestids=ids.copy()
    return best,bestcost<1e99,bestids,calls

