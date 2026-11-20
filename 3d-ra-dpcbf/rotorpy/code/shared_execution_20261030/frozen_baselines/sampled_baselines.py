"""Future-free sample-and-hold certificates for the unchanged C3/Park barriers.

For a fixed input, psi = h_dot + kh*h and psi_dot = h_ddot + kh*h_dot.
Outward-rounded interval second-order differentiation encloses psi_dot for
EVERY input in [-2,2]^3 throughout [t,t+dt]. The current r,w,ao and known
||ao||<=.8, ||jo||<=1.1 bounds alone form the reachable tube. No obstacle tape,
future samples, finite-difference estimates or empirically fitted factors enter.
Enforcing B@u+C >= dt*max(0,-psi_dot_interval.lower)+2e-7 therefore gives
psi(t+tau,u)>=2e-7 (up to the declared numerical QP tolerance) on the hold.
The baseline equations, including all derivatives of Park's lambda, are fixed.
No certificate is claimed if the tube intersects D<=R0 or arithmetic overflows;
this is a certificate-domain failure, not a proof of physical infeasibility.

Interval jets store [value, first time derivative, second time derivative],
each as [lower, upper]. Natural interval extensions deliberately overbound
correlated quantities. Their conservatism is not a method-performance claim.
"""
import numpy as np
from numba import njit
from baseline_coefficients import KL,KM

A_OBS=.8
J_OBS=1.1
U_COMPONENT=2.
QP_PAD=2e-7

@njit(cache=True)
def add(a,b):
    return np.array([np.nextafter(a[0]+b[0],-np.inf),np.nextafter(a[1]+b[1],np.inf)])

@njit(cache=True)
def neg(a):
    return np.array([-a[1],-a[0]])

@njit(cache=True)
def sub(a,b):return add(a,neg(b))

@njit(cache=True)
def mul(a,b):
    vals=np.array([a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]])
    return np.array([np.nextafter(np.min(vals),-np.inf),np.nextafter(np.max(vals),np.inf)])

@njit(cache=True)
def square(a):
    lo=0. if a[0]<=0.<=a[1] else min(a[0]*a[0],a[1]*a[1])
    hi=max(a[0]*a[0],a[1]*a[1])
    return np.array([max(0.,np.nextafter(lo,-np.inf)),np.nextafter(hi,np.inf)])

@njit(cache=True)
def reciprocal(a):
    if a[0]<=0.<=a[1]:return np.array([-np.inf,np.inf])
    return np.array([np.nextafter(1./a[1],-np.inf),np.nextafter(1./a[0],np.inf)])

@njit(cache=True)
def scale(a,c):return mul(a,np.array([c,c]))

@njit(cache=True)
def constant(c):
    j=np.zeros((3,2));j[0,:]=c;return j

@njit(cache=True)
def jadd(a,b):
    out=np.empty((3,2))
    for k in range(3):out[k]=add(a[k],b[k])
    return out

@njit(cache=True)
def jneg(a):
    out=np.empty((3,2))
    for k in range(3):out[k]=neg(a[k])
    return out

@njit(cache=True)
def jsub(a,b):return jadd(a,jneg(b))

@njit(cache=True)
def jmul(a,b):
    out=np.empty((3,2));out[0]=mul(a[0],b[0])
    out[1]=add(mul(a[1],b[0]),mul(a[0],b[1]))
    out[2]=add(add(mul(a[2],b[0]),scale(mul(a[1],b[1]),2.)),mul(a[0],b[2]))
    return out

@njit(cache=True)
def jsquare(a):
    out=np.empty((3,2));out[0]=square(a[0]);out[1]=scale(mul(a[0],a[1]),2.)
    out[2]=scale(add(square(a[1]),mul(a[0],a[2])),2.)
    return out

@njit(cache=True)
def compose(a,value,df,ddf):
    out=np.empty((3,2));out[0]=value;out[1]=mul(df,a[1])
    out[2]=add(mul(ddf,square(a[1])),mul(df,a[2]));return out

@njit(cache=True)
def jinv(a):
    v=reciprocal(a[0]);df=neg(square(v));ddf=scale(mul(square(v),v),2.)
    return compose(a,v,df,ddf)

@njit(cache=True)
def jsqrt(a):
    v=np.array([np.nextafter(np.sqrt(a[0,0]),-np.inf),np.nextafter(np.sqrt(a[0,1]),np.inf)])
    inv=reciprocal(v);df=scale(inv,.5);ddf=scale(mul(square(inv),inv),-.25)
    return compose(a,v,df,ddf)

@njit(cache=True)
def tube_jets(r,w,ao,dt):
    """Uniform tube for all fixed admissible u, including endpoints."""
    rj=np.empty((3,3,2));wj=np.empty((3,3,2))
    T=np.array([0.,np.nextafter(dt,np.inf)])
    halfT2=scale(square(T),.5)
    for l in range(3):
        rp=np.array([np.nextafter(r[l],-np.inf),np.nextafter(r[l],np.inf)])
        wp=np.array([np.nextafter(w[l],-np.inf),np.nextafter(w[l],np.inf)])
        ao_point=np.array([ao[l],ao[l]])
        ao_range=add(ao_point,mul(np.array([-J_OBS,J_OBS]),T))
        ao_range[0]=max(-A_OBS,ao_range[0]);ao_range[1]=min(A_OBS,ao_range[1])
        ar=add(ao_range,np.array([-U_COMPONENT,U_COMPONENT]))
        wr=add(wp,mul(ar,T))
        rr=add(rp,add(mul(wp,T),mul(ar,halfT2)))
        rj[l,0]=rr;rj[l,1]=wr;rj[l,2]=ar
        wj[l,0]=wr;wj[l,1]=ar;wj[l,2]=np.array([-J_OBS,J_OBS])
    return rj,wj

@njit(cache=True)
def barrier_jet(rj,wj,R0,method):
    d2=constant(0.);s2=constant(1e-12);rw=constant(0.)
    for l in range(3):
        d2=jadd(d2,jsquare(rj[l]));s2=jadd(s2,jsquare(wj[l]));rw=jadd(rw,jmul(rj[l],wj[l]))
    l2=jsub(d2,constant(R0*R0))
    if l2[0,0]<=0. or d2[0,0]<=0. or s2[0,0]<=0.:return np.full((3,2),np.nan)
    L=jsqrt(l2);S=jsqrt(s2)
    if method==2:return jadd(rw,jmul(L,S))
    D=jsqrt(d2);q=jmul(rw,jinv(D))
    Z=jsub(jsub(s2,constant(1e-12)),jsquare(q))
    return jadd(jadd(q,jmul(constant(KL),jmul(jmul(L,Z),jinv(S)))),jmul(constant(KM),L))

@njit(cache=True)
def c3_psi_derivative_lower(rj,wj,R0,ao,kh,dt):
    """Sharper one-sided bound, retaining the C3 velocity Hessian sign.

    H_ww = L*(I/S - w w^T/S^3) is PSD (S^2=w.w+eps^2).
    Dropping only a^T H_ww a gives
      psi_dot >= c0 + ca.a + gw.j,
      c0=S*w2/L-S*(r.w)^2/L^3+kh*(w2+S*(r.w)/L),
      ca=3w+S*r/L+2*(r.w)*w/(L*S)+kh*(r+L*w/S),
      gw=r+L*w/S.
    State intervals bound these coefficients. The control box support function
    and ||ao(t)-ao(0)||<=J*dt, ||j||<=J bound the last two terms. This keeps
    the nonnegative curvature term nonnegative rather than interval-subtracting
    its two individually large components. It does not change the barrier.
    """
    d2=np.array([0.,0.]);w2=np.array([0.,0.]);rw=np.array([0.,0.])
    for l in range(3):
        d2=add(d2,square(rj[l,0]));w2=add(w2,square(wj[l,0]));rw=add(rw,mul(rj[l,0],wj[l,0]))
    l2=sub(d2,np.array([R0*R0,R0*R0]));s2=add(w2,np.array([1e-12,1e-12]))
    L=np.array([np.nextafter(np.sqrt(l2[0]),-np.inf),np.nextafter(np.sqrt(l2[1]),np.inf)])
    S=np.array([np.nextafter(np.sqrt(s2[0]),-np.inf),np.nextafter(np.sqrt(s2[1]),np.inf)])
    il=reciprocal(L);ispeed=reciprocal(S);il3=mul(square(il),il)
    c0=add(sub(mul(mul(S,w2),il),mul(mul(S,square(rw)),il3)),scale(add(w2,mul(mul(S,rw),il)),kh))
    aopart=np.array([0.,0.]);l1=np.array([0.,0.]);a2=np.array([0.,0.]);g2=np.array([0.,0.])
    for l in range(3):
        r=rj[l,0];w=wj[l,0]
        gw=add(r,mul(mul(L,w),ispeed))
        ca=add(add(scale(w,3.),mul(mul(S,r),il)),add(scale(mul(mul(mul(rw,w),il),ispeed),2.),scale(gw,kh)))
        aopart=add(aopart,scale(ca,ao[l]))
        ca_abs=max(abs(ca[0]),abs(ca[1]));gw_abs=max(abs(gw[0]),abs(gw[1]))
        l1=add(l1,np.array([ca_abs,ca_abs]))
        a2=add(a2,square(np.array([ca_abs,ca_abs])));g2=add(g2,square(np.array([gw_abs,gw_abs])))
    anorm=np.nextafter(np.sqrt(a2[1]),np.inf);gnorm=np.nextafter(np.sqrt(g2[1]),np.inf)
    loss=add(scale(l1,U_COMPONENT),add(scale(np.array([anorm,anorm]),J_OBS*np.nextafter(dt,np.inf)),scale(np.array([gnorm,gnorm]),J_OBS)))
    return sub(add(c0,aopart),loss)[0]


@njit(cache=True)
def hold_certificates(x,obs,par,dt,method):
    """Return margins and [psi_dot_lo,psi_dot_hi,D_tube_lo,S_tube_lo]."""
    if method!=2 and method!=8:raise ValueError('Only unchanged C3BF/Park formulas are certified here.')
    n=len(obs);margins=np.full(n,np.inf);details=np.full((n,4),np.nan)
    for i in range(n):
        r=obs[i,:3]-x[:3];w=obs[i,3:6]-x[3:];ao=obs[i,6:9]
        R0=par[21]+par[22]+obs[i,9]
        if np.linalg.norm(ao)>A_OBS+1e-10:continue
        rj,wj=tube_jets(r,w,ao,dt);h=barrier_jet(rj,wj,R0,method)
        derivative=add(h[2],scale(h[1],par[3]))
        if not np.isfinite(derivative).all():continue
        if method==2:
            derivative[0]=max(derivative[0],c3_psi_derivative_lower(rj,wj,R0,ao,par[3],dt))
        # Only the most negative time derivative can violate psi>=0.
        magnitude=max(0.,-derivative[0])
        margins[i]=np.nextafter(dt*magnitude+QP_PAD,np.inf)
        d2=0.;s2=1e-12
        for l in range(3):d2+=square(rj[l,0])[0];s2+=square(wj[l,0])[0]
        details[i]=np.array([derivative[0],derivative[1],np.sqrt(d2),np.sqrt(s2)])
    return margins,details
