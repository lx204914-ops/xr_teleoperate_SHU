"""One-sided interval certificate for held commands, unchanged RA barrier.

Bounds the derivative of F=Psi_nominal-(tracking+robust)*||B|| over
20 ms and the actual Cartesian command box. No future trajectory is read.
Every elementary interval operation is rounded outwards; no fastmath.
The controller still adds the SAME tracking*||B(t0)|| reserve separately.
"""
import numpy as np
from numba import njit

@njit(inline='always')
def down(x):return np.nextafter(x,-np.inf)
@njit(inline='always')
def up(x):return np.nextafter(x,np.inf)
@njit(inline='always')
def pt(x):return (x,x)
@njit(inline='always')
def add(a,b):return (down(a[0]+b[0]),up(a[1]+b[1]))
@njit(inline='always')
def neg(a):return (-a[1],-a[0])
@njit(inline='always')
def sub(a,b):return add(a,neg(b))
@njit(inline='always')
def mul(a,b):
 x=a[0]*b[0];y=a[0]*b[1];z=a[1]*b[0];w=a[1]*b[1]
 return (down(min(x,y,z,w)),up(max(x,y,z,w)))
@njit(inline='always')
def scale(c,a):return mul(pt(c),a)
@njit(inline='always')
def sq(a):
 hi=up(max(a[0]*a[0],a[1]*a[1]));lo=0. if a[0]<=0<=a[1] else down(min(a[0]*a[0],a[1]*a[1]))
 return (max(0.,lo),hi)
@njit(inline='always')
def div(a,b):
 if b[0]<=0<=b[1]:return (-np.inf,np.inf)
 return mul(a,(down(1./b[1]),up(1./b[0])))
@njit(inline='always')
def root(a):return (max(0.,down(np.sqrt(max(0.,a[0])))),up(np.sqrt(max(0.,a[1]))))
@njit(inline='always')
def vec_dot(a,b):
 v=pt(0.)
 for j in range(3):v=add(v,mul((a[j,0],a[j,1]),(b[j,0],b[j,1])))
 return v
@njit(inline='always')
def vec_norm(a):
 v=pt(0.)
 for j in range(3):v=add(v,sq((a[j,0],a[j,1])))
 return root(v)

@njit(cache=True)
def radius_tube(R,V,ref,omega,dt):
 # Gronwall-style algebraic bound on |Rdd|, no transcendental evaluation.
 w2=mul(pt(omega),pt(omega));a0=sub(mul(w2,sub(pt(ref),pt(R))),scale(2*omega,pt(V)))
 numerator=add(pt(max(abs(a0[0]),abs(a0[1]))),scale(dt,scale(abs(V),w2)))
 denominator=sub(sub(pt(1.),pt(2*omega*dt)),scale(dt*dt,w2))
 if denominator[0]<=0:raise ValueError('Radius-tube denominator not positive')
 ab=div(numerator,denominator)[1];ti=(0.,dt)
 vv=add(pt(V),(-up(dt*ab),up(dt*ab)))
 rr=add(add(pt(R),scale(V,ti)),(-up(.5*dt*dt*ab),up(.5*dt*dt*ab)))
 thirdmax=up(omega*omega*max(abs(vv[0]),abs(vv[1]))+2*omega*ab)
 aa=add(a0,(-up(dt*thirdmax),up(dt*thirdmax)))
 third=neg(add(scale(omega*omega,vv),scale(2*omega,aa)))
 return rr,vv,aa,third

@njit(cache=True)
def derivative_cell(r0,w0,ao0,R,V,ref,par,dt,ulo,uhi,tracking=.5,jerk=1.1):
 beta=par[1]/(par[0]*par[19]);kh=par[3];kk=1/par[0];cc=tracking+par[16]
 rr=np.empty((3,2));ww=np.empty((3,2));acc=np.empty((3,2));ti=(0.,dt);t2=(0.,up(dt*dt));t3=(0.,up(dt*dt*dt))
 ew=up(tracking*dt+.5*jerk*dt*dt);er=up(.5*tracking*dt*dt+jerk*dt*dt*dt/6)
 for j in range(3):
  a0=(down(ao0[j]-uhi[j]),up(ao0[j]-ulo[j]))
  rp=add(add(pt(r0[j]),scale(w0[j],ti)),scale(.5,mul(a0,t2)))
  wp=add(pt(w0[j]),mul(a0,ti))
  rr[j]=add(rp,(-er,er));ww[j]=add(wp,(-ew,ew))
  aot=(max(-.8,down(ao0[j]-jerk*dt)),min(.8,up(ao0[j]+jerk*dt)))
  acc[j]=sub(aot,(ulo[j],uhi[j]))
 DD=vec_norm(rr)
 if DD[0]<=1e-9:return -np.inf,np.inf
 ee=np.empty((3,2))
 for j in range(3):ee[j]=div((rr[j,0],rr[j,1]),DD)
 qq=vec_dot(ee,ww);zz=np.empty((3,2))
 for j in range(3):zz[j]=sub((ww[j,0],ww[j,1]),mul(qq,(ee[j,0],ee[j,1])))
 ZZ=pt(0.)
 for j in range(3):ZZ=add(ZZ,sq((zz[j,0],zz[j,1])))
 rad,vr,ar,third=radius_tube(R,V,ref,par[2],dt)
 ss=sub(DD,rad);vs=sub(qq,vr);invD=div(pt(1.),DD);invD2=sq(invD)
 K=add(pt(kk),scale(beta,ZZ));normg=root(add(pt(1.),scale(4*beta*beta,mul(sq(ss),ZZ))))
 Lq=add(sub(K,scale(2*beta,mul(mul(ss,ZZ),invD))),pt(kh))
 Lz=sub(add(add(sub(invD,scale(2*beta,mul(mul(ss,qq),invD))),scale(beta,vs)),scale(kh*beta,ss)),scale(2*cc*beta*beta,div(sq(ss),normg)))
 Cr=sub(Lq,scale(2*beta,mul(mul(ss,ZZ),invD)))
 Cz=add(add(scale(2.,Lz),invD),scale(2*beta,sub(vs,mul(mul(ss,qq),invD))))
 C0=sub(sub(mul(mul(Lq,ZZ),invD),scale(2.,mul(mul(mul(Lz,qq),ZZ),invD))),mul(mul(ZZ,qq),invD2))
 C0=sub(sub(C0,third),mul(K,ar))
 C0=sub(C0,scale(2*beta,mul(mul(mul(vs,qq),ZZ),invD)))
 C0=add(C0,scale(2*beta,mul(mul(mul(ss,sq(qq)),ZZ),invD2)))
 C0=add(C0,scale(kh,add(neg(ar),mul(K,vs))))
 C0=sub(C0,scale(4*cc*beta*beta,div(mul(mul(ss,vs),ZZ),normg)))
 a_r=vec_dot(ee,acc);a_z=vec_dot(zz,acc);ap=np.empty((3,2));grad=np.empty((3,2));ap2=pt(0.)
 for j in range(3):
  ap[j]=sub((acc[j,0],acc[j,1]),mul(a_r,(ee[j,0],ee[j,1])))
  ap2=add(ap2,sq((ap[j,0],ap[j,1])))
  grad[j]=add(add(mul(Lq,(ee[j,0],ee[j,1])),scale(2.,mul(Lz,(zz[j,0],zz[j,1])))),scale(2*beta,mul(ss,(ap[j,0],ap[j,1]))))
 nominal=add(add(add(C0,mul(Cr,a_r)),mul(Cz,a_z)),scale(2*beta,mul(ss,ap2)))
 result=sub(sub(nominal,scale(tracking,vec_norm(grad))),scale(jerk,normg))
 return result[0],ZZ[1]

@njit(cache=True)
def interval_margin_rows(x,obs,R,V,ref,par,B,C,dt=.02):
 n=len(obs);m=np.empty(n);zu=np.empty(n);low=np.empty(n);refined=0
 lo=np.full(3,-1.5);hi=np.full(3,1.5)
 for i in range(n):
  r=obs[i,:3]-x[:3];w=obs[i,3:6]-x[3:];a=obs[i,6:9]
  lb,z=derivative_cell(r,w,a,R[i],V[i],ref[i],par,dt,lo,hi)
  margin=up(dt*max(0.,-lb))+2e-7
  # Refine only nonredundant certificates. Every unrefined row remains enforced.
  if C[i]-margin-.5*np.linalg.norm(B[i])<1.5*np.sum(np.abs(B[i])):
   refined+=1;best=np.inf
   for mask in range(8):
    l=np.empty(3);h=np.empty(3)
    for j in range(3):
     if (mask>>j)&1:l[j]=0.;h[j]=1.5
     else:l[j]=-1.5;h[j]=0.
    bound,_=derivative_cell(r,w,a,R[i],V[i],ref[i],par,dt,l,h)
    best=min(best,bound)
   # The partitions cover the original command box. Either bound is valid.
   lb=max(lb,best);margin=up(dt*max(0.,-lb))+2e-7
  m[i]=margin;zu[i]=z;low[i]=lb
 return m,zu,low,refined
