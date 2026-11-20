"""A future-free bound on |d/dt(hdot+k_h*h)| under held u and Rref.

Requires exact current p,v,a, ||a_o||<=ao_bound and ||a_o'||<=jerk_bound.
The added L*dt margin retains the same barrier and makes sample-and-hold
implementation imply the continuous-time CBF condition, when QP is feasible.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def interval_margins(log,R,V,ref,par,dt,method,ao_bound=.8,jerk_bound=1.1,control_norm=np.sqrt(12.),robust_acceleration_only=False):
    beta=par[1]/(par[0]*par[19]);omega=par[2];kh=par[3]
    relative_acc=ao_bound+control_norm
    global_Rdd=2*omega*omega*par[5]
    margins=np.empty(len(log));z_upper=np.empty(len(log));Lvalues=np.empty(len(log))
    for i in range(len(log)):
        D=log[i,0];q=log[i,1];Z=log[i,2];s=log[i,6]
        W=np.sqrt(q*q+Z)+dt*relative_acc
        Dlo=D-dt*W;Dhi=D+dt*W
        if Dlo<=1e-9:
            margins[i]=np.inf;z_upper[i]=np.inf;Lvalues[i]=np.inf;continue
        z=np.sqrt(Z)+dt*(2*relative_acc+2*W*W/Dlo);z=min(z,W)
        zsq=z*z;qabs=min(W,abs(q)+dt*(zsq/Dlo+relative_acc))
        z_upper[i]=zsq
        if method==1:
            k=par[7]
            L=6*W*relative_acc+(0. if robust_acceleration_only else 2*Dhi*jerk_bound)+4*k*(W*W+Dhi*relative_acc)+2*k*k*Dhi*W
        else:
            rd=V[i];rdd=omega*omega*(ref[i]-R[i])-2*omega*V[i]
            if method==3 or method==7:rd=0.;rdd=0.
            vb=abs(rd)+dt*global_Rdd
            rthird=omega*omega*vb+2*omega*global_Rdd
            rddb=abs(rdd)+dt*rthird
            if method==3 or method==7:vb=0.;rthird=0.;rddb=0.
            sdot=qabs+vb;sabs=abs(s)+dt*sdot
            qdot=zsq/Dlo+relative_acc
            zdot=relative_acc+qdot+qabs*z/Dlo
            Zdot=2*z*relative_acc+2*qabs*zsq/Dlo
            K=1/par[0]+beta*zsq
            n=np.sqrt(1+4*beta*beta*sabs*sabs*zsq)
            ndot=z/Dlo+2*beta*(sdot*z+sabs*zdot)
            hdot=zsq/Dlo+rddb+K*sdot+2*beta*sabs*qabs*zsq/Dlo+n*relative_acc
            L=(Zdot/Dlo+zsq*qabs/(Dlo*Dlo)+rthird
               +beta*Zdot*sdot+K*(qdot+rddb)
               +2*beta*((sdot*qabs*zsq+sabs*qdot*zsq+sabs*qabs*Zdot)/Dlo+sabs*qabs*qabs*zsq/(Dlo*Dlo))
               +ndot*relative_acc+(0. if robust_acceleration_only else n*jerk_bound)+kh*hdot)
        Lvalues[i]=L;margins[i]=dt*L+2e-7
    return margins,z_upper,Lvalues
