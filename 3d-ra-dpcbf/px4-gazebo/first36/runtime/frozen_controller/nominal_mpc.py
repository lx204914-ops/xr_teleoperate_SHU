"""Obstacle-free finite-horizon, box-constrained double-integrator MPC.

The reference approaches the goal at the declared cruise speed. Future obstacle
states are never supplied. Three axes share the same condensed temporal Hessian.
"""
import numpy as np
from numba import njit

def matrices(horizon=12,dt=.12,position_weight=2.,velocity_weight=6.,
             input_weight=.20,increment_weight=.35):
    n=horizon;P=np.zeros((n,n));V=np.zeros((n,n))
    for l in range(n):
        for j in range(l+1):
            P[l,j]=dt*dt*(l-j+.5);V[l,j]=dt
    difference=np.eye(n)
    for j in range(1,n):difference[j,j-1]=-1.
    H=position_weight*P.T@P+velocity_weight*V.T@V+input_weight*np.eye(n)+increment_weight*difference.T@difference
    step=1/np.linalg.eigvalsh(H)[-1]
    return H,P,V,float(step),np.arange(1,n+1)*dt,np.array([position_weight,velocity_weight,increment_weight])

@njit(cache=True)
def solve(x,goal,cruise,previous,warm,H,P,V,step,times,weights,limit=2.):
    delta=goal-x[:3];dist=np.sqrt(np.dot(delta,delta));direction=delta/max(dist,1e-12)
    desired=cruise*np.tanh(dist/1.6)*direction
    n=len(times);ep=np.empty((n,3));ev=np.empty((n,3))
    for l in range(n):
        ref=x[:3]+min(dist,times[l]*np.linalg.norm(desired))*direction
        ep[l]=x[:3]+times[l]*x[3:]-ref;ev[l]=x[3:]-desired
    f=weights[0]*P.T@ep+weights[1]*V.T@ev;f[0]-=weights[2]*previous
    U=warm.copy();Y=U.copy();momentum=1.;residual=1e20;iterations=0
    for iteration in range(600):
        grad=H@Y+f
        nextU=np.minimum(limit,np.maximum(-limit,Y-step*grad))
        next_momentum=.5*(1+np.sqrt(1+4*momentum*momentum))
        Y=nextU+(momentum-1)/next_momentum*(nextU-U);U=nextU;momentum=next_momentum
        if iteration%10==0:
            grad=H@U+f
            residual=np.max(np.abs(U-np.minimum(limit,np.maximum(-limit,U-step*grad))))/step
            if residual<1e-6:iterations=iteration+1;break
        iterations=iteration+1
    grad=H@U+f
    residual=np.max(np.abs(U-np.minimum(limit,np.maximum(-limit,U-step*grad))))/step
    return U[0].copy(),U,iterations,residual
