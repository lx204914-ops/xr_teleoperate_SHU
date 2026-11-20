"""Numerical kernels for the current all-obstacle linear-clearance RA-DPCBF.

The new simulation uses MPC in nominal_mpc.py. Historical biased guidance,
old scenario integrators and state-reset code are deliberately not entry points.
Method numbers 0/1/2/3/4/5/6/7 identify explicit optional ablations; main is 5.
"""
from pathlib import Path
import json
import numpy as np
from numba import njit
DEFAULT=np.array(json.loads((Path(__file__).resolve().parents[1]/'method_config.json').read_text(encoding='utf8'))['parameters'],dtype=float)
PARAM_NAMES=['Tr','gamma','omega','kh','tau','span','nu','distance_gain','unused_guide','cruise','unused_velocity_gain','unused_delay','unused_disturbance','unused_pos_error','unused_vel_error','unused_acc_error','robust_margin','fixed_radius_extra','initial_radius_fraction','Zmax','prediction_margin','robot_radius','base_margin']
METHODS={0:'nominal',1:'distance_hocbf',2:'collision_cone',3:'fixed_radius_ablation',4:'dynamic_distance',5:'ra_dpcbf',6:'omit_Rdot_ablation',7:'fixed_large_radius_ablation'}


@njit(cache=True)
def radius_ref(q, base, par):
    c=.5*(np.sqrt(q*q+.03**2)-q)
    prediction_margin=par[20] if len(par)>20 else .04
    desired=base+par[4]*c+c*c/(2*1.2)+prediction_margin
    return min(max(desired,base),base+par[5]),desired


@njit(cache=True)
def boundary_flow(R,V,ref,dt,omega):
    decay=np.exp(-omega*dt)
    for i in range(len(R)):
        y=R[i]-ref[i]; b=V[i]+omega*y
        R[i]=ref[i]+(y+b*dt)*decay
        V[i]=(V[i]-omega*b*dt)*decay


@njit(cache=True)
def constraints(x,os,R,V,ref,par,method):
    n=len(os); B=np.empty((n,3)); C=np.empty(n)
    # D,q,Z,R,Rdot,Rref,s,h,c,Bx,By,Bz,residual,Rdes,z1,z2
    log=np.empty((n,16))
    # 19-column vectors are archived pre-correction runs with Zscale=4.
    # Canonical runs always provide the explicit twentieth parameter.
    z_scale=par[19] if len(par)>19 else 4.
    beta=par[1]/(par[0]*z_scale)
    if method==4: beta=0.
    for i in range(n):
        r=os[i,:3]-x[:3]; w=os[i,3:6]-x[3:]
        D=max(np.sqrt(np.sum(r*r)),1e-12); e=r/D
        q=np.dot(e,w); z=w-q*e; Z=np.dot(z,z)
        base=(par[21]+par[22] if len(par)>22 else .4)+os[i,9]
        radius=R[i]; rd=V[i]; rdd=par[2]**2*(ref[i]-R[i])-2*par[2]*V[i]
        if method==3 or method==7:
            radius=base+(par[17] if method==7 else 0.); rd=0.; rdd=0.
        if method==1 or method==2 or method==0:
            radius=base; rd=0.; rdd=0.
        s=D-radius
        if method==1:
            H=D*D-base*base; Hd=2*np.dot(r,w); k=par[7]
            h=Hd+k*H
            B[i]=-2*r
            C[i]=2*np.dot(w,w)+2*np.dot(r,os[i,6:9])+2*k*Hd+k*k*H
        elif method==2:
            # Smooth speed only, with exact derivative of that regularization.
            d=np.sqrt(max(D*D-base*base,1e-14))
            speed=np.sqrt(np.dot(w,w)+1e-12)
            h=np.dot(r,w)+d*speed
            B[i]=-r-d*w/speed
            A=np.dot(w,w)+np.dot(r,os[i,6:9])+D*q/d*speed+d*np.dot(w,os[i,6:9])/speed
            C[i]=A+par[3]*h
        else:
            K=1/par[0]+beta*Z
            h=q-(0. if method==6 else rd)+K*s
            B[i]=-e-2*beta*s*z
            A=Z/D+np.dot(e,os[i,6:9])-(0. if method==6 else rdd)+K*(q-rd)
            A+=2*beta*s*np.dot(z,os[i,6:9])-2*beta*s*q*Z/D
            C[i]=A+par[3]*h
        C[i]-=par[16]*np.sqrt(np.sum(B[i]*B[i]))
        _,des=radius_ref(q,base,par)
        # Stable LOS basis (all diagnostics use the same definition).
        axis=np.zeros(3); axis[np.argmin(np.abs(e))]=1.
        b1=np.cross(e,axis); b1/=np.sqrt(np.dot(b1,b1)); b2=np.cross(e,b1)
        log[i,:9]=np.array([D,q,Z,radius,rd,ref[i],s,h,C[i]])
        log[i,9:12]=B[i]; log[i,12]=np.nan; log[i,13]=des
        log[i,14]=np.dot(z,b1); log[i,15]=np.dot(z,b2)
    return B,C,log


@njit(cache=True)
def feasible(u,A,b,tol=2e-8):
    for i in range(len(b)):
        if np.dot(A[i],u)-b[i]<-tol: return False
    return True


@njit(cache=True)
def project_qp(target,B,C):
    """Exact 3D strictly convex projection; all halfspaces are always checked.

    Coordinate dual iteration first. If it fails KKT, enumerate independent
    active sets (1/2/3 constraints). Feasible nonnegative multipliers certify
    optimality. A polyhedron projection has such a set in R^3.
    """
    n=len(C); m=n+6; A=np.zeros((m,3)); b=np.zeros(m)
    for i in range(n):
        scale=np.sqrt(np.sum(B[i]*B[i]))
        if scale<1e-14:
            if C[i]<-1e-10: return target.copy(),False,0
            scale=1.
        A[i]=B[i]/scale; b[i]=-C[i]/scale
    for j in range(3):
        A[n+2*j,j]=1.; A[n+2*j+1,j]=-1.
        b[n+2*j]=-2.; b[n+2*j+1]=-2.
    if feasible(target,A,b): return target.copy(),True,0
    u=target.copy(); lam=np.zeros(m)
    for sweep in range(120):
        for i in range(m):
            change=max(0.,lam[i]+b[i]-np.dot(A[i],u))-lam[i]
            u+=change*A[i]; lam[i]+=change
        err=0.
        for i in range(m):
            res=np.dot(A[i],u)-b[i]
            err=max(err,max(-res,abs(res) if lam[i]>1e-10 else 0.))
        if err<2e-9: return u,True,sweep+1
        if np.max(lam)>1e6: break
    # Single active constraint.
    for i in range(m):
        li=b[i]-np.dot(A[i],target)
        if li<0: continue
        u=target+li*A[i]
        if feasible(u,A,b): return u,True,121
    # Two active constraints.
    for i in range(m):
        yi=b[i]-np.dot(A[i],target)
        for j in range(i):
            d=np.dot(A[i],A[j]); det=1-d*d
            if det<1e-12: continue
            yj=b[j]-np.dot(A[j],target)
            li=(yi-d*yj)/det; lj=(yj-d*yi)/det
            if li< -1e-9 or lj< -1e-9: continue
            u=target+li*A[i]+lj*A[j]
            if feasible(u,A,b): return u,True,122
    # Three independent normals; cross products avoid a general inverse.
    for i in range(m):
        for j in range(i):
            for k in range(j):
                crossjk=np.cross(A[j],A[k]); det=np.dot(A[i],crossjk)
                if abs(det)<1e-10: continue
                crosski=np.cross(A[k],A[i]); crossij=np.cross(A[i],A[j])
                u=(b[i]*crossjk+b[j]*crosski+b[k]*crossij)/det
                delta=u-target
                li=np.dot(delta,crossjk)/det
                lj=np.dot(delta,crosski)/det
                lk=np.dot(delta,crossij)/det
                if min(li,lj,lk)<-1e-8: continue
                if feasible(u,A,b): return u,True,123
    return np.zeros(3),False,124

