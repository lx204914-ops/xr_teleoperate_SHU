"""Independent nonperiodic C2 obstacle motion. Geometry comes from scene_tapes.py.
Acceleration is slew limited; obstacle steering does not use the robot state.
"""
import numpy as np
from numba import njit
VERSION='continuous_manoeuvres_reference_initial_states_20261016'
TAPE_DT=.01
ACC_LIMIT=.72
JERK_LIMIT=1.1

def manoeuvre_schedule(n,duration,seed):
    commands=np.zeros((int(np.ceil(duration/TAPE_DT))+1,n,3))
    counts=[]
    for i in range(n):
        rng=np.random.default_rng(np.random.SeedSequence([seed,31011,i]))
        t=0.;count=0
        while t<duration+TAPE_DT:
            hold=rng.uniform(1.3,3.8)
            direction=rng.normal(size=3);direction[2]*=.65
            direction/=np.linalg.norm(direction)
            speed=rng.uniform(.3,.95)
            begin=int(t/TAPE_DT);end=min(len(commands),int((t+hold)/TAPE_DT)+1)
            commands[begin:end,i]=speed*direction
            t+=hold;count+=1
        counts.append(count)
    return commands,np.array(counts)


@njit(cache=True)
def build_tape(p0,v0,radii,workspace,commands):
    n=len(radii);dt=TAPE_DT;steps=len(commands)-1
    tape=np.zeros((steps+1,n,10));p=p0.copy();v=v0.copy();a=np.zeros((n,3))
    for k in range(steps+1):
        tape[k,:,:3]=p;tape[k,:,3:6]=v;tape[k,:,6:9]=a;tape[k,:,9]=radii
        if k==steps:break
        target=np.zeros((n,3))
        for i in range(n):
            # Keep initial crossing motion before independent random manoeuvres.
            blend=min(1.,max(0.,(k*dt-5.)/2.))
            desired=(1.-blend)*v0[i]+blend*commands[k,i]
            target[i]=.85*(desired-v[i])
            for j in range(3):
                lo=workspace[j,0]+radii[i]+.25;hi=workspace[j,1]-radii[i]-.25
                lower=p[i,j]-lo;upper=hi-p[i,j]
                stopping=.4+v[i,j]**2/(2*.42)
                if v[i,j]<0 and lower<stopping+1.2:
                    target[i,j]+=.85*max(0.,(stopping+1.2-lower)/(stopping+1.2))
                if v[i,j]>0 and upper<stopping+1.2:
                    target[i,j]-=.85*max(0.,(stopping+1.2-upper)/(stopping+1.2))
                if lower<.5:target[i,j]+=.85*(.5-lower)
                if upper<.5:target[i,j]-=.85*(.5-upper)
        for i in range(n):
            for j in range(i+1,n):
                r=p[j]-p[i];distance=np.sqrt(np.dot(r,r));e=r/max(distance,1e-9)
                relative=v[j]-v[i];closing=max(0.,-np.dot(relative,e))
                threshold=radii[i]+radii[j]+1.+closing*1.3+closing*closing/(2*.5)
                if distance<threshold:
                    strength=1.2*min(2.,(threshold-distance)/max(.5,threshold-radii[i]-radii[j]))
                    target[i]-=strength*e;target[j]+=strength*e
        for i in range(n):
            length=np.sqrt(np.dot(target[i],target[i]))
            if length>ACC_LIMIT:target[i]*=ACC_LIMIT/length
            change=(target[i]-a[i])*(1.-np.exp(-dt/.35))
            change_norm=np.sqrt(np.dot(change,change))
            if change_norm>JERK_LIMIT*dt:change*=JERK_LIMIT*dt/change_norm
            jerk=change/dt
            p[i]+=dt*v[i]+.5*dt*dt*a[i]+dt**3/6.*jerk
            v[i]+=dt*a[i]+.5*dt*dt*jerk
            a[i]+=change
    return tape

