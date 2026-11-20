"""Acceleration-interface calibration ONLY, not a navigation controller."""
import numpy as np
from shared_execution_box import bounds

def command(t,previous,dt,tau):
 # Exercise both horizontal directions at the existing nominal authority .8.
 target=np.array([.8 if int(t/.5)%2==0 else -.8,
                  .8 if int((t+.25)/.5)%2==0 else -.8,
                  .3*np.cos(np.pi*t)])
 lo,hi,_=bounds(previous,dt,tau)
 return np.minimum(hi,np.maximum(lo,target))
