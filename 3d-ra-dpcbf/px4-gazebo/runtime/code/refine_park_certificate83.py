"""Refine only uncertified rows; final all-obstacle acceptance unchanged."""
import numpy as np
from certificate_pool import candidate_lower

def certified_lower(x,obs,par,u,dt,delta):
    lower=candidate_lower(x,obs,par,u,dt,delta,4)
    for segments in (8,16,32):
        bad=np.flatnonzero(lower<2e-7)
        if not len(bad):break
        lower[bad]=candidate_lower(x,obs[bad],par,u,dt,delta,segments)
    return lower
