"""Exact IEEE-754 binary64 nextafter(x, +/-inf), without general libm dispatch."""
import numpy as np
from numba import njit

@njit(cache=True)
def _nan_mask(bits):
    x=np.uint64(bits).view(np.float64)
    return np.float64(np.nextafter(x,np.inf)).view(np.uint64)^bits

NAN_MASK=_nan_mask(np.uint64(0x7ff0000000000001))
assert NAN_MASK in (np.uint64(0),np.uint64(0x0008000000000000))

@njit(inline='always')
def up(x):
    bits=np.float64(x).view(np.uint64)
    if np.isnan(x):return np.uint64(bits|NAN_MASK).view(np.float64)
    if x==np.inf:return x
    if x==0.:return np.uint64(1).view(np.float64)
    if x>0.:return np.uint64(bits+np.uint64(1)).view(np.float64)
    return np.uint64(bits-np.uint64(1)).view(np.float64)

@njit(inline='always')
def down(x):
    bits=np.float64(x).view(np.uint64)
    if np.isnan(x):return np.uint64(bits|NAN_MASK).view(np.float64)
    if x==-np.inf:return x
    if x==0.:return np.uint64(0x8000000000000001).view(np.float64)
    if x>0.:return np.uint64(bits-np.uint64(1)).view(np.float64)
    return np.uint64(bits+np.uint64(1)).view(np.float64)
