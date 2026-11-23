"""Allocation-free interval evaluation of the exact Park gradient and CBF."""
import numpy as np
from bit_round_helpers75 import up, down
from numba import njit

@njit(inline='always')
def ia(a, b):
    return (down(a[0] + b[0]), up(a[1] + b[1]))

@njit(inline='always')
def ine(a):
    return (-a[1], -a[0])

@njit(inline='always')
def isu(a, b):
    return ia(a, ine(b))

@njit(inline='always')
def im(a, b):
    p = a[0] * b[0]
    q = a[0] * b[1]
    r = a[1] * b[0]
    s = a[1] * b[1]
    return (down(min(p, q, r, s)), up(max(p, q, r, s)))

@njit(inline='always')
def isc(a, s):
    return im(a, (s, s))

@njit(inline='always')
def iq(a):
    low = 0.0 if a[0] <= 0.0 <= a[1] else min(a[0] * a[0], a[1] * a[1])
    return (max(0.0, down(low)), up(max(a[0] * a[0], a[1] * a[1])))

@njit(inline='always')
def ir(a):
    if a[0] <= 0.0 <= a[1]:
        return (-np.inf, np.inf)
    return (down(1.0 / a[1]), up(1.0 / a[0]))

@njit(inline='always')
def isq(a):
    if a[0] <= 0:
        return (np.nan, np.nan)
    return (down(np.sqrt(a[0])), up(np.sqrt(a[1])))

@njit(cache=True)
def evaluate(rr, ww, ar, R0, kh, delta):
    d2 = ia(ia(iq(rr[0]), iq(rr[1])), iq(rr[2]))
    s2 = ia(ia(ia(iq(ww[0]), iq(ww[1])), iq(ww[2])), (1e-12, 1e-12))
    D = isq(d2)
    L = isq(isu(d2, (R0 * R0, R0 * R0)))
    S = isq(s2)
    if not np.isfinite(L[0]) or not np.isfinite(S[0]):
        return -np.inf
    iD = ir(D)
    iL = ir(L)
    iS = ir(S)
    iS3 = im(iq(iS), iS)
    e = (im(rr[0], iD), im(rr[1], iD), im(rr[2], iD))
    q = ia(ia(im(e[0], ww[0]), im(e[1], ww[1])), im(e[2], ww[2]))
    z = (isu(ww[0], im(q, e[0])), isu(ww[1], im(q, e[1])), isu(ww[2], im(q, e[2])))
    Z = ia(ia(iq(z[0]), iq(z[1])), iq(z[2]))
    lam = isc(im(L, iS), 0.1)
    h = ia(ia(q, im(lam, Z)), isc(L, 0.5))
    common = im(ia(isc(im(Z, iS), 0.1), (0.5, 0.5)), im(D, iL))
    phi = (0.0, 0.0)
    bn = (0.0, 0.0)
    for l in range(3):
        gr = isu(ia(im(z[l], iD), im(common, e[l])), isc(im(im(im(lam, q), z[l]), iD), 2.0))
        gw = isu(ia(e[l], isc(im(lam, z[l]), 2.0)), isc(im(im(im(L, Z), ww[l]), iS3), 0.1))
        phi = ia(phi, ia(im(gr, ww[l]), im(gw, ar[l])))
        v = max(abs(gw[0]), abs(gw[1]))
        bn = ia(bn, iq((v, v)))
    phi = ia(phi, isc(h, kh))
    return isu(phi, isc((down(np.sqrt(bn[1])), up(np.sqrt(bn[1]))), delta))[0]

@njit(cache=True, nogil=True)
def candidate_lower(x, obs, par, u, dt, delta, segments):
    out = np.full(len(obs), -np.inf)
    ta = np.empty(segments)
    tb = np.empty(segments)
    half_lo = np.empty(segments)
    half_hi = np.empty(segments)
    re_table = np.empty(segments)
    we_table = np.empty(segments)
    for j in range(segments):
        a = dt * j / segments
        b = up(dt * (j + 1) / segments)
        ta[j] = a
        tb[j] = b
        half = isc(iq((a, b)), 0.5)
        half_lo[j] = half[0]
        half_hi[j] = half[1]
        re_table[j] = up(1.1 * b ** 3 / 6.0 + delta * b * b / 2.0)
        we_table[j] = up(1.1 * b * b / 2.0 + delta * b)
    rr = np.empty((3, 2))
    ww = np.empty((3, 2))
    ar = np.empty((3, 2))
    r0 = np.empty(3)
    w0 = np.empty(3)
    a0 = np.empty(3)
    for i in range(len(obs)):
        for l in range(3):
            r0[l] = obs[i, l] - x[l]
            w0[l] = obs[i, l + 3] - x[l + 3]
            a0[l] = obs[i, l + 6] - u[l]
        R0 = par[21] + par[22] + obs[i, 9]
        minimum = np.inf
        for j in range(segments):
            b = tb[j]
            T = (ta[j], b)
            half = (half_lo[j], half_hi[j])
            re = re_table[j]
            we = we_table[j]
            for l in range(3):
                rr[l] = ia(ia(ia((r0[l], r0[l]), isc(T, w0[l])), isc(half, a0[l])), (-re, re))
                ww[l] = ia(ia((w0[l], w0[l]), isc(T, a0[l])), (-we, we))
                ar[l] = ia((a0[l], a0[l]), (-1.1 * b, 1.1 * b))
            lower = evaluate(rr, ww, ar, R0, par[3], delta)
            if not np.isfinite(lower):
                minimum = -np.inf
                break
            minimum = min(minimum, lower)
        out[i] = minimum
    return out

# Rounding platform probe version2; invalidate prior compiled dependency cache.

# Platform rounding version3, explicit NumPy scalar bitcast.
