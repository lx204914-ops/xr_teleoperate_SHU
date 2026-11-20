"""Computable checks, not a claim of recursive feasibility."""
import numpy as np
from scipy.optimize import linprog
from .certificates import (InitialState, initialization_report, initial_state,
    initial_common_margin, local_interval_from_verified_bounds, finite_interval_terms,
    finite_interval_check, current_common_set, common_direction_check)


def boundary_bounds(cfg):
    return {"radius_rate_bound": cfg.omega*cfg.radius_span,
            "radius_acceleration_bound": 2*cfg.omega**2*cfg.radius_span}


def single_box_margin(barrier, cfg):
    """Exact instantaneous feasibility test for ONE CBF and the input box."""
    return float(barrier['A'] + cfg.cbf_gain*barrier['h']
                 + np.asarray(cfg.acceleration_limit) @ np.abs(barrier['B']))


def common_box_margin(barriers, cfg):
    """max_{u in U} min_i (A_i + B_i.u + k_h h_i)."""
    if not barriers:
        return float('inf')
    constraints = np.array([np.r_[-b['B'], 1.0] for b in barriers])
    constants = np.array([b['A']+cfg.cbf_gain*b['h'] for b in barriers])
    result = linprog([0, 0, 0, -1], A_ub=constraints, b_ub=constants,
                     bounds=[(-x, x) for x in cfg.acceleration_limit]+[(None, None)],
                     method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x[-1])


def sufficient_bound(cfg, s_max, q_max, D_min):
    """ONE-obstacle sufficient certificate on s>=0, h>=0, Z<=Zmax,
    s<=s_max, |q|<=q_max and D>=D_min, with the prescribed boundary
    initialization and ||a_o||<=a_o_bound. Not a multi-obstacle certificate.
    """
    if min(s_max, q_max) < 0 or D_min <= 0:
        raise ValueError("Invalid certification domain.")
    H = (2*cfg.omega**2*cfg.radius_span
         + (1+cfg.transverse_weight)**2*s_max/cfg.clearance_time**2
         + 2*cfg.transverse_weight*s_max*q_max/(cfg.clearance_time*D_min))
    return dict(H=H, domain_s_max=s_max, domain_q_max=q_max, domain_D_min=D_min,
                domain_Z_max=cfg.transverse_speed_sq_scale,
                assumptions='0<=s<=s_max, h>=0, |q|<=q_max, D>=D_min, Z<=Zmax; bounded obstacle acceleration and prescribed boundary initialization',
                sufficient_margin=min(cfg.acceleration_limit)-cfg.obstacle_acceleration_bound-H,
                scope='instantaneous, within declared domain; not recursive feasibility')
