"""The one active parameter file, shared by execution and certificate checks."""
from pathlib import Path
import hashlib
import json
import numpy as np
from ra_dpcbf.core import Config

ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    filename=Path(path or ROOT/'method_config.json')
    cfg = json.loads(filename.read_text(encoding='utf-8-sig'))
    if 'active_config' in cfg:
        filename=filename.parent/cfg['active_config']
        cfg=json.loads(filename.read_text(encoding='utf-8-sig'))
    p = np.asarray(cfg['parameters'], dtype=float)
    if p.shape != (23,) or not np.isfinite(p).all():
        raise ValueError('The active method requires 23 finite, explicitly named parameters.')
    # The imported fast QP and interval backend currently use this declared box.
    if cfg['acceleration_component_limits'] != [2., 2., 2.] or cfg['obstacle_acceleration_bound'] != .8:
        raise ValueError('This backend requires axis bounds ±2 and obstacle bound .8; its braking scale is 1.2.')
    if cfg['obstacle_jerk_bound'] != 1.1:
        raise ValueError('The imported detour hold bound currently uses the declared jerk bound 1.1.')
    if not (p[0] > 0 and 0 <= p[1] <= 1 and p[2] > 0 and p[3] > 0 and p[5] > 0 and p[19] > 0):
        raise ValueError('Invalid Tr, gamma_perp, omega, kh, radius span or Zmax.')
    if p[20] != 0 or cfg.get('prediction_margin', 0) != 0:
        raise ValueError('The latest manuscript has no additional constant prediction margin.')
    if cfg['gamma_perp'] != p[1] or cfg['Zmax'] != p[19]:
        raise ValueError('Duplicated named/vector parameters disagree.')
    if cfg['robot_radius'] != p[21] or cfg['base_margin'] != p[22]:
        raise ValueError('Robot radius and base margin disagree with the vector.')
    if p[18] != 0 or p[21] <= 0 or p[22] < 0:
        raise ValueError('No heuristic guide term; positive robot radius and nonnegative base margin required.')
    d = cfg['detour']
    if not (d['nu_req'] > 0 and 0 < d['eta'] < 1 and d['gamma_min'] > 0 and
            d['k_perp'] > 0 and d['s_on'] > 0 and d['tau_off'] > 0):
        raise ValueError('Invalid detour gains, thresholds or confirmation time.')
    dt, fine = cfg['control_dt'], cfg['monitor_dt']
    if dt <= 0 or fine <= 0 or abs(dt / fine - round(dt / fine)) > 1e-8:
        raise ValueError('control_dt must be a positive integer multiple of monitor_dt.')
    lifecycle=cfg.get('lifecycle_rule')
    allowed=('goal_clear_primary_receding_nominal_safe_for_tau_off',
             'base_safe_qp_default_exceptional_activation_then_original_detour_with_goal_recovery_yield',
             'goal_cost_direction_with_base_safe_recovery')
    if lifecycle not in allowed:
        raise ValueError('Unknown detour lifecycle declaration.')
    if lifecycle in allowed[1:]:
        c=cfg.get('conditional_detour',{})
        required=('s_on','feasibility_reserve_on_mps2','feasibility_check_period_s','feasibility_dwell_s',
            'radial_brake_on_mps2','radial_to_transverse_ratio','radial_braking_dwell_s',
            'stagnation_window_s','stagnation_max_progress_m','stagnation_ignore_goal_radius_m',
            'stagnation_dwell_s','release_dwell_s','rearm_dwell_s')
        if any(key not in c or not np.isfinite(c[key]) or c[key]<=0 for key in required):
            raise ValueError('Conditional-detour thresholds must be present, finite and positive.')
        if list(c.get('acceleration_component_limits',()))!=cfg['acceleration_component_limits']:
            raise ValueError('Conditional feasibility reserve must use the controller input box.')
    return cfg


def core_config(cfg):
    p = np.asarray(cfg['parameters'], dtype=float)
    return Config(robot_radius=p[21], base_margin=p[22],
        acceleration_limit=tuple(cfg['acceleration_component_limits']),
        obstacle_acceleration_bound=cfg['obstacle_acceleration_bound'],
        obstacle_jerk_bound=cfg['obstacle_jerk_bound'], reaction_delay=p[4],
        prediction_margin=p[20], radius_span=p[5], omega=p[2], clearance_time=p[0],
        transverse_weight=p[1], transverse_speed_sq_scale=p[19], cbf_gain=p[3],
        control_dt=cfg['control_dt'], integration_dt=cfg['monitor_dt'],
        qp_smoothing=p[6], cruise_speed=p[9], velocity_gain=p[10])


def config_sha256(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode('utf8')).hexdigest()


def output_root(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT.parent / p
