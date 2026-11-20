"""Connect manuscript certificates to the same state and parameters as control."""
import numpy as np
from configuration import core_config
from ra_dpcbf.core import Boundary
from ra_dpcbf.certificates import (initialization_report,initial_state,
    initial_common_margin,finite_interval_check,current_common_set)


def check_initial_state(x,obs,R,V,ref,config,horizon=None):
    cfg=core_config(config)
    base=cfg.robot_radius+cfg.base_margin+obs[:,9]
    boundaries=[Boundary(float(R[i]),float(V[i]),float(ref[i]),float(base[i]),
                        float(base[i]+cfg.radius_span)) for i in range(len(obs))]
    T=config['feasibility_checks']['certificate_horizon_s'] if horizon is None else horizon
    verification=config['feasibility_checks']['numeric_verification_margin']
    initial=initialization_report(x,obs,boundaries,cfg)
    short=initial_common_margin(x,obs,boundaries,cfg,verification_margin=verification)
    finite=finite_interval_check(initial_state(x,obs,boundaries,cfg),cfg,T,
                                 verification_margin=verification)
    return dict(initialization=initial,initial_common_margin=short,finite_interval=finite,
        sampling_guarantee_from_continuous_certificate=False,
        scope='The latest manuscript sufficient conditions. A failed certificate is not a failed control QP.')


def check_current_sets(B,C,config,margins=None):
    limits=config['acceleration_component_limits']
    raw=current_common_set(B,C,limits)
    executable=current_common_set(B,C,limits,margins=margins) if margins is not None else None
    return dict(continuous=raw,sampling_tightened=executable)
