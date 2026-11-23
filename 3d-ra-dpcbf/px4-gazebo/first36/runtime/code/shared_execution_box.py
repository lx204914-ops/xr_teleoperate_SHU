"""Common deployment command slew envelope, never a post-QP output filter.

For the first-order execution surrogate a_dot=(u-a)/tau and a held command,
||e(k+)|| <= exp(-dt/tau)||e((k-1)+)|| + ||du||. Thus
||du|| <= budget*(1-exp(-dt/tau)) preserves that surrogate's error ball.
The inner axis cube uses 1/sqrt(3). Real X500 validity MUST be separately
measured; this model is NOT a proved tracking-error bound for PX4/Gazebo.
"""
import numpy as np

def bounds(previous,dt,tau,budget=.5,authority=1.5):
 if dt<=0 or tau<=0 or budget<=0:raise ValueError('Invalid execution envelope')
 delta=budget*(-np.expm1(-dt/tau));axis=delta/np.sqrt(3.)
 previous=np.asarray(previous,dtype=float)
 if previous.shape!=(3,) or not np.isfinite(previous).all() or np.max(abs(previous))>authority+1e-7:raise ValueError('Previous physical command outside authority')
 return np.maximum(-authority,previous-axis),np.minimum(authority,previous+axis),float(delta)

def proposed_rows(controller):
 lo,hi,delta=bounds(controller.previous,controller.config['control_dt'],controller.execution_time_scale)
 controller.slew_diagnostic=dict(enabled=True,norm_change_limit=delta,lo=lo.tolist(),hi=hi.tolist(),scope='First-order execution design; actual aircraft budget still independently audited')
 return np.r_[-lo,hi]
