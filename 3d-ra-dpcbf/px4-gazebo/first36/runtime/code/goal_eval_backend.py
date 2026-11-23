"""Compiled evaluation of the unchanged candidate cost and progress equations."""
import numpy as np
from numba import njit

@njit(cache=True)
def evaluate(x,goal,nominal,plan,previous,control,ds,cruise,weights):
    n=len(plan);tt=(np.arange(n)+1)*ds
    delta=goal-x[:3];distance=np.linalg.norm(delta);direction=delta/max(distance,1e-12)
    desired=cruise*np.tanh(distance/1.6)*direction
    U=plan.copy();U[0]=control
    v=x[3:].copy();p=x[:3].copy();value=0.
    last=previous.copy()
    for k in range(n):
        p=p+ds*v+.5*ds*ds*U[k];v=v+ds*U[k]
        ref=x[:3]+min(distance,tt[k]*np.linalg.norm(desired))*direction
        value+=(weights[0]*np.sum((p-ref)**2)+weights[1]*np.sum((v-desired)**2)
                +weights[2]*np.sum(U[k]**2)+weights[3]*np.sum((U[k]-last)**2))
        last=U[k]
    end=np.linalg.norm(goal-p)
    return value,end,distance-end

ORIGINAL=None
def fast_goal_evaluation(self,control):
    c=self.context;cfg=self.full_config;n=cfg['mpc_horizon']
    U=np.tile(c['nominal'],(n,1)) if c['plan'] is None else np.asarray(c['plan'])
    if U.shape!=(n,3):raise ValueError('The nominal MPC plan must have horizon by 3 entries')
    weights=np.array([cfg['mpc_position_weight'],cfg['mpc_velocity_weight'],cfg['mpc_input_weight'],cfg['mpc_increment_weight']])
    cost,end,progress=evaluate(c['x'],c['goal'],c['nominal'],U,c['previous'],np.asarray(control),cfg['mpc_prediction_dt'],cfg['cruise_speed'],weights)
    return dict(cost=float(cost),goal_distance_T=float(end),predicted_progress_m=float(progress))

def install():
    global ORIGINAL
    from goal_aware_detour import GoalAwareDetour
    if ORIGINAL is None:ORIGINAL=GoalAwareDetour.goal_evaluation
    # Compile before PX4 starts; no JIT initialization during a control tick.
    evaluate(np.zeros(6),np.ones(3),np.zeros(3),np.zeros((12,3)),np.zeros(3),np.zeros(3),.1,1.5,np.ones(4))
    GoalAwareDetour.goal_evaluation=fast_goal_evaluation
