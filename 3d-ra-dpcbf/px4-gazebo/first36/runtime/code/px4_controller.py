"""Platform adapter: frozen safety/radius/MPC/detour equations, explicit PX4 settings."""
from pathlib import Path
import numpy as np
from px4_adapter_base import DeployedController,load_config
ROOT=Path(__file__).resolve().parents[1]
class Px4Controller(DeployedController):
    def __init__(self,dt=.02,execution_time_scale=.1,nominal_cruise=1.0,mpc_input_limit=.8,execution_policy='px4_reference'):
        self.config=load_config(ROOT/'frozen_controller/method_config.json')
        self.config['experiment_selector']='progress_with_tick_hysteresis'
        self.config['control_dt']=float(dt)
        self.config['parameters'][9]=float(nominal_cruise)
        self.config['cruise_speed']=float(nominal_cruise)
        self.mpc_input_limit=float(mpc_input_limit)
        self.execution_policy=execution_policy
        self.execution_time_scale=float(execution_time_scale)
        self.config['execution_smoothing']=self.execution_time_scale/dt
        settings_file=ROOT/'execution_adaptation.json'
        if settings_file.exists():
            import json
            settings=json.loads(settings_file.read_text())
            self.config['shared_execution_slew']=bool(settings.get('shared_execution_slew',False))
        self.ready=False
    def reset(self,*args,**kwargs):
        super().reset(*args,**kwargs)
        if self.execution_policy=='px4_reference':
            from px4_safety_filter import Px4SafetyFilter
            from stable_progress import StableProgressDetour
            # Reuse the exact frozen original optional-start/transport/exit law.
            # The RotorPy deployment's extra +0.5 detour guarantee conflicted
            # with its own 0.25 measured-acceleration neighbourhood at rest.
            mechanism=StableProgressDetour(self.config['detour'],self.config)
            self.filter=Px4SafetyFilter(self.config,'conditional_detour');self.filter.mechanism=mechanism;self.filter.prepare(len(self.ids)+6)
    def step(self,*args,**kwargs):
        u,status,diagnostics=super().step(*args,**kwargs)
        if u is not None and self.execution_policy=='px4_reference' and self.filter.bumpless_release:
            self.execution_reference=u.copy();self.execution_reference_rate=np.zeros(3)
        return u,status,diagnostics
    def advance_interval(self,state,obstacles):
        count=round(self.config['control_dt']/self.config['monitor_dt'])
        for _ in range(count):self.advance_auxiliary(state,obstacles,self.config['monitor_dt'])
