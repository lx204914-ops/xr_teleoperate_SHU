from context47 import ROOT
from configuration import load_config
from safety_filter import SafetyFilter
from stable_progress import StableProgressDetour
from no_reselection import NoReselection

def bindings(method):
 cfg=load_config(ROOT/'frozen_controller/method_config.json')
 cfg['experiment_selector']='unchanged_goal_cost' if method=='no_progress_guard' else 'progress_with_tick_hysteresis'
 def factory(config,variant='conditional_detour'):
  f=SafetyFilter(config,'no_detour' if method=='no_detour' else 'conditional_detour')
  if method not in ('no_detour','no_progress_guard'):
   f.mechanism=(NoReselection if method=='no_reselection' else StableProgressDetour)(config['detour'],config)
   if method=='fixed_strength':f.mechanism.fixed_strength=True
  factory.current=f
  return f
 variant={'fixed_radius':'fixed_radius','no_transverse':'no_transverse','no_detour':'no_detour'}.get(method,'conditional_detour')
 return cfg,factory,variant
