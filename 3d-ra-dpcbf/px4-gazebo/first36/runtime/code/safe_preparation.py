"""Native position-control preparation; frozen obstacles never moved or hidden.

The RA episode starts only after returning to its original aerial start. A
blocked ground-to-start vertical segment is not part of the RA benchmark.
"""
from pathlib import Path
import json,time,numpy as np
BODY_RADIUS=.4
PREPARATION_MARGIN=.15

def segment_gap(a,b,centers,radii):
 d=b-a;relative=centers-a;alpha=np.clip(relative@d/max(float(d@d),1e-15),0.,1.)
 return float(np.min(np.linalg.norm(relative-alpha[:,None]*d,axis=1)-radii-BODY_RADIUS))

def execute(link,z,gz,tape,initial_offset,out):
 if tape is None:return link.takeoff(z)
 deadline=time.monotonic()+5
 while link.position is None and time.monotonic()<deadline:link.poll();time.sleep(.005)
 if link.position is None:raise RuntimeError('Preparation has no valid PX4 position')
 transform=np.asarray(gz.physics[1:4])-link.position[:3];start=np.asarray(gz.physics[1:4]);target=np.r_[link.position[:2],z];goal=target+transform
 centers=tape[0,:,:3]+initial_offset;radii=tape[0,:,9];direct=segment_gap(start,goal,centers,radii)
 record=dict(episode_not_started=True,obstacles_fixed_at_original_t0=True,main_method_not_active=True,body_radius_m=BODY_RADIUS,extra_preparation_margin_m=PREPARATION_MARGIN,direct_ascent_clearance_m=direct,world_to_PX4_translation=transform.tolist())
 if direct>=PREPARATION_MARGIN:
  record.update(route='original_vertical_takeoff',planned_waypoints_world=[start.tolist(),goal.tolist()]);(Path(out)/'preparation_route.json').write_text(json.dumps(record,indent=2));return link.takeoff(z)
 options=[];low=start.copy();low[2]=max(.65,start[2]);
 for dx in np.arange(-1.4,1.5,.7):
  for dy in np.arange(-5.,5.1,.5):
   side=low+np.array([dx,dy,0.]);top=side.copy();top[2]=goal[2];way=[start,low,side,top,goal]
   clearance=min(segment_gap(a,b,centers,radii) for a,b in zip(way,way[1:]))
   if clearance>=PREPARATION_MARGIN:options.append((sum(np.linalg.norm(b-a) for a,b in zip(way,way[1:])),clearance,way))
 if not options:raise RuntimeError('No collision-free native preparation route; frozen obstacles preserved')
 _,clearance,way=min(options,key=lambda item:item[0]);record.update(route='native_static_sphere_checked_waypoints',planned_waypoints_world=[v.tolist() for v in way],planned_minimum_clearance_m=clearance,waypoint_receipts=[])
 (Path(out)/'preparation_route.json').write_text(json.dumps(record,indent=2));actual_gap=float('inf');airborne=False
 for world_target in way[1:]:
  waypoint=world_target-transform;tic=time.monotonic();lastsend=0.;stable=None;lastmode=0.;lastarm=0.
  while time.monotonic()-tic<65:
   link.poll();now=time.monotonic()
   if link.armed and np.asarray(gz.physics)[3]>.45:airborne=True
   if airborne and (link.mode!=6 or not link.armed):raise RuntimeError('Native preparation lost Offboard/arming; failsafe preserved')
   if now-lastsend>.04:
    link.position_command(waypoint);lastsend=now
    if now-tic>4 and link.mode!=6 and now-lastmode>3:link.command(176,1,6);lastmode=now
    if link.mode==6 and not link.armed and now-lastarm>4:link.command(400,1);lastarm=now
   truth=np.asarray(gz.physics[1:4]);gap=float(np.min(np.linalg.norm(centers-truth,axis=1)-radii-BODY_RADIUS));actual_gap=min(actual_gap,gap)
   if gap<=0:raise RuntimeError('Native preparation physical obstacle envelope violated')
   if link.armed and np.linalg.norm(link.position[:3]-waypoint)<.15 and np.linalg.norm(link.position[3:])<.15:
    if stable is None:stable=now
    if now-stable>2:break
   else:stable=None
   time.sleep(.003)
  else:raise RuntimeError('Native preparation waypoint did not stabilize')
  record['waypoint_receipts'].append(dict(target_world=world_target.tolist(),actual_world=list(gz.physics[1:4]),PX4_state=link.position.tolist()));print('PREPARATION_WAYPOINT_REACHED',world_target.tolist(),flush=True)
  record['actual_minimum_clearance_m']=actual_gap;(Path(out)/'preparation_route.json').write_text(json.dumps(record,indent=2))
 record.update(completed=True,final_original_start_target=target.tolist());(Path(out)/'preparation_route.json').write_text(json.dumps(record,indent=2));return target
