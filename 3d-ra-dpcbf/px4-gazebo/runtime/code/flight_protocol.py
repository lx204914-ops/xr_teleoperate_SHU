"""One active PX4 transport implementation; no controller-install side effects."""
import time,math
import numpy as np
from pymavlink import mavutil
DT=.02
def enu(v):return np.array([v[1],v[0],-v[2]],dtype=float)
def ned(v):return enu(v)
class Link:
    def __init__(self,execution='acceleration'):
        self.execution=execution;self.velocity_reference=None
        self.m=mavutil.mavlink_connection('udpin:127.0.0.1:14540',source_system=245)
        hb=self.m.wait_heartbeat(timeout=35)
        if hb is None or hb.autopilot!=mavutil.mavlink.MAV_AUTOPILOT_PX4:raise RuntimeError('No local PX4 heartbeat')
        self.position=None;self.acceleration=None;self.cov=None;self.armed=False;self.mode=None;self.ground=False
        self.quaternion=None;self.acceleration_wall=None;self.acceleration_records=[];self.last_imu_stamp=None
        self.position_stamp=None;self.wall_position=None;self.text=[];self.acks=[];self.position_records=[];self.attitude=None
        self.requested_stream_intervals_us={32:4000,31:4000,105:4000,245:100000,83:20000,85:4000}
        for msg,interval in self.requested_stream_intervals_us.items():
            self.command(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,msg,interval)
        self.last_hb=0.
    def command(self,cmd,*args):
        args=list(args)+[0.]*7
        self.m.mav.command_long_send(self.m.target_system,self.m.target_component,cmd,0,*args[:7])
    def poll(self):
        while True:
            m=self.m.recv_match(blocking=False)
            if m is None:break
            kind=m.get_type()
            if kind=='LOCAL_POSITION_NED':
                self.position=np.r_[enu([m.x,m.y,m.z]),enu([m.vx,m.vy,m.vz])]
                self.position_stamp=m.time_boot_ms*.001;self.wall_position=time.monotonic()
                self.position_records.append([self.position_stamp,*self.position])
            elif kind=='LOCAL_POSITION_NED_COV':
                self.cov=m
            elif kind=='ATTITUDE_QUATERNION':self.quaternion=np.array([m.q1,m.q2,m.q3,m.q4])
            elif kind=='HIGHRES_IMU' and self.quaternion is not None:
                w,a,b,c=self.quaternion
                R=np.array([[1-2*(b*b+c*c),2*(a*b-c*w),2*(a*c+b*w)],
                    [2*(a*b+c*w),1-2*(a*a+c*c),2*(b*c-a*w)],
                    [2*(a*c-b*w),2*(b*c+a*w),1-2*(a*a+b*b)]])
                acc=enu(R@np.array([m.xacc,m.yacc,m.zacc])+np.array([0.,0.,9.80665]))
                stamp=m.time_usec*1e-6;delta=.004 if self.last_imu_stamp is None else max(.0001,min(.1,stamp-self.last_imu_stamp))
                alpha=1-np.exp(-delta/.005)
                self.acceleration=acc if self.acceleration is None else self.acceleration+alpha*(acc-self.acceleration)
                self.acceleration_wall=time.monotonic();self.last_imu_stamp=stamp
                self.acceleration_records.append([stamp,*acc,*self.acceleration])
            elif kind=='ATTITUDE':self.attitude=m
            elif kind=='HEARTBEAT' and m.get_srcComponent()==1:
                self.armed=bool(m.base_mode&mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED);self.mode=(int(m.custom_mode)>>16)&255
            elif kind=='EXTENDED_SYS_STATE':self.ground=m.landed_state==mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND
            elif kind=='STATUSTEXT':self.text.append(dict(text=m.text,severity=m.severity))
            elif kind=='COMMAND_ACK':self.acks.append(dict(command=m.command,result=m.result))
        if time.monotonic()-self.last_hb>.8:
            self.m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,mavutil.mavlink.MAV_AUTOPILOT_INVALID,0,0,0);self.last_hb=time.monotonic()
    def position_command(self,target):
        n=ned(target);mask=8|16|32|64|128|256|1024|2048
        self.m.mav.set_position_target_local_ned_send(int((self.position_stamp or 0)*1000),self.m.target_system,self.m.target_component,mavutil.mavlink.MAV_FRAME_LOCAL_NED,mask,*n,0,0,0,0,0,0,0,0)
    def acceleration_command(self,u):
        a=ned(u);mask=1|2|4|8|16|32|1024|2048;v=np.zeros(3)
        if self.execution=='velocity_ff':
            if self.velocity_reference is None:self.velocity_reference=self.position[3:].copy()
            v=ned(self.velocity_reference);self.velocity_reference+=np.asarray(u)*DT;mask=1|2|4|1024|2048
            if np.linalg.norm(self.velocity_reference)>5:raise RuntimeError('Velocity reference outside test envelope')
        assert not (mask&512) and np.isfinite(a).all()
        self.m.mav.set_position_target_local_ned_send(int((self.position_stamp or 0)*1000),self.m.target_system,self.m.target_component,mavutil.mavlink.MAV_FRAME_LOCAL_NED,mask,0,0,0,*v,*a,0,0)
    def takeoff(self,z):
        tic=time.monotonic();target=None;lastsend=0.;stable=None;lastmode=0.;lastarm=0.
        while time.monotonic()-tic<65:
            self.poll();now=time.monotonic()
            if self.position is not None and target is None:target=np.r_[self.position[:2],z]
            if target is not None and now-lastsend>.04:
                self.position_command(target);lastsend=now
                if now-tic>4 and self.mode!=6 and now-lastmode>3:
                    self.command(mavutil.mavlink.MAV_CMD_DO_SET_MODE,1,6);lastmode=now
                if self.mode==6 and not self.armed and now-lastarm>4:
                    self.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,1);lastarm=now
            if self.armed and self.position is not None and np.linalg.norm(self.position[:3]-target)<.15 and np.linalg.norm(self.position[3:])<.15:
                if stable is None:stable=now
                if now-stable>2:return target
            else:stable=None
            time.sleep(.003)
        raise RuntimeError('Native position takeoff not stable: '+str(self.text[-8:]))
    def land(self):
        self.command(mavutil.mavlink.MAV_CMD_NAV_LAND,0,0,0,math.nan,math.nan,math.nan,math.nan)
        tic=time.monotonic()
        while time.monotonic()-tic<30:
            self.poll()
            if self.ground and not self.armed:return True
            time.sleep(.02)
        return False

