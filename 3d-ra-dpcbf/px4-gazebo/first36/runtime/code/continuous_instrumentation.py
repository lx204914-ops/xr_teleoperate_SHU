"""Execution-only instrumentation. No controller equations or scene edits."""
import time
import numpy as np
from flight_protocol import Link,enu
from gz_bridge import Gazebo

class TimedGazebo(Gazebo):
    def __init__(self):
        self.clock_receipts=[];self.physics_receipts=[]
        super().__init__()
    def clock(self,msg):
        wall=time.monotonic();super().clock(msg)
        self.clock_receipts.append([wall,self.sim])
    def vehicle_truth(self,msg):
        wall=time.monotonic();super().vehicle_truth(msg)
        if len(msg.data)==14:self.physics_receipts.append([wall,float(msg.data[0])])

class TimedLink(Link):
    def __init__(self,*args,**kwargs):
        self.receive_trace=[];self.send_trace=[];self.timesync=[];self.last_sync=0.
        super().__init__(*args,**kwargs)
        old_recv=self.m.recv_match
        def recv(*a,**kw):
            message=old_recv(*a,**kw)
            if message is not None:
                wall=time.monotonic();kind=message.get_type()
                if kind in ('LOCAL_POSITION_NED','HIGHRES_IMU','POSITION_TARGET_LOCAL_NED','TIMESYNC','DEBUG_VECT'):
                    record=dict(kind=kind,wall=wall,system=message.get_srcSystem(),component=message.get_srcComponent())
                    if kind=='LOCAL_POSITION_NED':record['stamp']=message.time_boot_ms*.001
                    if kind=='HIGHRES_IMU':record['stamp']=message.time_usec*1e-6
                    if kind=='POSITION_TARGET_LOCAL_NED':
                        record.update(stamp=message.time_boot_ms*.001,mask=message.type_mask,
                                      velocity_enu=enu([message.vx,message.vy,message.vz]).tolist(),
                                      acceleration_enu=enu([message.afx,message.afy,message.afz]).tolist())
                    if kind=='DEBUG_VECT':
                        name=message.name.decode('ascii',errors='ignore') if isinstance(message.name,bytes) else str(message.name)
                        name=name.rstrip(chr(0))
                        if name.startswith('CBFRX') and name[5:].isdigit() and len(name)==8:
                            record.update(stamp=message.time_usec*1e-6,accepted_wire_sequence=int(name[5:]),acceleration_enu=enu([message.x,message.y,message.z]).tolist())
                    if kind=='TIMESYNC' and message.tc1>0:
                        record.update(echo_ns=message.ts1,px4_ns=message.tc1,rtt_ms=(wall-message.ts1*1e-9)*1000)
                        if 0<record['rtt_ms']<1000:self.timesync.append(record.copy())
                    self.receive_trace.append(record)
            return message
        self.m.recv_match=recv
        old_send=self.m.mav.set_position_target_local_ned_send
        def send(*a,**kw):
            wire_sequence=int(self.m.mav.seq);begin=time.monotonic();old_send(*a,**kw);end=time.monotonic()
            self.send_trace.append(dict(wall_begin=begin,wall_end=end,wire_sequence=wire_sequence,wire_stamp=a[0]*.001,mask=int(a[4]),
                position_enu=enu(a[5:8]).tolist(),velocity_enu=enu(a[8:11]).tolist(),acceleration_enu=enu(a[11:14]).tolist()))
        self.m.mav.set_position_target_local_ned_send=send
    def poll(self):
        super().poll();now=time.monotonic()
        if now-self.last_sync>.25:
            self.m.mav.timesync_send(0,time.monotonic_ns());self.last_sync=now
