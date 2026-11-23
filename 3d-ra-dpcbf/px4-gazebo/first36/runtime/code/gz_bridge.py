"""Read Gazebo clock/odometry without substituting them for PX4 estimates."""
import sys,threading,time,os
os.environ['GZ_IP']='127.0.0.1'
sys.path.append('/usr/lib/python3/dist-packages')
from gz.transport13 import Node
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.odometry_pb2 import Odometry
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.double_v_pb2 import Double_V
from gz.msgs10.world_control_pb2 import WorldControl
from rpc_client import RpcClient
class Gazebo:
    def __init__(self):
        self.node=Node();self.rpc=RpcClient();self.sim=None;self.odom=None;self.samples=[];self.mutex=threading.Lock();self.obstacle_poses={};self.obstacle_pose_samples=[];self.physics=None;self.physics_samples=[];self.motion_status=None;self.step_ack_lost=0
        assert self.node.subscribe(Clock,'/world/ra_lab/clock',self.clock)
        assert self.node.subscribe(Odometry,'/ra/odometry',self.odometry)
        assert self.node.subscribe(Pose_V,'/world/ra_lab/pose/info',self.poses)
        assert self.node.subscribe(Pose_V,'/ra/obstacle_actual',self.poses)
        self.motion_publisher=self.node.advertise('/ra/motion_epoch',Double_V)
        assert self.node.subscribe(Double_V,'/ra/vehicle_truth',self.vehicle_truth)
        assert self.node.subscribe(Double_V,'/ra/motion_status',self.motion_state)
    def motion_state(self,msg):self.motion_status=list(msg.data)
    def vehicle_truth(self,msg):
        if len(msg.data)==14:self.physics=list(msg.data);self.physics_samples.append(self.physics.copy())
    def motion_command(self,epoch,offset):
        m=Double_V();m.data.extend([float(epoch),*map(float,offset)])
        if not self.motion_publisher.publish(m):raise RuntimeError('Obstacle motion publish failed')
        return list(m.data)
    def set_paused(self,pause):
        req=WorldControl();req.pause=bool(pause)
        # Pause/resume is idempotent. Verify the clock effect, not only RPC ACKs.
        for _ in range(3):
            self.rpc.request('/world/ra_lab/control',req,WorldControl,Boolean,100)
            time.sleep(.015);before=self.sim;time.sleep(.025);after=self.sim
            if before is not None and after is not None:
                if pause and abs(after-before)<1e-9:return
                if not pause and after>before+1e-6:return
        raise RuntimeError('World pause/resume effect was not observed')
    def step(self,steps):
        req=WorldControl();req.pause=True;req.multi_step=int(steps)
        ok,r=self.rpc.request('/world/ra_lab/control',req,WorldControl,Boolean,5)
        # A lost RPC acknowledgement is not evidence that the step was not run.
        # Never resend a relative step: the caller verifies the exact clock target.
        if not ok or not r.data:self.step_ack_lost+=1
        time.sleep(.002)
    def poses(self,msg):
        t=msg.header.stamp.sec+1e-9*msg.header.stamp.nsec
        for p in msg.pose:
            if p.name.startswith('ra_obstacle_'):
                i=int(p.name.rsplit('_',1)[1]);row=[t,i,p.position.x,p.position.y,p.position.z]
                self.obstacle_poses[i]=row;self.obstacle_pose_samples.append(row)
    def clock(self,msg):self.sim=msg.sim.sec+1e-9*msg.sim.nsec
    def odometry(self,msg):
        t=msg.header.stamp.sec+1e-9*msg.header.stamp.nsec
        p=msg.pose.position;q=msg.pose.orientation;v=msg.twist.linear;w=msg.twist.angular
        row=[t,p.x,p.y,p.z,v.x,v.y,v.z,q.x,q.y,q.z,q.w,w.x,w.y,w.z]
        with self.mutex:self.odom=row;self.samples.append(row)
    def create_sphere(self,name,radius,position):
        x,y,z=position
        sdf=f'''<sdf version="1.9"><model name="{name}"><static>true</static><pose>{x} {y} {z} 0 0 0</pose><link name="body"><collision name="collision"><geometry><sphere><radius>{radius}</radius></sphere></geometry></collision><visual name="visual"><geometry><sphere><radius>{radius}</radius></sphere></geometry><material><ambient>0.5 0.5 0.5 1</ambient><diffuse>0.5 0.5 0.5 1</diffuse></material></visual></link></model></sdf>'''
        req=EntityFactory();req.sdf=sdf;req.name=name;req.allow_renaming=False
        ok,response=self.rpc.request('/world/ra_lab/create',req,EntityFactory,Boolean,3000)
        if not ok or not response.data:raise RuntimeError('Gazebo obstacle creation failed')
    def set_poses(self,names,obs):
        req=Pose_V()
        for name,o in zip(names,obs):
            p=req.pose.add();p.name=name;p.position.x=float(o[0]);p.position.y=float(o[1]);p.position.z=float(o[2]);p.orientation.w=1.
        ok,response=self.rpc.request('/world/ra_lab/set_pose_vector',req,Pose_V,Boolean,300)
        if not ok or not response.data:raise RuntimeError('Gazebo obstacle update failed')
