"""Controller-parameter transport for a UAV-attached native velocity diagram.

Each obstacle uses w=v_obstacle-v_UAV and X=p_actual+tau*w_boundary.
The UAV center is the zero-velocity schematic origin, NOT the current velocity
point. Parameters and radius are held from ONE timestamped controller snapshot.
"""
from pathlib import Path
import sys,time,json,os
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.append('/usr/lib/python3/dist-packages')
from gz.transport13 import Node
from gz.msgs10.marker_pb2 import Marker
from gz.msgs10.marker_v_pb2 import Marker_V
from gz.msgs10.double_v_pb2 import Double_V
from gz.msgs10.gui_camera_pb2 import GUICamera
from gz.msgs10.boolean_pb2 import Boolean
from rpc_client import RpcClient
PALETTE=((.267,.005,.329),(.281,.156,.469),(.244,.287,.537),(.191,.407,.556),(.148,.512,.557),(.119,.619,.537),(.208,.719,.473),(.431,.808,.346),(.710,.869,.169),(.993,.906,.144))
TAU_VIS=.6
TANGENT_LIMIT=1.5
LETTERS={'S':[(.6,1,0,1),(0,1,0,.5),(0,.5,.6,.5),(.6,.5,.6,0),(.6,0,0,0)],'T':[(0,1,.6,1),(.3,1,.3,0)],'A':[(0,0,.3,1),(.3,1,.6,0),(.12,.45,.48,.45)],'R':[(0,0,0,1),(0,1,.6,1),(.6,1,.6,.5),(.6,.5,0,.5),(.3,.5,.6,0)],'G':[(.6,1,0,1),(0,1,0,0),(0,0,.6,0),(.6,0,.6,.5),(.6,.5,.3,.5)],'O':[(0,0,0,1),(0,1,.6,1),(.6,1,.6,0),(.6,0,0,0)],'L':[(0,1,0,0),(0,0,.6,0)]}

class NativeOverlay:
    def __init__(self,out,start,goal,parameters):
        self.method=os.environ.get("CBF_COMPARE_METHOD","ra")
        self.out=Path(out);self.start=np.asarray(start);self.goal=np.asarray(goal);self.par=np.asarray(parameters)
        self.audit_frames=[];self.snapshots=[];self.last_save=-1.
        self.warming=False;self.last=-1.;self.count=0;self.publish_failures=0;self.applied_t=-1.;self.applied_frames=0;self.model_mesh=False
        self.trail=[];self.speeds=[];self.coefficients={};self.last_message=None;self.send_ms=[];self.native_metrics=[]
        rpc=RpcClient();m=GUICamera();m.pose.position.x=9.;m.pose.position.y=-14.;m.pose.position.z=12.
        q=Rotation.from_euler('xyz',[0,.42,np.pi/2]).as_quat()
        for key,value in zip(('x','y','z','w'),q):setattr(m.pose.orientation,key,float(value))
        self.camera_ack=False;deadline=time.monotonic()+30.
        while time.monotonic()<deadline:
            ok,response=rpc.request('/gui/move_to/pose',m,GUICamera,Boolean,500)
            if ok and response.data:self.camera_ack=True;break
            time.sleep(.2)
        self.rpc=rpc
        # A live process/camera RPC is insufficient: verify actual Render events.
        # An empty, invisible marker is only a transport/readiness handshake.
        deadline=time.monotonic()+30.
        while time.monotonic()<deadline:
            ready=Marker_V();m=ready.marker.add();m.ns='readiness';m.id=0;m.type=Marker.LINE_LIST
            m.pose.orientation.w=1.;m.scale.x=m.scale.y=m.scale.z=1.;m.material.diffuse.a=0.
            self.publish(ready);time.sleep(.10)
            if self.applied_frames>=3:break
        if self.applied_frames<3:
            (self.out/'display_startup_diagnostic.json').write_text(json.dumps(dict(camera_ack=self.camera_ack,applied_frames=self.applied_frames,service_connected=True)))
            raise RuntimeError('Native GUI never acknowledged three real render cycles; mission not started')
    def publish(self,array):
        ok,response=self.rpc.request('/ra/native_geometry_rpc',array,Marker_V,Double_V,2000)
        if ok:self.applied(response)
        return bool(ok)

    def prepare(self,state,truth,obstacles,barriers,stamp):
        self.warming=True;before=self.applied_frames;deadline=time.monotonic()+45.
        while time.monotonic()<deadline:
            self.last=-1.;self.update(0.,state,truth,obstacles,barriers,np.zeros(3),0.,stamp)
            time.sleep(.15)
            if self.applied_frames>=before+4:break
        if self.applied_frames<before+4:raise RuntimeError('Dense full-geometry rendering did not warm up before mission')
        self.warming=False;self.last=-1.;self.trail=[];self.speeds=[];self.snapshots=[];self.audit_frames=[];self.coefficients={}

    def frame_audit(self,msg):
        self.audit_frames.append(list(msg.data))
    def applied(self,msg):
        if len(msg.data)>=5:
            self.applied_t=float(msg.data[0]);self.applied_frames=int(msg.data[1]);self.model_mesh=bool(msg.data[2]);self.native_metrics=list(map(float,msg.data[4:]))
    def marker(self,array,ns,identity,kind,color,alpha=1.):
        m=array.marker.add();m.ns=ns;m.id=identity;m.type=kind;m.pose.orientation.w=1.;m.scale.x=m.scale.y=m.scale.z=1.
        c=m.material.diffuse;c.r,c.g,c.b=map(float,color);c.a=alpha;return m
    def points(self,m,xyz):
        for row in np.asarray(xyz).reshape(-1,3):
            p=m.point.add();p.x,p.y,p.z=map(float,row)
    def lines(self,array,identity,xyz,color,alpha=1.,namespace='route'):
        m=self.marker(array,namespace,identity,Marker.LINE_LIST,color,alpha);self.points(m,xyz);return m
    def text(self,array,identity,value,position,color,size=.35):
        segments=[];position=np.asarray(position)
        for j,c in enumerate(value):
            for x,z,a,b in LETTERS[c]:segments.append([position+size*np.array([x+.8*j,0,z]),position+size*np.array([a+.8*j,0,b])])
        self.lines(array,identity,segments,color)
    def update(self,t,state,truth,obstacles,barriers,command,gap,controller_stamp):
        if t-self.last<.019:return
        begin=time.perf_counter();self.last=t;p=np.asarray(truth[1:4]);array=Marker_V();array.header.stamp.sec=int(t);array.header.stamp.nsec=round((t-int(t))*1e9)
        audit=array.header.data.add();audit.key='audit_path';audit.value.append(str(self.out/'native_frames.bin'))
        if self.warming:array.header.data.add().key='prepare'
        self.trail.append(p.copy());self.speeds.append(float(np.linalg.norm(truth[4:7])))
        if len(self.trail)>1:
            xyz=np.asarray(self.trail);m=self.lines(array,1,np.stack([xyz[:-1],xyz[1:]],axis=1),(.82,.06,.03))
            for speed in np.repeat(self.speeds[:-1],2):
                f=np.clip(speed/2.2,0,1);c=m.materials.add().diffuse;c.r=.97-.25*f;c.g=.55-.5*f;c.b=.45-.43*f;c.a=1.
        self.lines(array,2,[[self.start,self.goal]],(.58,.61,.64),.5)
        for identity,where,color in ((3,self.start,(.55,.55,.55)),(4,self.goal,(.92,.66,.01))):
            angle=np.linspace(0,2*np.pi,41);ring=where+np.c_[.16*np.cos(angle),.16*np.sin(angle),np.zeros(len(angle))]
            self.lines(array,identity,np.stack([ring[:-1],ring[1:]],axis=1),color)
        self.text(array,5,'START',self.start+[0,0,.65],(.23,.28,.31));self.text(array,6,'GOAL',self.goal+[0,0,.65],(.85,.58,.0))
        params=[]
        for j,(o,b) in enumerate(zip(obstacles,barriers)):
            relative=o[:3]-state[:3];distance=np.linalg.norm(relative);e=relative/distance
            lam=self.par[1]*b[6]/(self.par[0]*self.par[19]);vertex=b[6]/self.par[0]-b[4]
            mode={'ra':0,'ma_cbf_vo':1,'c3bf':2,'park':3}[self.method]
            if self.method=='park':
                from baseline_coefficients import KL,KM
                rad=self.par[21]+self.par[22]+o[9];d=np.sqrt(max(distance*distance-rad*rad,1e-12));speed=np.sqrt(np.sum((o[3:6]-state[3:])**2)+1e-12)
                lam=KL*d/speed;vertex=KM*d
            elif self.method!='ra':lam=0.;vertex=0.
            if lam<0:raise RuntimeError('Negative clearance: no ordinary safe paraboloid geometry')
            # q_boundary=-mu-lambda*rho^2; original obstacle-minus-UAV sign.
            m=self.marker(array,'ra_parameters',j,Marker.POINTS,PALETTE[j%len(PALETTE)])
            m.scale.x=TAU_VIS;m.scale.y=TANGENT_LIMIT;m.scale.z=mode # fixed isotropic seconds and tangent-speed radius
            self.points(m,[e,[vertex,lam,b[7]],[b[3],b[4],b[5]],[o[9],self.par[2],controller_stamp],
                [self.par[21]+self.par[22]+o[9],self.par[5],b[6]],o[:3],o[3:6],[self.par[0],self.par[1]/(self.par[0]*self.par[19]),t],o[3:6]-state[3:]])
            params.append([float(vertex),float(lam),float(b[3]),float(b[4]),float(b[5])])
        if not self.warming:self.snapshots.append((float(t),float(controller_stamp),np.asarray(truth).copy(),np.asarray(state).copy(),np.asarray(obstacles).copy(),np.asarray(barriers).copy(),np.asarray(params)))
        self.coefficients[round(t,6)]=params
        while len(self.coefficients)>128:self.coefficients.pop(next(iter(self.coefficients)))
        if not self.publish(array):self.publish_failures+=1
        self.last_message=array;self.count+=1;self.send_ms.append((time.perf_counter()-begin)*1000)
        if t-self.last_save>.095:self.save();self.last_save=t
    def save(self):
        params=self.coefficients.get(round(self.applied_t,6),[])
        result=dict(method=self.method,geometry_kind={"ra":"True RA paraboloid and dynamic radius","park":"Actual Park parabola section","c3bf":"Fixed-size direction cone schematic and actual relative-velocity direction","ma_cbf_vo":"Actual relative-velocity directions; no RA surface or halo"}[self.method],risk_halos_enabled=self.method=="ra",native_GUI=True,renderer='Actual backend is recorded from Ogre2 log, not inferred from configured preference',sent_batches=self.count,
            marker_batches=self.applied_frames,marker_failures=self.publish_failures,last_sent_sim_s=self.last,last_render_sim_s=self.applied_t,
            camera=[dict(camera_service_ack=self.camera_ack)],actual_X500_mesh_replica=self.model_mesh,display_aircraft_scale=1.7,physical_aircraft_modified=False,
            last_curvature_vertex=[[v[1],v[0]] for v in params],last_risk_radius=[v[2] for v in params],
            velocity_coordinates='w=v_obstacle-v_UAV; per-obstacle relative-velocity schematic',local_map='X=actual_UAV_position+0.6*w_boundary; UAV center is zero origin, not current velocity point',uniform_scale_seconds=TAU_VIS,fixed_tangent_radius_mps=TANGENT_LIMIT,
            vertex_formula='p_actual-tau*(s/Tr-Rdot)*e',curvature_formula='gamma*s/(Tr*Zmax)',
            surface_style='Original relative-velocity sign; opens away from obstacle; fixed tangent disk; no attitude-driven rotation',
            risk_style='Actual controller R held together with mu/lambda at the same snapshot stamp; no prediction or artificial radius animation',
            arrow_style='Red true obstacle velocity; surface-rooted glyph attached to actual sphere position every render',
            native_metrics=self.native_metrics,Python_parameter_transport_p95_ms=float(np.percentile(self.send_ms,95)) if self.send_ms else None)
        if len(self.native_metrics)>=10:
            names=('render_CPU_ms','surfaces','surface_equation_error','UAV_origin_attachment_error_m','arrow_anchor_error_m','minimum_arrow_direction_dot','risk_radius_min_m','risk_radius_max_m','motion_stream_obstacles','parent_attached_spheres')
            result['native_checks']=dict(zip(names,self.native_metrics))
        tmp=self.out/'native_view_status.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(self.out/'native_view_status.json')
    def close(self,completed=False,failure=None):
        if self.last_message is not None:
            m=Marker_V();m.header.CopyFrom(self.last_message.header)
            if completed:m.header.data.add().key='completed'
            elif failure:m.header.data.add().key='failed'
            self.publish(m);time.sleep(.15)
        self.save()
        if self.snapshots:
            np.savez_compressed(self.out/'display_snapshots.npz',
                control_t=np.array([v[0] for v in self.snapshots]),controller_stamp=np.array([v[1] for v in self.snapshots]),
                truth=np.array([v[2] for v in self.snapshots]),state=np.array([v[3] for v in self.snapshots]),
                obstacles=np.array([v[4] for v in self.snapshots]),barriers=np.array([v[5] for v in self.snapshots]),
                parameters=np.array([v[6] for v in self.snapshots]),tau_vis=TAU_VIS,tangent_limit=TANGENT_LIMIT)
        from native_audit_reader import records
        rows,_=records(self.out/'native_frames.bin');self.audit_frames=rows
        self.rpc.close()
        if self.audit_frames:np.savez_compressed(self.out/'native_frame_audit.npz',frames=np.asarray(self.audit_frames))
