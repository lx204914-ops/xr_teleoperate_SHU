"""Execution-only controller for RotorPy; no outer-loop safety equations changed.
Uses current simulated rotor RPM / wind truth. This is not a flight-tested controller.
"""
import numpy as np
from scipy.spatial.transform import Rotation
CONFIG = dict(name="geometric_acceleration_derivative_v5", motor_feedback_gain=4.0,
              attitude_kd_rule="2*sqrt(original_kp)",
              feedforward_rate_limit_rad_s=2.0, feedforward_acc_limit_rad_s2=20.0, aero_feedforward=True,
              reference_jerk_differentiability_threshold_m_s3=20.0,
              attitude_gains_source="kp inherited from RotorPy; kd set by attitude_kd_rule")
class AccelerationExecutor:
    def __init__(self, vehicle):
        self.vehicle=vehicle
        self.previous_Rd=None
        self.wd=np.zeros(3)
        self.wd_stage1=np.zeros(3)
        self.previous_acceleration=None
        self.reference_jerk=np.zeros(3)
        self.previous_aero=None
        self.clipped_steps=0
        self.total_steps=0
        self.max_unclipped=0.
        self.min_unclipped=np.inf
    def command(self,state,acceleration,dt,new_reference=True,reference_dt=None):
        v=self.vehicle
        R=Rotation.from_quat(state["q"]).as_matrix()
        b3=R[:,2]
        rotor=state["rotor_speeds"]
        Fbody,Mbody=v.compute_body_wrench(state["w"],rotor,R.T@(state["v"]-state["wind"]))
        thrust=v.k_eta*np.sum(rotor**2)
        force_aero=R@Fbody-thrust*b3
        ideal_motor_moment=v.f_to_TM[1:]@(v.k_eta*rotor**2)
        moment_aero=Mbody-ideal_motor_moment
        F=v.mass*(np.asarray(acceleration)+np.array([0.,0.,v.g]))
        if CONFIG["aero_feedforward"]: F=F-force_aero
        norm=np.linalg.norm(F)
        if norm<1e-8:raise ValueError("Desired thrust direction undefined")
        b3d=F/norm
        c1=np.array([1.,0.,0.]);cross=np.cross(b3d,c1)
        if np.linalg.norm(cross)<1e-6:raise ValueError("Yaw reference singular")
        b2d=cross/np.linalg.norm(cross);b1d=np.cross(b2d,b3d)
        Rd=np.column_stack([b1d,b2d,b3d])
        if new_reference:
            h=dt if reference_dt is None else reference_dt
            if self.previous_acceleration is None:
                jerk=np.zeros(3)
            else:
                jerk=(np.asarray(acceleration)-self.previous_acceleration)/h
            # Discontinuities have no finite flatness derivative: use feedback
            # there, not a fictitious impulse. The acceleration itself is untouched.
            if np.linalg.norm(jerk)>CONFIG["reference_jerk_differentiability_threshold_m_s3"]:
                jerk=np.zeros(3)
            self.reference_jerk=jerk
            self.previous_acceleration=np.asarray(acceleration).copy()
        aero_dot=np.zeros(3) if self.previous_aero is None else (force_aero-self.previous_aero)/dt
        self.previous_aero=force_aero.copy()
        Fdot=v.mass*self.reference_jerk-aero_dot
        b3dot=(Fdot-b3d*(b3d@Fdot))/norm
        crossdot=np.cross(b3dot,c1)
        b2dot=(crossdot-b2d*(b2d@crossdot))/np.linalg.norm(cross)
        b1dot=np.cross(b2dot,b3d)+np.cross(b2d,b3dot)
        Rdot=np.column_stack([b1dot,b2dot,b3dot])
        Sref=Rd.T@Rdot
        wd_next=np.array([-Sref[1,2],Sref[0,2],-Sref[0,1]])
        wd_next*=min(1.,CONFIG["feedforward_rate_limit_rad_s"]/max(np.linalg.norm(wd_next),1e-12))
        wdot_d=(wd_next-self.wd)/dt
        wdot_d*=min(1.,CONFIG["feedforward_acc_limit_rad_s2"]/max(np.linalg.norm(wdot_d),1e-12))
        self.wd=wd_next;self.previous_Rd=Rd
        S=.5*(Rd.T@R-R.T@Rd)
        eR=np.array([-S[1,2],S[0,2],-S[0,1]])
        wd_body=R.T@Rd@self.wd
        kd=2.*np.sqrt(v.kp_att)
        angular_acc=-v.kp_att*eR-kd*(state["w"]-wd_body)
        angular_acc+=-np.cross(state["w"],wd_body)+R.T@Rd@wdot_d
        moment=v.inertia@angular_acc+np.cross(state["w"],v.inertia@state["w"])
        if CONFIG["aero_feedforward"]:moment-=moment_aero
        wrench=np.r_[np.dot(F,b3),moment]
        f=v.TM_to_f@wrench
        target=np.sqrt(np.maximum(f,0.)/v.k_eta)
        unclipped=rotor+CONFIG["motor_feedback_gain"]*(target-rotor)
        cmd=np.clip(unclipped,v.rotor_speed_min,v.rotor_speed_max)
        clipped=bool(np.any(cmd!=unclipped) or np.any(f<0))
        self.clipped_steps+=clipped;self.total_steps+=1
        self.max_unclipped=max(self.max_unclipped,float(np.max(unclipped)))
        self.min_unclipped=min(self.min_unclipped,float(np.min(unclipped)))
        return dict(cmd_motor_speeds=cmd)
