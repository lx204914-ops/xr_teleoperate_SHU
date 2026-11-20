"""The agreed method; no squared-distance eta, boundary reset or safety slack.

Obstacle arrays are [px, py, pz, vx, vy, vz, ax, ay, az, radius].
Robot arrays are [px, py, pz, vx, vy, vz]. SI units throughout.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    robot_radius: float = 0.30
    base_margin: float = 0.15
    acceleration_limit: tuple = (2.0, 2.0, 2.0)
    obstacle_acceleration_bound: float = 0.8
    obstacle_jerk_bound: float = 1.1
    reaction_delay: float = 0.23864762737569936
    prediction_margin: float = 0.0
    closing_epsilon: float = 0.03
    radius_span: float = 0.8587176689796248
    omega: float = 0.1637582081875905
    clearance_time: float = 0.8
    transverse_weight: float = 1.0
    transverse_speed_sq_scale: float = 9.0
    cbf_gain: float = 1.2778780864702888
    control_dt: float = 0.0025
    integration_dt: float = 0.0005
    qp_smoothing: float = 0.035
    cruise_speed: float = 2.0
    velocity_gain: float = 2.0
    goal_tolerance: float = 0.25
    stop_speed: float = 0.20

    def __post_init__(self):
        positive = [self.robot_radius, self.closing_epsilon, self.radius_span,
                    self.omega, self.clearance_time, self.transverse_speed_sq_scale,
                    self.cbf_gain, self.control_dt, self.integration_dt,
                    self.cruise_speed, self.velocity_gain, self.goal_tolerance,
                    self.stop_speed, *self.acceleration_limit]
        if len(self.acceleration_limit) != 3 or not np.all(np.isfinite(positive)) or min(positive) <= 0:
            raise ValueError("Positive, finite configuration values are required.")
        nonnegative=[self.base_margin,self.obstacle_acceleration_bound,self.obstacle_jerk_bound,
                     self.reaction_delay,self.prediction_margin,self.transverse_weight,self.qp_smoothing]
        if not np.all(np.isfinite(nonnegative)) or min(nonnegative)<0:
            raise ValueError("Margins, bounds and weights must be nonnegative.")
        if min(self.acceleration_limit) <= self.obstacle_acceleration_bound:
            raise ValueError("Relative braking authority must be positive.")
        if self.transverse_weight > 1:
            raise ValueError("The documented design range is 0 <= gamma_perp <= 1.")
        ratio = self.control_dt / self.integration_dt
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("control_dt must be a multiple of integration_dt.")

    @property
    def beta(self):
        return self.transverse_weight / (self.clearance_time * self.transverse_speed_sq_scale)


@dataclass(frozen=True)
class Boundary:
    radius: float
    rate: float
    reference: float
    base: float
    maximum: float

    def acceleration(self, cfg):
        return cfg.omega**2 * (self.reference - self.radius) - 2*cfg.omega*self.rate


def los(robot, obstacle):
    r = np.asarray(obstacle[:3]) - np.asarray(robot[:3])
    w = np.asarray(obstacle[3:6]) - np.asarray(robot[3:6])
    D = float(np.linalg.norm(r))
    if D <= 1e-10:
        raise ValueError("LOS is undefined at coincident centers.")
    e = r / D
    q = float(e @ w)
    z = w - q*e
    return D, e, q, z, float(z @ z)


def reference_radius(q, obstacle_radius, cfg):
    """Return raw desired, bounded reference and fixed base radii (metres)."""
    c = 0.5*(np.hypot(q, cfg.closing_epsilon) - q)
    base = cfg.robot_radius + obstacle_radius + cfg.base_margin
    effective = min(cfg.acceleration_limit) - cfg.obstacle_acceleration_bound
    desired = base + cfg.reaction_delay*c + c*c/(2*effective) + cfg.prediction_margin
    reference = float(np.clip(desired, base, base + cfg.radius_span))
    return float(desired), reference, float(base)


def initialize_boundary(robot, obstacle, cfg):
    _, _, q, _, _ = los(robot, obstacle)
    _, ref, base = reference_radius(q, float(obstacle[9]), cfg)
    return Boundary(base, 0.0, ref, base, base + cfg.radius_span)


def set_reference(robot, obstacle, boundary, cfg):
    _, _, q, _, _ = los(robot, obstacle)
    _, ref, base = reference_radius(q, float(obstacle[9]), cfg)
    if abs(base-boundary.base) > 1e-10:
        raise ValueError("Obstacle radius must remain fixed.")
    return Boundary(boundary.radius, boundary.rate, ref, base, boundary.maximum)


def boundary_step(boundary, dt, cfg):
    """Exact critical-damping flow under a held reference; no clipping/reset.

    R(t)=ref + (x0 + (v0+omega*x0)*t)*exp(-omega*t).
    Continuous R/Rdot; reference updates may make Rddot piecewise continuous.
    """
    if dt < 0:
        raise ValueError("Time step must be nonnegative.")
    x = boundary.radius - boundary.reference
    b = boundary.rate + cfg.omega*x
    decay = np.exp(-cfg.omega*dt)
    radius = boundary.reference + (x+b*dt)*decay
    rate = (boundary.rate - cfg.omega*b*dt)*decay
    if radius < boundary.base-1e-10 or radius > boundary.maximum+1e-10:
        raise ValueError("Boundary left its invariant interval; inspect initialization.")
    return Boundary(float(radius), float(rate), boundary.reference, boundary.base, boundary.maximum)


def evaluate(robot, obstacle, boundary, cfg):
    D, e, q, z, Z = los(robot, obstacle)
    s = D - boundary.radius
    K = 1/cfg.clearance_time + cfg.beta*Z
    Rdd = boundary.acceleration(cfg)
    h = q - boundary.rate + K*s
    ao = np.asarray(obstacle[6:9])
    B = -e - 2*cfg.beta*s*z
    A = (Z/D + e@ao - Rdd + K*(q-boundary.rate)
         + 2*cfg.beta*s*(z@ao) - 2*cfg.beta*s*q*Z/D)
    # Frozen-time velocity-space boundary: q = vertex - curvature*Z.
    curvature = cfg.beta*s
    vertex = boundary.rate - s/cfg.clearance_time
    return dict(D=D, e=e, q=q, z=z, Z=Z, s=s, K=K, h=float(h),
                A=float(A), B=B, R=boundary.radius, Rdot=boundary.rate,
                Rddot=Rdd, Rref=boundary.reference, curvature=2*abs(curvature),
                vertex=vertex, mu_base=s/cfg.clearance_time,
                mu_effective=-vertex,quadratic_coefficient=curvature,
                vertex_principal_curvature_magnitude=2*abs(curvature),
                base_margin=D-boundary.base,
                geometric_clearance=D-cfg.robot_radius-obstacle[9])


def surface_q(transverse_1, transverse_2, vertex, quadratic_coefficient):
    """Coefficient lambda, distinct from geometric vertex curvature 2*lambda."""
    return vertex - quadratic_coefficient*(np.asarray(transverse_1)**2 + np.asarray(transverse_2)**2)


def los_basis(e):
    """Orthonormal transverse basis; geometry itself is independent of this choice."""
    axis = np.eye(3)[np.argmin(np.abs(e))]
    b1 = axis - e*(axis@e)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(e, b1)
    return b1, b2
