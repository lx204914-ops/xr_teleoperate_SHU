"""Control-affine barrier expressions for spherical obstacles."""

from dataclasses import dataclass

import numpy as np

from .config import BarrierConfig, RiskConfig
from .smooth import sigmoid, softplus


@dataclass(frozen=True)
class RiskResult:
    distance: float
    clearance: float
    positive_clearance: float
    parallel_speed: float
    closing_speed: float
    ttc: float
    distance_term: float
    closing_term: float
    ttc_term: float
    risk: float


@dataclass(frozen=True)
class BarrierResult:
    h: float
    risk: RiskResult
    relative_position: np.ndarray
    relative_velocity: np.ndarray
    los_direction: np.ndarray
    parallel_speed: float
    transverse_velocity: np.ndarray
    transverse_speed_sq: float
    adaptive_radius: float
    geometric_gap: float
    relative_speed: float
    lam: float
    mu: float


@dataclass(frozen=True)
class BaseHOCBFResult:
    value: float
    derivative: float
    safe_radius: float
    clearance: float


def _state6(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError("state must be [px, py, pz, vx, vy, vz].")
    return vector


def _obstacle10(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (10,) or not np.all(np.isfinite(vector)):
        raise ValueError("obstacle must contain position, velocity, acceleration, radius.")
    if vector[9] <= 0.0:
        raise ValueError("Obstacle radius must be positive.")
    return vector


def _relative_geometry(
    state: np.ndarray,
    obstacle: np.ndarray,
    distance_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float, np.ndarray, float]:
    robot = _state6(state)
    obstacle = _obstacle10(obstacle)
    relative_position = obstacle[:3] - robot[:3]
    relative_velocity = obstacle[3:6] - robot[3:]
    distance = float(np.linalg.norm(relative_position))
    if distance <= distance_epsilon:
        raise ValueError("Robot and obstacle centers are too close to define LoS.")
    los = relative_position / distance
    parallel = float(los @ relative_velocity)
    transverse = relative_velocity - los * parallel
    transverse_sq = float(transverse @ transverse)
    return (
        relative_position,
        relative_velocity,
        distance,
        los,
        parallel,
        transverse,
        transverse_sq,
    )


def compute_smooth_risk(
    state: np.ndarray,
    obstacle: np.ndarray,
    robot_radius: float,
    config: RiskConfig = RiskConfig(),
) -> RiskResult:
    if robot_radius <= 0.0:
        raise ValueError("Robot radius must be positive.")
    obstacle = _obstacle10(obstacle)
    _, _, distance, _, parallel, _, _ = _relative_geometry(state, obstacle, 1e-12)
    clearance = distance - robot_radius - float(obstacle[9])
    positive_clearance = config.distance_scale * float(
        softplus(clearance / config.distance_scale, config.clearance_beta)
    )
    closing = config.closing_scale * float(
        softplus(-parallel / config.closing_scale, config.closing_beta)
    )
    ttc = positive_clearance / (closing + config.ttc_epsilon)
    distance_term = float(np.exp(-positive_clearance / config.distance_scale))
    closing_term = float(np.tanh(closing / config.closing_scale))
    ttc_term = float(np.exp(-ttc / config.ttc_scale))
    risk = (
        config.weight_distance * distance_term
        + config.weight_closing * closing_term
        + config.weight_ttc * ttc_term
    )
    return RiskResult(
        distance,
        float(clearance),
        float(positive_clearance),
        parallel,
        closing,
        float(ttc),
        distance_term,
        closing_term,
        ttc_term,
        float(risk),
    )


def ra_barrier_value(
    state: np.ndarray,
    obstacle: np.ndarray,
    robot_radius: float,
    risk_config: RiskConfig = RiskConfig(),
    barrier_config: BarrierConfig = BarrierConfig(),
) -> BarrierResult:
    obstacle = _obstacle10(obstacle)
    (
        relative_position,
        relative_velocity,
        distance,
        los,
        parallel,
        transverse,
        transverse_sq,
    ) = _relative_geometry(state, obstacle, barrier_config.distance_epsilon)
    risk = compute_smooth_risk(state, obstacle, robot_radius, risk_config)
    base_radius = robot_radius + float(obstacle[9])
    adaptive_radius = barrier_config.safety_scale * (
        base_radius + barrier_config.gamma_radius * risk.risk
    )
    gap_argument = distance * distance - adaptive_radius * adaptive_radius
    geometric_gap = float(
        np.sqrt(
            softplus(gap_argument, barrier_config.gap_beta)
            + barrier_config.gap_epsilon**2
        )
    )
    relative_speed = float(
        np.sqrt(
            relative_velocity @ relative_velocity
            + barrier_config.speed_epsilon**2
        )
    )
    lam = barrier_config.k_lambda * geometric_gap / relative_speed
    mu = barrier_config.k_mu * geometric_gap
    h = parallel + lam * transverse_sq + mu
    return BarrierResult(
        float(h),
        risk,
        relative_position,
        relative_velocity,
        los,
        parallel,
        transverse,
        transverse_sq,
        float(adaptive_radius),
        geometric_gap,
        relative_speed,
        float(lam),
        float(mu),
    )


def ra_barrier_affine_terms(
    state: np.ndarray,
    obstacle: np.ndarray,
    robot_radius: float,
    risk_config: RiskConfig = RiskConfig(),
    barrier_config: BarrierConfig = BarrierConfig(),
) -> tuple[float, np.ndarray, BarrierResult]:
    robot = _state6(state)
    obstacle = _obstacle10(obstacle)
    value = ra_barrier_value(
        robot, obstacle, robot_radius, risk_config, barrier_config
    )
    distance = value.risk.distance
    parallel = value.parallel_speed
    transverse = value.transverse_velocity
    transverse_sq = value.transverse_speed_sq
    los = value.los_direction
    obstacle_acceleration = obstacle[6:9]

    parallel_dot_0 = transverse_sq / distance + float(los @ obstacle_acceleration)
    parallel_u = -los
    transverse_sq_dot_0 = (
        2.0 * float(transverse @ obstacle_acceleration)
        - 2.0 * parallel * transverse_sq / distance
    )
    transverse_sq_u = -2.0 * transverse

    clearance_sigma = float(
        sigmoid(
            value.risk.clearance / risk_config.distance_scale,
            risk_config.clearance_beta,
        )
    )
    closing_sigma = float(
        sigmoid(
            -parallel / risk_config.closing_scale,
            risk_config.closing_beta,
        )
    )
    positive_clearance_dot = clearance_sigma * parallel
    closing_dot_0 = -closing_sigma * parallel_dot_0
    closing_u = -closing_sigma * parallel_u
    closing_denominator = value.risk.closing_speed + risk_config.ttc_epsilon
    ttc_dot_0 = (
        positive_clearance_dot * closing_denominator
        - value.risk.positive_clearance * closing_dot_0
    ) / closing_denominator**2
    ttc_u = (
        -value.risk.positive_clearance * closing_u / closing_denominator**2
    )

    distance_term_dot = (
        -value.risk.distance_term
        / risk_config.distance_scale
        * positive_clearance_dot
    )
    closing_term_factor = (
        1.0 - value.risk.closing_term**2
    ) / risk_config.closing_scale
    closing_term_dot_0 = closing_term_factor * closing_dot_0
    closing_term_u = closing_term_factor * closing_u
    ttc_term_factor = -value.risk.ttc_term / risk_config.ttc_scale
    ttc_term_dot_0 = ttc_term_factor * ttc_dot_0
    ttc_term_u = ttc_term_factor * ttc_u
    risk_dot_0 = (
        risk_config.weight_distance * distance_term_dot
        + risk_config.weight_closing * closing_term_dot_0
        + risk_config.weight_ttc * ttc_term_dot_0
    )
    risk_u = (
        risk_config.weight_closing * closing_term_u
        + risk_config.weight_ttc * ttc_term_u
    )

    radius_factor = barrier_config.safety_scale * barrier_config.gamma_radius
    radius_dot_0 = radius_factor * risk_dot_0
    radius_u = radius_factor * risk_u
    gap_argument = distance * distance - value.adaptive_radius**2
    gap_argument_dot_0 = (
        2.0 * distance * parallel - 2.0 * value.adaptive_radius * radius_dot_0
    )
    gap_argument_u = -2.0 * value.adaptive_radius * radius_u
    gap_factor = float(sigmoid(gap_argument, barrier_config.gap_beta)) / (
        2.0 * value.geometric_gap
    )
    gap_dot_0 = gap_factor * gap_argument_dot_0
    gap_u = gap_factor * gap_argument_u

    speed_dot_0 = float(
        value.relative_velocity @ obstacle_acceleration
    ) / value.relative_speed
    speed_u = -value.relative_velocity / value.relative_speed
    lambda_dot_0 = barrier_config.k_lambda * (
        gap_dot_0 / value.relative_speed
        - value.geometric_gap * speed_dot_0 / value.relative_speed**2
    )
    lambda_u = barrier_config.k_lambda * (
        gap_u / value.relative_speed
        - value.geometric_gap * speed_u / value.relative_speed**2
    )
    mu_dot_0 = barrier_config.k_mu * gap_dot_0
    mu_u = barrier_config.k_mu * gap_u

    offset = (
        parallel_dot_0
        + lambda_dot_0 * transverse_sq
        + value.lam * transverse_sq_dot_0
        + mu_dot_0
    )
    coefficient = (
        parallel_u
        + lambda_u * transverse_sq
        + value.lam * transverse_sq_u
        + mu_u
    )
    return float(offset), np.asarray(coefficient, dtype=float), value


def ra_barrier_rate(
    state: np.ndarray,
    obstacle: np.ndarray,
    control: np.ndarray,
    robot_radius: float,
    risk_config: RiskConfig = RiskConfig(),
    barrier_config: BarrierConfig = BarrierConfig(),
) -> float:
    control = np.asarray(control, dtype=float).reshape(-1)
    if control.shape != (3,):
        raise ValueError("control must contain three values.")
    offset, coefficient, _ = ra_barrier_affine_terms(
        state, obstacle, robot_radius, risk_config, barrier_config
    )
    return float(offset + coefficient @ control)


def base_hocbf_affine_terms(
    state: np.ndarray,
    obstacle: np.ndarray,
    robot_radius: float,
    margin: float = 0.15,
    gain_1: float = 2.0,
    gain_2: float = 2.0,
) -> tuple[float, np.ndarray, BaseHOCBFResult]:
    if robot_radius <= 0.0 or margin < 0.0 or min(gain_1, gain_2) <= 0.0:
        raise ValueError("HOCBF radii and gains are invalid.")
    robot = _state6(state)
    obstacle = _obstacle10(obstacle)
    relative_position = obstacle[:3] - robot[:3]
    relative_velocity = obstacle[3:6] - robot[3:]
    safe_radius = robot_radius + float(obstacle[9]) + margin
    value = float(relative_position @ relative_position - safe_radius**2)
    derivative = float(2.0 * relative_position @ relative_velocity)
    offset = float(
        2.0 * relative_velocity @ relative_velocity
        + 2.0 * relative_position @ obstacle[6:9]
        + (gain_1 + gain_2) * derivative
        + gain_1 * gain_2 * value
    )
    coefficient = -2.0 * relative_position
    clearance = float(np.linalg.norm(relative_position) - safe_radius)
    return (
        offset,
        coefficient,
        BaseHOCBFResult(value, derivative, safe_radius, clearance),
    )
