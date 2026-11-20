"""Map outer-loop acceleration commands to VelocityAviary actions."""

import numpy as np


def _vector3(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values.")
    return vector


def nominal_goal_acceleration(
    position: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
    position_gain: float,
    velocity_gain: float,
    max_acceleration: float,
) -> np.ndarray:
    if min(position_gain, velocity_gain, max_acceleration) <= 0.0:
        raise ValueError("Guidance gains and acceleration limit must be positive.")
    position = _vector3(position, "position")
    velocity = _vector3(velocity, "velocity")
    goal = _vector3(goal, "goal")
    command = position_gain * (goal - position) - velocity_gain * velocity
    return np.clip(command, -max_acceleration, max_acceleration)


def acceleration_to_velocity_action(
    velocity: np.ndarray,
    acceleration: np.ndarray,
    dt: float,
    speed_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    if dt <= 0.0 or speed_limit <= 0.0:
        raise ValueError("dt and speed_limit must be positive.")
    velocity = _vector3(velocity, "velocity")
    acceleration = _vector3(acceleration, "acceleration")
    command = velocity + dt * acceleration
    speed = float(np.linalg.norm(command))
    if speed > speed_limit:
        command = command * (speed_limit / speed)
        speed = speed_limit
    action = np.zeros(4, dtype=float)
    if speed > 1e-12:
        action[:3] = command / speed
        action[3] = speed / speed_limit
    return action, command

