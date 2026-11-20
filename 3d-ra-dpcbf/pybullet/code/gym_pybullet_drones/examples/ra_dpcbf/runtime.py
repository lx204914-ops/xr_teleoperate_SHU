"""Run the first physical CF2X scenes in PyBullet."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from PIL import Image
import pybullet as p

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.safety.flight_adapter import (
    acceleration_to_velocity_action,
    nominal_goal_acceleration,
)
from gym_pybullet_drones.safety.obstacles import swept_sphere_collision
from gym_pybullet_drones.safety.qp import HOCBFSafetyFilter, RADPCBFSafetyFilter
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from .scenarios import build_scene


@dataclass(frozen=True)
class SceneRunResult:
    initial_distance: float
    final_distance: float
    trajectory: np.ndarray
    output_dir: Path
    animation_path: Path | None
    minimum_clearance: float
    collision_events: int
    qp_failures: int
    maximum_risk: float
    maximum_slack: float


def should_place_trail_marker(
    previous_marker: np.ndarray,
    position: np.ndarray,
    spacing: float,
) -> bool:
    if spacing <= 0.0:
        raise ValueError("spacing must be positive.")
    return float(np.linalg.norm(position - previous_marker)) >= spacing


def _create_goal(goal: np.ndarray, client_id: int) -> int:
    visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.09,
        rgbaColor=[0.05, 0.85, 0.18, 1.0],
        physicsClientId=client_id,
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual,
        basePosition=goal,
        physicsClientId=client_id,
    )


def _capture_frame(client_id: int, drone_position: np.ndarray) -> Image.Image:
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=drone_position + np.array([0.28, 0.0, 0.02]),
        distance=1.15,
        yaw=42.0,
        pitch=-20.0,
        roll=0.0,
        upAxisIndex=2,
    )
    projection = p.computeProjectionMatrixFOV(58.0, 16.0 / 9.0, 0.05, 20.0)
    _, _, rgba, _, _ = p.getCameraImage(
        width=640,
        height=360,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client_id,
    )
    return Image.fromarray(np.asarray(rgba, dtype=np.uint8)[:, :, :3])


def run_scene(
    scene_name: str,
    method: str = "nominal",
    gui: bool = False,
    duration: float | None = None,
    output_dir: str | Path = "results/ra_dpcbf/scene",
    capture_video: bool = True,
) -> SceneRunResult:
    if method not in {"nominal", "hocbf", "ra_dpcbf"}:
        raise ValueError(f"Unsupported method: {method}")
    scene = build_scene(scene_name)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = VelocityAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=scene.start.reshape(1, 3),
        initial_rpys=np.zeros((1, 3)),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=48,
        gui=gui,
        record=False,
        obstacles=False,
        user_debug_gui=False,
        output_folder=str(output),
    )
    env.SPEED_LIMIT = 1.0
    client_id = env.getPyBulletClient()
    _create_goal(scene.goal, client_id)
    for obstacle in scene.obstacles:
        obstacle.create_body(client_id)
    safety_filter = HOCBFSafetyFilter(
        robot_radius=0.15,
        max_acceleration=2.5,
        margin=0.15,
        gain_1=2.0,
        gain_2=2.0,
    )
    ra_safety_filter = RADPCBFSafetyFilter(
        robot_radius=0.15,
        max_acceleration=2.5,
        base_margin=0.15,
        base_gain_1=2.0,
        base_gain_2=2.0,
        ra_alpha=1.5,
        slack_penalty=1e4,
    )

    run_duration = scene.duration if duration is None else duration
    action = np.zeros((1, 4), dtype=float)
    velocity_command = np.zeros(3, dtype=float)
    trajectory = [scene.start.copy()]
    frames: list[Image.Image] = []
    previous = scene.start.copy()
    previous_marker = scene.start.copy()
    trail_visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.025,
        rgbaColor=[0.05, 0.25, 0.95, 0.95],
        physicsClientId=client_id,
    )
    minimum_clearance = float("inf")
    collision_events = 0
    collision_active = False
    qp_failures = 0
    maximum_risk = 0.0
    maximum_slack = 0.0
    drone_id = int(env.getDroneIds()[0])
    try:
        for step in range(int(run_duration * env.CTRL_FREQ)):
            obs, _, _, _, _ = env.step(action)
            position = np.asarray(obs[0, 0:3], dtype=float)
            velocity = np.asarray(obs[0, 10:13], dtype=float)
            time = step * env.CTRL_TIMESTEP
            obstacle_states = [
                obstacle.update_body(time, client_id) for obstacle in scene.obstacles
            ]
            nominal_control = nominal_goal_acceleration(
                position,
                velocity,
                scene.goal,
                position_gain=1.2,
                velocity_gain=1.4,
                max_acceleration=2.5,
            )
            control = nominal_control
            obstacle_arrays = [
                    np.concatenate(
                        [
                            state.position,
                            state.velocity,
                            state.acceleration,
                            np.array([state.radius]),
                        ]
                    )
                    for state in obstacle_states
            ]
            if method == "hocbf":
                solution = safety_filter.solve(
                    np.concatenate([position, velocity]),
                    nominal_control,
                    obstacle_arrays,
                )
                control = solution.control
                if solution.status not in {"optimal", "optimal_inaccurate"}:
                    qp_failures += 1
            elif method == "ra_dpcbf":
                ra_solution = ra_safety_filter.solve(
                    np.concatenate([position, velocity]),
                    nominal_control,
                    obstacle_arrays,
                )
                control = ra_solution.control
                maximum_risk = max(
                    maximum_risk,
                    max(
                        (barrier.risk.risk for barrier in ra_solution.barriers),
                        default=0.0,
                    ),
                )
                maximum_slack = max(
                    maximum_slack,
                    max(ra_solution.slacks, default=0.0),
                )
                if ra_solution.status not in {"optimal", "optimal_inaccurate"}:
                    qp_failures += 1
            action[0], velocity_command = acceleration_to_velocity_action(
                velocity_command, control, env.CTRL_TIMESTEP, env.SPEED_LIMIT
            )
            trajectory.append(position.copy())
            collision = False
            for obstacle, state in zip(scene.obstacles, obstacle_states):
                clearance = float(
                    np.linalg.norm(position - state.position) - 0.15 - state.radius
                )
                minimum_clearance = min(minimum_clearance, clearance)
                collision = collision or clearance <= 0.0
                collision = collision or swept_sphere_collision(
                    previous,
                    position,
                    state.position,
                    0.15 + state.radius,
                )
                if obstacle.body_id is not None:
                    collision = collision or bool(
                        p.getContactPoints(
                            bodyA=drone_id,
                            bodyB=obstacle.body_id,
                            physicsClientId=client_id,
                        )
                    )
            if collision and not collision_active:
                collision_events += 1
            collision_active = collision
            if should_place_trail_marker(previous_marker, position, 0.06):
                p.createMultiBody(
                    baseMass=0.0,
                    baseVisualShapeIndex=trail_visual,
                    basePosition=position,
                    physicsClientId=client_id,
                )
                previous_marker = position.copy()
            if np.linalg.norm(position - previous) > 1e-4:
                p.addUserDebugLine(
                    previous,
                    position,
                    lineColorRGB=[0.05, 0.25, 0.95],
                    lineWidth=3.0,
                    lifeTime=0.0,
                    physicsClientId=client_id,
                )
                previous = position.copy()
            if capture_video and step % 4 == 0:
                frames.append(_capture_frame(client_id, position))
    finally:
        env.close()

    trajectory_array = np.asarray(trajectory, dtype=float)
    np.savetxt(
        output / "trajectory.csv",
        trajectory_array,
        delimiter=",",
        header="x,y,z",
        comments="",
    )
    animation_path = None
    if frames:
        gif_path = output / f"{scene_name}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // 12,
            loop=0,
        )
        animation_path = output / f"{scene_name}.mp4"
        imageio.mimsave(
            animation_path,
            [np.asarray(frame) for frame in frames],
            fps=12,
            codec="libx264",
            quality=8,
        )
    return SceneRunResult(
        initial_distance=float(np.linalg.norm(scene.goal - scene.start)),
        final_distance=float(np.linalg.norm(scene.goal - trajectory_array[-1])),
        trajectory=trajectory_array,
        output_dir=output,
        animation_path=animation_path,
        minimum_clearance=minimum_clearance,
        collision_events=collision_events,
        qp_failures=qp_failures,
        maximum_risk=maximum_risk,
        maximum_slack=maximum_slack,
    )
