"""Deterministic 3-D constant-velocity scenes for the density/radius study.

Scene rejection uses initial conditions only. It never runs a controller and
therefore cannot favor a method based on its eventual outcome.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from bootstrap import HERE, contract


def _key_seed(n: int, rmax: float, seed: int, attempt: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([20261019, 41017, int(n), int(round(1000 * rmax)), int(seed), int(attempt)])


def _pair_gap_ok(p: np.ndarray, radii: np.ndarray, gap: float) -> bool:
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if np.linalg.norm(p[i] - p[j]) - radii[i] - radii[j] < gap:
                return False
    return True


def _hard_initially_valid(start: np.ndarray, row0: np.ndarray) -> bool:
    """Initial hard-domain gate shared by all four implementations."""
    from configuration import load_config
    from dynamic_study.kernel import constraints
    from baseline_coefficients import coefficients
    from ma_cbf_vo import terms

    cfg = load_config(HERE / "current_method_conditional" / "method_config.json")
    par = np.asarray(cfg["parameters"], dtype=float)
    radius = par[21] + par[22] + row0[:, 9]
    zeros = np.zeros(len(row0))
    # Proposed method, C3BF, and Park-inspired DPCBF hard barriers.
    _, _, ra = constraints(start, row0, radius, zeros, radius, par, 5)
    _, _, c3 = coefficients(start, row0, radius, zeros, radius, par, 2)
    _, _, park = coefficients(start, row0, radius, zeros, radius, par, 8)
    *_, ma, __, ___ = terms(start, row0, par)
    arrays = (ra[:, 6], ra[:, 7], c3[:, 6], c3[:, 7], park[:, 6], park[:, 7], ma[:, 7])
    return all(np.isfinite(a).all() and float(np.min(a)) >= 0.0 for a in arrays)


def _candidate(n: int, rmax: float, seed: int, attempt: int) -> dict:
    spec = contract()["scene"]
    rng = np.random.default_rng(_key_seed(n, rmax, seed, attempt))
    workspace = np.asarray(spec["workspace_m"], dtype=float)
    start = np.asarray(spec["start_state"], dtype=float)
    goal = np.asarray(spec["goal_position_m"], dtype=float)
    horizon = float(spec["horizon_s"])
    dt = float(spec["tape_dt_s"])
    radii = rng.uniform(float(spec["minimum_obstacle_radius_m"]), rmax, size=n)
    p0 = np.empty((n, 3)); velocity = np.empty((n, 3))
    risky = max(1, int(round(0.08 * n)))

    # Conflict-bearing obstacles cross the nominal route at distinct times.
    for i in range(risky):
        accepted = False
        for _ in range(1000):
            tc = rng.uniform(3.5, 8.5)
            center = np.array([2.0 * tc + rng.uniform(-0.45, 0.45),
                               rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45)])
            direction = rng.normal(size=3); direction[0] *= 0.18
            direction /= np.linalg.norm(direction)
            speed = rng.uniform(0.38, 0.78)
            v = speed * direction
            p = center - tc * v
            pend = p + horizon * v
            lo = workspace[:, 0] + radii[i] + 0.05
            hi = workspace[:, 1] - radii[i] - 0.05
            if np.all(p >= lo) and np.all(p <= hi) and np.all(pend >= lo) and np.all(pend <= hi):
                p0[i] = p; velocity[i] = v; accepted = True; break
        if not accepted:
            raise RuntimeError("Could not place a conflict-bearing obstacle")

    # The remaining obstacles are background traffic with in-workspace endpoints.
    for i in range(risky, n):
        lo = workspace[:, 0] + radii[i] + 0.05
        hi = workspace[:, 1] - radii[i] - 0.05
        for _ in range(1000):
            p = rng.uniform(lo, hi)
            pend = rng.uniform(lo, hi)
            v = (pend - p) / horizon
            if 0.08 <= np.linalg.norm(v) <= 0.85:
                p0[i] = p; velocity[i] = v; break
        else:
            raise RuntimeError("Could not place a background obstacle")

    return {
        "p0": p0,
        "velocity": velocity,
        "radii": radii,
        "tape_dt": dt,
        "mode": 1,
        "revision": 1,
        "workspace": workspace,
        "start": start,
        "goal": goal,
        "waypoints": goal.reshape(1, 3),
        "same_reference_knots": True,
        "manoeuvre_seed": seed,
        "geometry_seed_source": "density_cv_v1 deterministic SeedSequence",
        "risk_bearing_count": risky,
        "attempt": attempt,
    }


def build_scene(n: int, rmax: float, seed: int, max_attempts: int = 1000) -> dict:
    spec = contract()["scene"]
    start = np.asarray(spec["start_state"], dtype=float)
    robot_radius = 0.3
    for attempt in range(max_attempts):
        scene = _candidate(n, rmax, seed, attempt)
        row0 = np.zeros((n, 10), dtype=float)
        row0[:, :3] = scene["p0"]; row0[:, 3:6] = scene["velocity"]; row0[:, 9] = scene["radii"]
        body_gap = np.linalg.norm(row0[:, :3] - start[:3], axis=1) - robot_radius - row0[:, 9]
        if float(body_gap.min()) < float(spec["minimum_initial_body_gap_m"]):
            continue
        if not _pair_gap_ok(row0[:, :3], row0[:, 9], float(spec["minimum_initial_obstacle_pair_gap_m"])):
            continue
        if not _hard_initially_valid(start, row0):
            continue
        horizon = float(spec["horizon_s"]); dt = float(spec["tape_dt_s"])
        steps = int(round(horizon / dt)); times = np.arange(steps + 1, dtype=float) * dt
        tape = np.zeros((steps + 1, n, 10), dtype=float)
        tape[:, :, :3] = scene["p0"][None, :, :] + times[:, None, None] * scene["velocity"][None, :, :]
        tape[:, :, 3:6] = scene["velocity"][None, :, :]
        tape[:, :, 9] = scene["radii"][None, :]
        scene["tape"] = tape
        scene.pop("p0"); scene.pop("velocity"); scene.pop("radii")
        scene["scene_id"] = f"cv_N{n:03d}_R{int(round(100*rmax)):02d}_S{seed:04d}"
        scene["tape_sha256"] = hashlib.sha256(scene["tape"].tobytes()).hexdigest()
        scene["metadata"] = {
            "scene_id": scene["scene_id"], "n": n, "rmax_m": rmax, "seed": seed,
            "accepted_attempt": attempt, "risk_bearing_count": scene["risk_bearing_count"],
            "tape_sha256": scene["tape_sha256"], "generator_version": spec["generator_version"],
        }
        return scene
    raise RuntimeError(f"No initially valid scene after {max_attempts} attempts: N={n}, rmax={rmax}, seed={seed}")


def save_scene(scene: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in scene.items() if isinstance(v, np.ndarray)}
    arrays.update(tape_dt=np.asarray(scene["tape_dt"]), mode=np.asarray(scene["mode"]), revision=np.asarray(scene["revision"]))
    np.savez_compressed(destination, **arrays)
    destination.with_suffix(".json").write_text(json.dumps(scene["metadata"], indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--rmax", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = build_scene(args.n, args.rmax, args.seed)
    out = args.out or HERE / "pilot" / "scenes" / f"{result['scene_id']}.npz"
    save_scene(result, out)
    print(json.dumps(result["metadata"], indent=2))
