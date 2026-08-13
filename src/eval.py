"""Evaluation script for GCF-DDPM trajectory generation.

Computes metrics (minADE, minFDE, endpoint error, diversity, goal hit rate,
b-minFDE, miss rate, off-road rate, jerk/acceleration/curvature statistics,
path efficiency, constraint violation rate) and generates evaluation
visualizations: metric distributions, per-scenario breakdown, and best/worst
scenario trajectory plots.
"""

import argparse
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from viz.viz_evaluation import plot_metric_distributions, plot_per_scenario_breakdown
from viz.viz_trajectory import plot_trajectories
from viz.style import save_figure


def compute_minADE(generated: np.ndarray, gt: np.ndarray) -> float:
    """Minimum Average Displacement Error over N generated trajectories."""
    dists = np.linalg.norm(generated - gt[np.newaxis], axis=-1)  # (N, T)
    return dists.mean(axis=-1).min()


def compute_minFDE(generated: np.ndarray, gt: np.ndarray) -> float:
    """Minimum Final Displacement Error."""
    dists = np.linalg.norm(generated[:, -1, :] - gt[-1, :][np.newaxis], axis=-1)  # (N,)
    return dists.min()


def compute_endpoint_error(generated: np.ndarray, goal: np.ndarray) -> float:
    """Average endpoint distance to goal across all generated trajectories."""
    dists = np.linalg.norm(generated[:, -1, :] - goal[np.newaxis], axis=-1)
    return dists.mean()


def compute_diversity(generated: np.ndarray) -> float:
    """Average pairwise distance between generated trajectory endpoints."""
    endpoints = generated[:, -1, :]
    n = len(endpoints)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(endpoints[i] - endpoints[j]))
    return np.mean(dists)


def compute_goal_hit_rate(generated: np.ndarray, goal: np.ndarray, threshold: float = 2.0) -> float:
    """Fraction of trajectories whose endpoint is within threshold of goal."""
    dists = np.linalg.norm(generated[:, -1, :] - goal[np.newaxis], axis=-1)
    return (dists < threshold).mean()


def compute_bminFDE(generated: np.ndarray, gt: np.ndarray) -> float:
    """Probability-weighted minFDE.

    Approximates trajectory probabilities from endpoint distribution
    using softmax over negative squared distances, then computes weighted FDE.

    Args:
        generated: (N, T, 2) generated trajectories in meters
        gt: (T, 2) ground truth in meters

    Returns:
        b-minFDE value
    """
    endpoints = generated[:, -1, :]
    gt_endpoint = gt[-1, :]
    dists = np.linalg.norm(endpoints - gt_endpoint[np.newaxis], axis=-1)
    neg_sq_dists = -(dists ** 2)
    temp = max(neg_sq_dists.std(), 1e-6)
    log_probs = neg_sq_dists / temp
    log_probs -= log_probs.max()
    probs = np.exp(log_probs)
    probs /= probs.sum()
    return float((probs * dists).sum())


def compute_miss_rate(generated: np.ndarray, gt: np.ndarray, threshold: float = 2.0) -> float:
    """Fraction where best-of-K FDE > threshold (1.0 = miss, 0.0 = hit)."""
    min_fde = compute_minFDE(generated, gt)
    return 1.0 if min_fde > threshold else 0.0


def compute_off_road_rate(trajectory: np.ndarray, drivable_areas: list) -> float:
    """Fraction of waypoints outside all drivable areas.

    Args:
        trajectory: (T, 2) trajectory in local meters
        drivable_areas: list of (N, 2) polygon arrays

    Returns:
        fraction of waypoints outside drivable area
    """
    if not drivable_areas:
        return float("nan")
    from matplotlib.path import Path
    paths = []
    for da in drivable_areas:
        da_arr = np.array(da)
        if len(da_arr) >= 3:
            paths.append(Path(da_arr))
    if not paths:
        return float("nan")
    outside_count = 0
    for point in trajectory:
        inside_any = any(path.contains_point(point) for path in paths)
        if not inside_any:
            outside_count += 1
    return outside_count / len(trajectory)


def compute_jerk_stats(trajectory: np.ndarray, dt: float = 0.1) -> dict:
    """Compute jerk statistics for a single trajectory.

    Args:
        trajectory: (T, 2) in meters
        dt: timestep in seconds

    Returns:
        dict with mean, max, violation_fraction (threshold 4.0 m/s^3)
    """
    if len(trajectory) < 4:
        return {"mean": 0.0, "max": 0.0, "violation_fraction": 0.0}
    vel = np.diff(trajectory, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    jerk_mag = np.linalg.norm(jerk, axis=-1)
    return {
        "mean": float(jerk_mag.mean()),
        "max": float(jerk_mag.max()),
        "violation_fraction": float((jerk_mag > 4.0).mean()),
    }


def compute_accel_stats(trajectory: np.ndarray, dt: float = 0.1) -> dict:
    """Compute acceleration statistics.

    Args:
        trajectory: (T, 2) in meters
        dt: timestep in seconds

    Returns:
        dict with mean, max, violation_fraction (threshold 3.0 m/s^2)
    """
    if len(trajectory) < 3:
        return {"mean": 0.0, "max": 0.0, "violation_fraction": 0.0}
    vel = np.diff(trajectory, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    acc_mag = np.linalg.norm(acc, axis=-1)
    return {
        "mean": float(acc_mag.mean()),
        "max": float(acc_mag.max()),
        "violation_fraction": float((acc_mag > 3.0).mean()),
    }


def compute_curvature_stats(trajectory: np.ndarray, dt: float = 0.1) -> dict:
    """Compute curvature statistics.

    Args:
        trajectory: (T, 2) in meters
        dt: timestep in seconds

    Returns:
        dict with mean_curvature, max_curvature, mean_curvature_rate
    """
    if len(trajectory) < 3:
        return {"mean_curvature": 0.0, "max_curvature": 0.0, "mean_curvature_rate": 0.0}
    dx = np.diff(trajectory[:, 0]) / dt
    dy = np.diff(trajectory[:, 1]) / dt
    ddx = np.diff(dx) / dt
    ddy = np.diff(dy) / dt
    dx = dx[1:]
    dy = dy[1:]
    cross = dx * ddy - dy * ddx
    speed_sq = dx ** 2 + dy ** 2 + 1e-8
    curvature = np.abs(cross) / (speed_sq ** 1.5)
    if len(curvature) > 1:
        curv_rate = np.abs(np.diff(curvature)) / dt
        mean_curv_rate = float(curv_rate.mean())
    else:
        mean_curv_rate = 0.0
    return {
        "mean_curvature": float(curvature.mean()),
        "max_curvature": float(curvature.max()),
        "mean_curvature_rate": mean_curv_rate,
    }


def compute_path_efficiency(trajectory: np.ndarray) -> float:
    """Straight-line distance / actual path length.

    A value of 1.0 means perfectly straight. Lower means more circuitous.

    Args:
        trajectory: (T, 2) in meters

    Returns:
        path efficiency ratio
    """
    if len(trajectory) < 2:
        return 1.0
    straight_dist = np.linalg.norm(trajectory[-1] - trajectory[0])
    path_len = np.linalg.norm(np.diff(trajectory, axis=0), axis=-1).sum()
    if path_len < 1e-8:
        return 1.0
    return float(straight_dist / path_len)


def compute_constraint_violation_rate(
    trajectory: np.ndarray,
    drivable_areas: list = None,
    max_jerk: float = 4.0,
    max_accel: float = 3.0,
    dt: float = 0.1,
) -> float:
    """Fraction of frames that violate any constraint.

    Constraints: off-road, jerk > threshold, accel > threshold.

    Args:
        trajectory: (T, 2) in meters
        drivable_areas: list of (N, 2) polygon arrays
        max_jerk: jerk threshold in m/s^3
        max_accel: acceleration threshold in m/s^2
        dt: timestep in seconds

    Returns:
        fraction of violating frames
    """
    T = len(trajectory)
    violations = np.zeros(T, dtype=bool)

    if T >= 4:
        vel = np.diff(trajectory, axis=0) / dt
        acc = np.diff(vel, axis=0) / dt
        jerk = np.diff(acc, axis=0) / dt
        acc_mag = np.linalg.norm(acc, axis=-1)
        jerk_mag = np.linalg.norm(jerk, axis=-1)
        # Acceleration violations: frames 2..T-2
        for i, mag in enumerate(acc_mag):
            violations[i + 1] |= (mag > max_accel)
        # Jerk violations: frames 3..T-3
        for i, mag in enumerate(jerk_mag):
            violations[i + 2] |= (mag > max_jerk)

    if drivable_areas:
        from matplotlib.path import Path
        paths = []
        for da in drivable_areas:
            da_arr = np.array(da)
            if len(da_arr) >= 3:
                paths.append(Path(da_arr))
        if paths:
            for i, point in enumerate(trajectory):
                inside_any = any(path.contains_point(point) for path in paths)
                if not inside_any:
                    violations[i] = True

    return float(violations.mean())


def compute_goal_diversity(sampled_goals: list) -> float:
    """Average pairwise distance between sampled goal endpoints.

    Args:
        sampled_goals: list of (2,) arrays in meters, one per generated trajectory

    Returns:
        average pairwise distance, or 0.0 if fewer than 2 goals
    """
    if len(sampled_goals) < 2:
        return 0.0
    goals = np.array(sampled_goals)
    n = len(goals)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(goals[i] - goals[j]))
    return float(np.mean(dists))


def compute_parking_rate(is_parking_list: list) -> dict:
    """Compute parking scenario rate and statistics.

    Args:
        is_parking_list: list of bool/float — is_parking flags per scenario

    Returns:
        dict with parking_count, total, rate
    """
    total = len(is_parking_list)
    parking_count = sum(1 for x in is_parking_list if float(x) > 0.5)
    return {
        "parking_count": parking_count,
        "total": total,
        "rate": parking_count / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="output/samples/results.pt")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for evaluation outputs (default: same as results)")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"No results file at {results_path}. Run inference first.")
        return

    results = torch.load(results_path, weights_only=False)

    timestamp = datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir) / timestamp
    else:
        output_dir = results_path.parent / "eval" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Eval output directory: {output_dir}")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        # Existing
        "minADE": [],
        "minFDE": [],
        "endpoint_error": [],
        "diversity": [],
        "goal_hit_rate": [],
        # New
        "bminFDE": [],
        "miss_rate": [],
        "off_road_rate": [],
        "jerk_mean": [],
        "jerk_max": [],
        "jerk_violation_frac": [],
        "accel_mean": [],
        "accel_max": [],
        "accel_violation_frac": [],
        "curvature_mean": [],
        "curvature_max": [],
        "curvature_rate_mean": [],
        "path_efficiency": [],
        "constraint_violation_rate": [],
        "goal_diversity": [],
        # Collision partner
        "collision_rate": [],
        "collision_mean_min_dist": [],
        "collision_rate_strict": [],
    }
    is_parking_flags = []

    for r in results:
        gt = r["gt_local"]           # (T, 2) in meters
        gen = np.array(r["generated_local"])  # (N, T, 2) in meters
        goal = r["goal_local"]       # (2,) in meters

        # Collect parking flag
        is_parking_flags.append(float(r.get("is_parking", 0.0)))

        # Existing metrics
        metrics["minADE"].append(compute_minADE(gen, gt))
        metrics["minFDE"].append(compute_minFDE(gen, gt))
        metrics["endpoint_error"].append(compute_endpoint_error(gen, goal))
        metrics["diversity"].append(compute_diversity(gen))
        metrics["goal_hit_rate"].append(compute_goal_hit_rate(gen, goal))

        # New metrics
        metrics["bminFDE"].append(compute_bminFDE(gen, gt))
        metrics["miss_rate"].append(compute_miss_rate(gen, gt))

        # Find best trajectory (closest to GT) for kinematic metrics
        best_idx = np.argmin(np.linalg.norm(gen - gt[np.newaxis], axis=-1).mean(axis=-1))
        best_traj = gen[best_idx]

        # Off-road rate (requires map data)
        drivable_areas = r.get("drivable_areas_local", [])
        if drivable_areas:
            metrics["off_road_rate"].append(compute_off_road_rate(best_traj, drivable_areas))
        else:
            metrics["off_road_rate"].append(float("nan"))

        # Kinematic metrics on best trajectory
        jerk_stats = compute_jerk_stats(best_traj)
        metrics["jerk_mean"].append(jerk_stats["mean"])
        metrics["jerk_max"].append(jerk_stats["max"])
        metrics["jerk_violation_frac"].append(jerk_stats["violation_fraction"])

        accel_stats = compute_accel_stats(best_traj)
        metrics["accel_mean"].append(accel_stats["mean"])
        metrics["accel_max"].append(accel_stats["max"])
        metrics["accel_violation_frac"].append(accel_stats["violation_fraction"])

        curv_stats = compute_curvature_stats(best_traj)
        metrics["curvature_mean"].append(curv_stats["mean_curvature"])
        metrics["curvature_max"].append(curv_stats["max_curvature"])
        metrics["curvature_rate_mean"].append(curv_stats["mean_curvature_rate"])

        metrics["path_efficiency"].append(compute_path_efficiency(best_traj))

        metrics["constraint_violation_rate"].append(
            compute_constraint_violation_rate(best_traj, drivable_areas)
        )

        # Goal diversity (from sampled goals if available)
        sampled_goals = r.get("sampled_goals_local", [])
        metrics["goal_diversity"].append(compute_goal_diversity(sampled_goals))

        # Collision partner metrics
        partner = r.get("partner")
        if partner is not None and partner.get("collision_rate") is not None:
            metrics["collision_rate"].append(partner["collision_rate"])
            metrics["collision_mean_min_dist"].append(partner["collision_mean_min_dist"])
            metrics["collision_rate_strict"].append(partner["collision_rate_strict"])

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    summary = {}
    for name, values in metrics.items():
        vals = [v for v in values if not np.isnan(v)]
        if not vals:
            mean, std = float("nan"), float("nan")
        else:
            mean = np.mean(vals)
            std = np.std(vals)
        print(f"  {name:30s}: {mean:.4f} +/- {std:.4f}")
        summary[name] = {"mean": float(mean), "std": float(std)}

    # Parking rate
    parking_stats = compute_parking_rate(is_parking_flags)
    print(f"\n  {'parking_count':30s}: {parking_stats['parking_count']}")
    print(f"  {'parking_rate':30s}: {parking_stats['rate']:.4f}")
    summary["parking"] = parking_stats

    # Metrics on non-parking scenarios only
    non_parking_mask = [float(x) < 0.5 for x in is_parking_flags]
    non_parking_count = sum(non_parking_mask)
    if non_parking_count > 0 and non_parking_count < len(is_parking_flags):
        print(f"\n  --- Non-parking scenarios only ({non_parking_count}/{len(is_parking_flags)}) ---")
        summary["non_parking"] = {}
        for name, values in metrics.items():
            vals = [v for v, m in zip(values, non_parking_mask) if m and not np.isnan(v)]
            if not vals:
                continue
            mean = np.mean(vals)
            std = np.std(vals)
            print(f"  {name:30s}: {mean:.4f} +/- {std:.4f}")
            summary["non_parking"][name] = {"mean": float(mean), "std": float(std)}

    # Save metrics JSON
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics.json to {output_dir / 'metrics.json'}")

    # Visualization: metric distributions
    # Only include non-NaN metrics for plotting
    plot_metrics = {k: [v for v in vs if not np.isnan(v)] for k, vs in metrics.items()}
    plot_metrics = {k: v for k, v in plot_metrics.items() if v}
    fig = plot_metric_distributions(plot_metrics)
    save_figure(fig, figures_dir / "eval_metric_distributions.png")
    print(f"Saved eval_metric_distributions.png")

    # Visualization: per-scenario breakdown
    fig = plot_per_scenario_breakdown(plot_metrics)
    if fig is not None:
        save_figure(fig, figures_dir / "eval_per_scenario.png")
        print(f"Saved eval_per_scenario.png")

    # Best and worst scenarios by minADE
    ade_values = np.array(metrics["minADE"])
    best_idx = int(np.argmin(ade_values))
    worst_idx = int(np.argmax(ade_values))

    for label, idx in [("best", best_idx), ("worst", worst_idx)]:
        r = results[idx]
        gen = np.array(r["generated_local"], dtype=np.float32)
        gt = np.array(r["gt_local"], dtype=np.float32)
        goal = np.array(r["goal_local"], dtype=np.float32)

        fig = plot_trajectories(
            gen, gt=gt, goal=goal,
            title=f"{label.capitalize()} Scenario (idx={idx}, minADE={ade_values[idx]:.3f})",
        )
        save_figure(fig, figures_dir / f"eval_{label}_scenario.png")
        print(f"Saved eval_{label}_scenario.png")

    print("=" * 50)


if __name__ == "__main__":
    main()
