"""Generate synthetic Argoverse 2 sample scenarios (parquet + json) matching the official av2 data format.

Usage:
    python generate_av2_samples.py              # default 2 scenarios
    python generate_av2_samples.py -n 10         # generate 10 scenarios
    python generate_av2_samples.py -n 50 -o data/train  # custom output dir
"""

import argparse
import uuid
import numpy as np
import pandas as pd
import json
from pathlib import Path

NUM_TIMESTEPS = 110  # 0~109, 11 seconds @10Hz
START_TS_NS = 31536000000000000
END_TS_NS = START_TS_NS + (NUM_TIMESTEPS - 1) * 100_000_000

CITIES = ["PIT", "ATX", "MIA", "PAO", "WDC"]
LANE_MARK_TYPES = [
    "DASH_SOLID_YELLOW", "DASH_SOLID_WHITE", "DASHED_WHITE", "DASHED_YELLOW",
    "DOUBLE_SOLID_YELLOW", "DOUBLE_SOLID_WHITE", "SOLID_YELLOW", "SOLID_WHITE",
    "SOLID_DASH_WHITE", "SOLID_DASH_YELLOW", "NONE",
]
OBJECT_TYPES = ["vehicle", "pedestrian", "cyclist", "motorcyclist", "bus"]
TRACK_CATEGORIES = [0, 1, 2, 3]  # TRACK_FRAGMENT, UNSCORED_TRACK, SCORED_TRACK, FOCAL_TRACK


def make_3d_points(xy_list, z=0.0):
    return [{"x": float(p[0]), "y": float(p[1]), "z": z} for p in xy_list]


def build_row(observed, track_id, object_type, object_category, timestep,
              position_x, position_y, heading, velocity_x, velocity_y,
              scenario_id, focal_track_id, city, map_id):
    return {
        "observed": observed,
        "track_id": track_id,
        "object_type": object_type,
        "object_category": object_category,
        "timestep": timestep,
        "position_x": position_x,
        "position_y": position_y,
        "heading": heading,
        "velocity_x": velocity_x,
        "velocity_y": velocity_y,
        "scenario_id": scenario_id,
        "start_timestamp": START_TS_NS,
        "end_timestamp": END_TS_NS,
        "num_timestamps": NUM_TIMESTEPS,
        "focal_track_id": focal_track_id,
        "city": city,
        "map_id": map_id,
    }


def generate_lane_segment(lane_id, cx, cy, heading, length, lane_width, is_intersection,
                          predecessors, successors, left_neighbor_id, right_neighbor_id):
    """Generate one lane segment with left/right boundaries offset from centerline."""
    n_pts = np.random.randint(4, 8)
    dx, dy = np.cos(heading), np.sin(heading)
    nx, ny = -np.sin(heading), np.cos(heading)  # normal (left)

    centerline_pts = []
    for i in range(n_pts):
        frac = i / (n_pts - 1)
        px = cx + frac * length * dx
        py = cy + frac * length * dy
        centerline_pts.append([px, py])

    left_pts = [[p[0] + lane_width / 2 * nx, p[1] + lane_width / 2 * ny] for p in centerline_pts]
    right_pts = [[p[0] - lane_width / 2 * nx, p[1] - lane_width / 2 * ny] for p in centerline_pts]

    return {
        "id": lane_id,
        "is_intersection": is_intersection,
        "lane_type": "VEHICLE",
        "left_lane_boundary": make_3d_points(left_pts),
        "right_lane_boundary": make_3d_points(right_pts),
        "left_lane_mark_type": np.random.choice(LANE_MARK_TYPES),
        "right_lane_mark_type": np.random.choice(LANE_MARK_TYPES),
        "predecessors": predecessors,
        "successors": successors,
        "left_neighbor_id": left_neighbor_id,
        "right_neighbor_id": right_neighbor_id,
    }


def generate_map(rng, focal_start_x, focal_start_y, focal_heading, n_lanes):
    """Generate a local road network around the focal vehicle's path."""
    lane_segments = {}
    lane_width = rng.uniform(3.0, 4.5)

    # Generate lanes along and around the focal path
    base_id = rng.integers(1000, 9999)
    for i in range(n_lanes):
        lane_id = int(base_id + i)
        # Spread lanes laterally around focal heading
        lateral_offset = (i - n_lanes // 2) * lane_width
        nx, ny = -np.sin(focal_heading), np.cos(focal_heading)
        cx = focal_start_x + lateral_offset * nx
        cy = focal_start_y + lateral_offset * ny
        length = rng.uniform(40, 120)
        is_intersection = rng.random() < 0.2

        preds = []
        succs = []
        left_id = int(base_id + i + 1) if i < n_lanes - 1 else None
        right_id = int(base_id + i - 1) if i > 0 else None

        # Connect sequential lanes
        if i > 0:
            preds.append(int(base_id + i - 1))
        if i < n_lanes - 1:
            succs.append(int(base_id + i + 1))

        lane_segments[str(lane_id)] = generate_lane_segment(
            lane_id, cx, cy, focal_heading, length, lane_width,
            is_intersection, preds, succs, left_id, right_id,
        )

    # Drivable area
    da_id = rng.integers(5000, 9999)
    half_w = n_lanes * lane_width / 2
    nx, ny = -np.sin(focal_heading), np.cos(focal_heading)
    dx, dy = np.cos(focal_heading), np.sin(focal_heading)
    corners = [
        [focal_start_x - half_w * nx, focal_start_y - half_w * ny],
        [focal_start_x + 60 * dx - half_w * nx, focal_start_y + 60 * dy - half_w * ny],
        [focal_start_x + 60 * dx + half_w * nx, focal_start_y + 60 * dy + half_w * ny],
        [focal_start_x + half_w * nx, focal_start_y + half_w * ny],
        [focal_start_x - half_w * nx, focal_start_y - half_w * ny],
    ]
    drivable_areas = {
        str(da_id): {"id": da_id, "area_boundary": make_3d_points(corners)},
    }

    # Pedestrian crossing (optional)
    ped_crossings = {}
    if rng.random() < 0.3:
        pc_id = rng.integers(7000, 9999)
        px = focal_start_x + rng.uniform(20, 40) * dx
        py = focal_start_y + rng.uniform(20, 40) * dy
        ped_crossings[str(pc_id)] = {
            "id": pc_id,
            "edge1": make_3d_points([[px - 3 * nx, py - 3 * ny], [px + 3 * nx, py + 3 * ny]]),
            "edge2": make_3d_points([[px - 3 * nx + 3 * dx, py - 3 * ny + 3 * dy],
                                     [px + 3 * nx + 3 * dx, py + 3 * ny + 3 * dy]]),
        }

    return {
        "lane_segments": lane_segments,
        "drivable_areas": drivable_areas,
        "pedestrian_crossings": ped_crossings,
    }


def generate_focal_trajectory(rng, start_x, start_y, heading):
    """Generate focal vehicle trajectory: straight or with a turn."""
    traj = []
    speed = rng.uniform(2.0, 8.0)
    turn = rng.random() < 0.3  # 30% chance of turning

    if turn:
        turn_dir = rng.choice([-1, 1])
        turn_start = rng.integers(50, 70)
        turn_end = turn_start + rng.integers(10, 25)
        turn_angle = turn_dir * rng.uniform(np.pi / 6, np.pi / 2)
    else:
        turn_start = turn_end = 999
        turn_angle = 0.0

    x, y = start_x, start_y
    cur_heading = heading
    for t in range(NUM_TIMESTEPS):
        vx = speed * np.cos(cur_heading)
        vy = speed * np.sin(cur_heading)
        traj.append((x, y, cur_heading, vx, vy))
        if turn_start <= t < turn_end:
            cur_heading += turn_angle / (turn_end - turn_start)
        x += vx * 0.1
        y += vy * 0.1

    return traj


def generate_neighbor_trajectory(rng, focal_traj, lane_width):
    """Generate a neighbor agent trajectory relative to focal."""
    offset = rng.choice([-1, 1]) * lane_width
    speed_ratio = rng.uniform(0.6, 1.2)
    appear_after = rng.integers(0, 30)
    disappear_before = rng.integers(80, 110)

    traj = []
    for t in range(NUM_TIMESTEPS):
        if t < appear_after or t >= disappear_before:
            continue
        fx, fy, fh, fvx, fvy = focal_traj[t]
        nx_dir = -np.sin(fh)
        ny_dir = np.cos(fh)
        x = fx + offset * nx_dir + rng.uniform(-1, 1)
        y = fy + offset * ny_dir + rng.uniform(-1, 1)
        heading = fh + rng.uniform(-0.05, 0.05)
        vx = fvx * speed_ratio + rng.uniform(-0.3, 0.3)
        vy = fvy * speed_ratio + rng.uniform(-0.3, 0.3)
        traj.append((t, x, y, heading, vx, vy))
    return traj


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def generate_scenario(rng, output_dir):
    """Generate one complete scenario (parquet + json).

    Uses the same integer map_id for both the parquet map_id field and the
    JSON filename so the dataset loader can find the map via the official
    layout path: log_map_archive_{map_id}.json
    """
    scenario_id = str(uuid.uuid4())
    short_id = scenario_id[:8]
    city = rng.choice(CITIES)
    map_id = int(rng.integers(1000, 99999))

    # Focal vehicle start state
    start_x = rng.uniform(0, 500)
    start_y = rng.uniform(0, 500)
    heading = rng.uniform(-np.pi, np.pi)
    lane_width = rng.uniform(3.0, 4.5)

    # Generate focal trajectory
    focal_traj = generate_focal_trajectory(rng, start_x, start_y, heading)

    # Build parquet rows
    rows = []
    for t in range(NUM_TIMESTEPS):
        x, y, h, vx, vy = focal_traj[t]
        rows.append(build_row(
            observed=t < 50, track_id="0", object_type="vehicle",
            object_category=3, timestep=t,
            position_x=x, position_y=y, heading=h,
            velocity_x=vx, velocity_y=vy,
            scenario_id=scenario_id, focal_track_id="0",
            city=city, map_id=map_id,
        ))

    # Generate neighbors (1~6)
    n_neighbors = rng.integers(1, 7)
    for i in range(n_neighbors):
        nb_traj = generate_neighbor_trajectory(rng, focal_traj, lane_width)
        if not nb_traj:
            continue
        nb_type = rng.choice(["vehicle", "vehicle", "vehicle", "bus", "cyclist"])
        nb_category = rng.choice([0, 1, 2])  # fragment, unscored, scored
        for t, x, y, h, vx, vy in nb_traj:
            rows.append(build_row(
                observed=t < 50, track_id=str(i + 1), object_type=nb_type,
                object_category=nb_category, timestep=t,
                position_x=x, position_y=y, heading=h,
                velocity_x=vx, velocity_y=vy,
                scenario_id=scenario_id, focal_track_id="0",
                city=city, map_id=map_id,
            ))

    # Optional pedestrian
    if rng.random() < 0.3:
        ped_start_t = rng.integers(30, 60)
        ped_end_t = min(ped_start_t + rng.integers(15, 40), 109)
        ped_x = focal_traj[ped_start_t][0] + rng.uniform(-5, 5)
        ped_y = focal_traj[ped_start_t][1] + rng.uniform(-5, 5)
        ped_heading = rng.uniform(-np.pi, np.pi)
        ped_speed = rng.uniform(0.5, 1.5)
        for t in range(ped_start_t, ped_end_t):
            rows.append(build_row(
                observed=t < 50, track_id=str(n_neighbors + 1),
                object_type="pedestrian", object_category=0, timestep=t,
                position_x=ped_x, position_y=ped_y,
                heading=ped_heading,
                velocity_x=ped_speed * np.cos(ped_heading),
                velocity_y=ped_speed * np.sin(ped_heading),
                scenario_id=scenario_id, focal_track_id="0",
                city=city, map_id=map_id,
            ))
            ped_x += ped_speed * np.cos(ped_heading) * 0.1
            ped_y += ped_speed * np.sin(ped_heading) * 0.1

    df = pd.DataFrame(rows)
    df.to_parquet(output_dir / f"scenario_{short_id}.parquet", engine="pyarrow")

    # Generate map — filename uses map_id (matches parquet map_id field)
    n_lanes = rng.integers(2, 6)
    map_data = generate_map(rng, start_x, start_y, heading, n_lanes)

    with open(output_dir / f"log_map_archive_{map_id}.json", "w") as f:
        json.dump(map_data, f, indent=2, default=_json_default)

    return short_id


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Argoverse 2 sample scenarios")
    parser.add_argument("-n", "--num", type=int, default=2, help="Number of scenarios to generate (default: 2)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory (default: av2_sample/)")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else Path(__file__).parent / "av2_sample"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    ids = []
    for i in range(args.num):
        sid = generate_scenario(rng, output_dir)
        ids.append(sid)
        if (i + 1) % 10 == 0 or i == args.num - 1:
            print(f"  Generated {i + 1}/{args.num}")

    print(f"\nDone. Generated {args.num} scenarios in {output_dir}/")
    print(f"Files: scenario_{{id}}.parquet + log_map_archive_{{id}}.json")
    print(f"IDs: {ids}")


if __name__ == "__main__":
    main()
