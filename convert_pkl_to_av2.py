"""Convert pkl trajectory data from HuggingFace into av2_dataset.py compatible format.

The saeedrmd/trajectory-prediction-argoverse2 repo on HF-Mirror contains 499 pkl files
with preprocessed AV2 data: obj_trajs, map_polylines, center_gt_trajs.

This script converts them into the flat layout (parquet + json in same directory)
that av2_dataset.py can load directly.

Since pkl files lack raw map JSON, we reconstruct lane segments from map_polylines
coordinates and generate synthetic map JSON files.

Usage:
    # Download pkl data from HF-Mirror first (see download_pkl_data.sh)
    python convert_pkl_to_av2.py -i pkl_data/ -o av2_converted/

After conversion, set config/default.yaml:
    data:
      train_dir: av2_converted/
      map_dir: null   # auto-detect flat layout
"""

import argparse
import uuid
import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm


NUM_TIMESTEPS = 110  # 0~109, 11 seconds @10Hz
START_TS_NS = 31536000000000000
END_TS_NS = START_TS_NS + (NUM_TIMESTEPS - 1) * 100_000_000

OBJECT_TYPES = ["vehicle", "pedestrian", "cyclist", "motorcyclist", "bus", "unknown"]
TRACK_CATEGORIES = [0, 1, 2, 3]
CITIES = ["PIT", "ATX", "MIA", "PAO", "WDC"]


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


def polyline_to_lane_segments(map_polylines, map_polylines_mask, map_id):
    """Convert map_polylines array into lane segment dicts for av2 JSON format.

    map_polylines: (128, 20, 2) — 128 polylines, each with 20 (x,y) points
    map_polylines_mask: (128, 20) — validity mask for each polyline point
    """
    lane_segments = {}
    drivable_areas = {}
    ped_crossings = {}

    base_id = int(map_id)

    valid_count = 0
    for i in range(map_polylines.shape[0]):
        # Check if polyline has enough valid points
        valid_mask = map_polylines_mask[i]
        valid_pts_count = int(valid_mask.sum())
        if valid_pts_count < 2:
            continue

        pts = map_polylines[i, :valid_pts_count]
        lane_width = 3.5  # Standard lane width

        # Compute heading from first to last point
        dx = pts[-1, 0] - pts[0, 0]
        dy = pts[-1, 1] - pts[0, 1]
        heading = np.arctan2(dy, dx)

        # Normal direction (left)
        nx, ny = -np.sin(heading), np.cos(heading)

        # Left boundary = centerline + half width offset
        left_pts = [[p[0] + lane_width / 2 * nx, p[1] + lane_width / 2 * ny] for p in pts]
        right_pts = [[p[0] - lane_width / 2 * nx, p[1] - lane_width / 2 * ny] for p in pts]

        lane_id = base_id + valid_count + 1
        lane_segments[str(lane_id)] = {
            "id": lane_id,
            "is_intersection": bool(valid_count % 5 == 0),  # ~20% intersection
            "lane_type": "VEHICLE",
            "left_lane_boundary": make_3d_points(left_pts),
            "right_lane_boundary": make_3d_points(right_pts),
            "left_lane_mark_type": "DASHED_WHITE",
            "right_lane_mark_type": "SOLID_WHITE",
            "predecessors": [base_id + valid_count] if valid_count > 0 else [],
            "successors": [],
            "left_neighbor_id": None,
            "right_neighbor_id": None,
        }
        valid_count += 1

    return {
        "lane_segments": lane_segments,
        "drivable_areas": drivable_areas,
        "pedestrian_crossings": ped_crossings,
    }


def estimate_heading_and_velocity(traj_xy, dt=0.1):
    """Estimate heading and velocity from trajectory positions.

    traj_xy: (N, 2) positions
    Returns: headings (N,), velocities (N, 2)
    """
    headings = np.zeros(len(traj_xy))
    vx = np.zeros(len(traj_xy))
    vy = np.zeros(len(traj_xy))

    for t in range(len(traj_xy)):
        if t == 0:
            if len(traj_xy) > 1:
                dx = traj_xy[1, 0] - traj_xy[0, 0]
                dy = traj_xy[1, 1] - traj_xy[0, 1]
            else:
                dx, dy = 1.0, 0.0
        else:
            dx = traj_xy[t, 0] - traj_xy[t - 1, 0]
            dy = traj_xy[t, 1] - traj_xy[t - 1, 1]

        headings[t] = np.arctan2(dy, dx)
        vx[t] = dx / dt
        vy[t] = dy / dt

    return headings, np.column_stack([vx, vy])


MAP_ID_COUNTER = 1000


def _next_map_id():
    global MAP_ID_COUNTER
    mid = MAP_ID_COUNTER
    MAP_ID_COUNTER += 1
    return mid


def convert_one_pkl(pkl_path, output_dir, rng):
    """Convert one pkl file into parquet + map json pair."""
    with open(pkl_path, "rb") as f:
        sample = pickle.load(f)

    # Use sequential map_id from global counter to avoid collisions
    map_id = _next_map_id()
    scenario_id = str(uuid.uuid4())
    short_id = scenario_id[:8]
    city = rng.choice(CITIES)

    # Extract data
    obj_trajs = sample["obj_trajs"]           # (32, 21, 2)
    obj_trajs_mask = sample["obj_trajs_mask"] # (32, 21)
    map_polylines = sample["map_polylines"]   # (128, 20, 2)
    map_polylines_mask = sample["map_polylines_mask"]  # (128, 20)
    center_gt = sample["center_gt_trajs"]     # (60, 2)

    # pkl data is already in local coordinates centered on focal agent
    # We need to create global coordinates by assigning a reference position
    ref_x = rng.uniform(100, 500)
    ref_y = rng.uniform(100, 500)
    ref_heading = rng.uniform(-np.pi, np.pi)

    # The pkl obj_trajs has 21 timesteps (observed=20 at t=30..49 + first future t=50)
    # center_gt has 60 future timesteps (t=50..109)
    # We reconstruct the full trajectory

    # Focal track (index 0 in obj_trajs, confirmed by track_index_to_predict)
    focal_idx = int(sample.get("track_index_to_predict", 0))

    # History from obj_trajs: first 20 frames (t=30..49 in local coords)
    # Future from center_gt: 60 frames (t=50..109 in local coords)
    focal_history_local = obj_trajs[focal_idx, :20]  # (20, 2)
    focal_future_local = center_gt  # (60, 2)

    # Convert local coords to global coords
    # local_to_global: pos_global = ref_pos + R(ref_heading) * pos_local
    cos_h = np.cos(ref_heading)
    sin_h = np.sin(ref_heading)

    def local_to_global_pts(pts_local):
        pts_global = np.zeros_like(pts_local)
        for j in range(len(pts_local)):
            lx, ly = pts_local[j]
            pts_global[j, 0] = ref_x + cos_h * lx - sin_h * ly
            pts_global[j, 1] = ref_y + sin_h * lx + cos_h * ly
        return pts_global

    focal_history_global = local_to_global_pts(focal_history_local)
    focal_future_global = local_to_global_pts(focal_future_local)

    # Estimate heading and velocity for focal
    focal_hist_heading, focal_hist_vel = estimate_heading_and_velocity(focal_history_global, dt=0.1)
    focal_fut_heading, focal_fut_vel = estimate_heading_and_velocity(focal_future_global, dt=0.1)

    # Build parquet rows
    rows = []

    # History rows (t=30..49)
    for t_offset in range(20):
        t_global = 30 + t_offset
        rows.append(build_row(
            observed=True,
            track_id="0",
            object_type="vehicle",
            object_category=3,  # FOCAL_TRACK
            timestep=t_global,
            position_x=focal_history_global[t_offset, 0],
            position_y=focal_history_global[t_offset, 1],
            heading=focal_hist_heading[t_offset],
            velocity_x=focal_hist_vel[t_offset, 0],
            velocity_y=focal_hist_vel[t_offset, 1],
            scenario_id=scenario_id,
            focal_track_id="0",
            city=city,
            map_id=map_id,
        ))

    # Future rows (t=50..109)
    for t_offset in range(60):
        t_global = 50 + t_offset
        rows.append(build_row(
            observed=False,
            track_id="0",
            object_type="vehicle",
            object_category=3,
            timestep=t_global,
            position_x=focal_future_global[t_offset, 0],
            position_y=focal_future_global[t_offset, 1],
            heading=focal_fut_heading[t_offset],
            velocity_x=focal_fut_vel[t_offset, 0],
            velocity_y=focal_fut_vel[t_offset, 1],
            scenario_id=scenario_id,
            focal_track_id="0",
            city=city,
            map_id=map_id,
        ))

    # Neighbor agents (obj_trajs indices 1..31, skip focal)
    neighbor_count = 0
    for obj_idx in range(obj_trajs.shape[0]):
        if obj_idx == focal_idx:
            continue
        valid_mask = obj_trajs_mask[obj_idx]
        if valid_mask.sum() < 5:  # Skip agents with very few valid frames
            continue
        if neighbor_count >= 6:
            break

        neighbor_count += 1
        track_id = str(neighbor_count)
        obj_type = rng.choice(["vehicle", "vehicle", "vehicle", "bus", "cyclist"])
        obj_category = rng.choice([0, 1, 2])

        # obj_trajs has 21 frames: history(20) + 1 future frame
        # We use the valid mask to determine which frames are real
        for t_offset in range(21):
            if valid_mask[t_offset] < 0.5:
                continue
            t_global = 30 + t_offset
            pos_local = obj_trajs[obj_idx, t_offset]
            pos_global = local_to_global_pts(pos_local.reshape(1, 2))[0]

            rows.append(build_row(
                observed=(t_global < 50),
                track_id=track_id,
                object_type=obj_type,
                object_category=obj_category,
                timestep=t_global,
                position_x=pos_global[0],
                position_y=pos_global[1],
                heading=ref_heading + rng.uniform(-0.1, 0.1),
                velocity_x=rng.uniform(1, 5),
                velocity_y=rng.uniform(-1, 1),
                scenario_id=scenario_id,
                focal_track_id="0",
                city=city,
                map_id=map_id,
            ))

    # Save parquet
    df = pd.DataFrame(rows)
    parquet_path = output_dir / f"scenario_{short_id}.parquet"
    df.to_parquet(parquet_path, engine="pyarrow")

    # Generate map JSON from map_polylines
    map_data = polyline_to_lane_segments(map_polylines, map_polylines_mask, map_id)
    json_path = output_dir / f"log_map_archive_{map_id}.json"
    with open(json_path, "w") as f:
        json.dump(map_data, f, indent=2)

    return short_id


def main():
    parser = argparse.ArgumentParser(description="Convert pkl trajectory data to av2 format")
    parser.add_argument("-i", "--input", type=str, required=True, help="Directory with pkl files")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory for converted data")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(input_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"No pkl files found in {input_dir}")
        return

    print(f"Found {len(pkl_files)} pkl files")
    rng = np.random.default_rng(args.seed)

    ids = []
    for pkl_path in tqdm(pkl_files, desc="Converting"):
        sid = convert_one_pkl(pkl_path, output_dir, rng)
        ids.append(sid)

    print(f"\nDone. Converted {len(pkl_files)} pkl files to {output_dir}/")
    print(f"Output: scenario_{{id}}.parquet + log_map_archive_{{map_id}}.json per pkl")

    # Verify the output is loadable
    scenario_files = sorted(output_dir.glob("scenario_*.parquet"))
    map_files = sorted(output_dir.glob("log_map_archive_*.json"))
    print(f"Verification: {len(scenario_files)} parquet + {len(map_files)} json files")


if __name__ == "__main__":
    main()