"""Argoverse 2 dataset for goal-conditioned trajectory generation.

All conditions (goal, map, neighbors) are always provided.
CFG condition dropout is applied during training for classifier-free guidance.
Diffusion model learns residual = trajectory - prior, where prior is a
Hermite spline (default) or linear interpolation from history endpoint to goal.
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional

from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet
from av2.map.map_api import ArgoverseStaticMap

from data.coordinate_utils import global_to_local, velocity_global_to_local
from data.map_encoding import encode_map_tokens
from data.neighbor_encoding import encode_neighbor_tokens
from data.normalization import normalize, denormalize, normalize_residual, compute_prior, compute_hermite_prior, compute_lane_prior
from data.normalization import (
    compute_chord_dir, residual_to_chord_frame, chord_frame_to_residual,
    normalize_residual_chord, denormalize_residual_chord,
    pack_chord, unpack_chord,
    DT, VELOCITY_SCALE, ACCELERATION_SCALE,
)


def _compute_centerline_from_boundaries(left_xyz, right_xyz):
    """Compute centerline from left/right lane boundaries by midpoints.

    Left and right boundaries may have different numbers of points,
    so we resample both to a common arc-length parameterization.
    """
    # Compute arc lengths for each boundary
    def arc_lengths(pts):
        diffs = np.diff(pts[:, :2], axis=0)
        seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
        s = np.zeros(len(pts))
        s[1:] = np.cumsum(seg_lens)
        return s

    s_left = arc_lengths(left_xyz)
    s_right = arc_lengths(right_xyz)
    total_left = s_left[-1]
    total_right = s_right[-1]

    if total_left < 0.5 or total_right < 0.5:
        return None

    # Resample both boundaries at N evenly-spaced arc-length points
    N = max(len(left_xyz), len(right_xyz), 10)
    s_interp = np.linspace(0, 1, N)

    left_interp = np.column_stack([
        np.interp(s_interp, s_left / total_left, left_xyz[:, i])
        for i in range(left_xyz.shape[1])
    ])
    right_interp = np.column_stack([
        np.interp(s_interp, s_right / total_right, right_xyz[:, i])
        for i in range(right_xyz.shape[1])
    ])

    center_xyz = (left_interp + right_interp) / 2.0
    return center_xyz


class Argoverse2Dataset(Dataset):
    """Loads Argoverse 2 scenarios and produces model inputs.

    Supports two directory layouts:

    1. **Official Argoverse 2 layout** (recommended):
       data_dir points to the split directory (e.g. .../train/), and map_dir
       points to the log_map_archive directory. Each scenario lives in its own
       subdirectory, and maps are stored centrally.
           data_dir/
           ├── {scenario_id}/scenario_{scenario_id}.parquet
           └── ...
           map_dir/   (e.g. .../log_map_archive/)
           ├── {log_id}/log_map_archive_{log_id}.json
           └── ...

    2. **Flat layout** (synthetic / av2_sample):
       Both parquet and JSON files live in the same directory.
       Set map_dir=None (default) and it will be inferred from data_dir.
           data_dir/
           ├── scenario_{id}.parquet
           ├── log_map_archive_{id}.json
           └── ...

    The map_dir is resolved with this priority:
      a) Explicit map_dir parameter
      b) data_dir/../log_map_archive/  (official layout: data_dir=train/ → map_dir=log_map_archive/)
      c) data_dir itself (flat layout: map files are in same dir as parquet files)
    """

    def __init__(
        self,
        data_dir: str,
        n_future: int = 60,
        n_history: int = 20,
        history_start: int = 30,
        n_lanes: int = 24,
        lane_feat_dim: int = 46,
        n_neighbors: int = 6,
        split: str = "train",
        drop_goal_p: float = 0.1,
        drop_map_p: float = 0.05,
        drop_neighbor_p: float = 0.1,
        return_scene_data: bool = False,
        map_dir: Optional[str] = None,
        prior_type: str = "hermite",
        residual_frame: str = "chord",
        filter_parking: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.n_future = n_future
        self.n_history = n_history
        self.history_start = history_start
        self.n_lanes = n_lanes
        self.lane_feat_dim = lane_feat_dim
        self.n_neighbors = n_neighbors
        self.split = split
        self.drop_goal_p = drop_goal_p
        self.drop_map_p = drop_map_p
        self.drop_neighbor_p = drop_neighbor_p
        self.return_scene_data = return_scene_data
        self.prior_type = prior_type
        self.residual_frame = residual_frame
        self.filter_parking = filter_parking

        # Resolve map directory
        if map_dir is not None:
            self.map_dir = Path(map_dir)
        else:
            # Try official layout: data_dir/../log_map_archive/
            candidate = self.data_dir.parent / "log_map_archive"
            if candidate.is_dir():
                self.map_dir = candidate
            else:
                # Flat layout: maps are in same directory as parquet files
                self.map_dir = self.data_dir

        self.scenario_files = sorted(self.data_dir.glob("**/scenario_*.parquet"))
        if not self.scenario_files:
            raise FileNotFoundError(f"No scenario parquet files found in {data_dir}")

        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        # Index mapping for parking filter: _valid_indices[i] gives the
        # physical file index for virtual index i.  None until apply_parking_filter().
        self._valid_indices: Optional[list] = None

    def __len__(self) -> int:
        if self._valid_indices is not None:
            return len(self._valid_indices)
        return len(self.scenario_files)

    @property
    def training(self) -> bool:
        return self.split == "train"

    def apply_parking_filter(self):
        """Scan cache for parking scenarios and build _valid_indices mapping.

        Must be called after preload_cache() so that is_parking is available
        for every scenario.  When filter_parking=True, __len__ and __getitem__
        will only expose non-parking scenarios.
        """
        if not self.filter_parking:
            return
        if not self._cache:
            raise RuntimeError("apply_parking_filter() requires preload_cache() first")

        n_total = len(self.scenario_files)
        valid = []
        n_parking = 0
        for phys_idx in range(n_total):
            item = self._cache.get(phys_idx)
            if item is not None and item["is_parking"].item() > 0.5:
                n_parking += 1
                continue
            valid.append(phys_idx)

        self._valid_indices = valid
        print(f"  Parking filter: {n_parking}/{n_total} scenarios excluded "
              f"({n_parking/n_total*100:.1f}%), {len(valid)} remaining")

    def preload_cache(self, cache_path: str = "output/dataset_cache.pt"):
        """Load all samples into memory before DataLoader workers are forked.

        If a disk cache exists, load it directly (seconds). Otherwise, parse
        all parquet + JSON files and save the result for future runs.
        """
        from tqdm import tqdm
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            print(f"  Loading dataset cache from {cache_path} ...")
            data = torch.load(cache_path, map_location="cpu")
            self._cache = {int(k): v for k, v in data.items()}
            n = len(self._cache)
            est_mb = 0.0
            for item in self._cache.values():
                for v in item.values():
                    if isinstance(v, torch.Tensor):
                        est_mb += v.nelement() * v.element_size()
                    elif isinstance(v, dict):
                        for vv in v.values():
                            if isinstance(vv, torch.Tensor):
                                est_mb += vv.nelement() * vv.element_size()
                            elif isinstance(vv, list):
                                for vvv in vv:
                                    if isinstance(vvv, torch.Tensor):
                                        est_mb += vvv.nelement() * vvv.element_size()
                                    elif isinstance(vvv, np.ndarray):
                                        est_mb += vvv.nbytes
            est_mb = est_mb / 1024 / 1024
            print(f"  Loaded {n} samples from cache ({est_mb:.1f} MB)")
            if self.filter_parking:
                self.apply_parking_filter()
            return

        for i in tqdm(range(len(self)), desc="Preloading dataset", disable=False):
            if i not in self._cache:
                item = self._load_item(i)
                self._cache[i] = {k: (v.clone() if isinstance(v, torch.Tensor) else
                                      {kk: vv.clone() if isinstance(vv, torch.Tensor) else vv
                                       for kk, vv in v.items()} if isinstance(v, dict) else v)
                                  for k, v in item.items()}

        n = len(self._cache)
        est_mb = 0.0
        for item in self._cache.values():
            for v in item.values():
                if isinstance(v, torch.Tensor):
                    est_mb += v.nelement() * v.element_size()
                elif isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, torch.Tensor):
                            est_mb += vv.nelement() * vv.element_size()
                        elif isinstance(vv, list):
                            for vvv in vv:
                                if isinstance(vvv, torch.Tensor):
                                    est_mb += vvv.nelement() * vvv.element_size()
                                elif isinstance(vvv, np.ndarray):
                                    est_mb += vvv.nbytes
        est_mb = est_mb / 1024 / 1024
        print(f"  Cached {n} samples ({est_mb:.1f} MB)")

        print(f"  Saving dataset cache to {cache_path} ...")
        torch.save(self._cache, cache_path)
        print(f"  Cache saved ({cache_path.stat().st_size / 1024 / 1024:.1f} MB on disk)")

        if self.filter_parking:
            self.apply_parking_filter()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        phys_idx = self._valid_indices[idx] if self._valid_indices is not None else idx
        if phys_idx in self._cache:
            result = {k: (v.clone() if isinstance(v, torch.Tensor) else
                          {kk: vv.clone() if isinstance(vv, torch.Tensor) else vv
                           for kk, vv in v.items()} if isinstance(v, dict) else v)
                      for k, v in self._cache[phys_idx].items()}
        else:
            result = self._load_item(phys_idx)
            self._cache[phys_idx] = {k: (v.clone() if isinstance(v, torch.Tensor) else
                                    {kk: vv.clone() if isinstance(vv, torch.Tensor) else vv
                                     for kk, vv in v.items()} if isinstance(v, dict) else v)
                                for k, v in result.items()}

        # Drop scene_data when not requested (cache may contain it from a prior
        # inference run; training collate cannot handle variable-length lists).
        if not self.return_scene_data and "scene_data" in result:
            del result["scene_data"]

        # Stochastic CFG drops — must be re-applied every access
        if self.training:
            goal = result["goal"]
            map_tokens = result["map_tokens"]
            neighbor_tokens = result["neighbor_tokens"]
            if np.random.random() < self.drop_goal_p:
                goal = torch.zeros_like(goal)
            if np.random.random() < self.drop_map_p:
                map_tokens = torch.zeros_like(map_tokens)
            if np.random.random() < self.drop_neighbor_p:
                neighbor_tokens = torch.zeros_like(neighbor_tokens)
            result["goal"] = goal
            result["map_tokens"] = map_tokens
            result["neighbor_tokens"] = neighbor_tokens

        return result

    def _load_item(self, idx: int) -> Dict[str, torch.Tensor]:
        parquet_path = self.scenario_files[idx]

        scenario = load_argoverse_scenario_parquet(parquet_path)

        # Load map — try official layout first, then flat layout fallback
        log_id = str(scenario.map_id)
        static_map = None

        # 1) Official layout: map_dir/{log_id}/log_map_archive_{log_id}.json
        map_path = self.map_dir / log_id / f"log_map_archive_{log_id}.json"
        if map_path.is_file():
            try:
                static_map = ArgoverseStaticMap.from_json(map_path)
            except Exception:
                pass

        # 2) Flat layout: search by scenario file ID in map_dir
        if static_map is None:
            scenario_file_id = parquet_path.stem.replace("scenario_", "")
            for mc in self.map_dir.glob(f"**/log_map_archive_{scenario_file_id}*.json"):
                try:
                    static_map = ArgoverseStaticMap.from_json(mc)
                    break
                except Exception:
                    continue

        # 3) Last resort: search by log_id if different from scenario_file_id
        if static_map is None and log_id != scenario_file_id:
            for mc in self.map_dir.glob(f"**/log_map_archive_{log_id}*.json"):
                try:
                    static_map = ArgoverseStaticMap.from_json(mc)
                    break
                except Exception:
                    continue

        # Find focal track using focal_track_id
        focal_track = None
        focal_id = getattr(scenario, "focal_track_id", None)
        if focal_id:
            for track in scenario.tracks:
                if track.track_id == focal_id:
                    focal_track = track
                    break
        if focal_track is None:
            for track in scenario.tracks:
                if track.category.value == 3:  # FOCAL_TRACK
                    focal_track = track
                    break
        if focal_track is None:
            focal_track = scenario.tracks[0]

        # Reference position and heading at t=50
        t50_state = None
        for s in focal_track.object_states:
            if s.timestep == 50:
                t50_state = s
                break
        if t50_state is None:
            t50_state = focal_track.object_states[min(50, len(focal_track.object_states) - 1)]

        ref_pos = np.array(t50_state.position, dtype=np.float64)
        ref_heading = float(t50_state.heading)

        # --- Build lane segments dict early (needed for lane prior + map tokens) ---
        lane_segments = {}
        if static_map is not None:
            for ls_id, ls in static_map.vector_lane_segments.items():
                real_centerline = None
                try:
                    left_xyz = ls.left_lane_boundary.xyz
                    right_xyz = ls.right_lane_boundary.xyz
                    if left_xyz is not None and right_xyz is not None and len(left_xyz) >= 2 and len(right_xyz) >= 2:
                        cl_xyz = _compute_centerline_from_boundaries(left_xyz, right_xyz)
                        if cl_xyz is not None and len(cl_xyz) >= 2:
                            real_centerline = [{"x": p[0], "y": p[1], "z": p[2]} for p in cl_xyz.tolist()]
                except Exception:
                    pass

                lane_segments[str(ls_id)] = {
                    "id": ls.id,
                    "left_lane_boundary": [{"x": p[0], "y": p[1], "z": p[2]} for p in ls.left_lane_boundary.xyz.tolist()],
                    "right_lane_boundary": [{"x": p[0], "y": p[1], "z": p[2]} for p in ls.right_lane_boundary.xyz.tolist()],
                    "centerline": real_centerline,
                    "lane_type": ls.lane_type.value,
                    "is_intersection": ls.is_intersection,
                    "has_predecessor": len(ls.predecessors) > 0,
                    "has_successor": len(ls.successors) > 0,
                    "predecessors": [str(p) for p in ls.predecessors],
                    "successors": [str(s) for s in ls.successors],
                    "left_neighbor_id": str(ls.left_neighbor_id) if ls.left_neighbor_id is not None else None,
                    "right_neighbor_id": str(ls.right_neighbor_id) if ls.right_neighbor_id is not None else None,
                    "left_mark_type": ls.left_mark_type.value if hasattr(ls, "left_mark_type") else "NONE",
                    "right_mark_type": ls.right_mark_type.value if hasattr(ls, "right_mark_type") else "NONE",
                }

        # --- Future trajectory (local coords) ---
        future_positions = []
        future_headings_global = []
        for s in focal_track.object_states:
            if 50 <= s.timestep < 50 + self.n_future:
                future_positions.append(np.array(s.position))
                future_headings_global.append(s.heading)
        future_positions = np.array(future_positions)
        future_headings_global = np.array(future_headings_global, dtype=np.float64)

        if len(future_positions) < self.n_future:
            pad_pos = np.tile(future_positions[-1:], (self.n_future - len(future_positions), 1))
            future_positions = np.concatenate([future_positions, pad_pos])
            pad_h = np.full(self.n_future - len(future_headings_global), future_headings_global[-1])
            future_headings_global = np.concatenate([future_headings_global, pad_h])

        future_local, _ = global_to_local(future_positions, np.zeros(self.n_future), ref_pos, ref_heading)
        future_local = future_local.astype(np.float32)

        # --- History (local coords) ---
        history_positions = []
        history_headings_global = []
        history_velocities_global = []
        for s in focal_track.object_states:
            if self.history_start <= s.timestep < self.history_start + self.n_history:
                history_positions.append(np.array(s.position))
                history_headings_global.append(s.heading)
                history_velocities_global.append(np.array(s.velocity))
        history_positions = np.array(history_positions) if history_positions else np.zeros((1, 2))
        history_headings_global = np.array(history_headings_global, dtype=np.float64) if history_headings_global else np.zeros(1)
        history_velocities_global = np.array(history_velocities_global, dtype=np.float64) if history_velocities_global else np.zeros((1, 2))

        if len(history_positions) < self.n_history:
            pad_pos = np.tile(history_positions[-1:], (self.n_history - len(history_positions), 1))
            history_positions = np.concatenate([history_positions, pad_pos])
            pad_h = np.full(self.n_history - len(history_headings_global), history_headings_global[-1])
            history_headings_global = np.concatenate([history_headings_global, pad_h])
            pad_v = np.tile(history_velocities_global[-1:], (self.n_history - len(history_velocities_global), 1))
            history_velocities_global = np.concatenate([history_velocities_global, pad_v])

        history_local, _ = global_to_local(history_positions, np.zeros(self.n_history), ref_pos, ref_heading)
        history_local = history_local.astype(np.float32)
        # Transform velocities to local frame (rotation only — velocities are deltas)
        history_velocities_local = velocity_global_to_local(history_velocities_global, ref_heading).astype(np.float32)
        # Acceleration via finite difference of velocity: a[i] = (v[i] - v[i-1]) / dt
        # First frame has no predecessor; use forward difference for a[0].
        history_acc_local = np.zeros_like(history_velocities_local)
        history_acc_local[1:] = (history_velocities_local[1:] - history_velocities_local[:-1]) / DT
        history_acc_local[0] = history_acc_local[1] if self.n_history > 1 else 0.0

        # --- Goal (always from actual future endpoint, before CFG dropout) ---
        goal = future_local[-1].copy()

        # --- Headings for Hermite spline prior ---
        # start_heading: ensure tangent continuity with observed history.
        # Use travel direction (atan2 of last 2 position steps) when it is
        # consistent with the box heading; fall back to box heading when the
        # two diverge (side-slip, low speed, or mid-turn snap).
        start_heading_global = float(history_headings_global[-1])
        if len(history_positions) >= 2:
            dx = history_positions[-1, 0] - history_positions[-2, 0]
            dy = history_positions[-1, 1] - history_positions[-2, 1]
            if dx * dx + dy * dy > 0.04:  # >0.2 m apart
                pos_heading = float(np.arctan2(dy, dx))
                diff = pos_heading - start_heading_global
                diff = diff - 2 * np.pi * np.round(diff / (2 * np.pi))
                if abs(diff) < np.pi / 4:  # <45° divergence — trust positions
                    start_heading_global = pos_heading

        # end_heading: keep AV2 bounding-box heading (prediction target,
        # no continuity constraint at the endpoint).
        end_heading_global = float(future_headings_global[-1])

        start_heading = start_heading_global - ref_heading
        end_heading = end_heading_global - ref_heading

        # --- Parking detection (before prior computation) ---
        speeds = []
        for s in focal_track.object_states:
            if self.history_start <= s.timestep < 50 + self.n_future:
                vx, vy = s.velocity
                speeds.append(math.sqrt(vx * vx + vy * vy))
        is_parking = False
        if speeds:
            from data.normalization import _PARKING_SPEED_THRESH, _PARKING_MIN_CONSECUTIVE_STEPS
            max_consecutive = 0
            current = 0
            for sp in speeds:
                if sp < _PARKING_SPEED_THRESH:
                    current += 1
                    max_consecutive = max(max_consecutive, current)
                else:
                    current = 0
            is_parking = max_consecutive >= _PARKING_MIN_CONSECUTIVE_STEPS

        # --- Map tokens (moved before prior computation so lane_boundaries_local
        #     and drivable_areas_local are available for compute_lane_prior) ---
        map_tokens = np.zeros((self.n_lanes, self.lane_feat_dim), dtype=np.float32)
        map_mask = np.zeros(self.n_lanes, dtype=np.float32)
        lane_boundaries_local = []
        drivable_areas_local = []
        ped_crossings_local = []
        if static_map is not None:
            map_tokens, map_mask, lane_boundaries_local = encode_map_tokens(lane_segments, ref_pos, ref_heading, self.n_lanes)

            # Drivable areas
            drivable_areas_global = []
            for da_id, da in static_map.vector_drivable_areas.items():
                try:
                    da_pts = da.xyz[:, :2]  # (N, 2)
                    da_local, _ = global_to_local(da_pts, np.zeros(len(da_pts)), ref_pos, ref_heading)
                    drivable_areas_local.append(da_local.astype(np.float32))
                    drivable_areas_global.append(da_pts.astype(np.float32))
                except Exception:
                    pass

            # Pedestrian crossings
            for pc_id, pc in static_map.vector_pedestrian_crossings.items():
                try:
                    e1 = pc.edge1.xyz[:, :2]  # (2, 2)
                    e2 = pc.edge2.xyz[:, :2]  # (2, 2)
                    e1_local, _ = global_to_local(e1, np.zeros(2), ref_pos, ref_heading)
                    e2_local, _ = global_to_local(e2, np.zeros(2), ref_pos, ref_heading)
                    ped_crossings_local.append({
                        "edge1": e1_local.astype(np.float32),
                        "edge2": e2_local.astype(np.float32),
                    })
                except Exception:
                    pass

        # --- Prior and residual (in meters, before normalization) ---
        history_end = history_local[-1].copy()
        if self.prior_type == "lane":
            prior = compute_lane_prior(
                history_end, goal, lane_segments, ref_pos, ref_heading,
                start_heading, end_heading, self.n_future,
                lane_boundaries_local=lane_boundaries_local,
                drivable_areas_local=drivable_areas_local,
            )
        elif self.prior_type == "hermite":
            prior = compute_hermite_prior(history_end, goal, start_heading, end_heading, self.n_future,
                                          drivable_areas_local=drivable_areas_local)
        else:
            prior = compute_prior(history_end, goal, self.n_future)
        residual = future_local - prior  # what the diffusion model learns

        # --- Normalize ---
        future_norm = normalize(future_local)
        history_norm = normalize(history_local)
        history_velocity_norm = history_velocities_local / VELOCITY_SCALE  # (T_hist, 2)
        history_acc_norm = history_acc_local / ACCELERATION_SCALE          # (T_hist, 2)
        # History tensor: [x, y, vx, vy, ax, ay] all normalized to ~[-1, 1]
        history_full_norm = np.concatenate(
            [history_norm, history_velocity_norm, history_acc_norm], axis=-1
        ).astype(np.float32)
        goal_norm = normalize(goal.reshape(1, 2)).flatten()

        # Chord-frame or isotropic residual normalization
        if self.residual_frame == "chord":
            chord_dir, chord_len, chord_valid = compute_chord_dir(history_end, goal)
            if chord_valid:
                r_lon, r_lat = residual_to_chord_frame(residual, history_end, goal)
                r_lon_n, r_lat_n = normalize_residual_chord(r_lon, r_lat)
                residual_norm = pack_chord(r_lon_n, r_lat_n)  # (T, 2)
                use_chord = True
            else:
                residual_norm = normalize_residual(residual)  # fallback
                use_chord = False
                chord_dir = np.zeros(2, dtype=np.float32)
        else:  # "xy" legacy
            residual_norm = normalize_residual(residual)
            use_chord = False
            chord_dir = np.zeros(2, dtype=np.float32)
        prior_norm = normalize(prior)  # for potential use

        # --- Neighbor tokens (always provided) ---
        neighbor_tokens = np.zeros((self.n_neighbors, self.n_history + 1, 6), dtype=np.float32)
        neighbor_mask = np.zeros(self.n_neighbors, dtype=np.float32)
        neighbor_trajs_local = []
        neighbor_positions_local = []
        neighbor_full_trajs_local = []
        tracks_dicts = []
        for track in scenario.tracks:
            if track.category.value == 3:
                continue
            states = []
            for s in track.object_states:
                states.append({
                    "timestep": s.timestep,
                    "position": s.position,
                    "heading": s.heading,
                    "velocity": s.velocity,
                    "observed": s.observed,
                })
            tracks_dicts.append({
                "track_id": track.track_id,
                "object_type": track.object_type.value,
                "object_category": track.category.value,
                "object_states": states,
            })
        if tracks_dicts:
            neighbor_tokens, neighbor_mask, neighbor_trajs_local, neighbor_positions_local, neighbor_full_trajs_local = encode_neighbor_tokens(
                tracks_dicts, ref_pos, ref_heading, self.n_neighbors, self.n_history, self.n_future, self.history_start
            )

        # --- CFG dropout applied in __getitem__ — keep full conditions here ---

        def _to_tensor(arr):
            return torch.tensor(arr, dtype=torch.float32)

        result = {
            "trajectory": _to_tensor(residual_norm),   # diffusion target: normalized residual
            "history": _to_tensor(history_full_norm),     # (T_hist, 6) [x, y, vx, vy, ax, ay] normalized
            "history_pos": _to_tensor(history_norm),      # (T_hist, 2) positions only (for aux losses)
            "history_velocity": _to_tensor(history_velocity_norm),  # (T_hist, 2) normalized
            "history_acceleration": _to_tensor(history_acc_norm),   # (T_hist, 2) normalized
            "goal": _to_tensor(goal_norm),
            "prior": _to_tensor(prior_norm),            # normalized prior for reconstruction
            "map_tokens": _to_tensor(map_tokens),
            "map_mask": _to_tensor(map_mask),
            "neighbor_tokens": _to_tensor(neighbor_tokens),
            "neighbor_mask": _to_tensor(neighbor_mask),
            "ref_pos": _to_tensor(ref_pos),
            "ref_heading": torch.tensor(ref_heading, dtype=torch.float32),
            "start_heading": torch.tensor(start_heading, dtype=torch.float32),
            "end_heading": torch.tensor(end_heading, dtype=torch.float32),
            "start_heading_global": torch.tensor(start_heading_global, dtype=torch.float32),
            "end_heading_global": torch.tensor(end_heading_global, dtype=torch.float32),
            "use_chord_frame": torch.tensor(float(use_chord), dtype=torch.float32),
            "chord_dir": torch.tensor(chord_dir, dtype=torch.float32),  # (2,)
            "is_parking": torch.tensor(float(is_parking), dtype=torch.float32),
        }

        if self.return_scene_data:
            result["scene_data"] = {
                "lane_boundaries_local": lane_boundaries_local,
                "drivable_areas_local": drivable_areas_local,
                "drivable_areas_global": drivable_areas_global,
                "ped_crossings_local": ped_crossings_local,
                "neighbor_trajs_local": neighbor_trajs_local,
                "neighbor_positions_local": neighbor_positions_local,
                "neighbor_full_trajs_local": neighbor_full_trajs_local,
                "focal_history_local": history_local.copy(),
                "lane_segments": lane_segments,
            }

        return result
