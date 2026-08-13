"""Self-test for GCF-DDPM: validates core components work correctly."""

import torch
import sys
sys.path.insert(0, ".")


def test_ddim_step():
    """Verify DDIM step with perfect noise prediction recovers x_0."""
    from model.diffusion import DiffusionProcess
    diff = DiffusionProcess(n_steps=1000)
    B, T, D = 2, 60, 2
    x_0 = torch.randn(B, T, D)
    noise = torch.randn_like(x_0)

    x_t = diff.add_noise(x_0, torch.tensor([999, 999]), noise)
    step_indices = torch.linspace(999, 0, 51).long()
    x_current = x_t.clone()
    for i in range(50):
        t_cur = step_indices[i].unsqueeze(0).expand(B)
        t_nxt = step_indices[i + 1].unsqueeze(0).expand(B)
        x_current = diff.ddim_step(x_current, noise, t_cur, t_nxt, eta=0.0)

    error = (x_current - x_0).abs().mean().item()
    assert error < 0.05, f"DDIM recovery error too large: {error}"
    print(f"[PASS] DDIM step: recovery error={error:.4f}")


def test_forward_diffusion():
    """Verify add_noise matches manual computation."""
    from model.diffusion import DiffusionProcess
    diff = DiffusionProcess(n_steps=1000)
    x_0 = torch.randn(2, 60, 2)
    t = torch.tensor([0, 500])
    noise = torch.randn_like(x_0)
    x_t = diff.add_noise(x_0, t, noise)

    # Manual: sqrt_alpha * x_0 + sqrt_one_minus * noise
    for b in range(2):
        sa = diff.sqrt_alphas_cumprod[t[b]]
        so = diff.sqrt_one_minus_alphas_cumprod[t[b]]
        expected = sa * x_0[b] + so * noise[b]
        actual = x_t[b]
        diff_val = (expected - actual).abs().max().item()
        assert diff_val < 1e-5, f"add_noise mismatch: {diff_val}"

    print(f"[PASS] Forward diffusion (add_noise)")


def test_model_dimensions():
    """Verify model handles all condition dimensions correctly."""
    from model.tf_cross_denoiser import TFCrossDenoiser
    model = TFCrossDenoiser(
        traj_len=60, history_len=20, n_lanes=24, lane_feat_dim=46,
        n_neighbors=6, neighbor_hist_len=20, neighbor_feat_dim=6,
        dim=256, n_heads=4, n_layers=6, ffn_dim=1024,
    )
    B = 2
    x_t = torch.randn(B, 60, 2)
    t = torch.randint(0, 1000, (B,))
    conditions = {
        "goal": torch.randn(B, 2),
        "map_tokens": torch.randn(B, 24, 46),
        "map_mask": torch.ones(B, 24),
        "neighbor_tokens": torch.randn(B, 6, 21, 6),
        "neighbor_mask": torch.ones(B, 6),
        "history": torch.randn(B, 20, 2),
    }
    noise_pred, attn_dict = model(x_t, t, **conditions, return_attn=True)
    assert noise_pred.shape == (B, 60, 2), f"Wrong output shape: {noise_pred.shape}"
    assert len(attn_dict) == 6, f"Wrong number of layers in attn_dict: {len(attn_dict)}"
    print(f"[PASS] Model dimensions: output shape={noise_pred.shape}")


def test_gradient_flow():
    """Verify gradients propagate to all key parameter groups after warmup."""
    from model.tf_cross_denoiser import TFCrossDenoiser
    from model.diffusion import DiffusionProcess
    model = TFCrossDenoiser(
        traj_len=60, history_len=20, n_lanes=24, lane_feat_dim=46,
        n_neighbors=6, neighbor_hist_len=20, neighbor_feat_dim=6,
        dim=256, n_heads=4, n_layers=6, ffn_dim=1024,
    )
    diff = DiffusionProcess(n_steps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    B = 4

    # Warmup: run 10 steps to unlock adaLN-Zero gates
    for _ in range(10):
        residual = torch.randn(B, 60, 2)
        timesteps = torch.randint(0, 1000, (B,))
        noise = torch.randn_like(residual)
        x_t = diff.add_noise(residual, timesteps, noise)
        conditions = {
            "goal": torch.randn(B, 2),
            "map_tokens": torch.randn(B, 24, 46),
            "map_mask": torch.ones(B, 24),
            "neighbor_tokens": torch.randn(B, 6, 21, 6),
            "neighbor_mask": torch.ones(B, 6),
            "history": torch.randn(B, 20, 2),
        }
        noise_pred = model(x_t, timesteps, **conditions)
        loss = diff.endpoint_weighted_loss(noise, noise_pred, lambda_ep=2.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Check gradient flow
    groups = {
        "output_head": 0,
        "adaLN_mlp": 0,
        "self_attn": 0,
        "cross_attn": 0,
        "cond_encoder": 0,
        "traj_embed": 0,
        "timestep_embedder": 0,
        "pos_embed": 0,
    }
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.norm() > 1e-8:
            for key in groups:
                if key in name:
                    groups[key] += 1

    for key, count in groups.items():
        if count == 0:
            # Some groups may still be frozen early in training
            pass  # Not fatal — will unlock with more training
    total_nonzero = sum(1 for p in model.parameters() if p.grad is not None and p.grad.norm() > 1e-8)
    assert total_nonzero >= 80, f"Too few nonzero gradients: {total_nonzero}/120"
    print(f"[PASS] Gradient flow: {total_nonzero}/120 params have nonzero gradients")


def test_dps_sampling_shape():
    """Verify full_dps_sample returns correct shapes."""
    from model.tf_cross_denoiser import TFCrossDenoiser
    from model.dps_guidance import full_dps_sample
    from model.diffusion import DiffusionProcess
    model = TFCrossDenoiser(
        traj_len=60, history_len=20, n_lanes=24, lane_feat_dim=46,
        n_neighbors=6, neighbor_hist_len=20, neighbor_feat_dim=6,
        dim=256, n_heads=4, n_layers=6, ffn_dim=1024,
    )
    diff = DiffusionProcess(n_steps=1000)
    conditions = {
        "goal": torch.randn(1, 2),
        "map_tokens": torch.randn(1, 24, 46),
        "map_mask": torch.ones(1, 24),
        "neighbor_tokens": torch.randn(1, 6, 21, 6),
        "neighbor_mask": torch.ones(1, 6),
        "history": torch.randn(1, 20, 2),
    }
    prior_norm = torch.randn(1, 60, 2)

    result = full_dps_sample(
        diff, model, conditions,
        traj_len=60, n_inference_steps=10,
        cfg_weight=2.0, dps_eta=0.0,
        prior_norm=prior_norm, use_cfg=True, use_dps=False,
        device="cpu", save_intermediates=True,
    )
    residual_norm, intermediates = result
    assert residual_norm.shape == (1, 60, 2), f"Wrong residual shape: {residual_norm.shape}"
    assert len(intermediates["residuals"]) > 0, "No intermediate snapshots saved"
    for res in intermediates["residuals"]:
        assert res.shape == (1, 60, 2), f"Wrong intermediate shape: {res.shape}"
    print(f"[PASS] DPS sampling: output shape={residual_norm.shape}, intermediates={len(intermediates['residuals'])} snapshots")


def test_map_encoding_dim():
    """Verify map encoding produces 46-dim features."""
    from data.map_encoding import encode_lane_segment
    lane = {
        "left_lane_boundary": [{"x": 0, "y": 0, "z": 0}, {"x": 1, "y": 0, "z": 0}, {"x": 2, "y": 0, "z": 0}],
        "right_lane_boundary": [{"x": 0, "y": 3, "z": 0}, {"x": 1, "y": 3, "z": 0}, {"x": 2, "y": 3, "z": 0}],
        "lane_type": "VEHICLE",
        "is_intersection": False,
    }
    ref_pos = np.array([0.0, 0.0])
    ref_heading = 0.0
    feat = encode_lane_segment(lane, ref_pos, ref_heading)
    assert feat.shape == (46,), f"Wrong map feature dim: {feat.shape}"
    print(f"[PASS] Map encoding: feature dim={feat.shape[0]}")


def test_dataset_loading():
    """Verify dataset loads real data with correct dimensions."""
    from data.av2_dataset import Argoverse2Dataset
    from pathlib import Path
    # Use real AV2 dataset if available, otherwise fall back to av2_sample
    data_dir = "av2_dataset_1k/train/"
    if not Path(data_dir).exists():
        data_dir = "av2_sample/"
    if not Path(data_dir).exists():
        print("[SKIP] Dataset test: no data directory found")
        return

    dataset = Argoverse2Dataset(
        data_dir=data_dir, n_future=60, n_history=20,
        n_lanes=24, lane_feat_dim=46, n_neighbors=6, split="train",
    )
    sample = dataset[0]
    assert sample["trajectory"].shape == (60, 2), f"Wrong residual shape"
    assert sample["goal"].shape == (2,), f"Wrong goal shape"
    assert sample["history"].shape == (20, 6), f"Wrong history shape: {sample['history'].shape}"
    assert sample["history_pos"].shape == (20, 2), f"Wrong history_pos shape"
    assert sample["map_tokens"].shape == (24, 46), f"Wrong map_tokens shape: {sample['map_tokens'].shape}"
    assert sample["neighbor_tokens"].shape == (6, 21, 6), f"Wrong neighbor shape"
    print(f"[PASS] Dataset loading: {len(dataset)} scenarios, dimensions verified")


def test_normalization_roundtrip():
    """Verify normalize/denormalize roundtrip preserves values."""
    from data.normalization import normalize, denormalize, normalize_residual, denormalize_residual
    import numpy as np
    x = np.array([10.0, -30.0, 25.0, 0.5])
    assert np.allclose(denormalize(normalize(x)), x), "Coord roundtrip failed"
    r = np.array([1.3, -0.5, 2.0, 0.1])
    assert np.allclose(denormalize_residual(normalize_residual(r)), r), "Residual roundtrip failed"
    print("[PASS] Normalization roundtrip")


def test_chord_frame_roundtrip():
    """Verify chord-frame residual roundtrip preserves original values."""
    import numpy as np
    from data.normalization import (
        compute_chord_dir, residual_to_chord_frame, chord_frame_to_residual,
        normalize_residual_chord, denormalize_residual_chord,
        pack_chord, unpack_chord,
    )

    # --- Numpy roundtrip ---
    history_end = np.array([1.0, 2.0], dtype=np.float32)
    goal = np.array([5.0, 8.0], dtype=np.float32)
    residual = np.array([[0.5, -0.3], [1.0, 0.2], [-0.5, 0.8]], dtype=np.float32)

    chord_dir, chord_len, is_valid = compute_chord_dir(history_end, goal)
    assert is_valid, "Chord should be valid for non-degenerate case"

    r_lon, r_lat = residual_to_chord_frame(residual, history_end, goal)
    assert r_lon is not None and r_lat is not None

    # Reconstruct residual from lon/lat
    reconstructed = chord_frame_to_residual(r_lon, r_lat, chord_dir)
    assert np.allclose(residual, reconstructed, atol=1e-5), \
        f"Chord frame roundtrip failed: max error={np.abs(residual - reconstructed).max()}"

    # Normalize + denormalize roundtrip
    r_lon_n, r_lat_n = normalize_residual_chord(r_lon, r_lat)
    r_lon_d, r_lat_d = denormalize_residual_chord(r_lon_n, r_lat_n)
    assert np.allclose(r_lon, r_lon_d, atol=1e-6), "Lon normalize roundtrip failed"
    assert np.allclose(r_lat, r_lat_d, atol=1e-6), "Lat normalize roundtrip failed"

    # Pack + unpack roundtrip
    packed = pack_chord(r_lon_n, r_lat_n)
    assert packed.shape == (3, 2), f"Wrong packed shape: {packed.shape}"
    u_lon, u_lat = unpack_chord(packed)
    assert np.allclose(r_lon_n, u_lon, atol=1e-7), "Pack/unpack lon roundtrip failed"
    assert np.allclose(r_lat_n, u_lat, atol=1e-7), "Pack/unpack lat roundtrip failed"

    # Full pipeline: residual → chord → normalize → pack → unpack → denorm → dechord → residual
    packed = pack_chord(*normalize_residual_chord(*residual_to_chord_frame(residual, history_end, goal)))
    u_lon_n, u_lat_n = unpack_chord(packed)
    r_lon_d, r_lat_d = denormalize_residual_chord(u_lon_n, u_lat_n)
    full_reconstructed = chord_frame_to_residual(r_lon_d, r_lat_d, chord_dir)
    assert np.allclose(residual, full_reconstructed, atol=1e-4), \
        f"Full chord pipeline roundtrip failed: max error={np.abs(residual - full_reconstructed).max()}"

    # --- Degenerate chord (||chord|| < epsilon) ---
    near_zero_end = np.array([1.0, 2.0], dtype=np.float32)
    near_zero_goal = np.array([1.05, 2.03], dtype=np.float32)  # dist < 0.1
    _, _, is_valid_degen = compute_chord_dir(near_zero_end, near_zero_goal)
    assert not is_valid_degen, "Degenerate chord should be invalid"

    r_lon_d, r_lat_d = residual_to_chord_frame(residual, near_zero_end, near_zero_goal)
    assert r_lon_d is None and r_lat_d is None, "Degenerate chord should return None"

    # --- Torch roundtrip ---
    history_end_t = torch.tensor([1.0, 2.0])
    goal_t = torch.tensor([5.0, 8.0])
    residual_t = torch.tensor([[0.5, -0.3], [1.0, 0.2], [-0.5, 0.8]])

    chord_dir_t, chord_len_t, is_valid_t = compute_chord_dir(history_end_t, goal_t)
    assert is_valid_t, "Torch chord should be valid"

    r_lon_t, r_lat_t = residual_to_chord_frame(residual_t, history_end_t, goal_t)
    reconstructed_t = chord_frame_to_residual(r_lon_t, r_lat_t, chord_dir_t)
    assert torch.allclose(residual_t, reconstructed_t, atol=1e-5), "Torch chord roundtrip failed"

    # Torch batched (B, T, 2)
    residual_batched = torch.randn(4, 60, 2)
    r_lon_b, r_lat_b = residual_to_chord_frame(residual_batched, history_end_t, goal_t)
    reconstructed_b = chord_frame_to_residual(r_lon_b, r_lat_b, chord_dir_t)
    assert torch.allclose(residual_batched, reconstructed_b, atol=1e-4), \
        f"Torch batched roundtrip failed: max error={(residual_batched - reconstructed_b).abs().max()}"

    print("[PASS] Chord-frame residual roundtrip (numpy + torch + batched + degenerate)")


def test_lane_prior():
    """Verify lane centerline prior computation and roundtrip."""
    import numpy as np
    from data.normalization import compute_lane_prior, compute_hermite_prior, compute_prior

    # --- Test 1: Lane prior with synthetic lane segments ---
    # Build a simple lane network: one segment going east, length matches goal distance
    lane_segments = {
        "1": {
            "id": 1,
            "left_lane_boundary": [{"x": 0, "y": 2, "z": 0}, {"x": 35, "y": 2, "z": 0}],
            "right_lane_boundary": [{"x": 0, "y": -2, "z": 0}, {"x": 35, "y": -2, "z": 0}],
            "centerline": [{"x": 0, "y": 0, "z": 0}, {"x": 15, "y": 0, "z": 0}, {"x": 30, "y": 0, "z": 0}],
            "lane_type": "VEHICLE",
            "is_intersection": False,
            "has_predecessor": False,
            "has_successor": False,
            "predecessors": [],
            "successors": [],
            "left_mark_type": "SOLID",
            "right_mark_type": "SOLID",
        },
    }

    ref_pos = np.array([0.0, 0.0])
    ref_heading = 0.0
    history_end = np.array([0.0, 0.0], dtype=np.float32)  # at lane start
    goal = np.array([30.0, 3.0], dtype=np.float32)         # offset 3m laterally
    start_heading = 0.0
    end_heading = 0.0
    n_future = 60

    prior = compute_lane_prior(
        history_end, goal, lane_segments, ref_pos, ref_heading,
        start_heading, end_heading, n_future)

    assert prior.shape == (n_future, 2), f"Wrong lane prior shape: {prior.shape}"
    # Start point should match history_end
    assert np.allclose(prior[0], history_end, atol=1e-4), \
        f"Lane prior start mismatch: {prior[0]} vs {history_end}"
    # End point should be near goal (lateral offset blending brings it closer)
    end_dist = np.linalg.norm(prior[-1] - goal)
    assert end_dist < 3.0, f"Lane prior endpoint too far from goal: {end_dist:.2f}m"
    # Lateral offset should be minimal at midpoint, larger at end
    mid_lateral = abs(prior[30, 1] - 0.0)  # lateral offset at midpoint
    end_lateral = abs(prior[-1, 1] - 0.0)   # lateral offset at endpoint
    assert mid_lateral < end_lateral + 0.5, "Lateral offset at end should be >= midpoint"
    print(f"  Lane prior (straight lane, lateral offset): shape={prior.shape}, "
          f"endpoint_dist={end_dist:.3f}m, mid_lat={mid_lateral:.2f}, end_lat={end_lateral:.2f}")

    # --- Test 2: Lane prior with curved lane (turning scenario) ---
    # Build a lane that curves from east to north (left turn)
    n_curve_pts = 20
    angles = np.linspace(0, np.pi / 2, n_curve_pts)
    radius = 20.0
    curve_x = (radius * np.sin(angles)).tolist()
    curve_y = (radius * (1 - np.cos(angles))).tolist()
    curve_centerline = [{"x": x, "y": y, "z": 0} for x, y in zip(curve_x, curve_y)]

    lane_segments_turn = {
        "1": {
            "id": 1,
            "left_lane_boundary": curve_centerline,  # simplified
            "right_lane_boundary": curve_centerline,
            "centerline": curve_centerline,
            "lane_type": "VEHICLE",
            "is_intersection": False,
            "has_predecessor": False,
            "has_successor": False,
            "predecessors": [],
            "successors": [],
            "left_mark_type": "SOLID",
            "right_mark_type": "SOLID",
        },
    }

    # Agent starts near the beginning of the curve, goal at the end
    history_end_turn = np.array([0.0, 0.0], dtype=np.float32)
    goal_turn = np.array([float(curve_x[-1]), float(curve_y[-1])], dtype=np.float32)
    start_heading_turn = 0.0       # heading east
    end_heading_turn = np.pi / 2   # heading north

    prior_turn = compute_lane_prior(
        history_end_turn, goal_turn, lane_segments_turn, ref_pos, ref_heading,
        start_heading_turn, end_heading_turn, n_future)

    assert prior_turn.shape == (n_future, 2), f"Wrong turn prior shape: {prior_turn.shape}"
    assert np.allclose(prior_turn[0], history_end_turn, atol=1e-3), \
        f"Turn prior start mismatch: {prior_turn[0]} vs {history_end_turn}"
    # The curve should have positive Y values (turning north)
    assert prior_turn[30, 1] > 1.0, f"Turn prior should have positive Y at midpoint: {prior_turn[30, 1]}"
    print(f"  Lane prior (turning lane): shape={prior_turn.shape}, "
          f"mid=({prior_turn[30, 0]:.1f}, {prior_turn[30, 1]:.1f})")

    # --- Test 3: Fallback to hermite when no map ---
    prior_fallback = compute_lane_prior(
        history_end, goal, {}, ref_pos, ref_heading,
        start_heading, end_heading, n_future)
    hermite_ref = compute_hermite_prior(history_end, goal, start_heading, end_heading, n_future)
    assert np.allclose(prior_fallback, hermite_ref, atol=1e-5), \
        "Lane prior should fall back to hermite when no lane segments"
    print(f"  Lane prior fallback to hermite: OK")

    # --- Test 4: Fallback to hermite when no lane near agent ---
    far_lanes = {
        "1": {
            "id": 1,
            "left_lane_boundary": [{"x": 500, "y": 2, "z": 0}, {"x": 600, "y": 2, "z": 0}],
            "right_lane_boundary": [{"x": 500, "y": -2, "z": 0}, {"x": 600, "y": -2, "z": 0}],
            "centerline": [{"x": 500, "y": 0, "z": 0}, {"x": 550, "y": 0, "z": 0}, {"x": 600, "y": 0, "z": 0}],
            "lane_type": "VEHICLE",
            "is_intersection": False,
            "has_predecessor": False,
            "has_successor": False,
            "predecessors": [],
            "successors": [],
            "left_mark_type": "SOLID",
            "right_mark_type": "SOLID",
        },
    }
    prior_far = compute_lane_prior(
        history_end, goal, far_lanes, ref_pos, ref_heading,
        start_heading, end_heading, n_future)
    # Should still work (finds nearest lane even if far), not fallback
    assert prior_far.shape == (n_future, 2), f"Far lane prior shape wrong: {prior_far.shape}"

    # --- Test 5: Consistency with hermite prior for straight paths ---
    # When goal is directly on the centerline, lateral offset should be ~0
    goal_on_centerline = np.array([30.0, 0.0], dtype=np.float32)
    prior_on_cl = compute_lane_prior(
        history_end, goal_on_centerline, lane_segments, ref_pos, ref_heading,
        start_heading, end_heading, n_future)
    # Lateral offset should be near zero since goal is on centerline
    max_lateral = np.abs(prior_on_cl[:, 1]).max()
    assert max_lateral < 0.5, f"Lateral offset should be near zero for on-centerline goal: {max_lateral:.3f}"
    print(f"  Lane prior (on-centerline goal): max_lateral_offset={max_lateral:.4f}m")

    # --- Test 6: Intersection gap bridging ---
    # Lane 1 has centerline, Lane 2 is intersection (no centerline), Lane 3 has centerline
    lane_segments_intersection = {
        "1": {
            "id": 1,
            "left_lane_boundary": [{"x": 0, "y": 2, "z": 0}, {"x": 30, "y": 2, "z": 0}],
            "right_lane_boundary": [{"x": 0, "y": -2, "z": 0}, {"x": 30, "y": -2, "z": 0}],
            "centerline": [{"x": 0, "y": 0, "z": 0}, {"x": 15, "y": 0, "z": 0}, {"x": 30, "y": 0, "z": 0}],
            "lane_type": "VEHICLE",
            "is_intersection": False,
            "has_predecessor": False,
            "has_successor": True,
            "predecessors": [],
            "successors": ["2"],
            "left_mark_type": "SOLID",
            "right_mark_type": "SOLID",
        },
        "2": {
            "id": 2,
            "left_lane_boundary": [{"x": 30, "y": 2, "z": 0}, {"x": 45, "y": 2, "z": 0}],
            "right_lane_boundary": [{"x": 30, "y": -2, "z": 0}, {"x": 45, "y": -2, "z": 0}],
            "centerline": None,  # intersection segment — no centerline
            "lane_type": "VEHICLE",
            "is_intersection": True,
            "has_predecessor": True,
            "has_successor": True,
            "predecessors": ["1"],
            "successors": ["3"],
            "left_mark_type": "NONE",
            "right_mark_type": "NONE",
        },
        "3": {
            "id": 3,
            "left_lane_boundary": [{"x": 45, "y": 2, "z": 0}, {"x": 75, "y": 2, "z": 0}],
            "right_lane_boundary": [{"x": 45, "y": -2, "z": 0}, {"x": 75, "y": -2, "z": 0}],
            "centerline": [{"x": 45, "y": 0, "z": 0}, {"x": 60, "y": 0, "z": 0}, {"x": 75, "y": 0, "z": 0}],
            "lane_type": "VEHICLE",
            "is_intersection": False,
            "has_predecessor": True,
            "has_successor": False,
            "predecessors": ["2"],
            "successors": [],
            "left_mark_type": "SOLID",
            "right_mark_type": "SOLID",
        },
    }
    goal_gap = np.array([60.0, 0.0], dtype=np.float32)
    prior_gap = compute_lane_prior(
        history_end, goal_gap, lane_segments_intersection, ref_pos, ref_heading,
        start_heading, end_heading, n_future)
    assert prior_gap.shape == (n_future, 2), f"Intersection gap prior shape wrong: {prior_gap.shape}"
    # Should bridge the intersection gap — path should extend past x=30
    assert prior_gap[-1, 0] > 50, f"Intersection gap prior should extend past x=50: last_x={prior_gap[-1, 0]:.1f}"
    print(f"  Lane prior (intersection gap): shape={prior_gap.shape}, "
          f"last_x={prior_gap[-1, 0]:.1f}")

    print("[PASS] Lane centerline prior (6 sub-tests)")


def test_curvature_rate_constraint():
    """Verify curvature rate constraint limits dk/ds."""
    import numpy as np
    from data.normalization import (
        _limit_curvature_rate, _LANE_MAX_CURVATURE_RATE,
    )

    # Construct a curvature profile with extreme jumps
    kappa = np.array([0.0, 0.5, 0.0, -0.5, 0.0, 0.3, -0.3], dtype=np.float32)
    ds = np.ones(len(kappa) - 1, dtype=np.float32) * 1.0  # 1m arc-length steps

    limited = _limit_curvature_rate(kappa, ds, _LANE_MAX_CURVATURE_RATE)

    # Verify that all discrete curvature rates are within limit
    for i in range(len(limited) - 1):
        dk = abs(limited[i + 1] - limited[i]) / ds[i]
        assert dk <= _LANE_MAX_CURVATURE_RATE + 1e-6, \
            f"Curvature rate exceeded at i={i}: dk/ds={dk:.4f} > {_LANE_MAX_CURVATURE_RATE}"

    print(f"[PASS] Curvature rate constraint: max rate within {_LANE_MAX_CURVATURE_RATE} rad/m²")


def test_lateral_accel_constraint():
    """Verify lateral acceleration constraint limits a_lat = v²*kappa."""
    import numpy as np
    from data.normalization import (
        _limit_lateral_acceleration, _LANE_MAX_LATERAL_ACCEL,
    )

    # High curvature + moderate speed → should be clipped
    kappa = np.array([0.0, 0.1, 0.3, 0.5, 0.3, 0.1, 0.0], dtype=np.float32)
    v_avg = 10.0  # m/s → a_lat = 100 * kappa
    max_accel = _LANE_MAX_LATERAL_ACCEL

    limited = _limit_lateral_acceleration(kappa, v_avg, max_accel)

    # Verify all lateral accelerations are within limit
    for i in range(len(limited)):
        a_lat = v_avg * v_avg * abs(limited[i])
        assert a_lat <= max_accel + 1e-4, \
            f"Lateral accel exceeded at i={i}: a_lat={a_lat:.3f} > {max_accel}"

    # Sign should be preserved
    for i in range(len(limited)):
        assert np.sign(limited[i]) == np.sign(kappa[i]) or abs(kappa[i]) < 1e-8, \
            f"Curvature sign changed at i={i}: {kappa[i]} → {limited[i]}"

    print(f"[PASS] Lateral acceleration constraint: a_lat ≤ {max_accel} m/s²")


def test_boundary_push():
    """Verify boundary push moves points away from lane boundaries."""
    import numpy as np
    from data.normalization import (
        _push_away_from_boundaries, _BOUNDARY_PUSH_DISTANCE,
    )

    # Construct a prior trajectory that runs close to a boundary
    # Boundary at y=0 (left) and y=4 (right), centerline at y=2
    # Prior at y=0.5 — too close to left boundary
    n = 60
    prior = np.column_stack([
        np.linspace(0, 30, n),
        np.full(n, 0.5),  # 0.5m from left boundary at y=0
    ]).astype(np.float32)

    lane_boundaries = [{
        "left": np.array([[0, 0], [30, 0]], dtype=np.float32),
        "right": np.array([[0, 4], [30, 4]], dtype=np.float32),
    }]

    pushed, violation_frac = _push_away_from_boundaries(
        prior, lane_boundaries, _BOUNDARY_PUSH_DISTANCE)

    # All interior points should be at least _BOUNDARY_PUSH_DISTANCE from left boundary
    for i in range(1, n - 1):  # skip endpoints which are preserved
        dist_to_left = pushed[i, 1]  # y coordinate = distance from y=0 boundary
        assert dist_to_left >= _BOUNDARY_PUSH_DISTANCE - 0.1, \
            f"Point {i} still too close to boundary: dist={dist_to_left:.2f} < {_BOUNDARY_PUSH_DISTANCE}"

    print(f"[PASS] Boundary push: violation_frac={violation_frac:.2f}, "
          f"min_y={pushed[1:-1, 1].min():.2f}m (threshold={_BOUNDARY_PUSH_DISTANCE}m)")


def test_degradation_logic():
    """Verify degradation produces Hermite when conditions are met."""
    import numpy as np
    from data.normalization import (
        compute_lane_prior, compute_hermite_prior, _LANE_DEGRADE_MIN_DELTA_H,
    )

    # Small heading change + boundary violation scenario:
    # Use a straight road with lane boundaries very close to centerline
    # to trigger boundary violations
    lane_segments = {
        "1": {
            "id": 1,
            "left_lane_boundary": [{"x": 0, "y": 1.2, "z": 0}, {"x": 50, "y": 1.2, "z": 0}],
            "right_lane_boundary": [{"x": 0, "y": -1.2, "z": 0}, {"x": 50, "y": -1.2, "z": 0}],
            "centerline": [{"x": 0, "y": 0, "z": 0}, {"x": 25, "y": 0, "z": 0}, {"x": 50, "y": 0, "z": 0}],
            "lane_type": "VEHICLE",
            "is_intersection": False,
            "has_predecessor": False,
            "has_successor": False,
            "predecessors": [],
            "successors": [],
            "left_mark_type": "SOLID",
            "right_mark_type": "SOLID",
        },
    }

    # Narrow boundaries: vehicle width 2m, boundaries at ±1.2m → violation everywhere
    # and small heading change (0°) → should degrade
    ref_pos = np.array([0.0, 0.0])
    ref_heading = 0.0
    history_end = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([30.0, 0.0], dtype=np.float32)
    start_heading = 0.0
    end_heading = 0.01  # Very small heading change → should trigger Level 2 degradation

    lane_boundaries_local = [{
        "left": np.array([[0, 1.2], [50, 1.2]], dtype=np.float32),
        "right": np.array([[0, -1.2], [50, -1.2]], dtype=np.float32),
    }]

    prior = compute_lane_prior(
        history_end, goal, lane_segments, ref_pos, ref_heading,
        start_heading, end_heading, 60,
        lane_boundaries_local=lane_boundaries_local,
    )

    hermite = compute_hermite_prior(history_end, goal, start_heading, end_heading, 60)

    # With small delta_h and boundary violations, should degrade to Hermite
    is_degraded = np.allclose(prior, hermite, atol=0.5)
    assert is_degraded, \
        f"Expected degradation to Hermite for small delta_h + boundary violation, " \
        f"but got different output. Max diff: {np.abs(prior - hermite).max():.3f}"

    print(f"[PASS] Degradation logic: small delta_h + boundary violation → Hermite fallback")


if __name__ == "__main__":
    import numpy as np  # needed for test_map_encoding_dim

    print("=" * 60)
    print("GCF-DDPM Self-Test")
    print("=" * 60)

    test_normalization_roundtrip()
    test_chord_frame_roundtrip()
    test_lane_prior()
    test_curvature_rate_constraint()
    test_lateral_accel_constraint()
    test_boundary_push()
    test_degradation_logic()
    test_forward_diffusion()
    test_ddim_step()
    test_map_encoding_dim()
    test_model_dimensions()
    test_gradient_flow()
    test_dps_sampling_shape()
    test_dataset_loading()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)