"""Compare serial/parallel preprocessing on a small real AV2 subset."""
import argparse
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.av2_dataset import Argoverse2Dataset


def assert_equal(left, right, path="cache"):
    assert type(left) is type(right), f"{path}: types differ"
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype, path
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6, msg=path)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype, path
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6, err_msg=path)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_equal(left[key], right[key], f"{path}.{key}")
    elif is_dataclass(left):
        for field in fields(left):
            assert_equal(getattr(left, field.name), getattr(right, field.name), f"{path}.{field.name}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_equal(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    files = sorted(Path(args.data_dir).glob("**/scenario_*.parquet"))
    count = min(args.scenarios, len(files))
    selected = [files[i * (len(files) - 1) // max(count - 1, 1)] for i in range(count)]
    with tempfile.TemporaryDirectory(prefix="preload_check_", dir=work) as temporary:
        temporary = Path(temporary)
        sample_dir = temporary / "scenarios"
        sample_dir.mkdir()
        for parquet in selected:
            scene_dir = sample_dir / parquet.parent.name
            scene_dir.mkdir()
            for source in [parquet, *parquet.parent.glob("log_map_archive_*.json")]:
                (scene_dir / source.name).symlink_to(source.resolve())
        keys = ("n_future", "n_history", "n_lanes", "lane_feat_dim", "n_neighbors", "prior_type", "residual_frame")
        kwargs = {key: cfg["data"][key] for key in keys}
        kwargs.update(data_dir=str(sample_dir), split="eval", return_scene_data=True, filter_parking=True)
        datasets, seconds = {}, {}
        for name, workers in (("serial", 1), ("parallel", args.workers)):
            dataset = Argoverse2Dataset(**kwargs)
            started = perf_counter()
            dataset.preload_cache(str(temporary / f"{name}.pt"), preload_workers=workers)
            seconds[name] = perf_counter() - started
            datasets[name] = dataset
        serial, parallel = datasets["serial"], datasets["parallel"]
        assert len(serial._cache) == len(parallel._cache) == count
        assert_equal(serial._cache, parallel._cache)
        assert_equal(serial._valid_indices, parallel._valid_indices)
        reloaded = Argoverse2Dataset(**kwargs)

        def unexpected_parse(*unused):
            raise AssertionError("Existing cache should load without parsing")

        reloaded._load_item = unexpected_parse
        reloaded.preload_cache(str(temporary / "parallel.pt"), preload_workers=args.workers)
        assert_equal(serial._cache, reloaded._cache)
        assert_equal(serial._valid_indices, reloaded._valid_indices)
        report = dict(status="PASS", scenarios=count, workers=args.workers,
                      serial_seconds=seconds["serial"], parallel_seconds=seconds["parallel"],
                      speedup=seconds["serial"] / seconds["parallel"],
                      non_parking_count=len(parallel), cache_reload="PASS")
        (work / "preload_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
