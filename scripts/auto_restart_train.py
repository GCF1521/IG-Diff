#!/usr/bin/env python3
"""Auto-restart training loop: keeps restarting from latest.pt after DDP crashes
until training completes naturally (epoch 700 or early stop).

Unlike the bash version, this runs synchronously — it waits for each training
process to finish before deciding to restart. This avoids the port conflict
issue where multiple torchrun processes start at the same time.

Usage:
  python scripts/auto_restart_train.py --gpus 0,1,2,3,4,5
  python scripts/auto_restart_train.py --gpus 0,1,2,3,4,5 --resume /path/to/latest.pt
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


WORKSPACE = "/workspace"
LOG_DIR = Path(WORKSPACE) / "output" / "logs"
CKPT_DIR = Path(WORKSPACE) / "output" / "checkpoints" / "train"


def find_latest_checkpoint():
    """Find the most recent latest.pt across all training run dirs."""
    candidates = sorted(CKPT_DIR.glob("*/latest.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else None


def run_training(gpus, resume_ckpt, log_file):
    """Run a single training process synchronously. Returns (exit_code, log_path)."""
    cmd = [
        sys.executable, "-m", "src.train",
        "--config", "config/default.yaml",
        "--gpus", gpus,
    ]
    if resume_ckpt:
        cmd.extend(["--resume", resume_ckpt])

    env = {**os.environ, "PYTHONPATH": WORKSPACE}
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_file}")

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd, cwd=WORKSPACE, env=env,
            stdout=f, stderr=subprocess.STDOUT,
        )
        proc.wait()
        return proc.returncode


def check_completion(log_file):
    """Check if training completed naturally (epoch 700 or early stop)."""
    if not os.path.exists(log_file):
        return False
    with open(log_file, "r", errors="ignore") as f:
        content = f.read()
    # Training complete message
    if "Training complete" in content:
        return True
    # Reached final epoch
    if re.search(r"Epoch 700/700: 100%", content):
        return True
    # Early stopping triggered (still considered a natural end)
    if "Early stopping triggered" in content:
        return True
    return False


def get_last_epoch(log_file):
    """Extract the last completed epoch number from the log."""
    if not os.path.exists(log_file):
        return None
    with open(log_file, "r", errors="ignore") as f:
        content = f.read()
    matches = re.findall(r"Epoch (\d+)/700", content)
    if matches:
        return int(matches[-1])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume from. If omitted, use latest.")
    parser.add_argument("--max-restarts", type=int, default=30)
    parser.add_argument("--sleep-between", type=int, default=30,
                        help="Seconds to wait between restarts (let GPU memory release).")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Determine initial resume checkpoint
    if args.resume:
        resume_ckpt = args.resume
    else:
        resume_ckpt = find_latest_checkpoint()
    print(f"Initial resume checkpoint: {resume_ckpt or '<fresh>'}")

    for attempt in range(1, args.max_restarts + 1):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = str(LOG_DIR / f"train_autorestart_{ts}.log")

        print("")
        print("=" * 60)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Restart attempt {attempt}/{args.max_restarts}")
        print(f"GPUs: {args.gpus}")
        print(f"Resume: {resume_ckpt or '<fresh>'}")
        print("=" * 60)

        exit_code = run_training(args.gpus, resume_ckpt, log_file)
        print(f"Process exit code: {exit_code}")

        # Check if training completed naturally
        if check_completion(log_file):
            print("")
            print("=" * 60)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"TRAINING COMPLETE")
            print(f"Final log: {log_file}")
            print("=" * 60)
            with open(log_file, "r", errors="ignore") as f:
                lines = f.readlines()
                print("".join(lines[-30:]))
            return 0

        # Crash: find newest latest.pt for next restart
        new_resume = find_latest_checkpoint()
        last_epoch = get_last_epoch(log_file)
        if new_resume is None:
            print(f"ERROR: No latest.pt found after crash — cannot restart")
            print(f"Last 50 lines of {log_file}:")
            with open(log_file, "r", errors="ignore") as f:
                lines = f.readlines()
                print("".join(lines[-50:]))
            return 1

        print(f"Crash detected at epoch {last_epoch}. "
              f"Will restart from: {new_resume}")
        print(f"Last 10 lines of crashed run:")
        with open(log_file, "r", errors="ignore") as f:
            lines = f.readlines()
            print("".join(lines[-10:]))

        resume_ckpt = new_resume
        print(f"Waiting {args.sleep_between}s for GPU memory release...")
        time.sleep(args.sleep_between)

    print(f"\nReached MAX_RESTARTS={args.max_restarts}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
