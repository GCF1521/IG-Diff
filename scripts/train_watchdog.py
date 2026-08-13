#!/usr/bin/env python3
"""Watchdog: detect existing training crash and restart until natural completion.

Two modes:
  - If a torchrun/src.train is already running: wait for it to exit, then take over.
  - If no training running: start one immediately from latest.pt.

After takeover, behaves like auto_restart_train.py: synchronous restart loop,
sleep between restarts, until "Training complete" / "Epoch 700" / early stop.

Usage:
  nohup python scripts/train_watchdog.py --gpus 0,1,2,3,4,5 \
    > /workspace/output/logs/watchdog_$(date +%Y%m%d_%H%M%S).log 2>&1 &
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
    candidates = sorted(CKPT_DIR.glob("*/latest.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else None


def is_training_running():
    """Return True if a torchrun or src.train process is currently alive."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", "(torchrun.*src.train|python.*-m src.train)"],
            capture_output=True, text=True, check=False,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


def wait_for_existing_training_to_exit(poll_interval=60, max_wait=36000):
    """Block while an existing training is running. Returns when it exits."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Detecting existing training...")
    if not is_training_running():
        print("No existing training running.")
        return
    print(f"Existing training detected. Waiting for it to exit (poll every {poll_interval}s)...")
    last_log = ""
    while True:
        if not is_training_running():
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Existing training has exited.")
            return
        # Print a brief progress pulse
        try:
            log_files = sorted(LOG_DIR.glob("train_resume_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            if log_files:
                with open(log_files[0], "r", errors="ignore") as f:
                    lines = f.readlines()
                tail = "".join(lines[-1:]).strip()[:120]
                if tail != last_log:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] still running: {tail}")
                    last_log = tail
        except Exception:
            pass
        time.sleep(poll_interval)


def run_training(gpus, resume_ckpt, log_file):
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
    if not os.path.exists(log_file):
        return False
    with open(log_file, "r", errors="ignore") as f:
        content = f.read()
    if "Training complete" in content:
        return True
    if re.search(r"Epoch 700/700: 100%", content):
        return True
    if "Early stopping triggered" in content:
        return True
    return False


def get_last_epoch(log_file):
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
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max-restarts", type=int, default=30)
    parser.add_argument("--sleep-between", type=int, default=30)
    parser.add_argument("--poll-existing", type=int, default=60,
                        help="Poll interval (s) while waiting for existing training.")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: If a training is already running, wait for it to finish/crash.
    wait_for_existing_training_to_exit(poll_interval=args.poll_existing)

    # Step 2: Now we own the loop. Determine initial resume checkpoint.
    if args.resume:
        resume_ckpt = args.resume
    else:
        resume_ckpt = find_latest_checkpoint()
    print(f"Initial resume checkpoint: {resume_ckpt or '<fresh>'}")

    for attempt in range(1, args.max_restarts + 1):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = str(LOG_DIR / f"train_watchdog_{ts}.log")

        print("")
        print("=" * 60)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Restart attempt {attempt}/{args.max_restarts}")
        print(f"GPUs: {args.gpus}")
        print(f"Resume: {resume_ckpt or '<fresh>'}")
        print("=" * 60)

        exit_code = run_training(args.gpus, resume_ckpt, log_file)
        print(f"Process exit code: {exit_code}")

        if check_completion(log_file):
            print("")
            print("=" * 60)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TRAINING COMPLETE")
            print(f"Final log: {log_file}")
            print("=" * 60)
            with open(log_file, "r", errors="ignore") as f:
                lines = f.readlines()
                print("".join(lines[-30:]))
            return 0

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
