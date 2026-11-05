"""Print MQAR recall per key-value count from offline W&B run directories.

Each row uses the best epoch when the run logged one (train_mqar.py), otherwise
the final epoch (zoology.launch). Both values of "epoch" count from zero.

Usage: python collect_mqar.py /work/hdd/bgly/$USER/wandb/wandb
Runs that didn't finish have no summary file and are listed as skipped.
"""
import json
import re
import sys
from pathlib import Path

KV_PAIRS = [4, 8, 16, 32, 64, 128, 256]


def run_name(run_dir):
    config = (run_dir / "files" / "config.yaml").read_text()
    match = re.search(r"^run_id:\s*\n\s+value:\s*(\S+)", config, re.MULTILINE)
    return match.group(1) if match else run_dir.name


def main(root):
    rows, skipped = [], []
    for run_dir in sorted(Path(root).glob("offline-run-*")):
        summary_path = run_dir / "files" / "wandb-summary.json"
        if not summary_path.exists():
            skipped.append(run_dir.name)
            continue
        summary = json.loads(summary_path.read_text())
        prefix = "best" if "best/epoch" in summary else "valid"
        recall = [summary.get(f"{prefix}/num_kv_pairs/accuracy-{kv}", float("nan")) for kv in KV_PAIRS]
        epoch = summary.get("best/epoch", summary.get("epoch", -1))
        rows.append((run_name(run_dir), summary.get("state_size", 0), epoch, recall))

    print(f"{'run':<28}{'bytes':>7}{'epoch':>6}" + "".join(f"{'kv' + str(kv):>8}" for kv in KV_PAIRS))
    for name, size, epoch, recall in sorted(rows):
        print(f"{name:<28}{size:>7}{epoch:>6}" + "".join(f"{value:>8.4f}" for value in recall))
    for name in skipped:
        print(f"skipped, no summary: {name}")


if __name__ == "__main__":
    main(sys.argv[1])
