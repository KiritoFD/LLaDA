import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_loss_series(path: Path):
    runs = defaultdict(list)
    if not path.exists():
        raise FileNotFoundError(f"Loss file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "step" not in record or "loss" not in record:
                continue

            run_id = str(record.get("run_id", "unknown"))
            runs[run_id].append((int(record["step"]), float(record["loss"])) )

    for run_id, series in runs.items():
        series.sort(key=lambda x: x[0])
        runs[run_id] = series

    return runs


def plot_loss_curves(train_path: Path, eval_path: Path, output_path: Path, title_suffix: str = ""):
    train_runs = load_loss_series(train_path)
    eval_runs = load_loss_series(eval_path) if eval_path and eval_path.exists() else {}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False)
    fig.suptitle(f"LLaDA Training Loss Curves{title_suffix}")

    ax_train, ax_eval = axes
    ax_train.set_title("Training Loss")
    ax_train.set_xlabel("Step")
    ax_train.set_ylabel("Loss")

    if not train_runs:
        ax_train.text(0.5, 0.5, "No training data", ha="center", va="center", transform=ax_train.transAxes)
    else:
        for run_id, series in sorted(train_runs.items()):
            steps, losses = zip(*series)
            label = run_id[-8:] if len(run_id) > 8 else run_id
            ax_train.plot(steps, losses, label=label)
        ax_train.legend()
        ax_train.grid(True, linestyle="--", alpha=0.3)

    ax_eval.set_title("Evaluation Loss")
    ax_eval.set_xlabel("Step")
    ax_eval.set_ylabel("Loss")

    if not eval_runs:
        ax_eval.text(0.5, 0.5, "No evaluation data", ha="center", va="center", transform=ax_eval.transAxes)
    else:
        for run_id, series in sorted(eval_runs.items()):
            steps, losses = zip(*series)
            label = run_id[-8:] if len(run_id) > 8 else run_id
            ax_eval.plot(steps, losses, marker="o", linestyle="-", label=label)
        ax_eval.legend()
        ax_eval.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot training and evaluation loss curves from JSONL logs")
    parser.add_argument("--train", type=Path, default=Path("outputs_offline/train_loss.jsonl"), help="Path to training loss JSONL file")
    parser.add_argument("--eval", type=Path, default=Path("outputs_offline/eval_loss.jsonl"), help="Path to evaluation loss JSONL file")
    parser.add_argument("--output", type=Path, default=Path("outputs_offline/loss_curves.png"), help="Output image path")
    parser.add_argument("--title", type=str, default="", help="Optional title suffix")
    args = parser.parse_args()

    suffix = f" - {args.title}" if args.title else ""
    plot_loss_curves(args.train, args.eval, args.output, title_suffix=suffix)


if __name__ == "__main__":
    main()
