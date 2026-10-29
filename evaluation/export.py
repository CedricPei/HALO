"""Export comparable averages for successful checkpoints as one English CSV."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
import tempfile

from .runner import RESULTS_DIR, _current_runs


TOKEN_FIELDS = ("input_tokens", "completion_tokens", "reasoning_tokens")
SUMMARY_FIELDS = (
    "snapshot_utc", "model", "scope", "successful_checkpoints",
    "mean_retrieval_rounds", "mean_api_calls",
    *(f"mean_{field}" for field in TOKEN_FIELDS),
)


def build_summary(runs, snapshot_utc):
    models = sorted({run["model"] for run in runs})
    rows = []
    successes = {model: set() for model in models}
    seen = set()
    for run in runs:
        for checkpoint in run["checkpoints"]:
            if checkpoint["agent"].get("decision") != "final":
                continue
            key = (run["source_path"], checkpoint["event_id"])
            identity = (run["model"], *key)
            if identity in seen:
                raise ValueError(f"Duplicate selected checkpoint: {identity}")
            seen.add(identity)
            successes[run["model"]].add(key)
            rows.append({
                "model": run["model"],
                "source_path": run["source_path"],
                "event_id": checkpoint["event_id"],
                "retrieval_rounds": checkpoint["metrics"]["intermediate_round_count"],
                "api_calls": len(checkpoint["agent"]["model_calls"]),
                **{field: checkpoint["metrics"][field] for field in TOKEN_FIELDS},
            })
    common = set.intersection(*successes.values()) if models else set()
    for row in rows:
        row["common_success"] = int((row["source_path"], row["event_id"]) in common)
    summary = []
    for model in models:
        successful = [row for row in rows if row["model"] == model]
        for scope, selected in (
            ("all_success", successful),
            ("common_success", [row for row in successful if row["common_success"]]),
        ):
            summary.append({
                "snapshot_utc": snapshot_utc,
                "model": model,
                "scope": scope,
                "successful_checkpoints": len(selected),
                **{f"mean_{metric}": round(fmean(row[metric] for row in selected), 6)
                   if selected else ""
                   for metric in ("retrieval_rounds", "api_calls", *TOKEN_FIELDS)},
            })
    return summary


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                     dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="Repeat to restrict the comparison")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    runs = [run for run, _ in _current_runs()
            if not args.model or run["model"] in args.model]
    if not runs:
        parser.error("No current results match the selection")
    missing = set(args.model or ()) - {run["model"] for run in runs}
    if missing:
        parser.error(f"No current results for: {', '.join(sorted(missing))}")
    summary = build_summary(runs, datetime.now(timezone.utc).isoformat())
    path = args.output_dir / "evaluation_summary.csv"
    write_csv(path, SUMMARY_FIELDS, summary)
    print(f"{path}: {len(summary)} rows")


if __name__ == "__main__":
    main()
