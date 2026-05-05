#!/usr/bin/env python
"""Write full-task baseline predictions for the unified protocol."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["shortcut_prior", "eval_majority"],
        required=True,
        help=(
            "shortcut_prior uses the train-set metadata-prior answer stored in "
            "each row. eval_majority is a sanity baseline fitted on the supplied "
            "challenge distribution, not a valid training baseline."
        ),
    )
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.challenge)
    majority: dict[str, str] = {}
    if args.mode == "eval_majority":
        counts: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            counts[row["task"]][str(row["answer"])] += 1
        majority = {
            task: counter.most_common(1)[0][0]
            for task, counter in counts.items()
            if counter
        }

    predictions = []
    for row in rows:
        if args.mode == "shortcut_prior":
            prediction = row.get("shortcut_answer", "")
        else:
            prediction = majority.get(row["task"], "")
        predictions.append(
            {
                "example_id": row["example_id"],
                "prediction": prediction,
                "model": args.label or args.mode,
                "baseline_mode": args.mode,
            }
        )

    write_jsonl(predictions, args.output)
    print(f"Wrote {len(predictions):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
