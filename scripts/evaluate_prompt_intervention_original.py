#!/usr/bin/env python3
"""Evaluate targeted shortcut-intervention predictions."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rate(values: list[int]) -> float:
    return sum(values) / len(values) if values else float("nan")


def fmt(value: float) -> str:
    return "NA" if value != value else f"{value:.3f}"


def bootstrap(values: list[int], iters: int, seed: int) -> tuple[float, float]:
    if not values or iters <= 0:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    samples = []
    for _ in range(iters):
        samples.append(rate([values[rng.randrange(len(values))] for _ in values]))
    samples.sort()
    return samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]


def get_score(row: dict, role: str) -> float:
    for candidate in row.get("candidates", []):
        if candidate.get("role") == role:
            return float(candidate["score"])
    return float("nan")


def get_rank(row: dict, role: str) -> float:
    for candidate in row.get("candidates", []):
        if candidate.get("role") == role:
            return float(candidate["rank"])
    return float("nan")


def mean(values: list[float]) -> float:
    valid = [value for value in values if value == value]
    return sum(valid) / len(valid) if valid else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--label", default="model")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    predictions = load_jsonl(args.predictions)
    groups = defaultdict(
        lambda: {
            "truth_top": [],
            "shortcut_top": [],
            "negative_top": [],
            "truth_minus_shortcut": [],
            "shortcut_minus_truth": [],
            "truth_rank": [],
            "shortcut_rank": [],
        }
    )

    for row in predictions:
        task = row["task"]
        bucket = "conflict" if row.get("is_shortcut_conflict") else "aligned"
        top_role = row.get("top_role")
        truth_top = int(top_role in {"truth", "truth_shortcut"})
        shortcut_top = int(top_role in {"shortcut", "truth_shortcut"})
        negative_top = int(top_role == "negative")
        keys = [(task, bucket, "ALL"), (task, "all", "ALL")]
        if row.get("prior_name"):
            keys.extend([(task, bucket, row["prior_name"]), (task, "all", row["prior_name"])])
        truth_score = get_score(row, "truth")
        shortcut_score = get_score(row, "shortcut")
        truth_rank = get_rank(row, "truth")
        shortcut_rank = get_rank(row, "shortcut")
        for key in keys:
            values = groups[key]
            values["truth_top"].append(truth_top)
            values["shortcut_top"].append(shortcut_top)
            values["negative_top"].append(negative_top)
            if truth_score == truth_score and shortcut_score == shortcut_score:
                values["truth_minus_shortcut"].append(truth_score - shortcut_score)
                values["shortcut_minus_truth"].append(shortcut_score - truth_score)
            values["truth_rank"].append(truth_rank)
            values["shortcut_rank"].append(shortcut_rank)

    rows = []
    for (task, bucket, prior), values in sorted(groups.items()):
        if not values["truth_top"]:
            continue
        low, high = bootstrap(values["truth_top"], args.bootstrap_iters, args.seed)
        truth_rate = rate(values["truth_top"])
        shortcut_rate = rate(values["shortcut_top"])
        rows.append(
            {
                "label": args.label,
                "task": task,
                "bucket": bucket,
                "prior_name": prior,
                "n": len(values["truth_top"]),
                "truth_top_rate": truth_rate,
                "truth_ci_low": low,
                "truth_ci_high": high,
                "shortcut_top_rate": shortcut_rate,
                "negative_top_rate": rate(values["negative_top"]),
                "truth_shortcut_margin": truth_rate - shortcut_rate,
                "mean_truth_minus_shortcut_score": mean(values["truth_minus_shortcut"]),
                "mean_shortcut_minus_truth_score": mean(values["shortcut_minus_truth"]),
                "mean_truth_rank": mean(values["truth_rank"]),
                "mean_shortcut_rank": mean(values["shortcut_rank"]),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Shortcut Intervention Evaluation",
        "",
        f"- Label: {args.label}",
        f"- Prediction rows: {len(predictions):,}",
        "",
        "## Task Summary",
        "",
        "| task | bucket | n | truth top | 95% CI | shortcut top | negative top | TSM | mean truth-shortcut score | mean truth rank | mean shortcut rank |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["prior_name"] != "ALL":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    row["task"],
                    row["bucket"],
                    str(row["n"]),
                    fmt(row["truth_top_rate"]),
                    f"[{fmt(row['truth_ci_low'])},{fmt(row['truth_ci_high'])}]",
                    fmt(row["shortcut_top_rate"]),
                    fmt(row["negative_top_rate"]),
                    fmt(row["truth_shortcut_margin"]),
                    fmt(row["mean_truth_minus_shortcut_score"]),
                    fmt(row["mean_truth_rank"]),
                    fmt(row["mean_shortcut_rank"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- On conflict rows, `shortcut top` is the targeted shortcut-following signal.",
            "- `mean truth-shortcut score` below zero means the shortcut text is ranked above the truth text on average.",
            "- `negative top` separates shortcut errors from generic retrieval errors.",
            "",
            "## By Prior",
            "",
            "| prior | task | bucket | n | truth top | shortcut top | negative top | mean truth-shortcut score |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["prior_name"] == "ALL":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    row["prior_name"],
                    row["task"],
                    row["bucket"],
                    str(row["n"]),
                    fmt(row["truth_top_rate"]),
                    fmt(row["shortcut_top_rate"]),
                    fmt(row["negative_top_rate"]),
                    fmt(row["mean_truth_minus_shortcut_score"]),
                ]
            )
            + " |"
        )

    args.output_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
