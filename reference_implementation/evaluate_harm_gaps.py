#!/usr/bin/env python
"""Evaluate shortcut harm gaps from full-task predictions.

This script expects non-degenerate task predictions. It can be run on pairwise
diagnostic outputs, but it will only report SHG for task/prior groups that have
both aligned and conflict rows covered by predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def normalize(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rate(values: list[int]) -> float:
    return sum(values) / len(values) if values else float("nan")


def bootstrap_gap(aligned: list[int], conflict: list[int], iters: int, seed: int) -> tuple[float, float]:
    if not aligned or not conflict or iters <= 0:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    gaps = []
    for _ in range(iters):
        aligned_sample = [aligned[rng.randrange(len(aligned))] for _ in aligned]
        conflict_sample = [conflict[rng.randrange(len(conflict))] for _ in conflict]
        gaps.append(rate(aligned_sample) - rate(conflict_sample))
    gaps.sort()
    low = gaps[int(0.025 * (len(gaps) - 1))]
    high = gaps[int(0.975 * (len(gaps) - 1))]
    return low, high


def grouped_bootstrap_gap(
    aligned: list[tuple[str, int]], conflict: list[tuple[str, int]], iters: int, seed: int
) -> tuple[float, float]:
    if not aligned or not conflict or iters <= 0:
        return float("nan"), float("nan")
    aligned_groups: dict[str, list[int]] = defaultdict(list)
    conflict_groups: dict[str, list[int]] = defaultdict(list)
    for group_id, value in aligned:
        aligned_groups[str(group_id)].append(value)
    for group_id, value in conflict:
        conflict_groups[str(group_id)].append(value)
    aligned_keys = list(aligned_groups)
    conflict_keys = list(conflict_groups)
    if not aligned_keys or not conflict_keys:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    gaps = []
    for _ in range(iters):
        aligned_num = 0
        aligned_den = 0
        for _ in aligned_keys:
            key = aligned_keys[rng.randrange(len(aligned_keys))]
            values = aligned_groups[key]
            aligned_num += sum(values)
            aligned_den += len(values)
        conflict_num = 0
        conflict_den = 0
        for _ in conflict_keys:
            key = conflict_keys[rng.randrange(len(conflict_keys))]
            values = conflict_groups[key]
            conflict_num += sum(values)
            conflict_den += len(values)
        if aligned_den and conflict_den:
            gaps.append(aligned_num / aligned_den - conflict_num / conflict_den)
    gaps.sort()
    if not gaps:
        return float("nan"), float("nan")
    low = gaps[int(0.025 * (len(gaps) - 1))]
    high = gaps[int(0.975 * (len(gaps) - 1))]
    return low, high


def format_float(value: float) -> str:
    if value != value:
        return "NA"
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--task-spec", type=Path)
    parser.add_argument("--label", default="model")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    challenge = {row["example_id"]: row for row in load_jsonl(args.challenge)}
    predictions = load_jsonl(args.predictions)
    if args.task_spec:
        spec = json.loads(args.task_spec.read_text())
        candidate_counts = {
            task: max(1, len(task_info.get("candidates", [])))
            for task, task_info in spec.get("tasks", {}).items()
        }
    else:
        task_answers: dict[str, set[str]] = defaultdict(set)
        for row in challenge.values():
            task_answers[row.get("task", "")].add(normalize(row.get("answer", "")))
        candidate_counts = {task: max(1, len(labels)) for task, labels in task_answers.items()}

    grouped: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: {"aligned": [], "conflict": [], "aligned_grouped": [], "conflict_grouped": [], "shortcut_conflict": []}
    )
    ignored = 0
    for pred in predictions:
        row = challenge.get(pred.get("example_id"))
        if row is None:
            ignored += 1
            continue
        prediction = normalize(pred.get("prediction", ""))
        answer = normalize(row.get("answer", ""))
        shortcut = normalize(row.get("shortcut_answer", ""))
        task = row.get("task", "")
        prior = row.get("prior_name", "ALL")
        bucket = "conflict" if row.get("is_shortcut_conflict") else "aligned"
        group_id = row.get("soma_joinid", pred.get("example_id", ""))
        correct = int(prediction == answer)
        grouped[(task, "ALL")][bucket].append(correct)
        grouped[(task, prior)][bucket].append(correct)
        grouped[(task, "ALL")][f"{bucket}_grouped"].append((str(group_id), correct))
        grouped[(task, prior)][f"{bucket}_grouped"].append((str(group_id), correct))
        if bucket == "conflict":
            shortcut_match = int(prediction == shortcut)
            grouped[(task, "ALL")]["shortcut_conflict"].append(shortcut_match)
            grouped[(task, prior)]["shortcut_conflict"].append(shortcut_match)

    rows = []
    for (task, prior), values in sorted(grouped.items()):
        aligned = values["aligned"]
        conflict = values["conflict"]
        if not aligned and not conflict:
            continue
        aligned_acc = rate(aligned)
        conflict_acc = rate(conflict)
        shortcut_agreement = rate(values["shortcut_conflict"])
        shg = aligned_acc - conflict_acc if aligned and conflict else float("nan")
        should_interval = prior == "ALL" and (len(aligned) + len(conflict)) <= 20000
        low, high = (
            bootstrap_gap(aligned, conflict, args.bootstrap_iters, args.seed)
            if should_interval
            else (float("nan"), float("nan"))
        )
        grouped_low, grouped_high = (
            grouped_bootstrap_gap(values["aligned_grouped"], values["conflict_grouped"], args.bootstrap_iters, args.seed)
            if should_interval
            else (float("nan"), float("nan"))
        )
        tsm = conflict_acc - shortcut_agreement if conflict else float("nan")
        chance = 1.0 / candidate_counts.get(task, max(1, len(set(conflict))))
        truth = max(0.0, (conflict_acc - chance) / (1.0 - chance)) if conflict_acc == conflict_acc and chance < 1 else float("nan")
        rejection = 1.0 - shortcut_agreement if shortcut_agreement == shortcut_agreement else float("nan")
        csrs = float("nan")
        if truth == truth and rejection == rejection:
            csrs = 0.0 if truth + rejection <= 0 else 2.0 * truth * rejection / (truth + rejection)
        rows.append(
            {
                "label": args.label,
                "task": task,
                "prior_name": prior,
                "n_aligned": len(aligned),
                "n_conflict": len(conflict),
                "n_aligned_groups": len({group_id for group_id, _ in values["aligned_grouped"]}),
                "n_conflict_groups": len({group_id for group_id, _ in values["conflict_grouped"]}),
                "aligned_accuracy": aligned_acc,
                "conflict_accuracy": conflict_acc,
                "shortcut_agreement_conflict": shortcut_agreement,
                "truth_shortcut_margin": tsm,
                "csrs": csrs,
                "shortcut_harm_gap": shg,
                "shg_ci_low": low,
                "shg_ci_high": high,
                "grouped_shg_ci_low": grouped_low,
                "grouped_shg_ci_high": grouped_high,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Shortcut Harm Gap Report",
        "",
        f"- Label: {args.label}",
        f"- Challenge rows: {len(challenge):,}",
        f"- Prediction rows: {len(predictions):,}",
        f"- Ignored prediction rows: {ignored:,}",
        "",
        "| task | prior | n aligned | n conflict | aligned acc. | conflict acc. | shortcut agreement | TSM | CSRS | SHG | row 95% CI | grouped 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["task"],
                    row["prior_name"],
                    str(row["n_aligned"]),
                    str(row["n_conflict"]),
                    format_float(row["aligned_accuracy"]),
                    format_float(row["conflict_accuracy"]),
                    format_float(row["shortcut_agreement_conflict"]),
                    format_float(row["truth_shortcut_margin"]),
                    format_float(row["csrs"]),
                    format_float(row["shortcut_harm_gap"]),
                    f"[{format_float(row['shg_ci_low'])},{format_float(row['shg_ci_high'])}]",
                    f"[{format_float(row['grouped_shg_ci_low'])},{format_float(row['grouped_shg_ci_high'])}]",
                ]
            )
            + " |"
        )
    args.output_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
