#!/usr/bin/env python
"""Evaluate full-task unified-protocol predictions."""

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


def fmt(value: float) -> str:
    return "NA" if value != value else f"{value:.3f}"


def bootstrap(values: list[int], iters: int, seed: int) -> tuple[float, float]:
    if not values or iters <= 0:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    samples = []
    for _ in range(iters):
        sample = [values[rng.randrange(len(values))] for _ in values]
        samples.append(rate(sample))
    samples.sort()
    return samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]


def grouped_bootstrap(values: list[int], group_ids: list[str], iters: int, seed: int) -> tuple[float, float]:
    if not values or iters <= 0:
        return float("nan"), float("nan")
    groups: dict[str, list[int]] = defaultdict(list)
    for value, group_id in zip(values, group_ids):
        groups[str(group_id)].append(value)
    keys = list(groups)
    if not keys:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    samples = []
    for _ in range(iters):
        numerator = 0
        denominator = 0
        for _ in keys:
            sampled_key = keys[rng.randrange(len(keys))]
            sampled_values = groups[sampled_key]
            numerator += sum(sampled_values)
            denominator += len(sampled_values)
        samples.append(numerator / denominator if denominator else float("nan"))
    samples = sorted(value for value in samples if value == value)
    if not samples:
        return float("nan"), float("nan")
    return samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]


def macro_scores(y_true: list[str], y_pred: list[str]) -> tuple[float, float]:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return float("nan"), float("nan")
    f1s = []
    recalls = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        recalls.append(recall)
    return sum(f1s) / len(f1s), sum(recalls) / len(recalls)


def ontology_lookup(task_spec: dict) -> dict[str, set[str]]:
    lookup = {}
    for task, spec in task_spec.get("tasks", {}).items():
        for candidate in spec.get("candidates", []):
            ids = set(candidate.get("ontology_term_ids", []))
            if ids:
                lookup[(task, candidate["normalized_label"])] = ids
    return lookup


def ontology_match(task: str, pred_norm: str, answer_norm: str, lookup: dict) -> bool | None:
    pred_ids = lookup.get((task, pred_norm))
    answer_ids = lookup.get((task, answer_norm))
    if not pred_ids or not answer_ids:
        return None
    return bool(pred_ids & answer_ids)


def csrs(accuracy: float, shortcut_agreement: float, candidate_count: int) -> float:
    if accuracy != accuracy or shortcut_agreement != shortcut_agreement or candidate_count <= 1:
        return float("nan")
    chance = 1.0 / candidate_count
    truth = max(0.0, (accuracy - chance) / (1.0 - chance))
    rejection = 1.0 - shortcut_agreement
    if truth + rejection <= 0:
        return 0.0
    return 2.0 * truth * rejection / (truth + rejection)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--label", default="model")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    challenge = {row["example_id"]: row for row in load_jsonl(args.challenge)}
    preds = {row["example_id"]: row for row in load_jsonl(args.predictions)}
    spec = json.loads(args.task_spec.read_text())
    ont = ontology_lookup(spec)

    candidate_counts = {
        task: max(1, len(task_info.get("candidates", [])))
        for task, task_info in spec.get("tasks", {}).items()
    }

    groups = defaultdict(
        lambda: {
            "correct": [],
            "shortcut": [],
            "ontology_correct": [],
            "ontology_shortcut": [],
            "truth": [],
            "pred": [],
            "group_id": [],
        }
    )
    ignored = 0
    for example_id, pred_row in preds.items():
        row = challenge.get(example_id)
        if row is None:
            ignored += 1
            continue
        task = row["task"]
        bucket = "conflict" if row.get("is_shortcut_conflict") else "aligned"
        pred_norm = normalize(pred_row.get("prediction", ""))
        answer_norm = normalize(row.get("answer", ""))
        shortcut_norm = normalize(row.get("shortcut_answer", ""))
        group_id = row.get("soma_joinid", example_id)
        correct = int(pred_norm == answer_norm)
        shortcut = int(pred_norm == shortcut_norm)
        ont_correct = ontology_match(task, pred_norm, answer_norm, ont)
        ont_shortcut = ontology_match(task, pred_norm, shortcut_norm, ont)
        keys = [
            (task, "all", "ALL"),
            (task, bucket, "ALL"),
            (task, "all", row.get("prior_name", "unknown")),
            (task, bucket, row.get("prior_name", "unknown")),
        ]
        for key in keys:
            groups[key]["correct"].append(correct)
            groups[key]["shortcut"].append(shortcut)
            groups[key]["truth"].append(answer_norm)
            groups[key]["pred"].append(pred_norm)
            groups[key]["group_id"].append(str(group_id))
            if ont_correct is not None:
                groups[key]["ontology_correct"].append(int(ont_correct))
            if ont_shortcut is not None:
                groups[key]["ontology_shortcut"].append(int(ont_shortcut))

    rows = []
    for (task, bucket, prior), values in sorted(groups.items()):
        n = len(values["correct"])
        if not n:
            continue
        should_interval = prior == "ALL" and n <= 20000
        low, high = (
            bootstrap(values["correct"], args.bootstrap_iters, args.seed)
            if should_interval
            else (float("nan"), float("nan"))
        )
        group_low, group_high = (
            grouped_bootstrap(values["correct"], values["group_id"], args.bootstrap_iters, args.seed)
            if should_interval
            else (float("nan"), float("nan"))
        )
        macro_f1, balanced_acc = macro_scores(values["truth"], values["pred"])
        accuracy = rate(values["correct"])
        shortcut_agreement = rate(values["shortcut"])
        tsm = accuracy - shortcut_agreement
        counter_shortcut = (
            csrs(accuracy, shortcut_agreement, candidate_counts.get(task, len(set(values["truth"]))))
            if bucket == "conflict"
            else float("nan")
        )
        rows.append(
            {
                "label": args.label,
                "task": task,
                "bucket": bucket,
                "prior_name": prior,
                "n": n,
                "n_groups": len(set(values["group_id"])),
                "accuracy": accuracy,
                "accuracy_ci_low": low,
                "accuracy_ci_high": high,
                "grouped_accuracy_ci_low": group_low,
                "grouped_accuracy_ci_high": group_high,
                "shortcut_agreement": shortcut_agreement,
                "ontology_shortcut_agreement": rate(values["ontology_shortcut"]),
                "ontology_accuracy": rate(values["ontology_correct"]),
                "ontology_coverage": len(values["ontology_correct"]) / n if n else float("nan"),
                "macro_f1": macro_f1,
                "balanced_accuracy": balanced_acc,
                "truth_shortcut_margin": tsm,
                "csrs": counter_shortcut,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Unified Prediction Evaluation",
        "",
        f"- Label: {args.label}",
        f"- Challenge rows: {len(challenge):,}",
        f"- Prediction rows: {len(preds):,}",
        f"- Ignored prediction rows: {ignored:,}",
        "",
        "## Task Summary",
        "",
        "| task | bucket | n | groups | accuracy | row 95% CI | grouped 95% CI | shortcut agreement | ontology accuracy | ontology coverage | macro F1 | balanced acc. | TSM | CSRS |",
        "|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
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
                    str(row["n_groups"]),
                    fmt(row["accuracy"]),
                    f"[{fmt(row['accuracy_ci_low'])},{fmt(row['accuracy_ci_high'])}]",
                    f"[{fmt(row['grouped_accuracy_ci_low'])},{fmt(row['grouped_accuracy_ci_high'])}]",
                    fmt(row["shortcut_agreement"]),
                    fmt(row["ontology_accuracy"]),
                    fmt(row["ontology_coverage"]),
                    fmt(row["macro_f1"]),
                    fmt(row["balanced_accuracy"]),
                    fmt(row["truth_shortcut_margin"]),
                    fmt(row["csrs"]),
                ]
            )
            + " |"
        )
    args.output_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
