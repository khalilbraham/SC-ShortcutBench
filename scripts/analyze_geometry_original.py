#!/usr/bin/env python3
"""Analyze whether model score margins track metadata shortcut strength.

This is a mediation-style diagnostic:

1. train metadata gives a shortcut-prior strength, approximated by
   logit(prior_purity);
2. model inference gives a shortcut score margin,
   score(shortcut) - score(truth);
3. a positive association means stronger construction priors align with stronger
   model preference for the shortcut label.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def logit(value: float, eps: float = 1e-6) -> float:
    value = min(max(float(value), eps), 1.0 - eps)
    return math.log(value / (1.0 - value))


def rate(values: list[int]) -> float:
    return sum(values) / len(values) if values else float("nan")


def mean(values: list[float]) -> float:
    valid = [v for v in values if v == v]
    return sum(valid) / len(valid) if valid else float("nan")


def variance(values: list[float]) -> float:
    valid = [v for v in values if v == v]
    if len(valid) < 2:
        return float("nan")
    m = mean(valid)
    return sum((v - m) ** 2 for v in valid) / (len(valid) - 1)


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pairs) < 3:
        return float("nan")
    xvals = [x for x, _y in pairs]
    yvals = [y for _x, y in pairs]
    xm = mean(xvals)
    ym = mean(yvals)
    xv = sum((x - xm) ** 2 for x in xvals)
    yv = sum((y - ym) ** 2 for y in yvals)
    if xv <= 0 or yv <= 0:
        return float("nan")
    return sum((x - xm) * (y - ym) for x, y in pairs) / math.sqrt(xv * yv)


def ranks(values: list[float]) -> list[float]:
    indexed = sorted((value, idx) for idx, value in enumerate(values))
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][0] == indexed[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for _value, idx in indexed[i:j]:
            out[idx] = avg_rank
        i = j
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pairs) < 3:
        return float("nan")
    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def slope(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pairs) < 3:
        return float("nan")
    xvals = [x for x, _y in pairs]
    yvals = [y for _x, y in pairs]
    xm = mean(xvals)
    ym = mean(yvals)
    denom = sum((x - xm) ** 2 for x in xvals)
    if denom <= 0:
        return float("nan")
    return sum((x - xm) * (y - ym) for x, y in pairs) / denom


def fmt(value: float) -> str:
    return "NA" if value != value else f"{value:.3f}"


def candidate_score(row: dict, role: str) -> float:
    for candidate in row.get("candidates", []):
        if candidate.get("role") == role:
            return float(candidate["score"])
    return float("nan")


def get_scores(prediction: dict) -> tuple[float, float]:
    score_key_pairs = [
        ("score_answer", "score_shortcut"),
        ("answer_score", "shortcut_score"),
        ("score_answer_centroid", "score_shortcut_centroid"),
    ]
    for answer_key, shortcut_key in score_key_pairs:
        answer = prediction.get(answer_key)
        shortcut = prediction.get(shortcut_key)
        if answer is not None and shortcut is not None:
            return float(answer), float(shortcut)
    return candidate_score(prediction, "truth"), candidate_score(prediction, "shortcut")


def summarize(label: str, records: list[dict]) -> dict:
    prior_strength = [record["prior_strength"] for record in records]
    margin = [record["shortcut_minus_truth"] for record in records]
    shortcut_top = [int(record["shortcut_minus_truth"] > 0) for record in records]
    return {
        "group": label,
        "n": len(records),
        "mean_prior_purity": mean([record["prior_purity"] for record in records]),
        "mean_prior_strength_logit": mean(prior_strength),
        "mean_shortcut_minus_truth_score": mean(margin),
        "shortcut_preferred_rate": rate(shortcut_top),
        "truth_preferred_rate": 1.0 - rate(shortcut_top) if shortcut_top else float("nan"),
        "pearson_prior_strength_vs_margin": pearson(prior_strength, margin),
        "spearman_prior_strength_vs_margin": spearman(prior_strength, margin),
        "ols_slope_margin_on_prior_strength": slope(prior_strength, margin),
        "margin_variance": variance(margin),
    }


def quantile_bins(records: list[dict], n_bins: int) -> list[tuple[str, list[dict]]]:
    if not records:
        return []
    ordered = sorted(records, key=lambda row: row["prior_purity"])
    bins = []
    for i in range(n_bins):
        start = round(i * len(ordered) / n_bins)
        end = round((i + 1) * len(ordered) / n_bins)
        subset = ordered[start:end]
        if subset:
            bins.append((f"prior_purity_q{i + 1}", subset))
    return bins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--label", default="model")
    parser.add_argument("--conflicts-only", action="store_true")
    args = parser.parse_args()

    challenge = {row["example_id"]: row for row in load_jsonl(args.challenge)}
    predictions = load_jsonl(args.predictions)
    records = []
    ignored = 0
    missing_scores = 0
    for pred in predictions:
        row = challenge.get(pred.get("example_id"))
        if row is None:
            ignored += 1
            continue
        if args.conflicts_only and not row.get("is_shortcut_conflict"):
            continue
        answer_score, shortcut_score = get_scores(pred)
        if answer_score != answer_score or shortcut_score != shortcut_score:
            missing_scores += 1
            continue
        prior_purity = float(row.get("prior_purity", float("nan")))
        if prior_purity != prior_purity:
            continue
        records.append(
            {
                "example_id": row["example_id"],
                "task": row["task"],
                "prior_name": row["prior_name"],
                "prior_family": row.get("prior_family", ""),
                "is_shortcut_conflict": bool(row.get("is_shortcut_conflict")),
                "prior_purity": prior_purity,
                "prior_support": int(row.get("prior_support", 0)),
                "prior_strength": logit(prior_purity),
                "answer_score": answer_score,
                "shortcut_score": shortcut_score,
                "shortcut_minus_truth": shortcut_score - answer_score,
            }
        )

    grouped: dict[str, list[dict]] = {
        "ALL": records,
        "conflict": [row for row in records if row["is_shortcut_conflict"]],
        "aligned": [row for row in records if not row["is_shortcut_conflict"]],
    }
    for task in sorted({row["task"] for row in records}):
        grouped[f"task={task}"] = [row for row in records if row["task"] == task]
        grouped[f"task={task};conflict"] = [
            row for row in records if row["task"] == task and row["is_shortcut_conflict"]
        ]
    for prior in sorted({row["prior_name"] for row in records}):
        grouped[f"prior={prior}"] = [row for row in records if row["prior_name"] == prior]
        grouped[f"prior={prior};conflict"] = [
            row for row in records if row["prior_name"] == prior and row["is_shortcut_conflict"]
        ]
    for name, subset in quantile_bins(grouped["conflict"], 4):
        grouped[f"conflict;{name}"] = subset

    summaries = [summarize(name, subset) for name, subset in grouped.items() if subset]
    summaries.sort(key=lambda row: (0 if row["group"] in {"ALL", "conflict", "aligned"} else 1, row["group"]))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()) if summaries else [])
        if summaries:
            writer.writeheader()
            writer.writerows(summaries)

    lines = [
        "# Shortcut Mediation Score-Margin Analysis",
        "",
        f"- Label: {args.label}",
        f"- Prediction rows: {len(predictions):,}",
        f"- Matched scored rows: {len(records):,}",
        f"- Ignored rows absent from challenge: {ignored:,}",
        f"- Rows skipped because truth/shortcut scores were unavailable: {missing_scores:,}",
        "",
        "Prior strength is approximated as `logit(prior_purity)` because the released challenge rows store the shortcut mode probability but not the full train-set probability of the held-out true label under the same key.",
        "",
        "Positive `mean shortcut-truth score` means the model scores the shortcut text above the truth text on average.",
        "",
        "## Summary",
        "",
        "| group | n | prior purity | shortcut-truth score | shortcut preferred | Pearson | Spearman | slope |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    primary_groups = [
        "ALL",
        "conflict",
        "aligned",
        "task=cell_type_prediction;conflict",
        "task=disease_prediction;conflict",
        "task=tissue_general_prediction;conflict",
        "prior=celltype_to_tissue;conflict",
        "prior=tissue_to_disease;conflict",
        "prior=tissue_general_to_disease;conflict",
        "prior=combined_celltype_disease_to_tissue;conflict",
        "conflict;prior_purity_q1",
        "conflict;prior_purity_q2",
        "conflict;prior_purity_q3",
        "conflict;prior_purity_q4",
    ]
    summary_by_name = {row["group"]: row for row in summaries}
    for name in primary_groups:
        row = summary_by_name.get(name)
        if not row:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    row["group"],
                    str(row["n"]),
                    fmt(row["mean_prior_purity"]),
                    fmt(row["mean_shortcut_minus_truth_score"]),
                    fmt(row["shortcut_preferred_rate"]),
                    fmt(row["pearson_prior_strength_vs_margin"]),
                    fmt(row["spearman_prior_strength_vs_margin"]),
                    fmt(row["ols_slope_margin_on_prior_strength"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## All Groups",
            "",
            "| group | n | prior purity | shortcut-truth score | shortcut preferred | truth preferred | Pearson | Spearman | slope |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["group"],
                    str(row["n"]),
                    fmt(row["mean_prior_purity"]),
                    fmt(row["mean_shortcut_minus_truth_score"]),
                    fmt(row["shortcut_preferred_rate"]),
                    fmt(row["truth_preferred_rate"]),
                    fmt(row["pearson_prior_strength_vs_margin"]),
                    fmt(row["spearman_prior_strength_vs_margin"]),
                    fmt(row["ols_slope_margin_on_prior_strength"]),
                ]
            )
            + " |"
        )

    args.output_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
