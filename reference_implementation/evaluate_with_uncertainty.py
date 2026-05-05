#!/usr/bin/env python
"""Evaluate shortcut predictions with bootstrap CIs and permutation tests."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--permutation-iters", type=int, default=1000)
    parser.add_argument(
        "--group-key",
        default=None,
        help="Optional challenge-row field for grouped bootstrap, e.g. soma_joinid.",
    )
    parser.add_argument(
        "--ontology-ancestors",
        default=None,
        help=(
            "Optional JSON mapping ontology term IDs to ancestor term IDs. "
            "When supplied, ontology matching counts exact ID matches and "
            "parent/child matches."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def norm(text) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def label_match(prediction: str, answer: str) -> bool:
    pred = norm(prediction)
    ans = norm(answer)
    return pred == ans or bool(ans and ans in pred)


def exact_match(prediction: str, answer: str) -> bool:
    return norm(prediction) == norm(answer)


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prediction_map(path: str) -> dict[str, str]:
    preds = {}
    for row in load_jsonl(path):
        if "example_id" in row:
            preds[row["example_id"]] = row.get("prediction", "")
    return preds


def build_ontology_lookup(rows: list[dict]) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], int]]:
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in rows:
        target = row.get("target", "")
        label = norm(row.get("answer", ""))
        term_id = str(row.get("target_ontology_term_id", "")).strip()
        if target and label and term_id:
            counts[(target, label)][term_id] += 1
    lookup = {}
    ambiguous = {}
    for key, counter in counts.items():
        most_common = counter.most_common()
        lookup[key] = most_common[0][0]
        if len(counter) > 1:
            ambiguous[key] = len(counter)
    return lookup, ambiguous


def load_ontology_ancestors(path: str | None) -> dict[str, set[str]]:
    if not path:
        return {}
    with open(path) as handle:
        raw = json.load(handle)
    return {str(key): {str(value) for value in values} for key, values in raw.items()}


def ontology_id_for_label(row: dict, label: str, lookup: dict[tuple[str, str], str]) -> str:
    if norm(label) == norm(row.get("answer", "")):
        return str(row.get("target_ontology_term_id", "")).strip()
    return lookup.get((row.get("target", ""), norm(label)), "")


def ontology_match_ids(pred_id: str, label_id: str, ancestors: dict[str, set[str]]) -> bool:
    if not pred_id or not label_id:
        return False
    if pred_id == label_id:
        return True
    if not ancestors:
        return False
    return label_id in ancestors.get(pred_id, set()) or pred_id in ancestors.get(label_id, set())


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


def bootstrap_ci(values: list[int], iters: int, rng: random.Random) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    samples = []
    for _ in range(iters):
        samples.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return quantile(samples, 0.025), quantile(samples, 0.975)


def grouped_bootstrap_ci(
    values: list[int],
    group_ids: list[str],
    iters: int,
    rng: random.Random,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    grouped: dict[str, list[int]] = defaultdict(list)
    for group_id, value in zip(group_ids, values):
        grouped[str(group_id)].append(value)
    keys = list(grouped)
    if not keys:
        return float("nan"), float("nan")
    samples = []
    for _ in range(iters):
        total = 0
        count = 0
        for _ in range(len(keys)):
            key = keys[rng.randrange(len(keys))]
            selected = grouped[key]
            total += sum(selected)
            count += len(selected)
        samples.append(total / count if count else float("nan"))
    return quantile(samples, 0.025), quantile(samples, 0.975)


def permutation_p_value(
    predictions: list[str],
    labels: list[str],
    observed_rate: float,
    iters: int,
    rng: random.Random,
) -> float:
    if not predictions or not labels:
        return float("nan")
    ge = 0
    shuffled = list(labels)
    n = len(labels)
    for _ in range(iters):
        rng.shuffle(shuffled)
        rate = sum(
            label_match(prediction, label)
            for prediction, label in zip(predictions, shuffled)
        ) / n
        if rate >= observed_rate:
            ge += 1
    return (ge + 1) / (iters + 1)


def bucket_key(row: dict) -> str:
    return "conflict" if row.get("is_shortcut_conflict") else "aligned"


def summarize_group(
    name: tuple[str, str, str],
    rows: list[dict],
    bootstrap_iters: int,
    permutation_iters: int,
    group_key: str | None,
    ontology_lookup: dict[tuple[str, str], str],
    ontology_ancestors: dict[str, set[str]],
    rng: random.Random,
) -> dict:
    task, bucket, prior = name
    pred_texts = [row["prediction"] for row in rows]
    answers = [row["answer"] for row in rows]
    shortcuts = [row["shortcut_answer"] for row in rows]
    correct = [
        int(label_match(prediction, answer))
        for prediction, answer in zip(pred_texts, answers)
    ]
    shortcut_agree = [
        int(label_match(prediction, shortcut))
        for prediction, shortcut in zip(pred_texts, shortcuts)
    ]
    exact_correct = [
        int(exact_match(prediction, answer))
        for prediction, answer in zip(pred_texts, answers)
    ]
    exact_shortcut_agree = [
        int(exact_match(prediction, shortcut))
        for prediction, shortcut in zip(pred_texts, shortcuts)
    ]
    ontology_correct = []
    ontology_shortcut_agree = []
    ontology_correct_available = []
    ontology_shortcut_available = []
    for row, prediction in zip(rows, pred_texts):
        pred_id = ontology_id_for_label(row, prediction, ontology_lookup)
        answer_id = ontology_id_for_label(row, row["answer"], ontology_lookup)
        shortcut_id = ontology_id_for_label(row, row["shortcut_answer"], ontology_lookup)
        answer_available = bool(pred_id and answer_id)
        shortcut_available = bool(pred_id and shortcut_id)
        ontology_correct_available.append(int(answer_available))
        ontology_shortcut_available.append(int(shortcut_available))
        ontology_correct.append(int(answer_available and ontology_match_ids(pred_id, answer_id, ontology_ancestors)))
        ontology_shortcut_agree.append(int(shortcut_available and ontology_match_ids(pred_id, shortcut_id, ontology_ancestors)))
    accuracy = mean(correct)
    shortcut_rate = mean(shortcut_agree)
    acc_low, acc_high = bootstrap_ci(correct, bootstrap_iters, rng)
    shortcut_low, shortcut_high = bootstrap_ci(shortcut_agree, bootstrap_iters, rng)
    group_ids = [row.get(group_key, row.get("example_id", "")) for row in rows] if group_key else []
    group_acc_low, group_acc_high = (
        grouped_bootstrap_ci(correct, group_ids, bootstrap_iters, rng)
        if group_key
        else (float("nan"), float("nan"))
    )
    group_shortcut_low, group_shortcut_high = (
        grouped_bootstrap_ci(shortcut_agree, group_ids, bootstrap_iters, rng)
        if group_key
        else (float("nan"), float("nan"))
    )
    ontology_acc_low, ontology_acc_high = bootstrap_ci(ontology_correct, bootstrap_iters, rng)
    ontology_shortcut_low, ontology_shortcut_high = bootstrap_ci(ontology_shortcut_agree, bootstrap_iters, rng)
    ontology_group_acc_low, ontology_group_acc_high = (
        grouped_bootstrap_ci(ontology_correct, group_ids, bootstrap_iters, rng)
        if group_key
        else (float("nan"), float("nan"))
    )
    ontology_group_shortcut_low, ontology_group_shortcut_high = (
        grouped_bootstrap_ci(ontology_shortcut_agree, group_ids, bootstrap_iters, rng)
        if group_key
        else (float("nan"), float("nan"))
    )
    acc_perm_p = permutation_p_value(
        pred_texts, answers, accuracy, permutation_iters, rng
    )
    shortcut_perm_p = permutation_p_value(
        pred_texts, shortcuts, shortcut_rate, permutation_iters, rng
    )
    return {
        "task": task,
        "bucket": bucket,
        "prior_name": prior,
        "n": len(rows),
        "n_groups": len(set(group_ids)) if group_key else "",
        "accuracy": accuracy,
        "accuracy_ci_low": acc_low,
        "accuracy_ci_high": acc_high,
        "accuracy_group_ci_low": group_acc_low,
        "accuracy_group_ci_high": group_acc_high,
        "accuracy_perm_p": acc_perm_p,
        "shortcut_agreement": shortcut_rate,
        "shortcut_ci_low": shortcut_low,
        "shortcut_ci_high": shortcut_high,
        "shortcut_group_ci_low": group_shortcut_low,
        "shortcut_group_ci_high": group_shortcut_high,
        "shortcut_perm_p": shortcut_perm_p,
        "exact_accuracy": mean(exact_correct),
        "exact_shortcut_agreement": mean(exact_shortcut_agree),
        "ontology_accuracy": mean(ontology_correct),
        "ontology_accuracy_ci_low": ontology_acc_low,
        "ontology_accuracy_ci_high": ontology_acc_high,
        "ontology_accuracy_group_ci_low": ontology_group_acc_low,
        "ontology_accuracy_group_ci_high": ontology_group_acc_high,
        "ontology_shortcut_agreement": mean(ontology_shortcut_agree),
        "ontology_shortcut_ci_low": ontology_shortcut_low,
        "ontology_shortcut_ci_high": ontology_shortcut_high,
        "ontology_shortcut_group_ci_low": ontology_group_shortcut_low,
        "ontology_shortcut_group_ci_high": ontology_group_shortcut_high,
        "ontology_answer_coverage": mean(ontology_correct_available),
        "ontology_shortcut_coverage": mean(ontology_shortcut_available),
    }


def fmt(value: float) -> str:
    if value != value:
        return ""
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    challenge_rows = load_jsonl(args.challenge)
    challenge = {row["example_id"]: row for row in challenge_rows}
    ontology_lookup, ambiguous_ontology_labels = build_ontology_lookup(challenge_rows)
    ontology_ancestors = load_ontology_ancestors(args.ontology_ancestors)
    preds = prediction_map(args.predictions)
    label = args.label or Path(args.predictions).stem

    matched = []
    ignored = 0
    for example_id, prediction in preds.items():
        row = challenge.get(example_id)
        if row is None:
            ignored += 1
            continue
        enriched = dict(row)
        enriched["prediction"] = prediction
        enriched["shortcut_ontology_term_id"] = ontology_id_for_label(
            enriched,
            enriched.get("shortcut_answer", ""),
            ontology_lookup,
        )
        matched.append(enriched)

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in matched:
        task = row.get("task", "")
        bucket = bucket_key(row)
        prior = row.get("prior_name", "ALL")
        for key in [
            (task, bucket, "ALL"),
            (task, "all", "ALL"),
            (task, bucket, prior),
            (task, "all", prior),
        ]:
            groups.setdefault(key, []).append(row)

    summary_rows = [
        summarize_group(
            key,
            group,
            args.bootstrap_iters,
            args.permutation_iters,
            args.group_key,
            ontology_lookup,
            ontology_ancestors,
            rng,
        )
        for key, group in sorted(groups.items())
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{label}_uncertainty.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# Shortcut Evaluation With Uncertainty",
        "",
        f"- Label: {label}",
        f"- Challenge rows: {len(challenge):,}",
        f"- Prediction rows matched: {len(matched):,}",
        f"- Prediction rows ignored: {ignored:,}",
        f"- Bootstrap iterations: {args.bootstrap_iters:,}",
        f"- Permutation iterations: {args.permutation_iters:,}",
        f"- Grouped bootstrap key: `{args.group_key}`" if args.group_key else "- Grouped bootstrap key: none",
        f"- Ontology ancestor file: `{args.ontology_ancestors}`" if args.ontology_ancestors else "- Ontology ancestor file: none; ontology scoring uses exact term-id equality",
        f"- Ambiguous normalized label-to-ontology mappings: {len(ambiguous_ontology_labels):,}",
        "",
        "## Task Summary",
        "",
        "| task | bucket | n | groups | accuracy | 95% CI | group 95% CI | perm p | shortcut agreement | 95% CI | group 95% CI | perm p | ontology acc. | ontology shortcut | ontology coverage |",
        "|---|---|---:|---:|---:|---|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["prior_name"] != "ALL":
            continue
        lines.append(
            f"| {row['task']} | {row['bucket']} | {row['n']} | {row['n_groups']} | "
            f"{fmt(row['accuracy'])} | [{fmt(row['accuracy_ci_low'])}, {fmt(row['accuracy_ci_high'])}] | "
            f"[{fmt(row['accuracy_group_ci_low'])}, {fmt(row['accuracy_group_ci_high'])}] | "
            f"{fmt(row['accuracy_perm_p'])} | {fmt(row['shortcut_agreement'])} | "
            f"[{fmt(row['shortcut_ci_low'])}, {fmt(row['shortcut_ci_high'])}] | "
            f"[{fmt(row['shortcut_group_ci_low'])}, {fmt(row['shortcut_group_ci_high'])}] | "
            f"{fmt(row['shortcut_perm_p'])} | {fmt(row['ontology_accuracy'])} | "
            f"{fmt(row['ontology_shortcut_agreement'])} | {fmt(row['ontology_answer_coverage'])}/{fmt(row['ontology_shortcut_coverage'])} |"
        )

    lines.extend(
        [
            "",
            "## Prior-Level Conflict Rows",
            "",
            "| prior | task | n | accuracy | 95% CI | shortcut agreement | 95% CI | shortcut perm p |",
            "|---|---|---:|---:|---|---:|---|---:|",
        ]
    )
    conflict_rows = [
        row
        for row in summary_rows
        if row["bucket"] == "conflict" and row["prior_name"] != "ALL"
    ]
    conflict_rows.sort(
        key=lambda row: (row["shortcut_agreement"], row["n"]), reverse=True
    )
    for row in conflict_rows[:40]:
        lines.append(
            f"| {row['prior_name']} | {row['task']} | {row['n']} | "
            f"{fmt(row['accuracy'])} | [{fmt(row['accuracy_ci_low'])}, {fmt(row['accuracy_ci_high'])}] | "
            f"{fmt(row['shortcut_agreement'])} | [{fmt(row['shortcut_ci_low'])}, {fmt(row['shortcut_ci_high'])}] | "
            f"{fmt(row['shortcut_perm_p'])} |"
        )

    report_path = output_dir / f"{label}_uncertainty.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
