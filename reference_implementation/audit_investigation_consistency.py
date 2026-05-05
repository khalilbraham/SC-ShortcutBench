#!/usr/bin/env python
"""Reliability checks for the shortcut-bias investigation artifacts.

This script does not rerun model inference. It checks that the generated
benchmark, predictions, uncertainty CSVs, and paper artifacts are mutually
consistent and that the headline conflict-row metrics are not artifacts of the
soft substring matcher.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark"
LARGE = BENCHMARK / "large_neurips_v1"
PAPER = ROOT / "paper_neurips2026"

CHALLENGES = {
    "Balanced": LARGE / "data" / "balanced_shortcut_challenge.jsonl",
    "Decorrelated": LARGE / "data" / "decorrelated_control_challenge.jsonl",
}

PREDICTIONS = {
    ("Balanced", "C2S-Scale"): LARGE / "predictions" / "c2s_pythia410m_diverse_pairwise_balanced_max300.jsonl",
    ("Balanced", "Cell2Text"): LARGE / "predictions" / "cell2text_llama32_1b_pairwise_balanced_max300.jsonl",
    ("Balanced", "CellWhisperer"): LARGE / "predictions" / "cellwhisperer_retrieval_balanced_max500.jsonl",
    ("Balanced", "scGPT"): LARGE / "predictions" / "scgpt_embedding_probe_balanced_max100.jsonl",
    ("Decorrelated", "C2S-Scale"): LARGE / "predictions" / "c2s_pythia410m_diverse_pairwise_decorrelated_max300.jsonl",
    ("Decorrelated", "Cell2Text"): LARGE / "predictions" / "cell2text_llama32_1b_pairwise_decorrelated_max300.jsonl",
    ("Decorrelated", "CellWhisperer"): LARGE / "predictions" / "cellwhisperer_retrieval_decorrelated_max300.jsonl",
    ("Decorrelated", "scGPT"): LARGE / "predictions" / "scgpt_embedding_probe_decorrelated_max100.jsonl",
}

UNCERTAINTY = {
    ("Balanced", "C2S-Scale"): LARGE / "reports" / "uncertainty" / "c2s_balanced_max300_uncertainty.csv",
    ("Balanced", "Cell2Text"): LARGE / "reports" / "uncertainty" / "cell2text_balanced_max300_uncertainty.csv",
    ("Balanced", "CellWhisperer"): LARGE / "reports" / "uncertainty" / "cellwhisperer_balanced_max500_uncertainty.csv",
    ("Balanced", "scGPT"): LARGE / "reports" / "uncertainty" / "scgpt_balanced_max100_uncertainty.csv",
    ("Decorrelated", "C2S-Scale"): LARGE / "reports" / "uncertainty" / "c2s_decorrelated_max300_uncertainty.csv",
    ("Decorrelated", "Cell2Text"): LARGE / "reports" / "uncertainty" / "cell2text_decorrelated_max300_uncertainty.csv",
    ("Decorrelated", "CellWhisperer"): LARGE / "reports" / "uncertainty" / "cellwhisperer_decorrelated_max300_uncertainty.csv",
    ("Decorrelated", "scGPT"): LARGE / "reports" / "uncertainty" / "scgpt_decorrelated_max100_uncertainty.csv",
}

TASK_LABELS = {
    "cell_type_prediction": "Cell type",
    "disease_prediction": "Disease",
    "tissue_general_prediction": "Tissue",
}


def norm(text) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def soft_match(prediction: str, answer: str) -> bool:
    pred = norm(prediction)
    ans = norm(answer)
    return pred == ans or bool(ans and ans in pred)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_uncertainty(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["bucket"] == "conflict" and row["prior_name"] == "ALL":
                rows[row["task"]] = row
    return rows


def rate(num: int, den: int) -> float:
    return num / den if den else float("nan")


def fmt(value: float) -> str:
    return f"{value:.3f}"


def audit_challenge(name: str, rows: list[dict]) -> tuple[list[str], list[str], list[str]]:
    failures = []
    warnings = []
    lines = []
    ids = [row.get("example_id", "") for row in rows]
    duplicate_ids = len(ids) - len(set(ids))
    required = {
        "example_id",
        "soma_joinid",
        "task",
        "answer",
        "shortcut_answer",
        "is_shortcut_conflict",
        "prior_name",
    }
    missing_required = sum(bool(required.difference(row)) for row in rows)
    conflict_rows = [row for row in rows if row.get("is_shortcut_conflict")]
    aligned_rows = [row for row in rows if not row.get("is_shortcut_conflict")]
    conflict_unique_soma = len({row.get("soma_joinid") for row in conflict_rows})
    conflict_norm_equal = sum(norm(row["answer"]) == norm(row["shortcut_answer"]) for row in conflict_rows)
    aligned_norm_different = sum(norm(row["answer"]) != norm(row["shortcut_answer"]) for row in aligned_rows)
    answer_in_shortcut = sum(
        norm(row["answer"]) != norm(row["shortcut_answer"])
        and bool(norm(row["answer"]))
        and norm(row["answer"]) in norm(row["shortcut_answer"])
        for row in conflict_rows
    )
    shortcut_in_answer = sum(
        norm(row["answer"]) != norm(row["shortcut_answer"])
        and bool(norm(row["shortcut_answer"]))
        and norm(row["shortcut_answer"]) in norm(row["answer"])
        for row in conflict_rows
    )
    task_counts = Counter(row.get("task", "") for row in rows)
    conflict_task_counts = Counter(row.get("task", "") for row in conflict_rows)

    if duplicate_ids:
        failures.append(f"{name}: duplicate example_id count is {duplicate_ids}")
    if missing_required:
        failures.append(f"{name}: {missing_required} rows miss required fields")
    if conflict_norm_equal:
        failures.append(f"{name}: {conflict_norm_equal} conflict rows have equal normalized labels")
    if aligned_norm_different:
        failures.append(f"{name}: {aligned_norm_different} aligned rows have different normalized labels")
    if answer_in_shortcut or shortcut_in_answer:
        warnings.append(
            f"{name}: {answer_in_shortcut + shortcut_in_answer} conflict rows have substring-related label pairs; exact-vs-soft metric check is required"
        )

    lines.extend(
        [
            f"### {name} Challenge",
            "",
            f"- Rows: {len(rows):,}",
            f"- Conflict rows: {len(conflict_rows):,}",
            f"- Unique conflict `soma_joinid`: {conflict_unique_soma:,}",
            f"- Aligned rows: {len(aligned_rows):,}",
            f"- Duplicate example IDs: {duplicate_ids:,}",
            f"- Missing required-field rows: {missing_required:,}",
            f"- Conflict normalized-label collisions: {conflict_norm_equal:,}",
            f"- Aligned normalized-label mismatches: {aligned_norm_different:,}",
            f"- Conflict substring-ambiguous label pairs: {answer_in_shortcut + shortcut_in_answer:,}",
            f"- Task counts: {dict(task_counts)}",
            f"- Conflict task counts: {dict(conflict_task_counts)}",
            "",
        ]
    )
    return failures, warnings, lines


def audit_predictions(
    challenge_name: str,
    model: str,
    challenge_rows: dict[str, dict],
) -> tuple[list[str], list[str], list[str]]:
    failures = []
    warnings = []
    lines = []
    pred_path = PREDICTIONS[(challenge_name, model)]
    rows = load_jsonl(pred_path)
    uncertainty = load_uncertainty(UNCERTAINTY[(challenge_name, model)])
    duplicate_predictions = len(rows) - len({row.get("example_id", "") for row in rows})
    if duplicate_predictions:
        failures.append(f"{challenge_name}/{model}: duplicate prediction example IDs: {duplicate_predictions}")

    stats = defaultdict(lambda: Counter())
    task_soma_ids: dict[str, set[str]] = defaultdict(set)
    ignored = 0
    for pred_row in rows:
        row = challenge_rows.get(pred_row.get("example_id"))
        if row is None:
            ignored += 1
            continue
        if not row.get("is_shortcut_conflict"):
            continue
        task = row["task"]
        task_soma_ids[task].add(str(row.get("soma_joinid", "")))
        prediction = pred_row.get("prediction", "")
        pred_norm = norm(prediction)
        answer_norm = norm(row["answer"])
        shortcut_norm = norm(row["shortcut_answer"])
        stats[task]["n"] += 1
        stats[task]["soft_acc"] += int(soft_match(prediction, row["answer"]))
        stats[task]["soft_short"] += int(soft_match(prediction, row["shortcut_answer"]))
        stats[task]["exact_acc"] += int(pred_norm == answer_norm)
        stats[task]["exact_short"] += int(pred_norm == shortcut_norm)
        stats[task]["both_soft"] += int(
            soft_match(prediction, row["answer"]) and soft_match(prediction, row["shortcut_answer"])
        )
        stats[task]["soft_exact_diff"] += int(
            soft_match(prediction, row["answer"]) != (pred_norm == answer_norm)
            or soft_match(prediction, row["shortcut_answer"]) != (pred_norm == shortcut_norm)
        )
        stats[task]["other_prediction"] += int(pred_norm not in {answer_norm, shortcut_norm})

    if ignored:
        failures.append(f"{challenge_name}/{model}: ignored predictions not present in challenge: {ignored}")

    lines.extend(
        [
            f"### {challenge_name} / {model}",
            "",
            "| task | n (unique cells) | exact acc | exact shortcut | soft acc | soft shortcut | both soft | soft/exact diff | other predictions | report match |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for task in sorted(stats):
        item = stats[task]
        n = item["n"]
        soft_acc = rate(item["soft_acc"], n)
        soft_short = rate(item["soft_short"], n)
        exact_acc = rate(item["exact_acc"], n)
        exact_short = rate(item["exact_short"], n)
        report = uncertainty.get(task)
        report_match = "yes"
        if report is None:
            failures.append(f"{challenge_name}/{model}/{task}: missing uncertainty row")
            report_match = "missing"
        else:
            report_n = int(report["n"])
            report_acc = float(report["accuracy"])
            report_short = float(report["shortcut_agreement"])
            if report_n != n or abs(report_acc - soft_acc) > 1e-12 or abs(report_short - soft_short) > 1e-12:
                failures.append(
                    f"{challenge_name}/{model}/{task}: uncertainty CSV mismatch "
                    f"(report n/acc/short={report_n}/{report_acc:.6f}/{report_short:.6f}; "
                    f"recomputed={n}/{soft_acc:.6f}/{soft_short:.6f})"
                )
                report_match = "no"
        if item["both_soft"]:
            warnings.append(f"{challenge_name}/{model}/{task}: {item['both_soft']} conflict predictions match both answer and shortcut under soft matching")
        if item["soft_exact_diff"]:
            warnings.append(f"{challenge_name}/{model}/{task}: {item['soft_exact_diff']} rows differ between soft and exact matching")
        if item["other_prediction"]:
            warnings.append(f"{challenge_name}/{model}/{task}: {item['other_prediction']} predictions are neither candidate label exactly")

        lines.append(
            "| "
            + " | ".join(
                [
                    task,
                    f"{n} ({len(task_soma_ids[task])} cells)",
                    fmt(exact_acc),
                    fmt(exact_short),
                    fmt(soft_acc),
                    fmt(soft_short),
                    str(item["both_soft"]),
                    str(item["soft_exact_diff"]),
                    str(item["other_prediction"]),
                    report_match,
                ]
            )
            + " |"
        )
    lines.append("")
    return failures, warnings, lines


def audit_baseline(challenge_name: str, challenge_rows: dict[str, dict]) -> tuple[list[str], list[str], list[str]]:
    failures = []
    warnings = []
    lines = []
    path = LARGE / "predictions" / f"metadata_prior_baseline_{challenge_name.lower()}.jsonl"
    if not path.exists():
        failures.append(f"{challenge_name}: missing metadata prior baseline {path}")
        return failures, warnings, []
    preds = {row["example_id"]: row.get("prediction", "") for row in load_jsonl(path)}
    conflict = [row for row in challenge_rows.values() if row.get("is_shortcut_conflict")]
    matched = [row for row in conflict if row["example_id"] in preds]
    exact_truth = sum(norm(preds[row["example_id"]]) == norm(row["answer"]) for row in matched)
    exact_short = sum(norm(preds[row["example_id"]]) == norm(row["shortcut_answer"]) for row in matched)
    if len(matched) != len(conflict):
        failures.append(f"{challenge_name}: metadata prior baseline covers {len(matched)} / {len(conflict)} conflict rows")
    if exact_truth != 0:
        failures.append(f"{challenge_name}: metadata prior baseline truth matches {exact_truth} conflict rows")
    if exact_short != len(matched):
        failures.append(f"{challenge_name}: metadata prior baseline shortcut matches {exact_short} / {len(matched)} conflict rows")
    lines.extend(
        [
            f"### {challenge_name} Metadata-Prior Baseline",
            "",
            f"- Conflict rows covered: {len(matched):,} / {len(conflict):,}",
            f"- Exact truth matches on conflict rows: {exact_truth:,}",
            f"- Exact shortcut matches on conflict rows: {exact_short:,}",
            "",
        ]
    )
    return failures, warnings, lines


def audit_paper_artifacts() -> tuple[list[str], list[str], list[str]]:
    failures = []
    warnings = []
    lines = ["### Paper Artifacts", ""]
    main = PAPER / "main.tex"
    text = main.read_text()
    figure_refs = re.findall(r"figures/([^}]+)", text)
    missing_figures = [name for name in figure_refs if not (PAPER / "figures" / name).exists()]
    if missing_figures:
        failures.append(f"Paper missing figure files: {missing_figures}")
    caveats = {
        "scMMGPT no quantitative claim": "not a quantitative scMMGPT failure claim" in text,
        "not all predictions shortcuts": "They do not show that every model collapses to a shortcut baseline" in text,
        "not invalid biology": "does not deny these facts" in text,
        "risk channels not failures": "source-risk audit rather than a complete quantitative reproduction" in text,
    }
    for label, ok in caveats.items():
        if not ok:
            warnings.append(f"Missing caveat in paper: {label}")

    metric_rows = {}
    generated_metrics = PAPER / "generated" / "model_shortcut_reliance_metrics.csv"
    with generated_metrics.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric_rows[(row["model"], row["challenge"], row["task_label"])] = (
                float(row["accuracy"]),
                float(row["shortcut_agreement"]),
            )

    paper_table_rows = 0
    model_pattern = re.compile(
        r"^(CellWhisperer|C2S-Scale|scGPT|Cell2Text) & "
        r"(Balanced|Decorrelated) & "
        r"([0-9.]+) / ([0-9.]+) & "
        r"([0-9.]+) / ([0-9.]+) & "
        r"([0-9.]+) / ([0-9.]+) \\\\",
        re.MULTILINE,
    )
    for match in model_pattern.finditer(text):
        paper_table_rows += 1
        model, challenge = match.group(1), match.group(2)
        values = {
            "Cell type": (float(match.group(3)), float(match.group(4))),
            "Disease": (float(match.group(5)), float(match.group(6))),
            "Tissue": (float(match.group(7)), float(match.group(8))),
        }
        for task_label, paper_values in values.items():
            generated_values = metric_rows.get((model, challenge, task_label))
            if generated_values is None:
                failures.append(f"Paper table row missing generated metric for {model}/{challenge}/{task_label}")
                continue
            for paper_value, generated_value in zip(paper_values, generated_values):
                if abs(paper_value - round(generated_value, 3)) > 5e-4:
                    failures.append(
                        f"Paper table mismatch for {model}/{challenge}/{task_label}: "
                        f"paper={paper_values}, generated=({generated_values[0]:.3f}, {generated_values[1]:.3f})"
                    )
    if paper_table_rows != 8:
        failures.append(f"Expected 8 model-result rows in paper table, found {paper_table_rows}")

    lines.extend(
        [
            f"- Includegraphics references: {len(figure_refs):,}",
            f"- Missing figures: {missing_figures}",
            f"- Caveat checks: {caveats}",
            f"- Model table rows checked against generated CSV: {paper_table_rows}",
            "",
        ]
    )
    return failures, warnings, lines


def main() -> None:
    report_lines = [
        "# Reliability Audit",
        "",
        "This audit checks internal consistency of the shortcut benchmark, model predictions, uncertainty CSVs, and manuscript artifacts.",
        "",
    ]
    all_failures: list[str] = []
    all_warnings: list[str] = []
    challenge_maps = {}

    report_lines.append("## Challenge Integrity")
    report_lines.append("")
    for name, path in CHALLENGES.items():
        rows = load_jsonl(path)
        challenge_maps[name] = {row["example_id"]: row for row in rows}
        failures, warnings, lines = audit_challenge(name, rows)
        all_failures.extend(failures)
        all_warnings.extend(warnings)
        report_lines.extend(lines)

    report_lines.append("## Negative Controls")
    report_lines.append("")
    for name, challenge in challenge_maps.items():
        failures, warnings, lines = audit_baseline(name, challenge)
        all_failures.extend(failures)
        all_warnings.extend(warnings)
        report_lines.extend(lines)

    report_lines.append("## Prediction And Metric Consistency")
    report_lines.append("")
    for challenge_name, model in sorted(PREDICTIONS):
        failures, warnings, lines = audit_predictions(challenge_name, model, challenge_maps[challenge_name])
        all_failures.extend(failures)
        all_warnings.extend(warnings)
        report_lines.extend(lines)

    report_lines.append("## Manuscript Checks")
    report_lines.append("")
    failures, warnings, lines = audit_paper_artifacts()
    all_failures.extend(failures)
    all_warnings.extend(warnings)
    report_lines.extend(lines)

    report_lines.extend(
        [
            "## Summary",
            "",
            f"- Failures: {len(all_failures)}",
            f"- Warnings: {len(all_warnings)}",
            "",
        ]
    )
    if all_failures:
        report_lines.extend(["### Failures", ""])
        report_lines.extend(f"- {item}" for item in all_failures)
        report_lines.append("")
    if all_warnings:
        report_lines.extend(["### Warnings", ""])
        report_lines.extend(f"- {item}" for item in all_warnings)
        report_lines.append("")
    if not all_failures:
        report_lines.extend(
            [
                "No blocking consistency failures were found. Warnings are caveats to report transparently, not evidence that the headline conflict-row numbers are invalid.",
                "",
            ]
        )

    output = LARGE / "reports" / "reliability_audit.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report_lines))
    print(f"Wrote {output}")
    if all_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
