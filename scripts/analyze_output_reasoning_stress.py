#!/usr/bin/env python3
"""Output-level stress tests for shortcut-vs-reasoning claims.

These diagnostics are intentionally model-output-facing.  They complement the
metadata-prior audit by asking whether the predictions themselves behave like
stable expression-grounded decisions, or whether they move with shortcut labels,
high-purity priors, label artifacts, and choice-set changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_RUNS = [
    (
        "CellWhisperer balanced",
        "data/balanced_shortcut_challenge.jsonl",
        "predictions/cellwhisperer_retrieval_balanced_max500.jsonl",
    ),
    (
        "CellWhisperer decorrelated",
        "data/decorrelated_control_challenge.jsonl",
        "predictions/cellwhisperer_retrieval_decorrelated_max300.jsonl",
    ),
    (
        "C2S balanced",
        "data/balanced_shortcut_challenge.jsonl",
        "predictions/c2s_pythia410m_diverse_pairwise_balanced_max300.jsonl",
    ),
    (
        "C2S decorrelated",
        "data/decorrelated_control_challenge.jsonl",
        "predictions/c2s_pythia410m_diverse_pairwise_decorrelated_max300.jsonl",
    ),
    (
        "Cell2Text balanced",
        "data/balanced_shortcut_challenge.jsonl",
        "predictions/cell2text_llama32_1b_pairwise_balanced_max300.jsonl",
    ),
    (
        "Cell2Text decorrelated",
        "data/decorrelated_control_challenge.jsonl",
        "predictions/cell2text_llama32_1b_pairwise_decorrelated_max300.jsonl",
    ),
    (
        "scGPT balanced",
        "data/balanced_shortcut_challenge.jsonl",
        "predictions/scgpt_embedding_probe_balanced_max100.jsonl",
    ),
    (
        "scGPT decorrelated",
        "data/decorrelated_control_challenge.jsonl",
        "predictions/scgpt_embedding_probe_decorrelated_max100.jsonl",
    ),
]


def normalize(text: object) -> str:
    value = "" if text is None else str(text).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    vals = [v for v in values if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def rate(values: list[int]) -> float:
    return sum(values) / len(values) if values else float("nan")


def fmt(value: float) -> str:
    return "NA" if value != value else f"{value:.3f}"


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pairs) < 3:
        return float("nan")
    xvals = [x for x, _ in pairs]
    yvals = [y for _, y in pairs]
    xm = mean(xvals)
    ym = mean(yvals)
    xss = sum((x - xm) ** 2 for x in xvals)
    yss = sum((y - ym) ** 2 for y in yvals)
    if xss <= 0.0 or yss <= 0.0:
        return float("nan")
    return sum((x - xm) * (y - ym) for x, y in pairs) / math.sqrt(xss * yss)


def score_pair(pred: dict) -> tuple[float, float]:
    for answer_key, shortcut_key in [
        ("score_answer", "score_shortcut"),
        ("answer_score", "shortcut_score"),
        ("score_answer_centroid", "score_shortcut_centroid"),
    ]:
        answer = pred.get(answer_key)
        shortcut = pred.get(shortcut_key)
        if answer is not None and shortcut is not None:
            return float(answer), float(shortcut)
    truth = shortcut = float("nan")
    for candidate in pred.get("candidates", []):
        if candidate.get("role") == "truth":
            truth = float(candidate["score"])
        if candidate.get("role") == "shortcut":
            shortcut = float(candidate["score"])
    return truth, shortcut


def token_len(text: object) -> int:
    value = normalize(text)
    return len(value.split()) if value else 0


def dist(rows: list[dict], key: str) -> dict[str, float]:
    counts = Counter(normalize(row[key]) for row in rows if normalize(row[key]))
    total = sum(counts.values())
    if not total:
        return {}
    return {label: count / total for label, count in counts.items()}


def entropy(prob: dict[str, float]) -> float:
    return -sum(p * math.log(p) for p in prob.values() if p > 0)


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    labels = set(p) | set(q)
    if not labels:
        return float("nan")
    m = {label: 0.5 * p.get(label, 0.0) + 0.5 * q.get(label, 0.0) for label in labels}

    def kl(a: dict[str, float], b: dict[str, float]) -> float:
        out = 0.0
        for label in labels:
            av = a.get(label, 0.0)
            bv = b.get(label, 0.0)
            if av > 0 and bv > 0:
                out += av * math.log(av / bv)
        return out

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def enrich(challenge_path: Path, predictions_path: Path) -> list[dict]:
    challenge = {row["example_id"]: row for row in load_jsonl(challenge_path)}
    rows = []
    for pred in load_jsonl(predictions_path):
        source = challenge.get(pred.get("example_id"))
        if not source:
            continue
        truth_score, shortcut_score = score_pair(pred)
        prediction = pred.get("prediction", "")
        answer = source.get("answer", "")
        shortcut = source.get("shortcut_answer", "")
        metadata = source.get("metadata", {})
        margin = (
            shortcut_score - truth_score
            if truth_score == truth_score and shortcut_score == shortcut_score
            else float("nan")
        )
        rows.append(
            {
                "example_id": source["example_id"],
                "soma_joinid": str(source["soma_joinid"]),
                "task": source["task"],
                "prior_name": source.get("prior_name", ""),
                "prior_family": source.get("prior_family", ""),
                "answer": answer,
                "shortcut_answer": shortcut,
                "prediction": prediction,
                "answer_norm": normalize(answer),
                "shortcut_norm": normalize(shortcut),
                "prediction_norm": normalize(prediction),
                "is_conflict": bool(source.get("is_shortcut_conflict")),
                "prior_purity": float(source.get("prior_purity", float("nan"))),
                "prior_support": int(source.get("prior_support", 0)),
                "truth_score": truth_score,
                "shortcut_score": shortcut_score,
                "shortcut_minus_truth": margin,
                "truth_match": int(normalize(prediction) == normalize(answer)),
                "shortcut_match": int(normalize(prediction) == normalize(shortcut)),
                "answer_chars": len(str(answer)),
                "shortcut_chars": len(str(shortcut)),
                "answer_tokens": token_len(answer),
                "shortcut_tokens": token_len(shortcut),
                "shortcut_minus_answer_chars": len(str(shortcut)) - len(str(answer)),
                "shortcut_minus_answer_tokens": token_len(shortcut) - token_len(answer),
                "cell_type": metadata.get("cell_type", ""),
                "tissue_general": metadata.get("tissue_general", ""),
                "disease": metadata.get("disease", ""),
                "assay": metadata.get("assay", ""),
                "development_stage": metadata.get("development_stage", ""),
                "sex": metadata.get("sex", ""),
            }
        )
    return rows


def summarize_subset(name: str, rows: list[dict]) -> dict:
    conflict = [row for row in rows if row["is_conflict"]]
    scored = [row for row in conflict if row["shortcut_minus_truth"] == row["shortcut_minus_truth"]]
    equal_tokens = [row for row in scored if row["shortcut_minus_answer_tokens"] == 0]
    near_equal_chars = [row for row in scored if abs(row["shortcut_minus_answer_chars"]) <= 2]
    high_purity = [row for row in conflict if row["prior_purity"] >= 0.9]
    high_purity_scored = [
        row
        for row in high_purity
        if row["shortcut_minus_truth"] == row["shortcut_minus_truth"]
    ]
    pred_d = dist(conflict, "prediction")
    truth_d = dist(conflict, "answer")
    shortcut_d = dist(conflict, "shortcut_answer")
    js_pred_truth = js_divergence(pred_d, truth_d)
    js_pred_shortcut = js_divergence(pred_d, shortcut_d)
    return {
        "group": name,
        "n_conflict": len(conflict),
        "truth_accuracy": rate([row["truth_match"] for row in conflict]),
        "shortcut_agreement": rate([row["shortcut_match"] for row in conflict]),
        "mean_shortcut_minus_truth_score": mean(
            [row["shortcut_minus_truth"] for row in scored]
        ),
        "shortcut_score_preferred": rate(
            [int(row["shortcut_minus_truth"] > 0) for row in scored]
        ),
        "prior_margin_corr": pearson(
            [row["prior_purity"] for row in scored],
            [row["shortcut_minus_truth"] for row in scored],
        ),
        "support_margin_corr": pearson(
            [math.log1p(row["prior_support"]) for row in scored],
            [row["shortcut_minus_truth"] for row in scored],
        ),
        "length_margin_corr_chars": pearson(
            [row["shortcut_minus_answer_chars"] for row in scored],
            [row["shortcut_minus_truth"] for row in scored],
        ),
        "length_margin_corr_tokens": pearson(
            [row["shortcut_minus_answer_tokens"] for row in scored],
            [row["shortcut_minus_truth"] for row in scored],
        ),
        "equal_token_n": len(equal_tokens),
        "equal_token_shortcut_preferred": rate(
            [int(row["shortcut_minus_truth"] > 0) for row in equal_tokens]
        ),
        "near_equal_char_n": len(near_equal_chars),
        "near_equal_char_shortcut_preferred": rate(
            [int(row["shortcut_minus_truth"] > 0) for row in near_equal_chars]
        ),
        "high_purity_n": len(high_purity),
        "high_purity_shortcut_agreement": rate(
            [row["shortcut_match"] for row in high_purity]
        ),
        "high_purity_shortcut_preferred": rate(
            [int(row["shortcut_minus_truth"] > 0) for row in high_purity_scored]
        ),
        "prediction_entropy": entropy(pred_d),
        "truth_entropy": entropy(truth_d),
        "shortcut_entropy": entropy(shortcut_d),
        "js_prediction_truth": js_pred_truth,
        "js_prediction_shortcut": js_pred_shortcut,
        "output_prior_alignment": (
            js_pred_truth - js_pred_shortcut
            if js_pred_truth == js_pred_truth and js_pred_shortcut == js_pred_shortcut
            else float("nan")
        ),
    }


def same_cell_stress(rows: list[dict]) -> tuple[dict, list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["is_conflict"]:
            groups[(row["soma_joinid"], row["task"])].append(row)
    repeated = [group for group in groups.values() if len(group) >= 2]
    multi_shortcut = [
        group
        for group in repeated
        if len({row["shortcut_norm"] for row in group}) >= 2
    ]
    direct_tracking = []
    for group in multi_shortcut:
        matched_shortcut_predictions = {
            row["prediction_norm"]
            for row in group
            if row["shortcut_match"] and row["prediction_norm"]
        }
        if len(matched_shortcut_predictions) >= 2:
            direct_tracking.append(group)
    repeated_rows = [row for group in repeated for row in group]
    multi_rows = [row for group in multi_shortcut for row in group]
    direct_rows = [row for group in direct_tracking for row in group]
    summary = {
        "repeated_groups": len(repeated),
        "repeated_rows": len(repeated_rows),
        "repeated_truth_accuracy": rate([row["truth_match"] for row in repeated_rows]),
        "repeated_shortcut_agreement": rate(
            [row["shortcut_match"] for row in repeated_rows]
        ),
        "multi_shortcut_groups": len(multi_shortcut),
        "multi_shortcut_rows": len(multi_rows),
        "multi_shortcut_truth_accuracy": rate([row["truth_match"] for row in multi_rows]),
        "multi_shortcut_agreement": rate([row["shortcut_match"] for row in multi_rows]),
        "direct_tracking_groups": len(direct_tracking),
        "direct_tracking_rows": len(direct_rows),
        "direct_tracking_row_rate_among_multi": (
            len(direct_rows) / len(multi_rows) if multi_rows else float("nan")
        ),
        "all_truth_group_rate": rate(
            [int(all(row["truth_match"] for row in group)) for group in repeated]
        ),
        "any_shortcut_group_rate": rate(
            [int(any(row["shortcut_match"] for row in group)) for group in repeated]
        ),
    }
    examples = []
    ranked = sorted(
        repeated,
        key=lambda group: (
            rate([row["shortcut_match"] for row in group]),
            len({row["shortcut_norm"] for row in group}),
            len(group),
        ),
        reverse=True,
    )
    for group in ranked[:6]:
        first = group[0]
        examples.append(
            {
                "soma_joinid": first["soma_joinid"],
                "task": first["task"],
                "answer": first["answer"],
                "cell_type": first["cell_type"],
                "tissue_general": first["tissue_general"],
                "disease": first["disease"],
                "assay": first["assay"],
                "rows": [
                    {
                        "prior_name": row["prior_name"],
                        "prior_purity": row["prior_purity"],
                        "shortcut": row["shortcut_answer"],
                        "prediction": row["prediction"],
                        "shortcut_minus_truth": row["shortcut_minus_truth"],
                    }
                    for row in group
                ],
            }
        )
    return summary, examples


def high_margin_examples(rows: list[dict], n: int = 8) -> list[dict]:
    candidates = [
        row
        for row in rows
        if row["is_conflict"]
        and row["shortcut_match"]
        and row["shortcut_minus_truth"] == row["shortcut_minus_truth"]
    ]
    candidates.sort(
        key=lambda row: (
            row["shortcut_minus_truth"],
            row["prior_purity"],
            row["prior_support"],
        ),
        reverse=True,
    )
    return [
        {
            "example_id": row["example_id"],
            "task": row["task"],
            "prior_name": row["prior_name"],
            "prior_purity": row["prior_purity"],
            "answer": row["answer"],
            "shortcut": row["shortcut_answer"],
            "prediction": row["prediction"],
            "shortcut_minus_truth": row["shortcut_minus_truth"],
            "cell_type": row["cell_type"],
            "tissue_general": row["tissue_general"],
            "disease": row["disease"],
            "assay": row["assay"],
        }
        for row in candidates[:n]
    ]


def top_labels(rows: list[dict], key: str, n: int = 5) -> str:
    conflict = [row for row in rows if row["is_conflict"]]
    counts = Counter(normalize(row[key]) for row in conflict if normalize(row[key]))
    total = sum(counts.values())
    if not total:
        return "NA"
    return "; ".join(
        f"{label} ({count / total:.2f})" for label, count in counts.most_common(n)
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("large_neurips_v1"),
        help="Directory containing data/, predictions/, and reports/.",
    )
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--task-csv", type=Path)
    parser.add_argument("--examples-json", type=Path)
    args = parser.parse_args()

    base = args.benchmark_dir
    output_md = args.output_md or base / "reports" / "output_reasoning_stress_report.md"
    output_csv = args.output_csv or base / "reports" / "output_reasoning_stress_summary.csv"
    task_csv = args.task_csv or base / "reports" / "output_reasoning_stress_by_task.csv"
    examples_json = (
        args.examples_json
        or base / "reports" / "output_reasoning_stress_examples.json"
    )

    all_summaries = []
    task_summaries = []
    example_payload = {}
    per_run_rows: dict[str, list[dict]] = {}
    for label, challenge_rel, pred_rel in DEFAULT_RUNS:
        challenge_path = base / challenge_rel
        pred_path = base / pred_rel
        if not challenge_path.exists() or not pred_path.exists():
            continue
        rows = enrich(challenge_path, pred_path)
        per_run_rows[label] = rows
        summary = summarize_subset(label, rows)
        stress, examples = same_cell_stress(rows)
        summary.update(stress)
        summary["top_predictions"] = top_labels(rows, "prediction")
        summary["top_truths"] = top_labels(rows, "answer")
        summary["top_shortcuts"] = top_labels(rows, "shortcut_answer")
        all_summaries.append(summary)
        for task in sorted({row["task"] for row in rows}):
            task_rows = [row for row in rows if row["task"] == task]
            task_summary = summarize_subset(label, task_rows)
            task_summary["run"] = label
            task_summary["task"] = task
            task_summaries.append(task_summary)
        example_payload[label] = {
            "same_cell_examples": examples,
            "high_margin_shortcut_examples": high_margin_examples(rows),
        }

    write_csv(output_csv, all_summaries)
    write_csv(task_csv, task_summaries)
    examples_json.parent.mkdir(parents=True, exist_ok=True)
    examples_json.write_text(json.dumps(example_payload, indent=2) + "\n")

    lines = [
        "# Output-Level Reasoning Stress Report",
        "",
        "This report asks whether model decisions look stable under expression-grounded reasoning, or whether they show output-level symptoms of shortcut use.  It should be read as diagnostic evidence, not as proof that a model has no biological knowledge.",
        "",
        "## Main Stress Metrics",
        "",
        "| run | n conflict | truth acc. | shortcut agree | shortcut score pref. | margin | prior corr | support corr | equal-token n | equal-token shortcut pref. | high-purity n | high-purity shortcut agree | OPDA | repeated groups | repeated shortcut agree | direct tracking groups |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["group"],
                    str(row["n_conflict"]),
                    fmt(row["truth_accuracy"]),
                    fmt(row["shortcut_agreement"]),
                    fmt(row["shortcut_score_preferred"]),
                    fmt(row["mean_shortcut_minus_truth_score"]),
                    fmt(row["prior_margin_corr"]),
                    fmt(row["support_margin_corr"]),
                    str(row["equal_token_n"]),
                    fmt(row["equal_token_shortcut_preferred"]),
                    str(row["high_purity_n"]),
                    fmt(row["high_purity_shortcut_agreement"]),
                    fmt(row["output_prior_alignment"]),
                    str(row["repeated_groups"]),
                    fmt(row["repeated_shortcut_agreement"]),
                    str(row["direct_tracking_groups"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "OPDA is `JS(prediction, truth) - JS(prediction, shortcut)`. Positive values mean the prediction distribution is closer to the shortcut-label distribution than to the truth-label distribution.",
            "",
            "## Task-Level Stress",
            "",
            "| run | task | n conflict | truth acc. | shortcut agree | shortcut score pref. | margin | prior corr | equal-token shortcut pref. | high-purity shortcut agree | length corr tokens |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in task_summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["run"],
                    row["task"],
                    str(row["n_conflict"]),
                    fmt(row["truth_accuracy"]),
                    fmt(row["shortcut_agreement"]),
                    fmt(row["shortcut_score_preferred"]),
                    fmt(row["mean_shortcut_minus_truth_score"]),
                    fmt(row["prior_margin_corr"]),
                    fmt(row["equal_token_shortcut_preferred"]),
                    fmt(row["high_purity_shortcut_agreement"]),
                    fmt(row["length_margin_corr_tokens"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Label Artifact Diagnostics",
            "",
            "| run | length corr chars | length corr tokens | near-equal-char n | near-equal-char shortcut pref. | pred entropy | truth entropy | shortcut entropy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in all_summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["group"],
                    fmt(row["length_margin_corr_chars"]),
                    fmt(row["length_margin_corr_tokens"]),
                    str(row["near_equal_char_n"]),
                    fmt(row["near_equal_char_shortcut_preferred"]),
                    fmt(row["prediction_entropy"]),
                    fmt(row["truth_entropy"]),
                    fmt(row["shortcut_entropy"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Distributional Output Modes",
            "",
            "| run | top predictions on conflicts | top truths | top shortcuts |",
            "|---|---|---|---|",
        ]
    )
    for row in all_summaries:
        lines.append(
            f"| {row['group']} | {row['top_predictions']} | {row['top_truths']} | {row['top_shortcuts']} |"
        )

    lines.extend(
        [
            "",
            "## Same-Cell Counterfactual Choice-Set Stress",
            "",
            "For the same `soma_joinid` and same task, the expression profile and true answer are fixed.  If multiple shortcut priors produce different shortcut answers and the model follows those changing shortcuts, the output is not invariant to the biological input.",
            "",
            "| run | repeated rows | repeated truth acc. | repeated shortcut agree | multi-shortcut groups | multi-shortcut rows | multi-shortcut truth acc. | multi-shortcut shortcut agree | all-truth group rate | any-shortcut group rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in all_summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["group"],
                    str(row["repeated_rows"]),
                    fmt(row["repeated_truth_accuracy"]),
                    fmt(row["repeated_shortcut_agreement"]),
                    str(row["multi_shortcut_groups"]),
                    str(row["multi_shortcut_rows"]),
                    fmt(row["multi_shortcut_truth_accuracy"]),
                    fmt(row["multi_shortcut_agreement"]),
                    fmt(row["all_truth_group_rate"]),
                    fmt(row["any_shortcut_group_rate"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## High-Margin Shortcut Examples",
            "",
            "The JSON sidecar stores concrete examples for manual paper inspection:",
            f"- `{examples_json}`",
            "",
            "## Interpretation",
            "",
            "- The output evidence is localized rather than a global collapse. OPDA is negative in the current runs, meaning the aggregate prediction distribution is usually closer to the truth-label distribution than to the shortcut-label distribution.",
            "- Same-cell repeated rows are useful but not yet the decisive counterfactual test, because most repeated rows expose the same shortcut label through multiple priors. They still show whether the same biological input repeatedly loses to a construction answer.",
            "- Length correlations reveal text-evaluation artifacts.  These should be reported transparently; if shortcut preference persists in equal-token or near-equal-character subsets, it is harder to dismiss as only label-length bias.",
            "- High-purity shortcut failures are the cleanest causal stress cases: the train prior is strong, the held-out truth contradicts it, and the model often follows the prior label.",
        ]
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_md}")
    print(f"Wrote {output_csv}")
    print(f"Wrote {task_csv}")
    print(f"Wrote {examples_json}")


if __name__ == "__main__":
    main()
