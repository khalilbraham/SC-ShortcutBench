#!/usr/bin/env python
"""Evaluate model predictions on the shortcut challenge JSONL."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", required=True)
    parser.add_argument(
        "--predictions",
        required=True,
        help="JSONL with example_id and prediction fields.",
    )
    parser.add_argument("--output", required=True)
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


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    challenge = {row["example_id"]: row for row in load_jsonl(args.challenge)}
    predictions = load_jsonl(args.predictions)

    totals: Counter = Counter()
    correct: Counter = Counter()
    shortcut_agree: Counter = Counter()
    prior_totals: Counter = Counter()
    prior_correct: Counter = Counter()
    prior_shortcut_agree: Counter = Counter()
    missing = 0
    matched = 0
    examples: dict[str, list[dict]] = defaultdict(list)

    for pred_row in predictions:
        example_id = pred_row.get("example_id")
        if example_id not in challenge:
            missing += 1
            continue
        matched += 1
        row = challenge[example_id]
        task = row["task"]
        bucket = "conflict" if row["is_shortcut_conflict"] else "aligned"
        key = (task, bucket)
        prior_name = row.get("prior_name")
        prediction = pred_row.get("prediction", "")
        totals[key] += 1
        totals[(task, "all")] += 1
        if label_match(prediction, row["answer"]):
            correct[key] += 1
            correct[(task, "all")] += 1
        if label_match(prediction, row["shortcut_answer"]):
            shortcut_agree[key] += 1
            shortcut_agree[(task, "all")] += 1
        if prior_name:
            for prior_key in ((prior_name, task, bucket), (prior_name, task, "all")):
                prior_totals[prior_key] += 1
                if label_match(prediction, row["answer"]):
                    prior_correct[prior_key] += 1
                if label_match(prediction, row["shortcut_answer"]):
                    prior_shortcut_agree[prior_key] += 1
        if len(examples[task]) < 5:
            examples[task].append(
                {
                    "example_id": example_id,
                    "prediction": prediction,
                    "answer": row["answer"],
                    "shortcut_answer": row["shortcut_answer"],
                    "is_shortcut_conflict": row["is_shortcut_conflict"],
                }
            )

    lines = [
        "# Shortcut Challenge Evaluation",
        "",
        f"- Challenge rows: {len(challenge):,}",
        f"- Prediction rows matched: {matched:,}",
        f"- Prediction rows ignored: {missing:,}",
        "",
        "| task | bucket | n | accuracy | shortcut_agreement |",
        "|---|---|---:|---:|---:|",
    ]
    for key in sorted(totals):
        task, bucket = key
        n = totals[key]
        lines.append(
            f"| {task} | {bucket} | {n} | {correct[key] / n:.3f} | {shortcut_agree[key] / n:.3f} |"
        )

    if prior_totals:
        lines.extend(
            [
                "",
                "## By Prior",
                "",
                "| prior | task | bucket | n | accuracy | shortcut_agreement |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for key in sorted(prior_totals):
            prior_name, task, bucket = key
            n = prior_totals[key]
            lines.append(
                f"| {prior_name} | {task} | {bucket} | {n} | "
                f"{prior_correct[key] / n:.3f} | "
                f"{prior_shortcut_agree[key] / n:.3f} |"
            )

    lines.extend(["", "## Example Predictions", ""])
    for task, task_examples in examples.items():
        lines.extend([f"### {task}", ""])
        for example in task_examples:
            lines.append(
                "- "
                + json.dumps(example, ensure_ascii=False)
            )
        lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
