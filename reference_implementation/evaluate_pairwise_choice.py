#!/usr/bin/env python
"""Evaluate pairwise true-vs-shortcut diagnostic predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def infer_choice(prediction: dict) -> str:
    if prediction.get("chosen_candidate") in {"answer", "shortcut_answer"}:
        return prediction["chosen_candidate"]
    answer_score = prediction.get("answer_score")
    shortcut_score = prediction.get("shortcut_score")
    if answer_score is None or shortcut_score is None:
        raise ValueError(
            f"Prediction for {prediction.get('example_id')} lacks chosen_candidate or scores."
        )
    return "answer" if float(answer_score) >= float(shortcut_score) else "shortcut_answer"


def main() -> None:
    args = parse_args()
    challenge = {row["example_id"]: row for row in load_jsonl(args.challenge)}
    predictions = load_jsonl(args.predictions)

    totals: Counter = Counter()
    answer_choices: Counter = Counter()
    shortcut_choices: Counter = Counter()
    missing = 0
    matched = 0
    examples: dict[str, list[dict]] = defaultdict(list)

    for pred in predictions:
        example_id = pred.get("example_id")
        if example_id not in challenge:
            missing += 1
            continue
        matched += 1
        row = challenge[example_id]
        task = row["task"]
        bucket = "conflict" if row["is_shortcut_conflict"] else "aligned"
        choice = infer_choice(pred)
        for key in ((task, bucket), (task, "all")):
            totals[key] += 1
            if choice == "answer":
                answer_choices[key] += 1
            elif choice == "shortcut_answer":
                shortcut_choices[key] += 1
        if len(examples[task]) < 5:
            examples[task].append(
                {
                    "example_id": example_id,
                    "chosen_candidate": choice,
                    "prediction": pred.get("prediction", ""),
                    "answer": row["answer"],
                    "shortcut_answer": row["shortcut_answer"],
                    "answer_score": pred.get("answer_score"),
                    "shortcut_score": pred.get("shortcut_score"),
                    "is_shortcut_conflict": row["is_shortcut_conflict"],
                }
            )

    lines = [
        "# Pairwise True-vs-Shortcut Evaluation",
        "",
        f"- Challenge rows: {len(challenge):,}",
        f"- Prediction rows matched: {matched:,}",
        f"- Prediction rows ignored: {missing:,}",
        "",
        "| task | bucket | n | answer_choice_rate | shortcut_choice_rate |",
        "|---|---|---:|---:|---:|",
    ]
    for key in sorted(totals):
        task, bucket = key
        n = totals[key]
        lines.append(
            f"| {task} | {bucket} | {n} | "
            f"{answer_choices[key] / n:.3f} | {shortcut_choices[key] / n:.3f} |"
        )

    lines.extend(["", "## Example Predictions", ""])
    for task, task_examples in examples.items():
        lines.extend([f"### {task}", ""])
        for example in task_examples:
            lines.append("- " + json.dumps(example, ensure_ascii=False))
        lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
