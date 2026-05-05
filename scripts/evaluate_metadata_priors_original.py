#!/usr/bin/env python
"""Evaluate arbitrary metadata-prior shortcut agreement.

Examples:
    tissue_general -> disease
    disease -> tissue_general
    assay -> tissue_general
    global prior -> disease

The prior is learned from the training split. On challenge rows where the prior
answer differs from the true answer, agreement with the prior is evidence that a
model may be using a construction shortcut rather than expression-grounded
reasoning.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from datasets import concatenate_datasets, load_from_disk


TARGET_TO_TASK = {
    "cell_type": "cell_type_prediction",
    "disease": "disease_prediction",
    "tissue_general": "tissue_general_prediction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--target", required=True, choices=sorted(TARGET_TO_TASK))
    parser.add_argument(
        "--features",
        nargs="*",
        default=[],
        help="Metadata features used to predict target. Empty means global target prior.",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional model prediction JSONL. If omitted, evaluates the metadata prior itself.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--baseline-output",
        default=None,
        help="Optional JSONL path for writing the prior baseline predictions.",
    )
    parser.add_argument("--name", default=None, help="Human-readable prior name for the report.")
    return parser.parse_args()


def is_hf_dataset_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "dataset_info.json").exists()
        or (path / "state.json").exists()
        or (path / "dataset_dict.json").exists()
    )


def find_dataset_dirs(root: Path) -> list[Path]:
    if is_hf_dataset_dir(root):
        return [root]
    return sorted(path for path in root.iterdir() if is_hf_dataset_dir(path))


def load_dataset_root(root: str, columns: Iterable[str]):
    dirs = find_dataset_dirs(Path(root))
    if not dirs:
        raise FileNotFoundError(f"No Hugging Face datasets found under {root}")
    datasets = []
    for dataset_dir in dirs:
        ds = load_from_disk(str(dataset_dir), keep_in_memory=False)
        keep = [col for col in columns if col in ds.column_names]
        ds = ds.select_columns(keep)
        datasets.append(ds)
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def norm(text) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def label_match(prediction: str, answer: str) -> bool:
    pred = norm(prediction)
    ans = norm(answer)
    return pred == ans or bool(ans and ans in pred)


def normalize_value(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metadata_value(row: dict, feature: str) -> str:
    metadata = row.get("metadata") or {}
    return normalize_value(metadata.get(feature) or row.get(feature))


def build_prior(
    train_root: str,
    features: list[str],
    target: str,
) -> tuple[dict[tuple[str, ...], str], str, dict[tuple[str, ...], int]]:
    columns = [*features, target]
    train_ds = load_dataset_root(train_root, columns=columns)
    frame = train_ds.to_pandas()
    for col in columns:
        if col not in frame:
            frame[col] = ""
        frame[col] = frame[col].astype("string").fillna("").str.strip()
    frame = frame[frame[target] != ""]
    if features:
        for feature in features:
            frame = frame[frame[feature] != ""]
    if frame.empty:
        return {}, "", {}

    global_mode = frame[target].mode().iat[0]
    if not features:
        return {(): global_mode}, global_mode, {(): len(frame)}

    grouped = frame.groupby([*features, target], dropna=False, observed=True).size().rename("n").reset_index()
    winners = grouped.loc[grouped.groupby(features, observed=True)["n"].idxmax()]
    lookup = {
        tuple(row[feature] for feature in features): row[target]
        for _, row in winners.iterrows()
    }
    support = {
        tuple(row[feature] for feature in features): int(row["n"])
        for _, row in winners.iterrows()
    }
    return lookup, global_mode, support


def prior_key(row: dict, features: list[str]) -> tuple[str, ...]:
    if not features:
        return ()
    return tuple(metadata_value(row, feature) for feature in features)


def build_prior_records(
    challenge_rows: list[dict],
    target: str,
    features: list[str],
    lookup: dict[tuple[str, ...], str],
    global_mode: str,
    support: dict[tuple[str, ...], int],
) -> list[dict]:
    task = TARGET_TO_TASK[target]
    records = []
    for row in challenge_rows:
        if row.get("task") != task:
            continue
        key = prior_key(row, features)
        prior_answer = lookup.get(key, global_mode)
        if not prior_answer:
            continue
        answer = normalize_value(row.get("answer"))
        records.append(
            {
                "example_id": row["example_id"],
                "task": task,
                "target": target,
                "features": features,
                "feature_values": dict(zip(features, key)),
                "answer": answer,
                "prior_answer": prior_answer,
                "prior_support": support.get(key, 0),
                "is_prior_conflict": prior_answer != answer,
            }
        )
    return records


def prediction_map(prediction_rows: list[dict]) -> dict[str, str]:
    return {
        row["example_id"]: row.get("prediction", "")
        for row in prediction_rows
        if "example_id" in row
    }


def main() -> None:
    args = parse_args()
    features = list(args.features)
    prior_name = args.name or (
        f"{' + '.join(features) if features else 'global'} -> {args.target}"
    )

    challenge_rows = load_jsonl(args.challenge)
    lookup, global_mode, support = build_prior(args.train_root, features, args.target)
    prior_records = build_prior_records(
        challenge_rows,
        args.target,
        features,
        lookup,
        global_mode,
        support,
    )

    if args.baseline_output:
        baseline_path = Path(args.baseline_output)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with baseline_path.open("w") as handle:
            for row in prior_records:
                handle.write(
                    json.dumps(
                        {
                            "example_id": row["example_id"],
                            "prediction": row["prior_answer"],
                        }
                    )
                    + "\n"
                )

    if args.predictions:
        preds = prediction_map(load_jsonl(args.predictions))
        run_label = args.predictions
    else:
        preds = {
            row["example_id"]: row["prior_answer"]
            for row in prior_records
        }
        run_label = f"{prior_name} baseline"

    totals: Counter = Counter()
    correct: Counter = Counter()
    prior_agree: Counter = Counter()
    examples: dict[str, list[dict]] = defaultdict(list)
    missing = 0
    matched = 0

    for row in prior_records:
        example_id = row["example_id"]
        if example_id not in preds:
            missing += 1
            continue
        matched += 1
        task = row["task"]
        bucket = "prior_conflict" if row["is_prior_conflict"] else "prior_aligned"
        prediction = preds[example_id]
        for key in ((task, bucket), (task, "all")):
            totals[key] += 1
            if label_match(prediction, row["answer"]):
                correct[key] += 1
            if label_match(prediction, row["prior_answer"]):
                prior_agree[key] += 1
        if len(examples[task]) < 5:
            examples[task].append(
                {
                    "example_id": example_id,
                    "prediction": prediction,
                    "answer": row["answer"],
                    "prior_answer": row["prior_answer"],
                    "feature_values": row["feature_values"],
                    "prior_support": row["prior_support"],
                    "is_prior_conflict": row["is_prior_conflict"],
                }
            )

    lines = [
        "# Metadata-Prior Shortcut Evaluation",
        "",
        f"- Prior: `{prior_name}`",
        f"- Run: `{run_label}`",
        f"- Challenge rows: {len(challenge_rows):,}",
        f"- Rows for target task: {len(prior_records):,}",
        f"- Prediction rows matched: {matched:,}",
        f"- Prediction rows missing: {missing:,}",
        "",
        "| task | bucket | n | accuracy | prior_agreement |",
        "|---|---|---:|---:|---:|",
    ]
    for key in sorted(totals):
        task, bucket = key
        n = totals[key]
        lines.append(
            f"| {task} | {bucket} | {n} | {correct[key] / n:.3f} | {prior_agree[key] / n:.3f} |"
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
