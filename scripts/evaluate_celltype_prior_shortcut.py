#!/usr/bin/env python
"""Evaluate tissue/disease predictions against a cell-type-only prior.

This tests a distinct shortcut from the full metadata shortcut benchmark:

    cell_type -> tissue_general
    cell_type -> disease

The prior is learned from the training split. A model agreeing with this prior
on rows where the true answer differs may be inferring metadata from cell type
instead of using expression-grounded evidence for tissue/disease.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from datasets import concatenate_datasets, load_from_disk


TASK_TO_TARGET = {
    "tissue_general_prediction": "tissue_general",
    "disease_prediction": "disease",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--challenge", required=True)
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional model prediction JSONL. If omitted, evaluates the prior baseline itself.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--baseline-output",
        default=None,
        help="Optional JSONL path for the cell-type-prior baseline predictions.",
    )
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


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_value(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_celltype_prior(train_root: str) -> dict[str, tuple[dict[str, str], str]]:
    train_ds = load_dataset_root(
        train_root,
        columns=["cell_type", "tissue_general", "disease"],
    )
    frame = train_ds.to_pandas()
    priors: dict[str, tuple[dict[str, str], str]] = {}
    for target in ("tissue_general", "disease"):
        sub = frame[["cell_type", target]].copy()
        sub["cell_type"] = sub["cell_type"].astype("string").fillna("").str.strip()
        sub[target] = sub[target].astype("string").fillna("").str.strip()
        sub = sub[(sub["cell_type"] != "") & (sub[target] != "")]
        if sub.empty:
            priors[target] = ({}, "")
            continue
        global_mode = sub[target].mode().iat[0]
        grouped = sub.groupby(["cell_type", target], dropna=False, observed=True).size().rename("n").reset_index()
        winners = grouped.loc[grouped.groupby("cell_type", observed=True)["n"].idxmax()]
        lookup = {row["cell_type"]: row[target] for _, row in winners.iterrows()}
        priors[target] = (lookup, global_mode)
    return priors


def celltype_from_row(row: dict) -> str:
    metadata = row.get("metadata") or {}
    return normalize_value(metadata.get("cell_type") or row.get("cell_type"))


def build_prior_records(challenge_rows: list[dict], priors: dict[str, tuple[dict[str, str], str]]) -> list[dict]:
    records = []
    for row in challenge_rows:
        task = row.get("task")
        if task not in TASK_TO_TARGET:
            continue
        target = TASK_TO_TARGET[task]
        cell_type = celltype_from_row(row)
        lookup, global_mode = priors[target]
        prior_answer = lookup.get(cell_type, global_mode)
        if not prior_answer:
            continue
        records.append(
            {
                "example_id": row["example_id"],
                "task": task,
                "answer": normalize_value(row.get("answer")),
                "cell_type": cell_type,
                "celltype_prior_answer": prior_answer,
                "is_celltype_prior_conflict": prior_answer != normalize_value(row.get("answer")),
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
    challenge_rows = load_jsonl(args.challenge)
    priors = build_celltype_prior(args.train_root)
    prior_records = build_prior_records(challenge_rows, priors)

    if args.baseline_output:
        baseline_path = Path(args.baseline_output)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with baseline_path.open("w") as handle:
            for row in prior_records:
                handle.write(
                    json.dumps(
                        {
                            "example_id": row["example_id"],
                            "prediction": row["celltype_prior_answer"],
                        }
                    )
                    + "\n"
                )

    if args.predictions:
        preds = prediction_map(load_jsonl(args.predictions))
        run_label = args.predictions
    else:
        preds = {
            row["example_id"]: row["celltype_prior_answer"]
            for row in prior_records
        }
        run_label = "cell-type-prior baseline"

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
        bucket = "celltype_prior_conflict" if row["is_celltype_prior_conflict"] else "celltype_prior_aligned"
        prediction = preds[example_id]
        for key in ((task, bucket), (task, "all")):
            totals[key] += 1
            if label_match(prediction, row["answer"]):
                correct[key] += 1
            if label_match(prediction, row["celltype_prior_answer"]):
                prior_agree[key] += 1
        if len(examples[task]) < 5:
            examples[task].append(
                {
                    "example_id": example_id,
                    "prediction": prediction,
                    "answer": row["answer"],
                    "celltype_prior_answer": row["celltype_prior_answer"],
                    "cell_type": row["cell_type"],
                    "is_celltype_prior_conflict": row["is_celltype_prior_conflict"],
                }
            )

    lines = [
        "# Cell-Type-Prior Shortcut Evaluation",
        "",
        f"- Run: `{run_label}`",
        f"- Challenge rows: {len(challenge_rows):,}",
        f"- Tissue/disease rows with cell-type prior: {len(prior_records):,}",
        f"- Prediction rows matched: {matched:,}",
        f"- Prediction rows missing: {missing:,}",
        "",
        "| task | bucket | n | accuracy | celltype_prior_agreement |",
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
