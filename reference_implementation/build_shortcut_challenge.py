#!/usr/bin/env python
"""Build a shortcut-vs-expression challenge set.

The benchmark does not assume a specific model. It emits JSONL tasks with:

- expression token ids from the cell dataset
- the true target label
- the metadata-only shortcut answer learned from the training split
- a conflict flag indicating whether shortcut_answer != answer

When a model is run on conflict rows, choosing shortcut_answer is evidence that
it is following atlas priors rather than expression-grounded signal.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import concatenate_datasets, load_from_disk


METADATA_COLUMNS = [
    "soma_joinid",
    "cell_type",
    "cell_type_ontology_term_id",
    "tissue",
    "tissue_ontology_term_id",
    "tissue_general",
    "tissue_general_ontology_term_id",
    "disease",
    "disease_ontology_term_id",
    "assay",
    "development_stage",
    "sex",
    "pathway1",
    "pathway2",
    "natural_desc",
    "natural_desc_base",
    "expression_state_desc",
    "expression_programs",
    "expression_program_scores",
    "expression_program_genes",
    "expression_background_key",
]

TASK_SPECS = [
    {
        "task": "tissue_general_prediction",
        "target": "tissue_general",
        "features": ("cell_type", "disease"),
        "question": "Predict the broad tissue of origin from expression only.",
    },
    {
        "task": "disease_prediction",
        "target": "disease",
        "features": ("cell_type", "tissue_general"),
        "question": "Predict the disease or normal state from expression only.",
    },
    {
        "task": "cell_type_prediction",
        "target": "cell_type",
        "features": ("tissue_general", "disease"),
        "question": "Predict the cell type from expression only.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True, help="HF dataset dir or directory containing HF dataset dirs.")
    parser.add_argument("--eval-root", required=True, help="HF dataset dir or directory containing HF dataset dirs.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--report", default=None, help="Optional markdown summary path.")
    parser.add_argument("--max-per-task-bucket", type=int, default=250)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=200_000,
        help="Maximum shuffled evaluation rows to scan before writing the challenge set. Use 0 for no limit.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-expression-state",
        action="store_true",
        help="Add expression-state generation tasks when expression_state_desc is present.",
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


def load_dataset_root(root: str, columns: Iterable[str] | None = None):
    dirs = find_dataset_dirs(Path(root))
    if not dirs:
        raise FileNotFoundError(f"No Hugging Face datasets found under {root}")

    datasets = []
    for dataset_dir in dirs:
        ds = load_from_disk(str(dataset_dir), keep_in_memory=False)
        if columns is not None:
            keep = [col for col in columns if col in ds.column_names]
            ds = ds.select_columns(keep)
        datasets.append(ds)
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def normalize(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_mode_lookup(train_ds, features: tuple[str, ...], target: str) -> tuple[dict[tuple[str, ...], str], str]:
    needed = [*features, target]
    missing = set(needed) - set(train_ds.column_names)
    if missing:
        return {}, ""

    frame = train_ds.select_columns(needed).to_pandas()
    for col in needed:
        frame[col] = frame[col].astype("string").fillna("").str.strip()
    frame = frame[frame[target] != ""]
    if frame.empty:
        return {}, ""

    global_mode = frame[target].mode().iat[0]
    grouped = frame.groupby(needed, dropna=False, observed=True).size().rename("n").reset_index()
    winners = grouped.loc[grouped.groupby(list(features), observed=True)["n"].idxmax()]
    lookup = {
        tuple(row[col] for col in features): row[target]
        for _, row in winners.iterrows()
    }
    return lookup, global_mode


def make_base_record(row: dict, max_input_tokens: int) -> dict:
    input_ids = row.get("input_ids") or []
    if not isinstance(input_ids, list):
        input_ids = list(input_ids)
    return {
        "soma_joinid": row.get("soma_joinid"),
        "cell2text_geneformer_token_ids": input_ids[:max_input_tokens],
        "metadata": {
            col: row.get(col)
            for col in METADATA_COLUMNS
            if col in row and col not in {"natural_desc", "natural_desc_base", "expression_state_desc"}
        },
        "reference_text": {
            col: row.get(col)
            for col in ["natural_desc", "natural_desc_base", "expression_state_desc"]
            if col in row
        },
    }


def add_label_tasks(
    records: list[dict],
    row: dict,
    row_i: int,
    lookups: dict[str, tuple[dict[tuple[str, ...], str], str]],
    bucket_counts: Counter,
    max_per_task_bucket: int,
    max_input_tokens: int,
) -> None:
    for spec in TASK_SPECS:
        target = spec["target"]
        if target not in row:
            continue
        answer = normalize(row[target])
        if not answer:
            continue
        lookup, global_mode = lookups[spec["task"]]
        key = tuple(normalize(row.get(col)) for col in spec["features"])
        shortcut_answer = lookup.get(key, global_mode)
        if not shortcut_answer:
            continue
        is_conflict = shortcut_answer != answer
        bucket = f"{spec['task']}:{'conflict' if is_conflict else 'aligned'}"
        if bucket_counts[bucket] >= max_per_task_bucket:
            continue

        record = make_base_record(row, max_input_tokens)
        record.update(
            {
                "example_id": f"{row.get('soma_joinid', row_i)}::{spec['task']}",
                "task": spec["task"],
                "question": spec["question"],
                "answer": answer,
                "shortcut_answer": shortcut_answer,
                "shortcut_features": list(spec["features"]),
                "shortcut_feature_values": {col: row.get(col) for col in spec["features"]},
                "is_shortcut_conflict": is_conflict,
            }
        )
        records.append(record)
        bucket_counts[bucket] += 1


def add_expression_state_task(
    records: list[dict],
    row: dict,
    row_i: int,
    bucket_counts: Counter,
    max_per_task_bucket: int,
    max_input_tokens: int,
) -> None:
    answer = normalize(row.get("expression_state_desc"))
    if not answer:
        return
    bucket = "expression_state_generation:all"
    if bucket_counts[bucket] >= max_per_task_bucket:
        return
    record = make_base_record(row, max_input_tokens)
    record.update(
        {
            "example_id": f"{row.get('soma_joinid', row_i)}::expression_state_generation",
            "task": "expression_state_generation",
            "question": "Describe expression programs elevated relative to matched background cells.",
            "answer": answer,
            "shortcut_answer": normalize(row.get("natural_desc_base") or row.get("natural_desc")),
            "shortcut_features": ["cell_type", "tissue_general", "disease"],
            "shortcut_feature_values": {
                "cell_type": row.get("cell_type"),
                "tissue_general": row.get("tissue_general"),
                "disease": row.get("disease"),
            },
            "is_shortcut_conflict": True,
        }
    )
    records.append(record)
    bucket_counts[bucket] += 1


def required_buckets(include_expression_state: bool) -> set[str]:
    buckets = {
        f"{spec['task']}:{bucket}"
        for spec in TASK_SPECS
        for bucket in ("conflict", "aligned")
    }
    if include_expression_state:
        buckets.add("expression_state_generation:all")
    return buckets


def buckets_are_full(bucket_counts: Counter, buckets: set[str], max_per_task_bucket: int) -> bool:
    return all(bucket_counts[bucket] >= max_per_task_bucket for bucket in buckets)


def write_report(path: Path, records: list[dict], args: argparse.Namespace) -> None:
    counts = Counter(record["task"] for record in records)
    conflict_counts = Counter(
        (record["task"], "conflict" if record["is_shortcut_conflict"] else "aligned")
        for record in records
    )
    lines = [
        "# Shortcut Challenge Build Report",
        "",
        f"- Train root: `{args.train_root}`",
        f"- Eval root: `{args.eval_root}`",
        f"- Output: `{args.output}`",
        f"- Examples: {len(records):,}",
        "",
        "## Task Counts",
        "",
        "| task | count | conflict | aligned |",
        "|---|---:|---:|---:|",
    ]
    for task, count in sorted(counts.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    task,
                    str(count),
                    str(conflict_counts[(task, "conflict")]),
                    str(conflict_counts[(task, "aligned")]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    train_columns = sorted(METADATA_COLUMNS)
    eval_columns = sorted({"input_ids", *METADATA_COLUMNS})
    train_ds = load_dataset_root(args.train_root, columns=train_columns)
    eval_ds = load_dataset_root(args.eval_root, columns=eval_columns).shuffle(seed=args.seed)

    lookups = {
        spec["task"]: build_mode_lookup(train_ds, spec["features"], spec["target"])
        for spec in TASK_SPECS
    }

    records: list[dict] = []
    bucket_counts: Counter = Counter()
    buckets_to_fill = required_buckets(args.include_expression_state)
    for row_i, row in enumerate(eval_ds):
        if args.max_eval_rows and row_i >= args.max_eval_rows:
            break
        add_label_tasks(
            records,
            row,
            row_i,
            lookups,
            bucket_counts,
            args.max_per_task_bucket,
            args.max_input_tokens,
        )
        if args.include_expression_state:
            add_expression_state_task(
                records,
                row,
                row_i,
                bucket_counts,
                args.max_per_task_bucket,
                args.max_input_tokens,
            )
        if buckets_are_full(bucket_counts, buckets_to_fill, args.max_per_task_bucket):
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    report = Path(args.report) if args.report else output.with_suffix(".md")
    write_report(report, records, args)
    print(f"Wrote {len(records):,} challenge examples to {output}")
    print(f"Wrote report to {report}")


if __name__ == "__main__":
    main()
