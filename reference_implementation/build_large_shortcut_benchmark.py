#!/usr/bin/env python
"""Build a large shortcut-bias benchmark manifest for dataset release.

This builder is intended for a paper-scale benchmark, not a smoke test. It
learns metadata priors from a training split, applies them to the held-out split,
and emits:

- a full metadata-only manifest over all selected held-out cells and priors
- balanced conflict/aligned JSONL challenge slices
- prior/bias summary tables and a benchmark card

The full manifest intentionally does not duplicate expression token arrays.
Models can recover expression by `soma_joinid` from the source AnnData files or
from the existing Cell2Text datasets. This keeps the benchmark release compact.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
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
    "assay_ontology_term_id",
    "development_stage",
    "sex",
]


@dataclass(frozen=True)
class PriorSpec:
    prior_name: str
    task: str
    target: str
    features: tuple[str, ...]
    family: str
    question: str


PRIOR_SPECS = [
    PriorSpec(
        "combined_celltype_disease_to_tissue",
        "tissue_general_prediction",
        "tissue_general",
        ("cell_type", "disease"),
        "combined_metadata",
        "Predict the broad tissue of origin from expression only.",
    ),
    PriorSpec(
        "combined_celltype_tissue_to_disease",
        "disease_prediction",
        "disease",
        ("cell_type", "tissue_general"),
        "combined_metadata",
        "Predict the disease or normal state from expression only.",
    ),
    PriorSpec(
        "combined_tissue_disease_to_celltype",
        "cell_type_prediction",
        "cell_type",
        ("tissue_general", "disease"),
        "combined_metadata",
        "Predict the cell type from expression only.",
    ),
    PriorSpec(
        "celltype_to_tissue",
        "tissue_general_prediction",
        "tissue_general",
        ("cell_type",),
        "biological_metadata",
        "Predict the broad tissue of origin from expression only.",
    ),
    PriorSpec(
        "celltype_to_disease",
        "disease_prediction",
        "disease",
        ("cell_type",),
        "biological_metadata",
        "Predict the disease or normal state from expression only.",
    ),
    PriorSpec(
        "tissue_general_to_disease",
        "disease_prediction",
        "disease",
        ("tissue_general",),
        "biological_metadata",
        "Predict the disease or normal state from expression only.",
    ),
    PriorSpec(
        "tissue_to_disease",
        "disease_prediction",
        "disease",
        ("tissue",),
        "biological_metadata",
        "Predict the disease or normal state from expression only.",
    ),
    PriorSpec(
        "disease_to_tissue",
        "tissue_general_prediction",
        "tissue_general",
        ("disease",),
        "biological_metadata",
        "Predict the broad tissue of origin from expression only.",
    ),
    PriorSpec(
        "tissue_general_to_celltype",
        "cell_type_prediction",
        "cell_type",
        ("tissue_general",),
        "biological_metadata",
        "Predict the cell type from expression only.",
    ),
    PriorSpec(
        "assay_to_tissue",
        "tissue_general_prediction",
        "tissue_general",
        ("assay",),
        "technical_metadata",
        "Predict the broad tissue of origin from expression only.",
    ),
    PriorSpec(
        "assay_to_disease",
        "disease_prediction",
        "disease",
        ("assay",),
        "technical_metadata",
        "Predict the disease or normal state from expression only.",
    ),
    PriorSpec(
        "sex_to_disease",
        "disease_prediction",
        "disease",
        ("sex",),
        "demographic_metadata",
        "Predict the disease or normal state from expression only.",
    ),
    PriorSpec(
        "development_stage_to_disease",
        "disease_prediction",
        "disease",
        ("development_stage",),
        "demographic_metadata",
        "Predict the disease or normal state from expression only.",
    ),
    PriorSpec(
        "global_to_disease",
        "disease_prediction",
        "disease",
        (),
        "label_imbalance",
        "Predict the disease or normal state from expression only.",
    ),
    PriorSpec(
        "global_to_tissue",
        "tissue_general_prediction",
        "tissue_general",
        (),
        "label_imbalance",
        "Predict the broad tissue of origin from expression only.",
    ),
    PriorSpec(
        "global_to_celltype",
        "cell_type_prediction",
        "cell_type",
        (),
        "label_imbalance",
        "Predict the cell type from expression only.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=0,
        help="0 means all eval rows. Use a small value only for debugging.",
    )
    parser.add_argument(
        "--min-prior-support",
        type=int,
        default=25,
        help="Minimum train rows for a feature-key mode to be used.",
    )
    parser.add_argument(
        "--min-prior-purity",
        type=float,
        default=0.6,
        help="Optional minimum mode fraction for a feature-key prior.",
    )
    parser.add_argument(
        "--balanced-per-prior-bucket",
        type=int,
        default=5000,
        help="Rows per prior/conflict-status bucket for balanced challenge JSONL.",
    )
    parser.add_argument(
        "--manifest-format",
        choices=["parquet", "jsonl_gz", "csv_gz"],
        default="parquet",
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


def normalize_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def normalize_value(value) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def make_frame(ds, columns: list[str], max_rows: int = 0) -> pd.DataFrame:
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    frame = ds.select_columns([col for col in columns if col in ds.column_names]).to_pandas()
    for col in columns:
        if col not in frame:
            frame[col] = ""
        frame[col] = normalize_series(frame[col])
    return frame


def mode_or_empty(series: pd.Series) -> str:
    series = series[series != ""]
    if series.empty:
        return ""
    return str(series.mode().iat[0])


def build_prior_lookup(
    train: pd.DataFrame,
    spec: PriorSpec,
    min_support: int,
    min_purity: float,
) -> tuple[dict[tuple[str, ...], dict], str, int]:
    target = spec.target
    features = list(spec.features)
    needed = [*features, target]
    frame = train[needed].copy()
    frame = frame[frame[target] != ""]
    for feature in features:
        frame = frame[frame[feature] != ""]
    global_mode = mode_or_empty(frame[target])
    if frame.empty or not global_mode:
        return {}, "", 0

    if not features:
        support = len(frame)
        mode_count = int((frame[target] == global_mode).sum())
        purity = mode_count / support if support else 0.0
        if support < min_support or purity < min_purity:
            return {}, global_mode, len(frame)
        return {
            (): {
                "answer": global_mode,
                "support": support,
                "mode_count": mode_count,
                "purity": purity,
                "n_unique_targets": int(frame[target].nunique()),
            }
        }, global_mode, len(frame)

    grouped = frame.groupby([*features, target], dropna=False, observed=True).size().rename("mode_count").reset_index()
    totals = frame.groupby(features, dropna=False, observed=True).size().rename("support").reset_index()
    merged = grouped.merge(totals, on=features, how="left")
    winners = merged.loc[merged.groupby(features, observed=True)["mode_count"].idxmax()].copy()
    unique_counts = frame.groupby(features, dropna=False, observed=True)[target].nunique().rename("n_unique_targets").reset_index()
    winners = winners.merge(unique_counts, on=features, how="left")
    winners["purity"] = winners["mode_count"] / winners["support"]
    winners = winners[
        (winners["support"] >= min_support)
        & (winners["purity"] >= min_purity)
    ]

    lookup: dict[tuple[str, ...], dict] = {}
    for _, row in winners.iterrows():
        key = tuple(str(row[feature]) for feature in features)
        lookup[key] = {
            "answer": str(row[target]),
            "support": int(row["support"]),
            "mode_count": int(row["mode_count"]),
            "purity": float(row["purity"]),
            "n_unique_targets": int(row["n_unique_targets"]),
        }
    return lookup, global_mode, len(frame)


def feature_key(row: pd.Series, features: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(normalize_value(row[feature]) for feature in features)


def ontology_id(row: pd.Series, target: str) -> str:
    col = f"{target}_ontology_term_id"
    if col in row:
        return normalize_value(row[col])
    return ""


def make_record(row: pd.Series, spec: PriorSpec, prior: dict) -> dict:
    answer = normalize_value(row[spec.target])
    shortcut_answer = prior["answer"]
    is_conflict = answer != shortcut_answer
    soma_joinid = normalize_value(row.get("soma_joinid"))
    example_id = f"{soma_joinid}::{spec.prior_name}::{spec.task}"
    feature_values = {
        feature: normalize_value(row.get(feature))
        for feature in spec.features
    }
    metadata = {
        col: normalize_value(row.get(col))
        for col in METADATA_COLUMNS
        if col in row
    }
    return {
        "example_id": example_id,
        "soma_joinid": soma_joinid,
        "task": spec.task,
        "question": spec.question,
        "target": spec.target,
        "target_ontology_term_id": ontology_id(row, spec.target),
        "answer": answer,
        "shortcut_answer": shortcut_answer,
        "is_shortcut_conflict": is_conflict,
        "prior_name": spec.prior_name,
        "prior_family": spec.family,
        "shortcut_features": list(spec.features),
        "shortcut_feature_values": feature_values,
        "prior_support": prior["support"],
        "prior_mode_count": prior["mode_count"],
        "prior_purity": prior["purity"],
        "prior_n_unique_targets": prior["n_unique_targets"],
        "metadata": metadata,
    }


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        flat = {
            key: value
            for key, value in record.items()
            if key not in {"metadata", "shortcut_features", "shortcut_feature_values"}
        }
        flat["shortcut_features"] = "|".join(record["shortcut_features"])
        for key, value in record["shortcut_feature_values"].items():
            flat[f"feature_{key}"] = value
        for key, value in record["metadata"].items():
            flat[f"metadata_{key}"] = value
        rows.append(flat)
    return pd.DataFrame(rows)


def write_manifest(records: list[dict], output_dir: Path, manifest_format: str) -> Path:
    if manifest_format == "parquet":
        path = output_dir / "metadata_prior_manifest.parquet"
        frame = records_to_frame(records)
        try:
            frame.to_parquet(path, index=False)
            return path
        except Exception as exc:
            fallback = output_dir / "metadata_prior_manifest.jsonl.gz"
            print(f"Parquet write failed ({exc}); falling back to {fallback}")
            write_jsonl_gz(records, fallback)
            return fallback
    if manifest_format == "csv_gz":
        path = output_dir / "metadata_prior_manifest.csv.gz"
        records_to_frame(records).to_csv(path, index=False)
        return path
    path = output_dir / "metadata_prior_manifest.jsonl.gz"
    write_jsonl_gz(records, path)
    return path


def write_jsonl_gz(records: Iterable[dict], path: Path) -> None:
    with gzip.open(path, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def sample_balanced(records: list[dict], per_bucket: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, bool], list[dict]] = {}
    for record in records:
        key = (record["prior_name"], bool(record["is_shortcut_conflict"]))
        buckets.setdefault(key, []).append(record)
    selected: list[dict] = []
    for key in sorted(buckets):
        bucket_records = buckets[key]
        rng.shuffle(bucket_records)
        selected.extend(bucket_records[:per_bucket])
    rng.shuffle(selected)
    return selected


def prior_baseline_records(records: Iterable[dict]) -> list[dict]:
    return [
        {
            "example_id": record["example_id"],
            "prediction": record["shortcut_answer"],
        }
        for record in records
    ]


def summarize_records(records: list[dict], specs: list[PriorSpec]) -> pd.DataFrame:
    rows = []
    spec_by_name = {spec.prior_name: spec for spec in specs}
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["prior_name"], []).append(record)
    for prior_name, group in sorted(grouped.items()):
        spec = spec_by_name[prior_name]
        n = len(group)
        conflicts = sum(bool(record["is_shortcut_conflict"]) for record in group)
        aligned = n - conflicts
        accuracy = aligned / n if n else math.nan
        conflict_rate = conflicts / n if n else math.nan
        rows.append(
            {
                "prior_name": prior_name,
                "family": spec.family,
                "task": spec.task,
                "target": spec.target,
                "features": "+".join(spec.features) if spec.features else "global",
                "eval_rows": n,
                "aligned_rows": aligned,
                "conflict_rows": conflicts,
                "conflict_rate": conflict_rate,
                "metadata_prior_accuracy": accuracy,
                "mean_prior_support": sum(record["prior_support"] for record in group) / n if n else math.nan,
                "mean_prior_purity": sum(record["prior_purity"] for record in group) / n if n else math.nan,
            }
        )
    return pd.DataFrame(rows)


def write_summary_markdown(
    path: Path,
    summary: pd.DataFrame,
    args: argparse.Namespace,
    manifest_path: Path,
    n_train: int,
    n_eval: int,
    n_records: int,
    n_balanced: int,
) -> None:
    lines = [
        "# Large Shortcut Benchmark Build Report",
        "",
        "## Build Inputs",
        "",
        f"- Train root: `{args.train_root}`",
        f"- Eval root: `{args.eval_root}`",
        f"- Train rows: {n_train:,}",
        f"- Eval rows scanned: {n_eval:,}",
        f"- Manifest records: {n_records:,}",
        f"- Balanced challenge records: {n_balanced:,}",
        f"- Minimum prior support: {args.min_prior_support:,}",
        f"- Minimum prior purity: {args.min_prior_purity:.3f}",
        f"- Manifest: `{manifest_path.name}`",
        "",
        "## Prior Summary",
        "",
        "| prior | family | task | features | eval | conflict | conflict_rate | prior_acc | mean_support | mean_purity |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.sort_values(["family", "prior_name"]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["prior_name"]),
                    str(row["family"]),
                    str(row["task"]),
                    str(row["features"]),
                    f"{int(row['eval_rows']):,}",
                    f"{int(row['conflict_rows']):,}",
                    f"{row['conflict_rate']:.3f}",
                    f"{row['metadata_prior_accuracy']:.3f}",
                    f"{row['mean_prior_support']:.1f}",
                    f"{row['mean_prior_purity']:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Rows marked as conflicts are the critical benchmark slice: the shortcut",
            "answer learned from training metadata disagrees with the held-out true",
            "label. A model that prefers `shortcut_answer` on these rows is following",
            "a construction prior instead of the held-out expression-grounded label.",
            "",
            "This benchmark is metadata-only by default. Use `soma_joinid` plus the",
            "source AnnData or Cell2Text dataset to retrieve expression for model",
            "inference. This avoids duplicating expression arrays or long token lists",
            "across multiple prior tasks.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_benchmark_card(path: Path, args: argparse.Namespace) -> None:
    text = f"""# Shortcut Reasoning Benchmark Card

## Summary

This is a large CELLxGENE-derived shortcut benchmark for testing whether
single-cell models rely on expression-grounded evidence or on dataset
construction priors.

## Intended Use

This benchmark tests whether single-cell models predict biological labels from
expression or from construction shortcuts in CELLxGENE-derived metadata.

## Core Evaluation

For each example, the benchmark provides:

- `answer`: held-out true label.
- `shortcut_answer`: label predicted by a metadata prior fitted only on the
  training split.
- `is_shortcut_conflict`: whether `answer != shortcut_answer`.

On conflict rows, high shortcut agreement is evidence of shortcut reliance.

## Included Shortcut Families

- Combined metadata shortcuts such as `(cell_type, tissue) -> disease`.
- Biological metadata shortcuts such as `cell_type -> tissue` and `tissue -> disease`.
- Technical shortcuts such as `assay -> tissue`.
- Demographic shortcuts such as `sex -> disease`.

## Release Notes

The full manifest is metadata-only to keep the benchmark compact. Expression
matrices should be joined by `soma_joinid`. The canonical release build uses
high-purity metadata priors so that conflict rows represent strong train-set
shortcuts that are wrong for held-out examples.

Build command parameters:

- Train root: `{args.train_root}`
- Eval root: `{args.eval_root}`
- Minimum prior support: `{args.min_prior_support}`
- Minimum prior purity: `{args.min_prior_purity}`
- Balanced rows per prior/bucket: `{args.balanced_per_prior_bucket}`

## Recommended Metrics

- Truth accuracy on conflict rows.
- Shortcut agreement on conflict rows.
- Difference in shortcut agreement between aligned and conflict rows.
- Per-prior-family reporting for biological, combined, technical, demographic,
  and other retained shortcut families.
- Bootstrap confidence intervals stratified by prior family and target task.
- Ontology-aware secondary scoring for parent/child labels.

## Caution

This benchmark demonstrates shortcut vulnerability. Claims about lack of
biological reasoning should be made only after model-native inference, large
conflict sets, confidence intervals, and decorrelated control splits.
"""
    path.write_text(text)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)

    needed_columns = sorted(set(METADATA_COLUMNS))
    train_ds = load_dataset_root(args.train_root, columns=needed_columns)
    eval_ds = load_dataset_root(args.eval_root, columns=needed_columns)
    n_train = len(train_ds)
    n_eval_total = len(eval_ds)
    train = make_frame(train_ds, needed_columns)
    eval_frame = make_frame(eval_ds, needed_columns, max_rows=args.max_eval_rows)
    n_eval = len(eval_frame)

    lookups = {}
    retained_specs = []
    for spec in PRIOR_SPECS:
        lookup, global_mode, n_target_train = build_prior_lookup(
            train,
            spec,
            min_support=args.min_prior_support,
            min_purity=args.min_prior_purity,
        )
        if lookup:
            lookups[spec.prior_name] = (lookup, global_mode, n_target_train)
            retained_specs.append(spec)

    records: list[dict] = []
    for spec in retained_specs:
        lookup, global_mode, _ = lookups[spec.prior_name]
        for _, row in eval_frame.iterrows():
            answer = normalize_value(row.get(spec.target))
            if not answer:
                continue
            key = feature_key(row, spec.features)
            prior = lookup.get(key)
            if prior is None:
                continue
            records.append(make_record(row, spec, prior))

    manifest_path = write_manifest(records, output_dir / "data", args.manifest_format)
    balanced = sample_balanced(records, args.balanced_per_prior_bucket, args.seed)
    balanced_path = output_dir / "data" / "balanced_shortcut_challenge.jsonl"
    write_jsonl(balanced, balanced_path)
    write_jsonl_gz((record for record in records if record["is_shortcut_conflict"]), output_dir / "data" / "all_conflicts.jsonl.gz")

    prior_predictions = prior_baseline_records(balanced)
    write_jsonl(prior_predictions, output_dir / "predictions" / "metadata_prior_baseline_balanced.jsonl")

    summary = summarize_records(records, retained_specs)
    summary_path = output_dir / "reports" / "prior_summary.csv"
    summary.to_csv(summary_path, index=False)
    write_summary_markdown(
        output_dir / "reports" / "build_report.md",
        summary,
        args,
        manifest_path,
        n_train,
        n_eval,
        len(records),
        len(balanced),
    )
    write_benchmark_card(output_dir / "BENCHMARK_CARD.md", args)

    metadata = {
        "train_root": args.train_root,
        "eval_root": args.eval_root,
        "n_train_rows": n_train,
        "n_eval_rows_total": n_eval_total,
        "n_eval_rows_scanned": n_eval,
        "n_manifest_records": len(records),
        "n_balanced_records": len(balanced),
        "min_prior_support": args.min_prior_support,
        "min_prior_purity": args.min_prior_purity,
        "balanced_per_prior_bucket": args.balanced_per_prior_bucket,
        "manifest_path": manifest_path.as_posix(),
        "balanced_challenge_path": balanced_path.as_posix(),
        "prior_summary_path": summary_path.as_posix(),
        "prior_specs": [
            {
                "prior_name": spec.prior_name,
                "task": spec.task,
                "target": spec.target,
                "features": list(spec.features),
                "family": spec.family,
            }
            for spec in retained_specs
        ],
    }
    (output_dir / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote manifest records: {len(records):,}")
    print(f"Wrote balanced challenge records: {len(balanced):,}")
    print(f"Wrote output dir: {output_dir}")


if __name__ == "__main__":
    main()
