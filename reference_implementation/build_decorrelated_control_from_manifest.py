#!/usr/bin/env python
"""Build a decorrelated control challenge from a shortcut manifest.

The large shortcut challenge shows where train-set metadata priors are wrong on
held-out rows. This control slice complements it: for each shortcut prior, it
finds held-out strata where the shortcut features map to multiple target labels,
then samples labels evenly within that stratum. The resulting rows are still
real cells, but the shortcut feature alone cannot solve the target task.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable

import pandas as pd


CORE_COLUMNS = [
    "example_id",
    "soma_joinid",
    "task",
    "question",
    "target",
    "target_ontology_term_id",
    "answer",
    "shortcut_answer",
    "is_shortcut_conflict",
    "prior_name",
    "prior_family",
    "shortcut_features",
    "prior_support",
    "prior_mode_count",
    "prior_purity",
    "prior_n_unique_targets",
    "decorrelation_group_id",
    "decorrelation_n_labels",
    "decorrelation_sample_per_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-labels-per-key",
        type=int,
        default=2,
        help="Minimum rows per target label inside a shortcut-feature stratum.",
    )
    parser.add_argument(
        "--per-label-per-key",
        type=int,
        default=50,
        help="Maximum rows sampled for each target label inside a stratum.",
    )
    parser.add_argument(
        "--max-rows-per-prior",
        type=int,
        default=20000,
        help="Cap rows per prior while preserving whole balanced strata. 0 means no cap.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if suffixes.endswith(".jsonl.gz"):
        return pd.read_json(path, lines=True)
    if suffixes.endswith(".csv.gz") or path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported manifest format: {path}")


def clean_value(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def clean_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return clean_value(value).lower() in {"1", "true", "yes"}


def clean_int(value) -> int:
    text = clean_value(value)
    if not text:
        return 0
    return int(float(text))


def clean_float(value) -> float:
    text = clean_value(value)
    if not text:
        return 0.0
    return float(text)


def shortcut_features(row: pd.Series) -> list[str]:
    return [
        feature
        for feature in clean_value(row.get("shortcut_features")).split("|")
        if feature
    ]


def group_id(prior_name: str, features: list[str], values: tuple[str, ...]) -> str:
    feature_text = ",".join(
        f"{feature}={value}" for feature, value in zip(features, values)
    )
    return f"{prior_name}::{feature_text}"


def sample_prior(
    prior_frame: pd.DataFrame,
    min_labels_per_key: int,
    per_label_per_key: int,
    max_rows_per_prior: int,
    rng: random.Random,
) -> tuple[pd.DataFrame, dict]:
    if prior_frame.empty:
        return prior_frame, {}

    first = prior_frame.iloc[0]
    features = shortcut_features(first)
    if not features:
        return prior_frame.iloc[0:0].copy(), {}

    feature_cols = [f"feature_{feature}" for feature in features]
    missing_cols = [col for col in feature_cols if col not in prior_frame]
    if missing_cols:
        return prior_frame.iloc[0:0].copy(), {}

    frame = prior_frame.copy()
    frame = frame[frame["answer"].astype("string").fillna("").str.strip() != ""]
    for col in feature_cols:
        frame = frame[frame[col].astype("string").fillna("").str.strip() != ""]
    if frame.empty:
        return frame, {}

    units = []
    annotations: dict[int, dict] = {}
    grouped = frame.groupby(feature_cols, dropna=False, sort=True)
    for raw_key, group in grouped:
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        key = tuple(clean_value(value) for value in key)
        label_counts = group["answer"].value_counts()
        eligible_labels = [
            label
            for label, count in label_counts.items()
            if count >= min_labels_per_key
        ]
        if len(eligible_labels) < 2:
            continue

        sample_per_label = min(
            per_label_per_key,
            min(int(label_counts[label]) for label in eligible_labels),
        )
        if sample_per_label < min_labels_per_key:
            continue

        indices = []
        for label in sorted(eligible_labels):
            label_indices = group.index[group["answer"] == label].tolist()
            rng.shuffle(label_indices)
            indices.extend(label_indices[:sample_per_label])

        prior_name = clean_value(first.get("prior_name"))
        unit = {
            "indices": indices,
            "group_id": group_id(prior_name, features, key),
            "n_labels": len(eligible_labels),
            "sample_per_label": sample_per_label,
        }
        units.append(unit)

    rng.shuffle(units)
    selected_indices: list[int] = []
    included_units = []
    for unit in units:
        next_size = len(selected_indices) + len(unit["indices"])
        if max_rows_per_prior and selected_indices and next_size > max_rows_per_prior:
            continue
        selected_indices.extend(unit["indices"])
        included_units.append(unit)
        if max_rows_per_prior and len(selected_indices) >= max_rows_per_prior:
            break

    for unit in included_units:
        for index in unit["indices"]:
            annotations[index] = {
                "decorrelation_group_id": unit["group_id"],
                "decorrelation_n_labels": unit["n_labels"],
                "decorrelation_sample_per_label": unit["sample_per_label"],
            }

    selected = frame.loc[selected_indices].copy()
    for column in [
        "decorrelation_group_id",
        "decorrelation_n_labels",
        "decorrelation_sample_per_label",
    ]:
        selected[column] = [annotations[index][column] for index in selected.index]

    summary = {
        "prior_name": clean_value(first.get("prior_name")),
        "prior_family": clean_value(first.get("prior_family")),
        "task": clean_value(first.get("task")),
        "target": clean_value(first.get("target")),
        "features": "+".join(features),
        "rows": len(selected),
        "groups": len(included_units),
        "unique_answers": int(selected["answer"].nunique()) if not selected.empty else 0,
        "conflict_rows": int(selected["is_shortcut_conflict"].astype(bool).sum()) if not selected.empty else 0,
        "metadata_prior_accuracy": float((selected["answer"] == selected["shortcut_answer"]).mean()) if not selected.empty else 0.0,
        "mean_group_labels": sum(unit["n_labels"] for unit in included_units) / len(included_units) if included_units else 0.0,
        "mean_sample_per_label": sum(unit["sample_per_label"] for unit in included_units) / len(included_units) if included_units else 0.0,
    }
    return selected, summary


def row_to_record(row: pd.Series) -> dict:
    record = {
        "example_id": clean_value(row.get("example_id")),
        "soma_joinid": clean_value(row.get("soma_joinid")),
        "task": clean_value(row.get("task")),
        "question": clean_value(row.get("question")),
        "target": clean_value(row.get("target")),
        "target_ontology_term_id": clean_value(row.get("target_ontology_term_id")),
        "answer": clean_value(row.get("answer")),
        "shortcut_answer": clean_value(row.get("shortcut_answer")),
        "is_shortcut_conflict": clean_bool(row.get("is_shortcut_conflict")),
        "prior_name": clean_value(row.get("prior_name")),
        "prior_family": clean_value(row.get("prior_family")),
        "shortcut_features": shortcut_features(row),
        "prior_support": clean_int(row.get("prior_support")),
        "prior_mode_count": clean_int(row.get("prior_mode_count")),
        "prior_purity": clean_float(row.get("prior_purity")),
        "prior_n_unique_targets": clean_int(row.get("prior_n_unique_targets")),
        "decorrelation_group_id": clean_value(row.get("decorrelation_group_id")),
        "decorrelation_n_labels": clean_int(row.get("decorrelation_n_labels")),
        "decorrelation_sample_per_label": clean_int(row.get("decorrelation_sample_per_label")),
    }
    record["shortcut_feature_values"] = {
        feature: clean_value(row.get(f"feature_{feature}"))
        for feature in record["shortcut_features"]
    }
    record["metadata"] = {
        column.removeprefix("metadata_"): clean_value(row.get(column))
        for column in row.index
        if column.startswith("metadata_") and clean_value(row.get(column))
    }
    return record


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def write_summary(path: Path, summary_rows: list[dict]) -> None:
    if not summary_rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def write_report(path: Path, summary_rows: list[dict], n_rows: int, args: argparse.Namespace) -> None:
    lines = [
        "# Decorrelated Control Build Report",
        "",
        f"- Manifest: `{args.manifest}`",
        f"- Decorrelated rows: {n_rows:,}",
        f"- Minimum rows per target label/key: {args.min_labels_per_key:,}",
        f"- Maximum rows per target label/key: {args.per_label_per_key:,}",
        f"- Maximum rows per prior: {args.max_rows_per_prior:,}",
        "",
        "| prior | task | features | rows | groups | conflicts | prior_acc | mean_labels | mean_per_label |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["prior_name"]),
                    str(row["task"]),
                    str(row["features"]),
                    f"{int(row['rows']):,}",
                    f"{int(row['groups']):,}",
                    f"{int(row['conflict_rows']):,}",
                    f"{float(row['metadata_prior_accuracy']):.3f}",
                    f"{float(row['mean_group_labels']):.2f}",
                    f"{float(row['mean_sample_per_label']):.1f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Rows are sampled so each included shortcut-feature stratum has at",
            "least two target labels with equal representation. A metadata-only",
            "shortcut should therefore lose accuracy on this control even though",
            "the examples remain real held-out cells.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)

    manifest = read_manifest(Path(args.manifest))
    rng = random.Random(args.seed)

    selected_frames = []
    summary_rows = []
    for _, prior_frame in manifest.groupby("prior_name", sort=True):
        selected, summary = sample_prior(
            prior_frame,
            min_labels_per_key=args.min_labels_per_key,
            per_label_per_key=args.per_label_per_key,
            max_rows_per_prior=args.max_rows_per_prior,
            rng=rng,
        )
        if not selected.empty:
            selected_frames.append(selected)
            summary_rows.append(summary)

    if selected_frames:
        selected = pd.concat(selected_frames, ignore_index=True)
        selected = selected.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    else:
        selected = manifest.iloc[0:0].copy()

    records = [row_to_record(row) for _, row in selected.iterrows()]
    challenge_path = output_dir / "data" / "decorrelated_control_challenge.jsonl"
    write_jsonl(records, challenge_path)
    write_jsonl(
        (
            {"example_id": record["example_id"], "prediction": record["shortcut_answer"]}
            for record in records
        ),
        output_dir / "predictions" / "metadata_prior_baseline_decorrelated.jsonl",
    )
    write_summary(output_dir / "reports" / "decorrelated_control_summary.csv", summary_rows)
    write_report(
        output_dir / "reports" / "decorrelated_control_report.md",
        summary_rows,
        len(records),
        args,
    )

    metadata = {
        "manifest": args.manifest,
        "n_decorrelated_records": len(records),
        "min_labels_per_key": args.min_labels_per_key,
        "per_label_per_key": args.per_label_per_key,
        "max_rows_per_prior": args.max_rows_per_prior,
        "challenge_path": challenge_path.as_posix(),
        "summary_path": (output_dir / "reports" / "decorrelated_control_summary.csv").as_posix(),
    }
    (output_dir / "decorrelated_control_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Wrote decorrelated control records: {len(records):,}")
    print(f"Wrote {challenge_path}")


if __name__ == "__main__":
    main()
