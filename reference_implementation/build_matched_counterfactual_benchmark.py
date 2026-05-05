#!/usr/bin/env python
"""Build matched counterfactual metadata benchmarks.

These examples isolate shortcut mechanisms by holding one context fixed while
the target varies. Example: same cell type, multiple tissues; same tissue,
multiple diseases. The output is metadata-only and can be joined to expression
by `soma_joinid` and source file.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


SPECS = [
    {
        "name": "same_celltype_different_tissue",
        "task": "tissue_general_prediction",
        "anchor": ("cell_type",),
        "target": "tissue_general",
        "question": "Predict the broad tissue from expression for this cell type context.",
    },
    {
        "name": "same_celltype_different_disease",
        "task": "disease_prediction",
        "anchor": ("cell_type",),
        "target": "disease",
        "question": "Predict disease state from expression for this cell type context.",
    },
    {
        "name": "same_tissue_different_disease",
        "task": "disease_prediction",
        "anchor": ("tissue_general",),
        "target": "disease",
        "question": "Predict disease state from expression for this tissue context.",
    },
    {
        "name": "same_disease_different_tissue",
        "task": "tissue_general_prediction",
        "anchor": ("disease",),
        "target": "tissue_general",
        "question": "Predict tissue from expression for this disease context.",
    },
    {
        "name": "same_assay_different_disease",
        "task": "disease_prediction",
        "anchor": ("assay",),
        "target": "disease",
        "question": "Predict disease state from expression under a fixed assay context.",
    },
    {
        "name": "same_assay_different_tissue",
        "task": "tissue_general_prediction",
        "anchor": ("assay",),
        "target": "tissue_general",
        "question": "Predict tissue from expression under a fixed assay context.",
    },
    {
        "name": "same_celltype_tissue_different_disease",
        "task": "disease_prediction",
        "anchor": ("cell_type", "tissue_general"),
        "target": "disease",
        "question": "Predict disease when cell type and tissue are fixed.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="investigations/shortcut_bias_20260421/benchmark/source_heldout_v1/source_heldout_manifest.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="investigations/shortcut_bias_20260421/benchmark/matched_counterfactual_v1",
    )
    parser.add_argument("--split", default="eval_source_heldout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-labels-per-anchor", type=int, default=2)
    parser.add_argument("--min-rows-per-label", type=int, default=2)
    parser.add_argument("--max-rows-per-label", type=int, default=25)
    parser.add_argument("--max-anchors-per-spec", type=int, default=250)
    parser.add_argument("--max-total-rows", type=int, default=50000)
    return parser.parse_args()


def read_manifest(path: Path) -> pd.DataFrame:
    suffix = "".join(path.suffixes)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix.endswith(".csv.gz") or path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    needed = {
        "soma_joinid",
        "dataset_id",
        "donor_id",
        "assay",
        "tissue",
        "tissue_general",
        "cell_type",
        "disease",
        "development_stage",
        "sex",
        "source_file",
    }
    missing = sorted(needed - set(df.columns))
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    out = df.copy()
    for col in needed:
        out[col] = out[col].astype("string").fillna("").str.strip()
        out = out[out[col] != ""]
        out = out[out[col].str.lower() != "unknown"]
    return out.reset_index(drop=True)


def make_anchor_key(row: pd.Series, columns: tuple[str, ...]) -> str:
    return "|".join(f"{col}={row[col]}" for col in columns)


def build_spec(df: pd.DataFrame, spec: dict, args: argparse.Namespace, rng: random.Random) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    target = spec["target"]
    anchor_cols = list(spec["anchor"])

    anchor_units = []
    for anchor_values, group in df.groupby(anchor_cols, dropna=False, sort=False):
        label_counts = group[target].value_counts()
        labels = [label for label, count in label_counts.items() if count >= args.min_rows_per_label]
        if len(labels) < args.min_labels_per_anchor:
            continue
        anchor_values_tuple = anchor_values if isinstance(anchor_values, tuple) else (anchor_values,)
        min_count = min(int(label_counts[label]) for label in labels)
        sample_per_label = min(args.max_rows_per_label, min_count)
        # Favor anchors with more labels and less extreme imbalance.
        anchor_units.append(
            {
                "anchor_values": tuple(str(v) for v in anchor_values_tuple),
                "group": group,
                "labels": sorted(labels),
                "sample_per_label": sample_per_label,
                "n_labels": len(labels),
            }
        )

    anchor_units.sort(key=lambda u: (-u["n_labels"], u["sample_per_label"], u["anchor_values"]))
    anchor_units = anchor_units[: args.max_anchors_per_spec]

    for unit_index, unit in enumerate(anchor_units):
        context_id = f"{spec['name']}::{unit_index:05d}"
        anchor_text = "|".join(f"{col}={val}" for col, val in zip(anchor_cols, unit["anchor_values"]))
        for label in unit["labels"]:
            label_indices = unit["group"].index[unit["group"][target] == label].tolist()
            rng.shuffle(label_indices)
            for idx in label_indices[: unit["sample_per_label"]]:
                row = df.loc[idx]
                example_id = f"{row['soma_joinid']}::{spec['name']}::{spec['task']}"
                rows.append(
                    {
                        "example_id": example_id,
                        "soma_joinid": str(row["soma_joinid"]),
                        "task": spec["task"],
                        "question": spec["question"],
                        "answer": str(row[target]),
                        "target": target,
                        "counterfactual_family": spec["name"],
                        "anchor_features": "|".join(anchor_cols),
                        "anchor_key": anchor_text,
                        "context_id": context_id,
                        "context_n_labels": unit["n_labels"],
                        "context_sample_per_label": unit["sample_per_label"],
                        "dataset_id": str(row["dataset_id"]),
                        "donor_id": str(row["donor_id"]),
                        "assay": str(row["assay"]),
                        "tissue": str(row["tissue"]),
                        "tissue_general": str(row["tissue_general"]),
                        "cell_type": str(row["cell_type"]),
                        "disease": str(row["disease"]),
                        "development_stage": str(row["development_stage"]),
                        "sex": str(row["sex"]),
                        "source_file": str(row["source_file"]),
                    }
                )

    summary = {
        "counterfactual_family": spec["name"],
        "task": spec["task"],
        "anchor_features": "|".join(anchor_cols),
        "target": target,
        "anchors": len(anchor_units),
        "rows": len(rows),
        "unique_answers": len({row["answer"] for row in rows}),
        "mean_labels_per_context": (
            sum(unit["n_labels"] for unit in anchor_units) / len(anchor_units)
            if anchor_units
            else 0.0
        ),
    }
    return rows, summary


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def markdown_table(df: pd.DataFrame) -> str:
    frame = df.copy()
    for col in frame.columns:
        if pd.api.types.is_float_dtype(frame[col]):
            frame[col] = frame[col].map(lambda x: f"{x:.3f}")
    headers = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_manifest(Path(args.input))
    df = df[df["split"] == args.split].copy()
    df = clean_frame(df)

    all_rows: list[dict] = []
    summaries = []
    for spec in SPECS:
        rows, summary = build_spec(df, spec, args, rng)
        all_rows.extend(rows)
        summaries.append(summary)

    rng.shuffle(all_rows)
    if args.max_total_rows and len(all_rows) > args.max_total_rows:
        all_rows = all_rows[: args.max_total_rows]

    write_jsonl(output_dir / "matched_counterfactual_challenge.jsonl", all_rows)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "matched_counterfactual_summary.csv", index=False)

    report = [
        "# Matched Counterfactual Benchmark",
        "",
        f"- Input: `{args.input}`",
        f"- Split: `{args.split}`",
        f"- Rows: {len(all_rows):,}",
        "",
        "## Summary",
        "",
        markdown_table(summary_df),
        "",
        "## Interpretation",
        "",
        "Each context holds the anchor metadata fixed while requiring the target label to vary.",
        "A model that predicts the dominant anchor label instead of the row-specific answer is using a shortcut.",
        "",
    ]
    try:
        text = "\n".join(report)
    except ImportError:
        text = summary_df.to_csv(index=False)
    (output_dir / "MATCHED_COUNTERFACTUAL_REPORT.md").write_text(text)
    print(f"Wrote {len(all_rows):,} rows to {output_dir / 'matched_counterfactual_challenge.jsonl'}")


if __name__ == "__main__":
    main()
