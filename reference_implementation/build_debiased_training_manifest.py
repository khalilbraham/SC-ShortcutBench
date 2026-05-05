#!/usr/bin/env python
"""Build a decorrelated CELLxGENE training manifest.

The output is a metadata manifest for training cell-to-text or annotation models
with reduced construction shortcuts. It does not change expression matrices; it
selects a subset of rows whose source, assay, cell-type, tissue, and disease
pairs are capped so no single shortcut pair dominates training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PAIR_CAPS = [
    "dataset_id:disease",
    "dataset_id:tissue_general",
    "dataset_id:cell_type",
    "assay:disease",
    "assay:tissue_general",
    "cell_type:tissue_general",
    "cell_type:disease",
    "tissue_general:disease",
    "donor_id:disease",
]

DEFAULT_AUDIT_PRIORS = [
    "dataset_id:disease",
    "dataset_id:tissue_general",
    "dataset_id:assay",
    "dataset_id+cell_type:tissue_general",
    "dataset_id+assay:disease",
    "cell_type:tissue_general",
    "cell_type:disease",
    "tissue_general:disease",
    "assay:disease",
    "assay:tissue_general",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="investigations/shortcut_bias_20260421/benchmark/source_heldout_v1/source_heldout_manifest.csv.gz",
        help="CSV/CSV.GZ/Parquet manifest containing metadata rows.",
    )
    parser.add_argument(
        "--output-dir",
        default="investigations/shortcut_bias_20260421/benchmark/debiased_training_v1",
    )
    parser.add_argument("--train-split", default="train_source")
    parser.add_argument("--eval-split", default="eval_source_heldout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-total-rows", type=int, default=200000)
    parser.add_argument("--max-per-source", type=int, default=4000)
    parser.add_argument("--max-per-donor", type=int, default=300)
    parser.add_argument("--max-per-target-label", type=int, default=15000)
    parser.add_argument("--max-per-pair", type=int, default=300)
    parser.add_argument(
        "--audit-max-rows",
        type=int,
        default=100000,
        help="Maximum rows sampled from before/after manifests for bias-audit metrics. 0 means all rows.",
    )
    parser.add_argument(
        "--pair-cap",
        action="append",
        default=[],
        help="Optional pair cap in the form column_a:column_b:max_rows. May repeat.",
    )
    parser.add_argument(
        "--required-columns",
        default="dataset_id,donor_id,assay,tissue_general,cell_type,disease",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> pd.DataFrame:
    suffix = "".join(path.suffixes)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix.endswith(".csv.gz") or path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported manifest format: {path}")


def clean_frame(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in required_columns:
        if col not in out.columns:
            raise KeyError(f"Missing required column: {col}")
        out[col] = out[col].astype("string").fillna("").str.strip()
        out = out[out[col] != ""]
        out = out[out[col].str.lower() != "unknown"]
    return out.reset_index(drop=True)


def parse_pair_caps(args: argparse.Namespace) -> dict[tuple[str, str], int]:
    caps = {}
    for spec in DEFAULT_PAIR_CAPS:
        a, b = spec.split(":")
        caps[(a, b)] = args.max_per_pair
    for spec in args.pair_cap:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad --pair-cap {spec!r}; expected column_a:column_b:max_rows")
        a, b, cap = parts
        caps[(a, b)] = int(cap)
    return caps


def add_sampling_priority(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    tuple_cols = ["cell_type", "tissue_general", "disease"]
    stratum_counts = out.groupby(tuple_cols, dropna=False).size().rename("stratum_count")
    out = out.merge(stratum_counts.reset_index(), on=tuple_cols, how="left")
    source_counts = out.groupby("dataset_id", dropna=False).size().rename("source_count")
    out = out.merge(source_counts.reset_index(), on="dataset_id", how="left")
    out["_rand"] = rng.random(len(out))
    out = out.sort_values(["stratum_count", "source_count", "_rand"], ascending=[True, True, True])
    return out.reset_index(drop=True)


def cap_group(frame: pd.DataFrame, columns: list[str], cap: int) -> pd.DataFrame:
    if cap <= 0 or frame.empty:
        return frame
    return frame.groupby(columns, dropna=False, sort=False, group_keys=False).head(cap)


def build_sample(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    pair_caps = parse_pair_caps(args)
    selected = add_sampling_priority(df, args.seed)

    selected = cap_group(selected, ["dataset_id"], args.max_per_source)
    selected = cap_group(selected, ["donor_id"], args.max_per_donor)
    for target in ("disease", "tissue_general", "cell_type"):
        selected = cap_group(selected, [target], args.max_per_target_label)
    for pair, cap in pair_caps.items():
        selected = cap_group(selected, list(pair), cap)

    if args.max_total_rows and len(selected) > args.max_total_rows:
        selected = selected.head(args.max_total_rows)

    selected = selected.drop(columns=["stratum_count", "source_count", "_rand"])
    return selected.reset_index(drop=True)


def parse_prior(spec: str) -> tuple[tuple[str, ...], str]:
    features, target = spec.split(":")
    return tuple(features.split("+")), target


def weighted_mode_accuracy(df: pd.DataFrame, features: tuple[str, ...], target: str) -> float:
    if df.empty:
        return math.nan
    if features:
        counts = df.groupby(list(features) + [target], dropna=False).size().rename("n").reset_index()
        max_counts = counts.groupby(list(features), dropna=False)["n"].max()
        return float(max_counts.sum() / len(df))
    return float(df[target].value_counts(normalize=True).iloc[0])


def cramers_v(df: pd.DataFrame, features: tuple[str, ...], target: str) -> float:
    if df.empty:
        return math.nan
    if features:
        left = df[list(features)].astype(str).agg("|".join, axis=1)
    else:
        left = pd.Series(["GLOBAL"] * len(df), index=df.index)
    table = pd.crosstab(left, df[target])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n
    mask = expected > 0
    chi2 = (((observed - expected) ** 2) / np.where(mask, expected, 1.0))[mask].sum()
    phi2 = chi2 / n
    denom = min(table.shape[0] - 1, table.shape[1] - 1)
    if denom <= 0:
        return 0.0
    return float(math.sqrt(phi2 / denom))


def audit_frame(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows and len(frame) > max_rows:
        return frame.sample(n=max_rows, random_state=seed)
    return frame


def audit_bias(before: pd.DataFrame, after: pd.DataFrame, max_rows: int = 100000, seed: int = 42) -> pd.DataFrame:
    before_audit = audit_frame(before, max_rows, seed)
    after_audit = audit_frame(after, max_rows, seed)
    rows = []
    for spec in DEFAULT_AUDIT_PRIORS:
        features, target = parse_prior(spec)
        for name, frame in (("before", before_audit), ("after", after_audit)):
            n_labels = max(int(frame[target].nunique()), 1)
            chance = 1.0 / n_labels
            mode_acc = weighted_mode_accuracy(frame, features, target)
            normalized = max(0.0, (mode_acc - chance) / (1.0 - chance)) if n_labels > 1 else 0.0
            rows.append(
                {
                    "stage": name,
                    "prior": spec,
                    "features": "+".join(features) if features else "GLOBAL",
                    "target": target,
                    "rows": len(frame),
                    "n_target_labels": n_labels,
                    "mode_accuracy": mode_acc,
                    "normalized_mode_advantage": normalized,
                    "cramers_v": cramers_v(frame, features, target),
                }
            )
    metrics = pd.DataFrame(rows)
    pivot = metrics.pivot(index="prior", columns="stage", values=["mode_accuracy", "normalized_mode_advantage", "cramers_v"])
    flat = []
    for prior in pivot.index:
        row = {"prior": prior}
        for metric in ("mode_accuracy", "normalized_mode_advantage", "cramers_v"):
            before_value = float(pivot.loc[prior, (metric, "before")])
            after_value = float(pivot.loc[prior, (metric, "after")])
            row[f"{metric}_before"] = before_value
            row[f"{metric}_after"] = after_value
            row[f"{metric}_reduction"] = before_value - after_value
        flat.append(row)
    return pd.DataFrame(flat).sort_values("normalized_mode_advantage_before", ascending=False)


def write_report(output_dir: Path, before: pd.DataFrame, after: pd.DataFrame, audit: pd.DataFrame, args: argparse.Namespace) -> None:
    def md_table(df: pd.DataFrame) -> str:
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

    retention = len(after) / len(before) if len(before) else 0.0
    lines = [
        "# Debiased Training Manifest",
        "",
        "This manifest is a preprocessing output for training on a decorrelated CELLxGENE subset.",
        "It caps dominant source, donor, assay, cell-type, tissue, and disease shortcut pairs.",
        "",
        "## Configuration",
        "",
        f"- Input: `{args.input}`",
        f"- Train split: `{args.train_split}`",
        f"- Seed: `{args.seed}`",
        f"- Max total rows: {args.max_total_rows:,}",
        f"- Max per source: {args.max_per_source:,}",
        f"- Max per donor: {args.max_per_donor:,}",
        f"- Max per target label: {args.max_per_target_label:,}",
        f"- Default max per protected pair: {args.max_per_pair:,}",
        "",
        "## Output Size",
        "",
        f"- Candidate train rows after missing-value filtering: {len(before):,}",
        f"- Selected decorrelated train rows: {len(after):,}",
        f"- Retention: {retention:.3f}",
        "",
        "## Remaining Bias Audit",
        "",
        md_table(audit),
        "",
        "## Recommended Training Use",
        "",
        "Use `debiased_training_manifest.csv.gz` as the row-selection file for the next Cell2Text/CELLxGENE training run.",
        "Join expression by `soma_joinid` and source file, then generate descriptions only from fields that are part of the intended prediction target.",
        "Do not include `dataset_id`, donor identifiers, assay, or the target label itself in generated text unless the task explicitly requires them.",
        "",
        "## Pipeline",
        "",
        "1. Start from a source-held-out manifest, so evaluation sources never appear in training.",
        "2. Drop rows missing the core biological and technical metadata needed for balancing.",
        "3. Prioritize rare `(cell_type, tissue_general, disease)` strata during sampling.",
        "4. Enforce caps on source, donor, target-label, and protected-pair counts.",
        "5. Re-audit the selected manifest with mode-accuracy, normalized mode advantage, and Cramer's V.",
        "6. Train on the selected manifest and evaluate on shortcut-conflict, decorrelated-control, and source-held-out challenges.",
        "",
    ]
    (output_dir / "DEBIASED_TRAINING_PIPELINE.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_columns = [col.strip() for col in args.required_columns.split(",") if col.strip()]
    manifest = read_manifest(Path(args.input))
    train = manifest[manifest["split"] == args.train_split].copy()
    train = clean_frame(train, required_columns)
    selected = build_sample(train, args)
    audit = audit_bias(train, selected, max_rows=args.audit_max_rows, seed=args.seed)

    selected.to_csv(output_dir / "debiased_training_manifest.csv.gz", index=False, compression="gzip")
    audit.to_csv(output_dir / "debiased_training_bias_audit.csv", index=False)
    metadata = {
        "input": args.input,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "seed": args.seed,
        "candidate_train_rows": int(len(train)),
        "selected_rows": int(len(selected)),
        "retention": float(len(selected) / len(train) if len(train) else 0.0),
        "max_total_rows": args.max_total_rows,
        "max_per_source": args.max_per_source,
        "max_per_donor": args.max_per_donor,
        "max_per_target_label": args.max_per_target_label,
        "max_per_pair": args.max_per_pair,
        "audit_max_rows": args.audit_max_rows,
        "pair_caps": {"+".join(pair): cap for pair, cap in parse_pair_caps(args).items()},
    }
    (output_dir / "debiased_training_metadata.json").write_text(json.dumps(metadata, indent=2))
    write_report(output_dir, train, selected, audit, args)

    print(f"Selected {len(selected):,} rows from {len(train):,} candidate train rows")
    print(f"Wrote {output_dir / 'debiased_training_manifest.csv.gz'}")
    print(f"Wrote {output_dir / 'DEBIASED_TRAINING_PIPELINE.md'}")


if __name__ == "__main__":
    main()
