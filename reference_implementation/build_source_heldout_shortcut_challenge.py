#!/usr/bin/env python
"""Build source-heldout shortcut challenge from source-heldout manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


PRIORS = [
    ("celltype_to_tissue", "tissue_general_prediction", "tissue_general", ("cell_type",), "biological_metadata"),
    ("celltype_to_disease", "disease_prediction", "disease", ("cell_type",), "biological_metadata"),
    ("tissue_general_to_disease", "disease_prediction", "disease", ("tissue_general",), "biological_metadata"),
    ("tissue_to_disease", "disease_prediction", "disease", ("tissue",), "biological_metadata"),
    ("disease_to_tissue", "tissue_general_prediction", "tissue_general", ("disease",), "biological_metadata"),
    ("assay_to_tissue", "tissue_general_prediction", "tissue_general", ("assay",), "technical_metadata"),
    ("assay_to_disease", "disease_prediction", "disease", ("assay",), "technical_metadata"),
    (
        "combined_celltype_tissue_to_disease",
        "disease_prediction",
        "disease",
        ("cell_type", "tissue_general"),
        "combined_metadata",
    ),
    (
        "combined_celltype_disease_to_tissue",
        "tissue_general_prediction",
        "tissue_general",
        ("cell_type", "disease"),
        "combined_metadata",
    ),
    (
        "combined_tissue_disease_to_celltype",
        "cell_type_prediction",
        "cell_type",
        ("tissue_general", "disease"),
        "combined_metadata",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="investigations/shortcut_bias_20260421/benchmark/source_heldout_v1/source_heldout_manifest.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="investigations/shortcut_bias_20260421/benchmark/source_heldout_challenge_v1",
    )
    parser.add_argument("--train-split", default="train_source")
    parser.add_argument("--eval-split", default="eval_source_heldout")
    parser.add_argument("--min-support", type=int, default=25)
    parser.add_argument("--min-purity", type=float, default=0.60)
    parser.add_argument("--balanced-per-prior-bucket", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_manifest(path: Path) -> pd.DataFrame:
    suffix = "".join(path.suffixes)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix.endswith(".csv.gz") or path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["soma_joinid", "dataset_id", "donor_id", "assay", "tissue", "tissue_general", "cell_type", "disease", "source_file"]:
        out[col] = out[col].astype("string").fillna("").str.strip()
    return out


def key_from_row(row: pd.Series, features: tuple[str, ...]) -> tuple[str, ...]:
    if not features:
        return ("GLOBAL",)
    return tuple(str(row[f]) for f in features)


def learn_prior(train: pd.DataFrame, features: tuple[str, ...], target: str, min_support: int, min_purity: float) -> dict[tuple[str, ...], dict]:
    needed = list(features) + [target]
    frame = train.dropna(subset=needed).copy()
    for col in needed:
        frame = frame[frame[col].astype("string").fillna("").str.strip() != ""]
    group_cols = list(features) + [target] if features else [target]
    counts_df = frame.groupby(group_cols, dropna=False, sort=False).size().rename("n").reset_index()
    if features:
        key_cols = list(features)
        total = counts_df.groupby(key_cols, dropna=False)["n"].sum().rename("support")
        idx = counts_df.groupby(key_cols, dropna=False)["n"].idxmax()
        modes = counts_df.loc[idx].copy()
        modes = modes.merge(total.reset_index(), on=key_cols, how="left")
        unique_targets = counts_df.groupby(key_cols, dropna=False)[target].nunique().rename("n_unique_targets")
        modes = modes.merge(unique_targets.reset_index(), on=key_cols, how="left")
    else:
        support = int(counts_df["n"].sum())
        mode_row = counts_df.sort_values("n", ascending=False).iloc[0]
        modes = pd.DataFrame(
            [
                {
                    target: mode_row[target],
                    "n": int(mode_row["n"]),
                    "support": support,
                    "n_unique_targets": int(counts_df[target].nunique()),
                }
            ]
        )
    priors = {}
    for _, row in modes.iterrows():
        support = int(row["support"])
        answer = str(row[target])
        mode_count = int(row["n"])
        purity = mode_count / support if support else 0.0
        if support >= min_support and purity >= min_purity:
            key = tuple(str(row[f]) for f in features) if features else ("GLOBAL",)
            priors[key] = {
                "shortcut_answer": answer,
                "support": support,
                "mode_count": mode_count,
                "purity": purity,
                "n_unique_targets": int(row["n_unique_targets"]),
            }
    return priors


def build_examples(train: pd.DataFrame, eval_df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_rows = []
    for prior_name, task, target, features, family in PRIORS:
        priors = learn_prior(train, features, target, args.min_support, args.min_purity)
        covered = 0
        conflict = 0
        aligned = 0
        for _, row in eval_df.iterrows():
            if any(str(row.get(f, "")).strip() == "" for f in features + (target,)):
                continue
            key = key_from_row(row, features)
            prior = priors.get(key)
            if not prior:
                continue
            answer = str(row[target])
            shortcut = prior["shortcut_answer"]
            is_conflict = answer != shortcut
            covered += 1
            conflict += int(is_conflict)
            aligned += int(not is_conflict)
            rows.append(
                {
                    "example_id": f"{row['soma_joinid']}::{prior_name}::{task}",
                    "soma_joinid": str(row["soma_joinid"]),
                    "task": task,
                    "question": f"Predict {target} from expression under source-heldout evaluation.",
                    "target": target,
                    "answer": answer,
                    "shortcut_answer": shortcut,
                    "is_shortcut_conflict": bool(is_conflict),
                    "prior_name": prior_name,
                    "prior_family": family,
                    "shortcut_features": "|".join(features),
                    "prior_support": prior["support"],
                    "prior_mode_count": prior["mode_count"],
                    "prior_purity": prior["purity"],
                    "prior_n_unique_targets": prior["n_unique_targets"],
                    "dataset_id": str(row["dataset_id"]),
                    "donor_id": str(row["donor_id"]),
                    "assay": str(row["assay"]),
                    "tissue": str(row["tissue"]),
                    "tissue_general": str(row["tissue_general"]),
                    "cell_type": str(row["cell_type"]),
                    "disease": str(row["disease"]),
                    "source_file": str(row["source_file"]),
                }
            )

        accuracy = aligned / covered if covered else 0.0
        summary_rows.append(
            {
                "prior_name": prior_name,
                "family": family,
                "task": task,
                "target": target,
                "features": "|".join(features),
                "usable_keys": len(priors),
                "eval_rows": len(eval_df),
                "covered_rows": covered,
                "coverage": covered / len(eval_df) if len(eval_df) else 0.0,
                "metadata_prior_accuracy": accuracy,
                "conflict_rows": conflict,
                "conflict_rate": conflict / covered if covered else 0.0,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


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


def balanced_sample(df: pd.DataFrame, per_bucket: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    indices = []
    for (_, conflict), group in df.groupby(["prior_name", "is_shortcut_conflict"], dropna=False, sort=True):
        group_indices = group.index.tolist()
        rng.shuffle(group_indices)
        indices.extend(group_indices[:per_bucket])
    rng.shuffle(indices)
    return df.loc[indices].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = clean_frame(read_manifest(Path(args.input)))
    train = manifest[manifest["split"] == args.train_split].copy()
    eval_df = manifest[manifest["split"] == args.eval_split].copy()

    all_df, summary = build_examples(train, eval_df, args)
    balanced = balanced_sample(all_df, args.balanced_per_prior_bucket, args.seed)

    all_df.to_parquet(output_dir / "source_heldout_shortcut_manifest.parquet", index=False)
    write_jsonl(output_dir / "source_heldout_shortcut_challenge.jsonl", balanced.to_dict("records"))
    summary.to_csv(output_dir / "source_heldout_shortcut_summary.csv", index=False)

    report = [
        "# Source-Heldout Shortcut Challenge",
        "",
        f"- Train split: `{args.train_split}`",
        f"- Eval split: `{args.eval_split}`",
        f"- Full manifest rows: {len(all_df):,}",
        f"- Balanced challenge rows: {len(balanced):,}",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Interpretation",
        "",
        "This challenge evaluates biological and technical shortcuts under zero source overlap.",
        "Source priors based directly on dataset_id are intentionally excluded because eval dataset IDs are unseen.",
        "",
    ]
    (output_dir / "SOURCE_HELDOUT_CHALLENGE_REPORT.md").write_text("\n".join(report))
    print(f"Wrote {len(balanced):,} balanced rows to {output_dir / 'source_heldout_shortcut_challenge.jsonl'}")


if __name__ == "__main__":
    main()
