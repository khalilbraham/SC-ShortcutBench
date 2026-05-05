#!/usr/bin/env python
"""Build a dataset/source-held-out split manifest from h5ad metadata."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import anndata as ad
import pandas as pd


DEFAULT_COLUMNS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-col", default="dataset_id")
    parser.add_argument("--eval-source-fraction", type=float, default=0.2)
    parser.add_argument("--min-source-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean(value) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_obs(paths: list[str], columns: list[str]) -> pd.DataFrame:
    frames = []
    for path_text in paths:
        path = Path(path_text)
        adata = ad.read_h5ad(path, backed="r")
        try:
            present = [col for col in columns if col in adata.obs.columns]
            obs = adata.obs[present].copy()
        finally:
            adata.file.close()
        for col in columns:
            if col not in obs.columns:
                obs[col] = ""
            obs[col] = obs[col].map(clean)
        obs["source_file"] = f"{path.parent.name}/{path.name}"
        frames.append(obs[columns + ["source_file"]])
    return pd.concat(frames, ignore_index=True)


def label_counts(frame: pd.DataFrame, column: str, top_n: int = 12) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return dict(Counter(value for value in frame[column] if value).most_common(top_n))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = list(DEFAULT_COLUMNS)
    if args.source_col not in columns:
        columns.append(args.source_col)
    obs = load_obs(args.h5ad, columns)
    obs = obs[obs[args.source_col] != ""].copy()

    source_counts = obs[args.source_col].value_counts()
    eligible_sources = [
        source
        for source, count in source_counts.items()
        if count >= args.min_source_rows
    ]
    rng.shuffle(eligible_sources)
    n_eval_sources = max(1, int(round(len(eligible_sources) * args.eval_source_fraction)))
    eval_sources = set(eligible_sources[:n_eval_sources])

    obs["split"] = obs[args.source_col].map(
        lambda source: "eval_source_heldout" if source in eval_sources else "train_source"
    )

    manifest_path = output_dir / "source_heldout_manifest.csv.gz"
    obs.to_csv(manifest_path, index=False)

    train = obs[obs["split"] == "train_source"]
    eval_frame = obs[obs["split"] == "eval_source_heldout"]
    overlap = set(train[args.source_col]) & set(eval_frame[args.source_col])
    rows = [
        {
            "split": "train_source",
            "rows": len(train),
            "sources": train[args.source_col].nunique(),
            "donors": train["donor_id"].nunique() if "donor_id" in train else 0,
            "diseases": train["disease"].nunique() if "disease" in train else 0,
            "tissues": train["tissue_general"].nunique() if "tissue_general" in train else 0,
            "cell_types": train["cell_type"].nunique() if "cell_type" in train else 0,
        },
        {
            "split": "eval_source_heldout",
            "rows": len(eval_frame),
            "sources": eval_frame[args.source_col].nunique(),
            "donors": eval_frame["donor_id"].nunique() if "donor_id" in eval_frame else 0,
            "diseases": eval_frame["disease"].nunique() if "disease" in eval_frame else 0,
            "tissues": eval_frame["tissue_general"].nunique() if "tissue_general" in eval_frame else 0,
            "cell_types": eval_frame["cell_type"].nunique() if "cell_type" in eval_frame else 0,
        },
    ]
    summary_csv = output_dir / "source_heldout_summary.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    details = {
        "source_col": args.source_col,
        "seed": args.seed,
        "eval_source_fraction": args.eval_source_fraction,
        "min_source_rows": args.min_source_rows,
        "n_total_rows": len(obs),
        "n_total_sources": obs[args.source_col].nunique(),
        "n_eligible_sources": len(eligible_sources),
        "n_eval_sources": len(eval_sources),
        "source_overlap": sorted(overlap),
        "manifest": str(manifest_path),
        "summary_csv": str(summary_csv),
        "train_top_disease": label_counts(train, "disease"),
        "eval_top_disease": label_counts(eval_frame, "disease"),
        "train_top_tissue_general": label_counts(train, "tissue_general"),
        "eval_top_tissue_general": label_counts(eval_frame, "tissue_general"),
    }
    (output_dir / "source_heldout_metadata.json").write_text(json.dumps(details, indent=2))

    lines = [
        "# Source-Held-Out Split",
        "",
        f"- Source column: `{args.source_col}`",
        f"- Total rows: {len(obs):,}",
        f"- Total sources: {obs[args.source_col].nunique():,}",
        f"- Eligible sources: {len(eligible_sources):,}",
        f"- Eval held-out sources: {len(eval_sources):,}",
        f"- Source overlap between train/eval: {len(overlap)}",
        "",
        "| split | rows | sources | donors | diseases | tissues | cell types |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['rows']:,} | {row['sources']:,} | {row['donors']:,} | "
            f"{row['diseases']:,} | {row['tissues']:,} | {row['cell_types']:,} |"
        )
    lines.extend(
        [
            "",
            "## Why This Split Matters",
            "",
            "A source-held-out split prevents direct memorization of dataset IDs across train and eval.",
            "It should be used together with tissue/disease/cell-type balancing, because source holdout alone does not remove every biological or technical prior.",
            "",
            f"- Manifest: `{manifest_path}`",
            f"- Summary CSV: `{summary_csv}`",
        ]
    )
    report_path = output_dir / "source_heldout_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
