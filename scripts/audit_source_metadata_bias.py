#!/usr/bin/env python
"""Audit source-level shortcut priors in single-cell metadata.

This script learns simple metadata priors on a training split and evaluates
them on a held-out split. The goal is to quantify construction shortcuts such
as `dataset_id -> disease`, `donor_id -> tissue`, `assay -> cell_type`, and
combined source priors that can make text or retrieval tasks look like
reasoning while the answer is largely recoverable from acquisition metadata.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import pandas as pd


MISSING_STRINGS = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "unknown",
    "unannotated",
    "not applicable",
}


@dataclass(frozen=True)
class PriorSpec:
    prior_name: str
    target: str
    features: tuple[str, ...]
    family: str
    hypothesis: str


PRIOR_SPECS = [
    PriorSpec(
        "dataset_id_to_disease",
        "disease",
        ("dataset_id",),
        "source_dataset",
        "A source study/dataset can reveal the disease label.",
    ),
    PriorSpec(
        "dataset_id_to_tissue_general",
        "tissue_general",
        ("dataset_id",),
        "source_dataset",
        "A source study/dataset can reveal broad tissue.",
    ),
    PriorSpec(
        "dataset_id_to_tissue",
        "tissue",
        ("dataset_id",),
        "source_dataset",
        "A source study/dataset can reveal exact tissue.",
    ),
    PriorSpec(
        "dataset_id_to_cell_type",
        "cell_type",
        ("dataset_id",),
        "source_dataset",
        "A source study/dataset can reveal overrepresented cell types.",
    ),
    PriorSpec(
        "dataset_id_to_assay",
        "assay",
        ("dataset_id",),
        "source_dataset",
        "A source study/dataset can reveal technical protocol.",
    ),
    PriorSpec(
        "source_file_to_disease",
        "disease",
        ("source_file",),
        "split_artifact",
        "A prepared h5ad shard can reveal disease due to split/sharding structure.",
    ),
    PriorSpec(
        "source_file_to_tissue_general",
        "tissue_general",
        ("source_file",),
        "split_artifact",
        "A prepared h5ad shard can reveal tissue due to split/sharding structure.",
    ),
    PriorSpec(
        "source_file_to_cell_type",
        "cell_type",
        ("source_file",),
        "split_artifact",
        "A prepared h5ad shard can reveal cell type due to split/sharding structure.",
    ),
    PriorSpec(
        "donor_id_to_disease",
        "disease",
        ("donor_id",),
        "donor",
        "A donor identifier can leak cohort or disease state.",
    ),
    PriorSpec(
        "donor_id_to_tissue_general",
        "tissue_general",
        ("donor_id",),
        "donor",
        "A donor identifier can leak sampled tissue.",
    ),
    PriorSpec(
        "donor_id_to_cell_type",
        "cell_type",
        ("donor_id",),
        "donor",
        "A donor identifier can leak overrepresented cell types.",
    ),
    PriorSpec(
        "assay_to_disease",
        "disease",
        ("assay",),
        "technical",
        "Sequencing protocol can be confounded with disease cohorts.",
    ),
    PriorSpec(
        "assay_to_tissue_general",
        "tissue_general",
        ("assay",),
        "technical",
        "Sequencing protocol can be confounded with tissue.",
    ),
    PriorSpec(
        "assay_to_cell_type",
        "cell_type",
        ("assay",),
        "technical",
        "Sequencing protocol can be confounded with cell type.",
    ),
    PriorSpec(
        "assay_ontology_to_disease",
        "disease",
        ("assay_ontology_term_id",),
        "technical",
        "Assay ontology can leak disease cohorts.",
    ),
    PriorSpec(
        "assay_ontology_to_tissue_general",
        "tissue_general",
        ("assay_ontology_term_id",),
        "technical",
        "Assay ontology can leak tissue.",
    ),
    PriorSpec(
        "tissue_general_to_disease",
        "disease",
        ("tissue_general",),
        "biological_context",
        "Broad tissue can reveal disease due to cohort construction.",
    ),
    PriorSpec(
        "tissue_to_disease",
        "disease",
        ("tissue",),
        "biological_context",
        "Exact tissue can reveal disease due to cohort construction.",
    ),
    PriorSpec(
        "cell_type_to_disease",
        "disease",
        ("cell_type",),
        "biological_context",
        "Cell type can reveal disease due to label co-occurrence.",
    ),
    PriorSpec(
        "cell_type_to_tissue_general",
        "tissue_general",
        ("cell_type",),
        "biological_context",
        "Cell type can reveal tissue without expression reasoning.",
    ),
    PriorSpec(
        "tissue_general_to_cell_type",
        "cell_type",
        ("tissue_general",),
        "biological_context",
        "Tissue can reveal common cell types.",
    ),
    PriorSpec(
        "tissue_general_to_assay",
        "assay",
        ("tissue_general",),
        "technical_context",
        "Tissue can reveal the likely assay/protocol.",
    ),
    PriorSpec(
        "cell_type_to_assay",
        "assay",
        ("cell_type",),
        "technical_context",
        "Cell type can reveal likely technical protocol.",
    ),
    PriorSpec(
        "development_stage_to_disease",
        "disease",
        ("development_stage",),
        "demographic_context",
        "Developmental stage can be confounded with disease or normal state.",
    ),
    PriorSpec(
        "development_stage_to_tissue_general",
        "tissue_general",
        ("development_stage",),
        "demographic_context",
        "Developmental stage can be confounded with sampled tissue.",
    ),
    PriorSpec(
        "sex_to_disease",
        "disease",
        ("sex",),
        "demographic_context",
        "Sex can be confounded with disease cohorts.",
    ),
    PriorSpec(
        "dataset_id_tissue_general_to_disease",
        "disease",
        ("dataset_id", "tissue_general"),
        "source_plus_context",
        "Study plus tissue can reveal disease state.",
    ),
    PriorSpec(
        "dataset_id_cell_type_to_disease",
        "disease",
        ("dataset_id", "cell_type"),
        "source_plus_context",
        "Study plus cell type can reveal disease state.",
    ),
    PriorSpec(
        "dataset_id_assay_to_disease",
        "disease",
        ("dataset_id", "assay"),
        "source_plus_context",
        "Study plus assay can reveal disease state.",
    ),
    PriorSpec(
        "dataset_id_tissue_general_to_cell_type",
        "cell_type",
        ("dataset_id", "tissue_general"),
        "source_plus_context",
        "Study plus tissue can reveal cell type.",
    ),
    PriorSpec(
        "dataset_id_cell_type_to_tissue_general",
        "tissue_general",
        ("dataset_id", "cell_type"),
        "source_plus_context",
        "Study plus cell type can reveal tissue.",
    ),
    PriorSpec(
        "assay_tissue_general_to_disease",
        "disease",
        ("assay", "tissue_general"),
        "technical_plus_context",
        "Assay plus tissue can reveal disease.",
    ),
    PriorSpec(
        "assay_cell_type_to_disease",
        "disease",
        ("assay", "cell_type"),
        "technical_plus_context",
        "Assay plus cell type can reveal disease.",
    ),
    PriorSpec(
        "cell_type_tissue_general_to_disease",
        "disease",
        ("cell_type", "tissue_general"),
        "biological_combo",
        "Cell type plus tissue can reveal disease.",
    ),
    PriorSpec(
        "donor_id_cell_type_to_disease",
        "disease",
        ("donor_id", "cell_type"),
        "donor_plus_context",
        "Donor plus cell type can reveal disease.",
    ),
    PriorSpec(
        "global_to_disease",
        "disease",
        (),
        "label_imbalance",
        "The global majority label can reveal disease/normal state.",
    ),
    PriorSpec(
        "global_to_tissue_general",
        "tissue_general",
        (),
        "label_imbalance",
        "The global majority label can reveal tissue.",
    ),
    PriorSpec(
        "global_to_cell_type",
        "cell_type",
        (),
        "label_imbalance",
        "The global majority label can reveal cell type.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-h5ad", nargs="+", required=True)
    parser.add_argument("--eval-h5ad", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-support", type=int, default=25)
    parser.add_argument("--min-purity", type=float, default=0.0)
    parser.add_argument(
        "--max-conflicts-per-prior",
        type=int,
        default=2000,
        help="Maximum conflict examples to write per prior. 0 writes none.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=0,
        help="Debug option. 0 means read all rows from each h5ad obs table.",
    )
    return parser.parse_args()


def required_columns(specs: Iterable[PriorSpec]) -> list[str]:
    cols = {"soma_joinid"}
    for spec in specs:
        cols.add(spec.target)
        cols.update(spec.features)
    return sorted(c for c in cols if c != "source_file")


def normalize_value(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in MISSING_STRINGS:
        return None
    return text


def load_obs(paths: list[str], columns: list[str], max_rows_per_file: int) -> pd.DataFrame:
    frames = []
    for path_text in paths:
        path = Path(path_text)
        adata = ad.read_h5ad(path, backed="r")
        try:
            present = [col for col in columns if col in adata.obs.columns]
            obs = adata.obs[present].copy()
        finally:
            adata.file.close()

        if max_rows_per_file and len(obs) > max_rows_per_file:
            obs = obs.iloc[:max_rows_per_file].copy()

        for col in columns:
            if col not in obs.columns:
                obs[col] = None
        obs["source_file"] = f"{path.parent.name}/{path.name}"
        frames.append(obs[columns + ["source_file"]])

    data = pd.concat(frames, ignore_index=True)
    for col in data.columns:
        data[col] = data[col].map(normalize_value)
    return data


def key_columns(spec: PriorSpec) -> list[str]:
    if spec.features:
        return list(spec.features)
    return ["__global_key__"]


def with_key_frame(data: pd.DataFrame, spec: PriorSpec) -> pd.DataFrame:
    cols = list(spec.features) + [spec.target]
    if spec.features:
        work = data[cols].dropna(subset=cols).copy()
    else:
        work = data[[spec.target]].dropna(subset=[spec.target]).copy()
        work["__global_key__"] = "__all__"
    return work


def key_display(row: pd.Series, keys: list[str]) -> str:
    if keys == ["__global_key__"]:
        return "{}"
    return json.dumps({col: row[col] for col in keys}, sort_keys=True)


def learn_prior(
    train: pd.DataFrame,
    spec: PriorSpec,
    min_support: int,
    min_purity: float,
) -> tuple[pd.DataFrame, dict]:
    work = with_key_frame(train, spec)
    keys = key_columns(spec)
    if work.empty:
        return pd.DataFrame(), {"train_rows": 0, "train_keys": 0}

    counts = work.groupby(keys + [spec.target], dropna=False).size().reset_index(name="count")
    totals = counts.groupby(keys, dropna=False)["count"].sum().reset_index(name="support")
    n_targets = counts.groupby(keys, dropna=False)[spec.target].nunique().reset_index(name="n_targets")
    mode_idx = counts.groupby(keys, dropna=False)["count"].idxmax()
    modes = counts.loc[mode_idx].copy()
    modes = modes.rename(columns={spec.target: "shortcut_answer", "count": "mode_count"})
    lookup = modes.merge(totals, on=keys).merge(n_targets, on=keys)
    lookup["purity"] = lookup["mode_count"] / lookup["support"]
    lookup = lookup[(lookup["support"] >= min_support) & (lookup["purity"] >= min_purity)].copy()
    lookup["key_display"] = lookup.apply(lambda row: key_display(row, keys), axis=1)
    stats = {
        "train_rows": int(len(work)),
        "train_keys": int(len(totals)),
        "usable_keys": int(len(lookup)),
        "train_weighted_mode_accuracy": float(modes["mode_count"].sum() / totals["support"].sum()),
    }
    return lookup, stats


def evaluate_prior(eval_data: pd.DataFrame, spec: PriorSpec, lookup: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    work = with_key_frame(eval_data, spec)
    keys = key_columns(spec)
    if work.empty or lookup.empty:
        summary = {
            "eval_rows": int(len(work)),
            "covered_rows": 0,
            "coverage": 0.0,
            "accuracy": None,
            "conflict_rows": 0,
            "conflict_rate": None,
            "mean_key_purity_on_eval": None,
            "mean_key_support_on_eval": None,
            "unique_truth_labels": int(work[spec.target].nunique()) if not work.empty else 0,
            "unique_shortcut_answers": 0,
        }
        return summary, pd.DataFrame(), pd.DataFrame()

    merged = work.merge(
        lookup[keys + ["shortcut_answer", "support", "purity", "n_targets", "key_display"]],
        on=keys,
        how="left",
    )
    covered = merged.dropna(subset=["shortcut_answer"]).copy()
    if covered.empty:
        summary = {
            "eval_rows": int(len(work)),
            "covered_rows": 0,
            "coverage": 0.0,
            "accuracy": None,
            "conflict_rows": 0,
            "conflict_rate": None,
            "mean_key_purity_on_eval": None,
            "mean_key_support_on_eval": None,
            "unique_truth_labels": int(work[spec.target].nunique()),
            "unique_shortcut_answers": 0,
        }
        return summary, covered, pd.DataFrame()

    covered["is_conflict"] = covered[spec.target] != covered["shortcut_answer"]
    summary = {
        "eval_rows": int(len(work)),
        "covered_rows": int(len(covered)),
        "coverage": float(len(covered) / len(work)),
        "accuracy": float((~covered["is_conflict"]).mean()),
        "conflict_rows": int(covered["is_conflict"].sum()),
        "conflict_rate": float(covered["is_conflict"].mean()),
        "mean_key_purity_on_eval": float(covered["purity"].astype(float).mean()),
        "mean_key_support_on_eval": float(covered["support"].astype(float).mean()),
        "unique_truth_labels": int(covered[spec.target].nunique()),
        "unique_shortcut_answers": int(covered["shortcut_answer"].nunique()),
    }

    by_key = (
        covered.groupby("key_display", dropna=False)
        .agg(
            eval_rows=(spec.target, "size"),
            eval_accuracy=("is_conflict", lambda x: float((~x).mean())),
            shortcut_answer=("shortcut_answer", "first"),
            train_support=("support", "first"),
            train_purity=("purity", "first"),
            train_n_targets=("n_targets", "first"),
        )
        .reset_index()
        .sort_values(["eval_rows", "train_purity"], ascending=[False, False])
    )
    return summary, covered, by_key


def format_float(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def write_jsonl_gz(path: Path, rows: Iterable[dict]) -> None:
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    reports_dir = output_dir / "reports"
    data_dir = output_dir / "data"
    reports_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    columns = required_columns(PRIOR_SPECS)
    train = load_obs(args.train_h5ad, columns, args.max_rows_per_file)
    eval_data = load_obs(args.eval_h5ad, columns, args.max_rows_per_file)

    summary_rows = []
    key_rows = []
    conflict_rows = []

    for spec in PRIOR_SPECS:
        missing_cols = [col for col in (list(spec.features) + [spec.target]) if col not in train.columns]
        if missing_cols:
            continue
        lookup, train_stats = learn_prior(train, spec, args.min_support, args.min_purity)
        eval_stats, covered, by_key = evaluate_prior(eval_data, spec, lookup)
        row = {
            "prior_name": spec.prior_name,
            "target": spec.target,
            "features": "+".join(spec.features) if spec.features else "GLOBAL",
            "family": spec.family,
            "hypothesis": spec.hypothesis,
            "min_support": args.min_support,
            "min_purity": args.min_purity,
            **train_stats,
            **eval_stats,
        }
        summary_rows.append(row)

        if not by_key.empty:
            top_keys = by_key.head(25).copy()
            top_keys.insert(0, "prior_name", spec.prior_name)
            top_keys.insert(1, "target", spec.target)
            top_keys.insert(2, "features", "+".join(spec.features) if spec.features else "GLOBAL")
            key_rows.extend(top_keys.to_dict(orient="records"))

        if args.max_conflicts_per_prior and not covered.empty:
            conflicts = covered[covered["is_conflict"]].head(args.max_conflicts_per_prior)
            for _, conflict in conflicts.iterrows():
                payload = {
                    "prior_name": spec.prior_name,
                    "target": spec.target,
                    "features": list(spec.features),
                    "key": conflict["key_display"],
                    "truth": conflict[spec.target],
                    "shortcut_answer": conflict["shortcut_answer"],
                    "train_support": int(conflict["support"]),
                    "train_purity": float(conflict["purity"]),
                }
                for col in [
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
                ]:
                    if col in eval_data.columns and col in conflict.index:
                        payload[col] = conflict[col]
                conflict_rows.append(payload)

    summary_path = reports_dir / "source_bias_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    keys_path = reports_dir / "source_bias_top_keys.csv"
    if key_rows:
        with keys_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(key_rows[0].keys()))
            writer.writeheader()
            writer.writerows(key_rows)

    conflicts_path = data_dir / "source_bias_conflict_examples.jsonl.gz"
    if conflict_rows:
        write_jsonl_gz(conflicts_path, conflict_rows)

    summary_df = pd.DataFrame(summary_rows)
    summary_df["risk_score"] = summary_df["coverage"].fillna(0) * summary_df["accuracy"].fillna(0)
    display_df = summary_df.sort_values(
        ["risk_score", "accuracy", "coverage"], ascending=[False, False, False]
    )

    lines = [
        "# Source Metadata Bias Audit",
        "",
        "This audit learns shortcut priors on training h5ad metadata and evaluates them on held-out h5ad metadata.",
        "",
        f"- Train h5ad files: {len(args.train_h5ad)}",
        f"- Eval h5ad files: {len(args.eval_h5ad)}",
        f"- Train rows loaded: {len(train):,}",
        f"- Eval rows loaded: {len(eval_data):,}",
        f"- Minimum train support per prior key: {args.min_support:,}",
        f"- Minimum train purity per prior key: {args.min_purity:.2f}",
        "",
        "## Highest-Risk Priors",
        "",
        "| prior | family | target | features | coverage | held-out shortcut accuracy | conflict rate | train weighted mode acc | usable keys |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in display_df.head(20).iterrows():
        lines.append(
            f"| {row['prior_name']} | {row['family']} | {row['target']} | {row['features']} | "
            f"{format_float(row['coverage'])} | {format_float(row['accuracy'])} | "
            f"{format_float(row['conflict_rate'])} | {format_float(row['train_weighted_mode_accuracy'])} | "
            f"{int(row['usable_keys'])} |"
        )

    lines.extend(
        [
            "",
            "## All Priors",
            "",
            "| prior | family | target | features | eval rows | covered rows | coverage | accuracy | conflicts | conflict rate | mean key purity |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in display_df.iterrows():
        lines.append(
            f"| {row['prior_name']} | {row['family']} | {row['target']} | {row['features']} | "
            f"{int(row['eval_rows'])} | {int(row['covered_rows'])} | {format_float(row['coverage'])} | "
            f"{format_float(row['accuracy'])} | {int(row['conflict_rows'])} | "
            f"{format_float(row['conflict_rate'])} | {format_float(row['mean_key_purity_on_eval'])} |"
        )

    if key_rows:
        key_df = pd.DataFrame(key_rows)
        lines.extend(["", "## Representative High-Support Keys", ""])
        for prior_name in display_df["prior_name"].head(8):
            subset = key_df[key_df["prior_name"] == prior_name].head(5)
            if subset.empty:
                continue
            lines.extend(
                [
                    f"### {prior_name}",
                    "",
                    "| key | shortcut answer | eval rows | held-out key accuracy | train support | train purity |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )
            for _, key_row in subset.iterrows():
                lines.append(
                    f"| `{key_row['key_display']}` | {key_row['shortcut_answer']} | "
                    f"{int(key_row['eval_rows'])} | {format_float(key_row['eval_accuracy'])} | "
                    f"{int(key_row['train_support'])} | {format_float(key_row['train_purity'])} |"
                )
            lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- High held-out shortcut accuracy means the answer is often recoverable from metadata priors learned on train, without reading expression.",
            "- High coverage means the shortcut applies broadly across held-out rows, not only rare edge cases.",
            "- Conflict rows are the useful stress-test rows: the learned shortcut is wrong, so a model following it should fail.",
            "- Source-level shortcuts are especially important for models trained with dataset captions, metadata text, retrieval pairs, or study-level abstracts.",
            "",
            "## Outputs",
            "",
            f"- Summary CSV: `{summary_path}`",
            f"- Top keys CSV: `{keys_path}`",
            f"- Conflict examples: `{conflicts_path}`",
        ]
    )
    report_path = reports_dir / "source_bias_report.md"
    report_path.write_text("\n".join(lines))

    manifest = {
        "train_h5ad": args.train_h5ad,
        "eval_h5ad": args.eval_h5ad,
        "train_rows": len(train),
        "eval_rows": len(eval_data),
        "min_support": args.min_support,
        "min_purity": args.min_purity,
        "summary_csv": str(summary_path),
        "top_keys_csv": str(keys_path),
        "conflict_examples_jsonl_gz": str(conflicts_path),
        "report": str(report_path),
    }
    (output_dir / "source_bias_audit_metadata.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {report_path}")
    print(f"Wrote {summary_path}")
    if key_rows:
        print(f"Wrote {keys_path}")
    if conflict_rows:
        print(f"Wrote {conflicts_path}")


if __name__ == "__main__":
    main()
