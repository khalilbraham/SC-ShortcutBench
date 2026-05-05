#!/usr/bin/env python
"""Build ontology-backed candidate sets for unified model evaluation."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


TASK_TARGETS = {
    "cell_type_prediction": "cell_type",
    "tissue_general_prediction": "tissue_general",
    "disease_prediction": "disease",
}


def normalize_label(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_task_spec(challenge_paths: list[Path]) -> dict:
    labels: dict[str, dict[str, dict]] = defaultdict(dict)
    split_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for path in challenge_paths:
        split_name = path.stem.replace("_shortcut_challenge", "").replace("_control_challenge", "")
        for row in iter_jsonl(path):
            task = row.get("task")
            if task not in TASK_TARGETS:
                continue
            split_counts[task][split_name] += 1
            for key in ("answer", "shortcut_answer"):
                label = str(row.get(key, "")).strip()
                if not label:
                    continue
                norm = normalize_label(label)
                item = labels[task].setdefault(
                    norm,
                    {
                        "label": label,
                        "normalized_label": norm,
                        "ontology_term_ids": set(),
                        "seen_as_answer": 0,
                        "seen_as_shortcut": 0,
                    },
                )
                if key == "answer":
                    item["seen_as_answer"] += 1
                    term_id = row.get("target_ontology_term_id")
                    if term_id:
                        item["ontology_term_ids"].add(str(term_id))
                else:
                    item["seen_as_shortcut"] += 1

    task_specs = {}
    for task, entries in labels.items():
        candidates = []
        ambiguous = []
        for item in entries.values():
            term_ids = sorted(item["ontology_term_ids"])
            if len(term_ids) > 1:
                ambiguous.append(item["normalized_label"])
            candidates.append(
                {
                    "label": item["label"],
                    "normalized_label": item["normalized_label"],
                    "ontology_term_ids": term_ids,
                    "seen_as_answer": item["seen_as_answer"],
                    "seen_as_shortcut": item["seen_as_shortcut"],
                }
            )
        candidates.sort(key=lambda item: (-item["seen_as_answer"], item["normalized_label"]))
        task_specs[task] = {
            "target": TASK_TARGETS[task],
            "n_candidates": len(candidates),
            "n_ontology_backed_candidates": sum(bool(item["ontology_term_ids"]) for item in candidates),
            "ambiguous_normalized_labels": ambiguous,
            "split_rows": dict(split_counts[task]),
            "candidates": candidates,
        }

    return {
        "schema": "sc_shortcutbench_unified_task_spec_v1",
        "challenge_files": [path.as_posix() for path in challenge_paths],
        "tasks": task_specs,
        "notes": [
            "Candidate sets are built from answer and shortcut labels in the supplied benchmark files.",
            "Shortcut-only labels may lack ontology IDs unless they also appear as answers.",
            "Use external ontology closure files for ancestor-aware scoring.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = build_task_spec(args.challenge)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    for task, task_spec in spec["tasks"].items():
        print(
            f"{task}: {task_spec['n_candidates']} candidates, "
            f"{task_spec['n_ontology_backed_candidates']} ontology-backed"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
