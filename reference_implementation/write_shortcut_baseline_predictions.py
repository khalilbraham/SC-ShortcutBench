#!/usr/bin/env python
"""Write a metadata-shortcut baseline for the shortcut challenge.

This is the negative-control model. It never looks at expression tokens; it
answers with the train-split metadata prior stored in each benchmark row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(args.challenge) as challenge_handle, output.open("w") as output_handle:
        for line in challenge_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            output_handle.write(
                json.dumps(
                    {
                        "example_id": row["example_id"],
                        "prediction": row.get("shortcut_answer", ""),
                    }
                )
                + "\n"
            )
            written += 1

    print(f"Wrote {written:,} shortcut predictions to {output}")


if __name__ == "__main__":
    main()
