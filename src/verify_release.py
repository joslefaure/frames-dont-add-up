#!/usr/bin/env python3
"""Static integrity, anonymity, schema, and coverage checks for the artifact."""
from hashlib import sha256
import json
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile("u" + "1399652|MST" + "115399|/ho" + "me/|/wo" + "rk/|SB" + "ATCH|SL" + "URM", re.I)
EXCLUDED_PARTS = {".git", ".venv", "reproduced", "__pycache__"}


def main():
    manifest_path = ROOT / "ARTIFACT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            raise AssertionError(f"missing: {entry['path']}")
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise AssertionError(f"integrity mismatch: {entry['path']}")

    for path in ROOT.rglob("*"):
        if (not path.is_file() or path == manifest_path or EXCLUDED_PARTS.intersection(path.parts) or
                path.suffix in {".pdf", ".parquet", ".pyc"}):
            continue
        text = path.read_text(errors="ignore")
        if FORBIDDEN.search(text):
            raise AssertionError(f"non-anonymous or scheduler-specific text: {path.relative_to(ROOT)}")

    e3 = list((ROOT / "artifacts/causal_analysis/e3").glob("*/per_candidate.parquet"))
    e4 = list((ROOT / "artifacts/causal_analysis/e4").glob("*/per_pair_context.parquet"))
    if len(e3) != 18 or len(e4) != 18:
        raise AssertionError(f"expected 18 settings, found e3={len(e3)}, e4={len(e4)}")
    for path in e3:
        table = pd.read_parquet(path)
        if len(table) != 4000 or set(table.candidate_stratum) != {"evidence", "boundary", "outside"}:
            raise AssertionError(f"singleton coverage mismatch: {path}")
    for path in e4:
        table = pd.read_parquet(path)
        if len(table) != 7500 or table.groupby(["item_id", "context_index", "pair_type"]).size().ne(1).any():
            raise AssertionError(f"factorial coverage mismatch: {path}")

    for name in ("mdp3_dpp_question_only_analysis", "mdp3_dpp_rich_query_analysis"):
        cells = sorted((ROOT / "artifacts" / name).glob("*/per_item.parquet"))
        if len(cells) != 18:
            raise AssertionError(f"expected 18 MDP3/DPP cells in {name}, found {len(cells)}")
        total = 0
        for path in cells:
            table = pd.read_parquet(path)
            expected = 990 if table.benchmark.iloc[0] == "nextgqa" else 1000
            if len(table) != expected or set(table.condition) != {"mdp3_8", "plain_dpp_8"}:
                raise AssertionError(f"MDP3/DPP coverage mismatch: {path}")
            if table.duplicated(["item_id", "condition"]).any():
                raise AssertionError(f"duplicate MDP3/DPP rows: {path}")
            total += len(table)
        if total != 17940:
            raise AssertionError(f"expected 17,940 MDP3/DPP rows in {name}, found {total}")

    for name in ("mdp3_dpp_question_only_selections", "mdp3_dpp_rich_query_selections"):
        selections = pd.concat([pd.read_csv(path) for path in sorted(
            (ROOT / "artifacts" / name).glob("*.selections.csv"))], ignore_index=True)
        if len(selections) != 2990 or selections.duplicated(["item_id", "condition"]).any():
            raise AssertionError(f"MDP3/DPP selection coverage mismatch: {name}")

    registry = pd.read_csv(ROOT / "artifacts/qualitative_registry.csv")
    if len(registry) != 148:
        raise AssertionError(f"expected 148 qualitative cases, found {len(registry)}")
    print(f"PASS: {len(manifest['files'])} files match ARTIFACT_MANIFEST.json")
    print("PASS: anonymity scan, 18-setting schemas, factorial and MDP3/DPP coverage, and 148-case registry")


if __name__ == "__main__":
    main()
