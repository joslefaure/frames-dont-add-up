#!/usr/bin/env python3
"""Apply official MDP3 and a matched plain-DPP ablation to a frozen E2 pool."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "external/MDP3/vlmeval/smp/mdp3_frame_selector.py"
OFFICIAL_COMMIT = "45616806d1173f1a3d8b9c7210edcd764ea45c31"


def load_official_module(path):
    spec = importlib.util.spec_from_file_location("official_mdp3_frame_selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return json.loads(value)


def recover_candidate_paths(row, charged=64):
    selected_paths = parse_list(row.frame_paths)
    if not selected_paths:
        raise ValueError(f"no materialized candidates for {row.item_id}")
    example = Path(selected_paths[0])
    prefix, suffix = example.stem.rsplit("_", 1)
    if not suffix.isdigit():
        raise ValueError(f"unexpected candidate filename: {example}")
    paths = [example.with_name(f"{prefix}_{index:03d}{example.suffix}") for index in range(charged)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{row.item_id}: {len(missing)} pool frames missing; first={missing[0]}")
    return paths


def question_prompt(row, query_mode):
    if query_mode == "question_only":
        return row.question
    options = parse_list(row.options)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return row.question + "\n" + "\n".join(
        f"{letters[index]}. {option}" for index, option in enumerate(options))


def plain_dpp_indices(selector, image_embeds, text_embeds, budget):
    # Matched ablation: the official relevance/diversity kernel and greedy MAP
    # rule, but without MDP3's temporal segmentation/dynamic allocation.
    total = selector.kernel(torch.cat([text_embeds, image_embeds], dim=0))
    _, traces = selector.seqdpp_select_super_fast(total, offset=0, to_select_num=budget)
    return traces[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path,
                        help="Frozen E2 all_conditions.parquet for one benchmark")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--query-mode", choices=("question_only", "question_and_options"),
                        default="question_only")
    parser.add_argument("--official-file", type=Path, default=OFFICIAL,
                        help="MDP3 frame selector from the frozen official commit")
    args = parser.parse_args()
    if args.budget != 8:
        raise ValueError("the official implementation is frozen at K=8 for this first baseline")

    source = pd.read_parquet(args.input)
    if source.candidate_frame_count_charged.nunique() != 1 or int(
            source.candidate_frame_count_charged.iloc[0]) != 64:
        raise ValueError("MDP3 comparison requires the charged 64-frame E2 pool")
    representatives = source.sort_values(["sample_order", "condition"]).drop_duplicates("item_id")
    if args.limit is not None:
        representatives = representatives.head(args.limit)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.official_file.is_file():
        raise FileNotFoundError(
            f"official MDP3 selector not found at {args.official_file}; "
            f"check out commit {OFFICIAL_COMMIT} or pass --official-file")
    official = load_official_module(args.official_file)
    selector = official.MDP3(device=args.device)
    selector.n_selection = args.budget
    selector.segment_size = 32
    records = []
    selection_rows = []
    for position, row in enumerate(representatives.itertuples(index=False), 1):
        paths = recover_candidate_paths(row)
        cache = args.cache_dir / f"{row.item_id.replace(':', '_')}.pt"
        if cache.exists():
            embedded = torch.load(cache, map_location=args.device, weights_only=True)
            image_embeds = embedded["image_embeds"].to(args.device)
            text_embeds = embedded["text_embeds"].to(args.device)
        else:
            with torch.inference_mode():
                image_embeds, text_embeds = selector.vlm(
                    [str(path) for path in paths], question_prompt(row, args.query_mode))
            torch.save({"image_embeds": image_embeds.detach().cpu(),
                        "text_embeds": text_embeds.detach().cpu()}, cache)
        with torch.inference_mode():
            # Selection is set-valued; sort explicitly so every answerer receives
            # the selected frames in chronological candidate-pool order.
            mdp3 = sorted(map(int, selector._select_frames_fast(image_embeds, text_embeds)))
            dpp = sorted(map(int, plain_dpp_indices(selector, image_embeds, text_embeds, args.budget)))
        if len(mdp3) != args.budget or len(set(mdp3)) != args.budget:
            raise ValueError(f"invalid MDP3 selection for {row.item_id}: {mdp3}")
        if len(dpp) != args.budget or len(set(dpp)) != args.budget:
            raise ValueError(f"invalid DPP selection for {row.item_id}: {dpp}")
        if min(mdp3 + dpp) < 0 or max(mdp3 + dpp) >= 64:
            raise ValueError(f"out-of-pool selection for {row.item_id}: {mdp3}, {dpp}")
        times = np.linspace(0.0, float(row.duration), 64)
        base = row._asdict()
        for condition, indices in (("mdp3_8", mdp3), ("plain_dpp_8", dpp)):
            record = dict(base)
            record.update({
                "condition": condition,
                "frame_count": args.budget,
                "frame_times": json.dumps([float(times[index]) for index in indices]),
                "frame_paths": json.dumps([str(paths[index].resolve()) for index in indices]),
                "selected_candidate_indices": json.dumps(indices),
                "sample_order": int(row.sample_order),
            })
            records.append(record)
            selection_rows.append({"item_id": row.item_id, "benchmark": row.benchmark,
                                   "condition": condition,
                                   "selected_candidate_indices": json.dumps(indices)})
        print(f"[{position}/{len(representatives)}] {row.item_id} MDP3={mdp3} DPP={dpp}", flush=True)

    output = pd.DataFrame(records)
    if output.duplicated(["item_id", "condition"]).any():
        raise ValueError("duplicate item/selector output")
    counts = output.groupby("item_id").condition.nunique()
    if not (counts == 2).all():
        raise ValueError("every item must have both MDP3 and DPP")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    pd.DataFrame(selection_rows).to_csv(args.output.with_suffix(".selections.csv"), index=False)
    provenance = {
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output": str(args.output.resolve()),
        "items": int(output.item_id.nunique()),
        "rows": int(len(output)),
        "charged_candidate_pool": 64,
        "selection_budget": args.budget,
        "selector_query_mode": args.query_mode,
        "conditions": ["mdp3_8", "plain_dpp_8"],
        "chronological_output": True,
        "official_commit": OFFICIAL_COMMIT,
        "official_file": str(args.official_file.resolve()),
        "official_file_sha256": hashlib.sha256(args.official_file.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
