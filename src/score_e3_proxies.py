#!/usr/bin/env python3
"""Score frozen E3 candidates with question relevance and local visual change."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def image_features(model, processor, paths, batch_size):
    vectors = []
    for start in range(0, len(paths), batch_size):
        images = [Image.open(path).convert('RGB') for path in paths[start:start + batch_size]]
        inputs = processor(images=images, return_tensors='pt').to('cuda')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            z = model.get_image_features(**inputs)
        vectors.append(torch.nn.functional.normalize(z.float(), dim=-1).cpu())
    return torch.cat(vectors)


def text_feature(model, processor, question):
    inputs = processor(text=[question], padding=True, truncation=True, max_length=64,
                       return_tensors='pt').to('cuda')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        z = model.get_text_features(**inputs)
    return torch.nn.functional.normalize(z.float(), dim=-1).cpu()[0]


def dino_features(model, processor, paths, batch_size):
    vectors = []
    for start in range(0, len(paths), batch_size):
        images = [Image.open(path).convert('RGB') for path in paths[start:start + batch_size]]
        inputs = processor(images=images, return_tensors='pt').to('cuda')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(**inputs)
        z = result.pooler_output if getattr(result, 'pooler_output', None) is not None else result.last_hidden_state[:, 0]
        vectors.append(torch.nn.functional.normalize(z.float(), dim=-1).cpu())
    return torch.cat(vectors)


def decode_neighbor(video, time, duration, offset, target):
    cap = cv2.VideoCapture(str(video))
    frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    index = min(frames - 1, max(0, int(round(min(duration - 1e-3, max(0., time + offset)) * fps))))
    image = None
    for backoff in (0, -1, 1, -2, 2, -5, 5):
        candidate = index + backoff
        if not 0 <= candidate < frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        ok, bgr = cap.read()
        if ok:
            image = bgr
            break
    cap.release()
    if image is None:
        raise RuntimeError(f'cannot decode neighbor {video}@{time}')
    if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f'cannot write {target}')
    return str(target.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True, type=Path,
                        help='E3 candidates.jsonl from prepare_causal_candidates.py')
    parser.add_argument('--e3-input', required=True, type=Path,
                        help='E3 all_conditions.parquet, used to recover materialized candidate RGB paths')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--exclusion-log', type=Path,
                        help='Structured JSONL ledger for item-level neighbor-decode failures')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--siglip-model', default='google/siglip-so400m-patch14-384')
    parser.add_argument('--dino-model', default='facebook/dinov2-base')
    args = parser.parse_args()
    candidates = load_rows(args.candidates)
    prepared = pd.read_parquet(args.e3_input)
    # Candidate i occupies fixed slot 3 in every E3 replacement condition.
    materialized = prepared[prepared.candidate_index >= 0].copy()
    materialized['candidate_path'] = materialized.frame_paths.map(lambda value: json.loads(value)[3])
    lookup = materialized.drop_duplicates(['item_id', 'candidate_index']).set_index(['item_id', 'candidate_index']).candidate_path
    siglip = AutoModel.from_pretrained(args.siglip_model, torch_dtype=torch.bfloat16).to('cuda').eval()
    sigproc = AutoProcessor.from_pretrained(args.siglip_model)
    dino = AutoModel.from_pretrained(args.dino_model, torch_dtype=torch.bfloat16).to('cuda').eval()
    dinoproc = AutoImageProcessor.from_pretrained(args.dino_model)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    records, exclusions = [], []
    for position, item in enumerate(candidates, 1):
        times = list(map(float, item['candidate_times']))
        candidate_paths = [lookup[(item['item_id'], index)] for index in range(len(times))]
        sig_images = image_features(siglip, sigproc, candidate_paths, args.batch_size)
        relevance = text_feature(siglip, sigproc, item['question']) @ sig_images.T
        duration = float(item['duration'])
        # Local change is symmetric around each candidate, with a frozen scale
        # tied to the 64-frame candidate pool used by E2.
        offset = max(.25, duration / 64.)
        neighbor_paths = []
        failed = None
        for candidate_index, timestamp in enumerate(times):
            before = args.cache_dir / f'{item["item_id"].replace(":", "_")}_{candidate_index}_before.jpg'
            after = args.cache_dir / f'{item["item_id"].replace(":", "_")}_{candidate_index}_after.jpg'
            try:
                if not before.exists():
                    decode_neighbor(item['source_video_path'], timestamp, duration, -offset, before)
                if not after.exists():
                    decode_neighbor(item['source_video_path'], timestamp, duration, offset, after)
            except Exception as error:
                failed = {
                    'item_id': item['item_id'], 'benchmark': item['benchmark'],
                    'source_video_id': item['source_video_id'],
                    'source_video_path': item['source_video_path'],
                    'candidate_index': candidate_index, 'candidate_time': timestamp,
                    'neighbor_offset': offset, 'error_type': type(error).__name__,
                    'error': str(error), 'policy': 'exclude_item_from_proxy_alignment',
                }
                break
            neighbor_paths.extend([str(before), str(after)])
        if failed is not None:
            exclusions.append(failed)
            print(f'[{position}/{len(candidates)}] EXCLUDED {item["item_id"]}: {failed["error"]}', flush=True)
            continue
        neighbors = dino_features(dino, dinoproc, neighbor_paths, args.batch_size)
        change = 1 - (neighbors[0::2] * neighbors[1::2]).sum(-1)
        for candidate_index, timestamp in enumerate(times):
            records.append({
                'item_id': item['item_id'], 'benchmark': item['benchmark'],
                'source_video_id': item['source_video_id'], 'candidate_index': candidate_index,
                'candidate_time': timestamp, 'candidate_stratum': item['candidate_strata'][candidate_index],
                'siglip_relevance': float(relevance[candidate_index]),
                'dino_local_change': float(change[candidate_index]),
                'temporal_centrality': 1 - abs(timestamp / duration - .5) * 2,
                'annotated_membership': int(item['candidate_strata'][candidate_index] == 'evidence'),
            })
        print(f'[{position}/{len(candidates)}] {item["item_id"]}', flush=True)
    table = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)
    if args.exclusion_log:
        args.exclusion_log.parent.mkdir(parents=True, exist_ok=True)
        args.exclusion_log.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in exclusions))
    if table.empty:
        raise RuntimeError('proxy scoring produced no retained records')
    print(table.groupby(['benchmark', 'candidate_stratum']).size())
    print(json.dumps({'retained_items': int(table.item_id.nunique()),
                      'excluded_items': len(exclusions),
                      'exclusion_log': str(args.exclusion_log) if args.exclusion_log else None}))


if __name__ == '__main__':
    main()
