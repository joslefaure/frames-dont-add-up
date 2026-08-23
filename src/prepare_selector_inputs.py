#!/usr/bin/env python3
"""Materialize fixed-budget E2 selector interventions from a frozen manifest.

All selectors see the same chronological candidate pool.  Candidate RGB files
are shared by source video, while question-conditioned selections are written
as ordinary ``frame_paths`` rows consumable by evaluate_mcq.py.  The selection
record deliberately retains every exposed candidate count and score so that
selector latency and charged-frame accounting can be analysed separately from
answerer inference.
"""
import argparse
import hashlib
import heapq
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor


def stable(seed, *parts):
    return hashlib.sha256(':'.join(map(str, (seed,) + parts)).encode()).hexdigest()


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if count <= 0 or fps <= 0:
        raise RuntimeError(f'cannot read video metadata: {path}')
    return count, fps, count / fps


def decode_candidates(video, candidate_times, frame_dir, key):
    count, fps, duration = video_info(video)
    frame_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    paths = []
    for pos, timestamp in enumerate(candidate_times):
        target = frame_dir / f'{key}_{pos:03d}.jpg'
        if not target.exists():
            frame_index = max(0, min(count - 1, int(round(float(timestamp) * fps))))
            frame = None
            for offset in (0, -1, 1, -2, 2, -5, 5):
                index = frame_index + offset
                if not 0 <= index < count:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, candidate = cap.read()
                if ok:
                    frame = candidate
                    break
            if frame is None:
                cap.release()
                raise RuntimeError(f'cannot decode {video} at frame {frame_index}')
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                cap.release()
                raise RuntimeError(f'cannot write {target}')
        paths.append(str(target.resolve()))
    cap.release()
    return paths, duration


def image_embeddings(model, processor, paths, device, batch_size):
    result = []
    for start in range(0, len(paths), batch_size):
        images = [Image.open(path).convert('RGB') for path in paths[start:start + batch_size]]
        inputs = processor(images=images, return_tensors='pt').to(device)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            z = model.get_image_features(**inputs)
        result.append(torch.nn.functional.normalize(z.float(), dim=-1).cpu())
    return torch.cat(result)


def text_embedding(model, processor, text, device):
    inputs = processor(text=[text], padding=True, truncation=True, max_length=64,
                       return_tensors='pt').to(device)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        z = model.get_text_features(**inputs)
    return torch.nn.functional.normalize(z.float(), dim=-1).cpu()[0]


def dino_embeddings(model, processor, paths, device, batch_size):
    result = []
    for start in range(0, len(paths), batch_size):
        images = [Image.open(path).convert('RGB') for path in paths[start:start + batch_size]]
        inputs = processor(images=images, return_tensors='pt').to(device)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(**inputs)
        z = output.pooler_output if getattr(output, 'pooler_output', None) is not None else output.last_hidden_state[:, 0]
        result.append(torch.nn.functional.normalize(z.float(), dim=-1).cpu())
    return torch.cat(result)


def normalized(values):
    values = values.float()
    lo, hi = values.min(), values.max()
    return torch.zeros_like(values) if float(hi - lo) < 1e-12 else (values - lo) / (hi - lo)


def uniform_indices(total, budget):
    return np.linspace(0, total - 1, budget).round().astype(int).tolist()


def change_indices(change, total, budget):
    """Select both endpoints of the largest DINO transitions, then fill by change."""
    selected = []
    for edge in torch.argsort(change, descending=True).tolist():
        for index in (edge, edge + 1):
            if index not in selected:
                selected.append(index)
            if len(selected) == budget:
                return sorted(selected)
    for index in uniform_indices(total, budget):
        if index not in selected:
            selected.append(index)
        if len(selected) == budget:
            break
    return sorted(selected)


def diversity_indices(embeddings, budget):
    """Deterministic farthest-point traversal in frozen DINO embedding space."""
    mean = embeddings.mean(0, keepdim=True)
    selected = [int(torch.argmax(1 - embeddings @ torch.nn.functional.normalize(mean, dim=-1).T).item())]
    while len(selected) < budget:
        similarity = embeddings @ embeddings[selected].T
        candidate = int(torch.argmin(similarity.max(dim=1).values).item())
        if candidate in selected:
            candidate = next(i for i in range(len(embeddings)) if i not in selected)
        selected.append(candidate)
    return sorted(selected)


def relevance_diversity_indices(relevance, embeddings, budget, alpha):
    selected = [int(torch.argmax(relevance).item())]
    rel = normalized(relevance)
    while len(selected) < budget:
        diversity = 1 - (embeddings @ embeddings[selected].T).max(dim=1).values
        score = alpha * rel + (1 - alpha) * normalized(diversity)
        score[selected] = -float('inf')
        selected.append(int(torch.argmax(score).item()))
    return sorted(selected)


def bolt_its_indices(relevance, budget):
    """BOLT's released inverse-transform sampling rule (CVPR 2025)."""
    score = relevance.float().numpy().copy()
    score -= score.min()
    if score.max() <= 1e-12:
        return uniform_indices(len(score), budget)
    score /= score.max()
    # The official code's deterministic quantiles preserve both relevance and
    # temporal spread; repeated indices are retained as charged frame slots.
    cdf = np.cumsum(score / score.sum())
    return np.searchsorted(cdf, np.linspace(1 / budget, 1 - 1 / budget, budget)).astype(int).tolist()


def _aks_split(scores, indices, budget, depth, max_depth, t1, t2):
    mean, std = float(np.mean(scores)), float(np.std(scores))
    top = heapq.nlargest(min(budget, len(scores)), range(len(scores)), scores.__getitem__)
    mean_diff = float(np.mean([scores[i] for i in top]) - mean)
    if mean_diff > t1 and std > t2 or depth >= max_depth or len(scores) < 2:
        return [(scores, indices, depth)]
    midpoint = len(scores) // 2
    return _aks_split(scores[:midpoint], indices[:midpoint], budget, depth + 1, max_depth, t1, t2) + \
        _aks_split(scores[midpoint:], indices[midpoint:], budget, depth + 1, max_depth, t1, t2)


def aks_fixed_k_indices(relevance, budget, t1=.8, t2=-100., max_depth=5):
    """Official AKS recursive allocation plus a deterministic fixed-K fill.

    AKS's released allocator can emit fewer/more than K when binary segment
    sizes collide with its ``K / 2**depth`` rule. The confirmatory study holds
    K fixed, so we retain its allocated keys, trim by relevance if necessary,
    and fill remaining slots by relevance-ranked unselected candidates.
    """
    raw = relevance.float().numpy().copy()
    normalized_score = np.zeros_like(raw) if raw.max() - raw.min() <= 1e-12 else (raw - raw.min()) / (raw.max() - raw.min())
    selected = []
    for segment, indices, depth in _aks_split(normalized_score.tolist(), list(range(len(raw))), budget, 0, max_depth, t1, t2):
        amount = int(budget / 2 ** depth)
        selected.extend(indices[i] for i in heapq.nlargest(amount, range(len(segment)), segment.__getitem__))
    selected = list(dict.fromkeys(selected))
    ranked = torch.argsort(relevance, descending=True).tolist()
    selected = sorted(selected, key=lambda index: float(relevance[index]), reverse=True)[:budget]
    for index in ranked:
        if index not in selected:
            selected.append(index)
        if len(selected) == budget:
            break
    return sorted(selected)


def selections(relevance, dino, budget, seed, item_id, alpha):
    total = len(relevance)
    change = 1 - (dino[:-1] * dino[1:]).sum(-1)
    generator = torch.Generator().manual_seed(int(stable(seed, item_id, 'random')[:16], 16) % (2 ** 63 - 1))
    return {
        f'uniform{budget}': uniform_indices(total, budget),
        f'random{budget}': sorted(torch.randperm(total, generator=generator)[:budget].tolist()),
        f'siglip_relevance{budget}': sorted(torch.argsort(relevance, descending=True)[:budget].tolist()),
        f'dino_change{budget}': change_indices(change, total, budget),
        f'dino_diversity{budget}': diversity_indices(dino, budget),
        f'siglip_dino_submodular{budget}': relevance_diversity_indices(relevance, dino, budget, alpha),
        f'bolt_its{budget}': bolt_its_indices(relevance, budget),
        f'aks_fixedk{budget}': aks_fixed_k_indices(relevance, budget),
    }, change


def read_manifest(path, limit):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    rows.sort(key=lambda row: row['item_id'])
    return rows[:limit] if limit else rows


def sampling_intervals(intervals, duration, point_window):
    result = []
    for start, end in intervals:
        start, end = float(start), float(end)
        if start == end:
            start, end = start - point_window, end + point_window
        result.append([max(0., start), min(duration, end)])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20260720)
    parser.add_argument('--budget', type=int, default=8)
    parser.add_argument('--candidate-frames', type=int, default=64)
    parser.add_argument('--relevance-weight', type=float, default=.5)
    parser.add_argument('--point-window-seconds', type=float, default=2.)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--exclude-video-id', action='append', default=[],
                        help='Exclude a source video with an unrecoverable decode failure. '
                             'The exclusion is retained in provenance rather than silently skipped.')
    parser.add_argument('--skip-unrecoverable-video', action='store_true',
                        help='Continue after a decode failure, recording every affected item in provenance.')
    parser.add_argument('--siglip-model', default='google/siglip-so400m-patch14-384')
    parser.add_argument('--dino-model', default='facebook/dinov2-base')
    args = parser.parse_args()
    if not 0 <= args.relevance_weight <= 1:
        raise ValueError('--relevance-weight must be in [0, 1]')
    rows = read_manifest(args.manifest, args.limit)
    input_items = len(rows)
    excluded_video_ids = set(args.exclude_video_id)
    excluded_rows = [
        {'item_id': row['item_id'], 'source_video_id': row['source_video_id'],
         'reason': 'explicit_unrecoverable_decode_failure'}
        for row in rows if row['source_video_id'] in excluded_video_ids
    ]
    rows = [row for row in rows if row['source_video_id'] not in excluded_video_ids]
    if not rows:
        raise ValueError('all manifest rows were excluded')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda'
    siglip = AutoModel.from_pretrained(args.siglip_model, torch_dtype=torch.bfloat16).to(device).eval()
    sigproc = AutoProcessor.from_pretrained(args.siglip_model)
    dino = AutoModel.from_pretrained(args.dino_model, torch_dtype=torch.bfloat16).to(device).eval()
    dinoproc = AutoImageProcessor.from_pretrained(args.dino_model)
    by_video = defaultdict(list)
    for order, row in enumerate(rows):
        row['sample_order'] = order
        by_video[(row['benchmark'], row['source_video_id'], row['source_video_path'])].append(row)
    output_rows, timing = [], []
    for video_number, ((benchmark, video_id, video_path), questions) in enumerate(by_video.items(), 1):
        start = time.time()
        try:
            _, _, duration = video_info(video_path)
            evidence_windows = sampling_intervals(questions[0].get('evidence_intervals', []), duration,
                                                  args.point_window_seconds)
            timestamps = np.linspace(0, max(0., duration - 1e-3), args.candidate_frames).tolist()
            key = stable(args.seed, benchmark, video_id, args.candidate_frames)[:20]
            candidate_paths, _ = decode_candidates(video_path, timestamps, args.output_dir / 'candidate_frames', key)
        except RuntimeError as error:
            if not args.skip_unrecoverable_video:
                raise
            excluded_rows.extend({
                'item_id': row['item_id'], 'source_video_id': row['source_video_id'],
                'reason': 'candidate_pool_decode_failure', 'detail': str(error),
            } for row in questions)
            print(f'[{video_number}/{len(by_video)}] SKIP {benchmark}:{video_id}: {error}', flush=True)
            continue
        sig_images = image_embeddings(siglip, sigproc, candidate_paths, device, args.batch_size)
        dino_images = dino_embeddings(dino, dinoproc, candidate_paths, device, args.batch_size)
        for row in questions:
            relevance = text_embedding(siglip, sigproc, row['question'], device) @ sig_images.T
            proposed, change = selections(relevance, dino_images, args.budget, args.seed, row['item_id'], args.relevance_weight)
            for condition, indices in proposed.items():
                output_rows.append({
                    **row, 'condition': condition, 'duration': duration, 'frame_count': len(indices),
                    'sampling_evidence_intervals': json.dumps(evidence_windows),
                    'frame_times': json.dumps([round(float(timestamps[i]), 6) for i in indices]),
                    'frame_paths': json.dumps([candidate_paths[i] for i in indices]),
                    'candidate_frame_count_charged': args.candidate_frames,
                    'selected_candidate_indices': json.dumps(indices),
                    'siglip_relevance': json.dumps([round(float(x), 8) for x in relevance]),
                    'dino_adjacent_change': json.dumps([round(float(x), 8) for x in change]),
                    'selector_seed': args.seed,
                    'selector_relevance_weight': args.relevance_weight,
                })
        seconds = time.time() - start
        timing.append({'benchmark': benchmark, 'source_video_id': video_id, 'questions': len(questions),
                       'seconds': seconds, 'candidate_frames_charged': args.candidate_frames})
        print(f'[{video_number}/{len(by_video)}] {benchmark}:{video_id} q={len(questions)} sec={seconds:.1f}', flush=True)
    table = pd.DataFrame(output_rows).sort_values(['condition', 'sample_order'])
    table.to_parquet(args.output_dir / 'all_conditions.parquet', index=False)
    for condition, subset in table.groupby('condition'):
        subset.to_parquet(args.output_dir / f'{condition}.parquet', index=False)
    (args.output_dir / 'timing.json').write_text(json.dumps(timing, indent=2) + '\n')
    (args.output_dir / 'provenance.json').write_text(json.dumps({
        'manifest': str(args.manifest.resolve()), 'items': int(table.item_id.nunique()),
        'input_items': input_items,
        'excluded_rows': excluded_rows,
        'seed': args.seed,
        'budget': args.budget, 'candidate_frames_charged': args.candidate_frames,
        'relevance_weight': args.relevance_weight,
        'point_window_seconds': args.point_window_seconds,
        'conditions': sorted(table.condition.unique()),
    }, indent=2) + '\n')
    print(table.groupby(['benchmark', 'condition']).size())


if __name__ == '__main__':
    main()
