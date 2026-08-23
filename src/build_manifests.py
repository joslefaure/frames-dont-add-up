#!/usr/bin/env python3
"""Build output-independent census and causal manifests from official resources."""
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def stable_hash(seed, *parts):
    return hashlib.sha256(':'.join(map(str, (seed,) + parts)).encode()).hexdigest()


def normalise_options(values):
    return [re.sub(r'^\s*[A-E][.)]\s*', '', str(v)).strip() for v in list(values)]


def option_answer(answer, options):
    answer = str(answer).strip().upper()
    if re.fullmatch('[A-E]', answer):
        return answer
    for i, value in enumerate(options):
        if answer == str(value).strip().upper():
            return chr(65 + i)
    raise ValueError(f'cannot map answer {answer!r}')


def parse_timestamp(value):
    """Parse LVBench's MM:SS (or HH:MM:SS) timestamp into seconds."""
    pieces = [float(piece) for piece in str(value).strip().split(':')]
    if not pieces or len(pieces) > 3:
        raise ValueError(f'invalid timestamp {value!r}')
    seconds = 0
    for piece in pieces:
        if piece < 0:
            raise ValueError(f'invalid timestamp {value!r}')
        seconds = seconds * 60 + piece
    return float(seconds)


def parse_reference_window(value):
    """Convert the official LVBench start-end reference to numeric seconds."""
    match = re.fullmatch(r'\s*([0-9:.]+)\s*-\s*([0-9:.]+)\s*', str(value))
    if not match:
        raise ValueError(f'invalid reference window {value!r}')
    start, end = (parse_timestamp(part) for part in match.groups())
    if end < start:
        raise ValueError(f'reversed reference window {value!r}')
    return [[start, end]]


def parse_herbench_point(value):
    """Parse HERBench's M:SS:ms point-cue format into seconds."""
    parts = str(value).strip().split(':')
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f'invalid HERBench point cue {value!r}')
    minutes, seconds, milliseconds = map(int, parts)
    if seconds >= 60 or milliseconds >= 1000:
        raise ValueError(f'invalid HERBench point cue {value!r}')
    point = 60 * minutes + seconds + milliseconds / 1000
    return [[point, point]]


def nextgqa(cfg):
    with open(cfg['qa_csv'], newline='') as handle:
        qa = list(csv.DictReader(handle))
    grounding = json.loads(Path(cfg['grounding_json']).read_text())
    rows, excluded = [], defaultdict(int)
    for row in qa:
        video_id, qid = row['video_id'], row['qid']
        path = Path(cfg['video_dir']) / f'{video_id}.mp4'
        intervals = grounding.get(video_id, {}).get('location', {}).get(qid)
        if not path.is_file():
            excluded['missing_video'] += 1
            continue
        if not intervals:
            excluded['missing_grounding'] += 1
            continue
        options = [row[f'a{i}'] for i in range(5)]
        rows.append({
            'benchmark': 'nextgqa', 'item_id': f'nextgqa:{video_id}:{qid}',
            'source_video_id': video_id, 'source_video_path': str(path.resolve()),
            'question': row['question'], 'options': options,
            'answer': option_answer(row['answer'], options), 'question_type': row['type'],
            'evidence_intervals': intervals,
            'annotation_kind': cfg['annotation_kind'], 'duration_band': 'short'
        })
    return rows, dict(excluded)


def videomme(cfg):
    table = pd.read_parquet(cfg['qa_parquet'])
    rows, excluded = [], defaultdict(int)
    for _, row in table.iterrows():
        path = Path(cfg['video_dir']) / f'{row.videoID}.mp4'
        if not path.is_file():
            excluded['missing_video'] += 1
            continue
        options = normalise_options(row.options)
        rows.append({
            'benchmark': 'videomme', 'item_id': f'videomme:{row.question_id}',
            'source_video_id': str(row.videoID), 'source_video_path': str(path.resolve()),
            'question': str(row.question), 'options': options,
            'answer': option_answer(row.answer, options), 'question_type': str(row.task_type),
            'evidence_intervals': [], 'annotation_kind': cfg['annotation_kind'],
            'duration_band': str(row.duration), 'domain': str(row.domain),
            'sub_category': str(row.sub_category)
        })
    return rows, dict(excluded)


def lvbench(cfg):
    table = pd.read_parquet(cfg['qa_parquet'])
    rows, excluded = [], defaultdict(int)
    for _, row in table.iterrows():
        video_id = str(row.key)
        path = Path(cfg['video_dir']) / f'{video_id}.mp4'
        if not path.is_file():
            excluded['missing_video'] += 1
            continue
        question_lines, option_map = [], {}
        for line in str(row.question).splitlines():
            match = re.match(r'^\s*\(([A-E])\)\s*(.*)$', line)
            if match:
                option_map[match.group(1)] = match.group(2).strip()
            else:
                question_lines.append(line)
        if sorted(option_map) != list('ABCD'):
            excluded['unparseable_options'] += 1
            continue
        try:
            intervals = parse_reference_window(row.time_reference)
            annotation_kind = cfg['annotation_kind']
        except ValueError:
            # Retain every official question in the census.  Entries without a
            # usable interval are simply ineligible for interval-conditioned runs.
            intervals = []
            annotation_kind = 'reference_window_unusable'
            excluded['unusable_reference_window'] += 1
        question_type = '|'.join(map(str, list(row.question_type)))
        rows.append({
            'benchmark': 'lvbench', 'item_id': f'lvbench:{row.uid}',
            'source_video_id': video_id, 'source_video_path': str(path.resolve()),
            'question': '\n'.join(question_lines).strip(),
            'options': [option_map[letter] for letter in 'ABCD'],
            'answer': str(row.answer).strip().upper(), 'question_type': question_type,
            'evidence_intervals': intervals,
            'evidence_reference': str(row.time_reference),
            'annotation_kind': annotation_kind, 'duration_band': 'long',
            'category': str(row.type)
        })
    return rows, dict(excluded)


def herbench(cfg):
    table = pd.read_parquet(cfg['qa_parquet'])
    rows, excluded = [], defaultdict(int)
    for _, row in table.iterrows():
        relative_video = str(row.video_path).removeprefix('videos/')
        path = Path(cfg['video_dir']) / relative_video
        if not path.is_file():
            excluded['missing_video'] += 1
            continue
        try:
            metadata = json.loads(row.metadata_json)
            if 'timestamps' in metadata:
                intervals = [parse_reference_window(value)[0] for value in metadata['timestamps']]
            elif metadata.get('required_timestamps') != 'All':
                intervals = [parse_herbench_point(value)[0] for value in metadata['required_timestamps']]
            else:
                intervals = []
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            intervals = []
            excluded['unusable_evidence_timestamps'] += 1
        options = normalise_options(row.choices)
        rows.append({
            'benchmark': 'herbench', 'item_id': f'herbench:{row.question_id}',
            'source_video_id': str(row.video_id), 'source_video_path': str(path.resolve()),
            'question': str(row.question), 'options': options,
            'answer': option_answer(row.answer, options), 'question_type': str(row.task_type),
            'evidence_intervals': intervals,
            'annotation_kind': cfg['annotation_kind'] if intervals else 'unlocalized_evidence_requirement',
            'duration_band': 'medium', 'source_dataset': str(row.source_dataset),
        })
    return rows, dict(excluded)


def stratified_holdout(rows, size, seed):
    if len(rows) < size:
        raise RuntimeError(f'eligible rows {len(rows)} < requested holdout {size}')
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get('question_type', ''), row.get('duration_band', ''))].append(row)
    chosen = []
    total = len(rows)
    allocations = {}
    for key, group in groups.items():
        allocations[key] = min(len(group), int(round(size * len(group) / total)))
    while sum(allocations.values()) < size:
        key = max(groups, key=lambda k: len(groups[k]) - allocations[k])
        allocations[key] += 1
    while sum(allocations.values()) > size:
        key = max(allocations, key=allocations.get)
        allocations[key] -= 1
    for key, group in groups.items():
        chosen.extend(sorted(group, key=lambda x: stable_hash(seed, x['item_id']))[:allocations[key]])
    return sorted(chosen, key=lambda x: stable_hash(seed, 'order', x['item_id']))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', type=Path, default=Path('configs/registry.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('manifests/v1'))
    parser.add_argument('--benchmarks', nargs='+', default=['lvbench', 'nextgqa', 'herbench', 'videomme'])
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    builders = {'nextgqa': nextgqa, 'videomme': videomme, 'lvbench': lvbench, 'herbench': herbench}
    report = {'schema_version': registry['schema_version'], 'sampling': registry['sampling'], 'benchmarks': {}}
    for name in args.benchmarks:
        cfg = registry['benchmarks'][name]
        if cfg['status'] != 'ready_local':
            raise RuntimeError(f'{name} is not ready_local: {cfg["status"]}')
        rows, excluded = builders[name](cfg)
        rows = sorted(rows, key=lambda x: x['item_id'])
        write_jsonl(args.output_dir / 'census' / f'{name}.jsonl', rows)
        entry = {'eligible': len(rows), 'excluded': excluded,
                 'census_path': str((args.output_dir / 'census' / f'{name}.jsonl').resolve())}
        if cfg['annotation_kind'] != 'none_census_only':
            grounded_rows = [row for row in rows if row['evidence_intervals']]
            holdout = stratified_holdout(grounded_rows, registry['sampling']['causal_holdout_questions'], registry['sampling']['seed'])
            write_jsonl(args.output_dir / 'causal_holdout' / f'{name}.jsonl', holdout)
            entry['causal_holdout'] = len(holdout)
            entry['causal_eligible'] = len(grounded_rows)
            entry['causal_holdout_path'] = str((args.output_dir / 'causal_holdout' / f'{name}.jsonl').resolve())
        report['benchmarks'][name] = entry
        print(json.dumps({name: entry}, sort_keys=True), flush=True)
    (args.output_dir / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
