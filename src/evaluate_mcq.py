#!/usr/bin/env python3
"""Deterministic constrained multiple-choice evaluation over prepared frames."""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image


def prompt_for(row):
    choices = '\n'.join(f'({chr(65 + index)}) {text}' for index, text in enumerate(row.options))
    return f'Question: {row.question}\n{choices}\nAnswer with the option letter only.'


def load_model(name, registry, reproducibility):
    model_id = registry['models'][name]
    specification = reproducibility['models'].get(name)
    if specification is None or specification.get('repository') != model_id:
        raise ValueError(f'{name}: reproducibility registry does not match the model registry')
    revision = specification.get('revision')
    if not revision:
        raise ValueError(f'{name}: missing pinned model revision')
    if name.startswith('qwen3_vl'):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda').eval()
        processor = AutoProcessor.from_pretrained(model_id, revision=revision, min_pixels=200704, max_pixels=200704)
        return model, processor, 'qwen'
    if name.startswith('internvl35'):
        from transformers import AutoProcessor, InternVLForConditionalGeneration
        model = InternVLForConditionalGeneration.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda',
            trust_remote_code=True).eval()
        return model, AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=True), 'internvl'
    if name == 'perception_lm_8b':
        from transformers import AutoProcessor, PerceptionLMForConditionalGeneration
        model = PerceptionLMForConditionalGeneration.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.bfloat16, attn_implementation='eager', device_map='cuda').eval()
        return model, AutoProcessor.from_pretrained(model_id, revision=revision), 'perception_lm'
    if name == 'videollama3_7b':
        from transformers import AutoModelForCausalLM
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        import transformers.image_utils as image_utils
        if not hasattr(image_utils, 'VideoInput'):
            image_utils.VideoInput = object
        processor_class = get_class_from_dynamic_module(
            'processing_videollama3.Videollama3Qwen2Processor', model_id, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda',
            trust_remote_code=True).eval()
        return model, processor_class.from_pretrained(model_id, revision=revision), 'videollama3'
    raise NotImplementedError(f'{name}: adapter not implemented yet')


def model_inputs(kind, processor, frame_paths, text):
    images = [Image.open(path).convert('RGB') for path in frame_paths]
    content = [{'type': 'image', 'image': image} for image in images] + [{'type': 'text', 'text': text}]
    chat = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                        add_generation_prompt=True)
    if kind == 'qwen':
        kwargs = {'text': [chat], 'padding': True, 'return_tensors': 'pt'}
        if images:
            kwargs['images'] = images
        return processor(**kwargs).to('cuda')
    if kind == 'perception_lm':
        if not images:
            chat = processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}],
                tokenize=False, add_generation_prompt=True)
            return processor(text=[chat], padding=True, return_tensors='pt').to('cuda')
        chat = processor.apply_chat_template(
            [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': text}]}],
            tokenize=False, add_generation_prompt=True)
        return processor(text=[chat], videos=[images], padding=True, return_tensors='pt').to('cuda')
    if kind == 'videollama3':
        visual_prefix = '<image>' * len(images)
        inputs = processor(text=f'{visual_prefix}{text}', images=images if images else None,
                           return_tensors='pt').to('cuda')
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                inputs[key] = value.to(dtype=torch.bfloat16)
        return inputs
    return processor(text=chat, images=images if images else None, min_patches=1, max_patches=1,
                     return_tensors='pt').to('cuda')


def option_ids(processor, count):
    ids = []
    for letter in 'ABCDE'[:count]:
        token_ids = processor.tokenizer.encode(letter, add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(f'{letter} does not map to one token: {token_ids}')
        ids.append(token_ids[0])
    return ids


def record_key(record):
    """Stable identity for resumable rows across E1/E3/E4 intervention grids."""
    return (record['item_id'], record['condition'], record.get('context_index'),
            record.get('pair_type'), record.get('factorial_cell'))


def row_key(row):
    return (row.item_id, row.condition,
            None if 'context_index' not in row.index or pd.isna(row.context_index) else int(row.context_index),
            None if 'pair_type' not in row.index or pd.isna(row.pair_type) else row.pair_type,
            None if 'factorial_cell' not in row.index or pd.isna(row.factorial_cell) else row.factorial_cell)


def completed_keys(path):
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line:
            record = json.loads(line)
            done.add(record_key(record))
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--registry', type=Path, default=Path('configs/registry.json'))
    parser.add_argument('--reproducibility-config', type=Path,
                        default=Path('configs/reproducibility.json'))
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--conditions', nargs='+',
                        help='Optional exact condition subset; used only for a documented resource-bounded cell.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--empty-cache-every', type=int, default=0,
                        help='Release cached CUDA allocations every N completed rows (0 disables it).')
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    reproducibility = json.loads(args.reproducibility_config.read_text())
    table = pd.read_parquet(args.input).sort_values(['condition', 'sample_order'])
    if args.conditions:
        requested = set(args.conditions)
        observed = set(table.condition.unique())
        unknown = requested - observed
        if unknown:
            raise ValueError(f'conditions absent from input: {sorted(unknown)}')
        table = table[table.condition.isin(requested)]
    if args.limit:
        table = table.iloc[:args.limit]
    if args.overwrite and args.output.exists():
        args.output.unlink()
    done = completed_keys(args.output)
    model, processor, kind = load_model(args.model, registry, reproducibility)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('a') as handle:
        for _, row in table.iterrows():
            key = row_key(row)
            if key in done:
                continue
            frames = json.loads(row.frame_paths)
            valid_ids = option_ids(processor, len(row.options))
            inputs = model_inputs(kind, processor, frames, prompt_for(row))
            start = time.time()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores=True,
                    prefix_allowed_tokens_fn=lambda _batch, _ids, allowed=valid_ids: allowed,
                )
            logits = generated.scores[0][0].float()
            logprobs = torch.log_softmax(logits[valid_ids], dim=0).cpu().tolist()
            token = processor.batch_decode(generated.sequences[:, inputs.input_ids.shape[1]:],
                                           skip_special_tokens=True)[0].strip().upper()
            # Accuracy is defined from the same constrained option distribution
            # used for log odds, avoiding model-family-specific decode quirks.
            prediction = 'ABCDE'[max(range(len(logprobs)), key=lambda index: logprobs[index])]
            gold_index = ord(row.answer) - ord('A')
            gold_logprob = logprobs[gold_index]
            rest = torch.logsumexp(torch.tensor([value for index, value in enumerate(logprobs) if index != gold_index]), 0).item()
            result = {'model': args.model, 'item_id': row.item_id, 'benchmark': row.benchmark,
                      'condition': row.condition, 'answer': row.answer, 'prediction': prediction,
                      'response': token,
                      'correct': prediction == row.answer, 'option_logprobs': logprobs,
                      'gold_option_logprob': gold_logprob, 'gold_vs_rest_logodds': gold_logprob - rest,
                      'frame_count': int(row.frame_count), 'frame_times': json.loads(row.frame_times),
                      'seconds': time.time() - start}
            # Retain the intervention design with each outcome. This lets the
            # released JSONL support re-aggregation without hidden joins.
            for field in ('sample_order', 'question_type', 'duration_band', 'context_index',
                          'candidate_index', 'candidate_time', 'candidate_stratum', 'pair_type',
                          'factorial_cell', 'candidate_a', 'candidate_b', 'candidate_a_time',
                          'candidate_b_time', 'subset_mask', 'active_candidates', 'annotation_kind'):
                if field in row.index and not pd.isna(row[field]):
                    value = row[field]
                    result[field] = value.item() if hasattr(value, 'item') else value
            handle.write(json.dumps(result, sort_keys=True) + '\n')
            handle.flush()
            print(json.dumps(result, sort_keys=True), flush=True)
            if args.empty_cache_every and (len(done) + 1) % args.empty_cache_every == 0:
                del inputs, generated, logits
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
