"""Operational capacity gate on longest Stage 2 examples; probe weights are discarded."""

import argparse
from pathlib import Path
import time

from training.imf.recipe import initialized_model, read_json, write_json


def main(run, batch_size):
    import numpy as np
    import torch
    from lits.data.text_mel_datamodule import TextMelBatchCollate
    from training.majestic_scratch.config import parameter_digest
    from training.majestic_scratch.data import NaturalBuckets
    from training.stage2.data import Stage2Audio

    torch.set_num_threads(4)
    plan = read_json(run / 'plan.json')
    dataset = Stage2Audio(run / 'data', 'train', plan['data_statistics'])
    model = initialized_model(run)
    assert parameter_digest(model) == plan['initial_parameter_sha256']
    assert len(dataset) == plan['train_rows']
    accumulation = plan['effective_batch'] // (4 * batch_size)
    samplers = [NaturalBuckets(dataset, batch_size, rank, 4, plan['seed']) for rank in range(4)]
    partitions = [list(sampler) for sampler in samplers]
    assert len({len(partition) for partition in partitions}) == 1
    assert len(partitions[0]) % accumulation == 0
    assert len(partitions[0]) // accumulation == plan['steps_per_epoch']
    indices = [index for partition in partitions for batch in partition for index in batch]
    assert len(indices) == len(set(indices))
    longest = np.argsort(dataset.lengths)[-batch_size:].tolist()
    # Preserve the longest waveform, include maximum token width and both speakers.
    longest[0] = max(range(len(dataset)), key=lambda i: len(dataset.rows[i]['text_key']))
    for offset, speaker in enumerate(plan['active_speaker_ids'], 1):
        if not any(dataset.sources[i] == speaker for i in longest):
            pool = np.flatnonzero(dataset.sources == speaker)
            longest[offset] = int(pool[np.argmax(dataset.lengths[pool])])
    batch = TextMelBatchCollate(2)([dataset[i] for i in longest])
    batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    model = model.cuda().train()
    optimizer = model.configure_optimizers()
    if isinstance(optimizer, dict):
        optimizer = optimizer['optimizer']
    assert not optimizer.state
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        losses = model(batch['x'], batch['x_lengths'], batch['y'], batch['y_lengths'],
                       spks=batch['spks'], x_tones=batch['x_tones'])
        loss = sum(losses[:3])
    assert torch.isfinite(loss)
    loss.backward()
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        assert torch.isfinite(parameter.grad).all(), name
        group = ('speaker' if name.startswith('spk_emb.') else
                 'duration' if name.startswith('encoder.proj_w.') else
                 'text' if name.startswith('encoder.') else
                 'condition' if name.startswith('decoder.encoder.') else
                 'interval' if '.interval_projector.' in name else
                 'aux_v' if '.v_' in name else 'flow_u')
        groups[group] = groups.get(group, 0) + int(parameter.grad.count_nonzero() > 0)
    assert all(groups.get(group, 0) > 0 for group in
               ('speaker', 'duration', 'text', 'condition', 'interval', 'aux_v', 'flow_u'))
    assert all(model.spk_emb.weight.grad[speaker].norm() > 0 for speaker in plan['active_speaker_ids'])
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    # Exercise validation under the same inference_mode used by Lightning.
    with torch.inference_mode():
        losses = model(batch['x'], batch['x_lengths'], batch['y'], batch['y_lengths'],
                       spks=batch['spks'], x_tones=batch['x_tones'])
        assert all(torch.isfinite(value) for value in losses[:3])
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    report = dict(
        status='passed' if peak < total * 0.70 else 'insufficient_headroom',
        batch_size_per_gpu=batch_size, accumulate_grad_batches=accumulation,
        peak_allocated_gib=peak / 2**30, total_memory_gib=total / 2**30,
        seconds=time.monotonic() - started, optimizer_steps=1, probe_weights_discarded=True,
        gradient_groups=groups, train_rows=len(dataset), unique_rank_partition=True,
        optimizer_steps_per_epoch=len(partitions[0]) // accumulation,
        max_mel_frames=batch['y'].shape[-1], max_text_tokens=batch['x'].shape[-1],
        validation_inference_mode_passed=True,
    )
    write_json(run / f'capacity_{batch_size}.json', report)
    return 0 if report['status'] == 'passed' else 3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    args = parser.parse_args()
    try:
        raise SystemExit(main(args.run_dir, args.batch_size))
    except Exception as exc:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            write_json(args.run_dir / f'capacity_{args.batch_size}.json',
                       dict(status='oom', batch_size_per_gpu=args.batch_size))
            raise SystemExit(3)
        raise
