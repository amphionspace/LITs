"""Verify the mixed distiller against CFM_Causal.compute_loss's actual masks."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import torch

from lits.models.lits import LITS
from meanflow_distill import train_intmeanflow_distill as train
from meanflow_distill.interval_estimator import IntervalConditionedEstimator
from meanflow_distill.stage2_run import training_command


def observe(model):
    result = dict(condition_modes=[], decoder_modes=[], masks={})
    handles = []

    def condition(module, inputs, kwargs, output):
        result['condition_modes'].append(kwargs['streaming'])
        result['condition'] = output.detach().clone()

    def decoder(module, inputs, kwargs):
        result['decoder_modes'].append(kwargs['streaming'])

    def condition_mask(module, inputs):
        result['masks']['condition'] = inputs[1].detach().clone()

    handles.append(model.decoder.encoder.register_forward_hook(condition, with_kwargs=True))
    handles.append(model.decoder.encoder.encoders[0].register_forward_pre_hook(condition_mask))
    handles.append(model.decoder.estimator.register_forward_pre_hook(decoder, with_kwargs=True))
    estimator = getattr(model.decoder.estimator, 'base', model.decoder.estimator)
    for name, module in estimator.named_modules():
        if module.__class__.__name__ == 'BasicTransformerBlock':
            def attention(module, inputs, kwargs, name=name):
                result['masks'].setdefault(name, kwargs['attention_mask'].detach().clone())
            handles.append(module.register_forward_pre_hook(attention, with_kwargs=True))
    return result, handles


def remove(handles):
    for handle in handles:
        handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    opts = parser.parse_args()
    run = opts.run_dir
    plan = json.loads((run / 'plan.json').read_text())
    command = training_command(run, plan)
    index = next(i for i, arg in enumerate(command) if arg.endswith('/train_intmeanflow_distill.py'))
    args = train.build_arg_parser().parse_args(command[index + 1:])
    train.resolve_streaming_flags(args)
    args.student_t_grid = train.parse_student_t_grid(args.student_t_grid, args.student_steps)
    assert args.teacher_matched_streaming and not args.kv_cache_distill and not args.parallel_streaming
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(317)
    device = torch.device('cuda')
    teacher = LITS.load_from_checkpoint(args.teacher_ckpt, map_location='cpu', weights_only=False).to(device).eval()
    student = copy.deepcopy(teacher)
    student.decoder.estimator = IntervalConditionedEstimator(student.decoder.estimator).to(device)
    for model in (teacher, student):
        train.verify_teacher_streaming_geometry(model, args)
    train.freeze_all(teacher)
    parameters = train.set_trainable_decoder(student, False)
    student.decoder.estimator.train()
    dataset = train.TextPromptDataset(args.manifest, args.cleaner, 1, False, 0)
    rows = sorted(dataset.rows, key=lambda row: abs(len(row['tokens']) - 40))[:2]
    batch = train.collate_prompts(rows)
    x, lengths, spks, tones = [batch[k].to(device) for k in ('x', 'x_lengths', 'spks', 'x_tones')]
    hidden = teacher.get_hidden_mel(x, lengths, spks, x_tones=tones)
    mu_y, mask, speaker = [hidden[k].clone() for k in ('mu_y', 'y_mask', 'spks')]
    reports, conditions = [], []
    for mode in (False, True):
        reference, handles = observe(teacher)
        # This is the original teacher training loss, not a reimplemented mask.
        with patch('random.random', return_value=0.25 if mode else 0.75) as coin:
            with torch.no_grad():
                teacher.decoder.compute_loss(torch.randn_like(mu_y), mask, mu_y, speaker)
            assert coin.call_count == 1
        remove(handles)
        actual, handles = observe(teacher)
        learner, student_handles = observe(student)
        student.zero_grad(set_to_none=True)
        with patch('random.random', return_value=0.25 if mode else 0.75) as coin:
            with patch.object(teacher.decoder, 'encode_mu', side_effect=AssertionError('Inference cache path used')):
                with train.precision_context(args.precision, device):
                    loss, metrics = train.distill_loss(student, teacher, batch, args, device)
            assert coin.call_count == 1
        loss.backward()
        remove(handles + student_handles)
        assert reference['condition_modes'] == actual['condition_modes'] == [mode]
        assert reference['decoder_modes'] == [mode]
        assert actual['decoder_modes'] == [mode] * 16
        assert learner['decoder_modes'] == [mode] * 4
        # Compare actual attention masks at every decoder scale, including padding.
        for name, expected in reference['masks'].items():
            if name == 'condition':
                torch.testing.assert_close(actual['masks'][name], expected, atol=0, rtol=0)
            else:
                # BF16 rounds the large negative bias; visibility must be identical.
                torch.testing.assert_close(actual['masks'][name] == 0, expected == 0, atol=0, rtol=0)
                torch.testing.assert_close(learner['masks'][name] == 0, expected == 0, atol=0, rtol=0)
        # The reference above is FP32; compare condition outputs separately in FP32.
        with torch.no_grad():
            condition, _, _ = train.prepare_decoder_condition(
                teacher, x, lengths, spks, mode, tones, teacher_training_mask=True)
        torch.testing.assert_close(condition, reference['condition'], atol=0, rtol=0)
        conditions.append(condition)
        assert teacher.decoder._enc_offset == 0 and teacher.decoder._encoded_mu is None
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        assert all(p.grad is None for p in teacher.parameters())
        assert all(p.grad is None for name, p in student.named_parameters()
                   if not name.startswith('decoder.estimator.'))
        reports.append(dict(streaming=mode, original_masks_match=True,
            mu_matches_original_training_exactly=True, shared_mode=True,
            teacher_forwards=16, student_forwards=4, finite_gradients=True, metrics=metrics))
        print(json.dumps(reports[-1]), flush=True)
    condition_difference = float((conditions[0] - conditions[1]).abs().max())
    assert condition_difference > 1e-6, 'The two condition modes unexpectedly produced identical results'
    # A preceding different-length utterance cannot contaminate the next condition.
    with torch.no_grad():
        short = train.collate_prompts([min(dataset.rows, key=lambda row: len(row['tokens']))])
        train.prepare_decoder_condition(teacher, short['x'].to(device), short['x_lengths'].to(device),
            short['spks'].to(device), True, short['x_tones'].to(device), teacher_training_mask=True)
        repeat, _, _ = train.prepare_decoder_condition(
            teacher, x, lengths, spks, True, tones, teacher_training_mask=True)
    torch.testing.assert_close(repeat, conditions[1], atol=0, rtol=0)
    report = dict(status='passed', reports=reports, cross_utterance_cache_isolation=True,
                  condition_mode_max_difference=condition_difference,
                  mu_static_chunk_size=50, decoder_static_chunk_size=50, decoder_left_frames=20,
                  mu_encoder_left_chunks=-1)
    (run / 'teacher_mask_preflight.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
