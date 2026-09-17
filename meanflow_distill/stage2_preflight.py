"""Check actual Stage 2 weights, longest prompts, gradients and checkpoint reload."""
import copy
import json
import random
from pathlib import Path

import torch

from meanflow_distill.train_intmeanflow_distill import (
    TextPromptDataset, collate_prompts, prepare_decoder_condition, distill_loss,
    set_trainable_decoder, freeze_all, precision_context, teacher_trajectory,
    save_student_checkpoint, build_arg_parser, resolve_streaming_flags,
    parse_student_t_grid,
    verify_teacher_streaming_geometry,
)
from meanflow_distill.interval_estimator import IntervalConditionedEstimator
from meanflow_distill.stage2_support import StartupAudit, load_student
from lits.models.lits import LITS


def main():
    args = build_arg_parser().parse_args()
    resolve_streaming_flags(args)
    args.student_t_grid = parse_student_t_grid(args.student_t_grid, args.student_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    dataset = TextPromptDataset(args.manifest, args.cleaner, 1, False, 0)
    longest = max(dataset.rows, key=lambda row: len(row["tokens"]))
    teacher = LITS.load_from_checkpoint(args.teacher_ckpt, map_location="cpu", weights_only=False).to(device).eval()
    student = copy.deepcopy(teacher)
    student.decoder.estimator = IntervalConditionedEstimator(student.decoder.estimator).to(device)
    if args.teacher_matched_streaming:
        for model in (teacher, student):
            verify_teacher_streaming_geometry(model, args)
    if args.decoder_streaming:
        from meanflow_distill.kv_cache_distill import configure_decoder_streaming_context
        for model in (teacher, student):
            configure_decoder_streaming_context(model, decoder_left_frames=args.decoder_left_frames,
                                                static_chunk_size=args.distill_chunk_size)
    freeze_all(teacher)
    params = set_trainable_decoder(student, False)
    audit = StartupAudit(student, teacher, args.output_dir, 0)
    # Prove copied time embedding behaves exactly like the old estimator at t.
    with torch.no_grad():
        t = torch.tensor([0., .5, 1.], device=device)
        r = torch.tensor([0., 0., .5], device=device)
        actual = student.decoder.estimator._interval_time_emb(r, t)
        expected = teacher.decoder.estimator.time_mlp(teacher.decoder.estimator.time_embeddings(t))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        # Regression: tone duration masks must not cross-broadcast utterances.
        ids = torch.tensor([[int(teacher.tone_mark_token_ids[0]), 1, 2],
                            [1, int(teacher.tone_mark_token_ids[0]), 2]], device=device)
        durations = torch.tensor([[[99., 7., 8.]], [[9., 99., 10.]]], device=device)
        text_mask = torch.ones_like(durations)
        batched = teacher._clamp_inference_tone_durations(durations, ids, text_mask)
        individual = torch.cat([teacher._clamp_inference_tone_durations(
            durations[i:i+1], ids[i:i+1], text_mask[i:i+1]) for i in range(2)])
        assert batched.shape == durations.shape
        torch.testing.assert_close(batched, individual, atol=0, rtol=0)
        from lits.utils.infer_duration_floor import _cap_tensor_per_position
        caps = torch.tensor([[2., 3., 4.], [5., 6., 7.]], device=device)
        need = torch.tensor([[True, False, True], [False, True, False]], device=device)
        batched = _cap_tensor_per_position(durations, durations.squeeze(1), need, caps)
        individual = torch.cat([_cap_tensor_per_position(durations[i:i+1],
            durations[i:i+1].squeeze(1), need[i:i+1], caps[i:i+1]) for i in range(2)])
        assert batched.shape == durations.shape
        torch.testing.assert_close(batched, individual, atol=0, rtol=0)
    # Teacher distillation condition + Euler trajectory must match production decoding.
    example = collate_prompts([min(dataset.rows, key=lambda row: abs(len(row["tokens"])-40))])
    x, lengths, spks, tones = (example[k].to(device) for k in ("x", "x_lengths", "spks", "x_tones"))
    with torch.no_grad():
        mu, mask, speaker = prepare_decoder_condition(teacher, x, lengths, spks, False, tones)
        hidden = teacher.get_hidden_mel(x, lengths, spks, x_tones=tones)
        z = torch.randn_like(mu) * args.temperature
        states, _ = teacher_trajectory(teacher, mu, mask, speaker, z, args.teacher_steps, False)
        direct = teacher.decoder(hidden["mu_y"], hidden["y_mask"], args.teacher_steps,
                                 True, args.temperature, hidden["spks"], z=z, streaming=False)
        torch.testing.assert_close(states[-1], direct, atol=1e-5, rtol=1e-5)
        if args.decoder_streaming:
            from meanflow_distill.streaming_eval import decoder_chunks
            from meanflow_distill.kv_cache_distill import trajectory_kv_cache
            for model, grid in ((teacher, [i / args.teacher_steps for i in range(args.teacher_steps + 1)]),
                                (student, args.student_t_grid)):
                for frames in (76, 200, 317):
                    condition = torch.randn(1, mu.shape[1], frames, device=device)
                    valid = torch.ones(1, 1, frames, device=device)
                    noise = torch.randn(1, model.n_feats, frames, device=device)
                    state, _ = trajectory_kv_cache(model, condition, valid, speaker, noise, grid,
                        chunk_size=args.distill_chunk_size, pre_lookahead_len=args.pre_lookahead_len)
                    production = torch.cat(list(decoder_chunks(model, condition, valid, speaker,
                        noise, grid, args.distill_chunk_size)), dim=-1)
                    torch.testing.assert_close(state[-1], production, atol=1e-5, rtol=1e-5)
    torch.cuda.reset_peak_memory_stats()
    student.decoder.estimator.train()
    batch = collate_prompts([longest] * args.batch_size)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)
    with precision_context(args.precision, device):
        loss, metrics = distill_loss(student, teacher, batch, args, device)
    assert torch.isfinite(loss)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
    optimizer.step()
    audit.check(student, teacher, 1)
    peak = torch.cuda.max_memory_allocated() / 2**30
    path = save_student_checkpoint(student, args.output_dir, args, 1, metrics, optimizer)
    student.eval()
    restored = load_student(path, device)
    with torch.inference_mode():
        if args.decoder_streaming:
            from vocos.vocoder import load_vocos_vocoder
            vocoder, _ = load_vocos_vocoder('/119010446/LITs/vocos/generator.ckpt', device,
                                           Path(__file__).resolve().parents[1])
            from meanflow_distill.streaming_eval import synthesize_streaming
            torch.manual_seed(124)
            before = synthesize_streaming(student, vocoder, x, lengths, spks, tones,
                                          args.student_t_grid, args.temperature,
                                          chunk_size=args.distill_chunk_size,
                                          mu_streaming=args.mu_streaming)['audio']
            torch.manual_seed(124)
            after = synthesize_streaming(restored, vocoder, x, lengths, spks, tones,
                                         args.student_t_grid, args.temperature,
                                         chunk_size=args.distill_chunk_size,
                                         mu_streaming=args.mu_streaming)['audio']
        else:
            torch.manual_seed(124)
            before = student.synthesise(x, lengths, 2, spks=spks, x_tones=tones)["mel"]
            torch.manual_seed(124)
            after = restored.synthesise(x, lengths, 2, spks=spks, x_tones=tones)["mel"]
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    report = dict(status="passed", time_embedding_exact=True, teacher_matches_production=True,
                  streaming=args.decoder_streaming,
                  chunk_size=args.distill_chunk_size, decoder_left_frames=args.decoder_left_frames,
                  teacher_matched_streaming=args.teacher_matched_streaming,
                  mu_streaming=args.mu_streaming,
                  streaming_trajectory_matches_chunk_outer=bool(args.decoder_streaming),
                  batched_duration_matches_individual=True,
                  checkpoint_roundtrip_exact=True, batch_size=args.batch_size,
                  longest_tokens=len(longest["tokens"]), peak_memory_gib=peak,
                  grad_norm=float(norm), metrics=metrics)
    (args.output_dir / "preflight.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
