"""Isolated synthesis, ASR and voice-quality stages for fixed checkpoint evals."""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'data_generation/majestic_voice')]
DATA = Path('/119010446/tts-assets/data_24k/ljs_majestic')
ASSETS = Path('/119010446/tts-assets')


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def synthesize(args):
    import numpy as np
    import soundfile as sf
    import torch
    from lits.models.lits import LITS
    from vocos.vocoder import load_vocos_vocoder
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    if args.distilled:
        from meanflow_distill.stage2_support import load_student
        model = load_student(args.checkpoint)
    else:
        model = LITS.load_from_checkpoint(str(args.checkpoint), map_location='cpu', weights_only=False)
    assert model.n_spks == 2 and model.n_feats == 100 and model.n_vocab == 173
    model = model.to('cuda').eval()
    streaming = args.streaming or getattr(model, 'distill_streaming', False)
    geometry = None
    if streaming:
        from meanflow_distill.kv_cache_distill import configure_decoder_streaming_context
        saved = getattr(model, 'distill_streaming_config', {})
        estimator = getattr(model.decoder.estimator, 'base', model.decoder.estimator)
        chunk_size = (args.streaming_chunk_size if args.streaming_chunk_size is not None
                      else saved.get('distill_chunk_size', estimator.static_chunk_size))
        left_frames = (args.decoder_left_frames if args.decoder_left_frames is not None
                       else saved.get('decoder_left_frames', estimator.decoder_left_frames))
        mu_streaming = (args.mu_streaming if args.mu_streaming is not None
                        else saved.get('mu_streaming', False))
        if saved:
            assert chunk_size == saved['distill_chunk_size'], 'Evaluation chunk size differs from training'
            assert left_frames == saved['decoder_left_frames'], 'Evaluation left context differs from training'
            assert mu_streaming == saved['mu_streaming'], 'Evaluation conditioning differs from training'
        geometry = dict(chunk_size=chunk_size, decoder_left_frames=left_frames,
                        mu_streaming=mu_streaming, mel_cache_len=args.mel_cache_len,
                        mu_static_chunk_size=model.decoder.encoder.static_chunk_size,
                        mu_encoder_left_chunks=model.decoder.encoder_num_decoding_left_chunks)
        configure_decoder_streaming_context(model, decoder_left_frames=left_frames,
                                            static_chunk_size=chunk_size)
        write_json(args.output / 'streaming_geometry.json', geometry)
    vocoder, cfg = load_vocos_vocoder(str(args.vocoder_checkpoint), torch.device('cuda'), REPO)
    assert cfg.sampling_rate == 24000 and cfg.num_mels == 100 and cfg.hop_size == 384
    sources = records(DATA / 'eval_manifest.jsonl')
    if args.per_group_limit:
        groups = defaultdict(list)
        for row in sources:
            groups[row['group']].append(row)
        sources = [row for rows in groups.values() for row in rows[:args.per_group_limit]]
    if args.limit:
        sources = sources[:args.limit]
    with (args.output / 'synthesis.jsonl').open('w', buffering=1) as stream:
        for index, source in enumerate(sources):
            row = dict(source)
            started = time.monotonic()
            try:
                torch.manual_seed(20260910 + index)
                ids = torch.tensor(source['token_ids'], dtype=torch.long, device='cuda')[None]
                tones = torch.tensor(source['tone_ids'], dtype=torch.long, device='cuda')[None]
                lengths = torch.tensor([ids.shape[-1]], dtype=torch.long, device='cuda')
                speaker = torch.tensor([source['speaker']], dtype=torch.long, device='cuda')
                with torch.inference_mode():
                    # FM keeps 10 steps; iMF checkpoints carry their sampling budget.
                    steps = args.n_timesteps or getattr(model.decoder, 'default_n_timesteps', 10)
                    if streaming:
                        from meanflow_distill.streaming_eval import synthesize_streaming
                        grid = getattr(model, 'distill_t_grid', [i / steps for i in range(steps + 1)])
                        result = synthesize_streaming(model, vocoder, ids, lengths, speaker,
                            tones, grid, temperature=args.temperature,
                            chunk_size=geometry['chunk_size'], mel_cache_len=geometry['mel_cache_len'],
                            mu_streaming=geometry['mu_streaming'])
                        audio = result['audio'].cpu().numpy()
                    else:
                        result = model.synthesise(ids, lengths, steps, temperature=args.temperature, spks=speaker, x_tones=tones)
                        mel = result['mel']
                        assert torch.isfinite(mel).all(), 'Nonfinite predicted Mel'
                        assert 1 <= mel.shape[-1] <= 7500, 'Predicted duration outside 0..120 seconds'
                        audio = vocoder(mel)[..., :int(result['mel_lengths'][0]) * 384].squeeze().cpu().numpy()
                assert audio.ndim == 1 and len(audio) and np.isfinite(audio).all(), 'Invalid waveform'
                target = args.output / 'wavs' / (source['id'] + '.wav')
                target.parent.mkdir(parents=True, exist_ok=True)
                sf.write(target, np.clip(audio, -1, 1), 24000, subtype='PCM_16')
                row.update(audio=str(target), duration=len(audio) / 24000, flow_steps=steps,
                           streaming=streaming, vocoder_mode='chunked' if streaming else 'full',
                           streaming_geometry=geometry,
                           flow_objective=getattr(model.decoder, 'objective', 'cfm'),
                           synthesis_seconds=time.monotonic() - started)
            except Exception as exc:
                row['synthesis_error'] = f'{type(exc).__name__}: {exc}'
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            if (index + 1) % 25 == 0:
                print(json.dumps({'stage': 'synthesis', 'completed': index + 1, 'target': len(sources)}), flush=True)


def asr(args):
    import torch
    from qwen_asr import Qwen3ASRModel
    from quality_worker import cer, wer, normalize, tonal_match
    torch.set_num_threads(4)
    model = Qwen3ASRModel.from_pretrained(str(ASSETS / 'Qwen3-ASR-1.7B'), dtype=torch.bfloat16,
                                          device_map='cuda:0', attn_implementation='sdpa',
                                          max_inference_batch_size=8, max_new_tokens=256)
    rows = [r for r in records(args.output / 'synthesis.jsonl') if 'audio' in r]
    with (args.output / 'asr.jsonl').open('w', buffering=1) as stream:
        for offset in range(0, len(rows), 8):
            batch = rows[offset:offset + 8]
            predictions = model.transcribe(audio=[r['audio'] for r in batch], context='', language=None)
            assert len(predictions) == len(batch)
            for row, pred in zip(batch, predictions):
                result = {'id': row['id'], 'asr_text': pred.text, 'asr_language': pred.language,
                          'cer': cer(row['ref_text'], pred.text), 'wer': wer(row['ref_text'], pred.text),
                          'reference_characters': len(normalize(row['ref_text'])),
                          'exact_tonal_pinyin': tonal_match(row['ref_text'], pred.text)}
                # Denominator matches quality_worker.wer's normalization exactly.
                import re
                import unicodedata
                text = unicodedata.normalize('NFKC', row['ref_text']).lower().replace('’', "'")
                result['reference_words'] = len(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text))
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(json.dumps({'stage': 'asr', 'completed': min(offset + 8, len(rows))}), flush=True)


def metrics(args):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from quality_metrics import SpeakerMetrics, DNSMOS, audio16
    torch.set_num_threads(4)
    protocol = json.loads((DATA / 'eval_protocol.json').read_text())
    reference_paths = protocol['speaker_references']
    speakers = SpeakerMetrics(Path(reference_paths['0']), fast_matmul=False)
    references = {0: speakers.reference,
                  1: speakers.embeddings(audio16(Path(reference_paths['1'])))}
    group_references = {}
    for group, path in protocol.get('group_speaker_references', {}).items():
        expected = protocol.get('group_reference_sha256', {}).get(group)
        if expected:
            import hashlib
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        group_references[group] = speakers.embeddings(audio16(Path(path)))
    dnsmos = DNSMOS()
    rows = [r for r in records(args.output / 'synthesis.jsonl') if 'audio' in r]
    with (args.output / 'metrics.jsonl').open('w', buffering=1) as stream, ThreadPoolExecutor(8) as pool:
        for offset in range(0, len(rows), 16):
            batch = rows[offset:offset + 16]
            futures = {}
            for row in batch:
                futures[row['id']] = pool.submit(lambda path: dnsmos(audio16(path)), row['audio'])
            for row in batch:
                try:
                    speakers.reference = group_references.get(row['group'], references[row['speaker']])
                    score = speakers(audio16(row['audio']))
                    result = {'id': row['id'], **score, **futures[row['id']].result()}
                except Exception as exc:
                    result = {'id': row['id'], 'metrics_error': f'{type(exc).__name__}: {exc}'}
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(json.dumps({'stage': 'metrics', 'completed': min(offset + 16, len(rows))}), flush=True)


def summarize(args):
    synthesized = records(args.output / 'synthesis.jsonl')
    asr_rows = {r['id']: r for r in records(args.output / 'asr.jsonl')}
    metric_rows = {r['id']: r for r in records(args.output / 'metrics.jsonl')}
    groups = defaultdict(list)
    joined = []
    for row in synthesized:
        row = {**row, **asr_rows.get(row['id'], {}), **metric_rows.get(row['id'], {})}
        groups[row['group']].append(row)
        joined.append(row)
    report = {'status': 'complete', 'checkpoint': str(args.checkpoint), 'samples': len(joined), 'groups': {}}
    for group, rows in groups.items():
        errors = [r for r in rows if 'synthesis_error' in r or 'metrics_error' in r or 'cer' not in r]
        scores = {'samples': len(rows), 'evaluation_failures': len(errors)}
        for key in ('cer', 'wer', 'wavlm_similarity', 'camp_similarity', 'dnsmos_ovrl', 'dnsmos_sig', 'dnsmos_bak'):
            values = [r[key] for r in rows if key in r and math.isfinite(r[key])]
            scores[key + '_mean'] = sum(values) / len(values) if values else None
        for metric, denominator in [('cer', 'reference_characters'), ('wer', 'reference_words')]:
            valid = [r for r in rows if metric in r and r.get(denominator, 0)]
            total = sum(r[denominator] for r in valid)
            scores[metric + '_micro'] = sum(r[metric] * r[denominator] for r in valid) / total if total else None
        report['groups'][group] = scores
    (args.output / 'details.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in joined))
    write_json(args.output / 'summary.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def build_parser():
    """Share evaluation options with the Stage 2 compatibility entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['synthesize', 'asr', 'metrics', 'summarize'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--per-group-limit', type=int, default=0)
    parser.add_argument('--distilled', action='store_true')
    parser.add_argument('--streaming', action='store_true')
    parser.add_argument('--streaming-chunk-size', type=int)
    parser.add_argument('--decoder-left-frames', type=int)
    parser.add_argument('--mel-cache-len', type=int, default=8)
    parser.add_argument('--mu-streaming', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--n-timesteps', type=int)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--vocoder-checkpoint', type=Path, default=REPO / 'vocos/generator.ckpt')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    DATA = args.data_dir
    args.output.mkdir(parents=True, exist_ok=True)
    {'synthesize': synthesize, 'asr': asr, 'metrics': metrics, 'summarize': summarize}[args.stage](args)
