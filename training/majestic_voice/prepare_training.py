"""Build the balanced 24-kHz training set from audited generation outputs."""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from lits.text import text_to_sequence_with_tones

ASSETS = Path('/119010446/tts-assets')
DATA = ASSETS / 'data_24k/ljs_majestic'
VOICE = Path('/ai_sds_wuzz/DATA_TTS/MajesticVoice')
CLEANERS = ['en_zh_dict_mixed_rhyme_body_tone_cleaners']
SEED = 20260910


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def manifest_line(row):
    fields = [row['audio'], str(row['speaker'])]
    if 'start' in row:
        fields += [str(row['start']), str(row['end'])]
    return '|'.join(fields + [row['text']]) + '\n'


def main(args):
    DATA.mkdir(parents=True, exist_ok=True)
    cache_path = DATA / 'text_preflight.jsonl'
    cache = {r['text']: r for r in records(cache_path)} if cache_path.exists() else {}
    with cache_path.open('a', buffering=1) as stream:
        def annotate(row):
            text = row['text']
            if text not in cache:
                ids, tones, phonemes = text_to_sequence_with_tones(text, CLEANERS)
                cache[text] = {'text': text, 'tokens': ids, 'tones': tones, 'phonemes': phonemes}
                stream.write(json.dumps(cache[text], ensure_ascii=False) + '\n')
                if len(cache) % 1000 == 0:
                    print(json.dumps({'checked_unique_texts': len(cache)}), flush=True)
            info = cache[text]
            if not info['tokens'] or 2 in info['tokens'] or not all(0 <= t < 173 for t in info['tokens']):
                raise ValueError(f'Empty/unknown/out-of-range token: {text}')
            return {**row, 'text_key': tuple(info['tokens'])}

        def ljs(split):
            rows = []
            for line in (DATA / f'ljs_{split}.txt').read_text().splitlines():
                fields = line.split('|')
                assert len(fields) in (3, 5), fields
                wav, speaker, text = fields[0], fields[1], fields[-1]
                segment = {'start': float(fields[2]), 'end': float(fields[3])} if len(fields) == 5 else {}
                rows.append(annotate({'audio': wav, 'speaker': int(speaker), 'text': text,
                                      'language': 'English', 'source': 'LJSpeech', **segment}))
            return rows

        ljs_train, ljs_val = ljs('train'), ljs('val')
        if args.preflight_only:
            chinese = records(VOICE / 'quality/selected.jsonl')
            for row in chinese:
                annotate({'text': row['text']})
            print(json.dumps({'ljs_train_rows': len(ljs_train), 'ljs_val_rows': len(ljs_val),
                              'majestic_chinese_rows': len(chinese),
                              'unique_texts_checked': len(cache)}), flush=True)
            return

        frontend_exclusions = []

        def generated(root):
            audit = json.loads((root / 'quality/final_audit.json').read_text())
            assert audit['status'] == 'passed'
            job = json.loads((root / 'job.json').read_text())
            selected = records(root / 'quality/selected.jsonl')
            assert job['status'] == 'complete', f'Generation is still running: {root}'
            assert len(selected) == audit['accepted_texts']
            assert len(records(root / 'requests.jsonl')) == audit['source_texts'], 'Stale generation audit'
            rows = []
            for r in selected:
                try:
                    rows.append(annotate({'audio': r['final_24k'], 'speaker': 1, 'text': r['text'],
                                          'language': 'English' if root.name == 'english' else 'Chinese',
                                          'source': 'MajesticVoice', 'source_id': r['source_id'],
                                          'source_split': r['split']}))
                except ValueError as exc:
                    frontend_exclusions.append({'root': str(root), 'source_id': r['source_id'],
                                                'text': r['text'], 'reason': str(exc)})
            return rows

        chinese, english = generated(VOICE), generated(VOICE / 'english')
        (DATA / 'text_frontend_exclusions.json').write_text(json.dumps(frontend_exclusions, ensure_ascii=False, indent=2) + '\n')
        cn_pool = [r for r in chinese if r['source_split'] == 'train']
        cn_test = [r for r in chinese if r['source_split'] == 'test']
        en_pool = [r for r in english if r['source_split'] == 'train']
        en_heldout = [r for r in english if r['source_split'] == 'test']
        random.Random(SEED).shuffle(cn_pool)
        # Chinese test remains held out. Select validation by unique token sequence.
        test_keys = {r['text_key'] for r in cn_test}
        cn_pool = [r for r in cn_pool if r['text_key'] not in test_keys]
        cn_val, seen = [], set()
        for row in cn_pool:
            if row['text_key'] not in seen:
                cn_val.append(row)
                seen.add(row['text_key'])
            if len(cn_val) == 200:
                break
        cn_train = [r for r in cn_pool if r['text_key'] not in seen]
        # Eval utterances must be absent from every training voice at the phoneme level.
        # Preserve the source held-out split. Remove matching training texts
        # below across both voices, rather than discarding the held-out data.
        en_heldout = list({r['text_key']: r for r in reversed(en_heldout)}.values())
        random.Random(SEED + 1).shuffle(en_heldout)
        ljs_val_keys = {r['text_key'] for r in ljs_val}
        shared_validation = [r for r in en_heldout if r['text_key'] in ljs_val_keys]
        independent = [r for r in en_heldout if r['text_key'] not in ljs_val_keys]
        needed = max(0, 32 - len(shared_validation))
        en_val, en_test = shared_validation + independent[:needed], independent[needed:]
        if len(en_val) < 32 or len(en_test) < 16:
            raise RuntimeError(f'Need >=32 English validation and >=16 internal test utterances, got {len(en_val)}/{len(en_test)}; fixed external eval adds 200 English texts')
        validation = ljs_val + cn_val + en_val
        test = cn_test + en_test
        eval_rows = records(DATA / 'eval_manifest.jsonl')
        heldout_keys = {r['text_key'] for r in validation + test} | {tuple(r['token_ids']) for r in eval_rows}
        old_ljs_rows = len(ljs_train)
        ljs_train = [r for r in ljs_train if r['text_key'] not in heldout_keys]
        cn_train = [r for r in cn_train if r['text_key'] not in heldout_keys]
        en_pool = [r for r in en_pool if r['text_key'] not in heldout_keys]
        if len(en_pool) < 2756:
            raise RuntimeError(f'Need 2756 quality-passed training English texts, got {len(en_pool)}')
        # Source order prioritizes the historical 2756-text recipe; retain only one per text.
        en_train, en_seen = [], set()
        for row in en_pool:
            if row['text_key'] not in en_seen:
                en_train.append(row)
                en_seen.add(row['text_key'])
            if len(en_train) == 2756:
                break
        assert len(en_train) == 2756
        majestic = cn_train + en_train
        rng = random.Random(SEED)
        balanced_majestic = majestic + rng.choices(majestic, k=len(ljs_train) - len(majestic))
        training = ljs_train + balanced_majestic
        rng.shuffle(training)
        for split, rows in [('train', training), ('val', validation), ('test', test)]:
            (DATA / f'{split}.txt').write_text(''.join(manifest_line(r) for r in rows))
            (DATA / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        train_paths = {r['audio'] for r in training}
        assert not train_paths & {r['audio'] for r in validation + test}
        assert not {r['text_key'] for r in training} & heldout_keys
        summary = {'status': 'prepared', 'seed': SEED, 'sample_rate': 24000, 'n_feats': 100,
                   'n_vocab': 173, 'speakers': {'0': 'LJSpeech', '1': 'MajesticVoice'},
                   'train_rows': len(training), 'validation_rows': len(validation), 'test_rows': len(test),
                   'train_rows_per_speaker': dict(Counter(r['speaker'] for r in training)),
                   'majestic_unique_chinese_train': len(cn_train), 'majestic_unique_english_train': len(en_train),
                   'ljs_train_rows_removed_for_heldout_text_overlap': old_ljs_rows - len(ljs_train),
                   'train_unique_audio': len(train_paths), 'train_heldout_audio_overlap': 0,
                   'train_segment_rows': sum('start' in r for r in training),
                   'train_heldout_phoneme_overlap': 0, 'cleaners': CLEANERS,
                   'generated_rows_excluded_by_frontend': len(frontend_exclusions),
                   'source_chinese_audit': str(VOICE / 'quality/final_audit.json'),
                   'source_english_audit': str(VOICE / 'english/quality/final_audit.json')}
        (DATA / 'training_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    main(parser.parse_args())
