"""Read existing real-speech transcripts only; do not load audio or codec files."""
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from common import ROOT,atomic_json,norm
from prepare_corpus import classify,eligibility


def main():
    base=Path('/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training')
    source=base/'emilia-full-en-zh/train.jsonl';out=ROOT/'text/external/Emilia2';out.mkdir(parents=True,exist_ok=True)
    # These source validation recordings are never reused as synthesis text.
    heldout_text=set();heldout_groups=set();heldout_files=[]
    for path in sorted(base.glob('*/val.jsonl'))+sorted(base.glob('*/test.jsonl')):
        heldout_files.append(str(path))
        for line in path.open():
            r=json.loads(line);heldout_text.add(norm(r['text']))
            if isinstance(r.get('source'),dict) and r['source'].get('recording_id'):heldout_groups.add(r['source']['recording_id'])
    counts=Counter();seen=set();estimated=0;scanned=0;started=time.time()
    pattern=re.compile(rb'"text"\s*:\s*("(?:[^"\\]|\\.)*")')
    with source.open('rb') as inp,(out/'mixed_candidates.jsonl.tmp').open('w') as dest:
        for line in inp:
            scanned+=1
            match=pattern.search(line)
            if not match:counts['no_text']+=1;continue
            text=json.loads(match[1]);lang,han,words,letters=classify(text)
            if lang!='mixed':continue
            counts['mixed_seen']+=1
            reason=eligibility(dict(text=text,ids=[3]*20))
            if reason:counts[reason]+=1;continue
            key=norm(text)
            if key in seen or key in heldout_text:counts['duplicate_or_heldout_text']+=1;continue
            r=json.loads(line);meta=r.get('source',{});group=meta.get('recording_id')
            if not group or group in heldout_groups:counts['heldout_or_missing_recording']+=1;continue
            if meta.get('type') not in ('short','long','dialogue'):counts['unknown_source_type']+=1;continue
            seconds=han/4.5+words/2.5
            record=dict(text=text,source='Emilia2',source_id=r['id'],source_group='Emilia2/'+group,
                source_manifest=str(source),source_line=scanned,source_audio=r.get('audio_source'),
                source_seconds=r['duration'],estimated_seconds=seconds,source_metadata=meta,
                text_origin='real_speech_transcription',language='mixed')
            dest.write(json.dumps(record,ensure_ascii=False)+'\n');seen.add(key);estimated+=seconds;counts['prefiltered']+=1
            if counts['prefiltered']%1000==0:
                status=dict(scanned=scanned,counts=dict(counts),estimated_hours=estimated/3600,elapsed_seconds=time.time()-started)
                atomic_json(out/'scan_progress.json',status);print(json.dumps(status),flush=True);dest.flush()
            if estimated>=200*3600:break
    (out/'mixed_candidates.jsonl.tmp').replace(out/'mixed_candidates.jsonl')
    report=dict(status='prefilter_complete',source=str(source),source_size=source.stat().st_size,source_mtime_ns=source.stat().st_mtime_ns,
        scanned=scanned,counts=dict(counts),estimated_hours=estimated/3600,heldout_files=heldout_files,
        excluded_heldout_recordings=len(heldout_groups),audio_read=False,
        remaining_checks=['LITs frontend','historical and current pool exact/phoneme dedup','near duplicates before admission'],
        note='Duration is a text-rate estimate, not accepted synthesized hours.',completed_at_unix=time.time())
    atomic_json(out/'scan_report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
