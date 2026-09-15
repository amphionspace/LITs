"""Stage-one shared-speaker evaluation; no target-voice similarity claim."""
import argparse
import json
from pathlib import Path
from training.majestic_voice import evaluate_checkpoint as common

common.DATA=Path('/119010446/tts-assets/data_24k/foundation')

def metrics(args):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from quality_metrics import DNSMOS,audio16
    torch.set_num_threads(4)
    scorer=DNSMOS()
    rows=[r for r in common.records(args.output/'synthesis.jsonl') if 'audio' in r]
    def score(row):
        try:
            return {'id':row['id'],**scorer(audio16(row['audio']))}
        except Exception as exc:
            return {'id':row['id'],'metrics_error':f'{type(exc).__name__}: {exc}'}
    with ThreadPoolExecutor(8) as pool,(args.output/'metrics.jsonl').open('w') as stream:
        for row in pool.map(score,rows):
            stream.write(json.dumps(row)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['synthesize','asr','metrics','summarize'],required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--per-group-limit',type=int,default=0)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    {'synthesize':common.synthesize,'asr':common.asr,'metrics':metrics,'summarize':common.summarize}[args.stage](args)
