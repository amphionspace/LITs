"""Use the frozen stage-two evaluation protocol with the existing metrics stack."""
import argparse
import os
from pathlib import Path
from training.majestic_voice import evaluate_checkpoint as common

common.DATA=Path(os.environ['STAGE2_DATA'])

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['synthesize','asr','metrics','summarize'],required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--per-group-limit',type=int,default=0)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    getattr(common,args.stage)(args)
