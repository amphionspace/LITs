"""Monitor stage-two checkpoints with the 650-utterance speaker-aware protocol."""
import argparse
from pathlib import Path
from training.majestic_voice import watch_eval as common

common.CODE=Path(__file__).resolve().parent

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--training-pid',type=int,required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--interval-steps',type=int,default=2000)
    common.main(parser.parse_args())
