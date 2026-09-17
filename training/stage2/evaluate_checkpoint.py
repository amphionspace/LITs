"""Use the frozen stage-two evaluation protocol with the existing metrics stack."""
import os
from pathlib import Path
from training.common import evaluate_checkpoint as common

common.DATA=Path(os.environ['STAGE2_DATA'])

if __name__=='__main__':
    args=common.build_parser().parse_args()
    common.DATA=args.data_dir
    args.output.mkdir(parents=True,exist_ok=True)
    getattr(common,args.stage)(args)
