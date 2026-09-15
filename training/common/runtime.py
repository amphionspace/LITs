"""Interpreter paths for isolated synthesis, ASR and quality evaluation."""
from pathlib import Path

ASSETS = Path('/119010446/tts-assets')
PYTHONS = {'synthesize': ASSETS / 'chenmingjie/lits-tts/envs/lits/bin/python',
           'asr': ASSETS / '.venv-voxcpm2/bin/python',
           'metrics': Path('/119010446/UltraEval-Audio/envs/metrics/bin/python'),
           'summarize': ASSETS / 'chenmingjie/lits-tts/envs/lits/bin/python'}
