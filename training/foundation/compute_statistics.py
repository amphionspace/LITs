"""Deterministic, source-balanced training-only Mel normalization estimate."""
import concurrent.futures as cf
import json
import math
from pathlib import Path
import torch
from training.foundation.data import IndexedAudio

OUT = Path('/119010446/tts-assets/data_24k/foundation')

def main():
    torch.set_num_threads(1)
    dataset = IndexedAudio(OUT / 'dataset.sqlite', 'train', {'mel_mean': 0., 'mel_std': 1.}, 2048)
    # Open a separate SQLite connection for each thread by reading metadata first.
    rows = [dataset.row(i) for i in range(len(dataset))]
    import numpy as np
    import soundfile as sf
    import soxr
    from lits.utils.audio import mel_spectrogram
    mel_spectrogram(torch.zeros(1,4096),2048,100,24000,384,1536,0,12000)
    def analyze(r):
        audio, rate = sf.read(r['audio'], dtype='float32')
        if rate != 24000:
            audio = soxr.resample(audio,rate,24000,quality='HQ')
        mel = mel_spectrogram(torch.from_numpy(audio)[None],2048,100,24000,384,1536,0,12000).double()
        assert torch.isfinite(mel).all()
        return mel.sum().item(),mel.square().sum().item(),mel.numel()
    with cf.ThreadPoolExecutor(16) as pool:
        values = list(pool.map(analyze,rows))
    n = sum(r[2] for r in values)
    mean = math.fsum(r[0] for r in values)/n
    std = math.sqrt(math.fsum(r[1] for r in values)/n-mean*mean)
    result = {'mel_mean':mean,'mel_std':std,'method':'estimate from deterministic 2048 training utterances per source',
              'seed':20260911,'samples':len(rows),'mel_elements':n,'sample_ids':dataset.ids.tolist()}
    (OUT/'mel_statistics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sample_ids'}),flush=True)

if __name__=='__main__': main()
