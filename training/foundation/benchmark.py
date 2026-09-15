"""Measure full forward/backward/Adam memory on the longest real examples."""
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import hydra
from hydra import compose, initialize_config_dir
from training.foundation.data import IndexedAudio
from lits.data.text_mel_datamodule import TextMelBatchCollate

OUT = Path('/119010446/tts-assets/data_24k/foundation')

def main():
    torch.set_num_threads(4)
    torch.manual_seed(20260911)
    stats = json.loads((OUT/'mel_statistics.json').read_text())
    os.environ.update(TRAIN_FILELIST=str(OUT/'dataset.sqlite'), VALID_FILELIST=str(OUT/'dataset.sqlite'),
                      N_SPKS='2', MEL_MEAN=str(stats['mel_mean']), MEL_STD=str(stats['mel_std']), PROJECT_ROOT=str(Path(__file__).resolve().parents[2]))
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2]/'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=['experiment=en-zh','model.optimizer.lr=0.0001'])
    dataset = IndexedAudio(OUT/'dataset.sqlite','train',stats)
    longest = np.argsort(dataset.lengths)[-1000:]
    # Include the longest phoneme sequences among near-20-second utterances.
    longest = sorted(longest, key=lambda i:len(dataset.row(int(i))['ids']), reverse=True)[:192]
    examples = [dataset[int(i)] for i in longest]
    model = hydra.utils.instantiate(cfg.model).cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    results = []
    for size in json.loads(os.environ.get('BENCHMARK_BATCHES', '[16,32,48,64,80,96,112,128,144,160,192]')):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        batch = TextMelBatchCollate(2)(examples[:size])
        batch = {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        try:
            start = time.time()
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    losses = model(batch['x'],batch['x_lengths'],batch['y'],batch['y_lengths'],spks=batch['spks'],x_tones=batch['x_tones'])
                    loss = sum(losses[:3])
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step()
            torch.cuda.synchronize()
            row = {'batch_size':size,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                   'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'seconds_per_step':(time.time()-start)/2,
                   'mel_frames':batch['y'].shape[-1],'text_tokens':batch['x'].shape[-1], 'loss':float(loss.detach()),'status':'passed'}
            results.append(row)
            print(json.dumps(row),flush=True)
            if row['peak_allocated_gib'] > 68:
                break
            del losses,loss,batch
        except torch.cuda.OutOfMemoryError:
            results.append({'batch_size':size,'status':'oom'})
            print(json.dumps(results[-1]),flush=True)
            break
    valid = [r for r in results if r['status']=='passed' and r['peak_allocated_gib'] < 70]
    chosen = valid[-1]['batch_size']
    result = {'chosen_batch_size_per_gpu':chosen,'results':results,'benchmark_weights_discarded':True,
              'initialization_for_training':'random, independent of benchmark'}
    (OUT/os.environ.get('CAPACITY_FILE','capacity.json')).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
