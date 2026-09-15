"""Match recorded total duration while preserving the predicted timing pattern."""
import argparse
import json
from pathlib import Path


def main(root):
    import numpy as np
    import soundfile as sf
    import torch
    from lits.models.lits import LITS
    from vocos.vocoder import load_vocos_vocoder
    repo=Path(__file__).resolve().parents[2]
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.075)
    meta=json.loads((root/'metadata.json').read_text())
    model=LITS.load_from_checkpoint(meta['checkpoint'],map_location='cpu',weights_only=False).to('cuda:0').eval().requires_grad_(False)
    vocoder,_=load_vocos_vocoder(str(repo/'vocos/generator.ckpt'),torch.device('cuda:0'),repo)
    out=root/'duration_control';out.mkdir(exist_ok=False)
    selected=json.loads((root/'selected_samples.json').read_text())
    with (out/'synthesis.jsonl').open('w',buffering=1) as stream,torch.inference_mode():
        for index,row in enumerate(selected):
            if not row['real_audio'] or row['source']!='Premium':continue
            name=row['diagnostic_id'];d=json.loads((root/name/'diagnostic.json').read_text());target=d['real_frames']
            x=torch.tensor([row['ids']],device='cuda:0');tones=torch.tensor([row['tones']],device='cuda:0');xl=torch.tensor([x.shape[-1]],device='cuda:0');sid=torch.tensor([0],device='cuda:0')
            scale=target/d['predicted_frames']
            for _ in range(6):
                hidden=model.get_hidden_mel(x,xl,spks=sid,x_tones=tones,length_scale=scale)
                frames=int(hidden['y_max_length'])
                if abs(frames-target)<=1:break
                scale*=target/frames
            assert abs(frames-target)<=2,(name,frames,target)
            torch.manual_seed(2026091202+index);z=torch.randn(1,100,3000,device='cuda:0')
            mel=model.get_mel(hidden['mu_y'],hidden['y_mask'],10,1.,spks=hidden['spks'],streaming=False,z=z)[:,:,:frames]
            wave=vocoder(mel).squeeze().cpu().numpy()[:frames*384];assert np.isfinite(wave).all()
            mode='pred_matched_duration_10_s0';path=out/f'{name}.wav';sf.write(path,np.clip(wave,-1,1),24000,subtype='FLOAT')
            record=dict(id=f'{name}/{mode}',sample=name,mode=mode,source=row['source'],split=row['split'],group=row['group'],ref_text=row['text'],audio=str(path),duration=len(wave)/24000,
                length_scale=scale,target_frames=target,actual_frames=frames,original_predicted_frames=d['predicted_frames'])
            stream.write(json.dumps(record,ensure_ascii=False)+'\n');print(json.dumps(record,ensure_ascii=False),flush=True)
    print('DURATION_CONTROL_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
