"""Read a fixed byte snapshot of existing event files; write no training state."""
import json
import struct
from collections import defaultdict
from pathlib import Path
import numpy as np
from tensorboard.compat.proto.event_pb2 import Event

ROOT=Path('/119010446/tts-assets/training_runs')
OUT=Path('/119010446/tts-assets/diagnostics/imf_tensorboard_20260916')
RUNS={'imf':'ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915',
      'fm':'ljs_majestic_100h_backbone21k_20260914'}
for model,run in RUNS.items():
    series=defaultdict(list);images=defaultdict(list);files=[];sessions=[]
    for p in sorted((ROOT/run/'tensorboard/version_0').glob('events.out.tfevents.*')):
        size=p.stat().st_size;offset=0;records=0
        with p.open('rb') as f:
            while offset+12<=size:
                h=f.read(12)
                if len(h)<12:break
                length=struct.unpack('<Q',h[:8])[0]
                if offset+16+length>size:break
                payload=f.read(length);crc=f.read(4)
                if len(payload)!=length or len(crc)!=4:break
                e=Event.FromString(payload);offset+=16+length;records+=1
                if e.HasField('session_log'):
                    sessions.append({'file':str(p),'step':e.step,'status':e.session_log.status,'wall_time':e.wall_time})
                    if e.session_log.status==1:
                        # Match TensorBoard restart purging, including step-indexed images.
                        for tag in series:series[tag]=[x for x in series[tag] if x[0]<e.step]
                        for tag in images:images[tag]=[x for x in images[tag] if x[0]<e.step]
                for v in e.summary.value:
                    if v.HasField('simple_value'):
                        series[v.tag].append((e.step,e.wall_time,v.simple_value))
                    elif v.HasField('image'):
                        images[v.tag].append((e.step,e.wall_time,v.image.width,v.image.height))
        files.append({'path':str(p),'snapshot_bytes':size,'parsed_bytes':offset,'records':records})
        print(model,p.name,records,len(series),flush=True)
    arrays={tag:np.asarray(values,dtype=np.float64) for tag,values in series.items()}
    np.savez_compressed(OUT/f'{model}_scalars.npz',**arrays)
    metadata={'run':str(ROOT/run),'files':files,'sessions':sessions,'images':dict(images),
              'scalar_tags':len(arrays),'scalar_events':sum(len(v) for v in arrays.values())}
    (OUT/f'{model}_event_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(model,'DONE',metadata['scalar_tags'],metadata['scalar_events'],flush=True)
