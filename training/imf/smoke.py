"""Bounded real-checkpoint/data validation; no optimizer step or training launch."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time


def main(args):
    import hydra
    import lightning as L
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from lits.data.text_mel_datamodule import TextMelBatchCollate
    from lits.models.lits import LITS
    from lits.models.components.text_encoder import RotaryPositionalEmbeddings
    from lits.models.components.utils import EspnetRelPositionalEncoding
    from lits.utils.imf_checkpoint import checkpoint_statistics, load_imf_initial_weights
    from training.stage2.data import Stage2Audio

    root = Path(__file__).resolve().parents[2]
    os.environ.update(PROJECT_ROOT=str(root), STAGE2_DATA=str(args.data_dir),
                      TRAIN_FILELIST=str(args.data_dir/'train.txt'), VALID_FILELIST=str(args.data_dir/'val.txt'))
    torch.set_num_threads(2)
    torch.manual_seed(20260915)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    with initialize_config_dir(config_dir=str(root/'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=['experiment=en-zh-imf-stage2',
            f'init_ckpt_path={args.checkpoint}', 'trainer.max_steps=1'])
    cfg.data.data_statistics = checkpoint_statistics(checkpoint)
    model = hydra.utils.instantiate(cfg.model)
    migration = load_imf_initial_weights(model, checkpoint, reset_speaker_embeddings=True)
    for name, tensor in checkpoint['state_dict'].items():
        if name != 'spk_emb.weight':
            assert torch.equal(model.state_dict()[name], tensor), name
    del checkpoint
    dataset = Stage2Audio(args.data_dir, 'val', dict(cfg.data.data_statistics))
    indices = [int(pool[np.argmin(dataset.lengths[pool])])
               for speaker in sorted(set(dataset.sources))
               for pool in [np.flatnonzero(dataset.sources == speaker)]]
    if len(indices) < 2:
        indices += [i for i in np.argsort(dataset.lengths).tolist() if i not in indices][:2-len(indices)]
    batch = TextMelBatchCollate(2)([dataset[i] for i in indices])
    batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    model = model.to(args.device).train()
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.autocast(device_type=args.device.split(':')[0], dtype=torch.bfloat16):
        losses = model(batch['x'], batch['x_lengths'], batch['y'], batch['y_lengths'],
                       spks=batch['spks'], x_tones=batch['x_tones'])
        loss = sum(losses[:3])
    assert torch.isfinite(loss)
    loss.backward()
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            group = ('speaker' if name.startswith('spk_emb.') else
                     'duration' if name.startswith('encoder.proj_w.') else
                     'text_encoder' if name.startswith('encoder.') else
                     'condition_encoder' if name.startswith('decoder.encoder.') else
                     'interval' if '.interval_projector.' in name else
                     'aux_v' if '.v_' in name else 'flow_u_and_trunk')
            groups[group] = groups.get(group, 0) + int(torch.count_nonzero(parameter.grad) > 0)
    assert all(groups.get(name, 0) > 0 for name in
               ('speaker', 'duration', 'text_encoder', 'condition_encoder', 'interval', 'aux_v', 'flow_u_and_trunk')), groups
    for speaker in batch['spks'].unique():
        assert model.spk_emb.weight.grad[speaker].abs().sum() > 0
    report = dict(status='passed', source_checkpoint=str(args.checkpoint), data_dir=str(args.data_dir),
                  device=args.device, migration=migration, batch_indices=indices,
                  losses=[float(v.detach()) for v in losses[:3]],
                  imf_metrics={k: float(v) for k, v in model.decoder.last_loss_stats.items()},
                  nonzero_gradient_tensors=groups, optimizer_steps=0)
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        validation = model(batch['x'], batch['x_lengths'], batch['y'], batch['y_lengths'],
                           spks=batch['spks'], x_tones=batch['x_tones'])
        assert all(torch.isfinite(v) for v in validation[:3])

    with tempfile.TemporaryDirectory(prefix='lits-imf-smoke-') as directory:
        path = Path(directory)/'imf.ckpt'
        torch.save(dict(state_dict=model.state_dict(), hyper_parameters=dict(model.hparams),
                        global_step=0, epoch=0, **{'pytorch-lightning_version': L.__version__}), path)
        restored = LITS.load_from_checkpoint(str(path), map_location='cpu', weights_only=False).to(args.device).eval()
        assert all(torch.equal(tensor, restored.state_dict()[name]) for name, tensor in model.state_dict().items())
        outputs = []
        call_counts = []
        for tested in (model, restored):
            # Compare fresh utterances. The existing text/condition encoders
            # cache positions outside state_dict; mixed-precision caches are
            # runtime state and must not be compared with a cold FP32 reload.
            tested.decoder.reset_encoder_cache()
            for module in tested.modules():
                if isinstance(module, RotaryPositionalEmbeddings):
                    module.cos_cached = module.sin_cached = None
                elif isinstance(module, EspnetRelPositionalEncoding):
                    module.pe = None
            calls = []
            handle = tested.decoder.estimator.register_forward_pre_hook(lambda *_: calls.append(1))
            def no_aux(*_):
                raise AssertionError('Auxiliary v head executed during synthesis')
            aux_handle = tested.decoder.estimator.v_final_proj.register_forward_pre_hook(no_aux)
            torch.manual_seed(47)
            with torch.inference_mode():
                result = tested.synthesise(batch['x'][:1], batch['x_lengths'][:1],
                                           spks=batch['spks'][:1], x_tones=batch['x_tones'][:1])
            assert len(calls) == 2 and torch.isfinite(result['mel']).all()
            outputs.append(result['mel'])
            call_counts.append(len(calls))
            handle.remove()
            aux_handle.remove()
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    report.update(checkpoint_reload_exact=True, synthesis_calls=call_counts,
                  nonpersistent_caches_reset_before_synthesis=True,
                  sampling_time_grid=list(model.decoder.sampling_time_grid),
                  validation_inference_mode_passed=True, seconds=time.monotonic()-started)
    if args.device.startswith('cuda'):
        report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
