"""Shared, explicit configuration for iMF preflight and the production run."""

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def environment(run):
    plan = read_json(run / 'plan.json')
    stats = plan['data_statistics']
    return dict(
        os.environ, PROJECT_ROOT=str(REPO), PYTHONPATH=str(REPO),
        TRAIN_FILELIST=str(run / 'data/train.txt'), VALID_FILELIST=str(run / 'data/val.txt'),
        STAGE2_DATA=str(run / 'data'), N_SPKS='2',
        MEL_MEAN=str(stats['mel_mean']), MEL_STD=str(stats['mel_std']),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
        HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
    )


def overrides(run, batch_size, *, preflight=False):
    plan = read_json(run / 'plan.json')
    accumulation = plan['effective_batch'] // (len(plan['devices']) * batch_size)
    if accumulation * len(plan['devices']) * batch_size != plan['effective_batch']:
        raise ValueError('Microbatch does not divide the effective batch')
    result = [
        'experiment=en-zh-imf-stage2', 'ckpt_path=null',
        f'init_ckpt_path={run}/initialization.ckpt', 'init_reset_speaker_embeddings=false',
        f'run_name={run.name}', f'seed={plan["seed"]}', f'data.seed={plan["seed"]}',
        f'data.batch_size={batch_size}', 'data.num_workers=4',
        f'model.optimizer.lr={plan["peak_lr"]}',
        'trainer.devices=[0,1,2,3]', 'trainer.precision=bf16-mixed',
        f'+trainer.accumulate_grad_batches={accumulation}',
        f'trainer.max_steps={plan["max_steps"]}', f'trainer.max_epochs={plan["max_epochs"]}',
        'trainer.check_val_every_n_epoch=null',
        f'+trainer.val_check_interval={1000 * accumulation}',
        '+trainer.num_sanity_val_steps=2', 'trainer.gradient_clip_val=5.0',
        'callbacks.model_checkpoint.every_n_epochs=null',
        'callbacks.model_checkpoint.every_n_train_steps=1000',
        'callbacks.model_checkpoint.monitor=step',
        "callbacks.model_checkpoint.filename='step_{step:08.0f}'",
        'callbacks.model_checkpoint.save_top_k=3',
        '+callbacks.imf_schedule._target_=training.majestic_scratch.schedule.WarmupStableDecay',
        f'+callbacks.imf_schedule.warmup_steps={plan["warmup_steps"]}',
        f'+callbacks.imf_schedule.total_steps={plan["max_steps"]}',
        f'+callbacks.imf_schedule.peak_lr={plan["peak_lr"]}',
        f'+callbacks.imf_schedule.final_lr={plan["final_lr"]}',
        f'+callbacks.imf_schedule.decay_fraction={plan["decay_fraction"]}',
        '+callbacks.imf_audit._target_=training.imf.callbacks.IMFAudit',
        f'+callbacks.imf_audit.run_dir={run}',
        '+callbacks.training_state._target_=training.common.callbacks.TrainingState',
        'callbacks.rich_progress_bar=null', 'trainer.enable_progress_bar=false',
        'test=false', f'hydra.run.dir={run}',
    ]
    if preflight:
        result = [x for x in result if not x.lstrip('+').startswith(('callbacks.', 'trainer.max_', 'hydra.run.dir'))]
        result += [
            'callbacks.model_checkpoint=null', 'callbacks.best_prior_checkpoint=null',
            'callbacks.model_summary=null', 'callbacks.rich_progress_bar=null',
            '+trainer.enable_checkpointing=false',
            'trainer.max_steps=2', 'trainer.max_epochs=1',
            f'+trainer.limit_train_batches={2 * accumulation}', '+trainer.limit_val_batches=1',
            f'hydra.run.dir={run}/preflight_ddp',
        ]
    return result


def initialized_model(run):
    import hydra
    import lightning as L
    import torch
    from hydra import compose, initialize_config_dir
    from lits.utils.imf_checkpoint import checkpoint_statistics, load_imf_initial_weights

    plan = read_json(run / 'plan.json')
    os.environ.update(environment(run))
    with initialize_config_dir(config_dir=str(REPO / 'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=[
            'experiment=en-zh-imf-stage2', f'init_ckpt_path={run}/initialization.ckpt',
            f'trainer.max_steps={plan["max_steps"]}', f'model.optimizer.lr={plan["peak_lr"]}',
        ])
    L.seed_everything(plan['seed'], workers=True)
    checkpoint = torch.load(run / 'initialization.ckpt', map_location='cpu', weights_only=False)
    cfg.data.data_statistics = checkpoint_statistics(checkpoint)
    model = hydra.utils.instantiate(cfg.model)
    load_imf_initial_weights(model, checkpoint, reset_speaker_embeddings=False)
    return model
