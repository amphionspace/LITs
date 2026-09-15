"""Shared model configuration and auditable fresh/backbone initialization."""
import hashlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def resolve_budget(training, train_rows, effective_batch=192):
    batches=train_rows//effective_batch
    if batches<1:raise ValueError('Training data must contain at least one full global batch')
    if training.get('budget_unit')=='epochs':
        epochs=int(training['max_epochs'])
        if epochs<1:raise ValueError('max_epochs must be positive')
        steps=epochs*batches
    else:
        epochs=-1;steps=int(training['max_steps'])
    if steps<=int(training.get('warmup_steps',1000)):
        raise ValueError('Training budget must exceed LR warmup')
    return dict(max_steps=steps,max_epochs=epochs,steps_per_epoch=batches,
                approximate_epochs=steps/batches,budget_unit=training.get('budget_unit','steps'))


def environment(run, stats):
    return dict(PROJECT_ROOT=str(REPO), TRAIN_FILELIST=str(run/'data/train.txt'),
                VALID_FILELIST=str(run/'data/val.txt'), STAGE2_DATA=str(run/'data'),
                N_SPKS='2', MEL_MEAN=str(stats['mel_mean']), MEL_STD=str(stats['mel_std']))


def model_overrides(peak_lr):
    return ['experiment=en-zh', 'ckpt_path=null', 'init_ckpt_path=null',
            f'model.optimizer.lr={peak_lr}', 'model.prior_encoder_lr=null',
            'model.duration_predictor_lr=null', 'data.load_durations=false',
            'model.duration_constrained_mas=true', 'model.prior_loss=true',
            'model.aux_loss_decay_start_step=-1', 'model.aux_loss_decay_end_step=-1']


def make_model(run, stats, peak_lr, seed):
    import hydra
    import lightning as L
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    os.environ.update(environment(run, stats))
    with initialize_config_dir(config_dir=str(REPO/'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=model_overrides(peak_lr))
    L.seed_everything(seed, workers=True)
    model = hydra.utils.instantiate(cfg.model)
    return model, OmegaConf.to_container(cfg.model, resolve=True)


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        digest.update(name.encode()); digest.update(str(tuple(p.shape)).encode())
        digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def backbone_metadata(path, expected_sha256):
    import torch
    from training.stage2.prepare import digest
    path=Path(path)
    assert digest(path)==expected_sha256
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    h=checkpoint['hyper_parameters']
    assert (h['n_vocab'],h['n_feats'],h['n_spks'],h['spk_emb_dim'])==(173,100,2,64)
    return dict(source_checkpoint=str(path),source_sha256=expected_sha256,source_global_step=int(checkpoint['global_step']),
                data_statistics={k:float(checkpoint['state_dict'][k]) for k in ('mel_mean','mel_std')})


def load_backbone(model, path, expected_sha256):
    """Copy acoustic tensors only; retain independently seeded speaker rows."""
    import torch
    from training.stage2.prepare import digest
    path=Path(path);assert digest(path)==expected_sha256
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    source=checkpoint['state_dict'];target=model.state_dict()
    assert source.keys()==target.keys()
    random_speakers=model.spk_emb.weight.detach().clone()
    for key,value in source.items():
        assert value.shape==target[key].shape and torch.isfinite(value).all(),key
        if key!='spk_emb.weight':target[key]=value
    model.load_state_dict(target,strict=True)
    assert torch.equal(model.spk_emb.weight,random_speakers)
    assert all(torch.equal(value,source[key]) for key,value in model.state_dict().items() if key!='spk_emb.weight')
    assert not torch.equal(random_speakers,source['spk_emb.weight'])
    return dict(backbone_tensors_exact=True,speaker_rows_reset=True,optimizer_loaded=False,scheduler_loaded=False)


def make_initialized_model(run, plan):
    import torch
    from training.stage2.prepare import digest
    model,cfg=make_model(run,plan['data_statistics'],plan['peak_lr'],plan['seed'])
    if plan['mode']=='backbone_init':
        path=run/'initialization.ckpt';assert digest(path)==plan['initialization_sha256']
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['global_step']==0 and 'optimizer_states' not in checkpoint and 'lr_schedulers' not in checkpoint
        random_speakers=model.spk_emb.weight.detach().clone()
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        assert torch.equal(model.spk_emb.weight,random_speakers)
    return model,cfg
