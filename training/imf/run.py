"""Freeze an iMF recipe and start it after the preceding FM run and evaluation finish."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from training.imf.recipe import REPO, environment, overrides, read_json, write_json
from training.stage2.prepare import digest


def alive(pid):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def previous_run_ready(previous):
    launch = read_json(previous / 'launch.json')
    state = read_json(previous / 'training_state.json')
    if launch['status'] == 'failed':
        raise RuntimeError('Preceding FM training failed; automatic handoff is blocked')
    final_step = read_json(previous / 'plan.json')['max_steps']
    if launch['status'] != 'complete' or state['status'] != 'complete' or state['global_step'] < final_step:
        return False, f'FM training: step {state["global_step"]}/{final_step}'
    if launch.get('returncode') != 0 or not (previous / 'checkpoints/final.ckpt').is_file():
        raise RuntimeError('FM completion is missing its successful exit or final checkpoint')
    if alive(launch['training_pid']):
        return False, 'Waiting for the FM training process to exit'
    done_path = previous / 'eval/evaluated.jsonl'
    done = {r['global_step'] for r in map(json.loads, done_path.read_text().splitlines())} if done_path.exists() else set()
    expected = {int(p.stem.split('_')[-1]) for p in (previous / 'evaluation_checkpoints').glob('step_*.ckpt')}
    expected.add(final_step)
    if not expected <= done:
        if not alive(launch['eval_watcher_pid']):
            raise RuntimeError('FM evaluator exited before draining its checkpoint queue')
        return False, f'Waiting for FM evaluations: {sorted(expected - done)}'
    report = read_json(previous / f'eval/step_{final_step:08d}/summary.json')
    if report['status'] != 'complete' or report['samples'] != 650:
        raise RuntimeError('Final FM evaluation is incomplete')
    if any(group['evaluation_failures'] for group in report['groups'].values()):
        raise RuntimeError('Final FM evaluation contains failed samples; inspect before handoff')
    if alive(launch['eval_watcher_pid']):
        return False, 'Waiting for the FM evaluation process to exit'
    return True, 'FM training and all queued evaluations completed'


def gpu_users():
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def freeze_source(destination):
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise RuntimeError('Commit the reviewed iMF source before freezing a run')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=REPO).decode().split('\0')
    # The compiled extension is a build artifact; all Python/config/resource sources are tracked.
    names += [str(p.relative_to(REPO)) for p in (REPO / 'lits/utils/monotonic_align').glob('core*.so')]
    hashes = {}
    for name in sorted(set(filter(None, names))):
        source = REPO / name
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        else:
            shutil.copy2(source, target)
        hashes[name] = digest(target)
    if not list((destination / 'lits/utils/monotonic_align').glob('core*.so')):
        raise RuntimeError('Build the MAS extension before freezing the training source')
    if (destination / 'vocos/generator.ckpt').stat().st_size < 1_000_000:
        raise RuntimeError('Vocos weights must be materialized, not a Git LFS pointer')
    return dict(revision=revision, sha256=hashes)


def prepare_run(run, previous):
    """CPU-only: freeze existing manifests, initialization and executable source."""
    import hydra
    import lightning as L
    import torch
    from hydra import compose, initialize_config_dir
    from lits.utils.imf_checkpoint import load_imf_initial_weights
    from training.majestic_scratch.config import parameter_digest

    if run.exists():
        raise RuntimeError('Run directory already exists; preparation never overwrites a run')
    old = read_json(previous / 'plan.json')
    assert old['source_global_step'] == 21000 and old['max_epochs'] == 500
    assert digest(Path(old['source_checkpoint'])) == old['source_sha256']
    assert digest(previous / 'initialization.ckpt') == old['initialization_sha256']
    for name, expected in old['manifest_hashes'].items():
        assert digest(previous / 'data' / name) == expected, name
    run.mkdir(parents=True)
    shutil.copytree(previous / 'data', run / 'data')
    source = freeze_source(run / 'source')
    checkpoint = torch.load(previous / 'initialization.ckpt', map_location='cpu', weights_only=False)
    assert checkpoint['global_step'] == 0 and 'optimizer_states' not in checkpoint
    torch.set_num_threads(4)
    os.environ.update(PROJECT_ROOT=str(REPO), STAGE2_DATA=str(run / 'data'),
                      TRAIN_FILELIST=str(run / 'data/train.txt'), VALID_FILELIST=str(run / 'data/val.txt'), N_SPKS='2')
    L.seed_everything(old['seed'], workers=True)
    with initialize_config_dir(config_dir=str(REPO / 'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=[
            'experiment=en-zh-imf-stage2', f'init_ckpt_path={previous}/initialization.ckpt',
            f'trainer.max_steps={old["max_steps"]}', f'model.optimizer.lr={old["peak_lr"]}',
        ])
    cfg.data.data_statistics = old['data_statistics']
    model = hydra.utils.instantiate(cfg.model)
    migration = load_imf_initial_weights(model, checkpoint, reset_speaker_embeddings=False)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in checkpoint['state_dict'].items())
    torch.save(dict(state_dict=model.state_dict(), hyper_parameters=dict(model.hparams), epoch=0, global_step=0,
                    **{'pytorch-lightning_version': L.__version__}), run / 'initialization.ckpt')
    plan = {key: old[key] for key in (
        'source_checkpoint', 'source_sha256', 'source_global_step', 'data_statistics', 'train_rows',
        'validation_rows', 'test_rows', 'max_epochs', 'max_steps', 'peak_lr', 'final_lr', 'warmup_steps',
        'decay_fraction', 'effective_batch', 'seed', 'active_speaker_ids', 'diagnostic_steps',
        'manifest_hashes', 'unique_audio_hours', 'speakers', 'evaluation_samples',
    )}
    plan.update(
        mode='backbone_init', objective='imf', recipe='stage2_imf_from_identical_fm_initialization',
        previous_run=str(previous), data_dir=str(run / 'data'), devices=[0, 1, 2, 3],
        steps_per_epoch=old['sampling_plan']['optimizer_steps_per_epoch'],
        initialization_sha256=digest(run / 'initialization.ckpt'),
        initial_parameter_sha256=parameter_digest(model),
        fm_initialization=str(previous / 'initialization.ckpt'),
        fm_initialization_sha256=old['initialization_sha256'],
        initialization='exact FM starting tensors including seeded speaker rows; new interval/v head; fresh Adam',
        source_snapshot=source, sampling_time_grid=[0.0, 0.5, 1.0],
        loss_weights=dict(duration=1, prior=1, flow=1), capacity_candidates=[24, 16, 8, 4],
        batch_size_per_gpu=None, accumulate_grad_batches=None,
        evaluation_interval_steps=5000, validation_every_steps=1000,
    )
    write_json(run / 'plan.json', plan)
    write_json(run / 'migration.json', dict(migration, all_fm_starting_tensors_exact=True, speaker_rows_match_fm_start=True))
    write_json(run / 'control.json', dict(enabled=True))
    write_json(run / 'supervisor_status.json', dict(status='prepared', updated_at_unix=time.time()))
    print(json.dumps(dict(status='prepared', run_dir=str(run), source_revision=source['revision'])), flush=True)


def verify_artifacts(run):
    plan = read_json(run / 'plan.json')
    assert digest(run / 'initialization.ckpt') == plan['initialization_sha256']
    for name, expected in plan['manifest_hashes'].items():
        assert digest(run / 'data' / name) == expected, name
    for name, expected in plan['source_snapshot']['sha256'].items():
        assert digest(run / 'source' / name) == expected, name
        assert digest(REPO / name) == expected, f'Working source changed after preparation: {name}'
    return plan


def execute_probe(command, run, name, env):
    with (run / f'{name}.log').open('w') as log:
        return subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def launch_run(run):
    plan = verify_artifacts(run)
    env = environment(run)
    env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    for batch_size in plan['capacity_candidates']:
        code = execute_probe(
            [sys.executable, str(REPO / 'training/imf/preflight.py'), '--run-dir', str(run),
             '--batch-size', str(batch_size)], run, f'capacity_{batch_size}',
            dict(env, CUDA_VISIBLE_DEVICES='0'),
        )
        if code == 0:
            break
        if code != 3:
            raise RuntimeError(f'iMF capacity/gradient preflight failed for batch {batch_size}; inspect its log')
    else:
        raise RuntimeError('No tested microbatch met the memory/gradient gate')
    plan.update(batch_size_per_gpu=batch_size, accumulate_grad_batches=plan['effective_batch'] // (4 * batch_size))
    write_json(run / 'plan.json', plan)
    command = [sys.executable, str(REPO / 'lits/train.py'), *overrides(run, batch_size, preflight=True)]
    if execute_probe(command, run, 'preflight_ddp', env) != 0:
        raise RuntimeError('Four-GPU iMF preflight failed; formal training was not started')
    # Probe ranks have exited, so their CUDA allocations are gone before the
    # fresh production model is constructed from the frozen initialization.
    write_json(run / 'preflight.json', dict(status='passed', capacity=read_json(run / f'capacity_{batch_size}.json'),
                                          ddp_optimizer_steps=2, ddp_devices=4, probe_weights_discarded=True))
    if not read_json(run / 'control.json')['enabled']:
        raise RuntimeError('Handoff disabled during preflight; formal training was not started')
    if (run / 'launch.json').exists():
        raise RuntimeError('Existing launch record: never start this run from scratch again')
    command = [sys.executable, str(REPO / 'lits/train.py'), *overrides(run, batch_size)]
    with (run / 'training.log').open('w') as log:
        child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    with (run / 'eval_watcher.log').open('w') as log:
        watcher = subprocess.Popen(
            [sys.executable, str(REPO / 'training/majestic_scratch/watch_eval.py'), '--run-dir', str(run),
             '--training-pid', str(child.pid), '--gpu', '0'], cwd=REPO, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    launch = dict(status='running', training_pid=child.pid, eval_watcher_pid=watcher.pid,
                  command=command, started_at_unix=time.time(), max_steps=plan['max_steps'],
                  batch_size_per_gpu=batch_size, accumulate_grad_batches=plan['accumulate_grad_batches'])
    write_json(run / 'launch.json', launch)
    write_json(run / 'supervisor_status.json', dict(status='training', **{k: launch[k] for k in ('training_pid', 'eval_watcher_pid')}))
    code = child.wait()
    launch.update(status='complete' if code == 0 else 'failed', returncode=code, finished_at_unix=time.time())
    write_json(run / 'launch.json', launch)
    if code:
        raise RuntimeError(f'iMF training exited {code}; checkpoints retained, no automatic fresh restart')
    write_json(run / 'supervisor_status.json', dict(status='training_complete_evaluation_draining', updated_at_unix=time.time()))


def supervise(run):
    lock = (run / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read_json(run / 'plan.json')
    if (run / 'launch.json').exists():
        raise RuntimeError('Run already launched; automatic fresh restart is forbidden')
    try:
        while True:
            enabled = read_json(run / 'control.json')['enabled']
            ready, reason = previous_run_ready(Path(plan['previous_run'])) if enabled else (False, 'Handoff disabled')
            users = gpu_users() if ready else []
            if users:
                ready, reason = False, f'Waiting for GPU compute processes to release devices: {users}'
            write_json(run / 'supervisor_status.json', dict(
                status='preflight' if ready else 'waiting', reason=reason,
                supervisor_pid=os.getpid(), updated_at_unix=time.time(), previous_run=plan['previous_run'],
            ))
            if ready:
                launch_run(run)
                return
            time.sleep(30)
    except Exception as exc:
        write_json(run / 'supervisor_status.json', dict(status='failed', error=repr(exc),
                   supervisor_pid=os.getpid(), updated_at_unix=time.time(), automatic_training_restart=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'supervise'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--after-run', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.after_run is None:
            parser.error('prepare requires --after-run')
        prepare_run(args.run_dir.resolve(), args.after_run.resolve())
    else:
        supervise(args.run_dir.resolve())
