from pathlib import Path
from types import SimpleNamespace
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightning.pytorch.callbacks import ModelCheckpoint

from lits.callbacks.lr_downshift import LRDownshiftOnPlateau


def make_trainer(opt, metric_value, callbacks=()):
    return SimpleNamespace(
        optimizers=[opt],
        callback_metrics={"sub_loss/val_prior_loss": torch.tensor(metric_value)},
        sanity_checking=False,
        is_global_zero=True,
        world_size=1,
        global_step=0,
        logger=None,
        callbacks=list(callbacks),
    )


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Linear(2, 2)
        self.dec = torch.nn.Linear(2, 2)


def make_model_and_optimizer():
    model = TinyModel()
    opt = torch.optim.Adam(
        [
            {"params": model.enc.parameters(), "lr": 1e-4, "name": "prior_encoder"},
            {"params": model.dec.parameters(), "lr": 1e-4, "name": "decoder"},
        ]
    )
    return model, opt


def lr_of(opt, name):
    return next(g["lr"] for g in opt.param_groups if g["name"] == name)


def run_epochs(cb, opt, values, pl_module=None, callbacks=()):
    for v in values:
        cb.on_validation_epoch_end(make_trainer(opt, v, callbacks), pl_module)


def test_downshift_after_patience_epochs_above_best():
    _, opt = make_model_and_optimizer()
    cb = LRDownshiftOnPlateau(patience=5, factor=0.1, groups=["prior_encoder"])

    # Improving phase: no downshift.
    run_epochs(cb, opt, [1.10, 1.05, 1.00, 0.996])
    assert lr_of(opt, "prior_encoder") == 1e-4

    # 4 stagnant epochs: still armed but not fired.
    run_epochs(cb, opt, [1.00, 1.001, 1.002, 1.003])
    assert lr_of(opt, "prior_encoder") == 1e-4

    # 5th stagnant epoch fires the downshift; decoder untouched.
    run_epochs(cb, opt, [1.004])
    assert lr_of(opt, "prior_encoder") == 1e-5
    assert lr_of(opt, "decoder") == 1e-4
    assert cb.num_downshifts == 1
    # Without rewind, baseline resets to current value; counter cleared.
    assert abs(cb.best_metric - 1.004) < 1e-5
    assert cb.epochs_no_improve == 0


def test_second_downshift_and_cap():
    _, opt = make_model_and_optimizer()
    cb = LRDownshiftOnPlateau(patience=2, factor=0.1, groups=["prior_encoder"], max_downshifts=2)

    run_epochs(cb, opt, [1.0])  # sets baseline
    run_epochs(cb, opt, [1.01, 1.02])  # fires #1
    assert lr_of(opt, "prior_encoder") == 1e-5

    run_epochs(cb, opt, [1.03, 1.04])  # fires #2 (vs reset baseline 1.02)
    assert abs(lr_of(opt, "prior_encoder") - 1e-6) < 1e-12

    run_epochs(cb, opt, [1.05, 1.06, 1.07, 1.08])  # capped: no third downshift
    assert abs(lr_of(opt, "prior_encoder") - 1e-6) < 1e-12
    assert cb.num_downshifts == 2


def test_improvement_resets_counter():
    _, opt = make_model_and_optimizer()
    cb = LRDownshiftOnPlateau(patience=3, groups=["prior_encoder"])

    run_epochs(cb, opt, [1.0, 1.01, 1.02])  # 2 stagnant epochs
    run_epochs(cb, opt, [0.99])  # new best resets the counter
    run_epochs(cb, opt, [1.0, 1.0])  # only 2 stagnant epochs again
    assert lr_of(opt, "prior_encoder") == 1e-4
    assert cb.num_downshifts == 0


def test_rewind_restores_best_weights_and_keeps_current_lrs(tmp_path):
    model, opt = make_model_and_optimizer()

    # Snapshot the "best" state, then degrade the weights and change a lr
    # (as lr_override might have done after the ckpt was written).
    ckpt_path = tmp_path / "best_val_prior_000.ckpt"
    torch.save(
        {"state_dict": model.state_dict(), "optimizer_states": [opt.state_dict()]},
        ckpt_path,
    )
    best_weights = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    for g in opt.param_groups:
        if g["name"] == "decoder":
            g["lr"] = 5e-5

    ckpt_cb = ModelCheckpoint(monitor="sub_loss/val_prior_loss", mode="min")
    ckpt_cb.best_model_path = str(ckpt_path)

    cb = LRDownshiftOnPlateau(
        patience=2, factor=0.1, groups=["prior_encoder"], rewind_to_best=True
    )
    run_epochs(cb, opt, [1.0, 1.01, 1.02], pl_module=model, callbacks=[ckpt_cb])

    # Weights rolled back to the ckpt snapshot.
    for k, v in model.state_dict().items():
        assert torch.equal(v, best_weights[k]), k
    # Encoder lr downshifted from its CURRENT value; decoder keeps its
    # current (post-ckpt) lr instead of the one stored in the ckpt.
    assert lr_of(opt, "prior_encoder") == 1e-5
    assert lr_of(opt, "decoder") == 5e-5
    # With rewind the historical best stays the baseline.
    assert abs(cb.best_metric - 1.0) < 1e-5
    assert cb.num_downshifts == 1
    assert cb.epochs_no_improve == 0


def test_rewind_missing_ckpt_still_downshifts():
    model, opt = make_model_and_optimizer()
    ckpt_cb = ModelCheckpoint(monitor="sub_loss/val_prior_loss", mode="min")
    # best_model_path left empty: no ckpt saved yet.

    cb = LRDownshiftOnPlateau(
        patience=1, factor=0.1, groups=["prior_encoder"], rewind_to_best=True
    )
    run_epochs(cb, opt, [1.0, 1.01], pl_module=model, callbacks=[ckpt_cb])
    assert lr_of(opt, "prior_encoder") == 1e-5
    # Falls back to the no-rewind baseline reset.
    assert abs(cb.best_metric - 1.01) < 1e-5


def test_unique_state_keys_for_multiple_instances():
    a = LRDownshiftOnPlateau(monitor="sub_loss/val_prior_loss", groups=["prior_encoder"])
    b = LRDownshiftOnPlateau(monitor="sub_loss/val_dur_loss", groups=["duration_predictor"])
    assert a.state_key != b.state_key


def test_state_dict_roundtrip():
    cb = LRDownshiftOnPlateau()
    cb.best_metric = 0.996
    cb.epochs_no_improve = 3
    cb.num_downshifts = 1

    cb2 = LRDownshiftOnPlateau()
    cb2.load_state_dict(cb.state_dict())
    assert cb2.best_metric == 0.996
    assert cb2.epochs_no_improve == 3
    assert cb2.num_downshifts == 1


def test_disabled_is_noop():
    _, opt = make_model_and_optimizer()
    cb = LRDownshiftOnPlateau(enabled=False, patience=1, groups=["prior_encoder"])
    run_epochs(cb, opt, [1.0, 1.1, 1.2, 1.3])
    assert lr_of(opt, "prior_encoder") == 1e-4
