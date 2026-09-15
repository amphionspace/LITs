"""Checkpoint loading and startup checks for Stage 2 trajectory distillation."""
import hashlib
import json

import torch

from meanflow_distill.interval_estimator import IntervalConditionedEstimator


def tensor_hash(items):
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_hash(model):
    return tensor_hash((name, value) for name, value in model.named_parameters()
                       if not name.startswith("decoder.estimator."))


def estimator(model):
    value = model.decoder.estimator
    return getattr(value, "module", value)


class StartupAudit:
    def __init__(self, student, teacher, output, rank):
        self.output, self.rank = output, rank
        self.frozen = frozen_hash(student)
        assert self.frozen == frozen_hash(teacher)
        assert tensor_hash(estimator(student).base.state_dict().items()) == tensor_hash(
            estimator(teacher).state_dict().items())
        self.teacher_hash = tensor_hash(teacher.state_dict().items())
        self.initial = {name: value.detach().cpu().clone()
                        for name, value in estimator(student).named_parameters()}

    def check(self, student, teacher, step):
        assert frozen_hash(student) == self.frozen, "Frozen conditioning weights changed"
        assert tensor_hash(teacher.state_dict().items()) == self.teacher_hash, "Teacher changed"
        groups = {}
        for name, value in estimator(student).named_parameters():
            assert value.grad is not None and torch.isfinite(value.grad).all(), name
            key = name.split(".")[0] if not name.startswith("base.") else name.split(".")[1]
            groups[key] = groups.get(key, False) or not torch.equal(value.detach().cpu(), self.initial[name])
        assert all(groups.values()), groups
        report = dict(status="passed", global_step=step, frozen_parameters_unchanged=True,
                      teacher_unchanged=True, updated_estimator_groups=groups,
                      frozen_sha256=self.frozen, teacher_sha256=self.teacher_hash)
        if self.rank == 0:
            (self.output / f"startup_step_{step}.json").write_text(json.dumps(report, indent=2))


def load_student(path, device="cpu"):
    from lits.models.lits import LITS
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = LITS(**payload["hyper_parameters"])
    model.decoder.estimator = IntervalConditionedEstimator(model.decoder.estimator)
    model.load_state_dict(payload["state_dict"], strict=True)
    meta = payload["metadata"]
    steps = meta["student_steps"]
    grid = meta["student_t_grid"]
    if any(abs(t - i / steps) > 1e-7 for i, t in enumerate(grid)):
        raise ValueError("Stage 2 evaluator requires the trained uniform time grid")
    model.decoder.default_n_timesteps = steps
    model.decoder.objective = "meanflow_distillation"
    return model.to(device).eval()
