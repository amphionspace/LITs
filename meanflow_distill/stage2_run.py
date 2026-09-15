"""Supervise a planned Stage 2 distillation run and isolated checkpoint evaluations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def successful_summary(path):
    summary = json.loads(path.read_text())
    return summary["samples"] > 0 and all(
        group["evaluation_failures"] == 0 for group in summary["groups"].values())


def training_command(run, plan, preflight=False):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node", str(plan["world_size"]),
            str(REPO / "meanflow_distill/train_intmeanflow_distill.py"),
            "--lits-root", str(REPO), "--teacher-ckpt", plan["teacher"],
            "--manifest", str(run / "data/train.jsonl"),
            "--val-manifest", str(run / "data/val.jsonl"),
            "--output-dir", str(run / "preflight_ddp" if preflight else run),
            "--batch-size", str(plan["batch_per_gpu"]), "--max-text-len", "0",
            "--teacher-steps", "16", "--student-steps", "2", "--student-t-grid", "0,0.5,1",
            "--max-steps", "2" if preflight else str(plan["max_steps"]),
            "--lr", str(plan["lr"]), "--precision", plan["precision"],
            "--temperature", str(plan["temperature"]), "--seed", str(plan["seed"]),
            "--num-workers", "2", "--dist-backend", "nccl", "--audit-startup",
            "--save-every", "2" if preflight else "1000",
            "--val-every", "2" if preflight else "1000",
            "--val-batches", "2" if preflight else "10000", "--log-every", "10",
            "--no-kv-cache-distill", "--no-mu-streaming",
            "--no-teacher-decoder-streaming", "--no-decoder-streaming"]


def evaluate(run, step):
    plan = json.loads((run / "plan.json").read_text())
    out = run / "eval" / f"step_{step:07d}"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "checkpoints" / f"student_step_{step:07d}.pt"
    status = dict(status="running", global_step=step, checkpoint=str(checkpoint), started_at=time.time())
    write(out / "status.json", status)
    base = [sys.executable, str(REPO / "training/common/evaluate_checkpoint.py"),
            "--checkpoint", str(checkpoint), "--distilled", "--output", str(out),
            "--data-dir", str(Path(plan["eval_manifest"]).parent),
            "--vocoder-checkpoint", plan["vocoder"], "--temperature", str(plan["temperature"])]
    if step == 1000:
        base += ["--per-group-limit", "8"]
    for stage in ["synthesize", "asr", "metrics", "summarize"]:
        with (out / f"{stage}.log").open("ab") as log:
            result = subprocess.run(base + ["--stage", stage], cwd=REPO,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(status="failed", failed_stage=stage, returncode=result.returncode)
            write(out / "status.json", status)
            return result.returncode
    if not successful_summary(out / "summary.json"):
        status.update(status="failed", failed_stage="sample_validation")
        write(out / "status.json", status)
        return 1
    status.update(status="complete", completed_at=time.time(), summary=str(out / "summary.json"))
    write(out / "status.json", status)
    if step == 1000:
        # Establish matched full-set teacher baselines in the same evaluator.
        for nfe in (2, 16):
            baseline = run / "eval" / f"teacher_{nfe}step"
            baseline.mkdir(parents=True, exist_ok=True)
            baseline_args = [sys.executable, str(REPO / "training/common/evaluate_checkpoint.py"),
                "--checkpoint", plan["teacher"], "--output", str(baseline),
                "--data-dir", str(Path(plan["eval_manifest"]).parent),
                "--vocoder-checkpoint", plan["vocoder"], "--temperature", str(plan["temperature"]),
                "--n-timesteps", str(nfe)]
            write(baseline / "status.json", dict(status="running", n_timesteps=nfe))
            for stage in ["synthesize", "asr", "metrics", "summarize"]:
                with (baseline / f"{stage}.log").open("ab") as log:
                    result = subprocess.run(baseline_args + ["--stage", stage], cwd=REPO,
                                            stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    write(baseline / "status.json", dict(status="failed", stage=stage, code=result.returncode))
                    return result.returncode
            ok = successful_summary(baseline / "summary.json")
            write(baseline / "status.json", dict(status="complete" if ok else "failed", n_timesteps=nfe))
            if not ok:
                return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight-ddp", action="store_true")
    parser.add_argument("--evaluate", type=int)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    plan = json.loads((run / "plan.json").read_text())
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("MPLCONFIGDIR", str(run / ".matplotlib"))
    if args.evaluate is not None:
        return evaluate(run, args.evaluate)
    command = training_command(run, plan, args.preflight_ddp)
    if args.preflight_ddp:
        result = subprocess.run(command, cwd=REPO)
        if result.returncode == 0:
            write(run / "preflight_ddp_passed.json", dict(status="passed", command=command))
        return result.returncode
    assert json.loads((run / "preflight/preflight.json").read_text())["status"] == "passed"
    assert json.loads((run / "preflight_ddp_passed.json").read_text())["status"] == "passed"
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    assert current_commit == plan["source_commit"], "Code revision changed after preparation"
    for name, expected in plan["source_file_sha256"].items():
        assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256(Path(plan["teacher"]).read_bytes()).hexdigest() == plan["teacher_sha256"]
    for split, expected in plan["manifest_sha256"].items():
        assert hashlib.sha256((run / "data" / f"{split}.jsonl").read_bytes()).hexdigest() == expected
    # Lock the shared run directory; PIDs alone are ambiguous across hardware.
    import fcntl
    lock = (run / "supervisor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / "training_state.json").exists():
        raise RuntimeError("Run already has training state; use an explicit resume procedure")
    with (run / "console.log").open("ab") as log:
        training = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    write(run / "launch.json", dict(status="running", training_pid=training.pid,
          supervisor_pid=os.getpid(), hostname=socket.gethostname(), command=command, started_at=time.time()))
    evaluator = None
    attempted = set()
    while True:
        if evaluator is not None and evaluator.poll() is not None:
            evaluator = None
        if evaluator is None:
            for step in plan["eval_steps"]:
                ckpt = run / "checkpoints" / f"student_step_{step:07d}.pt"
                if step not in attempted and ckpt.exists():
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1")
                    with (run / "eval_watcher.log").open("ab") as log:
                        evaluator = subprocess.Popen([sys.executable, "-m", "meanflow_distill.stage2_run",
                            "--run-dir", str(run), "--evaluate", str(step)], cwd=REPO, env=env,
                            stdout=log, stderr=subprocess.STDOUT)
                    attempted.add(step)
                    break
        rc = training.poll()
        if rc is not None and evaluator is None:
            pending = [s for s in plan["eval_steps"] if s not in attempted and
                       (run / "checkpoints" / f"student_step_{s:07d}.pt").exists()]
            if not pending:
                break
        time.sleep(5)
    write(run / "supervisor_status.json", dict(status="complete" if rc == 0 else "failed",
          training_exit_code=rc, hostname=socket.gethostname(), completed_at=time.time(),
          attempted_evaluations=sorted(attempted)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
