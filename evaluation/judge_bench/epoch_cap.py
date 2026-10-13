"""Publish a user-requested epoch-boundary checkpoint from an active job.

The running trainer and its original scheduler/config are not patched. After the
complete boundary checkpoint is saved, only its verified training child is
stopped. Its existing supervisor restores the base server. The checkpoint then
becomes the final adapter with an explicit amendment and preserved exit evidence.
The waiting post-training supervisor is paused only during this transition.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

from .training_job import _write_json, gpu_processes


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def process(identity, *, allow_dead=False):
    path = Path("/proc") / str(identity["pid"])
    if not path.exists():
        if allow_dead:
            return None
        raise RuntimeError("Declared process is absent")
    stat = (path / "stat").read_text().split(") ", 1)[1].split()
    command = (path / "cmdline").read_bytes().replace(b"\x00", b" ").decode().strip()
    if (identity["uid"] != os.getuid() or path.stat().st_uid != identity["uid"] or os.getpgid(identity["pid"]) != identity["pgrp"]
            or stat[19] != identity["start_ticks"]):
        raise RuntimeError("PID/UID/PGRP/start identity changed; no signal sent")
    if stat[0] == "Z" and allow_dead:
        return None
    if command != identity["command"]:
        raise RuntimeError("Process command changed; no signal sent")
    return stat[0]


def send(identity, sig):
    process(identity)
    os.kill(identity["pid"], sig)


def wait_stopped(identity):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process(identity) in {"T", "t"}:
            return
        time.sleep(0.1)
    raise RuntimeError("Owned process did not stop")


def boundary_checkpoint(directory, request):
    state = read(directory / "trainer_state.json")
    if (state["global_step"] != request["target_step"]
            or not math.isclose(state["epoch"], request["requested_epochs"], abs_tol=1e-9)
            or state["max_steps"] != request["original_steps"]):
        raise ValueError("Checkpoint does not match the requested full epoch boundary")
    required = ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
                "scheduler.pt", "rng_state.pth", "training_args.bin", "tokenizer.json",
                "tokenizer_config.json", "chat_template.jinja", "trainer_state.json"]
    if any(not (directory / name).is_file() or not (directory / name).stat().st_size for name in required):
        raise ValueError("Boundary checkpoint is incomplete")
    return state


def validate_adapter(directory, metadata):
    # The merger will additionally check every tensor against the original base.
    import torch
    from safetensors import safe_open
    config = read(directory / "adapter_config.json")
    if config["r"] != metadata["config"]["lora_r"] or config["lora_alpha"] != metadata["config"]["lora_alpha"]:
        raise ValueError("Boundary adapter rank/scaling differs")
    expected = {"base_model.model." + target + suffix
                for target in metadata["target_modules"]
                for suffix in [".lora_A.weight", ".lora_B.weight"]}
    with safe_open(directory / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        if set(handle.keys()) != expected:
            raise ValueError("Boundary adapter target tensors differ")
        for name in handle.keys():
            value = handle.get_tensor(name)
            if value.dtype != torch.float32 or value.ndim != 2 or not torch.isfinite(value).all().item():
                raise ValueError("Boundary adapter contains invalid tensors")
    if torch.cuda.is_initialized():
        raise RuntimeError("Epoch-cap validation must remain CPU-only")


def run(request_path):
    request = read(request_path)
    output = Path(request["output_dir"])
    training = Path(request["training_dir"])
    post = Path(request["post_dir"])
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "job.json"
    with (output / "job.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(state_path) if state_path.exists() else {
            "status": "waiting_for_epoch_checkpoint", "started_utc": utc(),
            "request_sha256": sha(request_path), "source_sha256": sha(__file__),
        }
        if state["request_sha256"] != sha(request_path) or state["source_sha256"] != sha(__file__):
            raise ValueError("Epoch-cap request/source changed; preserve existing state")
        if state["status"] == "completed":
            return
        if state["status"] == "failed_before_training_stop":
            raise RuntimeError("Previous amendment failed before stopping training; inspect evidence before a new request")
        state.update(supervisor_pid=os.getpid(), resumed_utc=utc())
        _write_json(state_path, state)
        def advance(status, **extra):
            state.update(status=status, checked_utc=utc(), **extra)
            _write_json(state_path, state)
            print(json.dumps({"event": "epoch_cap", **state}), flush=True)

        def verify_sources():
            for name, evidence in request["frozen_files"].items():
                if sha(name) != evidence:
                    raise ValueError("A frozen training input/source/config changed")
            original = read(training / "training_metadata.json")
            if original["config"] != request["original_training_config"]:
                raise ValueError("Original training config changed")
            if (original["data"]["fit"]["items"] != 7200 or original["data"]["validation"]["items"] != 0
                    or request["target_step"] != 450 or request["requested_epochs"] != 1):
                raise ValueError("This amendment requires exactly one full 7200-item epoch")
            return original

        checkpoint = training / ("checkpoint-" + str(request["target_step"]))
        try:
            metadata = verify_sources()
            if state["status"] == "waiting_for_epoch_checkpoint":
                while True:
                    process(request["trainer"])
                    process(request["training_supervisor"])
                    process(request["post_supervisor"])
                    job = read(training / "job.json")
                    if job["status"] != "training":
                        raise RuntimeError("Training exited before the requested checkpoint")
                    # Trainer writes trainer_state.json after model, optimizer,
                    # scheduler and RNG; parsing it gates the boundary stop.
                    if (checkpoint / "trainer_state.json").exists():
                        try:
                            boundary_checkpoint(checkpoint, request)
                        except (json.JSONDecodeError, FileNotFoundError):
                            time.sleep(0.2)
                            continue
                        break
                    time.sleep(0.25)
                _, gpu = gpu_processes(0, request["gpu_uuid"])
                if {x["pid"] for x in gpu} != {request["trainer"]["pid"]}:
                    raise RuntimeError("GPU0 process ownership changed; no process stopped")
                post_job = read(post / "job.json")
                if (post_job["status"] != "waiting_for_training" or post_job.get("child_pid")
                        or post_job.get("completed_stages")):
                    raise RuntimeError("Post-training supervisor is no longer waiting")
                advance("pausing_post_supervisor")

            if state["status"] == "pausing_post_supervisor":
                send(request["post_supervisor"], signal.SIGSTOP)
                wait_stopped(request["post_supervisor"])
                advance("pausing_trainer")
            if state["status"] == "pausing_trainer":
                send(request["trainer"], signal.SIGSTOP)
                wait_stopped(request["trainer"])
                advance("publishing_boundary_adapter")

            if state["status"] == "publishing_boundary_adapter":
                boundary = boundary_checkpoint(checkpoint, request)
                progress = []
                for line in Path(request["training_log"]).read_text().splitlines():
                    if line.startswith('{"event": "training_progress"'):
                        progress.append(json.loads(line))
                state["observed_step_at_pause"] = progress[-1]["step"] if progress else None
                validate_adapter(checkpoint, metadata)
                artifacts = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                             for p in checkpoint.iterdir() if p.is_file()}
                _write_json(output / "checkpoint_manifest.json", artifacts)
                for name in ["job.json", "training_metadata.json"]:
                    evidence = output / ("before_stop_" + name)
                    if not evidence.exists():
                        shutil.copyfile(training / name, evidence)
                adapter = training / "adapter"
                names = [name for name in artifacts if name not in {
                    "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin", "trainer_state.json"}]
                if not adapter.exists():
                    stage = output / "adapter.staging"
                    stage.mkdir(exist_ok=True)
                    for name in names:
                        shutil.copyfile(checkpoint / name, stage / name)
                    stage.replace(adapter)
                if ({p.name for p in adapter.iterdir()} != set(names)
                        or any(sha(adapter / name) != artifacts[name]["sha256"] for name in names)):
                    raise ValueError("Final adapter differs from the boundary checkpoint")
                _write_json(training / "trainer_state.json", boundary)
                advance("terminating_training_child", adapter_sha256=artifacts["adapter_model.safetensors"]["sha256"])

            if state["status"] == "terminating_training_child":
                if process(request["trainer"], allow_dead=True):
                    send(request["trainer"], signal.SIGTERM)
                    send(request["trainer"], signal.SIGCONT)
                advance("waiting_for_original_supervisor_restoration")

            if state["status"] == "waiting_for_original_supervisor_restoration":
                deadline = time.monotonic() + 420
                while process(request["training_supervisor"], allow_dead=True):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Original training supervisor did not finish restoration")
                    time.sleep(1)
                job = read(training / "job.json")
                if job["status"] != "failed" or job.get("restore", {}).get("status") != "ready":
                    raise RuntimeError("Original supervisor did not preserve a failed exit and restore the base")
                if job.get("exit_code") != -signal.SIGTERM:
                    raise RuntimeError("Unexpected training exit; do not promote checkpoint")
                evidence = output / "original_supervisor_exit.json"
                if not evidence.exists():
                    _write_json(evidence, job)
                advance("finalizing_requested_one_epoch")

            if state["status"] == "finalizing_requested_one_epoch":
                boundary = boundary_checkpoint(checkpoint, request)
                original_exit = read(output / "original_supervisor_exit.json")
                amendment = {
                    "authorization": request["authorization"], "requested_utc": request["requested_utc"],
                    "requested_epochs": 1, "requested_optimizer_steps": 450,
                    "original_epochs": 2, "original_optimizer_steps": 900,
                    "original_config_preserved": True, "original_scheduler_preserved": True,
                    "method": "Publish complete checkpoint-450; stop owned training child; original supervisor restores base",
                    "checkpoint": str(checkpoint), "checkpoint_manifest_sha256": sha(output / "checkpoint_manifest.json"),
                    "adapter_sha256": state["adapter_sha256"], "controller_source_sha256": sha(__file__),
                    "request_sha256": sha(request_path), "original_exit_code": original_exit["exit_code"],
                    "original_exit_evidence": str(output / "original_supervisor_exit.json"),
                    "original_exit_evidence_sha256": sha(output / "original_supervisor_exit.json"),
                    "observed_step_at_pause": state.get("observed_step_at_pause"),
                    "final_adapter_optimizer_step": 450,
                    "completed_utc": utc(), "test_set_used": False,
                }
                _write_json(output / "completion.json", amendment)
                metadata.update(status="complete", global_step=450, epochs_completed=1.0,
                                full_epoch_training_complete=True, ready_for_judgebench_evaluation=True,
                                finished_utc=amendment["completed_utc"], epoch_cap_amendment=amendment,
                                effective_requested_epochs=1, effective_requested_optimizer_steps=450,
                                train_metrics={"availability": "Boundary checkpoint log_history; trainer was stopped after saving"},
                                gpu_monitor_summary_available=False)
                _write_json(training / "training_metadata.json", metadata)
                job = dict(original_exit, status="completed", completion_method="user_requested_full_epoch_checkpoint",
                           epoch_cap_amendment=amendment, requested_training_artifact_complete=True)
                _write_json(training / "job.json", job)
                advance("resuming_post_training_supervisor")

            if state["status"] == "resuming_post_training_supervisor":
                send(request["post_supervisor"], signal.SIGCONT)
                advance("completed", finished_utc=utc())
        except BaseException as error:
            # Before stopping the child, a failed amendment must not strand
            # healthy training. After termination, retain the blocked transition
            # and paused post supervisor for audited recovery.
            recovery = {}
            if state["status"] in {"pausing_post_supervisor", "pausing_trainer", "publishing_boundary_adapter"}:
                failed_stage = state["status"]
                for name in ["trainer", "post_supervisor"]:
                    try:
                        if process(request[name]) in {"T", "t"}:
                            send(request[name], signal.SIGCONT)
                            recovery[name] = "resumed"
                    except Exception as resume_error:
                        recovery[name] = type(resume_error).__name__
                state.update(status="failed_before_training_stop", failed_stage=failed_stage)
            state.update(error_type=type(error).__name__, error=str(error), recovery=recovery, failed_utc=utc())
            _write_json(state_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    run(args.request.resolve())


if __name__ == "__main__":
    main()
