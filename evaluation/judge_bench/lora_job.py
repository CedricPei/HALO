"""Persistently run one frozen LoRA job under its declared GPU admission policy.

This CPU supervisor never stops other workloads. It checks the GPU UUID and
three consecutive resource observations before loading the model, and locks the
job directory to prevent duplicate launches. SIGTERM cancels only its child.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_state(index: int, expected_uuid: str) -> dict:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = list(csv.reader(io.StringIO(output), skipinitialspace=True))
    row = next(row for row in rows if int(row[0]) == index)
    if row[1].strip() != expected_uuid:
        raise RuntimeError("Declared physical GPU UUID changed")
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True)
    processes = [{"pid": int(app[1]), "used_memory_mib": app[2].strip()}
                 for app in csv.reader(io.StringIO(apps), skipinitialspace=True)
                 if app and app[0].strip() == expected_uuid]
    state = {"index": index, "uuid": expected_uuid, "memory_used_mib": int(row[2]),
             "memory_total_mib": int(row[3]), "utilization_percent": int(row[4]),
             "memory_free_mib": int(row[5]),
             "compute_processes": processes}
    # Utilization can lag behind context teardown; use process and memory ownership.
    state["idle"] = not processes and state["memory_used_mib"] < 1024
    return state


def gpu_admitted(current: dict, policy: dict) -> bool:
    mode = policy.get("mode", "exclusive")
    if mode == "exclusive":
        return current["idle"]
    if mode != "shared" or policy.get("minimum_free_mib", 0) <= 0:
        raise ValueError("Invalid GPU admission policy")
    return current["memory_free_mib"] >= policy["minimum_free_mib"]


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root, cwd = Path(config["output_dir"]), Path(config["cwd"])
    if config["environment"].get("CUDA_VISIBLE_DEVICES") != str(config["gpu_index"]):
        raise ValueError("Training environment must expose exactly the declared GPU")
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "job.json"
    with (root / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            raise ValueError("Existing job state cannot be overwritten; inspect it before resuming")
        stat = Path(f"/proc/{os.getpid()}/stat").read_text()
        state = {"status": "preflight", "created_utc": utc(), "supervisor_pid": os.getpid(),
                 "supervisor_uid": os.getuid(), "supervisor_pgrp": os.getpgrp(),
                 "supervisor_start_ticks": int(stat[stat.rfind(")") + 2:].split()[19]),
                 "config_path": str(config_path.resolve()), "config_sha256": sha(config_path),
                 "supervisor_source_sha256": sha(Path(__file__)),
                 "gpu_index": config["gpu_index"], "gpu_uuid": config["gpu_uuid"],
                 "child_pid": None, "child_return_code": None}
        child = None

        def cancel(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")

        signal.signal(signal.SIGTERM, cancel)
        signal.signal(signal.SIGINT, cancel)

        def persist():
            state["updated_utc"] = utc()
            write_json(state_path, state)

        def verify_inputs():
            if sha(config_path) != state["config_sha256"]:
                raise ValueError("Launch config changed while queued")
            for name, expected in config["frozen_sha256"].items():
                if sha(Path(name)) != expected:
                    raise ValueError(f"A frozen training dependency changed: {name}")

        try:
            verify_inputs()
            admission = config.get("gpu_admission", {"mode": "exclusive"})
            if admission.get("mode") == "shared":
                limit = admission.get("allocator_limit_gib")
                reserve = admission.get("reserve_mib", 0)
                command = config["command"]
                if (not limit or reserve < 0 or "--cuda-memory-limit-gib" not in command
                        or float(command[command.index("--cuda-memory-limit-gib") + 1]) != limit
                        or admission["minimum_free_mib"] < limit * 1024 + reserve):
                    raise ValueError("Shared GPU policy requires a matching allocator cap and reserved headroom")
            idle_observations = 0
            while True:
                current = gpu_state(config["gpu_index"], config["gpu_uuid"])
                admitted = gpu_admitted(current, admission)
                idle_observations = idle_observations + 1 if admitted else 0
                state.update(status="waiting_for_gpu", gpu=current, gpu_admission=admission, idle_observations=idle_observations)
                persist()
                if idle_observations >= 3:
                    verify_inputs()
                    if gpu_admitted(gpu_state(config["gpu_index"], config["gpu_uuid"]), admission):
                        break
                    idle_observations = 0
                time.sleep(30)
            environment = os.environ.copy()
            environment.update(config["environment"])
            with (root / "training.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(config["command"], cwd=cwd, env=environment,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                state.update(status="training", child_pid=child.pid, training_started_utc=utc())
                persist()
                code = child.wait()
            state["child_return_code"] = code
            metadata_path = root / "training_metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                state["training_status"] = metadata.get("status")
                state["global_step"] = metadata.get("global_step")
                state["best_validation_checkpoint"] = metadata.get("best_internal_validation_checkpoint")
            complete = code == 0 and state.get("training_status") == "complete"
            state["status"] = "completed" if complete else "failed"
            if not complete:
                state["error"] = f"Trainer exit {code}, status {state.get('training_status')}"
        except KeyboardInterrupt as error:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            state.update(status="cancelled", error=str(error))
        except Exception as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if state["status"] in ("completed", "failed", "cancelled"):
                state["finished_utc"] = utc()
            persist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
