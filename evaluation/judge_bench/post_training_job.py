"""Durably wait for training, serve merged weights, and run the frozen evaluation.

Run on the declared Linux GPU host in its own session. This supervisor never
modifies training, the original checkpoint, or preserved baseline evidence.
"""

from __future__ import annotations

import argparse
import csv
import copy
import fcntl
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

from .training_job import _alive, _api_key, _utc, _write_json, gpu_processes, wait_serving


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _proc(pid: int) -> dict[str, Any] | None:
    root = Path(f"/proc/{pid}")
    try:
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "uid": root.stat().st_uid, "pgrp": int(fields[2]),
                "argv": (root / "cmdline").read_bytes().decode().rstrip("\0").split("\0")}
    except FileNotFoundError:
        return None


def _owned_server(pid: int, model_path: str, model_id: str) -> dict[str, Any]:
    process = _proc(pid)
    if not process or process["uid"] != os.getuid() or process["pgrp"] != pid:
        raise RuntimeError("Serving PID is not a live, independently grouped process owned by this user")
    argv = process["argv"]
    if ("serve" not in argv or model_path not in argv or "--served-model-name" not in argv
            or argv[argv.index("--served-model-name") + 1] != model_id):
        raise RuntimeError("Serving command does not match the declared checkpoint and model ID")
    return process


def stop_owned_server(pid: int, model_path: str, manifest: dict[str, Any], *, allow_foreign_gpu_processes: bool = False) -> None:
    _owned_server(pid, model_path, manifest["served_model_name"])
    _, processes = gpu_processes(manifest["gpu_index"], manifest["gpu_uuid"])
    for process in processes:
        details = _proc(process["pid"])
        if not details or details["pgrp"] != pid:
            if allow_foreign_gpu_processes:
                continue
            raise RuntimeError("Declared GPU contains a foreign process; no process was stopped")
        if details["uid"] != os.getuid():
            raise RuntimeError("Owned serving GPU process has a different user; no process was stopped")
    all_apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                               "--format=csv,noheader,nounits"],
                              check=True, capture_output=True, text=True, timeout=10)
    for row in csv.reader(io.StringIO(all_apps.stdout)):
        if not row:
            continue
        details = _proc(int(row[1].strip()))
        if details and details["pgrp"] == pid and row[0].strip() != manifest["gpu_uuid"]:
            raise RuntimeError("Serving process group is using another GPU; no process was stopped")
    for root in Path("/proc").iterdir():
        if not root.name.isdigit():
            continue
        details = _proc(int(root.name))
        if details and details["pgrp"] == pid and details["uid"] != os.getuid():
            raise RuntimeError("Serving process group contains a different user")
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        _, processes = gpu_processes(manifest["gpu_index"], manifest["gpu_uuid"])
        owned_gpu_processes = [process for process in processes
                               if (details := _proc(process["pid"])) and details["pgrp"] == pid]
        remaining = owned_gpu_processes if allow_foreign_gpu_processes else processes
        if _proc(pid) is None and not remaining:
            return
        time.sleep(2)
    raise RuntimeError("Owned serving did not release the GPU within 120 seconds")


def _run(command: list[str], cwd: Path, state: dict[str, Any], state_path: Path,
         *, environment: dict[str, str] | None = None) -> None:
    child = subprocess.Popen(command, cwd=cwd, env=environment)
    state.update(child_pid=child.pid, child_command=command)
    _write_json(state_path, state)
    code = child.wait()
    state.update(child_pid=None, child_exit_code=code)
    _write_json(state_path, state)
    if code:
        raise RuntimeError(f"Stage {state['status']} exited with code {code}")


def run(config_path: Path) -> int:
    config = _read(config_path)
    cwd = Path(config["cwd"])
    output = Path(config["job_output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "job.json"
    with (output / "job.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another post-training supervisor holds the job lock") from None
        state = _read(state_path) if state_path.exists() else {
            "schema_version": 1, "started_utc": _utc(), "status": "waiting_for_training",
            "config_sha256": _sha(config_path), "completed_stages": [],
        }
        if state["config_sha256"] != _sha(config_path):
            raise ValueError("Post-training configuration changed; preserve the existing job")
        if state.get("status") == "completed":
            return 0
        old_child = state.get("child_pid")
        if old_child and _alive(old_child):
            raise RuntimeError("A previous stage process is still alive; do not duplicate its work")
        state.update(supervisor_pid=os.getpid(), resumed_utc=_utc())
        _write_json(state_path, state)
        try:
            training_dir = Path(config["training_output_dir"])
            while True:
                if not (training_dir / "job.json").exists():
                    state.update(status="waiting_for_training", checked_utc=_utc(),
                                 training_status="supervisor_not_started_yet")
                    _write_json(state_path, state)
                    time.sleep(30)
                    continue
                training_job = _read(training_dir / "job.json")
                training = (_read(training_dir / "training_metadata.json")
                            if (training_dir / "training_metadata.json").exists() else {})
                if training_job["status"] == "completed":
                    if not (training_job.get("restore", {}).get("status") == "ready"
                            and training.get("status") == "complete"
                            and training.get("full_epoch_training_complete") is True
                            and training.get("ready_for_judgebench_evaluation") is True):
                        raise RuntimeError("Training completed without a verified final adapter/restored server")
                    break
                if training_job["status"] in {"failed", "blocked", "training_failed"}:
                    raise RuntimeError("Training or its serving restoration failed; evaluation was not started")
                state.update(status="waiting_for_training", checked_utc=_utc(),
                             training_status=training_job["status"])
                _write_json(state_path, state)
                time.sleep(30)

            baseline = _read(Path(config["baseline_run_dir"]) / "run_metadata.json")
            original = baseline["served_configuration"]
            launch = _read(Path(config["training_launch_config"]))
            if _sha(Path(config["training_launch_config"])) != training_job["launch_config_sha256"]:
                raise ValueError("Training launch configuration changed after training started")
            if (launch["gpu_index"] != original["gpu_index"] or launch["gpu_uuid"] != original["gpu_uuid"]
                    or launch["environment"].get("CUDA_VISIBLE_DEVICES") != str(original["gpu_index"])
                    or config["serving_python"] != launch["restore_serving_launcher"][0]
                    or launch["restore_model_id"] != original["served_model_name"]
                    or launch["restore_base_url"] != baseline["config"]["base_url"]):
                raise ValueError("Training/serving launch GPU, runtime or model identity differs from baseline")
            original_launcher = Path(original["launcher_path"])
            if _sha(original_launcher) != original["launcher_sha256"]:
                raise ValueError("The frozen base serving launcher has changed")
            adapter = training_dir / "adapter"
            merged = Path(config["merged_model_dir"])
            merge_manifest = merged / "merge_manifest.json"
            if "merged" not in state["completed_stages"]:
                state.update(status="merging_adapter", child_pid=None)
                _write_json(state_path, state)
                environment = os.environ.copy()
                environment.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                                   OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
                _run([config["training_python"], "-B", "-m", "evaluation.judge_bench.merge_adapter",
                      "--base-model", original["model_path"], "--adapter", str(adapter),
                      "--training-metadata", str(training_dir / "training_metadata.json"),
                      "--output-dir", str(merged), "--base-manifest", str(Path(config["baseline_run_dir"]) / "run_metadata.json"),
                      "--cpu-threads", "8", "--resume"],
                     cwd, state, state_path, environment=environment)
                state["completed_stages"].append("merged")
                _write_json(state_path, state)
            merge = _read(merge_manifest)
            server = copy.deepcopy(original)
            server.update(captured_utc=_utc(), model_path=str(merged),
                          model_artifacts=merge["model_artifacts"],
                          evaluated_model_name=original["served_model_name"],
                          adaptation={key: value for key, value in merge.items() if key != "model_artifacts"})
            source = original_launcher.read_text()
            old_argument = 'str(ROOT / "models/Qwen3.5-9B")'
            if source.count(old_argument) != 1:
                raise ValueError("Frozen serving launcher has no unique checkpoint argument")
            adapted_launcher = output / "serve_merged.py"
            adapted_source = source.replace(old_argument, repr(str(merged)))
            if adapted_launcher.exists():
                if adapted_launcher.read_text() != adapted_source:
                    raise ValueError("Recorded adapted serving launcher changed")
            else:
                adapted_launcher.write_text(adapted_source)
            server.update(launcher_path=str(adapted_launcher), launcher_sha256=_sha(adapted_launcher),
                          original_launcher_sha256=original["launcher_sha256"])
            server_path = output / "serving_config.json"
            if server_path.exists():
                preserved_server = _read(server_path)
                server["captured_utc"] = preserved_server["captured_utc"]
                if preserved_server != server:
                    raise ValueError("Recorded serving manifest changed; evaluation cannot resume")
            else:
                _write_json(server_path, server)
            versions = {name: importlib.metadata.version(name) for name in original["runtime_versions"]}
            if versions != original["runtime_versions"]:
                raise ValueError("Serving runtime versions changed from the frozen baseline")
            from .post_training_eval import validate_protocol
            validate_protocol(baseline, training, server,
                              input_path=Path(config["test_input"]), adapter_path=adapter)
            pid_file = Path(launch["restore_pid_file"])
            if "served" not in state["completed_stages"]:
                state.update(status="deploying_merged_weights", child_pid=None)
                _write_json(state_path, state)
                recorded_pid = int(pid_file.read_text().strip()) if pid_file.exists() else None
                previous_model = state.get("serving_model_path", original["model_path"])
                if recorded_pid and _proc(recorded_pid):
                    stop_owned_server(recorded_pid, previous_model, original)
                _, active = gpu_processes(original["gpu_index"], original["gpu_uuid"])
                if active:
                    raise RuntimeError("GPU is occupied; merged serving was not started")
                environment = os.environ.copy()
                environment.update(launch["environment"])
                with Path(launch["restore_log_file"]).open("ab", buffering=0) as log:
                    serving = subprocess.Popen([config["serving_python"], "-B", str(adapted_launcher)],
                                               cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                pid_file.write_text(str(serving.pid) + "\n")
                state.update(serving_pid=serving.pid, serving_model_path=str(merged))
                _write_json(state_path, state)
                restore_config = dict(launch, restore_timeout_seconds=240)
                ready = wait_serving(serving, restore_config)
                if ready["status"] != "ready":
                    raise RuntimeError("Merged serving did not become healthy")
                _owned_server(serving.pid, str(merged), original["served_model_name"])
                state.update(serving_readiness=ready)
                state["completed_stages"].append("served")
                _write_json(state_path, state)
            _owned_server(int(pid_file.read_text().strip()), str(merged), original["served_model_name"])

            if "plumbing" not in state["completed_stages"]:
                # One training-only request checks the actual checkpoint/API path;
                # its output is not retained or scored and does not select settings.
                from . import prompt
                from .runner import _request
                with Path(config["train_input"]).open() as stream:
                    row = json.loads(next(stream))
                frozen = baseline["config"]
                payload = {"model": frozen["model"], "messages": prompt.build_messages(row["input_context"]),
                           "temperature": frozen["temperature"], "top_p": frozen["top_p"],
                           "top_k": frozen["top_k"], "presence_penalty": frozen["presence_penalty"],
                           "max_tokens": frozen["max_tokens"], "seed": frozen["seed"], "stream": False,
                           "chat_template_kwargs": {"enable_thinking": frozen["thinking"] == "true"},
                           "response_format": {"type": "json_schema", "json_schema": {
                               "name": "judge_bench", "strict": True, "schema": prompt.JUDGE_SCHEMA}}}
                state.update(status="checking_training_only_request")
                _write_json(state_path, state)
                response = _request(frozen["base_url"] + "/chat/completions", _api_key(launch),
                                    payload=payload, timeout=frozen["request_timeout_seconds"])
                if not response.get("choices"):
                    raise RuntimeError("Training-only API request returned no completion")
                state["completed_stages"].append("plumbing")
                _write_json(state_path, state)

            if "evaluated" not in state["completed_stages"]:
                state.update(status="evaluating_800", child_pid=None)
                _write_json(state_path, state)
                _run([config["serving_python"], "-B", "-m", "evaluation.judge_bench.post_training_eval",
                      "--baseline-run-dir", config["baseline_run_dir"],
                      "--training-metadata", str(training_dir / "training_metadata.json"),
                      "--adapter-path", str(adapter), "--input", config["test_input"],
                      "--output-dir", config["after_run_dir"], "--server-config", str(server_path),
                      "--model", original["served_model_name"], "--env-file", launch["restore_env_file"], "--resume"],
                     cwd, state, state_path)
                state["completed_stages"].append("evaluated")
                _write_json(state_path, state)
            state.update(status="writing_comparison", child_pid=None)
            _write_json(state_path, state)
            comparison_command = [config["serving_python"], "-B", "-m", "evaluation.judge_bench.compare",
                  "--input", config["test_input"], "--before-run-dir", config["baseline_run_dir"],
                  "--after-run-dir", config["after_run_dir"], "--output-dir", config["comparison_dir"]]
            if config.get("split_manifest"):
                comparison_command.extend(["--split-manifest", config["split_manifest"]])
            _run(comparison_command, cwd, state, state_path)
            state.update(status="completed", finished_utc=_utc(), child_pid=None,
                         after_run_dir=config["after_run_dir"], comparison_dir=config["comparison_dir"])
            _write_json(state_path, state)
            return 0
        except Exception as error:
            state.update(status="failed", finished_utc=_utc(), error_type=type(error).__name__,
                         error=str(error), child_pid=None)
            _write_json(state_path, state)
            return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.config))


if __name__ == "__main__":
    main()
