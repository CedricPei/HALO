"""Run one training job, then restore its declared base-model serving process.

Launch this supervisor in a new session externally. Its training child inherits
the supervisor's process group; restored serving receives its own session.
Only standard-library imports are used, and no benchmark evaluation is started.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from typing import Any
import urllib.parse
import urllib.request


ENVIRONMENT_ALLOWLIST = frozenset({
    "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_MODULE_LOADING", "LD_LIBRARY_PATH", "LIBRARY_PATH", "PATH",
    "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR",
    "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
    "HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TORCH_HOME", "TORCH_EXTENSIONS_DIR",
    "TRITON_CACHE_DIR", "XDG_CACHE_HOME", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TOKENIZERS_PARALLELISM",
    "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
})


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _command(value: Any, name: str) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value or any(not isinstance(arg, str) or not arg for arg in value):
        raise ValueError(f"{name} must be a nonempty executable argument list")
    # API credentials belong in an existing environment file, never argv/config.
    if any("api-key" in arg.lower() or "api_key=" in arg.lower() for arg in value):
        raise ValueError(f"{name} must not contain API credentials; use restore_env_file")
    return value


def _resolve(value: str, cwd: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else cwd / path


def load_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Launch config must be a JSON object")
    permitted = {
        "command", "cwd", "environment", "output_dir", "gpu_index", "gpu_uuid",
        "restore_serving_launcher", "restore_pid_file", "restore_log_file",
        "restore_model_id", "restore_model_path", "restore_base_url",
        "restore_env_file", "restore_api_key_env", "restore_timeout_seconds", "temporary_root",
    }
    unknown = set(raw) - permitted
    if unknown:
        raise ValueError(f"Unknown launch config keys: {sorted(unknown)}")
    cwd = _resolve(raw["cwd"], path.resolve().parent).resolve()
    if not cwd.is_dir():
        raise ValueError("Training cwd does not exist")
    config = dict(raw, command=_command(raw["command"], "command"), cwd=str(cwd))
    environment = raw.get("environment", {})
    if not isinstance(environment, dict) or any(
        key not in ENVIRONMENT_ALLOWLIST or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValueError("Environment overrides must be public strings from ENVIRONMENT_ALLOWLIST")
    config["environment"] = environment
    config["output_dir"] = str(_resolve(raw["output_dir"], cwd).resolve())
    gpu_index = raw.get("gpu_index", 0)
    if not isinstance(gpu_index, int) or isinstance(gpu_index, bool) or gpu_index < 0:
        raise ValueError("gpu_index must be a nonnegative integer")
    config["gpu_index"] = gpu_index
    if "CUDA_VISIBLE_DEVICES" in environment and environment["CUDA_VISIBLE_DEVICES"] != str(gpu_index):
        raise ValueError("CUDA_VISIBLE_DEVICES must select exactly gpu_index")
    if raw.get("temporary_root"):
        config["temporary_root"] = str(_resolve(raw["temporary_root"], cwd).resolve())
    if raw.get("restore_serving_launcher"):
        config["restore_serving_launcher"] = _command(raw["restore_serving_launcher"], "restore_serving_launcher")
        for key in ("restore_pid_file", "restore_log_file"):
            if not raw.get(key):
                raise ValueError(f"{key} is required when restoring serving")
        for key in ("restore_pid_file", "restore_log_file", "restore_env_file", "restore_model_path"):
            if raw.get(key):
                config[key] = str(_resolve(raw[key], cwd).resolve())
        base_url = raw.get("restore_base_url", "http://127.0.0.1:8003/v1").rstrip("/")
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("restore_base_url must be HTTP(S), without credentials")
        config["restore_base_url"] = base_url
        timeout = raw.get("restore_timeout_seconds", 240)
        if not isinstance(timeout, (int, float)) or not 0 < timeout <= 240:
            raise ValueError("restore_timeout_seconds must be in (0,240]")
        config["restore_timeout_seconds"] = timeout
        config["restore_api_key_env"] = raw.get("restore_api_key_env", "QWEN35_LOCAL_API_KEY")
    return config


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def gpu_processes(gpu_index: int, expected_uuid: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    gpu_uuid = result.stdout.strip()
    if not gpu_uuid.startswith("GPU-") or "\n" in gpu_uuid or (expected_uuid and gpu_uuid != expected_uuid):
        raise RuntimeError("Declared GPU identity does not match nvidia-smi")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    processes = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row or row[0].strip() != gpu_uuid:
            continue
        if len(row) != 3:
            raise RuntimeError("Could not parse GPU compute-process identity")
        processes.append({"pid": int(row[1].strip()), "process_name": row[2].strip()})
    return gpu_uuid, processes


def _api_key(config: dict[str, Any]) -> str | None:
    name = config["restore_api_key_env"]
    if os.environ.get(name):
        return os.environ[name]
    if not config.get("restore_env_file"):
        return None
    with Path(config["restore_env_file"]).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].lstrip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1]
                return value or None
    return None


def wait_serving(process: subprocess.Popen, config: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + config["restore_timeout_seconds"]
    base_url = config["restore_base_url"]
    parsed = urllib.parse.urlsplit(base_url)
    health_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))
    key = _api_key(config)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    expected_model = config.get("restore_model_id")
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return {"status": "failed", "reason": "Serving launcher exited", "exit_code": process.returncode}
        try:
            remaining = max(0.1, deadline - time.monotonic())
            with urllib.request.urlopen(urllib.request.Request(health_url, headers=headers), timeout=min(5, remaining)) as response:
                if response.status != 200:
                    raise ValueError("Health endpoint was not ready")
            model_ids = None
            if expected_model:
                remaining = max(0.1, deadline - time.monotonic())
                with urllib.request.urlopen(urllib.request.Request(base_url + "/models", headers=headers), timeout=min(5, remaining)) as response:
                    body = json.load(response)
                model_ids = [item["id"] for item in body["data"]]
                if expected_model not in model_ids:
                    raise ValueError("Serving model identity differs from the declared base model")
            return {"status": "ready", "ready_utc": _utc(), "health_url": health_url,
                    "model_ids": model_ids, "readiness_wall_seconds": time.monotonic() - started}
        except Exception as error:
            # Error text/response bodies may contain credentials; retain type only.
            last_error = type(error).__name__
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    return {"status": "timeout", "last_error_type": last_error,
            "readiness_wall_seconds": time.monotonic() - started,
            "process_still_running": process.poll() is None}


def restore_serving(config: dict[str, Any], job: dict[str, Any], job_path: Path) -> dict[str, Any]:
    if not config.get("restore_serving_launcher"):
        return {"status": "not_requested"}
    try:
        gpu_uuid, processes = gpu_processes(config["gpu_index"], config.get("gpu_uuid"))
        restore = {"status": "checking", "gpu_uuid": gpu_uuid, "gpu_processes_before_restore": processes}
        if processes:
            return dict(restore, status="blocked", reason="GPU has active compute processes; none were stopped")
        pid_file = Path(config["restore_pid_file"])
        if pid_file.exists():
            old_pid = int(pid_file.read_text().strip())
            if old_pid > 0 and _alive(old_pid):
                return dict(restore, status="blocked", reason="Recorded serving PID is still alive", pid=old_pid)
        log_file = Path(config["restore_log_file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(config["environment"])
        with log_file.open("ab", buffering=0) as log:
            serving = subprocess.Popen(
                config["restore_serving_launcher"], cwd=config["cwd"], env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        pid_file.write_text(str(serving.pid) + "\n", encoding="utf-8")
        restore.update(status="starting", pid=serving.pid, started_utc=_utc(), log_file=str(log_file),
                       model_id=config.get("restore_model_id"), model_path=config.get("restore_model_path"))
        job["restore"] = restore
        _write_json(job_path, job)
        restore.update(wait_serving(serving, config))
        return restore
    except Exception as error:
        return {"status": "failed", "reason": "Serving restoration raised an error", "error_type": type(error).__name__}


def run(config_path: Path) -> int:
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    job_path = output_dir / "job.json"
    if job_path.exists():
        raise ValueError("job.json already exists; use a new job directory to preserve its history")
    job = {
        "schema_version": 1, "status": "starting", "supervisor_pid": os.getpid(),
        "started_utc": _utc(), "launch_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "command": config["command"], "cwd": config["cwd"], "environment": config["environment"],
        "gpu_index": config["gpu_index"], "gpu_uuid": config.get("gpu_uuid"),
        "benchmark_evaluation_started": False,
    }
    _write_json(job_path, job)
    try:
        gpu_uuid, processes = gpu_processes(config["gpu_index"], config.get("gpu_uuid"))
        job.update(gpu_uuid=gpu_uuid, gpu_processes_before_training=processes)
        if processes:
            job.update(status="blocked", reason="GPU has active compute processes; training was not started",
                       exit_code=1, finished_utc=_utc(), restore={"status": "not_attempted"})
            _write_json(job_path, job)
            return 1
    except Exception as error:
        job.update(status="blocked", reason="Could not verify GPU identity and availability",
                   error_type=type(error).__name__, exit_code=1, finished_utc=_utc(),
                   restore={"status": "not_attempted"})
        _write_json(job_path, job)
        return 1
    environment = os.environ.copy()
    environment.update(config["environment"])
    child = None
    exit_code = 1
    temporary_root = config.get("temporary_root")
    if temporary_root:
        Path(temporary_root).mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="judge-sft-", dir=temporary_root) if temporary_root else nullcontext(None)
    with temporary as temporary_dir:
        if temporary_dir:
            environment["TMPDIR"] = temporary_dir
            job["training_temporary_directory"] = temporary_dir
        try:
            child = subprocess.Popen(config["command"], cwd=config["cwd"], env=environment, start_new_session=False)
            job.update(status="training", child_pid=child.pid)
            _write_json(job_path, job)
            exit_code = child.wait()
            job.update(status="training_succeeded" if exit_code == 0 else "training_failed", exit_code=exit_code,
                       training_finished_utc=_utc())
        except BaseException as error:
            job.update(status="training_failed", error_type=type(error).__name__)
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            if child is not None:
                job.update(exit_code=child.returncode, training_finished_utc=_utc())
    job["training_temporary_directory_cleaned"] = bool(temporary_dir)
    _write_json(job_path, job)
    job["restore"] = restore_serving(config, job, job_path)
    restored = job["restore"]["status"] in {"ready", "not_requested"}
    job.update(status="completed" if exit_code == 0 and restored else "failed", finished_utc=_utc())
    _write_json(job_path, job)
    print(json.dumps({"event": "training_job_finished", **job}, ensure_ascii=False), flush=True)
    return exit_code if exit_code > 0 else 0 if exit_code == 0 and restored else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-config", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.launch_config))


if __name__ == "__main__":
    main()
