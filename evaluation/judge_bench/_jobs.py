"""Process ownership and service readiness used by 400-item validation jobs."""

from __future__ import annotations

import csv
import io
import json
import os
import signal
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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
