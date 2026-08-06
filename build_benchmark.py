"""Internal benchmark construction driver (not an evaluated-agent CLI)."""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmark_audit import audit_benchmark
from benchmark_plan import instance_schedule, load_plan, scenario_schedule
from creators import InstanceCreator, NoiseCreator, build_scenario


ROOT_DIR = Path(__file__).resolve().parent
DOMAINS_DIR = ROOT_DIR / "domains"
BENCHMARK_DIR = ROOT_DIR / "benchmark"
PRINT_LOCK = threading.Lock()


def _print(message):
    with PRINT_LOCK:
        print(message, flush=True)


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _prepare_domains(plan):
    """Materialize the ten human-defined Domain folders without an LLM call."""
    DOMAINS_DIR.mkdir(parents=True, exist_ok=True)
    for spec in plan["domains"]:
        domain_dir = DOMAINS_DIR / spec["domain_id"]
        (domain_dir / "instances").mkdir(parents=True, exist_ok=True)
        _write_json(domain_dir / "domain.json", spec)


def _create_instance(domain_spec, instance_spec, max_attempts):
    domain_id = domain_spec["domain_id"]
    instance_id = instance_spec["instance_id"]
    instance_dir = DOMAINS_DIR / domain_id / "instances" / instance_id
    if instance_dir.exists():
        _print(f"[{domain_id}/{instance_id}] Instance exists; resuming")
        return
    creator = InstanceCreator()
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            _print(
                f"[{domain_id}/{instance_id}] creating Instance "
                f"({attempt}/{max_attempts})"
            )
            creator.create_instance(domain_spec, instance_id)
            return
        except Exception as error:
            last_error = error
            _print(
                f"[{domain_id}/{instance_id}] Instance failed: "
                f"{type(error).__name__}: {error}"
            )
    raise RuntimeError(f"Instance creation failed: {last_error}")


def _build_one_instance(plan, domain_spec, instance_spec, max_instance_attempts):
    domain_id, instance_id = domain_spec["domain_id"], instance_spec["instance_id"]
    _create_instance(domain_spec, instance_spec, max_instance_attempts)
    instance_dir = DOMAINS_DIR / domain_id / "instances" / instance_id
    construction_started = (instance_dir / "instance_output.json").stat().st_mtime
    scenario_results = []

    for item in scenario_schedule(plan, instance_spec):
        trigger, scenario_id = item["trigger_type"], item["scenario_id"]
        public_path = instance_dir / "scenarios" / trigger / f"{scenario_id}.json"
        if public_path.exists():
            _print(f"[{domain_id}/{instance_id}] {scenario_id} exists; skipping")
            scenario_results.append({"scenario_id": scenario_id, "status": "existing"})
            continue
        _print(f"[{domain_id}/{instance_id}] generating {scenario_id}")
        result = build_scenario(
            domain_id,
            instance_id,
            trigger,
            reference_date=item["reference_date"],
        )
        _print(
            f"[{domain_id}/{instance_id}] accepted {scenario_id} after "
            f"{result['attempts']} attempt(s)"
        )
        scenario_results.append(
            {
                "scenario_id": scenario_id,
                "status": "generated",
                "attempts": result["attempts"],
                "rejected": result["rejected"],
            }
        )

    noise_validation_path = instance_dir / "noise_validation.json"
    if noise_validation_path.exists() and json.loads(
        noise_validation_path.read_text(encoding="utf-8")
    ).get("valid"):
        _print(f"[{domain_id}/{instance_id}] Noise exists; skipping")
        existing_noise = json.loads(
            (instance_dir / "noise_output.json").read_text(encoding="utf-8")
        )
        noise_result = {
            "status": "existing",
            "command_count": len(existing_noise["noise_commands"]),
        }
    else:
        noise_creator = NoiseCreator()
        noise_output_path = instance_dir / "noise_output.json"
        if noise_output_path.exists() and not noise_validation_path.exists():
            _print(f"[{domain_id}/{instance_id}] revalidating saved Noise")
            created_noise = noise_creator.commit_saved_noise(domain_id, instance_id)
        else:
            action = (
                "regenerating failed Noise"
                if noise_output_path.exists()
                else "generating final Noise"
            )
            _print(f"[{domain_id}/{instance_id}] {action}")
            created_noise = noise_creator.create_noise(domain_id, instance_id)
        noise_result = {
            "status": "generated",
            "command_count": len(created_noise["noise"]["noise_commands"]),
        }

    review_path = instance_dir / "review_output.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    completion_time = (instance_dir / "noise_validation.json").stat().st_mtime
    return {
        "domain_id": domain_id,
        "instance_id": instance_id,
        "status": "complete" if review.get("valid") else "review_failed",
        "elapsed_seconds": round(completion_time - construction_started, 2),
        "scenario_results": scenario_results,
        "noise_result": noise_result,
        "review": review,
    }


def build_benchmark(
    workers=4,
    domain_ids=None,
    instance_ids=None,
    max_instance_attempts=4,
):
    plan = load_plan()
    _prepare_domains(plan)
    selected_domains = [
        spec
        for spec in plan["domains"]
        if domain_ids is None or spec["domain_id"] in domain_ids
    ]
    known_domains = {spec["domain_id"] for spec in plan["domains"]}
    unknown_domains = set(domain_ids or []) - known_domains
    if unknown_domains:
        raise ValueError(f"unknown Domain ids: {sorted(unknown_domains)}")

    jobs = []
    known_instances = set()
    for domain_spec in selected_domains:
        for instance_spec in instance_schedule(plan, domain_spec):
            known_instances.add(instance_spec["instance_id"])
            if instance_ids is None or instance_spec["instance_id"] in instance_ids:
                jobs.append((domain_spec, instance_spec))
    unknown_instances = set(instance_ids or []) - known_instances
    if unknown_instances:
        raise ValueError(f"unknown Instance ids: {sorted(unknown_instances)}")

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=min(workers, max(len(jobs), 1))) as executor:
        future_map = {
            executor.submit(
                _build_one_instance,
                plan,
                domain_spec,
                instance_spec,
                max_instance_attempts,
            ): (domain_spec["domain_id"], instance_spec["instance_id"])
            for domain_spec, instance_spec in jobs
        }
        for future in as_completed(future_map):
            domain_id, instance_id = future_map[future]
            try:
                results.append(future.result())
            except Exception as error:
                failure = {
                    "domain_id": domain_id,
                    "instance_id": instance_id,
                    "error": f"{type(error).__name__}: {error}",
                }
                failures.append(failure)
                _print(f"[{domain_id}/{instance_id}] BUILD FAILED: {failure['error']}")

    run_elapsed = round(time.monotonic() - started, 2)
    report = {
        "complete": not failures and all(item["status"] == "complete" for item in results),
        "run_elapsed_seconds": run_elapsed,
        "construction_elapsed_seconds": max(
            (item["elapsed_seconds"] for item in results), default=run_elapsed
        ),
        "selected_instance_count": len(jobs),
        "rejected_candidate_count": sum(
            len(
                list(
                    (
                        DOMAINS_DIR
                        / domain["domain_id"]
                        / "instances"
                        / instance["instance_id"]
                    ).glob("scenarios/*/logs/failed_*")
                )
            )
            for domain, instance in jobs
        ),
        "results": sorted(results, key=lambda item: item["instance_id"]),
        "failures": sorted(failures, key=lambda item: item["instance_id"]),
    }
    _write_json(BENCHMARK_DIR / "build_report.json", report)

    full_build = len(jobs) == plan["design"]["target_instance_count"]
    audit = audit_benchmark() if full_build and report["complete"] else None
    return report, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--domain", action="append", dest="domain_ids")
    parser.add_argument("--instance", action="append", dest="instance_ids")
    parser.add_argument("--max-instance-attempts", type=int, default=4)
    args = parser.parse_args()
    report, audit = build_benchmark(
        workers=args.workers,
        domain_ids=args.domain_ids,
        instance_ids=args.instance_ids,
        max_instance_attempts=args.max_instance_attempts,
    )
    valid = report["complete"] and (audit is None or audit["valid"])
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
