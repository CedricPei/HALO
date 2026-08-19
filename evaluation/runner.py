"""Run proactive Agents checkpoint by checkpoint in isolated simulated state."""

import argparse
import json
import logging
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from command_parser import CommandParser
from interfaces.data import ROOT_DIR, load_construction_items
from llm_client import LLMClient, LLMProtocolError, LLMTransportError

from .intensity import action_intensity
from .prompt import (
    ACTION_COMMANDS,
    INSPECT_COMMANDS,
    build_agent_prompt,
)


DEFAULT_MODELS = ("gpt-5.6-sol",)
RESULTS_DIR = ROOT_DIR / "eval_results"
PREFIX_METRICS_PATH = RESULTS_DIR / "summary.json"
LOG_PATH = ROOT_DIR / "evaluation" / "evaluation.log"
LOGGER = logging.getLogger(__name__)
AUTO_TIMESTAMP_ACTIONS = {
    ("email", "create_draft"),
    ("email", "send_email"),
    ("messaging", "create_draft"),
    ("messaging", "send_message"),
    ("wallet", "create_draft"),
    ("wallet", "pay"),
    ("wallet", "refund_transaction"),
}


class AgentExecutionError(Exception):
    """An Agent command was well formed but could not be executed."""


def _slug(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


def _normalize_command(command):
    """Extract one command while tolerating harmless JSON-shape variation."""
    if not isinstance(command, dict):
        raise ValueError("command must be an object")
    allowed_fields = {"app", "function", "parameter"}
    if (
        not {"app", "function"} <= command.keys()
        or command.keys() - allowed_fields
    ):
        raise ValueError(
            "command must contain app, function, and optionally parameter"
        )
    if not all(
        isinstance(command.get(field), str) and command[field].strip()
        for field in ("app", "function")
    ):
        raise ValueError("command app and function must be non-empty strings")
    app = command["app"].strip().casefold()
    function = command["function"].strip().casefold()
    parameter = command.get("parameter")
    if parameter is not None and not isinstance(parameter, dict):
        raise ValueError("command parameter must be an object")
    return {
        "app": app,
        "function": function,
        "parameter": {} if parameter is None else parameter,
    }


def _surface_command(command, available, decision):
    command = _normalize_command(command)
    name = (command["app"], command["function"])
    if name not in available:
        raise ValueError(
            f"{command['app']}.{command['function']} is unavailable for {decision}"
        )
    if name == ("database", "execute"):
        sql = command["parameter"].get("sql", "")
        verb = (
            sql.lstrip().split(None, 1)[0].upper()
            if isinstance(sql, str) and sql.strip()
            else ""
        )
        if verb not in {"SELECT", "WITH", "PRAGMA"}:
            raise ValueError("Agent database access is read-only")
    if (
        decision == "final"
        and name in AUTO_TIMESTAMP_ACTIONS
        and "timestamp" in command["parameter"]
    ):
        raise ValueError(
            f"{command['app']}.{command['function']} supplies timestamp automatically"
        )
    return command


def _execute_final_action(parser, command, timestamp):
    execution_command = command
    if (command["app"], command["function"]) in AUTO_TIMESTAMP_ACTIONS:
        execution_command = {
            **command,
            "parameter": {**command["parameter"], "timestamp": timestamp},
        }
    try:
        return parser.execute_command(execution_command)
    except Exception as error:
        raise AgentExecutionError(
            f"{type(error).__name__}: {error}"
        ) from error


def _execute_inspection_commands(parser, commands):
    """Execute one validated intermediate round of Agent read commands."""
    results = []
    for command in commands:
        try:
            result = parser.execute_command(command)
        except Exception as error:
            raise AgentExecutionError(
                f"{type(error).__name__}: {error}"
            ) from error
        results.append({"command": command, "result": result})
    return results


TOKEN_METRICS = (
    "input_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
)
PREFIX_METRICS = (
    *TOKEN_METRICS,
    "latency_ms",
    "intermediate_round_count",
)


def _validate_response(response):
    if (
        not isinstance(response, dict)
        or not {"decision", "content"} <= response.keys()
    ):
        raise ValueError("Agent output must contain decision and content")
    decision = response["decision"]
    if isinstance(decision, str):
        decision = decision.strip().casefold()
    content = response["content"]
    if decision == "inspect":
        if isinstance(content, dict):
            content = [content]
        if not isinstance(content, list) or not content:
            raise ValueError("inspect content must be a non-empty command list")
        return {
            "decision": decision,
            "content": [
                _surface_command(item, INSPECT_COMMANDS, decision)
                for item in content
            ],
        }
    if decision != "final":
        raise ValueError("decision must be inspect or final")
    if content is None:
        return {"decision": decision, "content": None}
    if isinstance(content, list) and len(content) == 1:
        content = content[0]
    if not isinstance(content, dict):
        raise ValueError("final content must be null or exactly one command object")
    content = _normalize_command(content)
    available = (
        INSPECT_COMMANDS
        if (content["app"], content["function"]) in INSPECT_COMMANDS
        else ACTION_COMMANDS
    )
    content = _surface_command(content, available, decision)
    action_intensity(content)
    return {"decision": decision, "content": content}


def write_prefix_metrics():
    rows = []
    for path in sorted(RESULTS_DIR.glob("*/*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        for prefix_length, checkpoint in enumerate(run["checkpoints"], start=1):
            if checkpoint["agent"].get("decision") == "error":
                continue
            rows.append({
                "run_id": run["run_id"],
                "model": run["model"],
                "domain_id": run["domain_id"],
                "instance_id": run["instance_id"],
                "scenario_id": run["scenario_id"],
                "trigger_type": run["trigger_type"],
                "event_id": checkpoint["event_id"],
                "prefix_length": prefix_length,
                "action_intensity": checkpoint["agent"].get("action_intensity"),
                "metrics": {
                    key: checkpoint["metrics"][key]
                    for key in PREFIX_METRICS
                },
            })
    def summarize(selected):
        totals = {
            key: sum(row["metrics"][key] for row in selected)
            for key in PREFIX_METRICS
        }
        return {
            "prefix_action_count": len(selected),
            "totals": totals,
            "averages_per_prefix_action": {
                key: value / len(selected) for key, value in totals.items()
            },
        }

    report = {
        "models": {
            model: summarize([row for row in rows if row["model"] == model])
            for model in sorted({row["model"] for row in rows})
        },
        "prefix_actions": rows,
    }
    temporary = PREFIX_METRICS_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(PREFIX_METRICS_PATH)
    return report


def completed_jobs():
    """Return model/Scenario pairs already written by an earlier run."""
    jobs = set()
    for path in RESULTS_DIR.glob("*/*.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        jobs.add((result["model"], result["source_path"]))
    return jobs


class EvaluationRunner:
    def __init__(self):
        self.client = LLMClient()

    def _agent_checkpoint(self, model, parser, instance, event_prefix, execute_final):
        model_calls, checkpoint_results = [], []
        round_index = 0
        while True:
            prompt = build_agent_prompt(
                instance=instance,
                events=event_prefix,
                tool_results=checkpoint_results,
            )
            attempt_errors = []
            inspect_complete = False
            for output_attempt in range(1, 4):
                call_record = None
                try:
                    call = self.client.generate_json(prompt, model=model)
                    call_record = {
                        "round": round_index,
                        "output_attempt": output_attempt,
                        "raw_text": call["raw_text"],
                        "parsed": call["parsed"],
                        "usage": call["usage"],
                        "latency_ms": call["latency_ms"],
                        "endpoint": call["endpoint"],
                        "response_id": call["response_id"],
                    }
                    model_calls.append(call_record)
                    response = _validate_response(call["parsed"])
                    if response["decision"] == "inspect":
                        results = _execute_inspection_commands(
                            parser, response["content"]
                        )
                        checkpoint_results.extend(results)
                        round_index += 1
                        inspect_complete = True
                        break
                    intensity = action_intensity(response["content"])
                    if response["content"] is not None and intensity != "L0":
                        execute_final(response["content"])
                    return {
                        "decision": response["decision"],
                        "content": response["content"],
                        "action_intensity": intensity,
                        "intermediate_round_count": round_index,
                        "inspection_trace": checkpoint_results,
                        "model_calls": model_calls,
                    }
                except (LLMProtocolError, ValueError, AgentExecutionError) as error:
                    if isinstance(error, LLMProtocolError) and error.call:
                        model_calls.append({
                            "round": round_index,
                            "output_attempt": output_attempt,
                            **error.call,
                            "protocol_error": str(error),
                        })
                    elif call_record is not None:
                        call_record["attempt_error"] = (
                            f"{type(error).__name__}: {error}"
                        )
                    attempt_errors.append(f"{type(error).__name__}: {error}")
                    LOGGER.warning(
                        "[%s] Agent final-round attempt %s/3 failed; %s: %s",
                        model,
                        output_attempt,
                        "stopping" if output_attempt == 3 else "retrying final round",
                        error,
                    )
            if not inspect_complete:
                return {
                    "decision": "error",
                    "error": attempt_errors[-1],
                    "content": None,
                    "action_intensity": None,
                    "intermediate_round_count": round_index,
                    "inspection_trace": checkpoint_results,
                    "model_calls": model_calls,
                }

    def run(self, item, model):
        instance_dir = ROOT_DIR / "domains" / item["domain_id"] / "instances" / item["instance_id"]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = _slug(f"{timestamp}-{model}-{item['instance_id']}-{item['scenario_id']}")
        checkpoints = []
        events = list(item["scenario"]["events"].items())
        if any("time" not in event for _, event in events):
            raise ValueError(
                f"{item['scenario_id']} uses the legacy Event schema without time"
            )
        for checkpoint_index, (event_id, _) in enumerate(events):
            LOGGER.info(
                "[%s/%s/%s] checkpoint %s/%s",
                model,
                item["instance_id"],
                item["scenario_id"],
                checkpoint_index + 1,
                len(events),
            )
            event_prefix = []
            with tempfile.TemporaryDirectory() as temporary:
                episode_dir = Path(temporary) / item["instance_id"]
                shutil.copytree(instance_dir, episode_dir)
                parser = CommandParser.for_instance(episode_dir, item["instance_id"])
                for prefix_id, prefix_event in events[:checkpoint_index + 1]:
                    event_prefix.append({"event_id": prefix_id, **prefix_event})
                    parser.execute_command(
                        {
                            key: prefix_event[key]
                            for key in ("app", "function", "parameter")
                        }
                    )
                simulation_timestamp = event_prefix[-1]["time"]

                def execute_final(command):
                    with tempfile.TemporaryDirectory() as action_temporary:
                        action_dir = Path(action_temporary) / item["instance_id"]
                        shutil.copytree(episode_dir, action_dir)
                        action_parser = CommandParser.for_instance(
                            action_dir, item["instance_id"]
                        )
                        _execute_final_action(
                            action_parser, command, simulation_timestamp
                        )

                agent = self._agent_checkpoint(
                    model=model,
                    parser=parser,
                    instance=item["instance"],
                    event_prefix=event_prefix,
                    execute_final=execute_final,
                )
                calls = agent["model_calls"]
                metrics = {
                    **{
                        key: sum(call["usage"][key] for call in calls)
                        for key in TOKEN_METRICS
                    },
                    "latency_ms": sum(call["latency_ms"] for call in calls),
                    "intermediate_round_count": agent["intermediate_round_count"],
                }
                checkpoints.append({
                    "event_id": event_id,
                    "agent": agent,
                    "metrics": metrics,
                })
        totals = {
            key: sum(checkpoint["metrics"][key] for checkpoint in checkpoints)
            for key in PREFIX_METRICS
        }
        result = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "domain_id": item["domain_id"],
            "instance_id": item["instance_id"],
            "scenario_id": item["scenario_id"],
            "trigger_type": item["trigger_type"],
            "source_path": item["path"],
            "checkpoints": checkpoints,
            "totals": totals,
            "averages_per_prefix_action": {
                key: value / len(checkpoints)
                for key, value in totals.items()
            },
        }
        model_dir = RESULTS_DIR / _slug(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        output = model_dir / f"{run_id}.json"
        temporary_output = output.with_suffix(".tmp")
        temporary_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary_output.replace(output)
        return result, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--instance", action="append", dest="instances")
    parser.add_argument("--scenario")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")],
        force=True,
    )
    items = load_construction_items()
    if args.instances:
        items = [item for item in items if item["instance_id"] in args.instances]
    if args.scenario:
        items = [item for item in items if item["scenario_id"] == args.scenario]
    if not items:
        raise SystemExit("No matching accepted Scenarios")
    runner = EvaluationRunner()
    failures = 0
    completed = completed_jobs()
    jobs = [
        (model, item)
        for model in args.models or DEFAULT_MODELS
        for item in items
        if (model, item["path"]) not in completed
    ]
    if not jobs:
        write_prefix_metrics()
        print("All matching model/Scenario evaluations already exist.", flush=True)
        return
    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
        futures = {
            executor.submit(runner.run, item, model): (model, item)
            for model, item in jobs
        }
        for future in as_completed(futures):
            model, item = futures[future]
            try:
                result, path = future.result()
                print(
                    f"[{model}] {item['item_id']} -> {path.relative_to(ROOT_DIR)} "
                    f"({result['totals']['intermediate_round_count']} intermediate rounds, "
                    f"{result['totals']['input_tokens']} input + "
                    f"{result['totals']['completion_tokens']} completion + "
                    f"{result['totals']['reasoning_tokens']} reasoning, "
                    f"{result['totals']['total_tokens']} total, "
                    f"{result['totals']['latency_ms']:.0f} ms)",
                    flush=True,
                )
            except Exception as error:
                failures += 1
                print(f"[{model}] FAILED {item['item_id']}: {type(error).__name__}: {error}", flush=True)
                if isinstance(error, LLMTransportError):
                    for pending in futures:
                        pending.cancel()
                    break
    if not failures:
        write_prefix_metrics()
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
