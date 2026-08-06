"""Load the benchmark design and derive Instance and Scenario schedules."""

import json
from datetime import date, timedelta
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
PLAN_PATH = ROOT_DIR / "benchmark_plan.json"
TRIGGER_TYPES = ("dependency", "blockage", "risk", "opportunity")


def load_plan(path=PLAN_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def instance_schedule(plan, domain_spec):
    """Return the assigned IDs and trigger patterns for one Domain."""
    count = plan["design"]["instances_per_domain"]
    patterns = tuple(plan["trigger_order_patterns"])
    return [
        {
            "instance_id": f"{domain_spec['domain_id']}_{index:02d}",
            "pattern": patterns[(index - 1) % len(patterns)],
        }
        for index in range(1, count + 1)
    ]


def scenario_schedule(plan, instance_spec):
    """Return the fixed trigger, ID, and date sequence for one Instance."""
    order = plan["trigger_order_patterns"][instance_spec["pattern"]]
    expected_total = plan["design"]["scenarios_per_instance"]
    if len(order) != expected_total:
        raise ValueError(f"invalid trigger pattern for {instance_spec['instance_id']}")

    counts = {trigger: 0 for trigger in TRIGGER_TYPES}
    start = date(plan["design"]["benchmark_year"], 1, 8)
    schedule = []
    for index, trigger_type in enumerate(order):
        counts[trigger_type] += 1
        schedule.append(
            {
                "trigger_type": trigger_type,
                "scenario_id": f"{trigger_type}_{counts[trigger_type]}",
                "reference_date": (start + timedelta(days=30 * index)).isoformat(),
            }
        )

    expected = plan["design"]["scenarios_per_trigger"]
    if any(count != expected for count in counts.values()):
        raise ValueError(f"unbalanced trigger pattern for {instance_spec['instance_id']}")
    return schedule
