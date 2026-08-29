"""Balanced assignment of one task type across annotators."""

from __future__ import annotations

from benchmark_catalog import evaluation_groups


def assignment_plan(
    user_ids: list[str], review_type: str, items: list[dict]
) -> dict[str, list[str]]:
    plan = {user_id: [] for user_id in user_ids}
    if review_type == "audit":
        for copy_index in range(min(2, len(user_ids))):
            for item_index, item in enumerate(items):
                plan[user_ids[(item_index + copy_index) % len(user_ids)]].append(
                    item["item_id"]
                )
    else:
        for group_index, checkpoint_ids in enumerate(evaluation_groups(items)):
            plan[user_ids[group_index % len(user_ids)]].extend(checkpoint_ids)
    return plan
