"""Balanced full-batch and remaining-work assignment."""

from __future__ import annotations

from benchmark_catalog import evaluation_groups


def all_assignment_plan(
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


def remaining_assignment_plan(
    user_ids: list[str],
    review_type: str,
    items: list[dict],
    submitted_users: dict[str, set[str]],
) -> dict[str, list[str]]:
    """Distribute unfinished coverage across existing and new annotators."""
    plan = {user_id: [] for user_id in user_ids}

    if review_type == "audit":
        user_set = set(user_ids)
        user_order = {user_id: index for index, user_id in enumerate(user_ids)}
        for item in items:
            item_id = item["item_id"]
            completed = submitted_users.get(item_id, set())
            target = min(2, len(completed | user_set))
            for _ in range(max(target - len(completed), 0)):
                available = [
                    user_id
                    for user_id in user_ids
                    if user_id not in completed and item_id not in plan[user_id]
                ]
                if not available:
                    break
                user_id = min(
                    available,
                    key=lambda value: (len(plan[value]), user_order[value]),
                )
                plan[user_id].append(item_id)
        return plan

    for checkpoint_ids in evaluation_groups(items):
        remaining = [
            item_id for item_id in checkpoint_ids if not submitted_users.get(item_id)
        ]
        if not remaining:
            continue
        completed_in_group = {
            user_id
            for item_id in checkpoint_ids
            for user_id in submitted_users.get(item_id, set())
            if user_id in plan
        }
        candidates = completed_in_group or set(user_ids)
        user_id = min(candidates, key=lambda value: (len(plan[value]), value))
        plan[user_id].extend(remaining)
    return plan
