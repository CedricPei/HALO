"""Task assignment helpers for legacy packs and arbitrary annotator batches."""

from __future__ import annotations

from annotation_app.catalog import evaluation_groups


def balanced_assignment_plan(
    user_ids: list[str],
    audit_items: list[dict],
    eval_items: list[dict],
    audit_redundancy: int = 2,
) -> dict[str, dict[str, list[str]]]:
    if not user_ids:
        raise ValueError("At least one annotator is required.")
    if audit_redundancy < 1:
        raise ValueError("Audit redundancy must be at least 1.")
    if audit_redundancy > len(user_ids):
        raise ValueError(
            "Audit redundancy cannot exceed the number of annotators."
        )

    plan = {user_id: {"audit": [], "eval": []} for user_id in user_ids}
    for copy_index in range(audit_redundancy):
        for item_index, item in enumerate(audit_items):
            user_id = user_ids[(item_index + copy_index) % len(user_ids)]
            plan[user_id]["audit"].append(item["item_id"])

    for group_index, checkpoint_ids in enumerate(evaluation_groups(eval_items)):
        user_id = user_ids[group_index % len(user_ids)]
        plan[user_id]["eval"].extend(checkpoint_ids)
    return plan


def apply_assignment_plan(store, plan: dict[str, dict[str, list[str]]]) -> int:
    inserted = 0
    for user_id, assignments in plan.items():
        inserted += store.assign_tasks(user_id, "audit", assignments.get("audit", []))
        inserted += store.assign_tasks(user_id, "eval", assignments.get("eval", []))
    return inserted
