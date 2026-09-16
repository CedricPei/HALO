"""Balanced full-batch and remaining-work assignment."""

from __future__ import annotations

from collections import deque

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


def quota_audit_assignment_plan(
    user_ids: list[str],
    items: list[dict],
    submitted_users: dict[str, set[str]],
    pending_users: dict[str, set[str]] | None = None,
) -> dict[str, list[str]]:
    """Fill two review slots per item and an equal quota per selected account.

    Submitted work by selected accounts is fixed. Existing pending assignments
    are preferred where possible, but can move to satisfy the exact quota.
    """
    if len(user_ids) < 2 or len(user_ids) != len(set(user_ids)):
        raise ValueError("Select at least two distinct Audit accounts.")
    item_ids = [item["item_id"] for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("The Audit catalog contains duplicate item IDs.")
    if (2 * len(item_ids)) % len(user_ids):
        raise ValueError("The catalog cannot be divided equally among these accounts.")

    quota = 2 * len(item_ids) // len(user_ids)
    selected = set(user_ids)
    preferred = pending_users or {}
    fixed_by_item: dict[str, set[str]] = {}
    fixed_by_user = {user_id: 0 for user_id in user_ids}
    for item_id in item_ids:
        fixed = submitted_users.get(item_id, set()) & selected
        if len(fixed) > 2:
            raise ValueError(f"Item {item_id} has more than two fixed reviews.")
        fixed_by_item[item_id] = fixed
        for user_id in fixed:
            fixed_by_user[user_id] += 1
    if any(count > quota for count in fixed_by_user.values()):
        raise ValueError("A selected account has already submitted over its quota.")
    if any(
        item_id not in fixed_by_item and submitted_users.get(item_id, set()) & selected
        for item_id in submitted_users
    ):
        raise ValueError("A selected account has submitted an item outside the catalog.")

    source = 0
    first_item = 1
    first_user = first_item + len(item_ids)
    sink = first_user + len(user_ids)
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, capacity: int) -> None:
        graph[start].append([end, capacity, len(graph[end])])
        graph[end].append([start, 0, len(graph[start]) - 1])

    needed = 0
    for item_index, item_id in enumerate(item_ids):
        fixed = fixed_by_item[item_id]
        slots = 2 - len(fixed)
        needed += slots
        add_edge(source, first_item + item_index, slots)
        candidates = [user_id for user_id in user_ids if user_id not in fixed]
        candidates.sort(key=lambda user_id: user_id not in preferred.get(item_id, set()))
        for user_id in candidates:
            add_edge(first_item + item_index, first_user + user_ids.index(user_id), 1)
    for user_index, user_id in enumerate(user_ids):
        add_edge(first_user + user_index, sink, quota - fixed_by_user[user_id])

    flow = 0
    while flow < needed:
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        previous[source] = (-1, -1)
        queue = deque([source])
        while queue and previous[sink] is None:
            node = queue.popleft()
            for edge_index, edge in enumerate(graph[node]):
                if edge[1] > 0 and previous[edge[0]] is None:
                    previous[edge[0]] = (node, edge_index)
                    queue.append(edge[0])
                    if edge[0] == sink:
                        break
        if previous[sink] is None:
            raise ValueError("Existing submitted reviews prevent an equal Audit split.")
        amount = needed - flow
        node = sink
        while node != source:
            parent, edge_index = previous[node]
            amount = min(amount, graph[parent][edge_index][1])
            node = parent
        node = sink
        while node != source:
            parent, edge_index = previous[node]
            edge = graph[parent][edge_index]
            edge[1] -= amount
            graph[node][edge[2]][1] += amount
            node = parent
        flow += amount

    plan = {user_id: [] for user_id in user_ids}
    for item_index, item_id in enumerate(item_ids):
        for edge in graph[first_item + item_index]:
            user_index = edge[0] - first_user
            if 0 <= user_index < len(user_ids) and edge[1] == 0:
                plan[user_ids[user_index]].append(item_id)
    return plan
