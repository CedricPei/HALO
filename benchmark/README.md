# Proactive benchmark

The dataset is organized as `domains/<domain_id>/instances/<instance_id>`.

Each broad Domain is specified by a human in `benchmark_plan.json`. An Instance
contains one protagonist and one shared initial state. Its eight Scenarios add
their required initial data to that state, while every Scenario's Event stream
runs independently from it.

Construction uses four LLM roles—Instance Creator, combined Scenario Creator,
Reviewer, and one final Noise Creator—plus one deterministic Validator. The
Instance Creator sees existing Instances in its Domain, and Scenario Creator
sees accepted same-trigger Scenarios separately for duplicate avoidance. The
Scenario Creator emits Events and optional `ini_data_commands` together. A
candidate is committed only after Validator and Reviewer pass. After all eight
Scenarios, Noise Creator adds one validated batch of unrelated background data.
A complete audit writes only `benchmark/manifest.json`.
