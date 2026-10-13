# JudgeBench v3: GPT-6 Sol with training demonstrations

This condition evaluates the new fixed 800 targets through Tiantoken using exact
`gpt-6-sol`, HTTPS Chat Completions, and `reasoning_effort=none`. It runs on CPU
while the authorized Qwen v3 training continues on GPU0.

`fewshot.py` selects four annotations from the 7,200 training rows for each
target. Checkpoint identity is `(scenario_key, event_id)`. All rows at the target
checkpoint are excluded, across all source models. Selection uses deterministic
SHA256 ordering, without labels or comments. It maximizes distinct checkpoints:
772 targets have four distinct other checkpoints; 28 have only three and use a
second distinct annotated action at one of those other checkpoints, as authorized
by the user. All four examples have distinct item IDs and input-context hashes.

The background is identical across a target and its eligible examples and is
included once. Each demonstration includes its checkpoint prefix, retrieval
rounds, candidate action, six human labels, and available original human comments.
The target includes its own prefix/retrieval/action and no labels. Examples do not
make their observed events or prior interventions observations at the target.
The original judge rubric and seven-field output schema are preserved, with an
explicit demonstration-use instruction appended to the system prompt.

Prepare once, then preserve the selection and message hashes:

```bash
python -B -m evaluation.judge_bench.fewshot \
  --train outputs/judge_bench_v3/train.jsonl \
  --test outputs/judge_bench_v3/test.jsonl \
  --output-dir outputs/judge_bench_v3/runs/gpt-6-sol-4shot

python -B -m evaluation.judge_bench.relay_eval \
  --config outputs/judge_bench_v3/gpt6_sol_launch.json
```

The API key is read from the remote environment or `.env`; it is never retained
in benchmark artifacts. Every HTTP attempt reserves an ordinal on disk before
sending and records the request hash, seed, response, usage and failure. The first
pass has no immediate structural retries. Failed items receive one recovery phase
with up to two additional structural retries, independently of the two transient
transport retries per phase. Successful and fully exhausted items are not rerun
on resume. An interrupted request with unknown delivery consumes a transport
attempt. Gold labels and cross-field semantic inconsistencies never trigger retries.
An exclusive file lock prevents duplicate supervisors.

The run freezes 16 concurrent requests, a 16,384 completion-token cap, temperature
1, top_p 0.95, presence_penalty 1.5 and base seed 20261010. It uses strict JSON
schema and non-streaming HTTP response bodies. Qwen-specific parameters, including
top_k and chat_template_kwargs, are omitted. Relay acceptance of `reasoning_effort=none`
was verified using an unscored training-only request; missing independent reasoning
usage details are recorded as unknown rather than evidence of zero hidden tokens.

Saved artifacts include prepared messages, example provenance and source hashes,
per-item durable request states, all raw responses, final and first-pass predictions,
metrics, protocol, model registry/capability observations and a standalone report.
Main metrics count failures as incorrect; intensity excludes the 31 human-null
rows and uses denominator 769. Joint3 includes all 800 and requires matching null
where present. Class recalls, non-Timely/non-Appropriate subsets and prediction
bias are reported separately.

This is a within-benchmark condition: both train and test cover the same 400
scenarios. Qwen v3 zero-shot runs and this GPT few-shot run share the targets but
receive different prompting information. Their difference is not a matched
model-only comparison and cannot establish unseen-scenario generalization.
