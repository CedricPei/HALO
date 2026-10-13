# Requested one-epoch completion of the active v3 job

The user shortened the running 2-epoch/900-update experiment to one full pass
through all 7200 training rows, using the complete checkpoint at update 450.
The original launch, trainer, sampler, loss weights and 900-update cosine
schedule remain intact. This is an explicitly recorded epoch-boundary cap,
not a newly scheduled 450-update run. No test labels inform the decision.

`epoch_cap.py` runs as a separate CPU supervisor. Its frozen request records
the three original process identities, including UID, process group, Linux
start ticks and full command. It waits for `checkpoint-450/trainer_state.json`;
the installed Trainer writes this after model, optimizer, scheduler and RNG.
It checks the complete checkpoint and pauses only the declared waiting
post-training supervisor and training child. The adapter tensors must be
finite FP32 and cover exactly the trained targets. The final adapter is a
byte-for-byte copy of this boundary checkpoint.

The child receives SIGTERM followed by SIGCONT. The existing training
supervisor, which is never stopped, restores its original base-model service.
The original nonzero child exit and supervisor result are preserved in
`epoch_cap_job/original_supervisor_exit.json`; they are expected consequences
of the requested cap and must not be reported as an unrecorded successful
trainer return. After restoration is verified, training metadata records
`global_step=450`, `epochs_completed=1`, the original configuration and an
`epoch_cap_amendment`. The training job records completion of the requested
artifact while retaining its real `exit_code=-15`. The post supervisor is
then resumed and follows the existing CPU merge and frozen evaluation.

`epoch_cap_job/job.json` records each transition durably and has an exclusive
lock. A resumed controller cannot send signals to reused PIDs. Before the
training child is terminated, a failed transition resumes paused processes
and records `failed_before_training_stop`; inspect it before creating any
replacement request. After child termination, a failure preserves the exact
stage and the paused post supervisor for audited recovery, without inventing
a fresh training or evaluation budget. A completed controller performs no
additional work when resumed. A paused post supervisor during these recorded
transitions is intentional.

Retain the request, controller state/log, boundary checkpoint manifest,
pre-stop snapshots, original exit evidence and completion amendment with the
experiment. The final result is the one-epoch adapter even if a few subsequent
microbatches had begun when the trainer was paused; the last logged update at
pause is recorded separately. GPU-monitor shutdown statistics are unavailable
because the trainer was deliberately terminated; do not fabricate them.
