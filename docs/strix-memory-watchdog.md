# Strix host-memory watchdog

`scripts/strix_memory_watchdog.py` is an external Linux command wrapper for headless Strix Halo validation. It does not change model loading or cache sizing. It measures host-wide memory from procfs and controls the launched command's process group.

```sh
./scripts/strix_memory_watchdog.py -- ./build/bin/llama-server <arguments>
```

The wrapper performs these checks and actions:

- It refuses to launch if `/proc/swaps` contains any active entry.
- It calculates used memory as `MemTotal - MemAvailable`. Linux reports these fields in KiB, so the wrapper multiplies each value by 1024 and keeps all accounting as integer bytes.
- It sends `SIGTERM` to the process group at 116 GiB used.
- It sends `SIGKILL` at 118 GiB used or 30 seconds after `SIGTERM`.
- It reports `grace_timeout` if any descendant requires `SIGKILL` after the soft-threshold grace period, even when the direct child exited earlier.
- It sends `SIGKILL` and fails if swap appears or required procfs data becomes unavailable during execution.
- It forwards wrapper `SIGHUP`, `SIGINT`, or `SIGTERM` to the process group, waits the configured grace period, then sends `SIGKILL` if any group member remains.
- It checks the process group after the direct child exits and cleans up remaining descendants before returning the child's classification.
- It applies the same bounded process-group cleanup if an unexpected post-launch error occurs.
- It propagates an unmonitored child exit code. A signal exit uses the shell convention `128 + signal`.

The 118 GiB emergency threshold leaves a 2 GiB sampling margin below the strict 120 GiB ceiling. The default sample interval is one second. This margin cannot guarantee the ceiling for a workload that can allocate more than 2 GiB between samples. Lower `--emergency-gib` or shorten `--sample-interval-seconds` for such a workload.

Use `--procfs-root` to select a different procfs mount or a test fixture. `--soft-gib`, `--emergency-gib`, `--grace-seconds`, and `--sample-interval-seconds` override the other defaults. The emergency threshold must remain below 120 GiB. The fail-closed timing bounds are a maximum 30-second grace, maximum one-second sample interval, and maximum five-second heartbeat age.

The wrapper writes timestamped JSON Lines records to standard error. Preflight, sample, signal, and final records include total, available, used, and peak-used bytes, swap entry count, child status, process-group status, threshold reason, and final classification where applicable. Signal records are written immediately after each process-group signal. Child standard input, standard output, and standard error are inherited unchanged.

## Watchdog-owned validation lease

Use all three artifact options together when another process must prove that it is inside the active watchdog process group:

```sh
./scripts/strix_memory_watchdog.py \
    --lease-path /run/deepseek-v41/watchdog-lease.json \
    --heartbeat-path /run/deepseek-v41/watchdog-heartbeat.json \
    --audit-path /run/deepseek-v41/watchdog-audit.jsonl \
    -- \
    python3 tools/deepseek-v41-trace/run_matrix.py <arguments>
```

The watchdog creates and exclusively locks the persistent audit before launch. It then starts an internal guardian as the new session and process-group leader; the guardian starts the supplied command in that same group without inheriting the private control pipe. After the guardian reports the payload PID, the watchdog atomically creates the lease and heartbeat. Existing artifact paths are rejected rather than overwritten. The payload receives the resolved paths through `STRIX_MEMORY_WATCHDOG_LEASE_PATH`, `STRIX_MEMORY_WATCHDOG_HEARTBEAT_PATH`, and `STRIX_MEMORY_WATCHDOG_AUDIT_PATH`. It also receives `STRIX_MEMORY_WATCHDOG_HEARTBEAT_MAX_AGE_SECONDS`.

The child can run before the first atomic lease rename. A matching preflight must retry the inherited lease path for a bounded interval and fail closed if a complete valid lease does not appear. It must not accept a lease path supplied separately by the operator. Consumers must require version 2; version 1 does not describe the guardian topology or timing policy and is rejected.

Lease format `strix-memory-watchdog-lease`, version 2, contains:

- `lease_id` and active/final `state`
- `watchdog_pid`, `watchdog_start_time_utc`, Linux `watchdog_start_time_ticks`, `watchdog_executable_path`, `watchdog_command_sha256`, `watchdog_script_path`, and `watchdog_script_sha256`
- exact `soft_bytes`, `emergency_bytes`, `strict_ceiling_bytes`, `grace_seconds`, and `sample_interval_seconds`
- `procfs_root`
- `guardian_pid`, payload `child_pid`, `child_process_group_id`, `command`, and `child_command_sha256`
- `heartbeat_path`, `max_heartbeat_age_seconds`, and `audit_path`
- device, inode, owner, and mode identity for atomic JSON artifacts, plus the watchdog-held audit descriptor identity
- the authoritative `final` audit record after termination

Heartbeat format `strix-memory-watchdog-heartbeat`, version 2, binds `lease_id`, watchdog PID/start ticks, child PID/process group, sequence, state, and update timestamps. Every memory sample first checks swap and memory thresholds, pulses the guardian through the private nonblocking pipe, then atomically replaces the heartbeat with the complete sample audit record and its persistent-audit record hash. It pulses again after persistence succeeds. A blocked audit or heartbeat write cannot delay the emergency signal; if persistence stalls past the guardian deadline, the guardian fails closed. A final heartbeat and final lease update remain on disk with the persistent JSONL audit; the watchdog does not delete this evidence.

The guardian uses Linux `PR_SET_PDEATHSIG` with a parent-race check. It kills its process group on watchdog death, control-pipe EOF/error, or a missed pulse deadline, including a stopped or wedged watchdog. When the watchdog sends a graceful signal, it also puts the guardian into a bounded grace mode and continues private pulses while it waits. This lets the watchdog own the configured grace deadline and record any `SIGKILL` escalation instead of letting the shorter heartbeat deadline preempt cleanup. If the grace control message or a cleanup pulse fails, the watchdog independently sends `SIGKILL` to the process group and reaps the child before it reports `signal_error`. The payload must call `start_process_group_lease_guard()` before it starts exporter descendants. This validates the lease with bounded startup retries, arms a second parent-death link to the guardian, and starts a thread that kills the process group if any validation or artifact operation fails or the watchdog evidence becomes stale.

A matching Linux preflight must verify all of the following:

- The inherited lease, heartbeat, and audit paths match the paths inside the lease.
- `/proc/<watchdog_pid>/exe` is the exact expected Python executable and argv position 1 is the exact repository watchdog script. `-c`, `-m`, helper-script, inert-argument, and interpreter-option substitutions are rejected.
- The watchdog command line itself supplies the exact 116/118 GiB thresholds, `/proc`, inherited artifact paths, timing policy, and command after `--`; the lease cannot override those expectations.
- `/proc/<watchdog_pid>/stat` start ticks and `/proc/<watchdog_pid>/cmdline` SHA-256 match the lease and remain stable across validation. A pidfd is held during validation when Linux provides `pidfd_open`.
- The topology is watchdog parent -> guardian process-group leader -> payload child. The current process must be inside `child_process_group_id`.
- The command identity is expected, the procfs root is `/proc`, and thresholds are exactly 116 GiB soft, 118 GiB emergency, and 120 GiB strict ceiling for the final run.
- Lease and heartbeat files are regular, mode 0600, owned by the current UID, opened with `O_NOFOLLOW`, and match their recorded device/inode identity.
- The heartbeat identity matches the lease, its monotonic timestamp is not older than `max_heartbeat_age_seconds`, and its audit-record hash exists in the persistent audit.
- The persistent audit matches the watchdog-held descriptor device/inode and remains exclusively locked by the live watchdog.

These checks reject accidental or helper-process substitution and make regular-file heartbeat forgery unable to keep the process group alive after private pulses stop. They are not a security boundary against intentionally hostile code running as the same UID; use a separately owned systemd user service or cgroup if that threat is in scope.

The guardian controls only the process group. A payload that deliberately calls `setsid()` can escape it. The correctness harness must not do that. If arbitrary payload code is in scope, launch the watchdog in a service/cgroup configured to kill every member when the unit stops.

Exit classifications are authoritative in the last final JSON record. If final artifact persistence fails after a primary safety failure, the primary classification and exit code remain unchanged and the artifact failure is listed in `secondary_errors`. Operational failures use these exit codes:

| Exit code | Classification |
| ---: | --- |
| 2 | procfs or configuration error |
| 3 | swap active at startup or detected during execution |
| 4 | soft threshold reached |
| 5 | emergency threshold reached |
| 6 | soft-threshold grace period expired |
| 7 | process-group signaling or termination failure |
| 8 | lease, heartbeat, or persistent audit failure |
| 70 | unexpected post-launch error |
| 127 | command launch failure |

No model, backend, or ROCm package is required to run the unit tests:

```sh
python3 tests/test_strix_memory_watchdog.py
```
