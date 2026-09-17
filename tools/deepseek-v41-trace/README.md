# DeepSeek V4.1 correctness traces

This directory defines version 2 of the cross-runtime trace format used by issue #48. It compares the unchanged published GGUF between llama.cpp and ds4 revision `bd66c402070042bf0a79ad6ece8242de4c93680c`.

Each trace is a directory:

- `manifest.json` records the model and prompt SHA-256 values, exact runtime revision/build, canonical executable path, loaded runtime-library paths and hashes, inference configuration, environment, runtime-specific host evidence, exact execution paths, and content-addressed audit references.
- `events.jsonl` is an ordered stream of content-addressed event records.
- `blobs/<sha256>.bin` stores canonical little-endian tensor bytes. This keeps complete logits and per-token state exact without embedding large numeric arrays in JSON.
- `audits/pre/<sha256>.json` and `audits/post/<sha256>.json` store immutable safety evidence from both sides of execution.
- `provenance/<sha256>.json` binds the exact prompt to its fixed corpus, published model, target token count, and prompt-builder executable.
- `bundle-signature.json` contains the detached OpenSSH signature envelope for the exact bundle file set.

Seal v1 signs `dsv41-trace-bundle-v1\n` followed by canonical JSON records for `manifest.json`, `events.jsonl`, every referenced event blob, every embedded audit, and prompt provenance. Each record binds its canonical relative path, byte count, and SHA-256. Validation rejects missing or added files, symlinks, hard links, nonregular files, path traversal, duplicate metadata references, noncanonical JSON or JSONL, duplicate JSON keys, truncation, concurrent replacement, and every post-seal mutation.

Signer trust is external to the bundle. `APPROVED_TRACE_SIGNERS` maps one restricted ASCII principal to one exact OpenSSH Ed25519 public key, runtime lane, and runtime profile, and is intentionally empty until a separately authorized run. Candidate and oracle signers are not interchangeable. Every signed manifest also binds an externally supplied 256-bit challenge, lane-specific run ID, and bounded validity window. Validation requires those expected values from outside the bundle and rejects missing, mismatched, reused within a comparison, cross-lane, not-yet-valid, or expired authorization. The bundle cannot provide a public key, authoritative principal, verifier path, or allowed-signers file. Validation uses only `/usr/bin/ssh-keygen` on macOS and Linux or `C:\Windows\System32\OpenSSH\ssh-keygen.exe` on Windows, rejects symlinks and unsupported `-Y` implementations, clears SSH-agent influence, and never searches `PATH`. The private signing key must be an owned restrictive regular file outside the bundle and is never copied or logged. Tests use explicit test-only verifiers with ephemeral lane-specific keys; production validation does not trust those keys.

Executable trust is also external to the bundle and to the reviewed source revision. `APPROVED_EXECUTABLE_APPROVERS`, `APPROVED_CANDIDATE_EXPORTERS`, and `APPROVED_PROMPT_BUILDERS` are intentionally empty in production source. An authorized run must receive a canonical detached approval policy plus its OpenSSH signature and an externally expected approver principal. Both files must be absolute canonical non-symlinked one-link regular files outside every protected output root. Their trusted root and owner come only from `APPROVED_EXECUTABLE_APPROVERS`; the fixed `ssh-keygen` verifies namespace `dsv41-executable-approval-v2` before either executable can run. This avoids an impossible same-revision self-reference: the policy records the artifact producer revision and exact executable hashes, while a separate `verifier_revision` records the harness revision that consumes the policy.

The candidate exporter approval binds the exact producer revision, base revision, binary diff SHA-256, canonical install root and exporter path, exporter SHA-256, runtime profile, complete embedded runtime receipt, and the exact revision, filename, and SHA-256 of the native Linux containment helper. The prompt-builder approval binds the same helper identity with its producer revision, canonical install/source roots and executable path, executable SHA-256, exact runtime profile and receipt, model and corpus identities, complete tokenizer policy, and exact prompt hash, byte count, context, and decode configuration for every authorized case. The tokenizer policy explicitly binds `add_bos`, `parse_special`, `detokenize_special`, leading-BOS removal, and exact token round-trip. The builder invocation, native result, signed provenance, execution authorization, candidate environment, and candidate manifest must all agree with it.

Every approved install tree is verified before launch. The install owner is externally selected and must differ from the unprivileged execution identity. Every ancestor is canonical, non-symlinked, trusted-owned, ACL-free, and not writable by the execution identity or group/other. Every approved executable, containment helper, and receipt library is a canonical, ACL-free, non-writable, one-link regular file with the exact owner and SHA-256. On Linux, the exporter, prompt builder, and containment helper are opened without following the final symlink and launched through retained `/proc/self/fd` descriptors. Receipt-library descriptors remain open through process completion, while the immutable canonical install hierarchy prevents pathname substitution during loader consumption. Both native tools report the actual canonical loaded project-library paths and hashes; the prompt provenance and candidate manifest require the loaded closure to equal the approved receipt before and after protected work. Ordinary user-owned install smoke verifies layout, RPATH, helper receipt, version, and bytes only; it does not establish production trusted-root authorization.

The llama runner opens the approved model once without following aliases, hashes that held read-only descriptor before launch, retains it through Linux containment, and directs the exporter to load through `/proc/self/fd`. The runner and exporter bind device, inode, owner, mode, link count, size, timestamps, descriptor flags, and SHA-256 before model initialization and verify the same descriptor plus the original pathname after tracing. The watchdog is validated against host `/proc` before containment and represented inside the private PID namespace by a retained live pidfd. The signed audit binds the host PID, process group, start time, executable, command, guardian, and child identities, while the candidate manifest separately requires the exporter to be namespace-local PID 2 with parent PID 1, its own process group and session, matching NSpid, and private procfs. Host numeric PIDs are never interpreted as namespace-local PIDs.

Prompt provenance version 2 records the lexical and resolved source root and corpus paths. The prompt builder accepts path aliases only when both resolve to the exact approved repository corpus and binds the canonical source identity into the signed provenance.

The signed execution authorization records the complete approval-policy SHA-256, verifier revision, tokenizer-policy SHA-256, and both selected approval record IDs, policy hashes, and install-trust hashes. Candidate-derived attestations and prompt provenance are evidence only and must exactly equal those external records. Seal creation performs the same semantic manifest, provenance, event, and coverage validation as standalone verification before invoking the signer.

Every manifest and memory audit declares that the expert cache and KV cache are memory-resident and that there are no external cache or state paths. Missing, substituted, or additional file-backed cache/state declarations fail closed.

The required hard-failure event components are `prompt.bytes`, `prompt.tokens`, `engram.row_ids`, `expert.ids`, `expert.weights`, `attn.source`, `attn.candidate_blocks`, `attn.candidates`, `logits.prefill`, `logits.decode`, and `decode.greedy_token`. `expert.ids` must declare `semantic_id_space: "original"`; cache slot IDs are rejected. Any graph tensor in the reserved `dsv41.trace.*` namespace with an unknown component, malformed suffix, or unexpected layer fails the exporter.

Every bundle carries one strict runtime-discriminated accelerator attestation. A llama.cpp candidate uses the `strix-rocm` kind: the selected backend device must map through its PCI identity and Linux KFD topology to `gfx_target_version=110501` (`gfx1151`). A ds4 oracle uses the `apple-metal` kind: the selected Metal device records its registry ID, reported architecture, unified-memory property, and recommended working-set size. Missing kinds, unknown kinds, cross-kind fields, duplicate JSON keys, and mixed evidence fail closed. Cross-runtime comparison does not require the two physical accelerators or PCI identities to match; each runtime proves its own execution environment, while the model, prompt, inference semantics, and complete output artifacts remain exact comparison inputs.

Internal tensors use raw ggml dimension order, and every dimension must be positive. The validator requires Engram rows as i32 `[24, token_count]`, original expert IDs as i32 `[6, token_count]`, router weights as f32 `[6, token_count]`, layer-0/1 raw attention-source rows as i32 `[128, token_count]`, compressed attention-source IDs as nonempty rank-2 i32 with width at most 512, layer-20 candidate blocks as nonempty rank-2 i32 with width at most 2048, propagated candidates as nonempty rank-2 i32 with width at most 512, and complete f32 logits as `[129280]`. Raw attention rows use physical ring IDs `0..127`, visible current-ubatch IDs `128..128+token_index`, and unavailable sentinel `128+token_count`; layers 0 and 1 must be byte-identical for each execution step. Original expert IDs must be within `0..383`.

Validate or compare bundles:

```sh
python3 tools/deepseek-v41-trace/trace_format.py validate TRACE \
  --signer-principal PRINCIPAL --lane strix-llama-candidate-v1 \
  --execution-challenge "$CHALLENGE" --run-id "$RUN_ID" \
  --candidate-exporter-policy-id "$CANDIDATE_EXPORTER_POLICY_ID" \
  --prompt-builder-policy-id "$PROMPT_BUILDER_POLICY_ID" \
  --approval-policy "$APPROVAL_POLICY" \
  --approval-signature "$APPROVAL_SIGNATURE" \
  --approval-principal "$APPROVAL_PRINCIPAL"
python3 tools/deepseek-v41-trace/trace_format.py compare DS4_TRACE LLAMA_TRACE \
  --left-signer-principal DS4_PRINCIPAL --right-signer-principal LLAMA_PRINCIPAL \
  --execution-challenge "$CHALLENGE" \
  --left-run-id "$DS4_RUN_ID" --right-run-id "$LLAMA_RUN_ID" \
  --right-candidate-exporter-policy-id "$CANDIDATE_EXPORTER_POLICY_ID" \
  --prompt-builder-policy-id "$PROMPT_BUILDER_POLICY_ID" \
  --approval-policy "$APPROVAL_POLICY" \
  --approval-signature "$APPROVAL_SIGNATURE" \
  --approval-principal "$APPROVAL_PRINCIPAL" \
  --report report.json
```

The first mismatch is reported by phase, decode step, exact token, layer, component, byte offset, flat element index, and per-token component element index. All required components use exact byte comparison. There is no tolerance mode. A ds4 bundle is invalid unless it reports revision `bd66c402070042bf0a79ad6ece8242de4c93680c`.

## Corpus matrix

Use only the target model and these repository files:

```text
tests/corpus/correctness-prose.txt
tests/corpus/correctness-code.txt
tests/corpus/correctness-structured.txt
tests/corpus/correctness-numeric.txt
```

Their SHA-256 values are fixed in `trace_format.py`; the matrix refuses modified corpus bytes. The published GGUF must have SHA-256 `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42`.

Start at context 32768. `llama-deepseek-v41-prompt-builder` loads only the GGUF vocabulary, repeats each repository corpus deterministically, truncates the token sequence to `context - decode_steps`, detokenizes it, and requires exact token round-trip before saving the prompt. `run_matrix.py` creates each prompt once before either runtime executes and reuses its exact bytes for every chunk-boundary case. Run prefill plus at least eight greedy decode steps in one reused context.

The initial correctness matrix uses the final admitted physical ubatch of 32. The worst-case routed expert union is `min(384, 6*32) = 192` experts per layer. The published GGUF metadata yields 398131200 bytes per cross-layer expert slot, so the required cache is exactly 76441190400 bytes, or 72900 MiB. The launchers reject other ubatch, slot, or cache-byte values. Host-memory admission still accounts the cache, staging, model, graph, state, outputs, current host use, and safety margin together before allocation.

Run the oracle matrix only from the final integration commit that contains the canonical watchdog, memory admission, and this correctness harness. Supply its exact revision, its immutable merge-base, and a newly computed base-to-head binary diff SHA-256 to `run_matrix.py`; do not reuse the standalone correctness PR head or diff identity.

ROCm on `gfx1151` is the primary acceptance backend:

```sh
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build-dsv41-trace-rocm \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DGGML_HIP=ON \
    -DGPU_TARGETS=gfx1151 \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-dsv41-trace-rocm --config Release -j "$(nproc)" --target \
  llama-deepseek-v41-trace \
  llama-deepseek-v41-prompt-builder \
  test-deepseek41-trace-manifest \
  test-deepseek41-trace-injected \
  test-backend-ops \
  test-deepseek41-schema \
  test-deepseek41-engram \
  test-deepseek41-expert \
  test-deepseek41-memory \
  test-deepseek41-runtime \
  test-deepseek41-trace-host

build-dsv41-trace-rocm/bin/test-backend-ops -b ROCm0 -o MUL_MAT_ID
build-dsv41-trace-rocm/bin/test-backend-ops -b ROCm0 -o MUL_MAT
build-dsv41-trace-rocm/bin/test-backend-ops -b ROCm0 -o GET_ROWS
build-dsv41-trace-rocm/bin/test-backend-ops -b ROCm0 -o SET_ROWS
build-dsv41-trace-rocm/bin/test-backend-ops -b ROCm0 -o CPY
```

Native Linux containment tests are not enabled by default because generic hosted runners do not provide the required mount policy. A dedicated single-purpose runner must set `-DLLAMA_DEEPSEEK_V41_NATIVE_CONTAINMENT_TESTS=ON` and use exactly one administrator-provisioned host configuration: `kernel.apparmor_restrict_unprivileged_userns=0`, or a narrowly scoped AppArmor allow policy for the unchanged receipt-bound helper user namespace, `MS_PRIVATE`, private procfs, and verification sequence. Exact AppArmor policy syntax is host-specific and is not supplied here. The test and production launcher still fail closed when the required namespace operations are unavailable.

Do not change host ROCm packages for this run. Vulkan can provide secondary coverage, but it cannot replace the required ROCm low-level and oracle evidence. The llama runner selects `ROCm0` explicitly, invokes the exact exporter for a pre-allocation device attestation, and rejects the run unless the backend PCI identity maps to exactly one KFD node reporting `gfx1151`. The native exporter repeats the query before model allocation and verifies that the loaded model still uses the same device.

Static repository builds skip this shared-library trace component instead of failing configuration.

Set `HIP_LAUNCH_BLOCKING=1` on the canonical watchdog command that owns the complete Strix matrix process group. The llama.cpp wrapper fails closed if this variable is absent or different, and every embedded Strix memory, swap, and watchdog audit records it. This Linux/ROCm setting is not an Apple Metal oracle requirement.

## Strix candidate execution gate

`run_llama.py` refuses model execution when swap is enabled, the canonical watchdog lease, heartbeat, or JSONL audit is missing or stale, another unrelated matching model workload is active, or any model/prompt/trace path fails the storage gate.

The approved watchdog revision is exactly `02235de637ae1ffe8aeef9479628962792190b22`, with `scripts/strix_memory_watchdog.py` SHA-256 `d2781a25f978dd2bc14fc113079aa2dbf513aa157b44da9d0d51d750daa6c94f`. That revision is an ancestor of this stack, so production validates the script in the exact candidate tree instead of copying it from another revision. Both Python validators and the native exporter reject every other revision or script hash.

The approved watchdog must own the complete matrix process group and expose its canonical validation and process-group lease-guard APIs. The wrappers verify its pinned script identity, Python executable and argv position, PID and Linux start time, exact command bytes, 116/118/120 GiB thresholds, `/proc` source, watchdog/guardian/matrix topology, current process group, child command hash, atomic lease/heartbeat identities, heartbeat freshness and persistent-audit record hash, and the watchdog-held audit lock. The direct matrix payload starts the canonical process-group lease guard before inference. The wrappers repeat validation before and after each runtime.

Use one empty directory on verified non-rotational NVMe for every Strix input and output. The Python launcher and both native tools resolve the nearest existing output parent through `/proc/self/mountinfo`, `/sys/dev/block`, and `/sys/class/block`. They require a resolvable local NVMe block device with `queue/rotational=0`; tmpfs, network filesystems, rotational disks, unknown devices, lexical or resolved `/mnt/bigspace` paths, and forbidden-root symlink escapes fail closed. Btrfs subvolume sources such as `/dev/nvme0n1p3[/home]` are resolved through the parent block device. `TMPDIR` is mandatory and must be an absolute literal pathname to an existing writable non-symlink directory on verified NVMe; there is no `/tmp` fallback. Relative paths and shell shorthand such as `~/tmp` are rejected because environment values are not shell-expanded for the launched process. These metadata commands do not execute the model:

```sh
MODEL=/mnt/models/DeepSeek-V4.1-Flash-Q2.gguf
REPO=/home/<user>/src/strix-llama-integration
RUN_ROOT=/home/<user>/dsv41-correctness
TMPDIR=/home/<user>/tmp/dsv41
export TMPDIR

test "$(stat -c %s "$MODEL")" = 365713686528
test "$(realpath "$MODEL")" = /mnt/models/DeepSeek-V4.1-Flash-Q2.gguf
test "$(realpath -m "$REPO")" = /home/<user>/src/strix-llama-integration
test "$(realpath -m "$RUN_ROOT")" = /home/<user>/dsv41-correctness
test "$(realpath -m "$TMPDIR")" = /home/<user>/tmp/dsv41
mkdir -p "$RUN_ROOT" "$TMPDIR"
findmnt -no SOURCE,FSTYPE,TARGET -T "$MODEL"
findmnt -no SOURCE,FSTYPE,TARGET -T /home
lsblk -d -o NAME,ROTA,TYPE,SIZE,MODEL
df -B1 /mnt/models /home
sha256sum "$MODEL"
PYTHONPATH=gguf-py python3 -m gguf.scripts.gguf_dump "$MODEL" > "$RUN_ROOT/model-metadata.txt"
git -C /home/<user>/src/ds4-v41 status --short
git -C /home/<user>/src/ds4-v41 rev-parse HEAD
test "$(awk 'NR > 1 { count++ } END { print count + 0 }' /proc/swaps)" = 0
```

The expected model digest is `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42`, both selected block devices report `ROTA=0`, and `/proc/swaps` has zero entries. The observed planning snapshot had 76366495744 bytes free on `/mnt/models` and 679635001344 bytes free under `/home`; recheck before every run. Keep the unchanged 365713686528-byte GGUF in place on `/mnt/models`. Do not copy the model or place builds, logs, traces, audit files, or temporary files there. Put all of those under `/home`, and never use `/mnt/bigspace`.

The exporter is intentionally external to the canonical ds4 checkout. It must be built from the pinned revision and emit this trace format without changing the canonical checkout. The launcher requires its trusted SHA-256 and canonical executed path, and rejects a bundle unless the exporter reports the pinned revision and its build path and SHA-256 match the executed file.

The llama.cpp exporter is built as `llama-deepseek-v41-trace`. It accepts the normal model, context, batch, ubatch, KV, Flash Attention, offload, and expert-cache arguments. `-bf` supplies the exact prompt bytes, `-n` is the number of greedy decode steps, and `-o` is the trace directory. It also requires `DSV41_TRACE_MEMORY_AUDIT`, `DSV41_TRACE_SWAP_AUDIT`, and `DSV41_TRACE_WATCHDOG_AUDIT` so every run points to its safety evidence. The content-addressed memory audit binds the preflight accelerator and storage attestations; the manifest binds the independently repeated native accelerator attestation.

Use `run_llama.py` on the validation host instead of calling the exporter directly. It verifies the detached executable approval policy before invoking the exporter, then applies the same zero-swap, watchdog, active-workload, exact-`gfx1151`, and proven-NVMe gates and embeds content-addressed preflight and postflight evidence in the trace. It requires the exact producer revision, immutable base revision, expected base-to-producer binary diff SHA-256, verifier repository path and revision, external executable approval, externally approved signer principal, and matching private signing key. The private key is never copied or logged. Its public half is derived with the fixed trusted `ssh-keygen` executable and must exactly match the source-controlled signer map before model execution.

The launcher checks the approved candidate exporter path, hash, device/inode, size, modification time, and change time before any exporter invocation and after each protected operation. The read-only `--dsv41-attest-build ROCm0` command loads the selected backend and emits the embedded runtime profile and receipt before model execution; the launcher requires exact equality with the external approval and repeats the attestation after trace completion. The native exporter embeds the full 40-character producer revision independently of dynamically loaded build-info, resolves its actual executable path, and validates every loaded `llama`, `ggml`, and enabled backend project library against the build-generated component receipt and selected runtime profile. Component name, filename, canonical path, SHA-256, role, and exact revision-bearing identity must match, and the measured loaded set must equal the predeclared profile set both immediately before protected trace generation and after it completes. Both snapshots and the completed loader-monitor receipt are signed. macOS also monitors loader additions during the protected interval. Missing, duplicated, unclassified, outside-root, renamed inside-root injected, catalogued-but-not-profile, changed, or late-loaded project libraries fail closed. Linux enumerates loaded ELF objects rather than arbitrary memory mappings and accepts non-project ROCm dependencies only from a root-owned, non-writable `/opt/rocm` installation. Production launchers and the native exporter reject dynamic-loader and `GGML_BACKEND_PATH` overrides.

Production installs no manifest-writing or runtime-path probe option. With `LLAMA_BUILD_TESTS`, CMake builds the non-installed `test-deepseek41-trace-manifest` harness from the same native writer implementation. Its input accepts only model, prompt, audit, expected-coverage, and event-count data. Runtime, accelerator, path, configuration, build, and environment evidence comes from measured local state, and CPU or Darwin output is explicitly test-only and cannot satisfy production candidate or signer requirements. The trace install component places the exact receipt libraries beside the tools and sets `@loader_path/../lib` on macOS or `$ORIGIN/../lib` on ELF so installed `--version` needs no loader override. The install test also hashes every installed component and requires exact equality with the receipt embedded after the original library link; install-time rewriting or re-signing fails.

`run_matrix.py --llama-only` runs the approved prompt builder on the Linux candidate host, copies the four repository corpora byte-for-byte into the NVMe result directory, verifies their fixed hashes, and verifies the approved builder path/hash, loaded runtime closure, complete tokenizer policy, and both original and copied corpus identities before prompt construction. It rechecks the builder, loaded libraries, and corpus identities after execution and requires the generated prompt hash, byte count, tokenizer policy, context, and decode configuration to equal the signed external approval before writing provenance. Pass the detached approval policy and signature, selected candidate and prompt approval IDs, both executables, the producer revision/base/diff identity, and the verifier checkout. Its default context matrix is 32768. Pass later contexts only after the 32K target passes. The Apple oracle is captured separately with `run_ds4.py`; compare completed per-case bundles with `trace_format.py compare`.

## Apple Metal oracle execution gate

The external ds4 exporter is not present in the pinned canonical checkout. Production remains blocked because the source-controlled `APPROVED_DS4_EXPORTERS` map is empty. A run requires a detached, externally signed executable-approval v2 policy with a selected `ds4_exporters` record. The record binds the `ds4` runtime role, `antirez/ds4` repository, pinned producer revision, distinct verifier revision, canonical install root and executable path, trusted owner, exact executable SHA-256, runtime profile, and exact dependency receipt. `run_ds4.py` verifies that policy before any exporter execution, requires a canonical non-symlink install hierarchy that is trusted-owned and not writable by the execution identity or an untrusted group, and requires the exporter and every dependency to be regular non-writable one-link trusted-owned files. Linux launches the retained exporter descriptor through `/proc/self/fd`. macOS launches the canonical path only after proving the immutable root and retains all measured descriptors through the launch; it does not claim descriptor-selected dyld loading. Test policies are explicit and cannot satisfy production defaults.

Linux production execution must use an administrator-provisioned dedicated service identity whose live supplementary-group list is empty. The service may select one trusted primary group for required device or file access, or use exact administrator-managed ACLs, but it must not add the execution identity to `render`, `video`, or other supplementary groups. A normal login session is not valid, and an unprivileged process cannot repair a nonempty list after launch. Python checks the live list before it opens helper descriptors, and the helper checks `getgroups(0, nullptr) == 0` before it sends `READY` or creates a namespace, then rechecks the inherited empty list after both user-namespace mapping boundaries and after its credential drop. The signed containment-helper receipt binds `zero-supplementary-groups-v1` and an exact empty group list. Administrators must run `llama-deepseek-v41-containment-helper --check-launcher-groups` as `ExecStartPre=` under the same `User=` and `Group=` as the matrix service; success prints `supplementary-groups=0`. `SupplementaryGroups=` can add groups and does not replace this runtime check. Any nonzero list or query failure exits 125 before target code.

Every exporter invocation repeats the install-root, executable, containment-helper, and dependency identity checks immediately before and after execution. Attestation calls have a fixed 60-second timeout and trace generation has a fixed 24-hour timeout, both bound by the signed runner script revision. Linux production execution requires a single-threaded supervisor and the receipt-bound native helper; Python does not use `fork()` at the containment boundary. Python launches the helper in a new session with `posix_spawn()` while all catchable signals are blocked and reset to default, obtains the helper pidfd before protocol release, and never uses numeric PID or process-group signaling. The helper verifies its inherited signal state, binds its lifetime to the exact Python parent with `PR_SET_PDEATHSIG`, and reports readiness before creating any target process. After `PREPARE`, `clone3(CLONE_NEWUSER | CLONE_NEWPID | CLONE_NEWNS | CLONE_PIDFD)` atomically creates a blocked namespace init and stable pidfd. The helper verifies the namespace-init parent binding, pins its `/proc` identity, writes one-entry UID and GID maps for the trusted init identity, creates a private session, makes PID 1 non-dumpable, makes mount propagation private, accepts only `EINVAL` when a locked inherited `/proc` cannot be detached, mounts a new `nosuid,nodev,noexec` procfs over `/proc`, and verifies `/proc/self` resolves to namespace PID 1. A separate bounded binary diagnostic pipe reports setup stage and errno without sharing protocol tokens or exposing host data. The helper sends the namespace pidfd to Python only after that setup is acknowledged, and target execution remains blocked until Python verifies both stable identities and sends `EXEC`. PID 1 then uses `clone3(CLONE_NEWUSER | CLONE_PIDFD)` to create a target bootstrap in a nested user namespace, maps only namespace UID and GID 65534 to the trusted parent namespace identity, and requires the target to create a separate session. Before `execve()`, the bootstrap verifies that the inherited supplementary-group list is still empty, locks securebits against root or set-ID capability regeneration, changes all real, effective, and saved credentials to 65534, drops every bounding, effective, permitted, inheritable, and ambient capability, sets `PR_SET_NO_NEW_PRIVS`, and verifies each state. It then installs an inherited seccomp user-notification filter for ptrace, process-vm access, PID 1 or process-group signaling, lifecycle `prctl`, credential or capability changes, session or process-group changes, and namespace entry. PID 1 owns the only listener, validates the bootstrap denial probes, and treats any later notification from any target descendant as fatal to the complete PID namespace. PID 1 re-verifies its parent-death signal, non-dumpable state, isolated session, and empty supplementary-group list before it confirms target isolation to the helper. The namespace init executes the target only after this handshake, then kills and reaps every remaining namespace member. If Python dies, the kernel kills the helper; helper death kills namespace PID 1; namespace PID 1 death kills the complete target namespace. If the dedicated zero-group service contract, unprivileged user, PID, or mount namespaces, nested target-user mapping, clone3, pidfds, credential or capability drops, securebits, no-new-privileges, seccomp notification, parent-death binding, private sessions, non-dumpable PID 1, private procfs, protocol identity, or helper receipt verification are unavailable, execution fails before target code. The helper sends `COMPLETE` only after the namespace is empty, reaped, and its pidfd is closed. If cleanup cannot prove that state, Python retains the stable containment authority and lock and poisons the supervisor against reuse. Other POSIX platforms fail closed because a process group is not a containment boundary. The Darwin process-group path is an explicit test-only fixture and is never production-valid. Windows creates the process suspended, records assignment and resume state, assigns it to a kill-on-close Job Object before resume, and directly terminates and reaps the exact suspended child if assignment fails. The `subprocess.Handle` wrapper retains sole ownership of the process handle and closes it exactly once; raw `CloseHandle` is not used for that wrapper. A structured completion result gates every post-attestation, and any containment acquisition, protocol, termination, assignment, reaping, empty-set proof, helper completion, or executable, runtime, helper, stream, pidfd, Job, process, or thread descriptor teardown failure keeps that result false. Captured output remains bytes until process-tree cleanup and all lower identity checks finish; DS4 strict UTF-8 decoding is then captured as the operation primary before post-build attestation. A non-attestation invocation always runs a post-invocation build attestation before its result or exception is honored when containment completion is proven; if execution, cleanup, postchecks, decoding, or post-attestation fail together, the launcher preserves the primary error and every typed secondary failure. The exporter must answer `--dsv41-attest-build` without loading the model and report its exact path, SHA-256, runtime profile, dependency receipt digest, and pre/post loaded-library closure. The launcher measures that evidence before and after device attestation, trace generation, and postflight device attestation. The signed authorization, oracle evidence, runner audits, and Seal v1 manifest bind the DS4 approval ID and hash, install-trust hash, canonical executable identity, runtime profile and receipt, loaded-library closure, and producer/verifier revisions. Missing or mismatched policy, mutable or aliased paths, hard links, component substitution, and changed build evidence fail closed. `run_ds4.py` also requires macOS arm64, at least 128 GiB of measured host memory, zero swap, no unrelated matching workload, an exact selected Metal device query, and an existing writable non-symlink `TMPDIR`. The model, prompt, output, harness repository, ds4 checkout, temporary directory, Python executable, runner script, and exporter must resolve through `df -P` to a volume that `diskutil info -plist` proves is internal solid-state storage backed by NVMe or Apple Fabric. SATA, network, virtual, disk-image, external/non-internal, non-solid-state, and incomplete device identities fail closed, as do lexical or resolved forbidden paths.

The ds4 memory audit binds the exact Metal accelerator, host model/OS/memory identity, and every storage record. The runner audit binds the Python runner process, UID, executable/script paths and hashes, exporter path/hash, external approval and install-trust hashes, runtime build/profile/receipt identity, producer and verifier revisions, pinned checkout path/revision, and exact command hash. The runner script must be inside the attested harness repository. The preflight and postflight accelerator and host identities must remain unchanged. These Apple audits replace Linux KFD, `/proc`, HIP, and Strix watchdog claims; the oracle must never fabricate those fields.

After the exporter and its complete runtime receipt are independently reviewed on an authorized 128 GiB or larger Apple oracle host, add the externally reviewed record to a signed executable-approval v2 policy and select it with `--ds4-exporter-policy-id`. Do not add test policy material to the production maps. A caller-provided digest or self-reported build record alone is not sufficient oracle provenance. The exporter must answer `--dsv41-attest-build` and `--dsv41-attest-device Metal0` without loading the model. The device query emits the strict `apple-metal` attestation. Its trace command interface is:

The unpublished `ds4gguf` documentation revision `e13893ffcb33e90c8852929303e188102df7a8f5` is provenance only. It is not an executable dependency, exporter approval, or fixture source. Executable tests and fixtures stay in this `strix-llama.cpp` stack.

```text
--model PATH --prompt-file PATH --output PATH --context N --decode-steps N --prefill-chunk 32 --device Metal0
```

It must emit a complete valid `dsv41-trace` bundle, report ds4 revision `bd66c402070042bf0a79ad6ece8242de4c93680c`, put its own executable SHA-256 in `manifest.json`, and report the same selected Metal device before and after execution. `run_ds4.py` embeds the platform-native memory, swap, runner, accelerator, host, storage, and exact-path evidence.

Capture the first llama.cpp matrix under the watchdog:

```sh
REPO=/home/<user>/src/strix-llama-integration
MODEL=/mnt/models/DeepSeek-V4.1-Flash-Q2.gguf
RUN_ROOT=/home/<user>/dsv41-correctness
CASE_ROOT="$RUN_ROOT/c32768"
CANDIDATE_REV="$(git -C "$REPO" rev-parse HEAD)"
BASE_REV=<full-immutable-oracle-revision>
DIFF_SHA256="$(git -C "$REPO" diff --binary --no-ext-diff "$BASE_REV" "$CANDIDATE_REV" -- | sha256sum | awk '{print $1}')"
CHALLENGE=<64-lowercase-hex-execution-challenge>
AUTH_ISSUED=<unix-seconds>
AUTH_EXPIRES=<unix-seconds-no-more-than-24h-after-issued>
APPROVAL_POLICY=<absolute-canonical-signed-policy-path>
APPROVAL_SIGNATURE=<absolute-canonical-detached-signature-path>
APPROVAL_PRINCIPAL=<source-approved-executable-approver>
CANDIDATE_EXPORTER_POLICY_ID=<approved-candidate-record-id>
PROMPT_BUILDER_POLICY_ID=<approved-prompt-builder-record-id>

mkdir -p "$CASE_ROOT/watchdog"
cd "$REPO"
HIP_LAUNCH_BLOCKING=1 python3 scripts/strix_memory_watchdog.py \
  --procfs-root /proc \
  --soft-gib 116 \
  --emergency-gib 118 \
  --grace-seconds 30 \
  --sample-interval-seconds 1 \
  --lease-path "$CASE_ROOT/watchdog/lease.json" \
  --heartbeat-path "$CASE_ROOT/watchdog/heartbeat.json" \
  --audit-path "$CASE_ROOT/watchdog/audit.jsonl" \
  --heartbeat-max-age-seconds 5 \
  -- \
  python3 tools/deepseek-v41-trace/run_matrix.py \
    --repo "$REPO" \
    --model "$MODEL" \
    --output "$CASE_ROOT/matrix" \
    --llama-runner "$REPO/tools/deepseek-v41-trace/run_llama.py" \
    --llama-exporter "$REPO/build-dsv41-trace-rocm/bin/llama-deepseek-v41-trace" \
    --llama-prompt-builder "$REPO/build-dsv41-trace-rocm/bin/llama-deepseek-v41-prompt-builder" \
    --candidate-revision "$CANDIDATE_REV" \
    --base-revision "$BASE_REV" \
    --candidate-diff-sha256 "$DIFF_SHA256" \
    --candidate-exporter-policy-id "$CANDIDATE_EXPORTER_POLICY_ID" \
    --prompt-builder-policy-id "$PROMPT_BUILDER_POLICY_ID" \
    --approval-policy "$APPROVAL_POLICY" \
    --approval-signature "$APPROVAL_SIGNATURE" \
    --approval-principal "$APPROVAL_PRINCIPAL" \
    --signer-principal "$LLAMA_SIGNER_PRINCIPAL" \
    --signing-key "$LLAMA_SIGNING_KEY" \
    --execution-challenge "$CHALLENGE" \
    --run-id-prefix "strix-llama-c32768" \
    --authorization-issued-unix "$AUTH_ISSUED" \
    --authorization-expires-unix "$AUTH_EXPIRES" \
    --llama-only \
    --contexts 32768 \
    --ubatches 32 \
    --batch 2048 \
    --device ROCm0 \
    --expert-cache-slots 192 \
    --expert-cache-mib 72900
```

Use a new empty output and watchdog directory for each later context. Repeat the same command after setting `CASE_ROOT` and changing `--contexts`:

```sh
CASE_ROOT="$RUN_ROOT/c65536"  # then use --contexts 65536
CASE_ROOT="$RUN_ROOT/c98304"  # then use --contexts 98304
CASE_ROOT="$RUN_ROOT/c131072" # then use --contexts 131072
```

On the separately authorized Apple oracle, place the unchanged GGUF, exact prompt, prompt-provenance record, harness checkout, pinned ds4 checkout, exporter, trace output, and `TMPDIR` on internal solid-state storage. Then run one case at a time:

```sh
export TMPDIR=/Users/oracle/dsv41/tmp
python3 tools/deepseek-v41-trace/run_ds4.py \
  --repo /Users/oracle/src/strix-llama.cpp \
  --checkout /Users/oracle/src/ds4-v41 \
  --exporter /opt/dsv41/ds4/bin/dsv41-trace-exporter \
  --exporter-sha256 <approved-exact-sha256> \
  --model /Users/oracle/models/DeepSeek-V4.1-Flash-Q2.gguf \
  --prompt /Users/oracle/dsv41/inputs/correctness-prose-c32768.txt \
  --prompt-provenance /Users/oracle/dsv41/inputs/correctness-prose-c32768.txt.provenance.json \
  --output /Users/oracle/dsv41/traces/correctness-prose-c32768-ub32 \
  --approval-policy "$APPROVAL_POLICY" \
  --approval-signature "$APPROVAL_POLICY_SIGNATURE" \
  --approval-principal "$APPROVAL_APPROVER_PRINCIPAL" \
  --ds4-exporter-policy-id "$DS4_EXPORTER_POLICY_ID" \
  --prompt-builder-policy-id "$PROMPT_BUILDER_POLICY_ID" \
  --corpus-name correctness-prose.txt \
  --corpus-sha256 2da590a37e3297767336c10b024a0de732d64bee4da5792596f8ddf49ea408d2 \
  --context 32768 \
  --decode-steps 8 \
  --prefill-chunk 32 \
  --device Metal0 \
  --signer-principal "$DS4_SIGNER_PRINCIPAL" \
  --signing-key "$DS4_SIGNING_KEY" \
  --execution-challenge "$CHALLENGE" \
  --run-id "apple-ds4-prose-c32768-ub32" \
  --authorization-issued-unix "$AUTH_ISSUED" \
  --authorization-expires-unix "$AUTH_EXPIRES"
```

Compare the completed bundle with the matching Strix bundle using `trace_format.py compare`. Repeat for all four corpora before expanding the context matrix.

## Strix bring-up evidence lane

The 128 GiB or larger Apple Metal run remains an external verification gate for the pinned ds4 cross-runtime oracle. A Strix bring-up can complete first, but it must report `BRINGUP PASS`, never `TARGET PASS`. It does not replace complete byte-identical ds4 logits.

The pinned `antirez/ds4@bd66c402070042bf0a79ad6ece8242de4c93680c` evidence anchors are:

| Evidence | SHA-256 | Limitation |
|---|---|---|
| `tests/test-vectors/README.md` | `0e59b2f2832bed8af0a91e6ff20962debf964cd2d1d141c086e52cfcc995a1c3` | Official-vector provenance and limitations |
| `tests/test-vectors/flash-0731/manifest.json` | `ebf237a5660a6851fb8085e77f532901a9d758208b25d7ed5af0b7af4b28f91b` | Official API provenance |
| `tests/test-vectors/flash-0731/official.vec` | `77ae699889bfaf1348768dcbe7ea2c72279ae86abb10470d3e1b08cd1fd82a83` | Official selected-token/top-logprob slice, not full logits |
| `tests/test-vectors/flash-0731/local-golden.vec` | `23d942ff3b9bb2a3f82927d11aa3ed1461e1f302071e788d0d95a5c165e47d3b` | Local tolerant top-64 drift anchor, not exact |
| `tests/test_engram.c` | `198a561d981f62518a9d28035480a7e220b99c156cde6d248b8baabd684cc74b` | Model-free Engram oracle |
| `ds4_engram.c` | `2b6ca468510ebf45ee298a905525bc7234dacad9a384bf2011eba19ba2c0bdf7` | Engram implementation under test |
| `ds4_engram.h` | `f84a264e0fe199d23a6f0c56fbbd19e222adc7009eed13c185af6403f68f7f5c` | Engram schema |
| `tests/test_deepseek41_metal.c` | `9197c2f9d65b380ce25be5334991e4bfaf40e82e6708e64f2b9329552412d28c` | Synthetic exact candidate/top-k oracle; source anchor only on Strix |
| `tests/test_deepseek41_graph.c` | `6dc786f831c93ae7f5aa56e7518f67657125f7c0fd35cead3c646eff3d3f9e09` | Synthetic routing and graph oracle |
| `tests/test_deepseek41_prefill.c` | `452774b9332d393822d84288eeb2d71ca1fb25f1ba30d20a49160d1787de734f` | Prefill boundary fixture |
| `tests/test_deepseek41_manifest.py` | `2d7aa1fc93805d9c97839c5de9eccc856628f13e6913c3bbc3f0c99b0827aeae` | Model manifest/schema oracle |
| `tests/test_deepseek41_conversion.py` | `a40b83062a9b91338773addd77fe62650296f4de607057cb4fd1329287f347b5c` | Conversion/schema fixture |
| `tests/test_deepseek41_gguf.c` | `8f41e049d5ec179c38a1a00ac61712db0306902f0ef74bcdb989d8993316ff35` | GGUF schema oracle |
| `gguf-tools/deepseek41_metadata.py` | `39300bbd504165b97de017edd72563377f50b8a5511ca7478252a86f31a0009b` | Conversion metadata source |
| `ds4.c` | `1776dbfed177ea14f3ce6cac1d8d0b1c1b44dfff2c2663769a9a5634aeec34e7` | Runtime schema source |

Verify these files from the pinned checkout before using their results. Do not copy an unpublished exporter or depend on a private ds4 remote.

```sh
python3 tools/deepseek-v41-trace/verify_ds4_anchors.py \
  --checkout /home/<user>/src/ds4-v41
```

`run_matrix.py --llama-only` captures all four repository corpora without claiming cross-runtime success. Run the final integration build twice with separate output directories, then run an equivalently instrumented immutable base build once. Keep every run under its own canonical watchdog invocation and use the exact ubatch/cache/ROCm arguments above.

For each case (`correctness-prose-c32768-ub32`, `correctness-code-c32768-ub32`, `correctness-structured-c32768-ub32`, and `correctness-numeric-c32768-ub32`), require both comparisons:

```sh
python3 tools/deepseek-v41-trace/trace_format.py compare-local self-consistency \
  "$RUN_A/llama/$CASE" "$RUN_B/llama/$CASE" \
  --left-signer-principal "$SIGNER_PRINCIPAL" \
  --right-signer-principal "$SIGNER_PRINCIPAL" \
  --execution-challenge "$CHALLENGE" \
  --left-run-id "$RUN_A_ID" --right-run-id "$RUN_B_ID" \
  --report "$REPORTS/$CASE-self.json"

python3 tools/deepseek-v41-trace/trace_format.py compare-local base-regression \
  "$BASE_RUN/llama/$CASE" "$RUN_A/llama/$CASE" \
  --left-signer-principal "$SIGNER_PRINCIPAL" \
  --right-signer-principal "$SIGNER_PRINCIPAL" \
  --execution-challenge "$CHALLENGE" \
  --left-run-id "$BASE_RUN_ID" --right-run-id "$RUN_A_ID" \
  --report "$REPORTS/$CASE-base.json"
```

Both commands compare every exact trace component, including complete prefill/decode logits, tokens, Engram rows, original expert IDs and weights, raw/compressed attention sources, and candidate propagation. `base-regression` also requires the base trace revision to equal the integrated trace's attested oracle revision. The report includes `cross_runtime_status: "INCOMPLETE"` even when it returns `BRINGUP PASS`.

The pinned ds4 evidence commands are:

```sh
git -C /home/<user>/src/ds4-v41 diff --quiet
git -C /home/<user>/src/ds4-v41 diff --cached --quiet
test "$(git -C /home/<user>/src/ds4-v41 rev-parse HEAD)" = bd66c402070042bf0a79ad6ece8242de4c93680c
/home/<user>/src/ds4-v41/tests/test_engram
DS4_TEST_MODEL="$MODEL" \
DS4_TEST_VECTOR_FILE=/home/<user>/src/ds4-v41/tests/test-vectors/flash-0731/official.vec \
  /home/<user>/src/ds4-v41/ds4_test --logprob-vectors
```

Capture the exact commands, executable hashes, stdout/stderr hashes, exit status, and watchdog artifacts. The official-vector result and pinned fixtures are supporting bring-up evidence only. The first `TARGET PASS` still requires the external pinned ds4 exporter and the full cross-runtime trace comparison.
