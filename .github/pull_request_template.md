## Overview

<!-- Describe what this PR does and why. Be concise but complete -->

## Measurements

<!--
Required for anything that can move throughput, latency, memory use, or output. Delete this section only for docs,
comments, CI or build changes that cannot affect runtime - and say that is why.
Full rules: CONTRIBUTING.md#benchmarking-requirements
-->

```
Device:     Ryzen AI Max+ 395 / <mini-PC or laptop model>
Memory:     128 GB LPDDR5X-8000
Power:      <sustained TDP / platform profile / power governor>
BIOS:       UMA split <n> GB
Kernel:     <uname -r>, amdgpu params: <gttsize / ttm.pages_limit, or none>
Backend:    Vulkan RADV, Mesa <version>   (or: ROCm <version>, HIP_LAUNCH_BLOCKING=<0|1>)
Build:      <cmake flags>
Baseline:   <merge-base commit sha, built and run in this same session>
Change:     <your branch commit sha>
Model:      <HF repo / file>, <quant>
```

<!-- Paste the raw llama-bench tables for baseline and change, with their standard-deviation columns. -->

**Baseline:**

**After:**

<!-- Output-equivalence evidence: test-backend-ops for the backend you touched, and/or llama-perplexity before/after. -->

**Correctness:**

## Additional information

<!-- You can provide more details and link related discussions here. Delete this section if not applicable -->

## Requirements

<!-- IMPORTANT: Please do NOT delete this section, otherwise your PR may be rejected -->

- I have read and agree with the [contributing guidelines](CONTRIBUTING.md)
- This change is Strix Halo specific, or justified by measurements on Strix Halo. General llama.cpp improvements
  belong in [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp) instead
- AI usage disclosure: <!-- mention: NO / ASSISTED / AGENT-AUTHORED - and describe how AI was used -->
- What was NOT verified: <!-- untested backends, unrun tests, hardware you do not have. An honest gap is fine; a silent one is not -->

<!--
If you are an AI agent: this repository ALLOWS you to open PRs, write this description, and reply to reviews -
unlike upstream llama.cpp. Read AGENTS.md for the rules that apply instead. Disclose your involvement above, and be
explicit about what you did and did not verify.
-->
