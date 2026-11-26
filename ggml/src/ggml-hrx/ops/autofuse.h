#pragma once

#include "ggml-hrx-common.h"

// Automatic fusion of element-wise and row reduction nodes. For each graph run, a plan groups contiguous node ranges
// into regions and builds one Loom kernel for each region.

struct ggml_hrx_autofuse;

// Makes the plan for cgraph. Call it before the nodes run.
void ggml_hrx_autofuse_plan(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph);

// If a planned region starts at node i, runs it and returns the index of its last node. Else returns -1.
// label is the region label for the op profiler. A failed dispatch is fatal (GGML_ABORT).
int ggml_hrx_autofuse_run(ggml_backend_hrx_context * ctx, int i, const char ** label);

void ggml_hrx_autofuse_free(ggml_hrx_autofuse * af);
