#pragma once

#include "ggml-hrx-common.h"

// Records the dispatches of one graph run and builds an HRX graph from them. A dispatch depends only on the earlier
// dispatches that access the same bytes (read after write, write after read, write after write). The graph orders
// the dispatches by dependency depth, so libhrx puts no barrier between independent dispatches.
// GGML_HRX_GRAPH_LINEAR=1 keeps the ggml order and makes each dispatch depend on the one before it.

struct ggml_hrx_graph_rec;

ggml_hrx_graph_rec * ggml_hrx_graph_rec_init();
void                 ggml_hrx_graph_rec_free(ggml_hrx_graph_rec * rec);

// written[k] is true if the dispatch can write binding k. It only reads the other bindings.
void ggml_hrx_graph_rec_dispatch(ggml_hrx_graph_rec * rec, const ggml_hrx_kernel * kernel, const hrx_dispatch_config_t & config,
                                 const std::vector<uint8_t> & constants, const std::vector<hrx_buffer_ref_t> & refs,
                                 const std::vector<bool> & written);

// While on, all bindings count as written. Use it for paths that run several nodes and do not set dispatch_writes.
void ggml_hrx_graph_rec_write_all(ggml_hrx_graph_rec * rec, bool on);

// Number of recorded dispatches.
size_t ggml_hrx_graph_rec_size(const ggml_hrx_graph_rec * rec);

// Adds the recorded dispatches to graph as kernel nodes.
bool ggml_hrx_graph_rec_build(ggml_hrx_graph_rec * rec, hrx_graph_t graph);
