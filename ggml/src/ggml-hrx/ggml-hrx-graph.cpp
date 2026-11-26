#include "ggml-hrx-graph.h"

#include <cinttypes>
#include <cstdlib>
#include <numeric>

// libhrx 0.2.0 (libhrx/src/libhrx/graph_analysis.c) cuts the creation order into partitions of 2048 nodes, one
// command buffer each. In a partition it puts a barrier before a node only if a dependency comes after the last
// barrier. A node at partition offset 16 or more with no dependency in its partition starts a new workstream: one
// more command buffer with its own system-scope entry and exit barriers.
static constexpr int GGML_HRX_GRAPH_PARTITION         = 2048;
static constexpr int GGML_HRX_GRAPH_MIN_STREAM_OFFSET = 16;

struct ggml_hrx_graph_rec {
    struct dispatch {
        hrx_executable_t              executable     = nullptr;
        uint32_t                      export_ordinal = 0;
        hrx_dispatch_config_t         config         = {};
        std::vector<uint8_t>          constants;
        std::vector<hrx_buffer_ref_t> refs;
        std::vector<int32_t>          deps;
    };
    // A byte range and its accesses since its last write.
    struct span {
        size_t               end    = 0;
        int32_t              writer = -1;
        std::vector<int32_t> readers;
    };
    using span_map = std::map<size_t, span>; // by first byte

    std::vector<dispatch>          dispatches;
    std::map<hrx_buffer_t, span_map> spans;
    bool                           write_all = false;
};

ggml_hrx_graph_rec * ggml_hrx_graph_rec_init() {
    return new ggml_hrx_graph_rec;
}

void ggml_hrx_graph_rec_free(ggml_hrx_graph_rec * rec) {
    delete rec;
}

size_t ggml_hrx_graph_rec_size(const ggml_hrx_graph_rec * rec) {
    return rec->dispatches.size();
}

void ggml_hrx_graph_rec_write_all(ggml_hrx_graph_rec * rec, bool on) {
    if (rec != nullptr) {
        rec->write_all = on;
    }
}

// Splits the span that contains pos, so that a span starts at pos.
static void ggml_hrx_span_split(ggml_hrx_graph_rec::span_map & m, size_t pos) {
    auto it = m.upper_bound(pos);
    if (it == m.begin()) {
        return;
    }
    --it;
    if (it->first < pos && pos < it->second.end) {
        ggml_hrx_graph_rec::span right = it->second;
        it->second.end = pos;
        m.emplace(pos, std::move(right));
    }
}

// First span that ends after pos.
static ggml_hrx_graph_rec::span_map::iterator ggml_hrx_span_find(ggml_hrx_graph_rec::span_map & m, size_t pos) {
    auto it = m.upper_bound(pos);
    if (it != m.begin() && std::prev(it)->second.end > pos) {
        --it;
    }
    return it;
}

void ggml_hrx_graph_rec_dispatch(ggml_hrx_graph_rec * rec, const ggml_hrx_kernel * kernel, const hrx_dispatch_config_t & config,
                                 const std::vector<uint8_t> & constants, const std::vector<hrx_buffer_ref_t> & refs,
                                 const std::vector<bool> & written) {
    const int32_t id = (int32_t) rec->dispatches.size();
    ggml_hrx_graph_rec::dispatch d;
    d.executable     = kernel->executable;
    d.export_ordinal = kernel->export_ordinal;
    d.config         = config;
    d.constants      = constants;
    d.refs           = refs;

    // Collect the dependencies before the spans change.
    for (size_t k = 0; k < refs.size(); ++k) {
        if (refs[k].length == 0) {
            continue;
        }
        const size_t a = refs[k].offset;
        const size_t b = a + refs[k].length;
        const bool   w = rec->write_all || written[k];
        auto &       m = rec->spans[refs[k].buffer];
        for (auto it = ggml_hrx_span_find(m, a); it != m.end() && it->first < b; ++it) {
            if (it->second.writer >= 0) {
                d.deps.push_back(it->second.writer);
            }
            if (w) {
                d.deps.insert(d.deps.end(), it->second.readers.begin(), it->second.readers.end());
            }
        }
    }
    std::sort(d.deps.begin(), d.deps.end());
    d.deps.erase(std::unique(d.deps.begin(), d.deps.end()), d.deps.end());

    // Reads first: a write of the same bytes by this dispatch then replaces them.
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t k = 0; k < refs.size(); ++k) {
            const bool w = rec->write_all || written[k];
            if (w != (pass == 1) || refs[k].length == 0) {
                continue;
            }
            const size_t a = refs[k].offset;
            const size_t b = a + refs[k].length;
            auto &       m = rec->spans[refs[k].buffer];
            ggml_hrx_span_split(m, a);
            ggml_hrx_span_split(m, b);
            auto it = m.lower_bound(a);
            if (w) {
                while (it != m.end() && it->first < b) {
                    it = m.erase(it);
                }
                ggml_hrx_graph_rec::span s;
                s.end    = b;
                s.writer = id;
                m.emplace(a, std::move(s));
                continue;
            }
            size_t pos = a;
            while (pos < b) {
                if (it == m.end() || it->first >= b || it->first > pos) {
                    // A gap: no access yet.
                    const size_t end = (it == m.end() || it->first >= b) ? b : it->first;
                    ggml_hrx_graph_rec::span s;
                    s.end = end;
                    s.readers.push_back(id);
                    m.emplace(pos, std::move(s));
                    pos = end;
                    continue;
                }
                if (it->second.readers.empty() || it->second.readers.back() != id) {
                    it->second.readers.push_back(id);
                }
                pos = it->second.end;
                ++it;
            }
        }
    }
    rec->dispatches.push_back(std::move(d));
}

bool ggml_hrx_graph_rec_build(ggml_hrx_graph_rec * rec, hrx_graph_t graph) {
    static const bool linear = getenv("GGML_HRX_GRAPH_LINEAR") != nullptr;
    const int32_t     n      = (int32_t) rec->dispatches.size();

    // Order the dispatches by dependency depth. Then one barrier separates each depth from the next.
    // Decode of a 4065-dispatch graph on gfx1151: 4061 -> 2521 barriers; device time 29.9 -> 27.1 ms at depth 0
    // and 38.3 -> 34.8 ms at depth 8192 (replays of the two recordings alternated in one process).
    std::vector<int32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    if (!linear) {
        std::vector<int32_t> level(n, 0);
        for (int32_t i = 0; i < n; ++i) {
            for (int32_t id : rec->dispatches[i].deps) {
                level[i] = std::max(level[i], level[id] + 1);
            }
        }
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return level[a] < level[b]; });
    }
    std::vector<int32_t> pos(n);
    for (int32_t p = 0; p < n; ++p) {
        pos[order[p]] = p;
    }

    std::vector<hrx_graph_node_t> nodes(n, nullptr);
    std::vector<hrx_graph_node_t> deps;
    int32_t last_barrier = 0;
    int64_t n_edges = 0, n_barriers = 0, n_anchors = 0;
    for (int32_t p = 0; p < n; ++p) {
        const ggml_hrx_graph_rec::dispatch & d = rec->dispatches[order[p]];
        const int32_t start = p - p % GGML_HRX_GRAPH_PARTITION;
        if (p == start) {
            last_barrier = start;
        }
        deps.clear();
        int32_t last_dep = -1;
        if (linear) {
            if (p > 0) {
                deps.push_back(nodes[p - 1]);
                last_dep = p - 1;
            }
        } else {
            for (int32_t id : d.deps) {
                deps.push_back(nodes[pos[id]]);
                last_dep = std::max(last_dep, pos[id]);
            }
            // Keep one workstream per partition: tie the node to the first node of its partition.
            if (last_dep < start && p - start >= GGML_HRX_GRAPH_MIN_STREAM_OFFSET) {
                deps.push_back(nodes[start]);
                last_dep = start;
                n_anchors++;
            }
        }
        // libhrx puts a barrier before a node if a dependency comes after the last barrier.
        if (last_dep >= last_barrier) {
            last_barrier = p;
            n_barriers++;
        }
        n_edges += (int64_t) deps.size();

        hrx_graph_kernel_node_attrs_t attrs = {};
        attrs.executable     = d.executable;
        attrs.export_ordinal = d.export_ordinal;
        attrs.config         = d.config;
        attrs.constants      = d.constants.data();
        attrs.constants_size = d.constants.size();
        attrs.bindings       = d.refs.data();
        attrs.binding_count  = d.refs.size();
        if (!GGML_HRX_CHECK(hrx_graph_add_kernel_node(graph, deps.data(), deps.size(), &attrs, &nodes[p]))) {
            return false;
        }
    }
    GGML_LOG_DEBUG("%s: %d dispatches, %" PRId64 " edges, %" PRId64 " barriers, %" PRId64 " anchors%s\n", __func__, n, n_edges,
                   n_barriers, n_anchors, linear ? " (linear)" : "");
    return true;
}
