#include "ops/autofuse.h"
#include "ggml-hrx-ops.h"

#include <array>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

// A region is a contiguous node range. All its compute nodes are F32 element-wise ops or row reductions over one row
// domain [R, D1, D2, D3]. Each node value is row-shaped [R, D1, D2, D3] or a row scalar [1, D1, D2, D3]. Tensors from
// outside the region (bindings) can have any strides and broadcast to the domain. Only region nodes that something
// outside the region reads are stored.
// A node can read a view of a region value: the value is computed again at the view indices (af_inst). If the view
// has another shape, the region moves to that shape and the earlier nodes become virtual (computed only through views).
// Copies of region values to other layouts and CONCATs of region values are extra stores (AF_NOP nodes). Independent
// neighbour regions can share one dispatch (af_part, af_flush).

// Part of the kernel signature.
enum af_op : uint8_t {
    AF_ADD, AF_SUB, AF_MUL, AF_DIV,
    AF_SCALE, AF_CLAMP, AF_FILL,
    AF_SQR, AF_SQRT, AF_LOG, AF_SIGMOID, AF_TANH, AF_EXP, AF_NEG, AF_ABS, AF_SILU, AF_RELU, AF_GELU, AF_SGN, AF_STEP,
    AF_COPY, // CPY, CONT, DUP and REPEAT of a binding, GET_ROWS
    AF_CONCAT,
    AF_SUM_ROWS, AF_RMS_NORM, AF_SOFT_MAX,
    AF_SSM_CONV, // sum over the taps k of a[k] * b[k]. dim: tap count
    AF_NOP,      // stores only: copies of region values to other layouts, CONCAT of region values
};

static constexpr int AF_MAX_NODES    = 96;
static constexpr int AF_MAX_BINDINGS = 32;
static constexpr int AF_MAX_OUTPUTS  = 16;
static constexpr int AF_MAX_TAPS     = 8;
// Rows from this length without RMS_NORM use 1024 lanes, as the soft_max kernel (ops/softmax.cpp). ADD + SOFT_MAX of
// [200058, 1] (sampler graph, gfx1151) took 624 us with 256 lanes.
static constexpr int64_t AF_WIDE_ROW = 16384;
// Rows kernels with at most this many workgroups keep the chunk values in registers (af_codegen::emit_rows_unrolled).
static constexpr int64_t AF_UNROLL_MAX_GROUPS = 40;
// Same limit as the direct ssm_conv kernel (ops/ssm-conv.cpp). Longer rows read x badly in this layout.
static constexpr int64_t AF_SSM_CONV_MAX_TOKENS = 16;
// Offsets and sizes are 32-bit launch scalars. The margin covers the index arithmetic of one row.
static constexpr int64_t AF_MAX_ELEMS = INT_MAX - 4096;

using af_dims = std::array<int64_t, GGML_MAX_DIMS>;

static bool af_is_reduction(int op) {
    return op == AF_SUM_ROWS || op == AF_RMS_NORM || op == AF_SOFT_MAX;
}

static bool af_is_pointwise(int op) {
    return !af_is_reduction(op) && op != AF_CONCAT && op != AF_NOP;
}

static int af_n_params(int op) {
    switch (op) {
        case AF_SCALE:
        case AF_CLAMP:    return 2;
        case AF_FILL:
        case AF_RMS_NORM:
        case AF_SOFT_MAX: return 1;
        default:          return 0;
    }
}

// An operand >= 0 is a region node. An operand < 0 is binding -1 - operand.
static constexpr int16_t AF_NONE = INT16_MIN;

struct af_snode {
    uint8_t op     = 0;
    uint8_t dim    = 0; // CONCAT dim, SSM_CONV taps
    int16_t a      = AF_NONE;
    int16_t b      = AF_NONE;
    bool    scalar = false; // row scalar
    bool    virt   = false; // no code: only copies of it at view indices (af_inst) are computed
};

struct af_sbinding {
    uint8_t mod    = 0;  // bit k: the index in dim k is i_k % ne_k
    int8_t  shift  = -1; // dim k with the index i_k - lo (CONCAT sources after the first), or -1
    uint8_t bc0    = 0;  // flat kernel with vec4: one value for each row (dim 0 broadcast)
    uint8_t vload  = 0;  // rows kernel with vec4: the row is contiguous, a chunk is one vector load
    uint8_t tap    = 0;  // reads at k * tap stride for the taps of SSM_CONV
    uint8_t gather = 0;  // GET_ROWS: one more offset, the row index from an I32 tensor times a stride
    int8_t  share  = -1; // binding with the same tensor and strides: only the offsets are new
    uint8_t pad    = 0;  // PAD: zero where an index is past the extent of the tensor
};

struct af_sout {
    int16_t node = 0;
    uint8_t vec  = 0; // rows kernel with vec4: the output row is contiguous, a chunk is one vector store
    uint8_t mask = 0; // stores only the domain box [lo, lo + n), at index i - lo. 2: the box in dim 0 is aligned to 4
};

// Everything that the generated source depends on. key() is the kernel cache key.
struct af_spec {
    bool                     rows  = false; // the region has reductions
    int                      lanes = 0;     // rows kernel: work items for each row. lanes == wg: one workgroup for each row
    int                      wg    = 0;
    bool                     vec4  = false; // 4 values for each step: chunks in the rows kernel, vectors in the flat kernel
    int                      nv    = 0;     // rows kernel with vec4: chunks for each lane, unrolled. 0: loops
    int                      dims  = 4;     // flat kernel: domain dims after the merge of contiguous dims (1, 2 or 4)
    std::vector<af_snode>    nodes;
    std::vector<af_sbinding> bindings;
    std::vector<int16_t>     csrc; // CONCAT sources (bindings). A CONCAT node has a = first entry, b = count.
    std::vector<af_sout>     outs;

    // Binary form of the fields above.
    std::string key() const {
        std::string k;
        k.reserve(32 + nodes.size() * sizeof(af_snode) + bindings.size() * sizeof(af_sbinding) + outs.size() * sizeof(af_sout));
        const int32_t head[6] = { rows, vec4, lanes, wg, nv, dims };
        k.append((const char *) head, sizeof(head));
        k.append((const char *) nodes.data(), nodes.size() * sizeof(af_snode));
        k.append("|", 1);
        k.append((const char *) bindings.data(), bindings.size() * sizeof(af_sbinding));
        k.append("|", 1);
        k.append((const char *) csrc.data(), csrc.size() * sizeof(int16_t));
        k.append("|", 1);
        k.append((const char *) outs.data(), outs.size() * sizeof(af_sout));
        return k;
    }

    std::string signature() const {
        std::string s = rows ? (vec4 ? "rows4" : "rows") : (vec4 ? "pw4" : "pw");
        s += " D" + std::to_string(dims) + " V" + std::to_string(nv);
        s += " L" + std::to_string(lanes) + " W" + std::to_string(wg) + " |";
        for (const af_snode & n : nodes) {
            s += " " + std::to_string(n.op) + "." + std::to_string(n.dim) + "." + std::to_string(n.a) + "." + std::to_string(n.b) +
                 (n.scalar ? "s" : "r");
        }
        s += " |";
        for (const af_sbinding & b : bindings) {
            s += " " + std::to_string(b.mod) + "." + std::to_string(b.shift) + "." + std::to_string(b.bc0) + "." + std::to_string(b.vload) +
                 "." + std::to_string(b.tap) + "." + std::to_string(b.gather) + "." + std::to_string(b.share) + "." + std::to_string(b.pad);
        }
        s += " |";
        for (const af_sout & o : outs) {
            s += " " + std::to_string(o.node) + "." + std::to_string(o.vec) + "." + std::to_string(o.mask);
        }
        s += " |";
        for (int16_t c : csrc) {
            s += " " + std::to_string(c);
        }
        return s;
    }
};

// Host side of a binding: the tensor and how the domain indices map to its elements.
struct af_bdata {
    const ggml_tensor * t     = nullptr;
    int64_t             lo    = 0;                // CONCAT: start of this source in the CONCAT dim
    af_dims             ne    = { 1, 1, 1, 1 };   // extent in each domain dim, 1 for a broadcast
    af_dims             ns    = { 0, 0, 0, 0 };   // element stride for each domain dim
    int64_t             off   = 0;                // element offset
    int64_t             tap   = 0;                // element stride of the taps
    const ggml_tensor * ids   = nullptr;          // gather: I32 row indices
    af_dims             ins   = { 0, 0, 0, 0 };   // gather: element stride of ids for each domain dim
    int64_t             ioff  = 0;
    int64_t             gs    = 0;                // gather: element stride of the source rows
    int64_t             gn    = 1;                // gather: source rows
    bool                fixed = false;            // reshaped, not shared with later operands
    bool                virt  = false;            // read only by virtual nodes

    bool same_layout(const af_bdata & o) const {
        return t == o.t && lo == o.lo && ne == o.ne && ns == o.ns && tap == o.tap && ids == o.ids && ins == o.ins && gs == o.gs && gn == o.gn;
    }

    bool same(const af_bdata & o) const { return same_layout(o) && off == o.off && ioff == o.ioff; }
};

// Host side of an output: a tensor and the element strides for each domain dim.
struct af_odata {
    const ggml_tensor * t    = nullptr;
    af_dims             ne   = { 1, 1, 1, 1 };
    af_dims             ns   = { 0, 0, 0, 0 };
    int64_t             off  = 0;
    bool                mask = false;
    af_dims             lo   = { 0, 0, 0, 0 };
    af_dims             n    = { 1, 1, 1, 1 };
};

// Region under construction.
struct af_builder {
    af_spec                           spec;
    std::vector<const ggml_tensor *>  tensors; // tensor of each region node, nullptr for the copies of af_inst
    std::vector<const ggml_tensor *>  ptens;   // tensor with the op params of each region node
    std::vector<int>                  index;   // graph index of each graph node in the region
    std::vector<af_bdata>             bd;      // in the order of spec.bindings
    // Shape and element strides of each region node, in domain dims. af_refine reshapes them.
    std::vector<af_dims>              node_ne, node_ns;
    // Stores of AF_NOP nodes: value node, destination, and the NOP node.
    struct dest {
        int      node;
        af_odata o;
        int      nop;
    };
    std::vector<dest>                 extra;
    std::vector<std::pair<const ggml_tensor *, int>> alias; // reshapes of region nodes that region nodes read
    std::map<std::string, int16_t>    inst;    // af_inst copies by node and index map
    int64_t                           dom[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
    bool                              dom_set = false;
    bool                              r_fixed = false;
    // The node being added has a shape that the domain splits (af_split_ok): grp maps its dims to domain dims.
    bool                              split_op = false;
    int                               split_grp[GGML_MAX_DIMS][2];
    af_dims                           split_ns;
    int                               last    = -1;  // graph index of the last region node
    // CONCAT nodes whose only reader is the next CONCAT. That one reads their sources. src: the sources, flattened.
    struct absorbed_concat {
        const ggml_tensor *              t;
        std::vector<const ggml_tensor *> src;
    };
    std::vector<absorbed_concat>      absorbed;
    // REPEAT nodes whose only reader is the next node, a GET_ROWS. That one reads the source (a broadcast).
    std::vector<std::pair<const ggml_tensor *, const ggml_tensor *>> rep;
    int                               n_absorbed = 0; // absorbed nodes that a region node reads
    const char *                      stop    = "";  // why the region ends, for the fusion trace

    int find(const ggml_tensor * t) const {
        for (size_t k = 0; k < tensors.size(); ++k) {
            if (tensors[k] == t && t != nullptr) {
                return (int) k;
            }
        }
        for (const auto & a : alias) {
            if (a.first == t) {
                return a.second;
            }
        }
        return -1;
    }

    // A region value or a view of one.
    int find_value(const ggml_tensor * t) const {
        const int k = find(t);
        if (k >= 0) {
            return k;
        }
        return t->view_src != nullptr ? find(t->view_src) : -1;
    }
};

struct af_region {
    int                              first  = 0;
    int                              last   = 0;
    ggml_hrx_kernel *                kernel = nullptr;
    std::string                      label;
    std::vector<int64_t>             workload;
    std::vector<int64_t>             scalars;
    std::vector<const ggml_tensor *> buffers;
    std::vector<const ggml_tensor *> outputs; // every tensor that the kernel writes
};

struct af_plan {
    std::vector<uint8_t>   fp;    // what the plan depends on, see af_fingerprint
    std::vector<int>       start; // region index for each graph node, or -1
    std::vector<af_region> regions;
};

struct ggml_hrx_autofuse {
    // Plans by first node and node count. A graph that runs again unchanged uses its plan again.
    std::map<std::pair<const void *, int>, af_plan> plans;
    const af_plan *                                 cur = nullptr;
    std::map<std::string, ggml_hrx_kernel *>        kernels; // by af_spec::key(). The device owns the kernels.
};

struct af_fp_tensor {
    const void * t;
    const void * data;
    const void * buf;
    const void * view_src;
    int64_t      ne[GGML_MAX_DIMS];
    size_t       nb[GGML_MAX_DIMS];
    int32_t      op;
    int32_t      type;
    int32_t      flags;
    int32_t      uses;
    int32_t      op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
};

static int32_t af_use_count(const ggml_cgraph * g, const ggml_tensor * t);

// The nodes and their sources: addresses, shapes, ops, params, flags and use counts.
static void af_fingerprint(const ggml_cgraph * g, std::vector<uint8_t> & fp) {
    fp.clear();
    auto put = [&](const ggml_tensor * t, bool node) {
        af_fp_tensor r;
        std::memset(&r, 0, sizeof(r));
        r.t        = t;
        r.data     = t->data;
        r.buf      = t->view_src != nullptr ? t->view_src->buffer : t->buffer;
        r.view_src = t->view_src;
        std::memcpy(r.ne, t->ne, sizeof(r.ne));
        std::memcpy(r.nb, t->nb, sizeof(r.nb));
        r.op    = t->op;
        r.type  = t->type;
        r.flags = t->flags;
        if (node) {
            r.uses = af_use_count(g, t);
            std::memcpy(r.op_params, t->op_params, sizeof(r.op_params));
        }
        const size_t n = fp.size();
        fp.resize(n + sizeof(r));
        std::memcpy(fp.data() + n, &r, sizeof(r));
    };
    for (int i = 0; i < g->n_nodes; ++i) {
        const ggml_tensor * t = g->nodes[i];
        put(t, true);
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            if (t->src[k] != nullptr) {
                put(t->src[k], false);
            }
        }
    }
}

static ggml_backend_buffer_t af_buffer(const ggml_tensor * t) {
    return t->view_src != nullptr ? t->view_src->buffer : t->buffer;
}

// Element strides and size of a tensor that a kernel reads or writes.
static bool af_tensor_ok(const ggml_tensor * t, ggml_type type = GGML_TYPE_F32) {
    int64_t s[GGML_MAX_DIMS];
    if (t->type != type || !ggml_hrx_elem_strides(t, s) || (int64_t) (ggml_nbytes(t) / ggml_type_size(type)) > AF_MAX_ELEMS) {
        return false;
    }
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        if (s[k] > AF_MAX_ELEMS || t->ne[k] > AF_MAX_ELEMS) {
            return false;
        }
    }
    return true;
}

static int af_op_of(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_ADD:      return AF_ADD;
        case GGML_OP_SUB:      return AF_SUB;
        case GGML_OP_MUL:      return AF_MUL;
        case GGML_OP_DIV:      return AF_DIV;
        case GGML_OP_SCALE:    return AF_SCALE;
        case GGML_OP_CLAMP:    return AF_CLAMP;
        case GGML_OP_FILL:     return AF_FILL;
        case GGML_OP_SQR:      return AF_SQR;
        case GGML_OP_SQRT:     return AF_SQRT;
        case GGML_OP_LOG:      return AF_LOG;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
        case GGML_OP_REPEAT:
        case GGML_OP_GET_ROWS:
        case GGML_OP_PAD:      return AF_COPY;
        case GGML_OP_CONCAT:   return AF_CONCAT;
        case GGML_OP_SUM_ROWS: return AF_SUM_ROWS;
        case GGML_OP_RMS_NORM: return AF_RMS_NORM;
        case GGML_OP_SOFT_MAX: return AF_SOFT_MAX;
        case GGML_OP_SSM_CONV: return AF_SSM_CONV;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(t)) {
                case GGML_UNARY_OP_SIGMOID: return AF_SIGMOID;
                case GGML_UNARY_OP_TANH:    return AF_TANH;
                case GGML_UNARY_OP_EXP:     return AF_EXP;
                case GGML_UNARY_OP_NEG:     return AF_NEG;
                case GGML_UNARY_OP_ABS:     return AF_ABS;
                case GGML_UNARY_OP_SILU:    return AF_SILU;
                case GGML_UNARY_OP_RELU:    return AF_RELU;
                case GGML_UNARY_OP_GELU:    return AF_GELU;
                case GGML_UNARY_OP_SGN:     return AF_SGN;
                case GGML_UNARY_OP_STEP:    return AF_STEP;
                default:                    return -1;
            }
        default: return -1;
    }
}

static af_dims af_ne(const ggml_tensor * t) {
    return { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
}

// Element strides. A size-1 dim has stride 0.
static af_dims af_ns(const ggml_tensor * t) {
    af_dims s;
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        s[k] = t->ne[k] == 1 ? 0 : (int64_t) (t->nb[k] / ggml_type_size(t->type));
    }
    return s;
}

// Contiguous element strides of shape ne. A size-1 dim has stride 0.
static af_dims af_cont_ns(const int64_t * ne) {
    af_dims s;
    int64_t st = 1;
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        s[k] = ne[k] == 1 ? 0 : st;
        st *= ne[k];
    }
    return s;
}

// Maps old dim k to the new dims [grp[k][0], grp[k][1]) of a reshape that only splits dims.
static bool af_split_dims(const int64_t * o, const int64_t * n, int grp[GGML_MAX_DIMS][2]) {
    int j = 0;
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        int64_t prod = 1;
        grp[k][0] = j;
        while (prod < o[k]) {
            if (j >= GGML_MAX_DIMS) {
                return false;
            }
            prod *= n[j++];
        }
        if (prod != o[k]) {
            return false;
        }
        grp[k][1] = j;
    }
    for (; j < GGML_MAX_DIMS; ++j) {
        if (n[j] != 1) {
            return false;
        }
    }
    return true;
}

static bool af_split_layout(const int grp[GGML_MAX_DIMS][2], const int64_t * o, const int64_t * n, af_dims & ne, af_dims & ns) {
    af_dims rne = { 1, 1, 1, 1 };
    af_dims rns = { 0, 0, 0, 0 };
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        if (ne[k] == 1) {
            continue;
        }
        if (ne[k] != o[k]) {
            return false;
        }
        int64_t st = ns[k];
        for (int j = grp[k][0]; j < grp[k][1]; ++j) {
            rne[j] = n[j];
            rns[j] = n[j] == 1 ? 0 : st;
            st *= n[j];
        }
    }
    ne = rne;
    ns = rns;
    return true;
}

static int af_bind_d(af_builder & b, const af_bdata & d, const af_sbinding & sb) {
    for (size_t k = 0; k < b.bd.size(); ++k) {
        const af_sbinding & o = b.spec.bindings[k];
        if (!b.bd[k].fixed && !b.bd[k].virt && b.bd[k].same(d) && o.mod == sb.mod && o.shift == sb.shift && o.tap == sb.tap &&
            o.gather == sb.gather && o.pad == sb.pad) {
            return (int) k;
        }
    }
    if ((int) b.bd.size() >= AF_MAX_BINDINGS || !af_tensor_ok(d.t) || (d.ids != nullptr && !af_tensor_ok(d.ids, GGML_TYPE_I32))) {
        return INT_MIN;
    }
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        if (d.ns[k] > AF_MAX_ELEMS || d.ins[k] > AF_MAX_ELEMS) {
            return INT_MIN;
        }
    }
    if (d.off > AF_MAX_ELEMS || d.ioff > AF_MAX_ELEMS || d.tap > AF_MAX_ELEMS || d.gs > AF_MAX_ELEMS) {
        return INT_MIN;
    }
    af_sbinding sbn = sb;
    for (size_t k = 0; k < b.bd.size() && sb.shift < 0; ++k) {
        const af_sbinding & o = b.spec.bindings[k];
        if (!b.bd[k].fixed && !b.bd[k].virt && o.share < 0 && b.bd[k].same_layout(d) && o.mod == sb.mod && o.shift == sb.shift &&
            o.tap == sb.tap && o.gather == sb.gather && o.pad == sb.pad) {
            sbn.share = (int8_t) k;
            break;
        }
    }
    b.bd.push_back(d);
    b.spec.bindings.push_back(sbn);
    return (int) b.bd.size() - 1;
}

// Returns the binding of an outside tensor. tne is the shape that the tensor broadcasts to.
static int af_bind(af_builder & b, const ggml_tensor * t, const int64_t * tne, int shift, int64_t lo) {
    af_sbinding sb;
    sb.shift = (int8_t) shift;
    for (int k = 0; k < GGML_MAX_DIMS && tne != nullptr; ++k) {
        if (t->ne[k] != 1 && t->ne[k] != tne[k]) {
            sb.mod |= (uint8_t) (1 << k);
        }
    }
    af_bdata d;
    d.t  = t;
    d.lo = lo;
    d.ne = af_ne(t);
    d.ns = af_ns(t);
    return af_bind_d(b, d, sb);
}

static bool af_same_rows(const af_builder & b, const ggml_tensor * t) {
    return t->ne[1] == b.dom[1] && t->ne[2] == b.dom[2] && t->ne[3] == b.dom[3];
}

// Map from the domain index i to the index j of region node x, for its view s: j_k = sum_d A[k][d] * i_d + c_k. x is
// contiguous. Each view dim with size > 1 must have the domain size and a stride that is a multiple of one x dim.
static bool af_view_map(const af_builder & b, int x, const ggml_tensor * s, int64_t A[GGML_MAX_DIMS][GGML_MAX_DIMS],
                        int64_t c[GGML_MAX_DIMS]) {
    const ggml_tensor * xt = b.tensors[x];
    const af_dims &     T  = b.node_ne[x];
    if (xt == nullptr || !ggml_is_contiguous(xt) || ggml_nelements(xt) != T[0] * T[1] * T[2] * T[3]) {
        return false;
    }
    if (s != xt && (s->view_src != xt || s->type != GGML_TYPE_F32)) {
        return false;
    }
    int64_t st[GGML_MAX_DIMS];
    st[0] = 1;
    for (int k = 1; k < GGML_MAX_DIMS; ++k) {
        st[k] = st[k - 1] * T[k - 1];
    }
    const size_t offs = s == xt ? 0 : s->view_offs;
    if (offs % 4 != 0) {
        return false;
    }
    int64_t rem = (int64_t) (offs / 4);
    std::memset(A, 0, sizeof(int64_t) * GGML_MAX_DIMS * GGML_MAX_DIMS);
    for (int k = GGML_MAX_DIMS - 1; k >= 0; --k) {
        c[k] = 0;
        if (T[k] > 1) {
            c[k] = rem / st[k];
            rem -= c[k] * st[k];
        }
    }
    if (rem != 0) {
        return false;
    }
    int64_t hi[GGML_MAX_DIMS];
    std::memcpy(hi, c, sizeof(hi));
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (s->ne[d] == 1) {
            continue;
        }
        if (s->ne[d] != b.dom[d] || s->nb[d] % 4 != 0) {
            return false;
        }
        const int64_t e  = (int64_t) (s->nb[d] / 4);
        int           kk = -1;
        for (int k = GGML_MAX_DIMS - 1; k >= 0 && kk < 0; --k) {
            if (T[k] > 1 && st[k] <= e) {
                kk = k;
            }
        }
        if (kk < 0 || e % st[kk] != 0) {
            return false;
        }
        A[kk][d] += e / st[kk];
        hi[kk] += (e / st[kk]) * (s->ne[d] - 1);
    }
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        if (hi[k] > T[k] - 1) {
            return false;
        }
    }
    return true;
}

// Binding p read at index j = A i + c of its domain, as a binding of the current domain.
static int af_derive(af_builder & b, int p, const int64_t A[GGML_MAX_DIMS][GGML_MAX_DIMS], const int64_t c[GGML_MAX_DIMS]) {
    const af_sbinding sb0 = b.spec.bindings[p];
    if (sb0.mod != 0 || sb0.shift >= 0 || sb0.pad) {
        return INT_MIN;
    }
    af_bdata d = b.bd[p];
    d.fixed    = false;
    d.virt     = false;
    for (int dd = 0; dd < GGML_MAX_DIMS; ++dd) {
        int64_t s = 0, is = 0;
        for (int k = 0; k < GGML_MAX_DIMS; ++k) {
            s += b.bd[p].ns[k] * A[k][dd];
            is += b.bd[p].ins[k] * A[k][dd];
        }
        d.ns[dd]  = s;
        d.ins[dd] = is;
        d.ne[dd]  = s != 0 || is != 0 ? b.dom[dd] : 1;
        if (b.dom[dd] == 1) {
            d.ns[dd]  = 0;
            d.ins[dd] = 0;
        }
    }
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        d.off += b.bd[p].ns[k] * c[k];
        d.ioff += b.bd[p].ins[k] * c[k];
    }
    af_sbinding sb;
    sb.tap    = sb0.tap;
    sb.gather = sb0.gather;
    return af_bind_d(b, d, sb);
}

static bool af_const0(const af_builder & b, int16_t o) {
    if (o == AF_NONE) {
        return true;
    }
    if (o >= 0) {
        return b.node_ne[o][0] == 1;
    }
    const af_bdata & d = b.bd[-1 - o];
    return d.ns[0] == 0 && d.ins[0] == 0;
}

// Copy of the point-wise region node x, read at index j = A i + c. Returns the operand of the copy, or AF_NONE.
static int16_t af_inst(af_builder & b, int x, const int64_t A[GGML_MAX_DIMS][GGML_MAX_DIMS], const int64_t c[GGML_MAX_DIMS]) {
    std::string key((const char *) &x, sizeof(x));
    key.append((const char *) A, sizeof(int64_t) * GGML_MAX_DIMS * GGML_MAX_DIMS);
    key.append((const char *) c, sizeof(int64_t) * GGML_MAX_DIMS);
    auto it = b.inst.find(key);
    if (it != b.inst.end()) {
        return it->second;
    }
    const af_snode sn = b.spec.nodes[x];
    if (!af_is_pointwise(sn.op) || (int) b.spec.nodes.size() >= AF_MAX_NODES) {
        return AF_NONE;
    }
    int16_t ops[2] = { sn.a, sn.b };
    for (int16_t & o : ops) {
        if (o == AF_NONE) {
            continue;
        }
        if (o < 0) {
            const int bi = af_derive(b, -1 - o, A, c);
            if (bi == INT_MIN) {
                return AF_NONE;
            }
            o = (int16_t) (-1 - bi);
            continue;
        }
        // An operand of x can broadcast to x: then its index is 0 in that dim.
        int64_t Ay[GGML_MAX_DIMS][GGML_MAX_DIMS];
        int64_t cy[GGML_MAX_DIMS];
        for (int k = 0; k < GGML_MAX_DIMS; ++k) {
            const bool bc = b.node_ne[o][k] == 1;
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                Ay[k][d] = bc ? 0 : A[k][d];
            }
            cy[k] = bc ? 0 : c[k];
        }
        o = af_inst(b, o, Ay, cy);
        if (o == AF_NONE) {
            return AF_NONE;
        }
    }
    if ((int) b.spec.nodes.size() >= AF_MAX_NODES) {
        return AF_NONE;
    }
    af_snode n = sn;
    n.a        = ops[0];
    n.b        = ops[1];
    n.virt     = false;
    n.scalar   = false;
    const bool c0 = af_const0(b, n.a) && af_const0(b, n.b);
    b.spec.nodes.push_back(n);
    b.tensors.push_back(nullptr);
    b.ptens.push_back(b.ptens[x]);
    b.node_ne.push_back({ c0 ? 1 : b.dom[0], b.dom[1], b.dom[2], b.dom[3] });
    b.node_ns.push_back({ 0, 0, 0, 0 });
    const int16_t r = (int16_t) (b.spec.nodes.size() - 1);
    b.inst[key]     = r;
    return r;
}

// Operand for source s of a node with shape tne. external_only: the op cannot take a region value.
static bool af_operand(af_builder & b, const ggml_tensor * s, const int64_t * tne, bool external_only, int16_t * out) {
    int       x = b.find(s);
    const int k = x;
    if (x < 0 && s->view_src != nullptr) {
        x = b.find(s->view_src);
    }
    if (x >= 0) {
        const af_snode & n = b.spec.nodes[x];
        if (external_only || n.op == AF_NOP) {
            b.stop = "op needs an outside source";
            return false;
        }
        if (b.split_op) {
            *out = (int16_t) x;
            return true;
        }
        if (k >= 0 && !n.virt) {
            *out = (int16_t) k;
            return true;
        }
        // A view of a region value: compute the value again at the view indices.
        int64_t A[GGML_MAX_DIMS][GGML_MAX_DIMS];
        int64_t c[GGML_MAX_DIMS];
        if (!af_view_map(b, x, s, A, c)) {
            b.stop = "view of a region value";
            return false;
        }
        bool ident = !n.virt && b.node_ne[x] == af_dims{ b.dom[0], b.dom[1], b.dom[2], b.dom[3] };
        for (int i = 0; i < GGML_MAX_DIMS && ident; ++i) {
            ident = c[i] == 0;
            for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                ident = ident && A[i][j] == (i == j && b.dom[i] > 1 ? 1 : 0);
            }
        }
        if (ident) {
            *out = (int16_t) x;
            return true;
        }
        const int16_t r = af_inst(b, x, A, c);
        if (r == AF_NONE) {
            b.stop = "view of a region value";
            return false;
        }
        *out = r;
        return true;
    }
    int bi = INT_MIN;
    if (b.split_op) {
        af_bdata d;
        d.t  = s;
        d.ne = af_ne(s);
        d.ns = af_ns(s);
        if (af_split_layout(b.split_grp, s->ne, b.dom, d.ne, d.ns)) {
            bi = af_bind_d(b, d, af_sbinding());
        }
    } else {
        bi = af_bind(b, s, tne, -1, 0);
    }
    if (bi == INT_MIN) {
        b.stop = "binding limit or layout";
        return false;
    }
    *out = (int16_t) (-1 - bi);
    return true;
}

// Sets the domain for a node of shape `shape` (the source of a reduction) and output t. Returns false if they do not fit.
static bool af_domain(af_builder & b, const ggml_tensor * shape, const ggml_tensor * t, bool red) {
    if (!b.dom_set) {
        for (int k = 0; k < GGML_MAX_DIMS; ++k) {
            b.dom[k] = shape->ne[k];
        }
        b.r_fixed = red || shape->ne[0] > 1;
        b.dom_set = true;
    } else {
        if (!af_same_rows(b, shape) || !af_same_rows(b, t)) {
            return false;
        }
        if (!b.r_fixed && shape->ne[0] > 1) {
            b.dom[0]  = shape->ne[0];
            b.r_fixed = true;
        } else if (!b.r_fixed && red) {
            b.r_fixed = true;
        }
        if (shape->ne[0] != b.dom[0] && (red || shape->ne[0] != 1)) {
            return false;
        }
    }
    return t->ne[0] == b.dom[0] || t->ne[0] == 1;
}

// A point-wise node whose shape the domain splits (for example [C] for a domain [64, C/64]) runs in the domain. It reads
// outside tensors of its shape with split strides, and contiguous region values of the domain shape by flat index.
static bool af_split_ok(af_builder & b, const ggml_tensor * t, int op) {
    if (!b.dom_set || !af_is_pointwise(op) || op == AF_SSM_CONV || t->op == GGML_OP_GET_ROWS || t->op == GGML_OP_REPEAT || t->op == GGML_OP_CPY ||
        !af_split_dims(t->ne, b.dom, b.split_grp)) {
        return false;
    }
    af_dims ne = af_ne(t);
    b.split_ns = af_ns(t);
    if (!af_split_layout(b.split_grp, t->ne, b.dom, ne, b.split_ns)) {
        return false;
    }
    const af_dims dom  = { b.dom[0], b.dom[1], b.dom[2], b.dom[3] };
    const af_dims cont = af_cont_ns(b.dom);
    for (int k = 0; k < GGML_MAX_SRC && op != AF_FILL; ++k) {
        const ggml_tensor * s = t->src[k];
        if (s == nullptr) {
            continue;
        }
        if (!ggml_are_same_shape(s, t)) {
            return false;
        }
        const int x = b.find_value(s);
        if (x < 0) {
            af_dims sne = af_ne(s), sns = af_ns(s);
            if (!af_split_layout(b.split_grp, s->ne, b.dom, sne, sns)) {
                return false;
            }
            continue;
        }
        if (b.spec.nodes[x].virt || b.spec.nodes[x].op == AF_NOP || b.tensors[x] == nullptr || b.node_ne[x] != dom || b.node_ns[x] != cont ||
            !ggml_is_contiguous(b.tensors[x]) || !ggml_is_contiguous(s) || (s != b.tensors[x] && s->view_offs != 0)) {
            return false;
        }
    }
    return true;
}

// The region moves to a new shape. The nodes so far become virtual. Only for point-wise regions.
static bool af_can_redomain(const af_builder & b, const ggml_tensor * t) {
    bool reads = false;
    for (int k = 0; k < GGML_MAX_SRC; ++k) {
        reads = reads || (t->src[k] != nullptr && b.find_value(t->src[k]) >= 0);
    }
    if (!reads || !b.absorbed.empty() || !b.extra.empty()) {
        return false;
    }
    for (const af_snode & n : b.spec.nodes) {
        if (!n.virt && !af_is_pointwise(n.op)) {
            return false;
        }
    }
    return true;
}

static void af_redomain(af_builder & b) {
    for (af_snode & n : b.spec.nodes) {
        n.virt = true;
    }
    for (af_bdata & d : b.bd) {
        d.virt = true;
    }
    b.dom_set = false;
    b.r_fixed = false;
    b.inst.clear();
}

// Copy of a region value, or of a permutation view of one, to another layout. The copy is not computed: the value is
// also stored with the strides of the copy. Returns 1 if added, 0 if t is not such a copy, -1 if it is but fails.
static int af_add_store(af_builder & b, const ggml_tensor * t, int op) {
    if (op == AF_CONCAT) {
        const int d = ggml_get_op_params_i32(t, 0);
        const ggml_tensor * srcs[2] = { t->src[0], t->src[1] };
        int x[2] = { -1, -1 };
        for (int k = 0; k < 2; ++k) {
            x[k] = srcs[k] != nullptr ? b.find(srcs[k]) : -1;
        }
        if (x[0] < 0 && x[1] < 0) {
            return 0;
        }
        // Region values with the domain shape. An outside source of that shape is copied.
        const af_dims dom = { b.dom[0], b.dom[1], b.dom[2], b.dom[3] };
        if (d < 0 || d >= GGML_MAX_DIMS || srcs[0] == nullptr || srcs[1] == nullptr) {
            b.stop = "concat params";
            return -1;
        }
        for (int k = 0; k < 2; ++k) {
            if (af_ne(srcs[k]) != dom) {
                b.stop = "concat of region values";
                return -1;
            }
            if (x[k] >= 0) {
                const af_snode & n = b.spec.nodes[x[k]];
                if (n.virt || n.op == AF_NOP || b.tensors[x[k]] != srcs[k] || b.node_ne[x[k]] != dom) {
                    b.stop = "concat of region values";
                    return -1;
                }
                continue;
            }
            af_snode c;
            c.op = AF_COPY;
            if (b.find_value(srcs[k]) >= 0 || !af_operand(b, srcs[k], b.dom, true, &c.a) || (int) b.spec.nodes.size() >= AF_MAX_NODES) {
                b.stop = "concat of region and outside values";
                return -1;
            }
            b.spec.nodes.push_back(c);
            b.tensors.push_back(nullptr);
            b.ptens.push_back(srcs[k]);
            b.node_ne.push_back(dom);
            b.node_ns.push_back({ 0, 0, 0, 0 });
            x[k] = (int) b.spec.nodes.size() - 1;
        }
        int64_t lo = 0;
        for (int k = 0; k < 2; ++k) {
            af_builder::dest e;
            e.node  = x[k];
            e.o.t   = t;
            e.o.ne  = dom;
            e.o.off = lo * (int64_t) (t->nb[d] / 4);
            e.nop   = (int) b.spec.nodes.size();
            for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                e.o.ns[j] = dom[j] == 1 ? 0 : (int64_t) (t->nb[j] / 4);
            }
            b.extra.push_back(e);
            lo += srcs[k]->ne[d];
        }
    } else if (t->op == GGML_OP_CPY || t->op == GGML_OP_CONT || t->op == GGML_OP_DUP) {
        // A copy of a region value x, or of a box of it with permuted dims: x is also stored with the strides of the
        // copy, in that box only. x is contiguous with the domain shape.
        const ggml_tensor * s  = t->src[0];
        const int           x  = b.find_value(s);
        if (x < 0) {
            return 0;
        }
        const af_snode &    n  = b.spec.nodes[x];
        const af_dims &     T  = b.node_ne[x];
        const ggml_tensor * xt = b.tensors[x];
        if (n.virt || n.op == AF_NOP || xt == nullptr || T != af_dims{ b.dom[0], b.dom[1], b.dom[2], b.dom[3] } || !ggml_is_contiguous(xt) ||
            ggml_nelements(xt) != T[0] * T[1] * T[2] * T[3] || !(s == xt || s->view_src == xt) || s->type != GGML_TYPE_F32 ||
            ggml_nelements(s) != ggml_nelements(t)) {
            return 0;
        }
        // Layout of t in the dims of s. CPY can change the shape.
        af_dims tne = af_ne(t), tns = af_ns(t);
        if (!ggml_are_same_shape(s, t)) {
            int grp[GGML_MAX_DIMS][2];
            if (!af_split_dims(t->ne, s->ne, grp) || !af_split_layout(grp, t->ne, s->ne, tne, tns)) {
                return 0;
            }
        }
        int64_t st[GGML_MAX_DIMS];
        st[0] = 1;
        for (int k = 1; k < GGML_MAX_DIMS; ++k) {
            st[k] = st[k - 1] * T[k - 1];
        }
        const size_t offs = s == xt ? 0 : s->view_offs;
        if (offs % 4 != 0) {
            return 0;
        }
        int64_t rem = (int64_t) (offs / 4);
        int64_t c[GGML_MAX_DIMS];
        for (int k = GGML_MAX_DIMS - 1; k >= 0; --k) {
            c[k] = T[k] > 1 ? rem / st[k] : 0;
            rem -= c[k] * st[k];
        }
        if (rem != 0) {
            return 0;
        }
        af_builder::dest e;
        e.node = x;
        e.o.t  = t;
        e.o.ne = T;
        e.o.ns = { 0, 0, 0, 0 };
        e.nop  = (int) b.spec.nodes.size();
        for (int k = 0; k < GGML_MAX_DIMS; ++k) {
            e.o.lo[k] = c[k];
            e.o.n[k]  = 1;
        }
        bool used[GGML_MAX_DIMS] = { false, false, false, false };
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (s->ne[d] == 1) {
                continue;
            }
            int kk = -1;
            for (int k = 0; k < GGML_MAX_DIMS && kk < 0; ++k) {
                if (!used[k] && T[k] > 1 && s->nb[d] % 4 == 0 && (int64_t) (s->nb[d] / 4) == st[k] && c[k] + s->ne[d] <= T[k]) {
                    kk = k;
                }
            }
            if (kk < 0) {
                return 0;
            }
            used[kk]     = true;
            e.o.ns[kk]   = tns[d];
            e.o.n[kk]    = s->ne[d];
        }
        for (int k = 0; k < GGML_MAX_DIMS; ++k) {
            e.o.mask = e.o.mask || e.o.lo[k] != 0 || e.o.n[k] != T[k];
        }
        if (t->op == GGML_OP_CPY && b.find_value(t->src[1]) >= 0) {
            b.stop = "copy into a region value";
            return -1;
        }
        b.extra.push_back(e);
    } else {
        return 0;
    }
    af_snode n;
    n.op = AF_NOP;
    b.spec.nodes.push_back(n);
    b.tensors.push_back(t);
    b.ptens.push_back(t);
    b.node_ne.push_back({ b.dom[0], b.dom[1], b.dom[2], b.dom[3] });
    b.node_ns.push_back({ 0, 0, 0, 0 });
    return 1;
}

// GET_ROWS of an outside F32 tensor (or of an absorbed REPEAT of one) as a gathered binding.
static bool af_gather_operand(af_builder & b, const ggml_tensor * t, int16_t * out) {
    const ggml_tensor * src = t->src[0];
    const ggml_tensor * ids = t->src[1];
    for (const auto & r : b.rep) {
        if (r.first == src) {
            src = r.second;
        }
    }
    if (ids == nullptr || ids->type != GGML_TYPE_I32 || b.find_value(src) >= 0 || b.find_value(ids) >= 0 || ids->ne[3] != 1) {
        b.stop = "get_rows sources";
        return false;
    }
    af_bdata d;
    d.t   = src;
    d.ids = ids;
    const af_dims ss = af_ns(src);
    const af_dims is = af_ns(ids);
    // dst[i0, i1, i2, i3] = src[i0, ids[i1, i2, i3], i2, i3]. A source dim of size 1 broadcasts.
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        if (k != 1 && src->ne[k] != 1 && src->ne[k] != t->ne[k]) {
            b.stop = "get_rows shape";
            return false;
        }
        d.ns[k] = k == 1 || t->ne[k] == 1 ? 0 : ss[k];
        d.ne[k] = t->ne[k] == 1 || (k != 1 && src->ne[k] == 1) ? 1 : t->ne[k];
        d.ins[k] = k == 0 || t->ne[k] == 1 ? 0 : is[k - 1];
    }
    d.ne[1] = t->ne[1];
    d.gs    = ss[1];
    d.gn    = src->ne[1];
    af_sbinding sb;
    sb.gather = 1;
    const int bi = af_bind_d(b, d, sb);
    if (bi == INT_MIN) {
        b.stop = "binding limit or layout";
        return false;
    }
    *out = (int16_t) (-1 - bi);
    return true;
}

// SSM_CONV with outside x and w, as taps: out[c, t, s] = sum_k x[t + k, c, s] * w[k, c].
static bool af_ssm_conv_operands(af_builder & b, const ggml_tensor * t, af_snode & n) {
    const ggml_tensor * x = t->src[0];
    const ggml_tensor * w = t->src[1];
    const int64_t       K = w->ne[0];
    if (b.find_value(x) >= 0 || b.find_value(w) >= 0 || K < 1 || K > AF_MAX_TAPS || t->ne[1] > AF_SSM_CONV_MAX_TOKENS || x->ne[3] != 1 ||
        w->ne[2] != 1 || w->ne[3] != 1 || x->ne[1] != w->ne[1] || t->ne[0] != x->ne[1] || t->ne[1] != x->ne[0] - K + 1 || t->ne[2] != x->ne[2]) {
        b.stop = "ssm_conv sources";
        return false;
    }
    const af_dims xs = af_ns(x);
    const af_dims ws = af_ns(w);
    af_bdata dx;
    dx.t   = x;
    dx.ne  = { t->ne[0], t->ne[1], t->ne[2], 1 };
    dx.ns  = { t->ne[0] == 1 ? 0 : xs[1], t->ne[1] == 1 ? 0 : (int64_t) (x->nb[0] / 4), t->ne[2] == 1 ? 0 : xs[2], 0 };
    dx.tap = (int64_t) (x->nb[0] / 4);
    af_bdata dw;
    dw.t   = w;
    dw.ne  = { t->ne[0], 1, 1, 1 };
    dw.ns  = { t->ne[0] == 1 ? 0 : ws[1], 0, 0, 0 };
    dw.tap = (int64_t) (w->nb[0] / 4);
    af_sbinding sb;
    sb.tap = 1;
    const int bx = af_bind_d(b, dx, sb);
    const int bw = bx == INT_MIN ? INT_MIN : af_bind_d(b, dw, sb);
    if (bw == INT_MIN) {
        b.stop = "binding limit or layout";
        return false;
    }
    n.a   = (int16_t) (-1 - bx);
    n.b   = (int16_t) (-1 - bw);
    n.dim = (uint8_t) K;
    return true;
}

// Adds node t to the region. On failure the region is unchanged, except for unused bindings.
static bool af_add_node_impl(af_builder & b, const ggml_tensor * t);

static bool af_add_node(af_builder & b, const ggml_tensor * t) {
    // A failed add can move the domain and add copies. Keep the old state.
    af_builder saved = b;
    b.split_op       = false;
    if (af_add_node_impl(b, t)) {
        if (b.spec.nodes.back().op != AF_NOP) {
            if (b.split_op) {
                b.node_ne.push_back({ b.dom[0], b.dom[1], b.dom[2], b.dom[3] });
                b.node_ns.push_back(b.split_ns);
            } else {
                b.node_ne.push_back(af_ne(t));
                b.node_ns.push_back(af_ns(t));
            }
        }
        b.split_op = false;
        return true;
    }
    const char * stop = b.stop;
    b                 = std::move(saved);
    b.stop            = stop;
    return false;
}

static bool af_add_node_impl(af_builder & b, const ggml_tensor * t) {
    const int op = af_op_of(t);
    if (op < 0) {
        b.stop = "op";
        return false;
    }
    if ((int) b.spec.nodes.size() >= AF_MAX_NODES) {
        b.stop = "node limit";
        return false;
    }
    if (!af_tensor_ok(t)) {
        b.stop = "type or layout";
        return false;
    }
    if (t->view_src != nullptr && b.find(t->view_src) >= 0) {
        b.stop = "writes into a region value";
        return false;
    }
    const ggml_tensor * s0 = t->src[0];
    const ggml_tensor * s1 = t->src[1];
    if (s0 == nullptr) {
        b.stop = "no source";
        return false;
    }
    if (b.dom_set) {
        const int st = af_add_store(b, t, op);
        if (st != 0) {
            return st > 0;
        }
    }

    // Domain. A reduction gives the row length of its input.
    const bool          red   = af_is_reduction(op);
    const ggml_tensor * shape = red ? s0 : t;
    {
        int64_t dom[GGML_MAX_DIMS];
        std::memcpy(dom, b.dom, sizeof(dom));
        const bool r_fixed = b.r_fixed;
        if (!af_domain(b, shape, t, red)) {
            std::memcpy(b.dom, dom, sizeof(dom));
            b.r_fixed = r_fixed;
            if (af_split_ok(b, t, op)) {
                b.split_op = true;
            } else if (af_can_redomain(b, t)) {
                af_redomain(b);
                if (!af_domain(b, shape, t, red)) {
                    b.stop = "domain";
                    return false;
                }
            } else {
                b.stop = "domain";
                return false;
            }
        }
    }
    if (b.dom[0] * b.dom[1] * b.dom[2] * b.dom[3] > AF_MAX_ELEMS) {
        b.stop = "size";
        return false;
    }
    const int64_t * tne = b.split_op ? b.dom : t->ne;

    af_snode n;
    n.op = (uint8_t) op;
    switch (op) {
        case AF_ADD:
        case AF_SUB:
        case AF_MUL:
        case AF_DIV:
            if (s1 == nullptr || !ggml_are_same_shape(s0, t) || !ggml_can_repeat(s1, t) || (b.split_op && !ggml_are_same_shape(s1, t))) {
                b.stop = "shape";
                return false;
            }
            if (!af_operand(b, s0, tne, false, &n.a) || !af_operand(b, s1, tne, false, &n.b)) {
                return false;
            }
            break;
        case AF_FILL:
            // FILL does not read its source.
            if (!ggml_are_same_shape(s0, t)) {
                b.stop = "shape";
                return false;
            }
            break;
        case AF_COPY:
            if (t->op == GGML_OP_GET_ROWS) {
                if (!af_gather_operand(b, t, &n.a)) {
                    return false;
                }
                break;
            }
            if (t->op == GGML_OP_PAD) {
                // Right padding only.
                bool ok = b.find_value(s0) < 0 && !b.split_op;
                for (int k = 0; k < GGML_MAX_DIMS; ++k) {
                    ok = ok && ggml_get_op_params_i32(t, 2 * k) == 0 && s0->ne[k] <= t->ne[k];
                }
                af_bdata d;
                d.t  = s0;
                d.ne = af_ne(s0);
                d.ns = af_ns(s0);
                af_sbinding sb;
                sb.pad = 1;
                const int bi = ok && ggml_get_op_params_i32(t, 8) == 0 ? af_bind_d(b, d, sb) : INT_MIN;
                if (bi == INT_MIN) {
                    b.stop = "pad";
                    return false;
                }
                n.a = (int16_t) (-1 - bi);
                break;
            }
            if (t->op == GGML_OP_REPEAT ? !ggml_can_repeat(s0, t) : !ggml_are_same_shape(s0, t)) {
                b.stop = "shape";
                return false;
            }
            if (t->op == GGML_OP_CPY && b.find_value(s1) >= 0) {
                b.stop = "copy into a region value";
                return false;
            }
            // A copy of a view of a region value reads the value at the view indices.
            if (!af_operand(b, s0, tne, t->op == GGML_OP_REPEAT, &n.a)) {
                return false;
            }
            break;
        case AF_SSM_CONV:
            if (!af_ssm_conv_operands(b, t, n)) {
                return false;
            }
            break;
        case AF_CONCAT: {
            const int d = ggml_get_op_params_i32(t, 0);
            if (s1 == nullptr || d < 0 || d >= GGML_MAX_DIMS) {
                b.stop = "concat params";
                return false;
            }
            if (ggml_hrx_concat_is_transposed(t)) {
                b.stop = "transposed concat source";
                return false;
            }
            std::vector<const ggml_tensor *> srcs;
            int n_absorbed = 0;
            for (const ggml_tensor * x : { s0, s1 }) {
                bool found = false;
                for (const af_builder::absorbed_concat & a : b.absorbed) {
                    if (a.t == x && ggml_get_op_params_i32(x, 0) == d) {
                        srcs.insert(srcs.end(), a.src.begin(), a.src.end());
                        n_absorbed += 1;
                        found = true;
                    }
                }
                if (!found) {
                    srcs.push_back(x);
                }
            }
            n.dim = (uint8_t) d;
            n.a   = (int16_t) b.spec.csrc.size();
            n.b   = (int16_t) srcs.size();
            int64_t lo = 0;
            for (size_t k = 0; k < srcs.size(); ++k) {
                const ggml_tensor * x = srcs[k];
                if (b.find_value(x) >= 0) {
                    b.stop = "op needs an outside source";
                    return false;
                }
                const int bi = af_bind(b, x, nullptr, k == 0 ? -1 : d, lo);
                if (bi == INT_MIN) {
                    b.stop = "binding limit or layout";
                    return false;
                }
                b.spec.csrc.push_back((int16_t) bi);
                lo += x->ne[d];
            }
            b.n_absorbed += n_absorbed;
            break;
        }
        case AF_SOFT_MAX:
            if (t->src[1] != nullptr || t->src[2] != nullptr || ggml_get_op_params_f32(t, 1) != 0.0f) {
                b.stop = "soft_max mask, sinks or ALiBi";
                return false;
            }
            if (!ggml_are_same_shape(s0, t) || !af_operand(b, s0, t->ne, false, &n.a)) {
                return false;
            }
            break;
        case AF_RMS_NORM:
            if (!(ggml_get_op_params_f32(t, 0) >= 0.0f)) {
                b.stop = "eps";
                return false;
            }
            if (!ggml_are_same_shape(s0, t) || !af_operand(b, s0, t->ne, false, &n.a)) {
                return false;
            }
            break;
        case AF_SUM_ROWS:
            if (!af_operand(b, s0, s0->ne, false, &n.a)) {
                return false;
            }
            break;
        default:
            if (!ggml_are_same_shape(s0, t)) {
                b.stop = "shape";
                return false;
            }
            if (!af_operand(b, s0, tne, false, &n.a)) {
                return false;
            }
            break;
    }
    // A reduction needs a row-shaped input.
    if (red && n.a >= 0 && b.node_ne[n.a][0] != b.dom[0]) {
        b.stop = "domain";
        return false;
    }
    if ((int) b.spec.nodes.size() >= AF_MAX_NODES) {
        b.stop = "node limit";
        return false;
    }
    b.spec.nodes.push_back(n);
    b.tensors.push_back(t);
    b.ptens.push_back(t);
    return true;
}

// Node t reads a contiguous reshape of a region node. If all region nodes are element-wise and the reshape only splits
// dims, the region moves to the new shape and the reshape becomes an alias of the node.
static void af_refine(af_builder & b, const ggml_tensor * t) {
    for (int si = 0; si < GGML_MAX_SRC; ++si) {
        const ggml_tensor * s = t->src[si];
        if (s == nullptr || s->view_src == nullptr || b.find(s) >= 0) {
            continue;
        }
        const int x = b.find(s->view_src);
        if (x < 0 || b.spec.nodes[x].virt) {
            continue;
        }
        const ggml_tensor * xt = b.tensors[x];
        if (s->data != xt->data || !ggml_is_contiguous(s) || !ggml_is_contiguous(xt) || ggml_nelements(s) != ggml_nelements(xt)) {
            continue;
        }
        for (size_t n = 0; n < b.spec.nodes.size(); ++n) {
            if (b.spec.nodes[n].virt) {
                continue;
            }
            if (!af_is_pointwise(b.spec.nodes[n].op) || b.node_ne[n][0] != b.dom[0] || b.node_ne[n] != b.node_ne[x]) {
                return;
            }
        }
        int grp[GGML_MAX_DIMS][2];
        if (!af_split_dims(b.dom, s->ne, grp)) {
            continue;
        }
        std::vector<af_bdata> bd = b.bd;
        std::vector<af_dims>  nne = b.node_ne, nns = b.node_ns;
        bool ok = true;
        for (size_t k = 0; k < bd.size() && ok; ++k) {
            if (bd[k].virt) {
                continue;
            }
            const af_sbinding & sb = b.spec.bindings[k];
            af_dims ine = bd[k].ne;
            ok = sb.mod == 0 && sb.shift < 0 && af_split_layout(grp, b.dom, s->ne, bd[k].ne, bd[k].ns) &&
                 (!sb.gather || af_split_layout(grp, b.dom, s->ne, ine, bd[k].ins));
        }
        for (size_t k = 0; k < nne.size() && ok; ++k) {
            if (!b.spec.nodes[k].virt) {
                ok = af_split_layout(grp, b.dom, s->ne, nne[k], nns[k]);
            }
        }
        if (!ok) {
            continue;
        }
        b.bd      = bd;
        b.node_ne = nne;
        b.node_ns = nns;
        for (af_bdata & d : b.bd) {
            d.fixed = true;
        }
        for (int k = 0; k < GGML_MAX_DIMS; ++k) {
            b.dom[k] = s->ne[k];
        }
        b.r_fixed = true;
        b.inst.clear();
        b.alias.push_back({ s, x });
    }
}

// A CONCAT whose only reader is the next compute node, a CONCAT in the same dim, is not computed: that CONCAT reads its
// sources directly. Thus a chain of CONCATs is one node.
static bool af_absorb_concat(const ggml_cgraph * g, int k, int last_max, af_builder & b) {
    const ggml_tensor * t = g->nodes[k];
    if (t->op != GGML_OP_CONCAT || t->view_src != nullptr || (t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || af_use_count(g, t) != 1 ||
        t->src[0] == nullptr || t->src[1] == nullptr || !af_tensor_ok(t) || b.find_value(t->src[0]) >= 0 || b.find_value(t->src[1]) >= 0) {
        return false;
    }
    int u = k + 1;
    while (u <= last_max && ggml_op_is_empty(g->nodes[u]->op)) {
        ++u;
    }
    if (u > last_max) {
        return false;
    }
    const ggml_tensor * c = g->nodes[u];
    if (c->op != GGML_OP_CONCAT || (c->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_get_op_params_i32(c, 0) != ggml_get_op_params_i32(t, 0) ||
        (c->src[0] != t && c->src[1] != t)) {
        return false;
    }
    af_builder::absorbed_concat a;
    a.t = t;
    for (const ggml_tensor * x : { t->src[0], t->src[1] }) {
        bool found = false;
        for (const af_builder::absorbed_concat & o : b.absorbed) {
            if (o.t == x && ggml_get_op_params_i32(x, 0) == ggml_get_op_params_i32(t, 0)) {
                a.src.insert(a.src.end(), o.src.begin(), o.src.end());
                found = true;
            }
        }
        if (!found) {
            a.src.push_back(x);
        }
    }
    b.absorbed.push_back(std::move(a));
    return true;
}

// A REPEAT that only broadcasts an outside tensor, and whose only reader is the next compute node, a GET_ROWS, is not
// computed: the GET_ROWS reads the source.
static bool af_absorb_repeat(const ggml_cgraph * g, int k, int last_max, af_builder & b) {
    const ggml_tensor * t = g->nodes[k];
    const ggml_tensor * s = t->src[0];
    if (t->op != GGML_OP_REPEAT || t->view_src != nullptr || (t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || af_use_count(g, t) != 1 ||
        s == nullptr || !af_tensor_ok(t) || !af_tensor_ok(s) || b.find_value(s) >= 0) {
        return false;
    }
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (s->ne[d] != 1 && s->ne[d] != t->ne[d]) {
            return false;
        }
    }
    int u = k + 1;
    while (u <= last_max && ggml_op_is_empty(g->nodes[u]->op)) {
        ++u;
    }
    if (u > last_max) {
        return false;
    }
    const ggml_tensor * c = g->nodes[u];
    if (c->op != GGML_OP_GET_ROWS || (c->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || c->src[0] != t) {
        return false;
    }
    b.rep.push_back({ t, s });
    return true;
}

// Builds the region that starts at node first and ends at or before node last_max.
static void af_build(const ggml_cgraph * g, int first, int last_max, af_builder & b) {
    b = af_builder();
    for (int k = first; k <= last_max; ++k) {
        const ggml_tensor * t = g->nodes[k];
        if (ggml_op_is_empty(t->op)) {
            continue;
        }
        if (ggml_is_empty(t) || (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            b.stop = "empty or not computed";
            break;
        }
        if (af_absorb_concat(g, k, last_max, b)) {
            continue;
        }
        if (af_absorb_repeat(g, k, last_max, b)) {
            continue;
        }
        if (af_op_of(t) >= 0) {
            af_refine(b, t);
        }
        const size_t n_rep = b.rep.size();
        if (!af_add_node(b, t)) {
            break;
        }
        if (t->op == GGML_OP_GET_ROWS && n_rep > 0 && b.rep.back().first == t->src[0]) {
            b.n_absorbed += 1;
        }
        b.index.push_back(k);
        b.last = k;
    }
}

static int32_t af_use_count(const ggml_cgraph * g, const ggml_tensor * t) {
    if (g->use_counts == nullptr || g->visited_hash_set.size == 0) {
        return -1;
    }
    const size_t pos = ggml_hash_find(&g->visited_hash_set, t);
    if (pos == GGML_HASHSET_FULL || !ggml_bitset_get(g->visited_hash_set.used, pos)) {
        return -1;
    }
    return g->use_counts[pos];
}

static bool af_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (af_buffer(a) != af_buffer(b)) {
        return false;
    }
    const uintptr_t a0 = (uintptr_t) a->data;
    const uintptr_t b0 = (uintptr_t) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

static bool af_in_region(const af_builder & b, const ggml_tensor * t) {
    for (const ggml_tensor * r : b.tensors) {
        if (r == t) {
            return true;
        }
    }
    return false;
}

static int af_reads_in(const ggml_cgraph * g, const af_builder & b, const ggml_tensor * t, int first, int depth);

// A view node whose readers are all region nodes or such views.
static bool af_consumed(const ggml_cgraph * g, const af_builder & b, const ggml_tensor * v, int first, int depth) {
    if ((v->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || depth > 8) {
        return false;
    }
    const int32_t uses = af_use_count(g, v);
    return uses > 0 && af_reads_in(g, b, v, first, depth + 1) == uses;
}

// Reads of t by region nodes, directly or through consumed views.
static int af_reads_in(const ggml_cgraph * g, const af_builder & b, const ggml_tensor * t, int first, int depth) {
    int n = 0;
    for (int k = first; k <= b.last; ++k) {
        const ggml_tensor * u = g->nodes[k];
        int                 m = 0;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            m += u->src[s] == t ? 1 : 0;
        }
        if (m == 0) {
            continue;
        }
        if (af_in_region(b, u) || (ggml_op_is_empty(u->op) && af_consumed(g, b, u, first, depth))) {
            n += m;
        }
    }
    return n;
}

// Sets the outputs and the row scalar flags, checks the aliasing rules and drops the virtual nodes. Returns false if the
// region cannot run fused.
static bool af_finish(const ggml_cgraph * g, int first, af_builder & b, std::vector<af_odata> & od, int min_ops) {
    const int     n_nodes = (int) b.spec.nodes.size();
    const int64_t R       = b.dom[0];
    b.spec.rows = false;
    b.spec.outs.clear();
    od.clear();
    if ((int) b.index.size() + b.n_absorbed < min_ops) {
        return false;
    }
    for (int k = 0; k < n_nodes; ++k) {
        af_snode &          n = b.spec.nodes[k];
        const ggml_tensor * t = b.tensors[k];
        n.scalar    = !n.virt && (n.op == AF_SUM_ROWS || (b.node_ne[k][0] == 1 && R > 1));
        b.spec.rows = b.spec.rows || (!n.virt && af_is_reduction(n.op));
        if (t == nullptr || n.op == AF_NOP) {
            continue;
        }
        const int32_t all_uses = af_use_count(g, t);
        const bool    out      = (t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || t->view_src != nullptr || all_uses <= 0 ||
                                 af_reads_in(g, b, t, first, 0) != all_uses;
        if (out && n.virt) {
            b.stop = "virtual value is read outside";
            return false;
        }
        if (out) {
            af_odata o;
            o.t  = t;
            o.ne = b.node_ne[k];
            o.ns = b.node_ns[k];
            b.spec.outs.push_back({ (int16_t) k, 0, 0 });
            od.push_back(o);
        }
    }
    std::vector<int> nop_of(od.size(), -1);
    for (const af_builder::dest & e : b.extra) {
        const bool al4 = e.o.lo[0] % 4 == 0 && e.o.n[0] % 4 == 0;
        b.spec.outs.push_back({ (int16_t) e.node, 0, (uint8_t) (!e.o.mask ? 0 : al4 ? 2 : 1) });
        od.push_back(e.o);
        nop_of.push_back(e.nop);
    }
    if ((int) od.size() > AF_MAX_OUTPUTS) {
        b.stop = "output limit";
        return false;
    }

    const int n_out = (int) od.size();
    for (int o = 0; o < n_out; ++o) {
        const int         k  = b.spec.outs[o].node;
        const af_snode &  n  = b.spec.nodes[k];
        const ggml_tensor * to = od[o].t;
        for (int p = 0; p < n_out; ++p) {
            // Parts of one CONCAT do not overlap.
            if (p != o && af_overlap(to, od[p].t) && !(nop_of[o] >= 0 && nop_of[o] == nop_of[p])) {
                b.stop = "outputs overlap";
                return false;
            }
        }
        // An output can use the memory of a binding only in place: same layout, read at the same index, and no reads
        // after the node that stores it.
        for (size_t bi = 0; bi < b.bd.size(); ++bi) {
            const af_bdata & d  = b.bd[bi];
            const bool       used_b = [&]() {
                for (const af_snode & r : b.spec.nodes) {
                    if (r.virt) {
                        continue;
                    }
                    if (r.op == AF_CONCAT) {
                        for (int c = 0; c < r.b; ++c) {
                            if (b.spec.csrc[r.a + c] == (int16_t) bi) {
                                return true;
                            }
                        }
                    } else if (r.a == (int16_t) (-1 - (int) bi) || r.b == (int16_t) (-1 - (int) bi)) {
                        return true;
                    }
                }
                return false;
            }();
            if (!used_b || (!af_overlap(to, d.t) && !(d.ids != nullptr && af_overlap(to, d.ids)))) {
                continue;
            }
            const af_sbinding & sb    = b.spec.bindings[bi];
            bool                exact = nop_of[o] < 0 && to->data == d.t->data && af_buffer(to) == af_buffer(d.t) && sb.mod == 0 && sb.shift < 0 &&
                                    !sb.tap && !sb.gather && !n.scalar && b.node_ne[k][0] == R && b.node_ne[k] == d.ne && od[o].ns == d.ns &&
                                    od[o].off == d.off;
            for (int j = 0; j < n_nodes && exact; ++j) {
                const af_snode & r     = b.spec.nodes[j];
                const int16_t    op_bi = (int16_t) (-1 - (int) bi);
                if (r.virt) {
                    continue;
                }
                if (r.op == AF_CONCAT) {
                    for (int c = 0; c < r.b; ++c) {
                        exact = exact && b.spec.csrc[r.a + c] != (int16_t) bi;
                    }
                } else if ((r.a == op_bi || r.b == op_bi) && (j > k || r.scalar)) {
                    exact = false;
                }
            }
            if (!exact) {
                b.stop = "output aliases an input";
                return false;
            }
        }
    }

    // Drop the virtual nodes and the bindings that only they read.
    std::vector<int> nmap(n_nodes, -1);
    int              n_keep = 0;
    for (int k = 0; k < n_nodes; ++k) {
        if (!b.spec.nodes[k].virt) {
            nmap[k] = n_keep++;
        }
    }
    std::vector<bool> used(b.bd.size(), false);
    for (int k = 0; k < n_nodes; ++k) {
        const af_snode & n = b.spec.nodes[k];
        if (n.virt) {
            continue;
        }
        if (n.op == AF_CONCAT) {
            for (int c = 0; c < n.b; ++c) {
                used[b.spec.csrc[n.a + c]] = true;
            }
            continue;
        }
        for (int16_t o : { n.a, n.b }) {
            if (o != AF_NONE && o < 0) {
                used[-1 - o] = true;
            } else if (o >= 0 && nmap[o] < 0) {
                b.stop = "reads a virtual value";
                return false;
            }
        }
    }
    for (size_t k = 0; k < used.size(); ++k) {
        if (used[k] && b.spec.bindings[k].share >= 0) {
            used[b.spec.bindings[k].share] = true;
        }
    }
    std::vector<int> bmap(b.bd.size(), -1);
    int              n_used = 0;
    for (size_t k = 0; k < used.size(); ++k) {
        if (used[k]) {
            b.bd[n_used]            = b.bd[k];
            b.spec.bindings[n_used] = b.spec.bindings[k];
            bmap[k]                 = n_used++;
        }
    }
    b.bd.resize(n_used);
    b.spec.bindings.resize(n_used);
    for (af_sbinding & sb : b.spec.bindings) {
        if (sb.share >= 0) {
            sb.share = (int8_t) bmap[sb.share];
        }
    }
    for (int16_t & c : b.spec.csrc) {
        c = (int16_t) (bmap[c] >= 0 ? bmap[c] : 0);
    }
    std::vector<af_snode>             nodes;
    std::vector<const ggml_tensor *>  tensors, ptens;
    std::vector<af_dims>              nne, nns;
    for (int k = 0; k < n_nodes; ++k) {
        if (nmap[k] < 0) {
            continue;
        }
        af_snode n = b.spec.nodes[k];
        if (n.op != AF_CONCAT) {
            for (int16_t * o : { &n.a, &n.b }) {
                if (*o != AF_NONE) {
                    *o = *o >= 0 ? (int16_t) nmap[*o] : (int16_t) (-1 - bmap[-1 - *o]);
                }
            }
        }
        nodes.push_back(n);
        tensors.push_back(b.tensors[k]);
        ptens.push_back(b.ptens[k]);
        nne.push_back(b.node_ne[k]);
        nns.push_back(b.node_ns[k]);
    }
    b.spec.nodes = std::move(nodes);
    b.tensors    = std::move(tensors);
    b.ptens      = std::move(ptens);
    b.node_ne    = std::move(nne);
    b.node_ns    = std::move(nns);
    for (af_sout & o : b.spec.outs) {
        o.node = (int16_t) nmap[o.node];
    }
    return true;
}

// Loom source generation.

struct af_gen {
    std::string src;
    int         tmp = 0;

    std::string t() { return "%t" + std::to_string(tmp++); }
    void        line(const std::string & s) { src += "  " + s + "\n"; }
};

static const char * af_helpers = R"(
func.def inline @af_f32(%bits: index) -> (f32) {
  %bits_i32 = index.cast %bits : index to i32
  %vec = vector.from_elements %bits_i32 : vector<1xi32>
  %fvec = vector.bitcast %vec : vector<1xi32> to vector<1xf32>
  %value = vector.extract %fvec[0] : vector<1xf32> -> f32
  func.return %value : f32
}

// The loads have no guard: a guard merge waits for the data. The index is clamped and the value selected.
func.def inline @af_load(%buf: buffer, %span: index, %idx: index) -> (f32) {
  %zero = index.constant 0 : offset
  %c1 = index.constant 1 : index
  %zf = scalar.constant 0.0 : f32
  %view = buffer.view %buf[%zero] : buffer -> view<[%span]xf32>
  %off = index.assume %idx [range(%idx, 0, 2147483646)] : index
  %ok = index.cmp ult, %off, %span : index
  %top = index.sub %span, %c1 : index
  %i0 = index.min %off, %top : index
  %i = index.assume %i0 [lt(%i0, %span)] : index
  %x = view.load %view[%i] : view<[%span]xf32> -> f32
  %r = scf.select %ok, %x, %zf : f32
  func.return %r : f32
}

func.def inline @af_store(%buf: buffer, %span: index, %idx: index, %value: f32) {
  %zero = index.constant 0 : offset
  %view = buffer.view %buf[%zero] : buffer -> view<[%span]xf32>
  %off = index.assume %idx [range(%idx, 0, 2147483646)] : index
  %ok = index.cmp ult, %off, %span : index
  scf.if %ok {
    view.store %value, %view[%off] : f32, view<[%span]xf32>
  }
  func.return
}
)";

// Row index of a gathered binding: ids[idx], clamped to [0, rows - 1].
static const char * af_gather = R"(
func.def inline @af_row(%buf: buffer, %span: index, %idx: index, %rows: index) -> (index) {
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %view = buffer.view %buf[%zero] : buffer -> view<[%span]xi32>
  %off = index.assume %idx [range(%idx, 0, 2147483646)] : index
  %top = index.sub %span, %c1 : index
  %i0 = index.min %off, %top : index
  %i = index.assume %i0 [lt(%i0, %span)] : index
  %x = view.load %view[%i] : view<[%span]xi32> -> i32
  %r = index.cast %x : i32 to index
  %pos = index.cmp sge, %r, %c0 : index
  %r0 = scf.select %pos, %r, %c0 : index
  %rtop = index.sub %rows, %c1 : index
  %r1 = index.min %r0, %rtop : index
  %row = index.assume %r1 [range(%r1, 0, 2147483646)] : index
  func.return %row : index
}
)";

static const char * af_vec4 = R"(
// span must be at least 4.
func.def inline @af_load4(%buf: buffer, %span: index, %idx: index) -> (vector<4xf32>) {
  %zero = index.constant 0 : offset
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %z4 = vector.splatf 0.0 : vector<4xf32>
  %span4 = index.assume %span [range(%span, 4, 2147483647)] : index
  %top = index.sub %span4, %c4 : index
  %limit0 = index.sub %span4, %c3 : index
  %off = index.assume %idx [range(%idx, 0, 2147483640)] : index
  %end = index.add %off, %c3 : index
  %ok = index.cmp ult, %end, %span4 : index
  %i0 = index.min %off, %top : index
  %i, %limit = index.assume %i0, %limit0 [lt(%i0, %limit0)] : index, index
  %extent = index.add %limit, %c3 : index
  %view = buffer.view %buf[%zero] : buffer -> view<[%extent]xf32>
  %x = vector.load %view[%i] : view<[%extent]xf32> -> vector<4xf32>
  %r = scf.select %ok, %x, %z4 : vector<4xf32>
  func.return %r : vector<4xf32>
}

func.def inline @af_store4(%buf: buffer, %span: index, %idx: index, %value: vector<4xf32>) {
  %zero = index.constant 0 : offset
  %c3 = index.constant 3 : index
  %view = buffer.view %buf[%zero] : buffer -> view<[%span]xf32>
  %off = index.assume %idx [range(%idx, 0, 2147483640)] : index
  %last = index.add %off, %c3 : index
  %ok = index.cmp ult, %last, %span : index
  scf.if %ok {
    vector.store %value, %view[%off] : vector<4xf32>, view<[%span]xf32>
  }
  func.return
}
)";

// Same as @ggml_hrx_tanh in elementwise/unary_f32.loom. Thus fused and plain TANH give the same values.
static const char * af_tanh = R"(
func.def inline @af_tanh(%x: f32) -> (f32) {
  %one = scalar.constant 1.0 : f32
  %two = scalar.constant 2.0 : f32
  %small_max = scalar.constant 0.625 : f32
  %k4 = scalar.constant -0.00570498872745 : f32
  %k3 = scalar.constant 0.0206390887954 : f32
  %k2 = scalar.constant -0.0537397155531 : f32
  %k1 = scalar.constant 0.133314422036 : f32
  %k0 = scalar.constant -0.333332819422 : f32
  %ax = scalar.absf %x : f32
  %ax2 = scalar.addf %ax, %ax : f32
  %e = scalar.expf<afn> %ax2 : f32
  %e1 = scalar.addf %e, %one : f32
  %q = scalar.divf %two, %e1 : f32
  %big_abs = scalar.subf %one, %q : f32
  %big = scalar.copysignf %big_abs, %x : f32
  %s = scalar.mulf %x, %x : f32
  %p3 = scalar.fmaf %k4, %s, %k3 : f32
  %p2 = scalar.fmaf %p3, %s, %k2 : f32
  %p1 = scalar.fmaf %p2, %s, %k1 : f32
  %p0 = scalar.fmaf %p1, %s, %k0 : f32
  %ps = scalar.mulf %p0, %s : f32
  %small = scalar.fmaf %ps, %x, %x : f32
  %is_small = scalar.cmpf olt, %ax, %small_max : f32
  %r = scf.select %is_small, %small, %big : f32
  func.return %r : f32
}
)";

struct af_codegen {
    const af_spec &  sp;
    af_gen           g;
    std::vector<int> param; // first param scalar of each node
    std::vector<int> avail; // rows kernel: first loop in which the node value is known
    int              n_loops = 1;

    // Names of the values in the current scope.
    std::vector<std::string>           row_val;    // row-shaped values, by node
    std::vector<std::string>           scalar_val; // row scalars, by node
    std::vector<std::string>           load_row;   // loads of the current element, by binding
    std::vector<std::string>           load_base;  // loads at index 0 of the row, by binding
    std::map<std::string, std::string> memo;       // other loads and row indices in the current scope
    std::string                        elem;       // index of the current element in dim 0

    explicit af_codegen(const af_spec & sp) : sp(sp) {}

    // A shared binding has its own offsets. The strides and buffers are those of the binding it shares.
    int  bsrc(int b) const { return sp.bindings[b].share >= 0 ? sp.bindings[b].share : b; }
    std::string bvar(int b, const char * name) const {
        const bool own = !strcmp(name, "off") || !strcmp(name, "ioff") || !strcmp(name, "base");
        return "%b" + std::to_string(own ? b : bsrc(b)) + name;
    }
    std::string bbuf(int b) const { return "%B" + std::to_string(bsrc(b)); }
    std::string ibuf(int b) const { return "%I" + std::to_string(bsrc(b)); }
    std::string ovar(int o, const char * name) const { return "%o" + std::to_string(o) + name; }
    std::string fparam(int n, int k) const { return "%f" + std::to_string(param[n] + k); }

    // Offset of each binding in the row. Not for the shifted CONCAT sources.
    void emit_bases() {
        for (size_t b = 0; b < sp.bindings.size(); ++b) {
            if (sp.bindings[b].shift >= 0) {
                continue;
            }
            std::string sum = bvar((int) b, "off");
            for (int k = 1; k < GGML_MAX_DIMS; ++k) {
                std::string j = "%i" + std::to_string(k);
                if (sp.bindings[b].mod & (1 << k)) {
                    const std::string m = g.t();
                    g.line(m + " = index.rem " + j + ", " + bvar(b, ("n" + std::to_string(k)).c_str()) + " : index");
                    j = m;
                }
                const std::string p = g.t(), a = g.t();
                g.line(p + " = index.mul " + j + ", " + bvar(b, ("s" + std::to_string(k)).c_str()) + " : index");
                g.line(a + " = index.add " + sum + ", " + p + " : index");
                sum = a;
            }
            g.line(bvar(b, "base") + " = index.add " + sum + ", %c0 : index");
        }
    }

    // Offset of output o in the row.
    // Index i - lo of a masked output in dim k, and whether i is in [lo, lo + n).
    std::pair<std::string, std::string> boxed(int o, int k, const std::string & i) {
        const std::string lo = ovar(o, ("lo" + std::to_string(k)).c_str()), n = ovar(o, ("n" + std::to_string(k)).c_str());
        const std::string m = g.t(), d = g.t(), ge = g.t(), lt = g.t(), ok = g.t();
        g.line(m + " = index.max " + i + ", " + lo + " : index");
        g.line(d + " = index.sub " + m + ", " + lo + " : index");
        g.line(ge + " = index.cmp uge, " + i + ", " + lo + " : index");
        g.line(lt + " = index.cmp ult, " + d + ", " + n + " : index");
        g.line(ok + " = scalar.andi " + ge + ", " + lt + " : i1");
        return { d, ok };
    }

    void emit_out_bases() {
        for (size_t o = 0; o < sp.outs.size(); ++o) {
            if (sp.outs[o].mask) {
                std::string sum = ovar((int) o, "off"), ok;
                for (int k = 1; k < GGML_MAX_DIMS; ++k) {
                    const auto        bx = boxed((int) o, k, "%i" + std::to_string(k));
                    const std::string p = g.t(), a = g.t();
                    g.line(p + " = index.mul " + bx.first + ", " + ovar((int) o, ("s" + std::to_string(k)).c_str()) + " : index");
                    g.line(a + " = index.add " + sum + ", " + p + " : index");
                    sum = a;
                    if (ok.empty()) {
                        ok = bx.second;
                    } else {
                        const std::string x = g.t();
                        g.line(x + " = scalar.andi " + ok + ", " + bx.second + " : i1");
                        ok = x;
                    }
                }
                g.line(ovar((int) o, "base") + " = index.add " + sum + ", %c0 : index");
                g.line(ovar((int) o, "ok") + " = scalar.andi " + ok + ", " + ok + " : i1");
                continue;
            }
            const std::string p1 = g.t(), p2 = g.t(), p3 = g.t(), a = g.t(), a2 = g.t();
            g.line(p1 + " = index.mul %i1, " + ovar(o, "s1") + " : index");
            g.line(p2 + " = index.mul %i2, " + ovar(o, "s2") + " : index");
            g.line(p3 + " = index.mul %i3, " + ovar(o, "s3") + " : index");
            g.line(a + " = index.add " + p1 + ", " + p2 + " : index");
            g.line(a2 + " = index.add " + a + ", " + p3 + " : index");
            g.line(ovar(o, "base") + " = index.add " + a2 + ", " + ovar(o, "off") + " : index");
        }
    }

    // Offset of the current element (or of index 0 of the row) of binding b, without taps.
    std::string addr(int b, bool scalar) {
        std::string off = bvar(b, "base");
        if (!scalar) {
            std::string j = elem;
            if (sp.bindings[b].mod & 1) {
                const std::string m = g.t();
                g.line(m + " = index.rem " + j + ", " + bvar(b, "n0") + " : index");
                j = m;
            }
            const std::string p = g.t(), a = g.t();
            g.line(p + " = index.mul " + j + ", " + bvar(b, "s0") + " : index");
            g.line(a + " = index.add " + off + ", " + p + " : index");
            off = a;
        }
        if (sp.bindings[b].gather) {
            const std::string r = row(b, scalar), p = g.t(), a = g.t();
            g.line(p + " = index.mul " + r + ", " + bvar(b, "gs") + " : index");
            g.line(a + " = index.add " + off + ", " + p + " : index");
            off = a;
        }
        return off;
    }

    // Source row of gathered binding b for the current element.
    std::string row(int b, bool scalar) {
        const std::string key = "row" + std::to_string(b) + (scalar ? "s" : "r");
        auto it = memo.find(key);
        if (it != memo.end()) {
            return it->second;
        }
        std::string sum = bvar(b, "ioff");
        for (int k = scalar ? 1 : 0; k < GGML_MAX_DIMS; ++k) {
            const std::string p = g.t(), a = g.t();
            g.line(p + " = index.mul " + (k == 0 ? elem : "%i" + std::to_string(k)) + ", " + bvar(b, ("is" + std::to_string(k)).c_str()) + " : index");
            g.line(a + " = index.add " + sum + ", " + p + " : index");
            sum = a;
        }
        const std::string r = g.t();
        g.line(r + " = func.call @af_row(" + ibuf(b) + ", " + bvar(b, "isp") + ", " + sum + ", " + bvar(b, "gn") +
               ") : (buffer, index, index, index) -> (index)");
        memo[key] = r;
        return r;
    }

    std::string load(int b, bool scalar) {
        std::vector<std::string> & m = scalar ? load_base : load_row;
        if (!m[b].empty()) {
            return m[b];
        }
        const std::string off = addr(b, scalar);
        std::string       x   = g.t();
        g.line(x + " = func.call @af_load(" + bbuf(b) + ", " + bvar(b, "sp") + ", " + off + ") : (buffer, index, index) -> (f32)");
        if (sp.bindings[b].pad) {
            // Zero past the extent in any dim.
            std::string in;
            for (int k = 0; k < GGML_MAX_DIMS; ++k) {
                const std::string c = g.t();
                g.line(c + " = index.cmp ult, " + (k == 0 ? (scalar ? std::string("%c0") : elem) : "%i" + std::to_string(k)) + ", " +
                       bvar(b, ("e" + std::to_string(k)).c_str()) + " : index");
                if (!in.empty()) {
                    const std::string a = g.t();
                    g.line(a + " = scalar.andi " + in + ", " + c + " : i1");
                    in = a;
                } else {
                    in = c;
                }
            }
            const std::string v = g.t();
            g.line(v + " = scf.select " + in + ", " + x + ", %zf : f32");
            x = v;
        }
        m[b] = x;
        return x;
    }

    std::string load_tap(int b, int k, bool scalar) {
        const std::string key = "tap" + std::to_string(b) + "." + std::to_string(k) + (scalar ? "s" : "r");
        auto it = memo.find(key);
        if (it != memo.end()) {
            return it->second;
        }
        const std::string base = addr(b, scalar), kc = g.t(), p = g.t(), a = g.t(), x = g.t();
        g.line(kc + " = index.constant " + std::to_string(k) + " : index");
        g.line(p + " = index.mul " + kc + ", " + bvar(b, "t") + " : index");
        g.line(a + " = index.add " + base + ", " + p + " : index");
        g.line(x + " = func.call @af_load(" + bbuf(b) + ", " + bvar(b, "sp") + ", " + a + ") : (buffer, index, index) -> (f32)");
        memo[key] = x;
        return x;
    }

    std::string operand(int16_t a, bool scalar) {
        if (a >= 0) {
            return sp.nodes[a].scalar ? scalar_val[a] : row_val[a];
        }
        return load(-1 - a, scalar);
    }

    // Index of CONCAT source b in dim d: i_d - lo, or 0 below lo.
    std::string shifted(const std::string & i, int b) {
        const std::string m = g.t(), r = g.t();
        g.line(m + " = index.max " + i + ", " + bvar(b, "lo") + " : index");
        g.line(r + " = index.sub " + m + ", " + bvar(b, "lo") + " : index");
        return r;
    }

    // Load of a shifted CONCAT source.
    std::string load_shifted(int b, int d, bool scalar) {
        std::string sum = bvar(b, "off");
        for (int k = 0; k < GGML_MAX_DIMS; ++k) {
            std::string j = k == 0 ? (scalar ? std::string("%c0") : elem) : "%i" + std::to_string(k);
            if (k == d) {
                j = shifted(j, b);
            }
            const std::string p = g.t(), a = g.t();
            g.line(p + " = index.mul " + j + ", " + bvar(b, ("s" + std::to_string(k)).c_str()) + " : index");
            g.line(a + " = index.add " + sum + ", " + p + " : index");
            sum = a;
        }
        const std::string x = g.t();
        g.line(x + " = func.call @af_load(" + bbuf(b) + ", " + bvar(b, "sp") + ", " + sum + ") : (buffer, index, index) -> (f32)");
        return x;
    }

    // CONCAT: source k is used from its lo up to the lo of source k + 1 in dim d. With 2 sources both load and a select
    // picks one. With more, branches load only the source of the element (the same for a whole row if d > 0).
    std::string emit_concat_from(int n, int k, bool scalar) {
        const af_snode & sn = sp.nodes[n];
        const int        d  = sn.dim;
        const bool       br = sn.b > 2;
        auto src = [&](int j) {
            const int bj = sp.csrc[sn.a + j];
            if (j > 0) {
                return load_shifted(bj, d, scalar);
            }
            if (!br) {
                return load(bj, scalar);
            }
            const std::string x = g.t();
            g.line(x + " = func.call @af_load(" + bbuf(bj) + ", " + bvar(bj, "sp") + ", " + addr(bj, scalar) + ") : (buffer, index, index) -> (f32)");
            return x;
        };
        if (k == sn.b - 1) {
            return src(k);
        }
        const int         bn = sp.csrc[sn.a + k + 1];
        const std::string id = d == 0 ? (scalar ? std::string("%c0") : elem) : "%i" + std::to_string(d);
        const std::string in = g.t(), v = g.t();
        if (!br) {
            const std::string x = src(k), y = emit_concat_from(n, k + 1, scalar);
            g.line(in + " = index.cmp ult, " + id + ", " + bvar(bn, "lo") + " : index");
            g.line(v + " = scf.select " + in + ", " + x + ", " + y + " : f32");
            return v;
        }
        g.line(in + " = index.cmp ult, " + id + ", " + bvar(bn, "lo") + " : index");
        g.line(v + " = scf.if " + in + " -> (f32) {");
        const std::string x = src(k);
        g.line("scf.yield " + x + " : f32");
        g.line("} else {");
        const std::string y = emit_concat_from(n, k + 1, scalar);
        g.line("scf.yield " + y + " : f32");
        g.line("}");
        return v;
    }

    std::string emit_concat(int n, bool scalar) {
        return emit_concat_from(n, 0, scalar);
    }

    // CONCAT in a dim above 0 of 4 values of a row, from e0. The sources are contiguous in dim 0. With more than 2
    // sources, branches (the same for the whole row) load only the source of the row.
    std::string emit_concat4_from(int n, int k, const std::string & e0) {
        const af_snode & sn = sp.nodes[n];
        const int        d  = sn.dim;
        const bool       br = sn.b > 2;
        auto src = [&](int j) {
            const int   bj = sp.csrc[sn.a + j];
            std::string off;
            if (j == 0) {
                off = g.t();
                g.line(off + " = index.add " + bvar(bj, "base") + ", " + e0 + " : index");
            } else {
                off = g.t();
                g.line(off + " = index.add " + e0 + ", " + bvar(bj, "off") + " : index");
                for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                    std::string ii = "%i" + std::to_string(i);
                    if (i == d) {
                        ii = shifted(ii, bj);
                    }
                    const std::string p = g.t(), a = g.t();
                    g.line(p + " = index.mul " + ii + ", " + bvar(bj, ("s" + std::to_string(i)).c_str()) + " : index");
                    g.line(a + " = index.add " + off + ", " + p + " : index");
                    off = a;
                }
            }
            const std::string x = g.t();
            g.line(x + " = func.call @af_load4(" + bbuf(bj) + ", " + bvar(bj, "sp") + ", " + off + ") : (buffer, index, index) -> (vector<4xf32>)");
            return x;
        };
        if (k == sn.b - 1) {
            return src(k);
        }
        const int         bn = sp.csrc[sn.a + k + 1];
        const std::string in = g.t(), v = g.t();
        if (!br) {
            const std::string x = src(k), y = emit_concat4_from(n, k + 1, e0);
            g.line(in + " = index.cmp ult, %i" + std::to_string(d) + ", " + bvar(bn, "lo") + " : index");
            g.line(v + " = scf.select " + in + ", " + x + ", " + y + " : vector<4xf32>");
            return v;
        }
        g.line(in + " = index.cmp ult, %i" + std::to_string(d) + ", " + bvar(bn, "lo") + " : index");
        g.line(v + " = scf.if " + in + " -> (vector<4xf32>) {");
        const std::string x = src(k);
        g.line("scf.yield " + x + " : vector<4xf32>");
        g.line("} else {");
        const std::string y = emit_concat4_from(n, k + 1, e0);
        g.line("scf.yield " + y + " : vector<4xf32>");
        g.line("}");
        return v;
    }

    // Value of node n in the current scope. Reductions are handled by the caller.
    std::string emit_node(int n) {
        const af_snode & sn = sp.nodes[n];
        const bool       sc = sn.scalar;
        auto bin = [&](const char * op) {
            const std::string a = operand(sn.a, sc), b = operand(sn.b, sc), v = g.t();
            g.line(v + " = " + op + " " + a + ", " + b + " : f32");
            return v;
        };
        auto un = [&](const char * op) {
            const std::string a = operand(sn.a, sc), v = g.t();
            g.line(v + " = " + op + " " + a + " : f32");
            return v;
        };
        switch (sn.op) {
            case AF_ADD: return bin("scalar.addf");
            case AF_SUB: return bin("scalar.subf");
            case AF_MUL: return bin("scalar.mulf");
            case AF_DIV: return bin("scalar.divf");
            case AF_SCALE: {
                const std::string a = operand(sn.a, sc), m = g.t(), v = g.t();
                g.line(m + " = scalar.mulf " + a + ", " + fparam(n, 0) + " : f32");
                g.line(v + " = scalar.addf " + m + ", " + fparam(n, 1) + " : f32");
                return v;
            }
            case AF_CLAMP: {
                // The CPU backend does MAX(MIN(x, max), min).
                const std::string a = operand(sn.a, sc), u = g.t(), v = g.t();
                g.line(u + " = scalar.minnumf " + a + ", " + fparam(n, 1) + " : f32");
                g.line(v + " = scalar.maxnumf " + u + ", " + fparam(n, 0) + " : f32");
                return v;
            }
            case AF_FILL:    return fparam(n, 0);
            case AF_SQR: {
                const std::string a = operand(sn.a, sc), v = g.t();
                g.line(v + " = scalar.mulf " + a + ", " + a + " : f32");
                return v;
            }
            case AF_SQRT:    return un("scalar.sqrtf");
            case AF_LOG:     return un("scalar.logf<afn>");
            case AF_SIGMOID: return un("scalar.logisticf");
            case AF_TANH: {
                const std::string a = operand(sn.a, sc), v = g.t();
                g.line(v + " = func.call @af_tanh(" + a + ") : (f32) -> (f32)");
                return v;
            }
            case AF_EXP:  return un("scalar.expf<afn>");
            case AF_NEG:  return un("scalar.negf");
            case AF_ABS:  return un("scalar.absf");
            case AF_SILU: return un("scalar.siluf");
            case AF_GELU: return un("scalar.geluf<tanh, afn>");
            case AF_SGN:  return un("scalar.signf");
            case AF_RELU:
            case AF_STEP: {
                const std::string a = operand(sn.a, sc), p = g.t(), v = g.t();
                g.line(p + " = scalar.cmpf ogt, " + a + ", %zf : f32");
                g.line(v + " = scf.select " + p + ", " + (sn.op == AF_RELU ? a : std::string("%onef")) + ", %zf : f32");
                return v;
            }
            case AF_COPY:   return operand(sn.a, sc);
            case AF_CONCAT: return emit_concat(n, sc);
            case AF_SSM_CONV: {
                // Same order as ssm/ssm_conv_f32.loom: acc + x * w for k = 0, 1, ...
                std::string acc = "%zf";
                for (int k = 0; k < sn.dim; ++k) {
                    const std::string x = load_tap(-1 - sn.a, k, sc), w = load_tap(-1 - sn.b, k, sc), p = g.t(), s = g.t();
                    g.line(p + " = scalar.mulf " + x + ", " + w + " : f32");
                    g.line(s + " = scalar.addf " + acc + ", " + p + " : f32");
                    acc = s;
                }
                return acc;
            }
            case AF_RMS_NORM: {
                const std::string x = operand(sn.a, false), v = g.t();
                g.line(v + " = scalar.mulf " + x + ", %rs" + std::to_string(n) + " : f32");
                return v;
            }
            case AF_SOFT_MAX: {
                const std::string e = emit_softmax_exp(n), v = g.t();
                g.line(v + " = scalar.mulf " + e + ", %is" + std::to_string(n) + " : f32");
                return v;
            }
            default: GGML_ABORT("autofuse: bad op %d", sn.op);
        }
    }

    std::string emit_softmax_exp(int n) {
        const std::string x = operand(sp.nodes[n].a, false), xs = g.t(), sh = g.t(), e = g.t();
        g.line(xs + " = scalar.mulf " + x + ", " + fparam(n, 0) + " : f32");
        g.line(sh + " = scalar.subf " + xs + ", %mx" + std::to_string(n) + " : f32");
        g.line(e + " = scalar.expf<afn> " + sh + " : f32");
        return e;
    }

    void new_scope(const std::string & e) {
        elem = e;
        std::fill(load_row.begin(), load_row.end(), std::string());
        std::fill(load_base.begin(), load_base.end(), std::string());
        std::fill(row_val.begin(), row_val.end(), std::string());
        memo.clear();
    }

    // Vector store of the 4 values of a row from e0. The output is contiguous in dim 0. A masked output (mask 2) has a
    // box in dim 0 that is aligned to 4.
    void store4(int o, const std::string & e0, const std::string & v) {
        std::string off = g.t();
        if (sp.outs[o].mask) {
            const auto bx = boxed(o, 0, e0);
            const std::string ok = g.t();
            g.line(off + " = index.add " + ovar(o, "base") + ", " + bx.first + " : index");
            g.line(ok + " = scalar.andi " + ovar(o, "ok") + ", " + bx.second + " : i1");
            g.line("scf.if " + ok + " {");
        } else {
            g.line(off + " = index.add " + ovar(o, "base") + ", " + e0 + " : index");
        }
        g.line("func.call @af_store4(%O" + std::to_string(o) + ", " + ovar(o, "sp") + ", " + off + ", " + v + ") : (buffer, index, index, vector<4xf32>)");
        if (sp.outs[o].mask) {
            g.line("}");
        }
    }

    void store(int o, const std::string & v, bool scalar) {
        std::string off = ovar(o, "base");
        if (sp.outs[o].mask) {
            const auto        bx = boxed(o, 0, scalar ? std::string("%c0") : elem);
            const std::string p = g.t(), a = g.t(), ok = g.t();
            g.line(p + " = index.mul " + bx.first + ", " + ovar(o, "s0") + " : index");
            g.line(a + " = index.add " + off + ", " + p + " : index");
            g.line(ok + " = scalar.andi " + ovar(o, "ok") + ", " + bx.second + " : i1");
            g.line("scf.if " + ok + " {");
            g.line("func.call @af_store(%O" + std::to_string(o) + ", " + ovar(o, "sp") + ", " + a + ", " + v + ") : (buffer, index, index, f32)");
            g.line("}");
            return;
        }
        if (!scalar) {
            const std::string p = g.t(), a = g.t();
            g.line(p + " = index.mul " + elem + ", " + ovar(o, "s0") + " : index");
            g.line(a + " = index.add " + off + ", " + p + " : index");
            off = a;
        }
        g.line("func.call @af_store(%O" + std::to_string(o) + ", " + ovar(o, "sp") + ", " + off + ", " + v + ") : (buffer, index, index, f32)");
    }

    std::string helpers() const {
        bool tanh = false, gather = false;
        for (const af_snode & n : sp.nodes) {
            tanh = tanh || n.op == AF_TANH;
        }
        for (const af_sbinding & b : sp.bindings) {
            gather = gather || b.gather;
        }
        std::string h = af_helpers;
        h += tanh ? af_tanh : "";
        h += gather ? af_gather : "";
        h += sp.vec4 ? af_vec4 : "";
        return h;
    }

    // Launch arguments: the scalars (name and lower bound; the host checks the ranges), then params and buffers.
    std::vector<std::pair<std::string, int>> sc;
    std::vector<std::string>                 bufs;
    int                                      n_params = 0;

    void make_args() {
        sc = { { "%n", 1 }, { "%R", 1 }, { "%D1", 1 }, { "%D2", 1 } };
        for (size_t b = 0; b < sp.bindings.size(); ++b) {
            const af_sbinding & sb = sp.bindings[b];
            const std::string   p  = "%b" + std::to_string(b);
            if (sb.share >= 0) {
                sc.push_back({ p + "off", 0 });
                if (sb.gather) {
                    sc.push_back({ p + "ioff", 0 });
                }
                continue;
            }
            for (int k = 0; k < GGML_MAX_DIMS; ++k) {
                sc.push_back({ p + "s" + std::to_string(k), 0 });
            }
            sc.push_back({ p + "sp", 1 });
            sc.push_back({ p + "off", 0 });
            for (int k = 0; k < GGML_MAX_DIMS; ++k) {
                if (sb.mod & (1 << k)) {
                    sc.push_back({ p + "n" + std::to_string(k), 1 });
                }
            }
            if (sb.shift >= 0) {
                sc.push_back({ p + "lo", 0 });
            }
            if (sb.tap) {
                sc.push_back({ p + "t", 0 });
            }
            if (sb.pad) {
                for (int k = 0; k < GGML_MAX_DIMS; ++k) {
                    sc.push_back({ p + "e" + std::to_string(k), 1 });
                }
            }
            if (sb.gather) {
                for (int k = 0; k < GGML_MAX_DIMS; ++k) {
                    sc.push_back({ p + "is" + std::to_string(k), 0 });
                }
                sc.push_back({ p + "isp", 1 });
                sc.push_back({ p + "ioff", 0 });
                sc.push_back({ p + "gs", 0 });
                sc.push_back({ p + "gn", 1 });
            }
        }
        for (size_t o = 0; o < sp.outs.size(); ++o) {
            const std::string p = "%o" + std::to_string(o);
            for (int k = 0; k < GGML_MAX_DIMS; ++k) {
                sc.push_back({ p + "s" + std::to_string(k), 0 });
            }
            sc.push_back({ p + "sp", 1 });
            sc.push_back({ p + "off", 0 });
            if (sp.outs[o].mask) {
                for (int k = 0; k < GGML_MAX_DIMS; ++k) {
                    sc.push_back({ p + "lo" + std::to_string(k), 0 });
                    sc.push_back({ p + "n" + std::to_string(k), 1 });
                }
            }
        }
        param.clear();
        n_params = 0;
        for (size_t n = 0; n < sp.nodes.size(); ++n) {
            param.push_back(n_params);
            n_params += af_n_params(sp.nodes[n].op);
        }
        bufs.clear();
        for (size_t b = 0; b < sp.bindings.size(); ++b) {
            if (sp.bindings[b].share < 0) {
                bufs.push_back("%B" + std::to_string(b));
            }
        }
        for (size_t b = 0; b < sp.bindings.size(); ++b) {
            if (sp.bindings[b].gather && sp.bindings[b].share < 0) {
                bufs.push_back("%I" + std::to_string(b));
            }
        }
        for (size_t o = 0; o < sp.outs.size(); ++o) {
            bufs.push_back("%O" + std::to_string(o));
        }
    }

    // Argument lists with types: index arguments, then buffers. Names get the prefix (for the kernel of several parts).
    std::string arg_decl_idx(const std::string & pre) const {
        std::string args;
        for (const auto & s : sc) {
            args += (args.empty() ? "" : ", ") + pre + s.first.substr(1) + "_in: index";
        }
        for (int k = 0; k < n_params; ++k) {
            args += ", " + pre + "p" + std::to_string(k) + ": index";
        }
        return args;
    }

    std::string arg_decl_buf(const std::string & pre) const {
        std::string args;
        for (const std::string & b : bufs) {
            args += (args.empty() ? "" : ", ") + pre + b.substr(1) + ": buffer";
        }
        return args;
    }

    std::string arg_decl(const std::string & pre) const {
        const std::string b = arg_decl_buf(pre);
        return arg_decl_idx(pre) + (b.empty() ? "" : ", " + b);
    }

    std::string arg_use(const std::string & pre) const {
        std::string args;
        for (const auto & s : sc) {
            args += ", " + pre + s.first.substr(1) + "_in";
        }
        for (int k = 0; k < n_params; ++k) {
            args += ", " + pre + "p" + std::to_string(k);
        }
        for (const std::string & b : bufs) {
            args += ", " + pre + b.substr(1);
        }
        return args;
    }

    std::string arg_types() const {
        std::string t;
        for (size_t k = 0; k < sc.size() + n_params; ++k) {
            t += ", index";
        }
        for (size_t k = 0; k < bufs.size(); ++k) {
            t += ", buffer";
        }
        return t;
    }

    void emit_prologue() {
        for (const auto & s : sc) {
            g.line(s.first + " = index.assume " + s.first + "_in [range(" + s.first + "_in, " + std::to_string(s.second) + ", 2147483647)] : index");
        }
        for (int k = 0; k < n_params; ++k) {
            g.line("%f" + std::to_string(k) + " = func.call @af_f32(%p" + std::to_string(k) + ") : (index) -> (f32)");
        }
        g.line("%c0 = index.constant 0 : index");
        g.line("%wg = index.constant " + std::to_string(sp.wg) + " : index");
        g.line("%zf = scalar.constant 0.0 : f32");
        g.line("%onef = scalar.constant 1.0 : f32");
        row_val.assign(sp.nodes.size(), std::string());
        scalar_val.assign(sp.nodes.size(), std::string());
        load_row.assign(sp.bindings.size(), std::string());
        load_base.assign(sp.bindings.size(), std::string());
    }

    void emit_header(const std::string & sym) {
        make_args();
        g.src  = "// Generated by the HRX auto-fusion. " + sp.signature() + "\n\ntarget.decl @ggml_wave64\n";
        g.src += helpers();
        g.src += "\nkernel.def target(@ggml_wave64) export(\"" + sym + "\") @" + sym + "(%w: index) {\n";
        g.line("%one = index.constant 1 : index");
        g.line("%wg = index.constant " + std::to_string(sp.wg) + " : index");
        if (sp.rows) {
            g.line("kernel.launch.config workgroups(%w, %one, %one) workgroup_size(%wg, %one, %one) : index");
        } else {
            const int per = sp.vec4 ? 4 * sp.wg : sp.wg;
            g.line("%per = index.constant " + std::to_string(per) + " : index");
            g.line("%per1 = index.constant " + std::to_string(per - 1) + " : index");
            g.line("%wr = index.add %w, %per1 : index");
            g.line("%groups = index.div %wr, %per : index");
            g.line("kernel.launch.config workgroups(%groups, %one, %one) workgroup_size(%wg, %one, %one) : index");
        }
        g.src += "} launch(" + arg_decl("%") + ") {\n";
        emit_prologue();
        g.line("%group = kernel.workgroup.id<x> : index");
        g.line("%item = kernel.workitem.id<x> : index");
    }

    void emit_body() {
        if (sp.rows) {
            emit_rows();
        } else {
            emit_pointwise();
        }
    }

    // The part as an inline function of its workgroup index and work item.
    std::string part_func(const std::string & name) {
        make_args();
        g.src = "\nfunc.def inline @" + name + "(%group: index, %item: index, " + arg_decl("%") + ") {\n";
        emit_prologue();
        emit_body();
        g.line("func.return");
        g.src += "}\n";
        return g.src;
    }

    // Flat grid over all domain elements. Row scalars are computed for each element with i0 = 0. With vec4, each work
    // item does 4 values of one row.
    void emit_pointwise() {
        g.line("%gbase = index.mul %group, %wg : index");
        g.line("%gi = index.add %gbase, %item : index");
        if (sp.vec4) {
            g.line("%c4v = index.constant 4 : index");
            g.line("%i = index.mul %gi, %c4v : index");
        } else {
            g.line("%i = index.add %gi, %c0 : index");
        }
        g.line("%in = index.cmp ult, %i, %n : index");
        g.line("scf.if %in {");
        if (sp.dims == 1) {
            g.line("%i0 = index.add %i, %c0 : index");
            g.line("%i1 = index.constant 0 : index");
            g.line("%i2 = index.constant 0 : index");
            g.line("%i3 = index.constant 0 : index");
        } else if (sp.dims == 2) {
            g.line("%i0 = index.rem %i, %R : index");
            g.line("%i1 = index.div %i, %R : index");
            g.line("%i2 = index.constant 0 : index");
            g.line("%i3 = index.constant 0 : index");
        } else {
            g.line("%i0 = index.rem %i, %R : index");
            g.line("%q0 = index.div %i, %R : index");
            g.line("%i1 = index.rem %q0, %D1 : index");
            g.line("%q1 = index.div %q0, %D1 : index");
            g.line("%i2 = index.rem %q1, %D2 : index");
            g.line("%i3 = index.div %q1, %D2 : index");
        }
        emit_bases();
        emit_out_bases();
        if (sp.vec4) {
            emit_pointwise4();
            g.line("}");
            return;
        }
        new_scope("%i0");
        g.line("%first = index.cmp eq, %i0, %c0 : index");
        for (size_t n = 0; n < sp.nodes.size(); ++n) {
            if (sp.nodes[n].op == AF_NOP) {
                continue;
            }
            const std::string v = emit_node((int) n);
            (sp.nodes[n].scalar ? scalar_val : row_val)[n] = v;
        }
        for (size_t o = 0; o < sp.outs.size(); ++o) {
            const int n = sp.outs[o].node;
            if (sp.nodes[n].scalar) {
                g.line("scf.if %first {");
                store((int) o, scalar_val[n], true);
                g.line("}");
            } else {
                store((int) o, row_val[n], false);
            }
        }
        g.line("}");
    }

    // All values are row-shaped. Each binding is contiguous in dim 0 or has one value for each row (bc0).
    void emit_pointwise4() {
        std::vector<std::vector<std::string>> x(sp.bindings.size());
        new_scope("%i0");
        for (size_t b = 0; b < sp.bindings.size(); ++b) {
            if (sp.bindings[b].bc0) {
                x[b].assign(4, load((int) b, false));
                continue;
            }
            const std::string p = g.t(), off = g.t(), v = g.t();
            g.line(p + " = index.mul %i0, " + bvar((int) b, "s0") + " : index");
            g.line(off + " = index.add " + bvar((int) b, "base") + ", " + p + " : index");
            g.line(v + " = func.call @af_load4(" + bbuf((int) b) + ", " + bvar((int) b, "sp") + ", " + off + ") : (buffer, index, index) -> (vector<4xf32>)");
            for (int j = 0; j < 4; ++j) {
                const std::string e = g.t();
                g.line(e + " = vector.extract " + v + "[" + std::to_string(j) + "] : vector<4xf32> -> f32");
                x[b].push_back(e);
            }
        }
        std::vector<std::vector<std::string>> y(sp.nodes.size());
        for (int j = 0; j < 4; ++j) {
            new_scope("%i0");
            for (size_t b = 0; b < sp.bindings.size(); ++b) {
                load_row[b] = x[b][j];
            }
            for (size_t n = 0; n < sp.nodes.size(); ++n) {
                if (sp.nodes[n].op == AF_NOP) {
                    y[n].push_back("");
                    continue;
                }
                row_val[n] = emit_node((int) n);
                y[n].push_back(row_val[n]);
            }
        }
        for (size_t o = 0; o < sp.outs.size(); ++o) {
            const int         n = sp.outs[o].node;
            const std::string v = g.t();
            g.line(v + " = vector.from_elements " + y[n][0] + ", " + y[n][1] + ", " + y[n][2] + ", " + y[n][3] + " : vector<4xf32>");
            store4((int) o, "%i0", v);
        }
    }

    std::string reduce(const std::string & v, const char * kind) {
        const std::string r = g.t();
        if (sp.lanes == sp.wg) {
            g.line(r + " = kernel.workgroup.reduce<" + kind + "> " + v + " : f32");
        } else if (sp.lanes == 64) {
            g.line(r + " = kernel.subgroup.reduce<" + kind + "> " + v + " : f32");
        } else {
            g.line(r + " = kernel.subgroup.reduce<" + kind + "> " + v + " {cluster_size = " + std::to_string(sp.lanes) + "} : f32");
        }
        return r;
    }

    // Marks the row-shaped nodes that are needed to compute node n.
    void need(int n, std::vector<bool> & needed) {
        if (n < 0 || sp.nodes[n].scalar || needed[n]) {
            return;
        }
        needed[n] = true;
        if (sp.nodes[n].op != AF_CONCAT) {
            need(sp.nodes[n].a == AF_NONE ? -1 : sp.nodes[n].a, needed);
            need(sp.nodes[n].b == AF_NONE ? -1 : sp.nodes[n].b, needed);
        }
    }

    // A reduction step in a loop. kind: 0 sum, 1 sum of squares, 2 max of x * scale, 3 sum of exp(x * scale - max).
    struct acc {
        int n;
        int kind;
    };

    // Values of one element in the current scope: the needed nodes, the reduction terms and, in the last loop, the stores.
    std::vector<std::string> emit_elem(const std::string & e, const std::vector<acc> & accs, const std::vector<bool> & needed, bool last,
                                       const std::vector<std::string> * preset = nullptr, std::vector<std::string> * out_vals = nullptr,
                                       const std::vector<std::string> * preset_node = nullptr) {
        new_scope(e);
        if (preset != nullptr) {
            for (size_t b = 0; b < preset->size(); ++b) {
                load_row[b] = (*preset)[b];
            }
        }
        for (size_t n = 0; n < sp.nodes.size(); ++n) {
            if (needed[n]) {
                row_val[n] = preset_node != nullptr && !(*preset_node)[n].empty() ? (*preset_node)[n] : emit_node((int) n);
            }
        }
        std::vector<std::string> x;
        for (const acc & r : accs) {
            x.push_back(r.kind == 3 ? emit_softmax_exp(r.n) : operand(sp.nodes[r.n].a, false));
        }
        if (last) {
            for (size_t o = 0; o < sp.outs.size(); ++o) {
                const int n = sp.outs[o].node;
                if (sp.nodes[n].scalar) {
                    continue;
                }
                if (out_vals != nullptr && sp.outs[o].vec) {
                    (*out_vals)[o] = row_val[n];
                } else {
                    store((int) o, row_val[n], false);
                }
            }
        }
        return x;
    }

    // Bindings that the row-shaped nodes in needed read directly.
    std::vector<bool> row_reads(const std::vector<bool> & needed) const {
        std::vector<bool> r(sp.bindings.size(), false);
        for (size_t n = 0; n < sp.nodes.size(); ++n) {
            const af_snode & sn = sp.nodes[n];
            if (!needed[n] || sn.op == AF_CONCAT) {
                continue;
            }
            if (sn.a < 0 && sn.a != AF_NONE) {
                r[-1 - sn.a] = true;
            }
            if (sn.b < 0 && sn.b != AF_NONE) {
                r[-1 - sn.b] = true;
            }
        }
        return r;
    }

    std::string update(const acc & r, const std::string & a, const std::string & x) {
        const std::string v = g.t();
        if (r.kind == 0 || r.kind == 3) {
            g.line(v + " = scalar.addf " + a + ", " + x + " : f32");
        } else if (r.kind == 1) {
            const std::string q = g.t();
            g.line(q + " = scalar.mulf " + x + ", " + x + " : f32");
            g.line(v + " = scalar.addf " + a + ", " + q + " : f32");
        } else {
            const std::string xs = g.t();
            g.line(xs + " = scalar.mulf " + x + ", " + fparam(r.n, 0) + " : f32");
            g.line(v + " = scalar.maxnumf " + a + ", " + xs + " : f32");
        }
        return v;
    }

    // Row scalar from the reduction result r.
    void finish_acc(const acc & a, const std::string & r) {
        const std::string ns = std::to_string(a.n);
        if (a.kind == 0) {
            scalar_val[a.n] = reduce(r, "addf");
        } else if (a.kind == 1) {
            const std::string s = reduce(r, "addf"), m = g.t(), e = g.t(), q = g.t();
            g.line(m + " = scalar.divf " + s + ", %Rf : f32");
            g.line(e + " = scalar.addf " + m + ", " + fparam(a.n, 0) + " : f32");
            g.line(q + " = scalar.sqrtf " + e + " : f32");
            g.line("%rs" + ns + " = scalar.divf %onef, " + q + " : f32");
        } else if (a.kind == 2) {
            const std::string m = reduce(r, "maxnumf");
            g.line("%mx" + ns + " = scalar.addf " + m + ", %zf : f32");
        } else {
            const std::string s = reduce(r, "addf");
            g.line("%is" + ns + " = scalar.divf %onef, " + s + " : f32");
        }
    }

    // One row for each group of lanes. Each loop over the row accumulates the reductions whose inputs are known, the
    // last loop stores the row-shaped outputs.
    void emit_rows() {
        const int n_nodes = (int) sp.nodes.size();
        avail.assign(n_nodes, 0);
        for (int n = 0; n < n_nodes; ++n) {
            const af_snode & sn = sp.nodes[n];
            int a = 0;
            if (sn.op != AF_CONCAT) {
                a = std::max(sn.a >= 0 ? avail[sn.a] : 0, sn.b >= 0 ? avail[sn.b] : 0);
            }
            avail[n] = a + (sn.op == AF_SOFT_MAX ? 2 : af_is_reduction(sn.op) ? 1 : 0);
            if (sn.op == AF_SOFT_MAX) {
                n_loops = std::max(n_loops, a + 2);
            } else if (af_is_reduction(sn.op)) {
                n_loops = std::max(n_loops, a + 1);
            }
        }
        for (const af_sout & o : sp.outs) {
            if (!sp.nodes[o.node].scalar) {
                n_loops = std::max(n_loops, avail[o.node] + 1);
            }
        }

        if (sp.lanes == sp.wg) {
            g.line("%row = index.add %group, %c0 : index");
            g.line("%lane = index.add %item, %c0 : index");
            g.line("%row_ok = index.cmp ult, %row, %n : index");
            g.line("%end = index.add %R, %c0 : index");
        } else {
            g.line("%lanes = index.constant " + std::to_string(sp.lanes) + " : index");
            g.line("%rpg = index.constant " + std::to_string(sp.wg / sp.lanes) + " : index");
            g.line("%grow = index.mul %group, %rpg : index");
            g.line("%lrow = index.div %item, %lanes : index");
            g.line("%lane = index.rem %item, %lanes : index");
            g.line("%row_raw = index.add %grow, %lrow : index");
            g.line("%row = index.assume %row_raw [range(%row_raw, 0, 2147483646)] : index");
            g.line("%row_ok = index.cmp ult, %row, %n : index");
            // Lanes past the last row do no loop iterations. They must still join the reductions.
            g.line("%end = scf.select %row_ok, %R, %lane : index");
        }
        g.line("%i1 = index.rem %row, %D1 : index");
        g.line("%rq = index.div %row, %D1 : index");
        g.line("%i2 = index.rem %rq, %D2 : index");
        g.line("%i3 = index.div %rq, %D2 : index");
        g.line("%Ri = index.cast %R : index to i32");
        g.line("%Rf = scalar.sitofp %Ri : i32 to f32");
        g.line("%lowest = scalar.constant -3.4028234663852886e+38 : f32");
        emit_bases();
        emit_out_bases();
        const std::string step = sp.lanes == sp.wg ? "%wg" : "%lanes";
        if (sp.vec4) {
            for (int j = 1; j <= 4; ++j) {
                g.line("%c" + std::to_string(j) + "i = index.constant " + std::to_string(j) + " : index");
            }
            g.line("%R3 = index.add %R, %c3i : index");
            g.line("%nch = index.div %R3, %c4i : index");
            g.line(sp.lanes == sp.wg ? "%endc = index.add %nch, %c0 : index" : "%endc = scf.select %row_ok, %nch, %lane : index");
        }

        // Row scalars outside the loops. They load at index 0 of their row.
        new_scope("%c0");
        if (sp.nv > 0) {
            emit_rows_unrolled(step);
            emit_scalar_stores();
            return;
        }

        for (int k = 0; k < n_loops; ++k) {
            emit_scalars(k);
            std::vector<acc> accs;
            std::vector<bool> needed(n_nodes, false);
            for (int n = 0; n < n_nodes; ++n) {
                const af_snode & sn = sp.nodes[n];
                const int a = sn.a >= 0 ? avail[sn.a] : 0;
                if (sn.op == AF_SUM_ROWS && a == k) {
                    accs.push_back({ n, 0 });
                } else if (sn.op == AF_RMS_NORM && a == k) {
                    accs.push_back({ n, 1 });
                } else if (sn.op == AF_SOFT_MAX && a == k) {
                    accs.push_back({ n, 2 });
                } else if (sn.op == AF_SOFT_MAX && a + 1 == k) {
                    accs.push_back({ n, 3 });
                } else {
                    continue;
                }
                need(sn.a, needed);
            }
            const bool last = k == n_loops - 1;
            if (last) {
                for (const af_sout & o : sp.outs) {
                    need(o.node, needed);
                }
            }

            std::string res, iters, types;
            for (size_t j = 0; j < accs.size(); ++j) {
                res   += (j ? ", " : "") + std::string("%r") + std::to_string(k) + "_" + std::to_string(j);
                iters += (j ? ", " : "") + std::string("%a") + std::to_string(k) + "_" + std::to_string(j) + " = " + (accs[j].kind == 2 ? "%lowest" : "%zf") + " : f32";
                types += (j ? ", f32" : "f32");
            }
            auto names = [&](const std::string & prefix) {
                std::vector<std::string> v;
                for (size_t j = 0; j < accs.size(); ++j) {
                    v.push_back(prefix + std::to_string(k) + "_" + std::to_string(j));
                }
                return v;
            };
            auto join = [](const std::vector<std::string> & v) {
                std::string r;
                for (size_t j = 0; j < v.size(); ++j) {
                    r += (j ? ", " : "") + v[j];
                }
                return r;
            };
            const std::string head_res = accs.empty() ? "" : res + " = ";
            const std::string head_its = accs.empty() ? "" : "(" + iters + ") -> (" + types + ")";
            const std::vector<std::string> a = names("%a");
            if (!sp.vec4) {
                const std::string c = "%c" + std::to_string(k + 1);
                g.line(head_res + "scf.for " + c + " = [%lane to %end step " + step + "]" + head_its + " {");
                const std::vector<std::string> x = emit_elem(c, accs, needed, last);
                std::vector<std::string> next;
                for (size_t j = 0; j < accs.size(); ++j) {
                    next.push_back(update(accs[j], a[j], x[j]));
                }
                if (!accs.empty()) {
                    g.line("scf.yield " + join(next) + " : " + types);
                }
                g.line("}");
            } else {
                // Chunks of 4 values, as in the rms_norm kernels. Thus the sums are the same as there.
                const std::string ch = "%ch" + std::to_string(k);
                g.line(head_res + "scf.for " + ch + " = [%lane to %endc step " + step + "]" + head_its + " {");
                const std::string e0 = g.t(), e3 = g.t(), full = g.t();
                g.line(e0 + " = index.mul " + ch + ", %c4i : index");
                g.line(e3 + " = index.add " + e0 + ", %c3i : index");
                g.line(full + " = index.cmp ult, " + e3 + ", %R : index");
                const std::vector<std::string> u = names("%u");
                g.line((accs.empty() ? "" : join(u) + " = ") + "scf.if " + full + (accs.empty() ? "" : " -> (" + types + ")") + " {");
                std::vector<std::vector<std::string>> xs;
                std::vector<std::vector<std::string>> ov(4, std::vector<std::string>(sp.outs.size()));
                emit_chunk(e0, accs, needed, last, xs, ov);
                emit_chunk_stores(e0, last, ov);
                std::vector<std::string> next;
                for (size_t j = 0; j < accs.size(); ++j) {
                    next.push_back(chunk_sum(xs, j, a[j]));
                }
                if (!accs.empty()) {
                    g.line("scf.yield " + join(next) + " : " + types);
                }
                g.line("} else {");
                // Row tail: at most 3 values.
                const std::string te = "%te" + std::to_string(k);
                const std::vector<std::string> t = names("%tt");
                std::string titers;
                for (size_t j = 0; j < accs.size(); ++j) {
                    titers += (j ? ", " : "") + t[j] + " = " + a[j] + " : f32";
                }
                const std::vector<std::string> tr = names("%tr");
                g.line((accs.empty() ? "" : join(tr) + " = ") + "scf.for " + te + " = [" + e0 + " to %R step %c1i]" +
                       (accs.empty() ? "" : "(" + titers + ") -> (" + types + ")") + " {");
                const std::vector<std::string> x = emit_elem(te, accs, needed, last);
                std::vector<std::string> tnext;
                for (size_t j = 0; j < accs.size(); ++j) {
                    tnext.push_back(update(accs[j], t[j], x[j]));
                }
                if (!accs.empty()) {
                    g.line("scf.yield " + join(tnext) + " : " + types);
                }
                g.line("}");
                if (!accs.empty()) {
                    g.line("scf.yield " + join(tr) + " : " + types);
                }
                g.line("}");
                if (!accs.empty()) {
                    g.line("scf.yield " + join(u) + " : " + types);
                }
                g.line("}");
            }
            new_scope("%c0");
            for (size_t j = 0; j < accs.size(); ++j) {
                finish_acc(accs[j], "%r" + std::to_string(k) + "_" + std::to_string(j));
            }
        }
        emit_scalars(n_loops);
        emit_scalar_stores();
    }

    // Row scalars that become known in loop k. They load at index 0 of their row.
    void emit_scalars(int k) {
        for (size_t n = 0; n < sp.nodes.size(); ++n) {
            if (sp.nodes[n].scalar && sp.nodes[n].op != AF_SUM_ROWS && avail[n] == k) {
                scalar_val[n] = emit_node((int) n);
            }
        }
    }

    // The reductions and stores of loop k, from the accumulators and the needed nodes.
    void loop_parts(int k, std::vector<acc> & accs, std::vector<bool> & needed) {
        const int n_nodes = (int) sp.nodes.size();
        accs.clear();
        needed.assign(n_nodes, false);
        for (int n = 0; n < n_nodes; ++n) {
            const af_snode & sn = sp.nodes[n];
            const int a = sn.a >= 0 ? avail[sn.a] : 0;
            if (sn.op == AF_SUM_ROWS && a == k) {
                accs.push_back({ n, 0 });
            } else if (sn.op == AF_RMS_NORM && a == k) {
                accs.push_back({ n, 1 });
            } else if (sn.op == AF_SOFT_MAX && a == k) {
                accs.push_back({ n, 2 });
            } else if (sn.op == AF_SOFT_MAX && a + 1 == k) {
                accs.push_back({ n, 3 });
            } else {
                continue;
            }
            need(sn.a, needed);
        }
        if (k == n_loops - 1) {
            for (const af_sout & o : sp.outs) {
                need(o.node, needed);
            }
        }
    }

    // Values of the current element scope.
    struct escope {
        std::vector<std::string>           row_val, load_row, load_base;
        std::map<std::string, std::string> memo;
        std::string                        elem;
    };

    void save(escope & e) const {
        e.row_val   = row_val;
        e.load_row  = load_row;
        e.load_base = load_base;
        e.memo      = memo;
        e.elem      = elem;
    }

    void restore(const escope & e) {
        row_val   = e.row_val;
        load_row  = e.load_row;
        load_base = e.load_base;
        memo      = e.memo;
        elem      = e.elem;
    }

    // Rows kernel with vec4 and nv chunks for each lane, unrolled: the values of a chunk stay in registers from the
    // first loop that needs them to the stores. Chunk j of a lane is lane + j * step, as in the rms_norm kernels.
    // R is a multiple of 4.
    void emit_rows_unrolled(const std::string & step) {
        const int n_nodes = (int) sp.nodes.size();
        const int nv      = sp.nv;
        std::vector<std::array<escope, 4>> st(nv);
        std::vector<std::string>           valid(nv), e0(nv);
        std::vector<std::vector<bool>>     loaded(nv, std::vector<bool>(sp.bindings.size(), false));
        for (int j = 0; j < nv; ++j) {
            const std::string jc = g.t(), o = g.t(), ch = g.t(), in = g.t();
            e0[j]    = g.t();
            valid[j] = g.t();
            g.line(jc + " = index.constant " + std::to_string(j) + " : index");
            g.line(o + " = index.mul " + jc + ", " + step + " : index");
            g.line(ch + " = index.add %lane, " + o + " : index");
            g.line(e0[j] + " = index.mul " + ch + ", %c4i : index");
            g.line(in + " = index.cmp ult, " + e0[j] + ", %R : index");
            g.line(valid[j] + " = scalar.andi %row_ok, " + in + " : i1");
            for (int e = 0; e < 4; ++e) {
                std::string ei = e0[j];
                if (e > 0) {
                    ei = g.t();
                    g.line(ei + " = index.add " + e0[j] + ", %c" + std::to_string(e) + "i : index");
                }
                st[j][e].row_val.assign(n_nodes, std::string());
                st[j][e].load_row.assign(sp.bindings.size(), std::string());
                st[j][e].load_base.assign(sp.bindings.size(), std::string());
                st[j][e].elem = ei;
            }
        }
        escope sc;
        save(sc);
        for (int k = 0; k < n_loops; ++k) {
            restore(sc);
            emit_scalars(k);
            save(sc);
            std::vector<acc>  accs;
            std::vector<bool> needed;
            loop_parts(k, accs, needed);
            const bool               last = k == n_loops - 1;
            const std::vector<bool>  reads = row_reads(needed);
            std::vector<std::string> cur(accs.size(), "%zf");
            for (int j = 0; j < nv; ++j) {
                for (size_t b = 0; b < sp.bindings.size(); ++b) {
                    if (!reads[b] || !sp.bindings[b].vload || loaded[j][b]) {
                        continue;
                    }
                    const std::string off = g.t(), v = g.t();
                    g.line(off + " = index.add " + bvar((int) b, "base") + ", " + e0[j] + " : index");
                    g.line(v + " = func.call @af_load4(" + bbuf((int) b) + ", " + bvar((int) b, "sp") + ", " + off + ") : (buffer, index, index) -> (vector<4xf32>)");
                    for (int e = 0; e < 4; ++e) {
                        const std::string x = g.t();
                        g.line(x + " = vector.extract " + v + "[" + std::to_string(e) + "] : vector<4xf32> -> f32");
                        st[j][e].load_row[b] = x;
                    }
                    loaded[j][b] = true;
                }
                for (int n = 0; n < n_nodes; ++n) {
                    const af_snode & sn = sp.nodes[n];
                    bool vec = needed[n] && sn.op == AF_CONCAT && sn.dim > 0 && st[j][0].row_val[n].empty();
                    for (int c = 0; c < sn.b && vec; ++c) {
                        vec = sp.bindings[sp.csrc[sn.a + c]].vload != 0;
                    }
                    if (!vec) {
                        continue;
                    }
                    const std::string v = emit_concat4_from(n, 0, e0[j]);
                    for (int e = 0; e < 4; ++e) {
                        const std::string x = g.t();
                        g.line(x + " = vector.extract " + v + "[" + std::to_string(e) + "] : vector<4xf32> -> f32");
                        st[j][e].row_val[n] = x;
                    }
                }
                std::vector<std::vector<std::string>> xs(4);
                for (int e = 0; e < 4; ++e) {
                    restore(st[j][e]);
                    for (int n = 0; n < n_nodes; ++n) {
                        if (needed[n] && row_val[n].empty()) {
                            row_val[n] = emit_node(n);
                        }
                    }
                    for (const acc & r : accs) {
                        xs[e].push_back(operand(sp.nodes[r.n].a, false));
                    }
                    save(st[j][e]);
                }
                for (size_t i = 0; i < accs.size(); ++i) {
                    const std::string v = chunk_sum(xs, i, cur[i]), sel = g.t();
                    g.line(sel + " = scf.select " + valid[j] + ", " + v + ", " + cur[i] + " : f32");
                    cur[i] = sel;
                }
                if (!last) {
                    continue;
                }
                g.line("scf.if " + valid[j] + " {");
                for (size_t o = 0; o < sp.outs.size(); ++o) {
                    const int n = sp.outs[o].node;
                    if (sp.nodes[n].scalar) {
                        continue;
                    }
                    if (sp.outs[o].vec) {
                        const std::string v = g.t();
                        g.line(v + " = vector.from_elements " + st[j][0].row_val[n] + ", " + st[j][1].row_val[n] + ", " + st[j][2].row_val[n] + ", " +
                               st[j][3].row_val[n] + " : vector<4xf32>");
                        store4((int) o, e0[j], v);
                        continue;
                    }
                    for (int e = 0; e < 4; ++e) {
                        restore(st[j][e]);
                        store((int) o, row_val[n], false);
                    }
                }
                g.line("}");
            }
            restore(sc);
            for (size_t i = 0; i < accs.size(); ++i) {
                finish_acc(accs[i], cur[i]);
            }
            save(sc);
        }
        restore(sc);
        emit_scalars(n_loops);
    }

    // The 4 values of a full chunk from e0: vector loads of the contiguous bindings and CONCATs, then the values.
    void emit_chunk(const std::string & e0, const std::vector<acc> & accs, const std::vector<bool> & needed, bool last,
                    std::vector<std::vector<std::string>> & xs, std::vector<std::vector<std::string>> & ov) {
        const std::vector<bool> reads = row_reads(needed);
        std::vector<std::vector<std::string>> pre(4, std::vector<std::string>(sp.bindings.size()));
        for (size_t b = 0; b < sp.bindings.size(); ++b) {
            if (!reads[b] || !sp.bindings[b].vload) {
                continue;
            }
            const std::string off = g.t(), v = g.t();
            g.line(off + " = index.add " + bvar((int) b, "base") + ", " + e0 + " : index");
            g.line(v + " = func.call @af_load4(" + bbuf((int) b) + ", " + bvar((int) b, "sp") + ", " + off + ") : (buffer, index, index) -> (vector<4xf32>)");
            for (int j = 0; j < 4; ++j) {
                pre[j][b] = g.t();
                g.line(pre[j][b] + " = vector.extract " + v + "[" + std::to_string(j) + "] : vector<4xf32> -> f32");
            }
        }
        std::vector<std::vector<std::string>> pre_node(4, std::vector<std::string>(sp.nodes.size()));
        for (size_t n = 0; n < sp.nodes.size(); ++n) {
            const af_snode & sn = sp.nodes[n];
            bool vec = needed[n] && sn.op == AF_CONCAT && sn.dim > 0;
            for (int c = 0; c < sn.b && vec; ++c) {
                vec = sp.bindings[sp.csrc[sn.a + c]].vload != 0;
            }
            if (!vec) {
                continue;
            }
            const std::string v = emit_concat4_from((int) n, 0, e0);
            for (int j = 0; j < 4; ++j) {
                pre_node[j][n] = g.t();
                g.line(pre_node[j][n] + " = vector.extract " + v + "[" + std::to_string(j) + "] : vector<4xf32> -> f32");
            }
        }
        for (int j = 0; j < 4; ++j) {
            std::string e = e0;
            if (j > 0) {
                e = g.t();
                g.line(e + " = index.add " + e0 + ", %c" + std::to_string(j) + "i : index");
            }
            xs.push_back(emit_elem(e, accs, needed, last, &pre[j], &ov[j], &pre_node[j]));
        }
    }

    void emit_chunk_stores(const std::string & e0, bool last, const std::vector<std::vector<std::string>> & ov) {
        for (size_t o = 0; o < sp.outs.size() && last; ++o) {
            if (!sp.outs[o].vec || ov[0][o].empty()) {
                continue;
            }
            const std::string v = g.t();
            g.line(v + " = vector.from_elements " + ov[0][o] + ", " + ov[1][o] + ", " + ov[2][o] + ", " + ov[3][o] + " : vector<4xf32>");
            store4((int) o, e0, v);
        }
    }

    // Sum of squares of a chunk into accumulator a, as @ggml_hrx_rms_norm_sq.
    std::string chunk_sum(const std::vector<std::vector<std::string>> & xs, size_t j, const std::string & a) {
        const std::string v = g.t(), q = g.t(), r = g.t();
        g.line(v + " = vector.from_elements " + xs[0][j] + ", " + xs[1][j] + ", " + xs[2][j] + ", " + xs[3][j] + " : vector<4xf32>");
        g.line(q + " = vector.mulf " + v + ", " + v + " : vector<4xf32>");
        g.line(r + " = vector.reduce<addf> " + q + ", " + a + " : vector<4xf32>, f32");
        return r;
    }

    void emit_scalar_stores() {
        bool scalar_out = false;
        for (const af_sout & o : sp.outs) {
            scalar_out = scalar_out || sp.nodes[o.node].scalar;
        }
        if (scalar_out) {
            g.line("%lane0 = index.cmp eq, %lane, %c0 : index");
            g.line("%first = scalar.andi %row_ok, %lane0 : i1");
            g.line("scf.if %first {");
            for (size_t o = 0; o < sp.outs.size(); ++o) {
                if (sp.nodes[sp.outs[o].node].scalar) {
                    store((int) o, scalar_val[sp.outs[o].node], true);
                }
            }
            g.line("}");
        }
    }

    std::string generate(const std::string & sym) {
        emit_header(sym);
        emit_body();
        g.line("kernel.return");
        g.src += "}\n";
        return g.src;
    }
};

// Kernel of several independent parts. Part k runs on the workgroups from the launch scalar gk_in (g0 = 0) on.
static std::string af_generate_parts(const std::vector<const af_spec *> & parts, const std::string & sym) {
    std::string head = "// Generated by the HRX auto-fusion, " + std::to_string(parts.size()) + " parts.\n\ntarget.decl @ggml_wave64\n";
    std::string helpers, funcs, args, bargs, calls;
    bool        tanh = false, gather = false, vec4 = false;
    for (size_t k = 0; k < parts.size(); ++k) {
        af_codegen gen(*parts[k]);
        const std::string fn  = sym + "_p" + std::to_string(k);
        const std::string pre = "%q" + std::to_string(k) + "_";
        funcs += gen.part_func(fn);
        for (const af_snode & n : parts[k]->nodes) {
            tanh = tanh || n.op == AF_TANH;
        }
        for (const af_sbinding & b : parts[k]->bindings) {
            gather = gather || b.gather;
        }
        vec4 = vec4 || parts[k]->vec4;
        if (k > 0) {
            args += ", %g" + std::to_string(k) + "_in: index";
        }
        args += (args.empty() ? "" : ", ") + gen.arg_decl_idx(pre);
        const std::string bd = gen.arg_decl_buf(pre);
        bargs += (bargs.empty() || bd.empty() ? "" : ", ") + bd;
        std::string types = "index, index" + gen.arg_types();
        calls += "  // part " + std::to_string(k) + ": " + parts[k]->signature() + "\n";
        calls += "  %call" + std::to_string(k) + "_g = index.sub %group, %g" + std::to_string(k) + " : index\n";
        calls += "  func.call @" + fn + "(%call" + std::to_string(k) + "_g, %item" + gen.arg_use(pre) + ") : (" + types + ")\n";
    }
    helpers = af_helpers;
    helpers += tanh ? af_tanh : "";
    helpers += gather ? af_gather : "";
    helpers += vec4 ? af_vec4 : "";
    std::string src = head + helpers + funcs;
    src += "\nkernel.def target(@ggml_wave64) export(\"" + sym + "\") @" + sym + "(%w: index) {\n";
    src += "  %one = index.constant 1 : index\n";
    src += "  %wg = index.constant " + std::to_string(parts[0]->wg) + " : index\n";
    src += "  kernel.launch.config workgroups(%w, %one, %one) workgroup_size(%wg, %one, %one) : index\n";
    src += "} launch(" + args + (bargs.empty() ? "" : ", " + bargs) + ") {\n";
    src += "  %group = kernel.workgroup.id<x> : index\n";
    src += "  %item = kernel.workitem.id<x> : index\n";
    src += "  %g0 = index.constant 0 : index\n";
    for (size_t k = 1; k < parts.size(); ++k) {
        src += "  %g" + std::to_string(k) + " = index.assume %g" + std::to_string(k) + "_in [range(%g" + std::to_string(k) + "_in, 1, 2147483647)] : index\n";
    }
    // Nested: part k if group < g(k + 1), else the rest.
    std::string body, tail;
    size_t      pos = 0;
    for (size_t k = 0; k < parts.size(); ++k) {
        const size_t e = calls.find("  // part " + std::to_string(k + 1) + ":", pos);
        const std::string call = calls.substr(pos, e == std::string::npos ? std::string::npos : e - pos);
        pos = e;
        if (k + 1 < parts.size()) {
            body += "  %in" + std::to_string(k) + " = index.cmp ult, %group, %g" + std::to_string(k + 1) + " : index\n";
            body += "  scf.if %in" + std::to_string(k) + " {\n" + call + "  } else {\n";
            tail += "  }\n";
        } else {
            body += call;
        }
    }
    src += body + tail + "  kernel.return\n}\n";
    return src;
}

// Shapes and element strides that the kernel sees. A size-1 dim has stride 0.
struct af_layout {
    int64_t              ne[GGML_MAX_DIMS];
    std::vector<int64_t> tne; // 4 for each binding, then 4 for each output
    std::vector<int64_t> ts;
};

// For the flat kernel, merges the dims that are contiguous in all tensors, and picks the vec4 form if all rows are
// contiguous or broadcast. Sets spec.dims, spec.vec4 and the binding flags.
static af_layout af_make_layout(af_builder & b, const std::vector<af_odata> & od) {
    af_layout l;
    const int n_b = (int) b.bd.size();
    const int n_t = n_b + (int) od.size();
    l.tne.resize(4 * n_t);
    l.ts.resize(4 * n_t);
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        l.ne[k] = b.dom[k];
    }
    for (int t = 0; t < n_t; ++t) {
        const af_dims & ne = t < n_b ? b.bd[t].ne : od[t - n_b].ne;
        const af_dims & ns = t < n_b ? b.bd[t].ns : od[t - n_b].ns;
        for (int k = 0; k < GGML_MAX_DIMS; ++k) {
            l.tne[4 * t + k] = ne[k];
            l.ts[4 * t + k]  = ns[k];
        }
    }
    b.spec.dims = 4;
    if (b.spec.rows) {
        for (int k = 0; k < n_b; ++k) {
            af_sbinding & sb = b.spec.bindings[k];
            sb.vload = b.spec.vec4 && l.ne[0] >= 4 && (sb.mod & 1) == 0 && sb.shift != 0 && !sb.tap && !sb.gather && !sb.pad && l.tne[4 * k] == l.ne[0] &&
                       l.ts[4 * k] == 1 ? 1 : 0;
        }
        for (size_t o = 0; o < od.size(); ++o) {
            const int t = n_b + (int) o;
            b.spec.outs[o].vec = b.spec.vec4 && b.spec.outs[o].mask != 1 && l.tne[4 * t] == l.ne[0] && l.ts[4 * t] == 1 ? 1 : 0;
        }
        return l;
    }
    b.spec.vec4 = false;
    bool flat = true, gather = false;
    for (const af_snode & n : b.spec.nodes) {
        flat = flat && !n.scalar && n.op != AF_CONCAT;
    }
    for (const af_sbinding & sb : b.spec.bindings) {
        gather = gather || sb.gather || sb.pad;
        // The vec4 flat kernel does not add the gathered row to the address.
        flat   = flat && !sb.tap && !sb.pad && !sb.gather;
    }
    for (const af_odata & o : od) {
        gather = gather || o.mask;
    }
    for (const af_sout & o : b.spec.outs) {
        flat = flat && o.mask != 1;
    }
    if (!gather) {
        const int dims = ggml_hrx_collapse_dims(l.ne, n_t, (int64_t (*)[GGML_MAX_DIMS]) l.tne.data(), (int64_t (*)[GGML_MAX_DIMS]) l.ts.data());
        b.spec.dims = dims == 1 ? 1 : dims == 2 ? 2 : 4;
    }
    bool vec4 = flat && l.ne[0] % 4 == 0;
    for (int t = 0; t < n_t && vec4; ++t) {
        const bool is_out = t >= n_b;
        const bool cont   = l.tne[4 * t] == l.ne[0] && l.ts[4 * t] == 1;
        vec4 = cont || (!is_out && l.tne[4 * t] == 1 && b.bd[t].ins[0] == 0);
    }
    b.spec.vec4 = vec4;
    for (int k = 0; k < n_b; ++k) {
        af_sbinding & sb = b.spec.bindings[k];
        sb.mod = 0;
        for (int d = 0; d < GGML_MAX_DIMS && !sb.pad; ++d) {
            if (l.tne[4 * k + d] != 1 && l.tne[4 * k + d] != l.ne[d]) {
                sb.mod |= (uint8_t) (1 << d);
            }
        }
        sb.bc0 = vec4 && l.tne[4 * k] == 1 ? 1 : 0;
    }
    return l;
}

// Launch scalars, in the order of the kernel arguments.
static bool af_scalars(const af_builder & b, const std::vector<af_odata> & od, const af_layout & l, af_region & r) {
    const int64_t rows = l.ne[1] * l.ne[2] * l.ne[3];
    std::vector<int64_t> & s = r.scalars;
    s = { b.spec.rows ? rows : l.ne[0] * rows, l.ne[0], l.ne[1], l.ne[2] };
    const int n_b = (int) b.bd.size();
    for (int k = 0; k < n_b; ++k) {
        const af_bdata &    d  = b.bd[k];
        const af_sbinding & sb = b.spec.bindings[k];
        if (sb.share >= 0) {
            s.push_back(d.off);
            if (sb.gather) {
                s.push_back(d.ioff);
            }
            continue;
        }
        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
            s.push_back(l.ts[4 * k + j]);
        }
        s.push_back((int64_t) (ggml_nbytes(d.t) / 4));
        s.push_back(d.off);
        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
            if (sb.mod & (1 << j)) {
                s.push_back(l.tne[4 * k + j]);
            }
        }
        if (sb.shift >= 0) {
            s.push_back(d.lo);
        }
        if (sb.tap) {
            s.push_back(d.tap);
        }
        if (sb.pad) {
            for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                s.push_back(l.tne[4 * k + j]);
            }
        }
        if (sb.gather) {
            for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                s.push_back(d.ins[j]);
            }
            s.push_back((int64_t) (ggml_nbytes(d.ids) / 4));
            s.push_back(d.ioff);
            s.push_back(d.gs);
            s.push_back(d.gn);
        }
        r.buffers.push_back(d.t);
    }
    for (int k = 0; k < n_b; ++k) {
        if (b.spec.bindings[k].gather && b.spec.bindings[k].share < 0) {
            r.buffers.push_back(b.bd[k].ids);
        }
    }
    for (size_t o = 0; o < od.size(); ++o) {
        const int t = n_b + (int) o;
        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
            s.push_back(l.ts[4 * t + j]);
        }
        s.push_back((int64_t) (ggml_nbytes(od[o].t) / 4));
        s.push_back(od[o].off);
        if (od[o].mask) {
            for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                s.push_back(od[o].lo[j]);
                s.push_back(od[o].n[j]);
            }
        }
        r.buffers.push_back(od[o].t);
        r.outputs.push_back(od[o].t);
    }
    for (size_t n = 0; n < b.spec.nodes.size(); ++n) {
        const ggml_tensor * t = b.ptens[n];
        uint32_t bits[2];
        switch (b.spec.nodes[n].op) {
            case AF_SCALE:
            case AF_CLAMP:
                std::memcpy(bits, t->op_params, sizeof(bits));
                s.push_back(bits[0]);
                s.push_back(bits[1]);
                break;
            case AF_FILL:
            case AF_RMS_NORM:
            case AF_SOFT_MAX:
                std::memcpy(bits, t->op_params, sizeof(bits[0]));
                s.push_back(bits[0]);
                break;
            default:
                break;
        }
    }
    for (int64_t v : s) {
        if (v < 0 || v > INT_MAX) {
            return false;
        }
    }
    for (const ggml_tensor * t : r.buffers) {
        if (!ggml_hrx_tensor_on_device(t)) {
            return false;
        }
    }
    return true;
}

static ggml_hrx_kernel * af_kernel(ggml_backend_hrx_context * ctx, const af_spec & spec) {
    ggml_hrx_autofuse * af  = ctx->autofuse;
    const std::string   sig = spec.key();
    auto it = af->kernels.find(sig);
    if (it != af->kernels.end()) {
        return it->second;
    }
    char sym[64];
    snprintf(sym, sizeof(sym), "ggml_autofuse_%016llx", (unsigned long long) std::hash<std::string>()(spec.signature()));
    af_codegen        gen(spec);
    const std::string source = gen.generate(sym);
    const std::string file   = std::string("autofuse/") + sym;

    ggml_hrx_kernel_desc desc;
    desc.file   = file.c_str();
    desc.symbol = sym;
    desc.source = &source;
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
    if (kernel == nullptr) {
        GGML_LOG_WARN("ggml_hrx: auto-fusion kernel %s does not compile, the region runs unfused:\n%s\n", sym, source.c_str());
    } else if (ctx->trace) {
        fprintf(stderr, "%s\n", source.c_str());
    }
    af->kernels[sig] = kernel;
    return kernel;
}

static void af_dump_nodes(const ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        fprintf(stderr, "afnode %d %s %s [%lld,%lld,%lld,%lld] nb[%zu,%zu,%zu,%zu] u=%d vs=%s |", i, ggml_op_desc(t), t->name, (long long) t->ne[0],
                (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], t->nb[0], t->nb[1], t->nb[2], t->nb[3], af_use_count(cgraph, t),
                t->view_src != nullptr ? t->view_src->name : "-");
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            const ggml_tensor * s = t->src[k];
            if (s != nullptr) {
                fprintf(stderr, " %s(%s)[%lld,%lld,%lld,%lld]", s->name, ggml_op_desc(s), (long long) s->ne[0], (long long) s->ne[1],
                        (long long) s->ne[2], (long long) s->ne[3]);
            }
        }
        fprintf(stderr, "\n");
    }
}

// A region ready to run: alone, or as one part of a kernel with several independent parts.
struct af_part {
    int                              first = 0;
    int                              last  = 0;
    int                              n_ops = 0;
    af_spec                          spec;
    std::vector<int64_t>             scalars;
    std::vector<const ggml_tensor *> buffers;
    std::vector<const ggml_tensor *> inputs;  // tensors that the part reads
    std::vector<const ggml_tensor *> outputs; // tensors that the part writes
    int64_t                          w      = 0; // launch argument of the kernel of this part alone
    int64_t                          groups = 0; // workgroups
    std::string                      label;
};

// Builds the longest region that starts at node i, with at least one op.
static bool af_make_part(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, af_part & p) {
    const ggml_hrx_tune * tune = ctx->device->tune;
    const bool            log  = ctx->trace;
    static thread_local af_builder b;
    std::vector<af_odata> od;
    // A region stops before the nodes of a fusion (ops/fuse.cpp).
    int  last_max = std::min(cgraph->n_nodes, ggml_hrx_fuse_next(ctx, i)) - 1;
    bool ok       = false;
    while (true) {
        af_build(cgraph, i, last_max, b);
        if (b.index.empty()) {
            break;
        }
        if (af_finish(cgraph, i, b, od, 1)) {
            ok = true;
            break;
        }
        if (b.index.size() < 2) {
            break;
        }
        // Try a shorter region.
        last_max = b.index[b.index.size() - 2];
    }
    if (log && b.last >= i) {
        std::string ops;
        for (const ggml_tensor * n : b.tensors) {
            if (n != nullptr) {
                ops += std::string(ops.empty() ? "" : ",") + ggml_op_desc(n);
            }
        }
        const int next = b.last + 1 < cgraph->n_nodes ? b.last + 1 : -1;
        fprintf(stderr, "autofuse: %s [%d..%d] n=%zu %s | stop: %s at %s\n", ok && b.index.size() + b.n_absorbed > 1 ? "region" : "single", i, b.last,
                b.index.size(), ops.c_str(), b.stop, next >= 0 ? ggml_op_desc(cgraph->nodes[next]) : "end");
    }
    if (!ok) {
        return false;
    }

    const int64_t R    = b.dom[0];
    bool          wide = b.spec.rows && R >= AF_WIDE_ROW;
    for (const af_snode & n : b.spec.nodes) {
        wide = wide && n.op != AF_RMS_NORM;
    }
    b.spec.wg    = (int) (wide ? 1024 : b.spec.rows ? tune->norm_wg_size : tune->elementwise_wg_size);
    // Same lane choice as the sum_rows and soft_max kernels. It depends only on R.
    b.spec.lanes = !b.spec.rows ? 0 : R <= 16 ? 16 : R <= 32 ? 32 : R <= 1024 ? 64 : b.spec.wg;
    // With only RMS_NORM reductions, the same lanes and value order as ops/norm.cpp. Then the sums are the same.
    b.spec.vec4 = b.spec.rows;
    for (const af_snode & n : b.spec.nodes) {
        b.spec.vec4 = b.spec.vec4 && (n.op == AF_RMS_NORM || !af_is_reduction(n.op));
    }
    if (b.spec.vec4) {
        if (R % 4 != 0 || R > tune->rms_norm_rows_max) {
            b.spec.lanes = b.spec.wg;
        } else {
            int64_t lanes = 16;
            while (lanes < tune->rms_norm_lanes_max && 4 * lanes < R) {
                lanes *= 2;
            }
            b.spec.lanes = (int) lanes;
        }
    }
    const int64_t rows = b.dom[1] * b.dom[2] * b.dom[3];
    const int64_t rpg  = !b.spec.rows || b.spec.lanes == b.spec.wg ? 1 : b.spec.wg / b.spec.lanes;
    // Few rows (decode): unroll the chunks of a lane and keep their values in registers between the loops.
    b.spec.nv = 0;
    if (b.spec.vec4 && R % 4 == 0) {
        const int64_t nv = (R + 4 * b.spec.lanes - 1) / (4 * b.spec.lanes);
        if (nv <= 8 && (rows + rpg - 1) / rpg <= AF_UNROLL_MAX_GROUPS) {
            b.spec.nv = (int) nv;
        }
    }

    const af_layout l = af_make_layout(b, od);
    af_region r;
    if (!af_scalars(b, od, l, r)) {
        return false;
    }
    p.first   = i;
    p.last    = b.last;
    p.n_ops   = (int) b.index.size() + b.n_absorbed;
    p.spec    = b.spec;
    p.scalars = std::move(r.scalars);
    p.buffers = std::move(r.buffers);
    p.outputs = std::move(r.outputs);
    for (const af_bdata & d : b.bd) {
        p.inputs.push_back(d.t);
        if (d.ids != nullptr) {
            p.inputs.push_back(d.ids);
        }
    }
    // The flat kernel sees the merged dims of the layout.
    if (b.spec.rows) {
        p.w      = (rows + rpg - 1) / rpg;
        p.groups = p.w;
    } else {
        const int64_t per = b.spec.vec4 ? 4 * b.spec.wg : b.spec.wg;
        p.w      = l.ne[0] * l.ne[1] * l.ne[2] * l.ne[3];
        p.groups = (p.w + per - 1) / per;
    }
    if (ctx->prof != nullptr) {
        p.label = "n=" + std::to_string(b.index.size()) + " ";
        bool first = true;
        for (const ggml_tensor * n : b.tensors) {
            if (n != nullptr) {
                p.label += std::string(first ? "" : ",") + ggml_op_desc(n);
                first = false;
            }
        }
    }
    return true;
}

// Part p can run in the same dispatch as the parts in group: it comes right after them and shares no memory that one
// of them writes.
static bool af_can_join(const ggml_cgraph * g, const std::vector<af_part> & group, const af_part & p) {
    if (group.size() >= 4 || p.spec.wg != group[0].spec.wg) {
        return false;
    }
    for (int k = group.back().last + 1; k < p.first; ++k) {
        const ggml_tensor * t = g->nodes[k];
        if (!ggml_op_is_empty(t->op) && !ggml_is_empty(t) && (t->flags & GGML_TENSOR_FLAG_COMPUTE) != 0) {
            return false;
        }
    }
    for (const af_part & q : group) {
        for (const ggml_tensor * o : q.outputs) {
            for (const ggml_tensor * t : p.inputs) {
                if (af_overlap(o, t)) {
                    return false;
                }
            }
            for (const ggml_tensor * t : p.outputs) {
                if (af_overlap(o, t)) {
                    return false;
                }
            }
        }
        for (const ggml_tensor * o : p.outputs) {
            for (const ggml_tensor * t : q.inputs) {
                if (af_overlap(o, t)) {
                    return false;
                }
            }
        }
    }
    return true;
}

static ggml_hrx_kernel * af_kernel_parts(ggml_backend_hrx_context * ctx, const std::vector<af_part> & group) {
    ggml_hrx_autofuse * af = ctx->autofuse;
    std::string         key = "parts";
    std::string         sig;
    std::vector<const af_spec *> specs;
    for (const af_part & p : group) {
        key += "|" + p.spec.key();
        sig += "|" + p.spec.signature();
        specs.push_back(&p.spec);
    }
    auto it = af->kernels.find(key);
    if (it != af->kernels.end()) {
        return it->second;
    }
    char sym[64];
    snprintf(sym, sizeof(sym), "ggml_autofuse_%016llx", (unsigned long long) std::hash<std::string>()(sig));
    const std::string source = af_generate_parts(specs, sym);
    const std::string file   = std::string("autofuse/") + sym;
    ggml_hrx_kernel_desc desc;
    desc.file   = file.c_str();
    desc.symbol = sym;
    desc.source = &source;
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
    if (kernel == nullptr) {
        GGML_LOG_WARN("ggml_hrx: auto-fusion kernel %s does not compile, the parts run alone:\n%s\n", sym, source.c_str());
    } else if (ctx->trace) {
        fprintf(stderr, "%s\n", source.c_str());
    }
    af->kernels[key] = kernel;
    return kernel;
}

static void af_add_region(af_plan & plan, af_region && r) {
    plan.start[r.first] = (int) plan.regions.size();
    plan.regions.push_back(std::move(r));
}

// Adds the regions of a group of parts to the plan: one dispatch for several parts, else each part with 2 or more ops.
static void af_flush(ggml_backend_hrx_context * ctx, af_plan & plan, std::vector<af_part> & group) {
    if (group.size() > 1) {
        af_region r;
        r.kernel = af_kernel_parts(ctx, group);
        if (r.kernel != nullptr) {
            r.first = group.front().first;
            r.last  = group.back().last;
            int64_t g0 = 0;
            for (size_t k = 0; k < group.size(); ++k) {
                if (k > 0) {
                    r.scalars.push_back(g0);
                }
                r.scalars.insert(r.scalars.end(), group[k].scalars.begin(), group[k].scalars.end());
                g0 += group[k].groups;
            }
            for (const af_part & p : group) {
                r.buffers.insert(r.buffers.end(), p.buffers.begin(), p.buffers.end());
                r.outputs.insert(r.outputs.end(), p.outputs.begin(), p.outputs.end());
                r.label += (r.label.empty() ? "AUTOFUSED " : " + ") + p.label;
            }
            r.workload = { g0 };
            bool ok = g0 <= INT_MAX;
            for (int64_t v : r.scalars) {
                ok = ok && v >= 0 && v <= INT_MAX;
            }
            if (ok) {
                af_add_region(plan, std::move(r));
                return;
            }
        }
    }
    for (af_part & p : group) {
        if (p.n_ops < 2) {
            continue;
        }
        af_region r;
        r.kernel = af_kernel(ctx, p.spec);
        if (r.kernel == nullptr) {
            continue;
        }
        r.first    = p.first;
        r.last     = p.last;
        r.scalars  = std::move(p.scalars);
        r.buffers  = std::move(p.buffers);
        r.outputs  = std::move(p.outputs);
        r.workload = { p.w };
        r.label    = p.label.empty() ? "" : "AUTOFUSED " + p.label;
        af_add_region(plan, std::move(r));
    }
}

void ggml_hrx_autofuse_plan(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph) {
    if (ctx->autofuse == nullptr) {
        ctx->autofuse = new ggml_hrx_autofuse;
    }
    ggml_hrx_autofuse * af = ctx->autofuse;
    af->cur = nullptr;
    if (cgraph->n_nodes == 0) {
        return;
    }
    static thread_local std::vector<uint8_t> fp;
    af_fingerprint(cgraph, fp);
    const std::pair<const void *, int> key = { cgraph->nodes[0], cgraph->n_nodes };
    auto it = af->plans.find(key);
    if (it != af->plans.end() && it->second.fp == fp) {
        af->cur = &it->second;
        return;
    }
    if (it == af->plans.end() && af->plans.size() >= 16) {
        af->plans.clear();
    }
    af_plan & plan = af->plans[key];
    plan.fp = fp;
    plan.start.assign(cgraph->n_nodes, -1);
    plan.regions.clear();
    af->cur = &plan;
    if (ctx->trace) {
        af_dump_nodes(cgraph);
    }

    std::vector<af_part> group;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        if (ggml_op_is_empty(t->op) || af_op_of(t) < 0 || ggml_hrx_fuse_next(ctx, i) == i) {
            continue;
        }
        af_part p;
        if (!af_make_part(ctx, cgraph, i, p)) {
            continue;
        }
        // Independent neighbours run in one dispatch, each on its own workgroups.
        if (!group.empty() && af_can_join(cgraph, group, p)) {
            group.push_back(std::move(p));
        } else {
            af_flush(ctx, plan, group);
            group.clear();
            group.push_back(std::move(p));
        }
        i = group.back().last;
    }
    af_flush(ctx, plan, group);
}

int ggml_hrx_autofuse_run(ggml_backend_hrx_context * ctx, int i, const char ** label) {
    const af_plan * plan = ctx->autofuse != nullptr ? ctx->autofuse->cur : nullptr;
    if (plan == nullptr || i >= (int) plan->start.size() || plan->start[i] < 0) {
        return -1;
    }
    const af_region & r = plan->regions[plan->start[i]];
    ctx->dispatch_writes = &r.outputs;
    const bool ok = ggml_hrx_dispatch(ctx, r.kernel, r.workload, r.scalars, r.buffers);
    ctx->dispatch_writes = nullptr;
    if (!ok) {
        GGML_ABORT("%s: auto-fused region at node %d failed", __func__, i);
    }
    *label = r.label.c_str();
    return r.last;
}

void ggml_hrx_autofuse_free(ggml_hrx_autofuse * af) {
    delete af;
}
