import sys, re
root = sys.argv[1]
def sub(path, old, new, count=1):
    p = root + '/' + path
    s = open(p).read()
    n = s.count(old)
    if n != count:
        sys.exit(f"{path}: expected {count} got {n} for: {old[:80]!r}")
    s = s.replace(old, new)
    open(p, 'w').write(s)

A = 'ggml/src/ggml-alloc.c'
sub(A, '#include <string.h>\n', '''#include <string.h>
#include <unistd.h>

// C15 instrumentation (GGML_C15=1): planned buffer sizes, the tensor that sets each chunk top, realloc reasons
static int ggml_c15_on(void) { static int v = -1; if (v < 0) { const char * e = getenv("GGML_C15"); v = e ? atoi(e) : 0; } return v; }
''')
sub(A, '''struct tallocr_chunk {
    struct free_block free_blocks[MAX_FREE_BLOCKS];
    int n_free_blocks;
    size_t max_size;
};''', '''struct tallocr_chunk {
    struct free_block free_blocks[MAX_FREE_BLOCKS];
    int n_free_blocks;
    size_t max_size;
    char top_name[GGML_MAX_NAME];
    int top_op;
    int64_t top_ne[4];
    int top_type;
    size_t top_off, top_size;
};''')
sub(A, '''    chunk->max_size = MAX(chunk->max_size, addr.offset + size);

    return addr;''', '''    if (tensor && addr.offset + size > chunk->max_size) {
        snprintf(chunk->top_name, sizeof(chunk->top_name), "%s", tensor->name);
        chunk->top_op = tensor->op;
        chunk->top_type = tensor->type;
        for (int k = 0; k < 4; k++) { chunk->top_ne[k] = tensor->ne[k]; }
        chunk->top_off = addr.offset;
        chunk->top_size = size;
    }
    chunk->max_size = MAX(chunk->max_size, addr.offset + size);

    return addr;''')
sub(A, '''    // set the node_allocs from the hash table
    if (galloc->n_nodes < graph->n_nodes) {''', '''    if (ggml_c15_on()) {
        static int c15_seq = 0;
        const int seq = c15_seq++;
        bool grow = false;
        for (int i = 0; i < galloc->n_buffers; i++) {
            bool seen = false;
            for (int j = 0; j < i; j++) { seen |= galloc->buf_tallocs[j] == galloc->buf_tallocs[i]; }
            if (seen) { continue; }
            size_t planned = 0;
            for (int c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) { planned += ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i], c); }
            const size_t cur = galloc->buffers[i] ? ggml_vbuffer_size(galloc->buffers[i]) : 0;
            grow |= planned > cur;
            GGML_LOG_INFO("C15A plan #%d galloc=%p nodes=%d leafs=%d no_alloc=%d buf=%s planned=%zu (%.4f MiB) cur=%zu (%.4f MiB)%s\\n",
                    seq, (void *) galloc, graph->n_nodes, graph->n_leafs, (int) no_alloc, ggml_backend_buft_name(galloc->bufts[i]),
                    planned, planned/1048576.0, cur, cur/1048576.0, planned > cur ? " GROW" : "");
            for (int c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
                const struct tallocr_chunk * ch = galloc->buf_tallocs[i]->chunks[c];
                GGML_LOG_INFO("C15A   top #%d %s chunk %d: '%s' op=%s type=%s ne=[%lld,%lld,%lld,%lld] off=%zu size=%zu end=%zu\\n",
                        seq, ggml_backend_buft_name(galloc->bufts[i]), c, ch->top_name, ggml_op_name((enum ggml_op) ch->top_op),
                        ggml_type_name((enum ggml_type) ch->top_type),
                        (long long) ch->top_ne[0], (long long) ch->top_ne[1], (long long) ch->top_ne[2], (long long) ch->top_ne[3],
                        ch->top_off, ch->top_size, ch->top_off + ch->top_size);
            }
        }
        const char * dir = getenv("GGML_C15_DUMP");
        const char * all = getenv("GGML_C15_DUMP_ALL");
        if (dir && (grow || (all && atoi(all)))) {
            char path[512];
            snprintf(path, sizeof(path), "%s/plan_%d_%04d.tsv", dir, (int) getpid(), seq);
            FILE * f = fopen(path, "w");
            if (f) {
                fprintf(f, "#galloc=%p nodes=%d leafs=%d\\n", (void *) galloc, graph->n_nodes, graph->n_leafs);
                for (int pass = 0; pass < 2; pass++) {
                    const int n = pass == 0 ? graph->n_nodes : graph->n_leafs;
                    for (int i = 0; i < n; i++) {
                        struct ggml_tensor * t = pass == 0 ? graph->nodes[i] : graph->leafs[i];
                        if (t->view_src || t->data) { continue; }
                        struct hash_node * hn = ggml_gallocr_hash_get(galloc, t);
                        if (hn->buffer_id < 0) { continue; }
                        fprintf(f, "%c\\t%d\\t%s\\t%s\\t%s\\t%d\\t%zu\\t%zu\\t%lld,%lld,%lld,%lld\\n", pass == 0 ? 'N' : 'L', i, t->name,
                                ggml_op_name(t->op), ggml_type_name(t->type), hn->buffer_id, hn->addr.offset,
                                ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], t),
                                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
                    }
                }
                fclose(f);
                GGML_LOG_INFO("C15A dump #%d -> %s\\n", seq, path);
            }
        }
    }

    // set the node_allocs from the hash table
    if (galloc->n_nodes < graph->n_nodes) {''')
# realloc reasons
sub(A, '''static bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (galloc->n_nodes != graph->n_nodes) {''', '''static bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (galloc->n_nodes != graph->n_nodes) {
        if (ggml_c15_on()) { GGML_LOG_INFO("C15R galloc=%p n_nodes %d -> %d\\n", (void *) galloc, galloc->n_nodes, graph->n_nodes); }''')
sub(A, '''    if (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG''', '''    if (galloc->n_leafs != graph->n_leafs) {
        if (ggml_c15_on()) { GGML_LOG_INFO("C15R galloc=%p n_leafs %d -> %d\\n", (void *) galloc, galloc->n_leafs, graph->n_leafs); }
#ifndef NDEBUG''')
sub(A, '''        if (!ggml_gallocr_node_needs_realloc(galloc, node, &node_alloc->dst)) {
#ifndef NDEBUG''', '''        if (!ggml_gallocr_node_needs_realloc(galloc, node, &node_alloc->dst)) {
            if (ggml_c15_on()) { GGML_LOG_INFO("C15R galloc=%p node %d '%s' op=%s size %zu > size_max %zu\\n", (void *) galloc, i, node->name, ggml_op_name(node->op),
                    node_alloc->dst.buffer_id >= 0 ? ggml_backend_buft_get_alloc_size(galloc->bufts[node_alloc->dst.buffer_id], node) : 0, node_alloc->dst.size_max); }
#ifndef NDEBUG''')
sub(A, '''            if (!ggml_gallocr_node_needs_realloc(galloc, src, &node_alloc->src[j])) {
#ifndef NDEBUG''', '''            if (!ggml_gallocr_node_needs_realloc(galloc, src, &node_alloc->src[j])) {
                if (ggml_c15_on()) { GGML_LOG_INFO("C15R galloc=%p node %d '%s' src %d '%s' size %zu > size_max %zu\\n", (void *) galloc, i, node->name, j, src->name,
                        node_alloc->src[j].buffer_id >= 0 ? ggml_backend_buft_get_alloc_size(galloc->bufts[node_alloc->src[j].buffer_id], src) : 0, node_alloc->src[j].size_max); }
#ifndef NDEBUG''')

C = 'src/llama-context.cpp'
sub(C, '''    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);''', '''    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);
    static const bool c15 = getenv("GGML_C15") && atoi(getenv("GGML_C15")) != 0;
    if (c15) {
        LLAMA_LOG_INFO("C15L graph_reserve ctx=%p ctx_type=%d n_tokens=%u n_seqs=%u n_outputs=%u split_only=%d sizes=%d\\n",
                (void *) this, (int) cparams.ctx_type, n_tokens, n_seqs, n_outputs, (int) split_only, sizes != nullptr);
    }''')
sub(C, '''    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\\n", __func__, status);''', '''    static const bool c15 = getenv("GGML_C15") && atoi(getenv("GGML_C15")) != 0;
    if (c15) {
        std::string sz;
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            const size_t a = ggml_backend_sched_get_buffer_size(sched.get(), backend_ptrs[i]);
            sz += format(" %s=%zu(exp %zu)%s", ggml_backend_buft_name(backend_buft[i]), a, backend_buf_exp_size[i], a > backend_buf_exp_size[i] ? "!OVER" : "");
        }
        llama_seq_id s0 = ubatch.n_seq_id && ubatch.n_seq_id[0] > 0 ? ubatch.seq_id[0][0] : -1;
        LLAMA_LOG_INFO("C15L ubatch ctx=%p ctx_type=%d gtype=%d n_tokens=%u n_seqs=%u n_seq_tokens=%u n_seqs_unq=%u n_outputs=%d token=%d embd=%d pos=[%d..%d] seq0=%d reused=%d nodes=%d%s\\n",
                (void *) this, (int) cparams.ctx_type, (int) gtype, ubatch.n_tokens, ubatch.n_seqs, ubatch.n_seq_tokens, ubatch.n_seqs_unq,
                (int) n_outputs, ubatch.token != nullptr, ubatch.embd != nullptr,
                ubatch.pos ? ubatch.pos[0] : -1, ubatch.pos ? ubatch.pos[ubatch.n_tokens-1] : -1, s0, (int) reused,
                ggml_graph_n_nodes(res->get_gf()), sz.c_str());
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\\n", __func__, status);''')

Q = 'src/models/qwen4exp.cpp'
sub(Q, '''        if (shortcut && n_kv <= width) {
            build_qsa_store_k(mctx_hyb, cur, il);
        } else {''', '''        static const bool c15 = getenv("GGML_C15") && atoi(getenv("GGML_C15")) != 0;
        if (c15) {
            LLAMA_LOG_INFO("C15Q layer il=%d r=%lld n_kv=%lld width=%lld route=%s q_first=%lld n_tokens=%lld n_outputs=%lld\\n", il, (long long) r, (long long) n_kv, (long long) width,
                    shortcut && n_kv <= width ? "dense-shortcut" : "selection", (long long) q_first, (long long) n_tokens, (long long) n_outputs);
        }
        if (shortcut && n_kv <= width) {
            build_qsa_store_k(mctx_hyb, cur, il);
        } else {''')
sub(Q, '''        inp = qsa.get();
        res->add_input(std::move(qsa));
        qsa_inps.emplace((uint32_t) r, inp);''', '''        static const bool c15 = getenv("GGML_C15") && atoi(getenv("GGML_C15")) != 0;
        if (c15) {
            LLAMA_LOG_INFO("C15Q inputs il=%d r=%lld n_kv=%lld n_blocks=%lld n_stream=%lld n_tps=%lld blk_bias=%d blocks=%d scalar=%d strip=%lld key_limits=%zu bias=%s[%lld,%lld,%lld] tail=%d pos0=%d posN=%d\\n",
                    il, (long long) r, (long long) n_kv, (long long) n_blocks, (long long) n_stream, (long long) n_tps, (int) blk_bias, (int) blocks, (int) scalar,
                    (long long) qsa->score_strip, qsa->score_key_limits.size(), ggml_type_name(qsa->bias->type),
                    (long long) qsa->bias->ne[0], (long long) qsa->bias->ne[1], (long long) qsa->bias->ne[2], (int) (qsa->tail_idxs != nullptr),
                    ubatch.pos ? ubatch.pos[0] : -1, ubatch.pos ? ubatch.pos[ubatch.n_tokens-1] : -1);
        }
        inp = qsa.get();
        res->add_input(std::move(qsa));
        qsa_inps.emplace((uint32_t) r, inp);''')
sub(Q, '''    const int64_t prep_blocks = fast_cache ? (n_tokens+3)/4+2 : n_blocks;''', '''    const int64_t prep_blocks = fast_cache ? (n_tokens+3)/4+2 : n_blocks;
    {
        static const bool c15 = getenv("GGML_C15") && atoi(getenv("GGML_C15")) != 0;
        if (c15) {
            LLAMA_LOG_INFO("C15Q cache il=%d storage=%d cached=%d fast=%d prep_blocks=%lld\\n", il, storage != nullptr, cached != nullptr, (int) fast_cache, (long long) prep_blocks);
        }
    }''')
print("ok")
