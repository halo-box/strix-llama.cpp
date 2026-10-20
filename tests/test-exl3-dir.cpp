// EXL3 directory importer, common/exl3-dir.cpp
//
// test-exl3-dir                            a small synthetic qwen35 EXL3 directory is written, imported and run
// test-exl3-dir --compare DIR CONT.gguf    the import of DIR against the converter's container of it: tensors, bytes, keys
// test-exl3-dir --dump DIR OUT.gguf        write the import as a GGUF file
//
// --compare hashes both sides with the same code: sha256 over the tensors sorted by name (name, type, ne, sha256 of the
// data), and sha256 over the keys sorted by name except the general.* provenance keys, which are listed apart

#include "exl3-dir.h"

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

extern "C" {
#include "hash/sha256/sha256.h"
}

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::string to_hex(const unsigned char * d, size_t n) {
    static const char digits[] = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) {
        s += digits[d[i] >> 4];
        s += digits[d[i] & 15];
    }
    return s;
}

struct sha256 {
    sha256_t s;
    sha256() { sha256_init(&s); }
    void add(const void * p, size_t n) { sha256_update(&s, (const unsigned char *) p, n); }
    template <typename T> void add_val(T v) { add(&v, sizeof(v)); }
    void add_str(const std::string & v) { add_val((uint64_t) v.size()); add(v.data(), v.size()); }
    std::string hex() {
        unsigned char d[SHA256_DIGEST_SIZE];
        sha256_final(&s, d);
        return to_hex(d, sizeof(d));
    }
};

static size_t gguf_elem_size(gguf_type t) {
    switch (t) {
        case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL:    return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:                         return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default:                                                             return 0;
    }
}

// the value of a key: type, then the value; strings with their length
static std::string kv_value(const gguf_context * g, int64_t i) {
    sha256 h;
    const gguf_type t = gguf_get_kv_type(g, i);
    h.add_val((int32_t) t);
    if (t == GGUF_TYPE_STRING) {
        h.add_str(gguf_get_val_str(g, i));
    } else if (t == GGUF_TYPE_ARRAY) {
        const gguf_type at = gguf_get_arr_type(g, i);
        const size_t    n  = gguf_get_arr_n(g, i);
        h.add_val((int32_t) at);
        h.add_val((uint64_t) n);
        if (at == GGUF_TYPE_STRING) {
            for (size_t j = 0; j < n; j++) {
                h.add_str(gguf_get_arr_str(g, i, j));
            }
        } else {
            h.add(gguf_get_arr_data(g, i), n*gguf_elem_size(at));
        }
    } else {
        h.add(gguf_get_val_data(g, i), gguf_elem_size(t));
    }
    return h.hex();
}

static bool kv_is_provenance(const std::string & key) {
    return key.rfind("general.", 0) == 0 && key != "general.architecture" && key != "general.type" &&
           key != "general.file_type" && key != "general.quantization_version" && key != "general.alignment";
}

struct structure {
    std::map<std::string, std::string> kv;         // key -> hash of the value
    std::map<std::string, std::string> provenance; // key -> value as text
    struct tensor {
        ggml_type   type;
        int64_t     ne[4];
        std::string sha;
    };
    std::map<std::string, tensor> tensors;
    std::vector<std::string>      order;
    std::string                   kv_hash;
    std::string                   tensor_hash;
};

static structure read_structure(const gguf_context * g, const std::function<void(int64_t, std::vector<uint8_t> &)> & data) {
    structure s;
    for (int64_t i = 0; i < gguf_get_n_kv(g); i++) {
        const std::string key = gguf_get_key(g, i);
        if (kv_is_provenance(key)) {
            s.provenance[key] = gguf_get_kv_type(g, i) == GGUF_TYPE_STRING ? gguf_get_val_str(g, i) : "(" + std::string(gguf_type_name(gguf_get_kv_type(g, i))) + ")";
        } else {
            s.kv[key] = kv_value(g, i);
        }
    }
    std::vector<uint8_t> buf;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        structure::tensor t;
        t.type = gguf_get_tensor_type(g, i);
        memcpy(t.ne, gguf_get_tensor_ne(g, i), sizeof(t.ne));
        buf.resize(gguf_get_tensor_size(g, i));
        data(i, buf);
        unsigned char d[SHA256_DIGEST_SIZE];
        sha256_hash(d, buf.data(), buf.size());
        t.sha = to_hex(d, sizeof(d));
        s.tensors[gguf_get_tensor_name(g, i)] = t;
        s.order.push_back(gguf_get_tensor_name(g, i));
    }
    sha256 hk;
    for (const auto & [k, v] : s.kv) {
        hk.add_str(k);
        hk.add_str(v);
    }
    s.kv_hash = hk.hex();
    sha256 ht;
    for (const auto & [name, t] : s.tensors) {
        ht.add_str(name);
        ht.add_val((int32_t) t.type);
        for (int j = 0; j < 4; j++) {
            ht.add_val(t.ne[j]);
        }
        ht.add_str(t.sha);
    }
    s.tensor_hash = ht.hex();
    return s;
}

static int compare(const std::string & dir, const std::string & container) {
    ggml_context * ctx = nullptr;
    gguf_init_params params = { true, &ctx };
    gguf_context * gc = gguf_init_from_file(container.c_str(), params);
    if (!gc) {
        fprintf(stderr, "cannot read %s\n", container.c_str());
        return 1;
    }
    std::ifstream f(container, std::ios::binary);
    const structure a = read_structure(gc, [&](int64_t i, std::vector<uint8_t> & buf) {
        f.seekg((std::streamoff) (gguf_get_data_offset(gc) + gguf_get_tensor_offset(gc, i)));
        if (!f.read((char *) buf.data(), (std::streamsize) buf.size())) {
            throw std::runtime_error("cannot read the container");
        }
    });

    const common_exl3_dir_ptr d = common_exl3_dir_open(dir);
    const gguf_context * gd = common_exl3_dir_gguf(d.get());
    const structure b = read_structure(gd, [&](int64_t i, std::vector<uint8_t> & buf) {
        common_exl3_dir_read(d.get(), gguf_get_tensor_name(gd, i), buf.data(), buf.size());
    });

    int n_diff = 0;
    for (const auto & [k, v] : a.kv) {
        if (!b.kv.count(k)) {
            printf("key only in the container: %s\n", k.c_str());
            n_diff++;
        } else if (b.kv.at(k) != v) {
            printf("key differs: %s\n", k.c_str());
            n_diff++;
        }
    }
    for (const auto & [k, v] : b.kv) {
        if (!a.kv.count(k)) {
            printf("key only in the import: %s\n", k.c_str());
            n_diff++;
        }
    }
    for (const auto & [name, t] : a.tensors) {
        if (!b.tensors.count(name)) {
            printf("tensor only in the container: %s\n", name.c_str());
            n_diff++;
            continue;
        }
        const auto & u = b.tensors.at(name);
        if (t.type != u.type || memcmp(t.ne, u.ne, sizeof(t.ne)) != 0) {
            printf("tensor type or shape differs: %s\n", name.c_str());
            n_diff++;
        } else if (t.sha != u.sha) {
            printf("tensor data differ: %s\n", name.c_str());
            n_diff++;
        }
    }
    for (const auto & [name, t] : b.tensors) {
        if (!a.tensors.count(name)) {
            printf("tensor only in the import: %s\n", name.c_str());
            n_diff++;
        }
    }
    for (const auto & [k, v] : a.provenance) {
        printf("provenance, container: %s = %s\n", k.c_str(), v.c_str());
    }
    for (const auto & [k, v] : b.provenance) {
        printf("provenance, import:    %s = %s\n", k.c_str(), v.c_str());
    }
    printf("container: %zu tensors sha256 %s, %zu keys sha256 %s\n", a.tensors.size(), a.tensor_hash.c_str(), a.kv.size(), a.kv_hash.c_str());
    printf("import:    %zu tensors sha256 %s, %zu keys sha256 %s\n", b.tensors.size(), b.tensor_hash.c_str(), b.kv.size(), b.kv_hash.c_str());
    printf("tensor order equal: %s\n", a.order == b.order ? "yes" : "no");
    printf("%s (%d differences)\n", n_diff == 0 ? "IDENTICAL" : "DIFFERENT", n_diff);

    gguf_free(gc);
    ggml_free(ctx);
    return n_diff == 0 ? 0 : 1;
}

static int dump(const std::string & dir, const std::string & out) {
    const common_exl3_dir_ptr d = common_exl3_dir_open(dir);
    const gguf_context * g = common_exl3_dir_gguf(d.get());
    std::vector<uint8_t> meta(gguf_get_meta_size(g));
    gguf_get_meta_data(g, meta.data());
    FILE * f = fopen(out.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", out.c_str());
        return 1;
    }
    const size_t align      = gguf_get_alignment(g);
    const size_t data_start = (meta.size() + align - 1)/align*align;
    bool ok = fwrite(meta.data(), 1, meta.size(), f) == meta.size();
    size_t pos = meta.size();
    std::vector<uint8_t> buf;
    for (int64_t i = 0; ok && i < gguf_get_n_tensors(g); i++) {
        const size_t offs = data_start + gguf_get_tensor_offset(g, i);
        const std::vector<uint8_t> pad(offs - pos, 0);
        buf.resize(gguf_get_tensor_size(g, i));
        common_exl3_dir_read(d.get(), gguf_get_tensor_name(g, i), buf.data(), buf.size());
        ok = fwrite(pad.data(), 1, pad.size(), f) == pad.size() && fwrite(buf.data(), 1, buf.size(), f) == buf.size();
        pos = offs + buf.size();
    }
    ok = fclose(f) == 0 && ok;
    printf("%s: %" PRId64 " tensors, %zu bytes\n", out.c_str(), gguf_get_n_tensors(g), pos);
    return ok ? 0 : 1;
}

// a safetensors file; the header is padded so that the data start at 2 mod 8, so every tensor starts at 2 mod 4 as in real EXL3 files
struct st_file {
    struct entry {
        std::string          name;
        std::string          dtype;
        std::vector<int64_t> shape;
        std::vector<uint8_t> data;
    };
    std::vector<entry> tensors;

    void add(const std::string & name, const std::string & dtype, const std::vector<int64_t> & shape, const std::vector<uint8_t> & data) {
        tensors.push_back({ name, dtype, shape, data });
    }

    void write(const fs::path & path) const {
        std::string hdr = "{";
        uint64_t offs = 0;
        for (size_t i = 0; i < tensors.size(); i++) {
            const entry & t = tensors[i];
            hdr += std::string(i ? ", " : "") + "\"" + t.name + "\": {\"dtype\": \"" + t.dtype + "\", \"shape\": [";
            for (size_t j = 0; j < t.shape.size(); j++) {
                hdr += (j ? ", " : "") + std::to_string(t.shape[j]);
            }
            hdr += "], \"data_offsets\": [" + std::to_string(offs) + ", " + std::to_string(offs + t.data.size()) + "]}";
            offs += t.data.size();
        }
        hdr += "}";
        while ((8 + hdr.size()) % 8 != 2) {
            hdr += ' ';
        }
        std::ofstream f(path, std::ios::binary);
        for (int i = 0; i < 8; i++) {
            f.put((char) ((uint64_t) hdr.size() >> 8*i));
        }
        f << hdr;
        for (const entry & t : tensors) {
            f.write((const char *) t.data.data(), t.data.size());
        }
    }
};

static uint32_t rng_state = 1;

static float rnd(float lo, float hi) {
    rng_state = rng_state*1664525u + 1013904223u;
    return lo + (hi - lo)*(float) (rng_state >> 8)/(float) (1u << 24);
}

static std::vector<uint8_t> rnd_data(const std::string & dtype, int64_t n, float lo, float hi) {
    std::vector<uint8_t> out;
    for (int64_t i = 0; i < n; i++) {
        const float x = rnd(lo, hi);
        if (dtype == "F32") {
            out.insert(out.end(), (const uint8_t *) &x, (const uint8_t *) &x + 4);
        } else if (dtype == "F16") {
            const ggml_fp16_t h = ggml_fp32_to_fp16(x);
            out.insert(out.end(), (const uint8_t *) &h, (const uint8_t *) &h + 2);
        } else if (dtype == "BF16") {
            const ggml_bf16_t h = ggml_fp32_to_bf16(x);
            out.insert(out.end(), (const uint8_t *) &h, (const uint8_t *) &h + 2);
        } else {
            const uint16_t h = (uint16_t) (rnd(0, 65536));
            out.insert(out.end(), (const uint8_t *) &h, (const uint8_t *) &h + 2);
        }
    }
    return out;
}

static void add_exl3(st_file & f, const std::string & base, int64_t k, int64_t n, int64_t K) {
    f.add(base + ".trellis", "I16", { k/16, n/16, 16*K }, rnd_data("I16", k/16*n/16*16*K, 0, 0));
    f.add(base + ".suh", "F16", { k }, rnd_data("F16", k, -1.0f, 1.0f));
    f.add(base + ".svh", "F16", { n }, rnd_data("F16", n, -0.05f, 0.05f));
    const uint32_t mcg = 0xCBAC1FED;
    f.add(base + ".mcg", "I32", {}, std::vector<uint8_t>((const uint8_t *) &mcg, (const uint8_t *) &mcg + 4));
}

static int n_fail = 0;

static void check(bool ok, const char * what) {
    printf("  %s: %s\n", ok ? "ok  " : "FAIL", what);
    n_fail += !ok;
}

// [K, V, unit] -> [V, K, unit]: the old index of new index p of a range of n_heads*unit elements
static int64_t v_head_src(int64_t p, int64_t K, int64_t V, int64_t unit) {
    const int64_t head = p / unit;
    return ((head % K)*V + head / K)*unit + p % unit;
}

// a qwen35 EXL3 checkpoint of 3 linear attention layers and 1 full attention layer, written in three files: two shards
// with an index, and lm_head.safetensors that the index does not list; the config declares an MTP layer the files do not
// have. With mtp the files also hold that MTP layer; with rows the token embedding is a row-ring table (K = 4, one group)
static void write_synthetic(const fs::path & dir, std::map<std::string, st_file::entry> & src, int64_t & n_params,
        bool mtp = false, bool rows = false) {
    const int64_t n_embd = 256, n_ff = 256, n_vocab = 384, head_dim = 256, n_head = 2, n_head_kv = 1;
    const int64_t n_kh = 2, n_vh = 4, head_k = 128, head_v = 128;
    fs::create_directories(dir);

    st_file s1, s2, head;
    const std::string m = "model.language_model.";
    if (rows) {
        // rings of 1 + 16*4 words: an fp16 scale, then the stream; signs +/-1
        std::vector<uint8_t> table = rnd_data("I16", n_vocab*65, 0, 0);
        for (int64_t r = 0; r < n_vocab; r++) {
            const ggml_fp16_t sc = ggml_fp32_to_fp16(rnd(0.01f, 0.02f));
            memcpy(table.data() + 2*r*65, &sc, 2);
        }
        std::vector<uint8_t> signs;
        for (int i = 0; i < 256; i++) {
            const ggml_fp16_t h = ggml_fp32_to_fp16(rnd(0, 1) < 0.5f ? -1.0f : 1.0f);
            signs.insert(signs.end(), (const uint8_t *) &h, (const uint8_t *) &h + 2);
        }
        s1.add(m + "embed_tokens.trellis", "I16", { n_vocab, 65 }, table);
        s1.add(m + "embed_tokens.signs",   "F16", { 1, 256 }, signs);
    } else {
        s1.add(m + "embed_tokens.weight", "F16", { n_vocab, n_embd }, rnd_data("F16", n_vocab*n_embd, -0.05f, 0.05f));
    }
    s2.add(m + "norm.weight", "BF16", { n_embd }, rnd_data("BF16", n_embd, -0.1f, 0.1f));
    for (int il = 0; il < 4; il++) {
        st_file & f = il < 2 ? s1 : s2;
        const std::string p = m + "layers." + std::to_string(il) + ".";
        f.add(p + "input_layernorm.weight",          "BF16", { n_embd }, rnd_data("BF16", n_embd, -0.1f, 0.1f));
        f.add(p + "post_attention_layernorm.weight", "BF16", { n_embd }, rnd_data("BF16", n_embd, -0.1f, 0.1f));
        add_exl3(f, p + "mlp.gate_proj", n_embd, n_ff, 4);
        add_exl3(f, p + "mlp.up_proj",   n_embd, n_ff, 4);
        add_exl3(f, p + "mlp.down_proj", n_ff, n_embd, 4);
        if (il < 3) {
            const int64_t qk = 2*n_kh*head_k, v = n_vh*head_v;
            add_exl3(f, p + "linear_attn.in_proj_qkv", n_embd, qk + v, 4);
            add_exl3(f, p + "linear_attn.in_proj_z",   n_embd, v, 4);
            add_exl3(f, p + "linear_attn.out_proj",    v, n_embd, 4);
            f.add(p + "linear_attn.in_proj_a.weight", "F16",  { n_vh, n_embd }, rnd_data("F16", n_vh*n_embd, -0.05f, 0.05f));
            f.add(p + "linear_attn.in_proj_b.weight", "F16",  { n_vh, n_embd }, rnd_data("F16", n_vh*n_embd, -0.05f, 0.05f));
            f.add(p + "linear_attn.A_log",            "F32",  { n_vh }, rnd_data("F32", n_vh, -1.0f, 1.0f));
            f.add(p + "linear_attn.dt_bias",          "BF16", { n_vh }, rnd_data("BF16", n_vh, -0.5f, 0.5f));
            f.add(p + "linear_attn.conv1d.weight",    "BF16", { qk + v, 1, 4 }, rnd_data("BF16", (qk + v)*4, -0.3f, 0.3f));
            f.add(p + "linear_attn.norm.weight",      "F32",  { head_v }, rnd_data("F32", head_v, 0.9f, 1.1f));
        } else {
            add_exl3(f, p + "self_attn.q_proj", n_embd, n_head*head_dim*2, 4);
            add_exl3(f, p + "self_attn.k_proj", n_embd, n_head_kv*head_dim, 4);
            add_exl3(f, p + "self_attn.v_proj", n_embd, n_head_kv*head_dim, 4);
            add_exl3(f, p + "self_attn.o_proj", n_head*head_dim, n_embd, 4);
            f.add(p + "self_attn.q_norm.weight", "BF16", { head_dim }, rnd_data("BF16", head_dim, -0.1f, 0.1f));
            f.add(p + "self_attn.k_norm.weight", "BF16", { head_dim }, rnd_data("BF16", head_dim, -0.1f, 0.1f));
        }
    }
    if (mtp) {
        add_exl3(s2, "mtp.fc", 2*n_embd, n_embd, 4);
        s2.add("mtp.pre_fc_norm_embedding.weight", "BF16", { n_embd }, rnd_data("BF16", n_embd, -0.1f, 0.1f));
        s2.add("mtp.pre_fc_norm_hidden.weight",    "BF16", { n_embd }, rnd_data("BF16", n_embd, -0.1f, 0.1f));
        s2.add("mtp.norm.weight",                  "BF16", { n_embd }, rnd_data("BF16", n_embd, -0.1f, 0.1f));
        const std::string p = "mtp.layers.0.";
        s2.add(p + "input_layernorm.weight",          "BF16", { n_embd }, rnd_data("BF16", n_embd, -0.1f, 0.1f));
        s2.add(p + "post_attention_layernorm.weight", "BF16", { n_embd }, rnd_data("BF16", n_embd, -0.1f, 0.1f));
        add_exl3(s2, p + "self_attn.q_proj", n_embd, n_head*head_dim*2, 4);
        add_exl3(s2, p + "self_attn.k_proj", n_embd, n_head_kv*head_dim, 4);
        add_exl3(s2, p + "self_attn.v_proj", n_embd, n_head_kv*head_dim, 4);
        add_exl3(s2, p + "self_attn.o_proj", n_head*head_dim, n_embd, 4);
        s2.add(p + "self_attn.q_norm.weight", "BF16", { head_dim }, rnd_data("BF16", head_dim, -0.1f, 0.1f));
        s2.add(p + "self_attn.k_norm.weight", "BF16", { head_dim }, rnd_data("BF16", head_dim, -0.1f, 0.1f));
        add_exl3(s2, p + "mlp.gate_proj", n_embd, n_ff, 4);
        add_exl3(s2, p + "mlp.up_proj",   n_embd, n_ff, 4);
        add_exl3(s2, p + "mlp.down_proj", n_ff, n_embd, 4);
    }
    s2.add("model.visual.blocks.0.norm1.weight", "BF16", { 16 }, rnd_data("BF16", 16, -1.0f, 1.0f));
    s2.add("model.visual.patch_embed.proj.weight", "BF16", { 16, 48 }, rnd_data("BF16", 16*48, -1.0f, 1.0f));
    add_exl3(head, "lm_head", n_embd, n_vocab, 6);

    s1.write(dir / "model-00001-of-00002.safetensors");
    s2.write(dir / "model-00002-of-00002.safetensors");
    head.write(dir / "lm_head.safetensors");

    n_params = 0;
    std::string index = "{\"metadata\": {}, \"weight_map\": {";
    bool first = true;
    for (const st_file * f : { &s1, &s2, &head }) {
        for (const auto & t : f->tensors) {
            src[t.name] = t;
            if (f != &head) {
                index += std::string(first ? "" : ", ") + "\"" + t.name + "\": \"model-0000" + (f == &s1 ? "1" : "2") + "-of-00002.safetensors\"";
                first = false;
            }
            auto ends_with = [&](const std::string & suffix) {
                return t.name.size() >= suffix.size() && t.name.compare(t.name.size() - suffix.size(), suffix.size(), suffix) == 0;
            };
            if (t.name.find("visual.") != std::string::npos || ends_with(".suh") || ends_with(".svh") || ends_with(".mcg") ||
                    t.name.rfind("mtp.", 0) == 0) {
                continue;
            }
            int64_t n = 1;
            for (const int64_t x : t.shape) {
                n *= x;
            }
            n_params += ends_with(".trellis") ? t.shape[0]*16*t.shape[1]*16 : n;
        }
    }
    std::ofstream(dir / "model.safetensors.index.json") << index << "}}";

    std::ofstream(dir / "config.json") << R"JSON({
 "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "tie_word_embeddings": true,
 "text_config": {"attn_output_gate": true, "full_attention_interval": 4, "head_dim": 256, "hidden_size": 256, "intermediate_size": 256,
  "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"], "linear_conv_kernel_dim": 4,
  "linear_key_head_dim": 128, "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_value_head_dim": 128,
  "max_position_embeddings": 4096, "mtp_num_hidden_layers": 1, "num_attention_heads": 2, "num_hidden_layers": 4, "num_key_value_heads": 1,
  "rms_norm_eps": 1e-06, "tie_word_embeddings": true, "vocab_size": 384, "eos_token_id": 300,
  "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25}},
 "vision_config": {"depth": 1},
 "quantization_config": {"quant_method": "exl3", "version": "0.0.22", "bits": 4.0, "head_bits": 6, "codebook": "mcg"}})JSON";
    // tensor_storage lists one module only: the importer must not take it as the inventory
    std::ofstream(dir / "quantization_config.json") << R"JSON({"quant_method": "exl3", "tensor_storage": {"model.language_model.layers.0.mlp.gate_proj": {}}})JSON";

    // the byte-level alphabet of GPT-2, two merges, added tokens in tokenizer.json and one only in tokenizer_config.json
    std::string vocab;
    for (int b = 0, n = 0; b < 256; b++) {
        const bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
        const int  cp     = direct ? b : 256 + n++;
        std::string u;
        if (cp < 0x80) {
            u = std::string(1, (char) cp);
        } else {
            u = std::string(1, (char) (0xC0 | (cp >> 6))) + (char) (0x80 | (cp & 0x3F));
        }
        if (u == "\"" || u == "\\") {
            u = "\\" + u;
        }
        vocab += "\"" + u + "\": " + std::to_string(b) + ", ";
    }
    vocab += "\"\xc4\xa0t\": 256, \"he\": 257";
    std::ofstream(dir / "tokenizer.json") << R"JSON({"version": "1.0", "truncation": null, "padding": null,
 "added_tokens": [{"id": 300, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
  {"id": 301, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
  {"id": 302, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
  {"id": 303, "content": "<think>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}],
 "normalizer": {"type": "NFC"},
 "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]},
 "post_processor": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
 "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
 "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "", "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
  "vocab": {)JSON" << vocab << R"JSON(}, "merges": [[")JSON" << "\\u0120" << R"JSON(", "t"], ["h", "e"]]}})JSON";
    std::ofstream(dir / "tokenizer_config.json") << R"JSON({"add_bos_token": false, "add_prefix_space": false, "bos_token": null,
 "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}",
 "added_tokens_decoder": {"300": {"content": "<|endoftext|>", "special": true}, "304": {"content": "<|audio_pad|>", "special": true}}})JSON";
}

static std::vector<uint8_t> import_data(const common_exl3_dir * d, const std::string & name) {
    const gguf_context * g = common_exl3_dir_gguf(d);
    const int64_t id = gguf_find_tensor(g, name.c_str());
    if (id < 0) {
        return {};
    }
    std::vector<uint8_t> out(gguf_get_tensor_size(g, id));
    common_exl3_dir_read(d, name.c_str(), out.data(), out.size());
    return out;
}

static float f32_of(const std::string & dtype, const uint8_t * p) {
    uint16_t h;
    memcpy(&h, p, 2);
    if (dtype == "BF16") {
        const uint32_t u = (uint32_t) h << 16;
        float x;
        memcpy(&x, &u, 4);
        return x;
    }
    if (dtype == "F16") {
        return ggml_fp16_to_fp32(h);
    }
    float x;
    memcpy(&x, p, 4);
    return x;
}

static int synthetic() {
    const fs::path dir = fs::temp_directory_path() / ("test-exl3-dir-" + std::to_string((unsigned) time(nullptr)));
    std::map<std::string, st_file::entry> src;
    int64_t n_params = 0;
    write_synthetic(dir / "model", src, n_params);
    const std::string m = "model.language_model.";

    printf("import of a synthetic qwen35 EXL3 directory (%s)\n", dir.string().c_str());
    const common_exl3_dir_ptr d = common_exl3_dir_open((dir / "model").string());
    const gguf_context * g = common_exl3_dir_gguf(d.get());
    auto u32 = [&](const char * key) { const int64_t id = gguf_find_key(g, key); return id < 0 ? UINT32_MAX : gguf_get_val_u32(g, id); };
    auto str = [&](const char * key) { const int64_t id = gguf_find_key(g, key); return id < 0 ? std::string() : std::string(gguf_get_val_str(g, id)); };

    check(gguf_get_n_tensors(g) == 26*3 + 30, "26 EXL3 linears and 30 other tensors, vision tower not loaded");
    check(gguf_find_tensor(g, "output.exl3_trellis") >= 0, "lm_head found in a file the index does not list, not replaced by the tied embedding");
    check(u32("qwen35.block_count") == 4 && gguf_find_key(g, "qwen35.nextn_predict_layers") < 0, "no MTP layer although config.json declares one");
    check(str("tokenizer.ggml.pre") == "qwen35" && str("exl3.codebook") == "mcg", "pre-tokenizer and codebook");
    check(u32("tokenizer.ggml.eos_token_id") == 302 && u32("tokenizer.ggml.padding_token_id") == 300, "eos and pad from tokenizer_config.json");
    {
        const int64_t kt = gguf_find_key(g, "tokenizer.ggml.token_type");
        const int64_t ks = gguf_find_key(g, "tokenizer.ggml.tokens");
        const int64_t km = gguf_find_key(g, "tokenizer.ggml.merges");
        const int32_t * tt = (const int32_t *) gguf_get_arr_data(g, kt);
        check(gguf_get_arr_n(g, ks) == 384 && tt[65] == 1 && tt[300] == 3 && tt[303] == 4 && tt[304] == 3 && tt[305] == 5 &&
              std::string(gguf_get_arr_str(g, ks, 304)) == "<|audio_pad|>" && std::string(gguf_get_arr_str(g, ks, 260)) == "[PAD260]" &&
              std::string(gguf_get_arr_str(g, km, 0)) == "\xc4\xa0 t", "tokens, token types and merges");
    }

    bool ok = true;
    {
        const auto & s = src.at(m + "layers.0.linear_attn.A_log");
        const auto   y = import_data(d.get(), "blk.0.ssm_a");
        for (int64_t p = 0; p < 4; p++) {
            const float x = f32_of("F32", s.data.data() + 4*v_head_src(p, 2, 2, 1));
            const float e = (float) -std::exp((double) x);
            ok = ok && memcmp(&e, y.data() + 4*p, 4) == 0;
        }
        check(ok, "ssm_a = -exp(A_log) in the tiled V head order, rounded once from float64");
    }
    {
        const auto & s = src.at(m + "layers.1.input_layernorm.weight");
        const auto   y = import_data(d.get(), "blk.1.attn_norm.weight");
        ok = y.size() == 4*256;
        for (int64_t i = 0; ok && i < 256; i++) {
            const float e = f32_of("BF16", s.data.data() + 2*i) + 1.0f;
            ok = memcmp(&e, y.data() + 4*i, 4) == 0;
        }
        check(ok, "zero-centered norm + 1 in F32");
    }
    {
        const auto & s = src.at(m + "layers.2.linear_attn.conv1d.weight");
        const auto   y = import_data(d.get(), "blk.2.ssm_conv1d.weight");
        ok = y.size() == 4*1024*4;
        for (int64_t r = 0; ok && r < 1024; r++) {
            const int64_t q = r < 512 ? r : 512 + v_head_src(r - 512, 2, 2, 128);
            for (int64_t c = 0; ok && c < 4; c++) {
                const float e = f32_of("BF16", s.data.data() + 2*(q*4 + c));
                ok = memcmp(&e, y.data() + 4*(r*4 + c), 4) == 0;
            }
        }
        check(ok, "conv1d squeezed, V channels reordered, F32");
    }
    {
        const auto & tr = src.at(m + "layers.0.linear_attn.in_proj_qkv.trellis");
        const auto & sv = src.at(m + "layers.0.linear_attn.in_proj_qkv.svh");
        const auto   yt = import_data(d.get(), "blk.0.attn_qkv.exl3_trellis");
        const auto   ys = import_data(d.get(), "blk.0.attn_qkv.exl3_svh");
        const int64_t tn = 64, tile = 64*2;
        ok = yt.size() == tr.data.size() && ys.size() == sv.data.size();
        for (int64_t kt = 0; ok && kt < 16; kt++) {
            for (int64_t j = 0; ok && j < tn; j++) {
                const int64_t q = j < 32 ? j : 32 + v_head_src(j - 32, 2, 2, 8);
                ok = memcmp(yt.data() + (kt*tn + j)*tile, tr.data.data() + (kt*tn + q)*tile, tile) == 0;
            }
        }
        for (int64_t i = 0; ok && i < 1024; i++) {
            const int64_t q = i < 512 ? i : 512 + v_head_src(i - 512, 2, 2, 128);
            ok = memcmp(ys.data() + 2*i, sv.data.data() + 2*q, 2) == 0;
        }
        check(ok, "in_proj_qkv: V tiles and svh reordered in whole 128-blocks, q and k untouched");
    }
    {
        const auto & tr = src.at(m + "layers.1.linear_attn.out_proj.trellis");
        const auto & su = src.at(m + "layers.1.linear_attn.out_proj.suh");
        const auto   yt = import_data(d.get(), "blk.1.ssm_out.exl3_trellis");
        const auto   ys = import_data(d.get(), "blk.1.ssm_out.exl3_suh");
        const int64_t row = 16*64*2;
        ok = yt.size() == tr.data.size() && ys.size() == su.data.size();
        for (int64_t kt = 0; ok && kt < 32; kt++) {
            ok = memcmp(yt.data() + kt*row, tr.data.data() + v_head_src(kt, 2, 2, 8)*row, row) == 0;
        }
        for (int64_t i = 0; ok && i < 512; i++) {
            ok = memcmp(ys.data() + 2*i, su.data.data() + 2*v_head_src(i, 2, 2, 128), 2) == 0;
        }
        check(ok, "out_proj: tile rows and suh reordered on the input side");
    }
    {
        const auto & s = src.at(m + "layers.2.linear_attn.in_proj_a.weight");
        const auto   y = import_data(d.get(), "blk.2.ssm_alpha.weight");
        ok = y.size() == s.data.size();
        for (int64_t r = 0; ok && r < 4; r++) {
            ok = memcmp(y.data() + r*512, s.data.data() + v_head_src(r, 2, 2, 1)*512, 512) == 0;
        }
        check(ok, "in_proj_a rows reordered, F16 kept");
    }
    check(import_data(d.get(), "token_embd.weight") == src.at(m + "embed_tokens.weight").data, "token_embd copied from an offset that is 2 mod 4");

    {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        llama_model * model = common_exl3_load_model((dir / "model").string(), mp);
        check(model != nullptr, "the model loads through llama_model_init_from_user");
        if (model) {
            check(llama_model_n_params(model) == (uint64_t) n_params, "parameters counted as the k x n weights of the linears");
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx     = 64;
            cp.n_batch   = 16;
            cp.n_ubatch  = 16;
            cp.n_threads = 4;
            llama_context * ctx = llama_init_from_model(model, cp);
            std::vector<llama_token> tokens = { 65, 66, 256, 257, 302 };
            ok = ctx && llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size())) == 0;
            const float * logits = ok ? llama_get_logits_ith(ctx, -1) : nullptr;
            for (int i = 0; ok && i < 384; i++) {
                ok = std::isfinite(logits[i]);
            }
            check(ok, "a prompt runs and gives finite logits");
            llama_free(ctx);
            llama_model_free(model);
        }
    }

    // a stored MTP layer, named after the trunk layers as the converter names it, and a quantized token embedding
    {
        std::map<std::string, st_file::entry> src2;
        int64_t n_params2 = 0;
        write_synthetic(dir / "mtp", src2, n_params2, true, true);
        const common_exl3_dir_ptr d2 = common_exl3_dir_open((dir / "mtp").string());
        const gguf_context * g2 = common_exl3_dir_gguf(d2.get());
        auto u32_2 = [&](const char * key) { const int64_t id = gguf_find_key(g2, key); return id < 0 ? UINT32_MAX : gguf_get_val_u32(g2, id); };
        const int64_t kr = gguf_find_key(g2, "qwen35.attention.recurrent_layers");
        check(u32_2("qwen35.block_count") == 5 && u32_2("qwen35.nextn_predict_layers") == 1 && gguf_get_arr_n(g2, kr) == 5 &&
              !((const int8_t *) gguf_get_arr_data(g2, kr))[4], "MTP layer counted from the tensors: block 4, full attention");
        check(gguf_get_n_tensors(g2) == (26 + 8)*3 + 30 + 7 + 1, "8 MTP linears and 7 other MTP tensors imported, embedding as rows and signs");
        check(import_data(d2.get(), "blk.4.nextn.eh_proj.exl3_trellis") == src2.at("mtp.fc.trellis").data &&
              import_data(d2.get(), "blk.4.attn_q.exl3_trellis") == src2.at("mtp.layers.0.self_attn.q_proj.trellis").data,
              "mtp.fc -> blk.4.nextn.eh_proj, mtp.layers.0 -> blk.4");
        {
            const auto & s = src2.at("mtp.pre_fc_norm_embedding.weight");
            const auto   y = import_data(d2.get(), "blk.4.nextn.enorm.weight");
            ok = y.size() == 4*256 && gguf_find_tensor(g2, "blk.4.nextn.hnorm.weight") >= 0 &&
                 gguf_find_tensor(g2, "blk.4.nextn.shared_head_norm.weight") >= 0;
            for (int64_t i = 0; ok && i < 256; i++) {
                const float e = f32_of("BF16", s.data.data() + 2*i) + 1.0f;
                ok = memcmp(&e, y.data() + 4*i, 4) == 0;
            }
            check(ok, "MTP norms: enorm, hnorm, shared_head_norm, + 1 in F32");
        }
        check(import_data(d2.get(), "token_embd.exl3r_trellis") == src2.at(m + "embed_tokens.trellis").data &&
              import_data(d2.get(), "token_embd.exl3r_signs") == src2.at(m + "embed_tokens.signs").data &&
              gguf_find_tensor(g2, "token_embd.weight") < 0 && u32_2("exl3.rows.version") == 1,
              "quantized token embedding carried as token_embd.exl3r_trellis/_signs, exl3.rows.version 1");

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mtp     = true;
        llama_model * model = common_exl3_load_model((dir / "mtp").string(), mp);
        check(model != nullptr && llama_model_n_layer_nextn(model) == 1, "the model loads with its MTP layer");
        if (model) {
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx     = 64;
            cp.n_batch   = 16;
            cp.n_ubatch  = 16;
            cp.n_threads = 4;
            llama_context * ctx = llama_init_from_model(model, cp);
            std::vector<llama_token> tokens = { 65, 66, 256, 257, 302 };
            ok = ctx && llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size())) == 0;
            const float * logits = ok ? llama_get_logits_ith(ctx, -1) : nullptr;
            for (int i = 0; ok && i < 384; i++) {
                ok = std::isfinite(logits[i]);
            }
            check(ok, "a prompt runs through the row-ring embedding and gives finite logits");
            llama_free(ctx);
            llama_model_free(model);
        }
    }

    // the same directory with one more file that repeats a tensor: refused, not overridden by the order of the files
    fs::copy(dir / "model", dir / "dup", fs::copy_options::recursive);
    st_file extra;
    extra.add(m + "norm.weight", "BF16", { 256 }, rnd_data("BF16", 256, -0.1f, 0.1f));
    extra.write(dir / "dup" / "extra.safetensors");
    std::string err;
    try {
        common_exl3_dir_open((dir / "dup").string());
    } catch (const std::exception & e) {
        err = e.what();
    }
    check(err.find("is in both") != std::string::npos, "a tensor name in two files is refused");

    fs::remove_all(dir);
    printf("%s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    try {
        if (argc == 4 && strcmp(argv[1], "--compare") == 0) {
            return compare(argv[2], argv[3]);
        }
        if (argc == 4 && strcmp(argv[1], "--dump") == 0) {
            return dump(argv[2], argv[3]);
        }
        if (argc != 1) {
            fprintf(stderr, "usage: %s [--compare DIR CONTAINER.gguf | --dump DIR OUT.gguf]\n", argv[0]);
            return 1;
        }
        llama_backend_init();
        const int ret = synthetic();
        llama_backend_free();
        return ret;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
