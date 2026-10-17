#include "exl3-dir.h"

#include "common.h"
#include "json.h"
#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "hash/hash.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

// see ggml/exl3/README.md and conversion/exl3.py
static const uint32_t EXL3_MARKER_MCG  = 0xCBAC1FED;
static const uint32_t EXL3_MARKER_MUL1 = 0x83DCD12D;
static const char *   EXL3_PERMUTATION = "linear_attn_v_heads_grouped_to_tiled";

// sha256 of json.dumps({"normalizer": ..., "pre_tokenizer": ...}, sort_keys=True) of tokenizer.json -> tokenizer.ggml.pre
// the converter hashes tokenizer output instead, which needs the HF tokenizer; a new entry must be checked against a converted file
static const struct {
    const char * fingerprint;
    const char * pre;
} EXL3_PRE_TOKENIZERS[] = {
    { "821105e4f95dacb2d90e189de93c8273343574e7d2aae0ea82fc552ff0bc84b9", "qwen35" }, // Qwen3.5
};

// one tensor of a safetensors file
struct st_tensor {
    std::string          dtype;
    std::vector<int64_t> shape;
    int                  shard = 0;
    uint64_t             offs  = 0; // file offset, often not aligned (2 mod 4 in real EXL3 files): only read into buffers
    uint64_t             size  = 0;
};

// one tensor of the import: the source tensor, its V-head reorder, its type conversion and arithmetic
struct exl3_out_tensor {
    std::string          name;
    std::string          src;
    ggml_type            type = GGML_TYPE_F32; // F32 converts the source, any other type is a copy of it
    std::vector<int64_t> shape;                // row-major shape of the result
    int                  arith = 0;            // 1: x + 1 (zero-centered norm), 2: -exp(x) (A_log)

    // [K, V, unit] -> [V, K, unit] on the elements [start, end) of axis `perm_axis`, see _reorder_v_heads in conversion/qwen.py
    int     perm_axis  = -1;
    int64_t perm_start = 0;
    int64_t perm_unit  = 0;
    int64_t perm_k     = 0;
    int64_t perm_v     = 0;
};

struct common_exl3_dir {
    std::vector<fs::path>                    shards;
    std::map<std::string, st_tensor>         st;
    std::vector<exl3_out_tensor>             out;
    std::unordered_map<std::string, size_t>  out_idx;
    gguf_context *                           meta     = nullptr;
    ggml_context *                           ctx_meta = nullptr;
    mutable std::vector<std::ifstream>       files;

    ~common_exl3_dir() {
        if (meta) {
            gguf_free(meta);
        }
        if (ctx_meta) {
            ggml_free(ctx_meta);
        }
    }
};

void common_exl3_dir_deleter::operator()(common_exl3_dir * dir) {
    delete dir;
}

static common_json exl3_read_json(const fs::path & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path.string());
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return common_json::parse(ss.str());
}

static bool str_ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool str_starts_with(const std::string & s, const std::string & prefix) {
    return s.compare(0, prefix.size(), prefix) == 0;
}

static std::string str_replace_all(std::string s, const std::string & from, const std::string & to) {
    for (size_t pos = s.find(from); pos != std::string::npos; pos = s.find(from, pos + to.size())) {
        s.replace(pos, from.size(), to);
    }
    return s;
}

// json.dumps(v, sort_keys=True) of Python: ", " and ": " separators, ASCII only, shortest float repr
static void exl3_py_json(const common_json & v, std::string & out) {
    if (v.is_null()) {
        out += "null";
    } else if (v.is_boolean()) {
        out += v.get<bool>() ? "true" : "false";
    } else if (v.is_number_integer()) {
        out += v.dump();
    } else if (v.is_number_float()) {
        const double x = v.get<double>();
        if (!std::isfinite(x)) {
            out += std::isnan(x) ? "NaN" : x > 0 ? "Infinity" : "-Infinity";
            return;
        }
        // shortest round trip digits, then the layout of repr(): fixed for 1e-4 <= |x| < 1e16, else d.ddde+XX
        char buf[64];
        for (int prec = 1; prec <= 17; prec++) {
            snprintf(buf, sizeof(buf), "%.*e", prec - 1, x);
            if (strtod(buf, nullptr) == x) {
                break;
            }
        }
        std::string s = buf;
        const size_t epos = s.find('e');
        const int    exp  = atoi(s.c_str() + epos + 1);
        std::string  digits;
        for (size_t i = 0; i < epos; i++) {
            if (isdigit((unsigned char) s[i])) {
                digits += s[i];
            }
        }
        const std::string sign = std::signbit(x) ? "-" : "";
        std::string r;
        if (exp >= -4 && exp < 16) {
            if (exp >= 0) {
                std::string ip = digits.substr(0, std::min(digits.size(), (size_t) exp + 1));
                ip += std::string(exp + 1 - ip.size(), '0');
                std::string fp = digits.size() > (size_t) exp + 1 ? digits.substr(exp + 1) : "0";
                r = ip + "." + fp;
            } else {
                r = "0." + std::string(-exp - 1, '0') + digits;
            }
        } else {
            r = digits.substr(0, 1) + (digits.size() > 1 ? "." + digits.substr(1) : "");
            char eb[16];
            snprintf(eb, sizeof(eb), "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
            r += eb;
        }
        out += sign + r;
    } else if (v.is_string()) {
        const std::string s = v.get<std::string>();
        out += '"';
        for (size_t i = 0; i < s.size(); ) {
            const unsigned char c = s[i];
            uint32_t cp = c;
            size_t   n  = 1;
            if (c >= 0xF0) {
                cp = c & 0x07; n = 4;
            } else if (c >= 0xE0) {
                cp = c & 0x0F; n = 3;
            } else if (c >= 0xC0) {
                cp = c & 0x1F; n = 2;
            }
            for (size_t k = 1; k < n && i + k < s.size(); k++) {
                cp = (cp << 6) | (s[i + k] & 0x3F);
            }
            i += n;
            char eb[16];
            switch (cp) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                case '\b': out += "\\b";  break;
                case '\f': out += "\\f";  break;
                default:
                    if (cp < 0x20 || cp > 0x7E) {
                        if (cp >= 0x10000) {
                            cp -= 0x10000;
                            snprintf(eb, sizeof(eb), "\\u%04x\\u%04x", 0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF));
                        } else {
                            snprintf(eb, sizeof(eb), "\\u%04x", cp);
                        }
                        out += eb;
                    } else {
                        out += (char) cp;
                    }
            }
        }
        out += '"';
    } else if (v.is_array()) {
        out += '[';
        bool first = true;
        for (const auto & e : v) {
            out += first ? "" : ", ";
            first = false;
            exl3_py_json(e, out);
        }
        out += ']';
    } else if (v.is_object()) {
        std::vector<std::string> keys;
        for (const auto & [k, e] : v.items()) {
            keys.push_back(k);
        }
        std::sort(keys.begin(), keys.end());
        out += '{';
        for (size_t i = 0; i < keys.size(); i++) {
            out += i ? ", " : "";
            exl3_py_json(common_json(keys[i]), out);
            out += ": ";
            exl3_py_json(v.at(keys[i]), out);
        }
        out += '}';
    }
}

static std::string exl3_py_json(const common_json & v) {
    std::string out;
    exl3_py_json(v, out);
    return out;
}

static size_t st_dtype_size(const std::string & dtype) {
    if (dtype == "F64" || dtype == "I64" || dtype == "U64") {
        return 8;
    }
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") {
        return 4;
    }
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") {
        return 2;
    }
    if (dtype == "I8" || dtype == "U8" || dtype == "BOOL" || dtype == "F8_E4M3" || dtype == "F8_E5M2") {
        return 1;
    }
    return 0;
}

static ggml_type st_dtype_ggml(const std::string & dtype) {
    if (dtype == "F32")  { return GGML_TYPE_F32;  }
    if (dtype == "F16")  { return GGML_TYPE_F16;  }
    if (dtype == "BF16") { return GGML_TYPE_BF16; }
    if (dtype == "I16")  { return GGML_TYPE_I16;  }
    if (dtype == "I32")  { return GGML_TYPE_I32;  }
    return GGML_TYPE_COUNT;
}

// every tensor of one safetensors file; a name that is already known is refused (exllamav3 lets the glob order pick one)
static void st_read_header(const fs::path & path, int shard, std::map<std::string, st_tensor> & st, std::map<std::string, std::string> & owner) {
    std::ifstream f(path, std::ios::binary);
    const uint64_t fsize = fs::file_size(path);
    unsigned char len[8];
    if (!f || fsize < 8 || !f.read((char *) len, 8)) {
        throw std::runtime_error("cannot read " + path.string());
    }
    uint64_t n = 0;
    for (int i = 7; i >= 0; i--) {
        n = (n << 8) | len[i];
    }
    if (n < 2 || n > fsize - 8 || n > (uint64_t) 256 << 20) {
        throw std::runtime_error(string_format("%s: bad safetensors header size %" PRIu64, path.string().c_str(), n));
    }
    std::string hdr(n, '\0');
    if (!f.read(&hdr[0], (std::streamsize) n)) {
        throw std::runtime_error("cannot read the header of " + path.string());
    }
    const uint64_t data_start = 8 + n;
    const common_json j = common_json::parse(hdr);
    if (!j.is_object()) {
        throw std::runtime_error(path.string() + ": the safetensors header is not an object");
    }
    for (const auto & [name, info] : j.items()) {
        if (name == "__metadata__") {
            continue;
        }
        st_tensor t;
        t.shard = shard;
        t.dtype = info.at("dtype").get<std::string>();
        const size_t es = st_dtype_size(t.dtype);
        if (es == 0) {
            throw std::runtime_error(string_format("%s: tensor %s has unknown dtype %s", path.string().c_str(), name.c_str(), t.dtype.c_str()));
        }
        uint64_t ne = 1;
        for (const auto & d : info.at("shape")) {
            const long long v = d.get<long long>();
            if (v < 0 || (v > 0 && ne > UINT64_MAX / (uint64_t) v)) {
                throw std::runtime_error(string_format("%s: tensor %s has a bad shape", path.string().c_str(), name.c_str()));
            }
            t.shape.push_back(v);
            ne *= (uint64_t) v;
        }
        const common_json & offs = info.at("data_offsets");
        const uint64_t b = offs.at(0).get<unsigned long long>();
        const uint64_t e = offs.at(1).get<unsigned long long>();
        if (offs.size() != 2 || b > e || e > fsize - data_start || ne > UINT64_MAX / es || e - b != ne*es) {
            throw std::runtime_error(string_format("%s: tensor %s has bad data offsets", path.string().c_str(), name.c_str()));
        }
        t.offs = data_start + b;
        t.size = e - b;
        const auto it = owner.find(name);
        if (it != owner.end()) {
            throw std::runtime_error(string_format("tensor %s is in both %s and %s", name.c_str(), it->second.c_str(), path.filename().string().c_str()));
        }
        owner.emplace(name, path.filename().string());
        st.emplace(name, std::move(t));
    }
}

// first key of `keys` that hp has, as find_hparam of the converter
static const common_json * exl3_find(const common_json & hp, std::initializer_list<const char *> keys) {
    for (const char * k : keys) {
        if (hp.contains(k)) {
            return &hp.at(k);
        }
    }
    return nullptr;
}

static int64_t exl3_get_int(const common_json & hp, const char * key) {
    if (!hp.contains(key) || !hp.at(key).is_number_integer()) {
        throw std::runtime_error(string_format("config.json: %s is missing or not an integer", key));
    }
    return hp.at(key).get<long long>();
}

static uint32_t exl3_u32(int64_t v, const char * what) {
    if (v < 0 || v > (int64_t) UINT32_MAX) {
        throw std::runtime_error(string_format("config.json: %s = %" PRId64 " is out of range", what, v));
    }
    return (uint32_t) v;
}

static bool exl3_looks_special(const std::string & t) {
    return t == "<pad>" || t == "<mask>" || t == "<2mass>" || t == "[@BOS@]" ||
           (str_starts_with(t, "<|") && str_ends_with(t, "|>")) ||
           (str_starts_with(t, "<\xef\xbd\x9c") && str_ends_with(t, "\xef\xbd\x9c>")) ||
           (str_starts_with(t, "<unused") && str_ends_with(t, ">"));
}

static std::string exl3_read_text(const fs::path & path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// a small JSON scanner: common_json keeps the order of object keys and parses the 250k keys of model.vocab in quadratic time
struct exl3_json_scan {
    const std::string & s;
    size_t p = 0;

    [[noreturn]] void fail() const {
        throw std::runtime_error(string_format("tokenizer.json: bad JSON at byte %zu", p));
    }
    void ws() {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) {
            p++;
        }
    }
    void expect(char c) {
        ws();
        if (p >= s.size() || s[p] != c) {
            fail();
        }
        p++;
    }
    bool next_is(char c) {
        ws();
        return p < s.size() && s[p] == c;
    }
    uint32_t hex4() {
        if (p + 4 > s.size()) {
            fail();
        }
        uint32_t v = 0;
        for (int i = 0; i < 4; i++) {
            const char c = s[p++];
            v = v*16 + (c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : (fail(), 0));
        }
        return v;
    }
    std::string str() {
        expect('"');
        std::string out;
        while (true) {
            if (p >= s.size()) {
                fail();
            }
            const char c = s[p++];
            if (c == '"') {
                return out;
            }
            if ((unsigned char) c < 0x20) {
                fail();
            }
            if (c != '\\') {
                out += c;
                continue;
            }
            const char e = p < s.size() ? s[p++] : (fail(), 0);
            switch (e) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00) {
                        if (p + 2 > s.size() || s[p] != '\\' || s[p + 1] != 'u') {
                            fail();
                        }
                        p += 2;
                        const uint32_t lo = hex4();
                        if (lo < 0xDC00 || lo >= 0xE000) {
                            fail();
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp < 0xE000) {
                        fail();
                    }
                    if (cp < 0x80) {
                        out += (char) cp;
                    } else if (cp < 0x800) {
                        out += (char) (0xC0 | (cp >> 6));
                        out += (char) (0x80 | (cp & 0x3F));
                    } else if (cp < 0x10000) {
                        out += (char) (0xE0 | (cp >> 12));
                        out += (char) (0x80 | ((cp >> 6) & 0x3F));
                        out += (char) (0x80 | (cp & 0x3F));
                    } else {
                        out += (char) (0xF0 | (cp >> 18));
                        out += (char) (0x80 | ((cp >> 12) & 0x3F));
                        out += (char) (0x80 | ((cp >> 6) & 0x3F));
                        out += (char) (0x80 | (cp & 0x3F));
                    }
                } break;
                default: fail();
            }
        }
    }
    void skip() {
        ws();
        if (p >= s.size()) {
            fail();
        }
        const char c = s[p];
        if (c == '"') {
            str();
        } else if (c == '{' || c == '[') {
            const char close = c == '{' ? '}' : ']';
            p++;
            if (next_is(close)) {
                p++;
                return;
            }
            while (true) {
                if (c == '{') {
                    str();
                    expect(':');
                }
                skip();
                if (next_is(',')) {
                    p++;
                    continue;
                }
                expect(close);
                return;
            }
        } else {
            const size_t b = p;
            while (p < s.size() && (isalnum((unsigned char) s[p]) || s[p] == '-' || s[p] == '+' || s[p] == '.')) {
                p++;
            }
            if (p == b) {
                fail();
            }
        }
    }
    // [b, e) of the value of `key` in the object at p, false if it has none
    bool member(const std::string & key, size_t & b, size_t & e) {
        expect('{');
        if (next_is('}')) {
            return false;
        }
        while (true) {
            const std::string k = str();
            expect(':');
            ws();
            const size_t vb = p;
            skip();
            if (k == key) {
                b = vb;
                e = p;
                return true;
            }
            if (next_is(',')) {
                p++;
                continue;
            }
            expect('}');
            return false;
        }
    }
};

// tokenizer.json, and its model.vocab apart; the rest of the file goes through common_json without the vocab
static common_json exl3_read_tokenizer(const fs::path & path, std::vector<std::pair<std::string, int64_t>> & vocab) {
    const std::string text = exl3_read_text(path);
    exl3_json_scan sc = { text };
    size_t mb = 0, me = 0, vb = 0, ve = 0;
    if (!sc.member("model", mb, me)) {
        throw std::runtime_error("tokenizer.json has no model");
    }
    sc.p = mb;
    if (!sc.member("vocab", vb, ve)) {
        throw std::runtime_error("tokenizer.json has no model.vocab");
    }
    sc.p = vb;
    sc.expect('{');
    if (!sc.next_is('}')) {
        while (true) {
            std::string tok = sc.str();
            sc.expect(':');
            sc.ws();
            const size_t ib = sc.p;
            while (sc.p < text.size() && isdigit((unsigned char) text[sc.p])) {
                sc.p++;
            }
            if (sc.p == ib || sc.p - ib > 18) {
                sc.fail();
            }
            vocab.emplace_back(std::move(tok), std::stoll(text.substr(ib, sc.p - ib)));
            if (sc.next_is(',')) {
                sc.p++;
                continue;
            }
            sc.expect('}');
            break;
        }
    }
    return common_json::parse(text.substr(0, vb) + "{}" + text.substr(ve));
}

// tokenizer.ggml.* and tokenizer.chat_template, as _set_vocab_gpt2 and gguf.SpecialVocab write them
static void exl3_set_vocab(gguf_context * meta, const fs::path & dir, const common_json & config, const common_json & hp) {
    std::vector<std::pair<std::string, int64_t>> vocab;
    const common_json tj = exl3_read_tokenizer(dir / "tokenizer.json", vocab);
    const bool has_tc = fs::is_regular_file(dir / "tokenizer_config.json");
    const common_json tc = has_tc ? exl3_read_json(dir / "tokenizer_config.json") : common_json::object();

    const common_json & model = tj.at("model");
    if (model.value("type", "") != "BPE") {
        throw std::runtime_error("tokenizer.json: only BPE tokenizers are supported");
    }

    std::string pre;
    {
        common_json fp = common_json::object();
        fp["normalizer"]    = tj.value("normalizer", common_json());
        fp["pre_tokenizer"] = tj.value("pre_tokenizer", common_json());
        const std::string s  = exl3_py_json(fp);
        const std::string fh = hash_sha256_hex(s.data(), s.size());
        for (const auto & e : EXL3_PRE_TOKENIZERS) {
            if (fh == e.fingerprint) {
                pre = e.pre;
            }
        }
        if (pre.empty()) {
            throw std::runtime_error("tokenizer.json: unknown pre-tokenizer " + fh + ", convert the model with convert_hf_to_gguf.py");
        }
    }

    // added tokens of tokenizer.json and tokenizer_config.json, as the HF tokenizer merges them
    struct added_token {
        std::string content;
        bool        special;
    };
    std::map<int64_t, added_token> added;
    auto add_token = [&](int64_t id, const common_json & a) {
        const added_token t = { a.at("content").get<std::string>(), a.value("special", false) };
        const auto it = added.find(id);
        if (it != added.end() && (it->second.content != t.content || it->second.special != t.special)) {
            throw std::runtime_error(string_format("added token %" PRId64 " differs between tokenizer.json and tokenizer_config.json", id));
        }
        added[id] = t;
    };
    const common_json added_tj = tj.value("added_tokens", common_json::array());
    for (const auto & a : added_tj) {
        add_token(a.at("id").get<long long>(), a);
    }
    if (tc.contains("added_tokens_decoder")) {
        for (const auto & [id, a] : tc.at("added_tokens_decoder").items()) {
            add_token(std::stoll(id), a);
        }
    }

    std::unordered_map<std::string, int64_t> vocab_ids;
    for (const auto & [t, id] : vocab) {
        if (!vocab_ids.emplace(t, id).second) {
            throw std::runtime_error("tokenizer.json: token " + t + " is twice in model.vocab");
        }
    }
    int64_t n_vocab = (int64_t) vocab.size();
    for (const auto & [id, t] : added) {
        n_vocab += vocab_ids.count(t.content) == 0;
    }
    if (hp.contains("vocab_size")) {
        n_vocab = exl3_get_int(hp, "vocab_size");
    }

    std::vector<std::string> tokens(n_vocab);
    std::vector<int32_t>     types(n_vocab, 0);
    for (const auto & [t, id] : vocab) {
        if (id >= n_vocab || types[id] != 0) {
            throw std::runtime_error(string_format("tokenizer.json: token id %" PRId64 " is not below vocab_size %" PRId64 " or is used twice", id, n_vocab));
        }
        tokens[id] = t;
        types[id]  = 1; // NORMAL
    }
    for (const auto & [id, t] : added) {
        if (id < 0 || id >= n_vocab || (types[id] != 0 && tokens[id] != t.content)) {
            throw std::runtime_error(string_format("added token %" PRId64 " is out of range or clashes with the vocabulary", id));
        }
        // the converter also round-trips a token that is not normalized through the HF tokenizer; for added tokens that gives the token itself
        if (t.special || exl3_looks_special(t.content)) {
            tokens[id] = t.content;
            types[id]  = 3; // CONTROL
        } else {
            tokens[id] = str_replace_all(t.content, "\xe2\x96\x81", " ");
            types[id]  = 4; // USER_DEFINED
        }
    }
    for (int64_t i = 0; i < n_vocab; i++) {
        if (types[i] == 0) {
            tokens[i] = "[PAD" + std::to_string(i) + "]";
            types[i]  = 5; // UNUSED
        }
    }

    std::vector<std::string> merges;
    for (const auto & m : model.value("merges", common_json::array())) {
        if (m.is_string()) {
            merges.push_back(m.get<std::string>());
        } else if (m.is_array() && m.size() == 2) {
            // a space inside a pair is written as U+0120
            merges.push_back(str_replace_all(m.at(0).get<std::string>(), " ", "\xc4\xa0") + " " +
                             str_replace_all(m.at(1).get<std::string>(), " ", "\xc4\xa0"));
        } else {
            throw std::runtime_error("tokenizer.json: unknown merges format");
        }
    }

    // special tokens and add_*_token flags, in the order gguf.SpecialVocab finds them
    static const char * typs[] = { "bos", "eos", "unk", "sep", "pad", "cls", "mask" };
    std::vector<std::pair<std::string, int64_t>> ids;
    std::vector<std::pair<std::string, bool>>    adds;
    auto find_id = [&](const std::string & typ) {
        return std::find_if(ids.begin(), ids.end(), [&](const auto & e) { return e.first == typ; });
    };
    auto find_add = [&](const std::string & typ) {
        return std::find_if(adds.begin(), adds.end(), [&](const auto & e) { return e.first == typ; });
    };
    auto set_id = [&](const std::string & typ, const common_json & id) {
        if (!id.is_number_integer()) {
            return;
        }
        if (id.get<long long>() < 0) {
            throw std::runtime_error("invalid value for special token type " + typ);
        }
        if (find_id(typ) == ids.end()) {
            ids.emplace_back(typ, id.get<long long>());
        }
    };

    std::map<std::string, std::string> content;
    for (const char * typ : typs) {
        const std::string key = std::string(typ) + "_token";
        if (tc.contains(key)) {
            const common_json & e = tc.at(key);
            if (e.is_string()) {
                content[typ] = e.get<std::string>();
            } else if (e.is_object() && e.contains("content") && e.at("content").is_string()) {
                content[typ] = e.at("content").get<std::string>();
            }
        }
    }
    if (content["bos"].empty() && !content["cls"].empty()) {
        content["bos"] = content["cls"];
    }
    if (content["eos"].empty() && !content["sep"].empty()) {
        content["eos"] = content["sep"];
    }
    if (tj.contains("post_processor") && !tj.at("post_processor").is_null()) {
        const common_json & pp = tj.at("post_processor");
        const common_json procs = pp.contains("processors") ? pp.at("processors") : common_json::array({ pp });
        for (const auto & p : procs) {
            const std::string type = p.value("type", "");
            if (type == "ByteLevel") {
                if (find_add("bos") == adds.end()) {
                    adds.emplace_back("bos", false);
                }
                if (find_add("eos") == adds.end()) {
                    adds.emplace_back("eos", false);
                }
            } else if (type == "RobertaProcessing" || type == "TemplateProcessing") {
                throw std::runtime_error("tokenizer.json: post_processor " + type + " is not supported, convert the model with convert_hf_to_gguf.py");
            }
        }
    }

    common_json chat_template;
    if (has_tc) {
        common_json alt;
        bool alt_is_list = false;
        if (fs::is_regular_file(dir / "chat_template.jinja")) {
            alt = exl3_read_text(dir / "chat_template.jinja");
            if (fs::is_directory(dir / "additional_chat_templates")) {
                for (const auto & e : fs::directory_iterator(dir / "additional_chat_templates")) {
                    alt_is_list = alt_is_list || e.path().extension() == ".jinja";
                }
            }
        } else if (fs::is_regular_file(dir / "chat_template.json")) {
            alt = exl3_read_json(dir / "chat_template.json").value("chat_template", common_json());
        }
        if (tc.contains("chat_template")) {
            chat_template = tc.at("chat_template");
        } else if (alt_is_list) {
            throw std::runtime_error("additional_chat_templates is not supported, convert the model with convert_hf_to_gguf.py");
        } else {
            chat_template = alt;
        }
        if (chat_template.is_array()) {
            throw std::runtime_error("a list of chat templates is not supported, convert the model with convert_hf_to_gguf.py");
        }
        if (!chat_template.is_null() && !chat_template.is_string()) {
            LOG_WRN("%s: bad type for chat_template in tokenizer_config.json - ignoring\n", __func__);
            chat_template = common_json();
        }

        for (const char * typ : typs) {
            const std::string add_key = std::string("add_") + typ + "_token";
            if (tc.contains(add_key) && tc.at(add_key).is_boolean()) {
                const bool v  = tc.at(add_key).get<bool>();
                const auto it = find_add(typ);
                if (it == adds.end()) {
                    adds.emplace_back(typ, v);
                } else if (it->second != v) {
                    LOG_WRN("%s: %s = %d in tokenizer_config.json differs from the post_processor, using the post_processor\n", __func__, add_key.c_str(), v);
                }
            }
            if (content[typ].empty()) {
                continue;
            }
            for (const auto & a : added_tj) {
                if (a.value("content", "") == content[typ]) {
                    set_id(typ, a.at("id"));
                    break;
                }
            }
        }
    }
    for (const char * typ : typs) {
        const std::string key = std::string(typ) + "_token_id";
        common_json id = config.value(key, common_json());
        if (id.is_null() && config.contains("text_config")) {
            id = config.at("text_config").value(key, common_json());
        }
        set_id(typ, id);
    }

    gguf_set_val_str(meta, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(meta, "tokenizer.ggml.pre", pre.c_str());
    {
        std::vector<const char *> ptrs;
        for (const auto & t : tokens) {
            ptrs.push_back(t.c_str());
        }
        gguf_set_arr_str (meta, "tokenizer.ggml.tokens", ptrs.data(), ptrs.size());
        gguf_set_arr_data(meta, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    }
    if (!merges.empty()) {
        std::vector<const char *> ptrs;
        for (const auto & m : merges) {
            ptrs.push_back(m.c_str());
        }
        gguf_set_arr_str(meta, "tokenizer.ggml.merges", ptrs.data(), ptrs.size());
    } else {
        LOG_WRN("%s: tokenizer.json has no merges\n", __func__);
    }

    static const std::map<std::string, const char *> id_keys = {
        { "bos", "tokenizer.ggml.bos_token_id" }, { "eos", "tokenizer.ggml.eos_token_id" }, { "unk", "tokenizer.ggml.unknown_token_id" },
        { "sep", "tokenizer.ggml.seperator_token_id" }, { "pad", "tokenizer.ggml.padding_token_id" }, { "mask", "tokenizer.ggml.mask_token_id" },
    };
    static const std::map<std::string, const char *> add_keys = {
        { "bos", "tokenizer.ggml.add_bos_token" }, { "eos", "tokenizer.ggml.add_eos_token" }, { "sep", "tokenizer.ggml.add_sep_token" },
    };
    for (const auto & [typ, id] : ids) {
        const auto it = id_keys.find(typ);
        if (it == id_keys.end()) {
            LOG_WRN("%s: no key for special token type %s with id %" PRId64 " - skipping\n", __func__, typ.c_str(), id);
            continue;
        }
        gguf_set_val_u32(meta, it->second, exl3_u32(id, it->second));
    }
    for (const auto & [typ, v] : adds) {
        const auto it = add_keys.find(typ);
        if (it != add_keys.end()) {
            gguf_set_val_bool(meta, it->second, v);
        }
    }
    if (chat_template.is_string()) {
        gguf_set_val_str(meta, "tokenizer.chat_template", chat_template.get<std::string>().c_str());
    }

    int nfc = -1;
    int lowercase = -1;
    int strip_accents = -1;
    std::function<void(const common_json &)> parse_normalizer = [&](const common_json & n) {
        if (!n.is_object()) {
            return;
        }
        const std::string type = n.value("type", "");
        if (type == "Lowercase") {
            lowercase = 1;
        } else if (type == "NFC") {
            nfc = 1;
        } else if (type == "StripAccents") {
            strip_accents = 1;
        } else if (type == "BertNormalizer") {
            if (n.contains("lowercase")) {
                lowercase = n.at("lowercase").get<bool>();
            }
            if (n.contains("strip_accents")) {
                strip_accents = n.at("strip_accents").get<bool>();
            }
        } else if (type == "Sequence") {
            for (const auto & e : n.value("normalizers", common_json::array())) {
                parse_normalizer(e);
            }
        }
    };
    parse_normalizer(tj.value("normalizer", common_json()));
    if (nfc >= 0) {
        gguf_set_val_bool(meta, "tokenizer.ggml.normalizer.nfc", nfc);
    }
    if (lowercase >= 0) {
        gguf_set_val_bool(meta, "tokenizer.ggml.normalizer.lowercase", lowercase);
    }
    if (strip_accents >= 0) {
        gguf_set_val_bool(meta, "tokenizer.ggml.normalizer.strip_accents", strip_accents);
    }
}

// HF linear (or EXL3 module) of a qwen35 layer -> GGUF base name
static const struct {
    const char * hf;
    const char * gguf;
} QWEN35_LINEARS[] = {
    { "mlp.gate_proj",           "ffn_gate"    },
    { "mlp.up_proj",             "ffn_up"      },
    { "mlp.down_proj",           "ffn_down"    },
    { "self_attn.q_proj",        "attn_q"      },
    { "self_attn.k_proj",        "attn_k"      },
    { "self_attn.v_proj",        "attn_v"      },
    { "self_attn.o_proj",        "attn_output" },
    { "linear_attn.in_proj_qkv", "attn_qkv"    },
    { "linear_attn.in_proj_z",   "attn_gate"   },
    { "linear_attn.out_proj",    "ssm_out"     },
    { "linear_attn.in_proj_a",   "ssm_alpha"   },
    { "linear_attn.in_proj_b",   "ssm_beta"    },
};

// other tensors of a qwen35 layer
static const struct {
    const char * hf;
    const char * gguf;
} QWEN35_LAYER_TENSORS[] = {
    { "input_layernorm.weight",          "attn_norm.weight"           },
    { "post_attention_layernorm.weight", "post_attention_norm.weight" },
    { "self_attn.q_norm.weight",         "attn_q_norm.weight"         },
    { "self_attn.k_norm.weight",         "attn_k_norm.weight"         },
    { "linear_attn.norm.weight",         "ssm_norm.weight"            },
    { "linear_attn.A_log",               "ssm_a"                      },
    { "linear_attn.dt_bias",             "ssm_dt.bias"                },
    { "linear_attn.conv1d.weight",       "ssm_conv1d.weight"          },
};

// "model.layers.<il>.<local>" -> il, local
static bool exl3_split_layer(const std::string & name, int64_t & il, std::string & local) {
    static const std::string prefix = "model.layers.";
    if (!str_starts_with(name, prefix)) {
        return false;
    }
    const size_t dot = name.find('.', prefix.size());
    if (dot == std::string::npos || dot == prefix.size()) {
        return false;
    }
    const std::string num = name.substr(prefix.size(), dot - prefix.size());
    if (num.find_first_not_of("0123456789") != std::string::npos) {
        return false;
    }
    il    = std::stoll(num);
    local = name.substr(dot + 1);
    return true;
}

// the <arch>.* and general.* keys of Qwen3_5TextModel.set_gguf_parameters
static void exl3_qwen35_hparams(gguf_context * meta, const common_json & hp, const std::string & name, uint32_t ftype) {
    // config fields the converter would turn into keys that this importer does not write
    static const char * refused[] = {
        "moe_intermediate_size", "shared_expert_intermediate_size", "num_local_experts", "num_experts", "n_routed_experts",
        "num_experts_per_tok", "num_experts_per_token", "top_k_experts", "n_group", "topk_group",
        "prefix_dense_intermediate_size", "dense_intermediate_size", "layer_norm_eps", "layer_norm_epsilon", "norm_epsilon",
        "local_rope_theta", "rope_local_theta", "rope_theta_local", "swa_rope_theta", "rope_local_base_freq",
        "score_function", "scoring_func", "score_func", "moe_router_activation", "moe_router_activation_func",
        "expert_selection_fn", "router_score_func",
    };
    for (const char * k : refused) {
        if (hp.contains(k) && !hp.at(k).is_null()) {
            throw std::runtime_error(string_format("config.json: %s is not supported by the qwen35 importer, convert the model with convert_hf_to_gguf.py", k));
        }
    }
    if (hp.contains("is_causal") && hp.at("is_causal").is_boolean() && !hp.at("is_causal").get<bool>()) {
        throw std::runtime_error("config.json: is_causal false is not supported by the qwen35 importer");
    }

    // rope parameters, with the top level values mirrored as TextModel.__init__ does
    common_json rope = hp.contains("rope_parameters") ? hp.at("rope_parameters") : hp.value("rope_scaling", common_json());
    if (rope.is_null()) {
        rope = common_json::object();
    }
    if (rope.contains("full_attention") || rope.contains("sliding_attention")) {
        throw std::runtime_error("config.json: per-attention-type rope parameters are not supported by the qwen35 importer");
    }
    if (!rope.contains("rope_theta")) {
        if (const common_json * v = exl3_find(hp, { "global_rope_theta", "rope_global_theta", "rope_theta_global", "rope_theta", "rotary_emb_base" })) {
            rope["rope_theta"] = *v;
        }
    }
    if (!rope.contains("rope_type") && rope.contains("type")) {
        rope["rope_type"] = rope.at("type");
    }
    if (!rope.contains("partial_rotary_factor")) {
        if (const common_json * v = exl3_find(hp, { "partial_rotary_factor", "rope_pct", "rope_percent" })) {
            rope["partial_rotary_factor"] = *v;
        }
    }
    if (rope.contains("rope_type") && !rope.at("rope_type").is_null() && rope.at("rope_type").get<std::string>() != "default") {
        throw std::runtime_error("config.json: rope_type " + rope.at("rope_type").get<std::string>() + " is not supported by the qwen35 importer");
    }

    const int64_t n_layer = exl3_get_int(hp, "num_hidden_layers");

    gguf_set_val_str(meta, "general.architecture", "qwen35");
    gguf_set_val_str(meta, "general.type", "model");
    gguf_set_val_str(meta, "general.name", name.c_str());

    // no MTP layer: block_count counts only the layers of the checkpoint
    gguf_set_val_u32(meta, "qwen35.block_count", exl3_u32(n_layer, "num_hidden_layers"));
    if (const common_json * v = exl3_find(hp, { "max_position_embeddings", "n_ctx", "n_positions", "max_length", "max_sequence_length", "model_max_length" })) {
        gguf_set_val_u32(meta, "qwen35.context_length", exl3_u32(v->get<long long>(), "context length"));
    }
    if (const common_json * v = exl3_find(hp, { "hidden_size", "n_embd", "dim" })) {
        gguf_set_val_u32(meta, "qwen35.embedding_length", exl3_u32(v->get<long long>(), "hidden_size"));
    }
    if (const common_json * v = exl3_find(hp, { "intermediate_size", "n_inner", "hidden_dim" })) {
        gguf_set_val_u32(meta, "qwen35.feed_forward_length", exl3_u32(v->get<long long>(), "intermediate_size"));
    }
    if (const common_json * v = exl3_find(hp, { "num_attention_heads", "n_head", "n_heads" })) {
        gguf_set_val_u32(meta, "qwen35.attention.head_count", exl3_u32(v->get<long long>(), "num_attention_heads"));
    }
    if (const common_json * v = exl3_find(hp, { "num_key_value_heads", "n_kv_heads" })) {
        gguf_set_val_u32(meta, "qwen35.attention.head_count_kv", exl3_u32(v->get<long long>(), "num_key_value_heads"));
    }
    std::vector<int32_t> sections = { 11, 11, 10, 0 };
    const bool has_sections = rope.contains("mrope_section");
    if (has_sections) {
        sections.clear();
        for (const auto & s : rope.at("mrope_section")) {
            sections.push_back((int32_t) s.get<long long>());
        }
        sections.resize(std::max<size_t>(sections.size(), 4), 0);
        sections.resize(4);
        gguf_set_arr_data(meta, "qwen35.rope.dimension_sections", GGUF_TYPE_INT32, sections.data(), sections.size());
    }
    if (rope.contains("rope_theta") && !rope.at("rope_theta").is_null()) {
        gguf_set_val_f32(meta, "qwen35.rope.freq_base", (float) rope.at("rope_theta").get<double>());
    }
    if (const common_json * v = exl3_find(hp, { "rms_norm_eps", "norm_eps" })) {
        gguf_set_val_f32(meta, "qwen35.attention.layer_norm_rms_epsilon", (float) v->get<double>());
    }
    const bool has_head_dim = hp.contains("head_dim") && !hp.at("head_dim").is_null();
    if (has_head_dim) {
        const uint32_t head_dim = exl3_u32(exl3_get_int(hp, "head_dim"), "head_dim");
        gguf_set_val_u32(meta, "qwen35.attention.key_length", head_dim);
        gguf_set_val_u32(meta, "qwen35.attention.value_length", head_dim);
    }
    gguf_set_val_u32(meta, "general.file_type", ftype);

    gguf_set_val_u32(meta, "qwen35.ssm.conv_kernel",    exl3_u32(exl3_get_int(hp, "linear_conv_kernel_dim"), "linear_conv_kernel_dim"));
    gguf_set_val_u32(meta, "qwen35.ssm.state_size",     exl3_u32(exl3_get_int(hp, "linear_key_head_dim"), "linear_key_head_dim"));
    gguf_set_val_u32(meta, "qwen35.ssm.group_count",    exl3_u32(exl3_get_int(hp, "linear_num_key_heads"), "linear_num_key_heads"));
    gguf_set_val_u32(meta, "qwen35.ssm.time_step_rank", exl3_u32(exl3_get_int(hp, "linear_num_value_heads"), "linear_num_value_heads"));
    gguf_set_val_u32(meta, "qwen35.ssm.inner_size",
        exl3_u32(exl3_get_int(hp, "linear_value_head_dim") * exl3_get_int(hp, "linear_num_value_heads"), "linear value size"));
    if (hp.contains("layer_types") && !hp.at("layer_types").is_null()) {
        std::vector<int8_t> recurrent;
        for (const auto & t : hp.at("layer_types")) {
            recurrent.push_back(t.get<std::string>() == "linear_attention");
        }
        if ((int64_t) recurrent.size() != n_layer) {
            throw std::runtime_error(string_format("config.json: layer_types has %zu entries, expected %" PRId64, recurrent.size(), n_layer));
        }
        gguf_set_arr_data(meta, "qwen35.attention.recurrent_layers", GGUF_TYPE_BOOL, recurrent.data(), recurrent.size());
    }
    gguf_set_val_u32(meta, "qwen35.full_attention_interval", exl3_u32(hp.value("full_attention_interval", (long long) 4), "full_attention_interval"));
    const int64_t rope_dim = has_head_dim ? exl3_get_int(hp, "head_dim") : exl3_get_int(hp, "hidden_size") / exl3_get_int(hp, "num_attention_heads");
    const double  partial  = rope.contains("partial_rotary_factor") ? rope.at("partial_rotary_factor").get<double>() : 0.25;
    gguf_set_val_u32(meta, "qwen35.rope.dimension_count", exl3_u32((int64_t) ((double) rope_dim * partial), "rope dimension count"));
    if (!has_sections) {
        gguf_set_arr_data(meta, "qwen35.rope.dimension_sections", GGUF_TYPE_INT32, sections.data(), sections.size());
    }
    gguf_set_val_u32(meta, "general.quantization_version", GGML_QNT_VERSION);
}

// [K, V, unit] -> [V, K, unit] on the elements [start, end) of one axis of a row-major tensor, the rest is copied
static void exl3_permute(const uint8_t * src, uint8_t * dst, const std::vector<int64_t> & shape, size_t es, const exl3_out_tensor & t) {
    int64_t outer = 1;
    int64_t inner = (int64_t) es;
    for (int i = 0; i < t.perm_axis; i++) {
        outer *= shape[i];
    }
    for (size_t i = t.perm_axis + 1; i < shape.size(); i++) {
        inner *= shape[i];
    }
    const int64_t dim  = shape[t.perm_axis];
    const int64_t unit = t.perm_unit*inner;
    GGML_ASSERT(dim - t.perm_start == t.perm_k*t.perm_v*t.perm_unit);
    for (int64_t o = 0; o < outer; o++) {
        const uint8_t * s = src + o*dim*inner;
        uint8_t       * d = dst + o*dim*inner;
        memcpy(d, s, t.perm_start*inner);
        for (int64_t g = 0; g < t.perm_k*t.perm_v; g++) {
            const int64_t v = g / t.perm_k;
            const int64_t k = g % t.perm_k;
            memcpy(d + t.perm_start*inner + g*unit, s + t.perm_start*inner + (k*t.perm_v + v)*unit, unit);
        }
    }
}

bool common_exl3_is_dir(const std::string & path) {
    std::error_code ec;
    return fs::is_directory(path, ec);
}

common_exl3_dir_ptr common_exl3_dir_open(const std::string & path) {
    const fs::path dir(path);
    common_exl3_dir_ptr d(new common_exl3_dir);

    const common_json config = exl3_read_json(dir / "config.json");
    common_json hp = config;
    if (config.contains("text_config") && config.at("text_config").is_object()) {
        for (const auto & [k, v] : config.at("text_config").items()) {
            hp[k] = v;
        }
    }
    // only the quant_method and the provenance are taken from it: quantization_config.json (tensor_storage) is not read,
    // it lists the modules of the default model walk only, the safetensors headers are the inventory
    const common_json qc = config.value("quantization_config", common_json());
    if (!qc.is_object() || qc.value("quant_method", "") != "exl3") {
        throw std::runtime_error("config.json has no quantization_config with quant_method exl3");
    }
    std::string arch;
    if (config.contains("architectures") && config.at("architectures").is_array() && !config.at("architectures").empty()) {
        arch = config.at("architectures").at(0).get<std::string>();
    }
    if (arch != "Qwen3_5ForConditionalGeneration" && arch != "Qwen3_5ForCausalLM") {
        throw std::runtime_error("only Qwen3.5 (qwen35) EXL3 models are loaded from a directory, not " + arch + ": convert the model with convert_hf_to_gguf.py");
    }
    if (fs::exists(dir / "modules.json")) {
        throw std::runtime_error("sentence-transformers pooling (modules.json) is not supported, convert the model with convert_hf_to_gguf.py");
    }

    // every *.safetensors file of the directory, also when there is no index and for side files the index does not list
    for (const auto & e : fs::directory_iterator(dir)) {
        if (e.is_regular_file() && e.path().extension() == ".safetensors") {
            d->shards.push_back(e.path());
        }
    }
    std::sort(d->shards.begin(), d->shards.end());
    if (d->shards.empty()) {
        throw std::runtime_error("no .safetensors files");
    }
    std::map<std::string, std::string> owner;
    for (size_t i = 0; i < d->shards.size(); i++) {
        st_read_header(d->shards[i], (int) i, d->st, owner);
        d->files.emplace_back(d->shards[i], std::ios::binary);
    }
    // an index is not needed, but it must agree with the files
    if (fs::is_regular_file(dir / "model.safetensors.index.json")) {
        const common_json index = exl3_read_json(dir / "model.safetensors.index.json");
        for (const auto & [name, file] : index.at("weight_map").items()) {
            const auto it = owner.find(name);
            if (it == owner.end() || it->second != file.get<std::string>()) {
                throw std::runtime_error("model.safetensors.index.json lists " + name + " in " + file.get<std::string>() + ", but it is not there");
            }
        }
    }

    // the converter's names drop "language_model."
    std::map<std::string, std::string> names;
    for (const auto & [orig, t] : d->st) {
        const std::string n = str_replace_all(orig, "language_model.", "");
        if (!names.emplace(n, orig).second) {
            throw std::runtime_error("tensors " + orig + " and " + names.at(n) + " have the same name " + n);
        }
    }

    int n_vision = 0;
    int n_mtp    = 0;
    std::map<std::string, std::map<std::string, std::string>> modules;
    std::map<std::string, std::string> plain;
    for (const auto & [n, orig] : names) {
        // the vision tower is for an mmproj, not for the text model
        if (str_starts_with(n, "model.visual.") || str_starts_with(n, "visual.")) {
            n_vision++;
            continue;
        }
        // MTP layers are not loaded: there is no EXL3 MTP support yet
        if (str_starts_with(n, "mtp.") || str_starts_with(n, "model.mtp.")) {
            n_mtp++;
            continue;
        }
        const size_t dot  = n.rfind('.');
        const std::string part = dot == std::string::npos ? "" : n.substr(dot + 1);
        if (part == "trellis" || part == "suh" || part == "svh" || part == "mcg" || part == "mul1" || part == "su" || part == "sv") {
            modules[n.substr(0, dot)][part] = n;
        } else {
            plain[n] = orig;
        }
    }
    for (auto it = modules.begin(); it != modules.end(); ) {
        if (it->second.count("trellis") == 0) {
            for (const auto & [part, n] : it->second) {
                plain[n] = names.at(n);
            }
            it = modules.erase(it);
        } else {
            ++it;
        }
    }
    if (modules.empty()) {
        throw std::runtime_error("quant_method is exl3 but the checkpoint has no .trellis tensors");
    }

    const int64_t n_layer   = exl3_get_int(hp, "num_hidden_layers");
    const int64_t n_k_heads = exl3_get_int(hp, "linear_num_key_heads");
    const int64_t n_v_heads = exl3_get_int(hp, "linear_num_value_heads");
    const int64_t head_k    = exl3_get_int(hp, "linear_key_head_dim");
    const int64_t head_v    = exl3_get_int(hp, "linear_value_head_dim");
    const bool    reorder   = n_k_heads > 0 && n_v_heads > 0 && n_k_heads != n_v_heads;
    if (reorder && n_v_heads % n_k_heads != 0) {
        throw std::runtime_error("config.json: linear_num_value_heads is not a multiple of linear_num_key_heads");
    }
    const int64_t n_v_per_k = reorder ? n_v_heads / n_k_heads : 1;
    const int64_t qk_dim    = 2*head_k*n_k_heads;
    const int64_t v_dim     = n_v_heads*head_v;

    // tie_word_embeddings may be set while lm_head is stored (the 4B EXL3 file): a stored head is always loaded
    const bool tie      = hp.value("tie_word_embeddings", false);
    const bool has_head = modules.count("lm_head") > 0 || plain.count("lm_head.weight") > 0;
    if (!has_head && !tie) {
        throw std::runtime_error("config.json does not tie the output head, but there is no lm_head");
    }
    if (has_head && tie) {
        LOG_INF("%s: tie_word_embeddings is set, but lm_head is stored: the stored head is used\n", __func__);
    }
    // the MTP layer count comes from the tensors, not from mtp_num_hidden_layers (the 4B EXL3 file declares 1 and has none)
    if (n_mtp > 0) {
        LOG_WRN("%s: %d MTP tensors are not loaded, EXL3 MTP layers are not supported yet\n", __func__, n_mtp);
    } else if (hp.value("mtp_num_hidden_layers", (long long) 0) > 0) {
        LOG_INF("%s: config.json declares %lld MTP layers, the checkpoint has none\n", __func__, hp.value("mtp_num_hidden_layers", (long long) 0));
    }
    if (n_vision > 0) {
        LOG_INF("%s: %d vision tensors are not loaded (text model)\n", __func__, n_vision);
    }

    auto gguf_base = [&](const std::string & base) -> std::string {
        if (base == "lm_head") {
            return "output";
        }
        int64_t il;
        std::string local;
        if (exl3_split_layer(base, il, local) && il < n_layer) {
            for (const auto & e : QWEN35_LINEARS) {
                if (local == e.hf) {
                    return "blk." + std::to_string(il) + "." + e.gguf;
                }
            }
        }
        return "";
    };
    auto read_raw = [&](const std::string & orig, void * dst) {
        const st_tensor & s = d->st.at(orig);
        std::ifstream & f = d->files[s.shard];
        f.seekg((std::streamoff) s.offs);
        if (!f.read((char *) dst, (std::streamsize) s.size)) {
            throw std::runtime_error("cannot read " + orig);
        }
    };

    // EXL3 linears, in the order of conversion/exl3.py
    std::string codebook;
    std::vector<std::string> permuted;
    for (const auto & [base, parts] : modules) {
        if (parts.count("su") || parts.count("sv")) {
            throw std::runtime_error(base + ": legacy packed su/sv EXL3 tensors are not supported");
        }
        if (parts.count("mcg") && parts.count("mul1")) {
            throw std::runtime_error(base + ": both .mcg and .mul1 codebook markers are present");
        }
        if (!parts.count("suh") || !parts.count("svh")) {
            throw std::runtime_error(base + ": EXL3 linear without .suh/.svh");
        }
        const std::string cb = parts.count("mcg") ? "mcg" : parts.count("mul1") ? "mul1" : "3inst";
        if (cb != "3inst") {
            const st_tensor & m = d->st.at(names.at(parts.at(cb)));
            if (m.dtype != "I32" || !m.shape.empty()) {
                throw std::runtime_error(base + ": malformed ." + cb + " marker");
            }
            uint8_t b[4];
            read_raw(names.at(parts.at(cb)), b);
            const uint32_t marker = (uint32_t) b[0] | (uint32_t) b[1] << 8 | (uint32_t) b[2] << 16 | (uint32_t) b[3] << 24;
            if (marker != (cb == "mcg" ? EXL3_MARKER_MCG : EXL3_MARKER_MUL1)) {
                throw std::runtime_error(string_format("%s: .%s marker holds 0x%08x", base.c_str(), cb.c_str(), marker));
            }
        }
        if (!codebook.empty() && codebook != cb) {
            throw std::runtime_error("EXL3 checkpoints that mix codebooks are not supported");
        }
        codebook = cb;

        const std::string & n_tr = names.at(parts.at("trellis"));
        const std::string & n_su = names.at(parts.at("suh"));
        const std::string & n_sv = names.at(parts.at("svh"));
        const st_tensor & tr = d->st.at(n_tr);
        const st_tensor & su = d->st.at(n_su);
        const st_tensor & sv = d->st.at(n_sv);
        if (tr.shape.size() != 3) {
            throw std::runtime_error(base + ": a " + std::to_string(tr.shape.size()) + "-D trellis (row-ring embedding or n-gram table) is not supported yet");
        }
        if (tr.dtype != "I16" || su.dtype != "F16" || sv.dtype != "F16") {
            throw std::runtime_error(base + ": unexpected EXL3 dtypes");
        }
        const int64_t k  = tr.shape[0]*16;
        const int64_t n  = tr.shape[1]*16;
        const int64_t tw = tr.shape[2];
        const bool rate_ok = (tw % 16 == 0 && tw >= 16 && tw <= 128) || (cb == "mul1" && (tw == 24 || tw == 40 || tw == 56));
        if (!rate_ok || k % 128 || n % 128 || su.shape != std::vector<int64_t>{ k } || sv.shape != std::vector<int64_t>{ n }) {
            throw std::runtime_error(base + ": unsupported EXL3 rate or inconsistent shapes");
        }
        const std::string gb = gguf_base(base);
        if (gb.empty()) {
            throw std::runtime_error("EXL3 linear " + base + " is not known to the qwen35 importer");
        }

        exl3_out_tensor t_tr = { gb + ".exl3_trellis", n_tr, GGML_TYPE_I16, tr.shape };
        exl3_out_tensor t_su = { gb + ".exl3_suh",     n_su, GGML_TYPE_F16, su.shape };
        exl3_out_tensor t_sv = { gb + ".exl3_svh",     n_sv, GGML_TYPE_F16, sv.shape };
        // the V head reorder of qwen35 on whole 128-blocks, a lossless reorder of tiles (H is block diagonal)
        if (reorder && base.find("linear_attn.") != std::string::npos) {
            if (head_v % 128 || qk_dim % 128) {
                throw std::runtime_error(base + ": V heads do not fill whole EXL3 128-blocks");
            }
            int     axis  = -1;
            int64_t start = 0;
            if (str_ends_with(base, ".in_proj_qkv") && n == qk_dim + v_dim) {
                axis = 1; start = qk_dim;
            } else if (str_ends_with(base, ".in_proj_z") && n == v_dim) {
                axis = 1;
            } else if (str_ends_with(base, ".out_proj") && k == v_dim) {
                axis = 0;
            } else {
                throw std::runtime_error(base + ": no EXL3 V head reorder for this projection");
            }
            t_tr.perm_axis = axis;
            t_tr.perm_start = start/16;
            t_tr.perm_unit  = head_v/16;
            t_tr.perm_k     = n_k_heads;
            t_tr.perm_v     = n_v_per_k;
            exl3_out_tensor & t_s = axis == 0 ? t_su : t_sv;
            t_s.perm_axis  = 0;
            t_s.perm_start = start;
            t_s.perm_unit  = head_v;
            t_s.perm_k     = n_k_heads;
            t_s.perm_v     = n_v_per_k;
            permuted.push_back(gb);
        }
        d->out.push_back(t_tr);
        d->out.push_back(t_su);
        d->out.push_back(t_sv);
    }

    // the other tensors, with the transforms of Qwen3NextModel and _LinearAttentionVReorderBase.modify_tensors
    for (const auto & [n, orig] : plain) {
        const st_tensor & s = d->st.at(orig);
        exl3_out_tensor t;
        t.src   = orig;
        t.shape = s.shape;
        int64_t il = -1;
        std::string local;
        if (n == "model.embed_tokens.weight") {
            t.name = "token_embd.weight";
        } else if (n == "model.norm.weight") {
            t.name = "output_norm.weight";
        } else if (n == "lm_head.weight") {
            t.name = "output.weight";
        } else if (exl3_split_layer(n, il, local) && il < n_layer) {
            for (const auto & e : QWEN35_LAYER_TENSORS) {
                if (local == e.hf) {
                    t.name = "blk." + std::to_string(il) + "." + e.gguf;
                }
            }
            for (const auto & e : QWEN35_LINEARS) {
                if (local == std::string(e.hf) + ".weight") {
                    t.name = "blk." + std::to_string(il) + "." + e.gguf + ".weight";
                }
            }
        }
        if (t.name.empty()) {
            throw std::runtime_error("tensor " + orig + " is not known to the qwen35 importer");
        }

        if (local == "linear_attn.conv1d.weight") {
            if (t.shape.size() != 3 || t.shape[1] != 1 || t.shape[2] <= 1) {
                throw std::runtime_error(orig + ": unexpected conv1d shape");
            }
            t.shape = { t.shape[0], t.shape[2] };
        }
        if (reorder && str_starts_with(local, "linear_attn.")) {
            auto set_perm = [&](int axis, int64_t start, int64_t unit) {
                if (t.shape.size() <= (size_t) axis || t.shape[axis] - start != n_v_heads*unit) {
                    throw std::runtime_error(orig + ": shape does not match the linear attention heads");
                }
                t.perm_axis  = axis;
                t.perm_start = start;
                t.perm_unit  = unit;
                t.perm_k     = n_k_heads;
                t.perm_v     = n_v_per_k;
            };
            if (local == "linear_attn.in_proj_qkv.weight") {
                set_perm(0, qk_dim, head_v);
            } else if (local == "linear_attn.in_proj_z.weight") {
                set_perm(0, 0, head_v);
            } else if (local == "linear_attn.in_proj_a.weight" || local == "linear_attn.in_proj_b.weight") {
                set_perm(0, 0, 1);
            } else if (local == "linear_attn.A_log" || local == "linear_attn.dt_bias") {
                if (t.shape.size() != 1) {
                    throw std::runtime_error(orig + ": expected a 1-D tensor");
                }
                set_perm(0, 0, 1);
            } else if (local == "linear_attn.conv1d.weight") {
                set_perm(0, qk_dim, head_v);
            } else if (local == "linear_attn.out_proj.weight") {
                set_perm(1, 0, head_v);
            }
        }
        if (local == "linear_attn.A_log") {
            t.arith = 2;
        } else if (str_ends_with(n, "norm.weight") && !str_ends_with(n, "linear_attn.norm.weight")) {
            t.arith = 1;
        }

        // the type rule of ModelBase.prepare_tensors for EXL3 checkpoints: 1-D, norms, conv1d and non-.weight tensors are F32, other tensors keep the stored type
        if (t.shape.size() <= 1 || str_ends_with(t.name, "_norm.weight") || str_ends_with(t.name, ".ssm_conv1d.weight") || !str_ends_with(t.name, ".weight")) {
            if (s.dtype != "F32" && s.dtype != "F16" && s.dtype != "BF16") {
                throw std::runtime_error(orig + ": unexpected dtype " + s.dtype);
            }
            t.type = GGML_TYPE_F32;
        } else if (s.dtype == "F16" || s.dtype == "BF16") {
            t.type = st_dtype_ggml(s.dtype);
        } else {
            throw std::runtime_error(orig + ": " + s.dtype + " matrices are not supported by the importer, convert the model with convert_hf_to_gguf.py");
        }
        if (t.arith && t.type != GGML_TYPE_F32) {
            throw std::runtime_error(orig + ": unexpected shape");
        }
        d->out.push_back(t);
    }

    // general.file_type: the type of the first float matrix, as the converter guesses the output type
    uint32_t ftype = 1; // MOSTLY_F16
    for (const auto & [n, orig] : names) {
        if (str_starts_with(n, "model.visual.") || str_starts_with(n, "visual.") || str_starts_with(n, "mtp.") || str_starts_with(n, "model.mtp.")) {
            continue;
        }
        const st_tensor & s = d->st.at(orig);
        if (s.shape.size() >= 2 && (s.dtype == "BF16" || s.dtype == "F16")) {
            ftype = s.dtype == "BF16" ? 32 : 1; // MOSTLY_BF16 or MOSTLY_F16
            break;
        }
    }

    d->meta = gguf_init_empty();
    exl3_qwen35_hparams(d->meta, hp, dir.filename().string().empty() ? dir.parent_path().filename().string() : dir.filename().string(), ftype);

    gguf_set_val_u32(d->meta, "exl3.version", 1);
    gguf_set_val_str(d->meta, "exl3.codebook", codebook.c_str());
    gguf_set_val_u32(d->meta, "exl3.hadamard_size", 128);
    if (!permuted.empty()) {
        std::vector<const char *> ptrs;
        for (const auto & p : permuted) {
            ptrs.push_back(p.c_str());
        }
        gguf_set_val_str(d->meta, "exl3.permutation", EXL3_PERMUTATION);
        gguf_set_arr_str(d->meta, "exl3.permuted_tensors", ptrs.data(), ptrs.size());
    }
    gguf_set_val_str(d->meta, "exl3.source.quantization_config", exl3_py_json(qc).c_str());

    exl3_set_vocab(d->meta, dir, config, hp);

    ggml_init_params params = { ggml_tensor_overhead()*d->out.size(), NULL, true };
    d->ctx_meta = ggml_init(params);
    for (size_t i = 0; i < d->out.size(); i++) {
        const exl3_out_tensor & t = d->out[i];
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t j = 0; j < t.shape.size(); j++) {
            ne[j] = t.shape[t.shape.size() - 1 - j];
        }
        ggml_tensor * cur = ggml_new_tensor(d->ctx_meta, t.type, std::max<int>(1, (int) t.shape.size()), ne);
        ggml_set_name(cur, t.name.c_str());
        if (!d->out_idx.emplace(t.name, i).second) {
            throw std::runtime_error("two tensors map to " + t.name);
        }
        gguf_add_tensor(d->meta, cur);
    }

    LOG_INF("%s: %s: %zu tensors (%zu EXL3 linears, codebook %s, %zu block-permuted) from %zu safetensors file(s)\n",
        __func__, path.c_str(), d->out.size(), modules.size(), codebook.c_str(), permuted.size(), d->shards.size());
    return d;
}

const gguf_context * common_exl3_dir_gguf(const common_exl3_dir * d) {
    return d->meta;
}

void common_exl3_dir_read(const common_exl3_dir * d, const char * name, void * dst, size_t size) {
    const auto it = d->out_idx.find(name);
    if (it == d->out_idx.end()) {
        throw std::runtime_error(string_format("tensor %s is not in the import", name));
    }
    const exl3_out_tensor & t = d->out[it->second];
    const st_tensor & s = d->st.at(t.src);
    int64_t ne = 1;
    for (const int64_t x : t.shape) {
        ne *= x;
    }
    if (size != (size_t) ne*ggml_type_size(t.type)) {
        throw std::runtime_error(string_format("tensor %s: %zu bytes asked, it has %zu", name, size, (size_t) ne*ggml_type_size(t.type)));
    }

    std::ifstream & f = d->files[s.shard];
    const bool copy = t.type != GGML_TYPE_F32 || s.dtype == "F32";
    std::vector<uint8_t> raw;
    uint8_t * buf = (uint8_t *) dst;
    if (t.perm_axis >= 0 || !copy) {
        raw.resize(s.size);
        buf = raw.data();
    }
    f.seekg((std::streamoff) s.offs);
    if (!f.read((char *) buf, (std::streamsize) s.size)) {
        throw std::runtime_error(string_format("cannot read tensor %s", t.src.c_str()));
    }
    if (t.perm_axis >= 0) {
        std::vector<uint8_t> tmp(s.size);
        exl3_permute(raw.data(), tmp.data(), t.shape, st_dtype_size(s.dtype), t);
        if (copy) {
            memcpy(dst, tmp.data(), s.size);
        } else {
            raw.swap(tmp);
        }
    }
    if (t.type != GGML_TYPE_F32) {
        return;
    }

    float * y = (float *) dst;
    if (!copy) {
        for (int64_t i = 0; i < ne; i++) {
            uint16_t h;
            memcpy(&h, raw.data() + 2*i, 2);
            if (s.dtype == "BF16") {
                const uint32_t u = (uint32_t) h << 16;
                memcpy(&y[i], &u, 4);
            } else {
                y[i] = ggml_fp16_to_fp32(h);
            }
        }
    }
    if (t.arith == 1) {
        for (int64_t i = 0; i < ne; i++) {
            y[i] = y[i] + 1.0f;
        }
    } else if (t.arith == 2) {
        // one rounding from float64, as the converter does for EXL3 checkpoints
        for (int64_t i = 0; i < ne; i++) {
            y[i] = (float) -std::exp((double) y[i]);
        }
    }
}

static void exl3_set_tensor_data(ggml_tensor * t, void * ud) {
    const auto * d = (const common_exl3_dir *) ud;
    const size_t n = ggml_nbytes(t);
    if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
        common_exl3_dir_read(d, ggml_get_name(t), t->data, n);
        return;
    }
    std::vector<uint8_t> buf(n);
    common_exl3_dir_read(d, ggml_get_name(t), buf.data(), n);
    ggml_backend_tensor_set(t, buf.data(), 0, n);
}

llama_model * common_exl3_load_model(const std::string & path, llama_model_params params) {
    if (!common_exl3_is_dir(path)) {
        return llama_model_load_from_file(path.c_str(), params);
    }
    try {
        const common_exl3_dir_ptr d = common_exl3_dir_open(path);
        return llama_model_init_from_user(d->meta, exl3_set_tensor_data, d.get(), params);
    } catch (const std::exception & e) {
        LOG_ERR("%s: cannot load %s: %s\n", __func__, path.c_str(), e.what());
        return nullptr;
    }
}

