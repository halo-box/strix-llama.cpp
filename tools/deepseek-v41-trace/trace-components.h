#pragma once

#include <optional>
#include <stdexcept>
#include <string>

struct dsv41_trace_descriptor {
    std::string component;
    int layer;
    const char * semantic_id_space;
};

inline bool dsv41_trace_expected_layer(const std::string & component, int layer) {
    if (component == "engram.row_ids") {
        return layer == 1 || layer == 14;
    }
    if (component == "expert.ids" || component == "expert.weights" || component == "attn.source") {
        return layer >= 0 && layer < 40;
    }
    if (component == "attn.candidate_blocks") {
        return layer == 20;
    }
    if (component == "attn.candidates") {
        return layer == 24 || layer == 28 || layer == 32 || layer == 36;
    }
    return false;
}

inline std::optional<dsv41_trace_descriptor> dsv41_trace_parse_name(const std::string & name) {
    struct prefix_entry {
        const char * prefix;
        const char * component;
        const char * semantic_id_space;
    };
    static const prefix_entry prefixes[] = {
        {"dsv41.trace.engram.row_ids.l", "engram.row_ids", nullptr},
        {"dsv41.trace.expert.ids.l", "expert.ids", "original"},
        {"dsv41.trace.expert.weights.l", "expert.weights", nullptr},
        {"dsv41.trace.attn.source.l", "attn.source", nullptr},
        {"dsv41.trace.attn.candidate_blocks.l", "attn.candidate_blocks", nullptr},
        {"dsv41.trace.attn.candidates.l", "attn.candidates", nullptr},
    };
    for (const prefix_entry & entry : prefixes) {
        const std::string prefix = entry.prefix;
        if (name.rfind(prefix, 0) != 0 || name.size() == prefix.size()) {
            continue;
        }
        int layer = 0;
        for (size_t index = prefix.size(); index < name.size(); ++index) {
            const char value = name[index];
            if (value < '0' || value > '9') {
                throw std::runtime_error("malformed reserved trace tensor name: " + name);
            }
            layer = 10*layer + value - '0';
            if (layer >= 40) {
                throw std::runtime_error("unexpected reserved trace tensor layer: " + name);
            }
        }
        if (!dsv41_trace_expected_layer(entry.component, layer)) {
            throw std::runtime_error("unexpected reserved trace tensor layer: " + name);
        }
        return dsv41_trace_descriptor{entry.component, layer, entry.semantic_id_space};
    }
    if (name.rfind("dsv41.trace.", 0) == 0) {
        throw std::runtime_error("malformed reserved trace tensor name: " + name);
    }
    return std::nullopt;
}

inline bool dsv41_trace_select_name(const std::string & name) {
    return dsv41_trace_parse_name(name).has_value();
}
