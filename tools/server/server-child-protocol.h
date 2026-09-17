#pragma once

#include <string_view>

inline constexpr std::string_view SERVER_CHILD_STATE_PREFIX = "cmd_child_to_router:state:";

inline std::string_view server_child_state_line(std::string_view line) {
    constexpr std::string_view ansi_reset = "\x1b[0m";

    // Colored logs write the reset after the newline, before the next raw protocol line.
    while (line.compare(0, ansi_reset.size(), ansi_reset) == 0) {
        line.remove_prefix(ansi_reset.size());
    }
    if (line.compare(0, SERVER_CHILD_STATE_PREFIX.size(), SERVER_CHILD_STATE_PREFIX) != 0) {
        return {};
    }
    return line;
}
