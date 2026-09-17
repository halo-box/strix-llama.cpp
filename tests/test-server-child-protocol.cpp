#include "server-child-protocol.h"

#include <cassert>
#include <string>
#include <string_view>

int main() {
    constexpr std::string_view state =
        "cmd_child_to_router:state:{\"state\":\"downloading\",\"payload\":{\"result\":\"download_finished\"}}\n";

    const std::string reset_state = std::string("\x1b[0m") + std::string(state);
    const std::string repeated_reset_state = std::string("\x1b[0m\x1b[0m") + std::string(state);
    const std::string colored_state = std::string("\x1b[31m") + std::string(state);

    assert(server_child_state_line(state) == state);
    assert(server_child_state_line(reset_state) == state);
    assert(server_child_state_line(repeated_reset_state) == state);
    assert(server_child_state_line(colored_state).empty());
    assert(server_child_state_line("ordinary child log\n").empty());
    assert(server_child_state_line("\x1b[0cmd_child_to_router:state:{}\n").empty());
    return 0;
}
