// The HTTP side of the server hooks. Built in llama-server-impl, which has server-http.cpp: server-context is also
// linked without it (test-server-component).

#include "server-hooks.h"

#include "server-http.h"

// link-time wrapper, see hooks/llama/llama-hooks.cmake
extern "C" bool __real__ZN19server_http_context5startEv(server_http_context * self);
extern "C" __attribute__((visibility("default"))) bool __wrap__ZN19server_http_context5startEv(server_http_context * self) {
    server_hooks_register_routes(*self);
    return __real__ZN19server_http_context5startEv(self);
}
