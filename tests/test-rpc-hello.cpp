// The RPC client's hello against servers of other protocol versions (ggml-rpc.cpp negotiate_hello, ggml-rpc.h). A fake
// server answers the HELLO with a chosen version; a forked child connects with ggml_backend_rpc_add_server. The client
// must go on (send its next command) only for a server minor in [RPC_PROTO_MIN_SERVER_MINOR_VERSION, RPC_PROTO_MINOR_VERSION].
// It accepted any older minor: an 8.0 server (GGML_OP_COUNT 101) then received the op values 101 / 102 of
// GGML_OP_SPARSE_ATTN_PLAN / GGML_OP_SPARSE_ATTN and crashed or computed no attention, instead of a version error.
// No device is used.
#include "ggml-rpc.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifndef RPC_PROTO_MIN_SERVER_MINOR_VERSION
#define RPC_PROTO_MIN_SERVER_MINOR_VERSION 0   // a ggml-rpc.h without the floor: the hello accepted every older minor
#endif

static bool recv_all(int fd, void * p, size_t n) {
    uint8_t * b = (uint8_t *) p;
    while (n > 0) {
        const ssize_t r = recv(fd, b, n, 0);
        if (r <= 0) {
            return false;
        }
        b += r;
        n -= (size_t) r;
    }
    return true;
}

static bool send_all(int fd, const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    while (n > 0) {
        const ssize_t r = send(fd, b, n, MSG_NOSIGNAL);
        if (r <= 0) {
            return false;
        }
        b += r;
        n -= (size_t) r;
    }
    return true;
}

// one client connection against a server announcing major.minor; true if the client sent a command after the hello
static bool client_goes_on(uint8_t major, uint8_t minor) {
    const int lfd = socket(AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a = {};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    if (bind(lfd, (sockaddr *) &a, sizeof(a)) != 0 || listen(lfd, 4) != 0) {
        perror("bind/listen");
        exit(2);
    }
    socklen_t al = sizeof(a);
    getsockname(lfd, (sockaddr *) &a, &al);
    const std::string endpoint = "127.0.0.1:" + std::to_string(ntohs(a.sin_port));

    fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) {
        close(lfd);
        setenv("GGML_NO_BACKTRACE", "1", 1);
        // a version mismatch aborts here (GGML_ABORT "RPC handshake failed"); an accepted hello asks the device count
        ggml_backend_rpc_add_server(endpoint.c_str());
        _exit(0);
    }
    const int fd = accept(lfd, nullptr, nullptr);
    bool goes_on = false;
    // | cmd (1) | size (8) | hello_req (24 bytes of conn_caps) |
    uint8_t cmd = 0;
    uint64_t size = 0;
    uint8_t req[64];
    if (fd >= 0 && recv_all(fd, &cmd, 1) && cmd == 14 && recv_all(fd, &size, 8) && size <= sizeof(req) && recv_all(fd, req, size)) {
        // | size (8) | major, minor, patch, padding, conn_caps (24 bytes, zero: no transport upgrade) |
        uint8_t rsp[28] = {};
        rsp[0] = major;
        rsp[1] = minor;
        const uint64_t rsize = sizeof(rsp);
        send_all(fd, &rsize, 8);
        send_all(fd, rsp, sizeof(rsp));
        uint8_t next = 0;
        goes_on = recv_all(fd, &next, 1);   // EOF when the client rejected the hello and exited
        if (goes_on) {
            // the device count request: answer 0 devices so that add_server returns
            uint64_t n = 0;
            uint8_t body[64];
            if (recv_all(fd, &n, 8) && n <= sizeof(body) && recv_all(fd, body, n)) {
                const uint64_t csize = 4;
                const uint32_t count = 0;
                send_all(fd, &csize, 8);
                send_all(fd, &count, 4);
            }
        }
    } else {
        printf("  fake server: no HELLO from the client\n");
    }
    if (fd >= 0) {
        close(fd);
    }
    close(lfd);
    int status = 0;
    for (int i = 0; i < 100; ++i) {
        if (waitpid(pid, &status, WNOHANG) == pid) {
            return goes_on;
        }
        usleep(50000);
    }
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    return goes_on;
}

int main() {
    int fail = 0;
    struct { uint8_t major, minor; bool want; } cases[] = {
        { RPC_PROTO_MAJOR_VERSION, RPC_PROTO_MINOR_VERSION, true },
        { RPC_PROTO_MAJOR_VERSION, (uint8_t) (RPC_PROTO_MINOR_VERSION + 1), false },
        // the 8.0 server of origin/master: its op enum ends at 101, before the two sparse-attention ops (ggml-rpc.h 8.1)
        { RPC_PROTO_MAJOR_VERSION, 0, false },
        { (uint8_t) (RPC_PROTO_MAJOR_VERSION - 1), RPC_PROTO_MINOR_VERSION, false },
    };
    printf("client %d.%d, oldest server minor %d\n", RPC_PROTO_MAJOR_VERSION, RPC_PROTO_MINOR_VERSION, RPC_PROTO_MIN_SERVER_MINOR_VERSION);
    for (const auto & c : cases) {
        const bool got = client_goes_on(c.major, c.minor);
        printf("  server %d.%d: client %s (want %s)\n", c.major, c.minor, got ? "goes on" : "rejects", c.want ? "goes on" : "rejects");
        if (got != c.want) {
            fail++;
        }
    }
    printf("test-rpc-hello: %s\n", fail == 0 ? "OK" : "FAILED");
    return fail == 0 ? 0 : 1;
}
