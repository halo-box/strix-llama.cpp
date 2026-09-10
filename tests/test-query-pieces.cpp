#include "fattn-masked-qk-layout.h"
#include <algorithm>
#include <cassert>
#include <cstdio>

int main() {
    size_t ranges = 0;
    for (int n = 32768; n <= 1050624; n += 256) {
        const int kv_tiles = n/32;
        const int tiles = 1024;
        const int64_t work = int64_t(tiles)*kv_tiles;
        for (unsigned blocks : {1u, 3u, 7u, 20u, 40u, 64u, 1024u}) {
            const unsigned pieces = fattn_query_piece_count(tiles, kv_tiles, blocks);
            assert(pieces > 0);
            for (unsigned b = 0; b < blocks; ++b) {
                const int64_t begin = int64_t(b)*work/blocks;
                const int64_t end = (int64_t(b) + 1)*work/blocks;
                int64_t cursor = begin;
                for (unsigned p = 0; p < pieces; ++p) {
                    const int64_t tile = begin/kv_tiles + int64_t(p)*fattn_query_tiles_per_piece;
                    const int64_t lo = std::max(begin, tile*kv_tiles);
                    const int64_t hi = std::min(end, (tile + fattn_query_tiles_per_piece)*kv_tiles);
                    if (lo >= hi) { continue; }
                    assert(lo == cursor);
                    assert(lo == begin || lo % kv_tiles == 0);
                    assert(hi == end || hi % kv_tiles == 0);
                    cursor = hi;
                }
                assert(cursor == end);
                ++ranges;
            }
        }
    }
    std::printf("PASS: %zu logical ranges partitioned exactly, KV lengths 32768..1050624\n", ranges);
}
