#pragma once
#include <cstddef>
#include <cstdint>
#include <climits>

// Zero means unsupported. Bound every int-indexed slice and Stream-K product
// before allocation; all intermediate arithmetic here is unsigned 64-bit.
constexpr size_t masked_qk_flag_count(int64_t t, int64_t n) {
    if (t <= 0 || n <= 0 || t > INT_MAX - 7 || n > INT_MAX - 255 || n % 256 != 0) {
        return 0;
    }
    const uint64_t a = (uint64_t(t) + 7)/8, b = uint64_t(n)/32;
    const uint64_t count = 2*a + 2*b + a*b;
    return 4*a*b <= INT_MAX && 400*a <= INT_MAX && count <= INT_MAX && count <= SIZE_MAX/sizeof(int)
        ? size_t(count) : 0;
}

static_assert(masked_qk_flag_count(2048, 34816) == 281216, "winning layout");
static_assert(masked_qk_flag_count(9, 256) == 36, "partial tile layout");
static_assert(masked_qk_flag_count(0, 256) == 0, "empty query");
static_assert(masked_qk_flag_count(2048, 257) == 0, "key alignment");
static_assert(masked_qk_flag_count(INT64_MAX, 256) == 0, "query overflow");
static_assert(masked_qk_flag_count(2048, INT64_MAX) == 0, "key overflow");
static_assert(masked_qk_flag_count(1048576, 1048576) == 0, "Stream-K overflow");

// Subdivide physical work only at output-tile boundaries. Logical Stream-K
// ranges (and hence their partial sums and fixup order) remain unchanged.
constexpr int fattn_query_tiles_per_piece = 4;
constexpr unsigned fattn_query_piece_count(int tiles, int kv_tiles, unsigned logical_blocks) {
    const int64_t work = int64_t(tiles)*kv_tiles;
    if (tiles <= 0 || kv_tiles <= 0 || logical_blocks == 0 || work > INT_MAX) {
        return 1;
    }
    int longest = 0;
    for (unsigned b = 0; b < logical_blocks; ++b) {
        const int64_t begin = int64_t(b)*work/logical_blocks;
        const int64_t end = (int64_t(b) + 1)*work/logical_blocks;
        if (begin < end) {
            const int span = int((end - 1)/kv_tiles - begin/kv_tiles + 1);
            longest = span > longest ? span : longest;
        }
    }
    return longest ? unsigned((longest + fattn_query_tiles_per_piece - 1)/fattn_query_tiles_per_piece) : 1;
}

static_assert(fattn_query_piece_count(1024, 66048/32, 20) == 13);
static_assert(fattn_query_piece_count(1024, 130048/32, 20) == 13);
static_assert(fattn_query_piece_count(1024, 1002240/32, 20) == 13);
static_assert(fattn_query_piece_count(1024, 1050624/32, 20) == 13);
static_assert(fattn_query_piece_count(1024, 130048/32, 40) == 7);
static_assert(fattn_query_piece_count(1024, 130048/32, 1024) == 1);
static_assert(fattn_query_piece_count(0, 4064, 20) == 1);
static_assert(fattn_query_piece_count(1024, 4064, 0) == 1);
static_assert(fattn_query_piece_count(INT_MAX, INT_MAX, 20) == 1);
