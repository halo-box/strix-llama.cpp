// C5 offline union statistics over captured selected-key rows (format of c5_capture / c3_capture).
// usage: c5_union file.rows [tag]
// Per layer (record) and averaged over the 12 layers:
//   q_cells  valid cells per query (0 <= c < nk), dup = duplicate valid cells per query, asc = rows already non-decreasing
//   q_quads  distinct physical quads (cell>>2) per query; qpb = q_quads / (q_cells/4) ("quads per logical block": 1.00 =
//            every selected block is one aligned quad); part = fraction of a query's quads with < 4 selected cells (tail incl.)
//   ovl      adjacent-query overlap |Q_i & Q_i+1| / |Q_i| (quads)
//   for G in {1,2,4,8}, groups of G consecutive queries (query = G*g + j, as QSA3_G grouping in qsa3_attn_kernel):
//     U/q    mean |union of quads| / mean q_quads
//     ch/q   ASC chunks of 4 quads per query = ceil(|U|/4)/G  (the chunk count today's kernel walks; K/V gather per chunk)
//     t12    WMMA 16-row tiles per query, today's packing (12 heads per query, rows packed, ceil(12G/16) tiles per chunk)
//     t12s   same, ASC order but skipping, per chunk, the tiles whose queries select none of the chunk's 4 quads
//     bch/q  BUCKET chunks per query: quads stably sorted by G-bit query-presence pattern, each bucket padded to 4
//     b12    tiles per query, BUCKET order, 12-row packing, tiles touched by the bucket's pattern
//     b16    tiles per query, BUCKET order, one 16-row tile per present query (plan-B.md:10 "tS/q")
//   ideal16 = q_quads/4 (one tile per query per 4 of its own quads), ideal12 = 0.75 * q_quads/4
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <string>

struct hdr { char name[64]; int32_t n_q, ns, nk, has_mask, pos0, kt; };

static int tiles12_mask(int G, uint32_t qpat) {
    // rows r = 12*j + h, tile = r/16; tile t active if any present query j has a row in [16t, 16t+16)
    const int T = (12 * G + 15) / 16;
    int n = 0;
    for (int t = 0; t < T; ++t) {
        bool a = false;
        for (int j = 0; j < G && !a; ++j) {
            if (!((qpat >> j) & 1)) continue;
            const int r0 = 12 * j, r1 = 12 * j + 11;
            if (r1 >= 16 * t && r0 <= 16 * t + 15) a = true;
        }
        n += a;
    }
    return n;
}

int main(int argc, char ** argv) {
    FILE * f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 1; }
    const char * tag = argc > 2 ? argv[2] : argv[1];
    const int Gs[4] = {1, 2, 4, 8};
    double acc[64] = {0}; int nl = 0;
    double hb[2][5] = {{0}}; // G4: [asc-skip, bucket] chunks by active 12-row tile count 0..3, [4] = chunks with all 4 queries present
    printf("# file %s\n", argv[1]);
    printf("%-6s %-2s %5s %6s %7s %6s %4s %5s %5s %5s", "tag", "L", "nk", "n_q", "q_cells", "q_quad", "qpb", "part", "asc", "ovl");
    for (int G : Gs) printf(" | G%d: %5s %6s %6s %6s %6s %6s %6s", G, "U/q", "ch/q", "t12", "t12s", "bch/q", "b12", "b16");
    printf(" | %7s %7s\n", "ideal16", "ideal12");
    hdr h;
    while (fread(&h, sizeof(h), 1, f) == 1) {
        std::vector<int32_t> rows((size_t) h.n_q * h.ns);
        if (fread(rows.data(), 4, rows.size(), f) != rows.size()) { fprintf(stderr, "short\n"); return 1; }
        const int nq = h.n_q;
        // per query: sorted unique (quad, cellmask)
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> Q(nq);
        double s_cells = 0, s_dup = 0, s_quads = 0, s_part = 0; int n_asc = 0;
        for (int q = 0; q < nq; ++q) {
            const int32_t * r = rows.data() + (size_t) q * h.ns;
            std::vector<int32_t> v; v.reserve(h.ns);
            bool asc = true; int32_t prev = INT32_MIN;
            for (int j = 0; j < h.ns; ++j) {
                const int32_t c = r[j];
                if (c < 0 || c >= h.nk) continue;
                if (c < prev) asc = false;
                prev = c; v.push_back(c);
            }
            n_asc += asc;
            std::sort(v.begin(), v.end());
            const size_t nv = v.size();
            v.erase(std::unique(v.begin(), v.end()), v.end());
            s_dup += (double) (nv - v.size());
            s_cells += (double) v.size();
            auto & qq = Q[q];
            for (int32_t c : v) {
                const uint32_t qd = (uint32_t) c >> 2, b = 1u << (c & 3);
                if (!qq.empty() && qq.back().first == qd) qq.back().second |= b; else qq.push_back({qd, b});
            }
            s_quads += (double) qq.size();
            int part = 0; for (auto & e : qq) part += e.second != 0xF;
            s_part += qq.empty() ? 0 : (double) part / qq.size();
        }
        double s_ovl = 0;
        for (int q = 0; q + 1 < nq; ++q) {
            const auto & a = Q[q]; const auto & b = Q[q + 1];
            size_t i = 0, j = 0, c = 0;
            while (i < a.size() && j < b.size()) { if (a[i].first < b[j].first) ++i; else if (a[i].first > b[j].first) ++j; else { ++c; ++i; ++j; } }
            s_ovl += a.empty() ? 0 : (double) c / a.size();
        }
        const double mq = s_quads / nq;
        double vals[64]; int k = 0;
        vals[k++] = h.nk; vals[k++] = nq; vals[k++] = s_cells / nq; vals[k++] = mq; vals[k++] = (s_quads / nq) / ((s_cells / nq) / 4.0);
        vals[k++] = s_part / nq; vals[k++] = (double) n_asc / nq; vals[k++] = s_ovl / (nq - 1);
        for (int G : Gs) {
            const int ng = (nq + G - 1) / G;
            const int T = (12 * G + 15) / 16;
            double sU = 0, sch = 0, st12s = 0, sbch = 0, sb12 = 0, sb16 = 0;
            std::vector<std::pair<uint32_t, uint32_t>> u; // (quad, G-bit presence pattern)
            for (int g = 0; g < ng; ++g) {
                u.clear();
                for (int j = 0; j < G && G * g + j < nq; ++j) for (auto & e : Q[G * g + j]) u.push_back({e.first, 1u << j});
                std::sort(u.begin(), u.end());
                size_t w = 0;
                for (size_t i = 0; i < u.size(); ++i) { if (w > 0 && u[w - 1].first == u[i].first) u[w - 1].second |= u[i].second; else u[w++] = u[i]; }
                u.resize(w);
                sU += w;
                const size_t ch = (w + 3) / 4;
                sch += ch;
                for (size_t c = 0; c < ch; ++c) {
                    uint32_t p = 0; for (size_t i = 4 * c; i < std::min(w, 4 * c + 4); ++i) p |= u[i].second;
                    st12s += tiles12_mask(G, p);
                    if (G == 4) { hb[0][tiles12_mask(G, p)] += 1; hb[0][4] += p == 0xF; }
                }
                // bucket: count per pattern
                std::vector<int> cnt(1u << G, 0);
                for (auto & e : u) cnt[e.second]++;
                for (uint32_t p = 1; p < (1u << G); ++p) {
                    if (!cnt[p]) continue;
                    const int bc = (cnt[p] + 3) / 4;
                    sbch += bc; sb12 += (double) bc * tiles12_mask(G, p);
                    if (G == 4) { hb[1][tiles12_mask(G, p)] += bc; hb[1][4] += p == 0xF ? bc : 0; } sb16 += (double) bc * __builtin_popcount(p);
                }
            }
            vals[k++] = (sU / ng) / mq;
            vals[k++] = sch / nq; vals[k++] = sch * T / nq; vals[k++] = st12s / nq;
            vals[k++] = sbch / nq; vals[k++] = sb12 / nq; vals[k++] = sb16 / nq;
        }
        vals[k++] = mq / 4.0; vals[k++] = 0.75 * mq / 4.0;
        printf("%-6s %-2d %5.0f %6.0f %7.1f %6.1f %4.3f %5.3f %5.3f %5.3f", tag, nl, vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]);
        for (int gi = 0; gi < 4; ++gi) { const double * x = vals + 8 + 7 * gi; printf(" | G%d: %5.2f %6.1f %6.1f %6.1f %6.1f %6.1f %6.1f", Gs[gi], x[0], x[1], x[2], x[3], x[4], x[5], x[6]); }
        printf(" | %7.1f %7.1f   dup=%.2f\n", vals[36], vals[37], s_dup / nq);
        for (int i = 0; i < k; ++i) acc[i] += vals[i];
        nl++;
    }
    printf("%-6s %-2s %5.0f %6.0f %7.1f %6.1f %4.3f %5.3f %5.3f %5.3f", tag, "MN", acc[0] / nl, acc[1] / nl, acc[2] / nl, acc[3] / nl, acc[4] / nl, acc[5] / nl, acc[6] / nl, acc[7] / nl);
    for (int gi = 0; gi < 4; ++gi) { const double * x = acc + 8 + 7 * gi; printf(" | G%d: %5.2f %6.1f %6.1f %6.1f %6.1f %6.1f %6.1f", Gs[gi], x[0] / nl, x[1] / nl, x[2] / nl, x[3] / nl, x[4] / nl, x[5] / nl, x[6] / nl); }
    printf(" | %7.1f %7.1f   layers=%d\n", acc[36] / nl, acc[37] / nl, nl);
    for (int o = 0; o < 2; ++o) { const double t = hb[o][1] + hb[o][2] + hb[o][3];
        printf("%-6s G4HIST %s chunks by active tiles (1,2,3): %.4f %.4f %.4f  all-4-queries %.4f\n", tag, o ? "BUCKET" : "ASC   ", hb[o][1] / t, hb[o][2] / t, hb[o][3] / t, hb[o][4] / t); }
    return 0;
}
