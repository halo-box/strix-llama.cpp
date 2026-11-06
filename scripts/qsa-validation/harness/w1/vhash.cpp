// prints bands_cpu.cpp's CACHE_HASH ... V f16 line for every layer of a rows + qkv fixture
#include "bands_ref.h"
using namespace bands;
int main(int argc, char ** argv) {
    fixture_reader R; if (!R.open(argv[1], argv[2])) return 1;
    fixture F;
    while (R.next(F, [](int) { return true; })) {
        std::vector<uint8_t> VC; build_cache(BT_F16, F.V16, F.nk, VC);
        printf("CACHE_HASH vhash L%d V f16 %016llx\n", F.L, (unsigned long long) fnv1a(VC.data(), VC.size()));
    }
    return 0;
}
