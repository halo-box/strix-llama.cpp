// Sinkhorn normalization of a token's 4x4 hc comb matrix, held in one 16-lane slice of a subgroup: lane idx = idst + hc*isrc
// (idst in bits 0..1, isrc in bits 2..3), so subgroupShuffleXor by 1|2 reduces a row and by 4|8 a column. v is this lane's
// logit (mixes * scale + base). Every lane of the subgroup must call it. Shared by DSV4_HC_COMB and the fused hc boundary
// (dsv4_hc_boundary.comp), so both normalize with the same arithmetic.
float dsv4_hc_sinkhorn(float v, const float eps, const uint n_iter) {
    // Softmax across destinations: the four lanes sharing an isrc.
    float vmax = max(v, subgroupShuffleXor(v, 1));
    vmax = max(vmax, subgroupShuffleXor(vmax, 2));
    v = exp(v - vmax);

    float sum = v + subgroupShuffleXor(v, 1);
    sum += subgroupShuffleXor(sum, 2);
    v = v / sum + eps;

    // Normalize columns: equal destination indices are four lanes apart.
    sum = v + subgroupShuffleXor(v, 4);
    sum += subgroupShuffleXor(sum, 8);
    v /= sum + eps;

    for (uint i = 1; i < n_iter; ++i) {
        sum = v + subgroupShuffleXor(v, 1);
        sum += subgroupShuffleXor(sum, 2);
        v /= sum + eps;

        sum = v + subgroupShuffleXor(v, 4);
        sum += subgroupShuffleXor(sum, 8);
        v /= sum + eps;
    }
    return v;
}
