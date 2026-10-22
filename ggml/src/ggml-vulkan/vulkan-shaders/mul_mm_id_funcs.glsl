#ifdef MUL_MAT_ID
shared u16vec2 row_ids[BN];
uint _ne1;

void load_row_ids(uint row_begin) {
    // fusion_flags bit 4 (GGML_VK_MMID_ROWID_FILL): fill the slots past _ne1 with token 0, so the aligned B loads, which read
    // row_ids for the whole tile, gather a valid and cache-hot row instead of whatever the array held
    const uint n_slots = (p.fusion_flags & 16u) != 0 ? BN : _ne1;
    for (uint i = gl_LocalInvocationIndex; i < n_slots; i += BLOCK_SIZE) {
        const uint packed_row_id = i < _ne1 ? data_row_map[p.row_ids_offset + row_begin + i] : 0u;
        row_ids[i] = u16vec2(packed_row_id & 0xffffu, packed_row_id >> 16);
    }
    barrier();
}
#endif // MUL_MAT_ID
