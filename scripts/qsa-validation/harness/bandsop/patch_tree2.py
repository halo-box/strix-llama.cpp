# R2-M-LAB-BANDS-OP: BANDS_PARTIAL=<dir> writes the decode partial buffer (before the merge) of every qsa_decode launch. Idempotent.
import os
T = os.path.expanduser('~/dev/wt_lab/bands/tree')
p = os.path.join(T, 'ggml/src/ggml-cuda/qsa-decode.cu'); s = open(p).read()
anchor = '    qsa_decode_merge<<<nq*nh, 256, 0, ctx.stream()>>>(partial.get(), (float *) dst->data, splits);\n'
add = '    if (const char * pd = getenv("BANDS_PARTIAL")) { static int seq = 0; std::vector<float> hp(size_t(nq)*nh*splits*258); CUDA_CHECK(cudaMemcpyAsync(hp.data(), partial.get(), hp.size()*4, cudaMemcpyDeviceToHost, ctx.stream())); CUDA_CHECK(cudaStreamSynchronize(ctx.stream())); char fn[512]; snprintf(fn, sizeof fn, "%s/part_%03d.f32", pd, seq++); FILE * f = fopen(fn, "wb"); fwrite(hp.data(), 4, hp.size(), f); fclose(f); fprintf(stderr, "BANDS_PARTIAL %s nq %d splits %d\\n", fn, nq, splits); }\n'
if add.strip() in s: print('already')
else:
    assert s.count(anchor) == 1; s = s.replace(anchor, add + anchor); open(p, 'w').write(s); print('patched')
if '#include <vector>' not in s:
    s = open(p).read(); s = '#include <vector>\n' + s; open(p, 'w').write(s)
