#include "ggml-cuda/fattn-common.cuh"
#include <algorithm>
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) { return 2; }
    const int t=atoi(argv[1]), n=atoi(argv[2]), wide=atoi(argv[3]);
    const int a=(t+7)/8, b=n/32, m=t>=1024?8*a:t;
    const size_t count=masked_qk_flag_count(t,n);
    if (!count) { return 2; }
    const size_t qs=24*256*(wide?2:1), ks=2*256*(wide?2:1);
    // No padded Q rows allocated; last-head exceptions include adjacent half fields.
    std::vector<float> q(size_t(t)*qs,0.5f);
    std::vector<ggml_fp16_t> k(size_t(n)*ks,0x3800), mask(size_t(m)*n,0xfc00);
    for (int j=0;j<t;++j) {
        for (int i=0;i<n;++i) {
            if (i<(n>=512?n-256:n) && (i<3 || (i<=n-t+j && i>n-t+j-64))) { mask[size_t(j)*n+i]=0; }
        }
    }
    float * dq; ggml_fp16_t * dk,* dm; int * df;
    CUDA_CHECK(cudaMalloc(&dq,q.size()*4)); CUDA_CHECK(cudaMalloc(&dk,k.size()*2));
    CUDA_CHECK(cudaMalloc(&dm,mask.size()*2)); CUDA_CHECK(cudaMalloc(&df,count*4));
    int previous_last_bound=-1;
    for (int step=0;step<5;++step) {
        const float scale=step==1?2.0f:0.0625f;
        q[(t-1)*qs+23*256+255]=step==1?65504.0f:step==2?NAN:0.5f;
        k[(n-1)*ks+256+255]=step==2?0x7c00:step==3?0x7e01:0x3800;
        mask[32]=step==1?0x7c00:step==2?0x7e01:0xfc00;
        if (n>=512) {
            for (int j=0;j<t;++j) {
                std::fill(mask.begin()+size_t(j)*n+n-256,mask.begin()+size_t(j+1)*n,step==1?0x7c00:0xfc00);
            }
        }
        if (m>t) { std::fill(mask.begin()+size_t(t)*n,mask.end(),step==4?0:0xfc00); }
        CUDA_CHECK(cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dk,k.data(),k.size()*2,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dm,mask.data(),mask.size()*2,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(df,0xa5,count*4));
        if (t>=1024) {
            flash_attn_mask_to_KV_max<8><<<dim3(a),dim3(128)>>>((const half2 *)dm,df,n/256,n/2,size_t(m)*n/2);
        }
        masked_qk_q_finite<<<dim3(a),dim3(256)>>>((const char *)dq,df+a,qs*4,256*4,scale,t,t<1024?df:nullptr,n);
        masked_qk_k_finite<<<dim3(2*b),dim3(256)>>>((const char *)dk,df+2*a,ks*2,256*2,b);
        masked_qk_mask_empty<<<dim3(a*b/8),dim3(32,8)>>>((const unsigned short *)dm,df+2*a+2*b,t,n,b);
        CUDA_CHECK(cudaGetLastError());
        std::vector<int> actual(count), expected(count,1);
        CUDA_CHECK(cudaMemcpy(actual.data(),df,count*4,cudaMemcpyDeviceToHost));
        for (int jt=0;jt<a;++jt) {
            int last=0;
            for (int j=0;j<8;++j) {
                const int row=jt*8+j;
                if (t>=1024) {
                    for (int i=0;i<n;++i) {
                        if ((mask[size_t(row)*n+i]&0x7fff)!=0x7c00) { last=std::max(last,i+1); }
                    }
                }
                if (row<t) {
                    for (int h=0;h<24;++h) {
                        for (int d=0;d<256;++d) {
                            const float qh=ggml_fp16_to_fp32(ggml_fp32_to_fp16(q[row*qs+h*256+d]));
                            const float sh=ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale));
                            expected[a+jt] &= (ggml_fp32_to_fp16(sh*qh)&0x7c00)!=0x7c00;
                        }
                    }
                }
            }
            expected[jt]=t<1024?n:(last+255)/256*256;
            for (int kb=0;kb<b;++kb) {
                for (int j=0;j<8;++j) {
                    for (int i=0;i<32;++i) {
                        expected[2*a+2*b+jt*b+kb] &= mask[size_t((jt*8+j)%t)*n+kb*32+i]==0xfc00;
                    }
                }
            }
        }
        for (int h=0;h<2;++h) {
            for (int kb=0;kb<b;++kb) {
                for (int i=0;i<32*256;++i) {
                    expected[2*a+h*b+kb] &= (k[(kb*32+i/256)*ks+h*256+i%256]&0x7c00)!=0x7c00;
                }
            }
        }
        for (size_t i=0;i<count;++i) {
            if (actual[i]!=expected[i]) { fprintf(stderr,"FLAG_FAIL t=%d n=%d step=%d index=%zu actual=%d expected=%d\n",t,n,step,i,actual[i],expected[i]); return 4; }
        }
        if (m>t && n>=512 && step==4) {
            if (previous_last_bound>n-256 || actual[a-1]!=n) { return 6; }
            printf("PHYSICAL_PADDING_PASS T=%d N=%d before=%d after=%d\n",t,n,previous_last_bound,actual[a-1]);
        }
        previous_last_bound=actual[a-1];
        printf("FLAG_PASS T=%d N=%d wide=%d step=%d all_ints=%zu qgrid=%d kgrid=%d emptygrid=%d\n",t,n,wide,step,count,a,2*b,a*b/8);
    }
    CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dk)); CUDA_CHECK(cudaFree(dm)); CUDA_CHECK(cudaFree(df));
}
