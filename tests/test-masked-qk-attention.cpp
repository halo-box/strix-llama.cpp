#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static float next_value(uint32_t & s) {
    s^=s<<13; s^=s>>17; s^=s<<5;
    return (int32_t(s&2047)-1024)*(1.0f/1024.0f);
}
static uint64_t hash(const void * p,size_t n) {
    uint64_t h=14695981039346656037ull;
    for (size_t i=0;i<n;++i) { h=(h^((const unsigned char *)p)[i])*1099511628211ull; }
    return h;
}
static void save(const std::string & path,const void * data,size_t n) {
    FILE * f=fopen(path.c_str(),"wb");
    if (!f || fwrite(data,1,n,f)!=n || fclose(f)!=0) { fprintf(stderr,"write failed %s\n",path.c_str()); exit(5); }
}
static void load(const std::string & path,void * data,size_t n) {
    FILE * f=fopen(path.c_str(),"rb");
    if (!f || fread(data,1,n,f)!=n || fgetc(f)!=EOF || fclose(f)!=0) { fprintf(stderr,"read failed %s\n",path.c_str()); exit(5); }
}

static int run_case(int argc,char ** argv,ggml_backend_t backend) {
    if (argc!=5) { fprintf(stderr,"usage: driver mode output-prefix mask-file|- reference-prefix|-\n"); return 2; }
    const std::string mode=argv[1],prefix=argv[2],maskfile=argv[3],reference=argv[4];
    const bool synthetic_replay=mode=="replay-synthetic",real_replay=mode=="replay-real";
    const bool replay=synthetic_replay||real_replay,perf=getenv("ATTN_TIMING")!=nullptr;
    if (getenv("GGML_CUDA_DISABLE_GRAPHS") || getenv("GGML_BACKEND_PATH")) { return 2; }
    const int64_t d=256,nq=getenv("ATTN_T")?atoll(getenv("ATTN_T")):2048;
    const int64_t nk=getenv("ATTN_N")?atoll(getenv("ATTN_N")):34816;
    const int64_t nm=nq>=1024?(nq+7)/8*8:nq;
    const int64_t past0=getenv("ATTN_DEPTH")?atoll(getenv("ATTN_DEPTH")):std::max(int64_t(0),nk-nq);
    const int64_t hq=mode=="heads" ? 16 : 24,hk=2;
    const bool no_mask=mode=="no-mask",padded=mode=="strided";
    const char * backenddir=getenv("ATTN_BACKEND_DIR");
    if (!backenddir) { return 2; }
    auto ctx=ggml_init({ggml_tensor_overhead()*32+ggml_graph_overhead(),nullptr,true});
    if (!backend || !ctx) { return 3; }
    const bool qwide=mode=="q-strided",native=mode=="native";
    auto qb=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,d,native?nq:hq*(qwide?2:1),native?hq:nq,1);
    auto q=qwide?ggml_view_4d(ctx,qb,d,hq,nq,1,qb->nb[1],qb->nb[2],qb->nb[3],0):qb;
    if (!native) { q=ggml_permute(ctx,q,0,2,1,3); }
    auto kb=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,d,hk*(padded?2:1),nk,1);
    auto vb=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,d,hk*(padded?2:1),nk,1);
    auto k=padded ? ggml_view_4d(ctx,kb,d,hk,nk,1,kb->nb[1],kb->nb[2],kb->nb[3],0) : kb;
    auto v=padded ? ggml_view_4d(ctx,vb,d,hk,nk,1,vb->nb[1],vb->nb[2],vb->nb[3],0) : vb;
    k=ggml_permute(ctx,k,0,2,1,3); v=ggml_permute(ctx,v,0,2,1,3);
    auto m=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,nk,nm,1,1);
    auto out=ggml_flash_attn_ext(ctx,q,k,v,no_mask?nullptr:m,mode=="qoverflow"?2.0f:0.0625f,0,0);
    ggml_flash_attn_ext_set_prec(out,GGML_PREC_F32);
    ggml_set_name(out,"continuation-attention");
    auto graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,out);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if (!buffer || !ggml_backend_supports_op(backend,out)) { return 3; }
    uint32_t seed=0x13579bdf;
    std::vector<float> original_q(ggml_nelements(qb));
    std::vector<ggml_fp16_t> original_k(ggml_nelements(kb)),original_v(ggml_nelements(vb));
    for (auto & x:original_q) { x=next_value(seed); }
    for (auto & x:original_k) { x=ggml_fp32_to_fp16(next_value(seed)); }
    for (auto & x:original_v) { x=ggml_fp32_to_fp16(next_value(seed)); }
    const int steps=synthetic_replay?5:real_replay?3:1;
    const int only=getenv("ATTN_STEP")?atoi(getenv("ATTN_STEP")):-1;
    printf("CONFIG mode=%s nk=%lld hq=%lld padded=%d graph_default=1 q=%p k=%p v=%p mask=%p graph=%p backend=%s\n",
        mode.c_str(),(long long)nk,(long long)hq,padded,q->data,k->data,v->data,m->data,(void *)graph,backenddir);
    for (int step=0;step<steps;++step) {
        if (only>=0 && step!=only) { continue; }
        auto qdata=original_q;
        auto kdata=original_k,vdata=original_v;
        std::vector<ggml_fp16_t> mask(size_t(nk)*nm,mode=="pad-finite"?0:0xfc00);
        for (int j=0;j<nq;++j) {
            for (int i=0;i<nk;++i) {
                const int past=past0+j;
                const bool visible=i<=past && (mode=="causal" || i<3 || i>past-2048);
                mask[size_t(j)*nk+i]=visible?0:0xfc00;
            }
        }
        if (mode=="qoverflow") { qdata[23*256]=65504.0f; }
        if (mode=="qnan") { qdata[23*256]=NAN; }
        if (mode=="kinf") { kdata[32*(padded?1024:512)+256]=0x7c00; }
        if (mode=="knan") { kdata[32*(padded?1024:512)+256]=0x7e01; }
        if (mode=="vinf") { vdata[32*512]=0x7c00; }
        if (mode=="vnan") { vdata[32*512]=0x7e01; }
        if (mode=="minf" || mode=="mnan") { mask[32]=mode=="minf"?0x7c00:0x7e01; }
        if (maskfile!="-") {
            std::string path=maskfile;
            if (real_replay && step==1) {
                const auto pos=path.rfind("mask-1.f16");
                if (pos==std::string::npos) { return 2; }
                path.replace(pos,10,"mask-12.f16");
            }
            load(path,mask.data(),mask.size()*2);
        }
        if (synthetic_replay && step==1) {
            qdata[23*256]=1.0e10f;
            kdata[32*512+256]=0x7c00;
            vdata[32*512]=0x7e01;
            for (int j=0;j<nq;++j) {
                for (int i=std::max(int64_t(3),nk-512);i<nk;++i) { mask[size_t(j)*nk+i]=0xfc00; }
            }
        }
        if (synthetic_replay && step==2) {
            for (size_t i=0;i<vdata.size();++i) { vdata[i]=(i%2?0x8000:0)|(i%1025); }
        }
        if (synthetic_replay && step==3) {
            for (size_t i=0;i<vdata.size();++i) { vdata[i]=i%2?0x8000:0; }
        }
        const auto label=prefix+"-step"+std::to_string(step);
        const uint64_t hashes[]={hash(qdata.data(),qdata.size()*4),hash(kdata.data(),kdata.size()*2),
            hash(vdata.data(),vdata.size()*2),hash(mask.data(),mask.size()*2)};
        char inputs[256];
        snprintf(inputs,sizeof(inputs),"seed=13579bdf Q=%016llx K=%016llx V=%016llx mask=%016llx\n",
            (unsigned long long)hashes[0],(unsigned long long)hashes[1],(unsigned long long)hashes[2],(unsigned long long)hashes[3]);
        save(label+".inputs",inputs,strlen(inputs));
        ggml_backend_tensor_set(qb,qdata.data(),0,qdata.size()*4);
        ggml_backend_tensor_set(kb,kdata.data(),0,kdata.size()*2);
        ggml_backend_tensor_set(vb,vdata.data(),0,vdata.size()*2);
        ggml_backend_tensor_set(m,mask.data(),0,mask.size()*2);
        auto compute=[&]() {
            if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) { exit(4); }
            ggml_backend_synchronize(backend);
        };
        std::vector<float> expected(ggml_nelements(out)),output(expected.size());
        const bool check=reference!="-";
        if (check) {
            load(reference+"-step"+std::to_string(step)+".f32",expected.data(),expected.size()*4);
            std::vector<char> input_reference(strlen(inputs));
            load(reference+"-step"+std::to_string(step)+".inputs",input_reference.data(),input_reference.size());
            if (memcmp(inputs,input_reference.data(),input_reference.size())) { return 8; }
        }
        // In replay mode, verify each execution, including capture and subsequent replay.
        for (int repeat=0;repeat<(replay?3:1);++repeat) {
            compute();
            ggml_backend_tensor_get(out,output.data(),0,output.size()*4);
            if (check && memcmp(output.data(),expected.data(),output.size()*4)) {
                save(label+".FIRST_FAILURE.f32",output.data(),output.size()*4);
                fprintf(stderr,"BIT_FAIL step=%d repeat=%d\n",step,repeat); return 7;
            }
        }
        save(label+".f32",output.data(),output.size()*4);
        size_t nonfinite=0; for (float x:output) { nonfinite+=!std::isfinite(x); }
        const bool exceptional=mode=="qoverflow" || mode=="qnan" || mode=="kinf" || mode=="knan" ||
            mode=="vinf" || mode=="vnan" || mode=="minf" || mode=="mnan" || (synthetic_replay && step==1);
        if (nonfinite && !exceptional) { return 6; }
        printf("RAW_PASS step=%d elements=%zu nonfinite=%zu checked=%d %s",step,output.size(),nonfinite,check,inputs);
        if (perf) {
            if (!check || replay) { return 2; }
            compute(); // One untimed warmup after the correctness invocation.
            int64_t samples[6];
            for (int i=0;i<6;++i) {
                ggml_backend_synchronize(backend);
                const auto start=std::chrono::steady_clock::now(); compute();
                samples[i]=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
            }
            ggml_backend_tensor_get(out,output.data(),0,output.size()*4);
            save(label+".after.f32",output.data(),output.size()*4);
            if (memcmp(output.data(),expected.data(),output.size()*4)) { fprintf(stderr,"BIT_FAIL after timing\n"); return 7; }
            printf("TIMING_BIT_PASS step=%d warmup=1 samples=6 scope=whole_graph_sync\n",step);
            for (int i=0;i<6;++i) { printf("sample_ns[%d]=%lld\n",i,(long long)samples[i]); }
        }
        ggml_backend_tensor_get(qb,qdata.data(),0,qdata.size()*4);
        ggml_backend_tensor_get(kb,kdata.data(),0,kdata.size()*2);
        ggml_backend_tensor_get(vb,vdata.data(),0,vdata.size()*2);
        ggml_backend_tensor_get(m,mask.data(),0,mask.size()*2);
        const uint64_t after[]={hash(qdata.data(),qdata.size()*4),hash(kdata.data(),kdata.size()*2),
            hash(vdata.data(),vdata.size()*2),hash(mask.data(),mask.size()*2)};
        if (memcmp(hashes,after,sizeof(hashes))) { return 8; }
        printf("INPUT_AFTER_PASS step=%d\n",step);
    }
    std::ifstream maps("/proc/self/maps"); std::string line;
    while (std::getline(maps,line)) { if (line.find("libggml")!=std::string::npos) { printf("MAP %s\n",line.c_str()); } }
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    return 0;
}

int main(int argc,char ** argv) {
    if (argc!=5 || !getenv("ATTN_BACKEND_DIR")) { return 2; }
    ggml_backend_load_all_from_path(getenv("ATTN_BACKEND_DIR"));
    auto dev=ggml_backend_dev_by_name("ROCm0");
    if (!dev || ggml_backend_dev_type(dev)!=GGML_BACKEND_DEVICE_TYPE_IGPU) { return 3; }
    auto backend=ggml_backend_dev_init(dev,nullptr);
    if (!backend) { return 3; }
    int rc=0;
    if (strcmp(argv[1],"resize-replay")) {
        rc=run_case(argc,argv,backend);
    } else {
        // Reuse the GPU backend/graph cache across changing Q and KV extents.
        const int shapes[][2]={{63,256},{1025,1280},{2048,14080},{9,256},{2048,34816}};
        for (int i=0;i<5 && !rc;++i) {
            const std::string t=std::to_string(shapes[i][0]),n=std::to_string(shapes[i][1]);
            setenv("ATTN_T",t.c_str(),1); setenv("ATTN_N",n.c_str(),1); setenv("ATTN_DEPTH","0",1);
            std::string mode="replay-synthetic",prefix=std::string(argv[2])+"-shape"+std::to_string(i);
            std::string reference=strcmp(argv[4],"-")?std::string(argv[4])+"-shape"+std::to_string(i):"-";
            char * args[]={argv[0],mode.data(),prefix.data(),argv[3],reference.data()};
            rc=run_case(5,args,backend);
        }
    }
    ggml_backend_free(backend);
    return rc;
}
