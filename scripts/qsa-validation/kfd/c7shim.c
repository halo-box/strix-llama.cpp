// C7 lab: LD_PRELOAD logger of HIP allocations, graph capture/instantiate, long syncs and llama_process boundaries.
// Output: C7SHIM_OUT (default /tmp/c7shim.<pid>.log), one line per event: kind t_ns dur_ns arg
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

static FILE * g_out;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static long long g_sync_min_ns = 1000000; // log syncs longer than 1 ms

static inline long long now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
static void out_open(void) {
    if (g_out) return;
    const char * p = getenv("C7SHIM_OUT");
    char buf[256];
    if (!p) { snprintf(buf, sizeof buf, "/tmp/c7shim.%d.log", (int) getpid()); p = buf; }
    g_out = fopen(p, "w");
    if (!g_out) g_out = stderr;
    setvbuf(g_out, NULL, _IOFBF, 1 << 20);
    const char * s = getenv("C7SHIM_SYNC_MIN_US");
    if (s) g_sync_min_ns = atoll(s) * 1000LL;
}
static void c7log(const char * kind, long long t0, long long dur, unsigned long long a, unsigned long long b) {
    pthread_mutex_lock(&g_mu);
    out_open();
    fprintf(g_out, "%s %lld %lld %llu %llx\n", kind, t0, dur, a, b);
    pthread_mutex_unlock(&g_mu);
}
static void flush_out(void) { pthread_mutex_lock(&g_mu); if (g_out) fflush(g_out); pthread_mutex_unlock(&g_mu); }
__attribute__((destructor)) static void fini(void) { flush_out(); }

#define REAL(name, ret, ...) static ret (*real_##name)(__VA_ARGS__); if (!real_##name) real_##name = (ret (*)(__VA_ARGS__)) dlsym(RTLD_NEXT, #name)

int hipMalloc(void ** p, size_t size) {
    REAL(hipMalloc, int, void **, size_t);
    long long t0 = now_ns(); int r = real_hipMalloc(p, size); long long d = now_ns() - t0;
    c7log("MALLOC", t0, d, (unsigned long long) size, (unsigned long long) (uintptr_t) (p ? *p : 0));
    return r;
}
int hipMallocManaged(void ** p, size_t size, unsigned int flags) {
    REAL(hipMallocManaged, int, void **, size_t, unsigned int);
    long long t0 = now_ns(); int r = real_hipMallocManaged(p, size, flags); long long d = now_ns() - t0;
    c7log("MALLOCM", t0, d, (unsigned long long) size, (unsigned long long) (uintptr_t) (p ? *p : 0));
    return r;
}
int hipExtMallocWithFlags(void ** p, size_t size, unsigned int flags) {
    REAL(hipExtMallocWithFlags, int, void **, size_t, unsigned int);
    long long t0 = now_ns(); int r = real_hipExtMallocWithFlags(p, size, flags); long long d = now_ns() - t0;
    c7log("MALLOCX", t0, d, (unsigned long long) size, (unsigned long long) (uintptr_t) (p ? *p : 0));
    return r;
}
int hipHostMalloc(void ** p, size_t size, unsigned int flags) {
    REAL(hipHostMalloc, int, void **, size_t, unsigned int);
    long long t0 = now_ns(); int r = real_hipHostMalloc(p, size, flags); long long d = now_ns() - t0;
    c7log("HOSTMALLOC", t0, d, (unsigned long long) size, (unsigned long long) (uintptr_t) (p ? *p : 0));
    return r;
}
int hipFree(void * p) {
    REAL(hipFree, int, void *);
    long long t0 = now_ns(); int r = real_hipFree(p); long long d = now_ns() - t0;
    c7log("FREE", t0, d, 0, (unsigned long long) (uintptr_t) p);
    return r;
}
int hipHostRegister(void * p, size_t size, unsigned int flags) {
    REAL(hipHostRegister, int, void *, size_t, unsigned int);
    long long t0 = now_ns(); int r = real_hipHostRegister(p, size, flags); long long d = now_ns() - t0;
    c7log("HOSTREG", t0, d, (unsigned long long) size, (unsigned long long) (uintptr_t) p);
    return r;
}
int hipStreamBeginCapture(void * s, int mode) {
    REAL(hipStreamBeginCapture, int, void *, int);
    long long t0 = now_ns(); int r = real_hipStreamBeginCapture(s, mode); long long d = now_ns() - t0;
    c7log("CAPBEGIN", t0, d, 0, (unsigned long long) (uintptr_t) s);
    return r;
}
int hipStreamEndCapture(void * s, void ** g) {
    REAL(hipStreamEndCapture, int, void *, void **);
    long long t0 = now_ns(); int r = real_hipStreamEndCapture(s, g); long long d = now_ns() - t0;
    c7log("CAPEND", t0, d, 0, (unsigned long long) (uintptr_t) s);
    return r;
}
int hipGraphInstantiate(void ** e, void * g, void * en, char * lb, size_t bs) {
    REAL(hipGraphInstantiate, int, void **, void *, void *, char *, size_t);
    long long t0 = now_ns(); int r = real_hipGraphInstantiate(e, g, en, lb, bs); long long d = now_ns() - t0;
    c7log("GINST", t0, d, 0, (unsigned long long) (uintptr_t) g);
    return r;
}
int hipGraphExecUpdate(void * e, void * g, void * en, void * res) {
    REAL(hipGraphExecUpdate, int, void *, void *, void *, void *);
    long long t0 = now_ns(); int r = real_hipGraphExecUpdate(e, g, en, res); long long d = now_ns() - t0;
    c7log("GUPDATE", t0, d, (unsigned long long) r, (unsigned long long) (uintptr_t) e);
    return r;
}
int hipGraphLaunch(void * e, void * s) {
    REAL(hipGraphLaunch, int, void *, void *);
    long long t0 = now_ns(); int r = real_hipGraphLaunch(e, s); long long d = now_ns() - t0;
    c7log("GLAUNCH", t0, d, 0, (unsigned long long) (uintptr_t) e);
    return r;
}
int hipStreamSynchronize(void * s) {
    REAL(hipStreamSynchronize, int, void *);
    long long t0 = now_ns(); int r = real_hipStreamSynchronize(s); long long d = now_ns() - t0;
    if (d >= g_sync_min_ns) c7log("SSYNC", t0, d, 0, (unsigned long long) (uintptr_t) s);
    return r;
}
int hipDeviceSynchronize(void) {
    REAL(hipDeviceSynchronize, int, void);
    long long t0 = now_ns(); int r = real_hipDeviceSynchronize(); long long d = now_ns() - t0;
    if (d >= g_sync_min_ns) c7log("DSYNC", t0, d, 0, 0);
    return r;
}
int hipEventSynchronize(void * e) {
    REAL(hipEventSynchronize, int, void *);
    long long t0 = now_ns(); int r = real_hipEventSynchronize(e); long long d = now_ns() - t0;
    if (d >= g_sync_min_ns) c7log("ESYNC", t0, d, 0, (unsigned long long) (uintptr_t) e);
    return r;
}
// llama boundaries (llama-bench calls these through the PLT of its impl library)
int llama_process(void * ctx, int type, void * batch) {
    REAL(llama_process, int, void *, int, void *);
    long long t0 = now_ns(); int r = real_llama_process(ctx, type, batch); long long d = now_ns() - t0;
    c7log("LPROC", t0, d, (unsigned long long) type, 0);
    return r;
}
void llama_synchronize(void * ctx) {
    REAL(llama_synchronize, void, void *);
    long long t0 = now_ns(); real_llama_synchronize(ctx); long long d = now_ns() - t0;
    c7log("LSYNC", t0, d, 0, 0);
    flush_out();
}
void llama_memory_clear(void * mem, int data) {
    REAL(llama_memory_clear, void, void *, int);
    long long t0 = now_ns(); real_llama_memory_clear(mem, data); long long d = now_ns() - t0;
    c7log("LCLEAR", t0, d, 0, 0);
}
