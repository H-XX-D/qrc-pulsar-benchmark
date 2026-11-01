#include "orbit_mpc_core.h"
static_assert(QRC_MPC_BITS >= 16 && QRC_MPC_BITS <= 32, "width");
static_assert(QRC_MPC_FRAC > 0 && QRC_MPC_FRAC < QRC_MPC_BITS-1, "fraction");
static_assert(QRC_MPC_ITERATIONS > 0, "iteration budget");
namespace {
const int64_t ONE = int64_t(1) << QRC_MPC_FRAC;
const int64_t LO = -(int64_t(1) << (QRC_MPC_BITS-1));
const int64_t HI = (int64_t(1) << (QRC_MPC_BITS-1))-1;
int64_t magnitude(int64_t x) { return x < 0 ? -x : x; }
int32_t bounded(int64_t x, bool &fault) {
    if (x < LO) { fault=true; return int32_t(LO); }
    if (x > HI) { fault=true; return int32_t(HI); }
    return int32_t(x);
}
int32_t rounded(int64_t x, bool &fault) {
    int64_t q = (magnitude(x)+(ONE>>1)) >> QRC_MPC_FRAC;
    return bounded(x < 0 ? -q : q, fault);
}
int32_t clip(int64_t x) { return int32_t(x < -ONE ? -ONE : x > ONE ? ONE : x); }
}
extern "C" int orbit_mpc_metadata(int field) {
    const int values[5]={1,QRC_MPC_BITS,QRC_MPC_FRAC,QRC_MPC_N,QRC_MPC_ITERATIONS};
    return field>=0 && field<5 ? values[field] : -1;
}
extern "C" int orbit_mpc_solve(const int32_t h[QRC_MPC_N*QRC_MPC_N],
    const int32_t g[QRC_MPC_N], const int32_t warm[QRC_MPC_N],
    int n, int32_t beta, int32_t tolerance, int valid,
    int32_t solution[QRC_MPC_N], int32_t *residual) {
    int32_t x[QRC_MPC_N], y[QRC_MPC_N], next[QRC_MPC_N];
    bool fault=false, invalid=false;
    *residual=0;
    for (int i=0;i<QRC_MPC_N;++i) {
        solution[i]=0; x[i]=clip(warm[i]); y[i]=x[i];
    }
    if (n<1 || n>QRC_MPC_N || beta<0 || beta>=ONE || tolerance<=0 || tolerance>ONE) return 1;
    if (!valid) return 2;
    for (int i=0;i<QRC_MPC_N;++i) {
        int64_t row=0;
        if (g[i]<LO || g[i]>HI || warm[i]<-ONE || warm[i]>ONE) invalid=true;
        if (i>=n && (g[i]!=0 || warm[i]!=0)) invalid=true;
        for (int j=0;j<QRC_MPC_N;++j) {
            int64_t a=h[i*QRC_MPC_N+j]; row+=magnitude(a);
            if (magnitude(a)>ONE || a!=h[j*QRC_MPC_N+i]) invalid=true;
            if ((i>=n || j>=n) && a!=0) invalid=true;
        }
        if (row>ONE) invalid=true;
    }
    if (invalid) return 1;
    // Fixed work budget; the inner dot accumulator is safe in int64 because
    // row l1 <= ONE and |y| <= 3*ONE, including accelerated extrapolation.
    for (int iteration=0;iteration<QRC_MPC_ITERATIONS;++iteration) {
        for (int i=0;i<QRC_MPC_N;++i) {
            int64_t dot=0;
            for (int j=0;j<QRC_MPC_N;++j) dot+=int64_t(h[i*QRC_MPC_N+j])*y[j];
            int32_t gradient=bounded(int64_t(rounded(dot,fault))+g[i],fault);
            next[i]=i<n ? clip(int64_t(y[i])-gradient) : 0;
        }
        for (int i=0;i<QRC_MPC_N;++i) {
            y[i]=bounded(int64_t(next[i])+rounded(int64_t(beta)*(int64_t(next[i])-x[i]),fault),fault);
            x[i]=next[i];
        }
    }
    int64_t maximum=0;
    for (int i=0;i<QRC_MPC_N;++i) {
        int64_t dot=0;
        for (int j=0;j<QRC_MPC_N;++j) dot+=int64_t(h[i*QRC_MPC_N+j])*x[j];
        int32_t gradient=bounded(int64_t(rounded(dot,fault))+g[i],fault);
        int64_t r=magnitude(int64_t(x[i])-clip(int64_t(x[i])-gradient));
        if (r>maximum) maximum=r;
    }
    *residual=bounded(maximum,fault);
    if (fault) return 4;
    if (maximum>tolerance) return 16;
    for (int i=0;i<QRC_MPC_N;++i) solution[i]=x[i];
    return 0;
}
