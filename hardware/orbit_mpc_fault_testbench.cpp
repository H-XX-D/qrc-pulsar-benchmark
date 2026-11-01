#include "orbit_mpc_core.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <climits>
int main() {
  const int32_t one=int32_t(1)<<QRC_MPC_FRAC;
  uint32_t rng=21071984; int counts[33]={0};
  for (int trial=0;trial<600;++trial) {
    int32_t h[QRC_MPC_N*QRC_MPC_N]={0},g[QRC_MPC_N]={0},w[QRC_MPC_N]={0},out[QRC_MPC_N]={0},res=0;
    int n=1+trial%QRC_MPC_N;
    for (int i=0;i<n;++i) {
      rng=1664525*rng+1013904223;
      h[i*QRC_MPC_N+i]=one/4;
      g[i]=(trial%3==0) ? ((i%2) ? INT32_MAX : INT32_MIN) : int32_t(rng%uint32_t(one))-one/2;
      w[i]=int32_t(rng%uint32_t(one))-one/2;
    }
    if (trial%5==0) h[0]=one*2;
    int status=orbit_mpc_solve(h,g,w,n,one/2,8,1,out,&res);
    assert(status>=0 && status<=32);counts[status]++;
    for (int i=0;i<QRC_MPC_N;++i) {
      if (status) assert(out[i]==0);
      else assert(out[i]>=-one && out[i]<=one);
    }
  }
  std::printf("bits=%d cases=600 success=%d invalid=%d range=%d budget=%d\n",QRC_MPC_BITS,counts[0],counts[1],counts[4],counts[16]);
}
