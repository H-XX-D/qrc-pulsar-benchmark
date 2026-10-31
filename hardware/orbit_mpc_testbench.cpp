#include "orbit_mpc_core.h"
#include <cassert>
#include <cstdlib>
int main() {
    const int32_t one=int32_t(1)<<QRC_MPC_FRAC;
    int32_t h[QRC_MPC_N*QRC_MPC_N]={0},g[QRC_MPC_N]={0},warm[QRC_MPC_N]={0},out[QRC_MPC_N]={0},res=0;
    for(int i=0;i<3;++i) {h[i*QRC_MPC_N+i]=one/2;g[i]=one/8;}
    assert(orbit_mpc_solve(h,g,warm,3,0,4,1,out,&res)==0);
    for(int i=0;i<3;++i) assert(std::abs(out[i]+one/4)<=2);
    assert(orbit_mpc_solve(h,g,warm,3,0,4,0,out,&res)==2);
    for(int i=0;i<QRC_MPC_N;++i) assert(out[i]==0);
    h[1]=one*2;
    assert(orbit_mpc_solve(h,g,warm,3,0,4,1,out,&res)==1);
    for(int i=0;i<QRC_MPC_N;++i) assert(out[i]==0);
}
