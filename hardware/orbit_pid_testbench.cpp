#include "orbit_pid_core.h"
#include <cassert>
int main(){
    const int32_t one=int32_t(1)<<QRC_PID_FRAC;
    int32_t e[3]={one,0,0},v[3]={0,0,0},ff[3]={0,0,0};
    int32_t c[6]={one/4,one/2,one/16,one,one,one/2},state[6]={},out[3]={};
    assert(orbit_pid_step(e,v,ff,c,1,state,out)==0);
    assert(out[0]==-5*one/16 && state[0]==one/16);
    assert(orbit_pid_step(e,v,ff,c,0,state,out)==2);
    assert(out[0]==0 && state[0]==one/16);
    c[4]=one/8;c[5]=one/8;
    const int32_t previous=state[0];
    assert(orbit_pid_step(e,v,ff,c,1,state,out)==8);
    assert(out[0]==-one/8 && state[0]==previous);
    return 0;
}
