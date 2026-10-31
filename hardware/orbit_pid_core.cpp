#include "orbit_pid_core.h"
static_assert(QRC_PID_BITS>=16 && QRC_PID_BITS<=32,"supported word width");
static_assert(QRC_PID_FRAC>0 && QRC_PID_FRAC<QRC_PID_BITS-1,"fraction width");
static const int64_t HI=(int64_t(1)<<(QRC_PID_BITS-1))-1;
static const int64_t LO=-(int64_t(1)<<(QRC_PID_BITS-1));
static int32_t sat(int64_t v, bool &fault) {
    if(v>HI){fault=true;return int32_t(HI);}
    if(v<LO){fault=true;return int32_t(LO);}
    return int32_t(v);
}
static int32_t add(int32_t a,int32_t b,bool &f){return sat(int64_t(a)+b,f);}
static int32_t sub(int32_t a,int32_t b,bool &f){return sat(int64_t(a)-b,f);}
static int32_t mul(int32_t a,int32_t b,bool &f){
    const int64_t x=int64_t(a)*int64_t(b), half=int64_t(1)<<(QRC_PID_FRAC-1);
    const int64_t rounded=x<0 ? -((-x+half)>>QRC_PID_FRAC) : ((x+half)>>QRC_PID_FRAC);
    return sat(rounded,f);
}
static int32_t limit(int32_t x,int32_t bound){return x>bound?bound:(x<-bound?-bound:x);}
static bool in_range(int32_t x){return int64_t(x)>=LO && int64_t(x)<=HI;}
extern "C" int orbit_pid_step(const int32_t e[3],const int32_t v[3],const int32_t ff[3],
    const int32_t c[6],int valid,int32_t state[6],int32_t out[3]){
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE s_axilite port=e bundle=control
    #pragma HLS INTERFACE s_axilite port=v bundle=control
    #pragma HLS INTERFACE s_axilite port=ff bundle=control
    #pragma HLS INTERFACE s_axilite port=c bundle=control
    #pragma HLS INTERFACE s_axilite port=state bundle=control
    #pragma HLS INTERFACE s_axilite port=out bundle=control
    #pragma HLS INTERFACE s_axilite port=valid bundle=control
    for(int j=0;j<3;++j)out[j]=0;
    for(int j=0;j<6;++j)if(!in_range(c[j]) || !in_range(state[j]))return 1;
    if(c[0]<0 || c[1]<0 || c[2]<0 || c[3]<0 || c[3]>(int64_t(1)<<QRC_PID_FRAC)
       || c[4]<=0 || c[5]<0 || c[5]>c[4])return 1;
    if(!valid)return 2;
    for(int j=0;j<3;++j)if(!in_range(e[j]) || !in_range(v[j]) || !in_range(ff[j]))return 1;
    bool fault=false;int status=0;int32_t next[6],commands[3];
    for(int j=0;j<3;++j){
        #pragma HLS UNROLL
        const int32_t derivative=add(state[j+3],mul(c[3],sub(v[j],state[j+3],fault),fault),fault);
        const int32_t increment=mul(c[2],e[j],fault);
        int32_t integral=limit(add(state[j],increment,fault),c[5]);
        const int32_t pd=sub(sub(ff[j],mul(c[0],e[j],fault),fault),mul(c[1],derivative,fault),fault);
        int32_t requested=sub(pd,integral,fault);
        // Conditional integration: freeze only when it worsens saturation.
        if((requested>c[4] && increment<0)||(requested<-c[4] && increment>0)){
            integral=state[j];requested=sub(pd,integral,fault);
        }
        commands[j]=limit(requested,c[4]);
        if(commands[j]!=requested)status|=8;
        next[j]=integral;next[j+3]=derivative;
    }
    if(fault)return 4;
    for(int j=0;j<6;++j)state[j]=next[j];
    for(int j=0;j<3;++j)out[j]=commands[j];
    return status;
}
