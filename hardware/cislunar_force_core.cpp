#include "cislunar_force_core.h"
#include <cmath>

static double norm(const double x[3]) {
    return std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
}
static double dot(const double x[3], const double y[3]) {
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
}
static double clamp(double x) { return x < -1 ? -1 : (x > 1 ? 1 : x); }

extern "C" int cislunar_acceleration(const double r[3], const double env[9],
                                    const double p[7], double out[3]) {
    #pragma HLS INLINE
    for (int i=0; i<3; ++i) if (!std::isfinite(r[i])) return 1;
    for (int i=0; i<9; ++i) if (!std::isfinite(env[i])) return 1;
    for (int i=0; i<7; ++i) if (!std::isfinite(p[i]) || p[i]<0) return 1;
    if (p[0]==0 || p[3]==0 || p[6]==0) return 1;
    double rn=norm(r), kn=norm(env+6);
    if (rn==0 || kn==0) return 1;
    double z=dot(r, env+6)/kn;
    double factor=1.5*p[4]*p[0]*p[3]*p[3]/(rn*rn*rn*rn*rn);
    double result[3];
    for (int j=0; j<3; ++j)
        result[j]=-p[0]*r[j]/(rn*rn*rn)
            +factor*((5*z*z/(rn*rn)-1)*r[j]-2*z*env[6+j]/kn);
    for (int body=0; body<2; ++body) {
        double d[3];
        for (int j=0; j<3; ++j) d[j]=r[j]-env[3*body+j];
        double dn=norm(d), bn=norm(env+3*body);
        if (dn==0 || bn==0) return 1;
        for (int j=0; j<3; ++j)
            result[j]-=p[1+body]*(d[j]/(dn*dn*dn)+env[3*body+j]/(bn*bn*bn));
    }
    if (p[5]>0) {
        const double solar_radius=695700e3, au=149597870700.0;
        double to_sun[3];
        for (int j=0; j<3; ++j) to_sun[j]=env[3+j]-r[j];
        double ds=norm(to_sun);
        if (ds<=solar_radius) return 1;
        double sun_angle=std::asin(solar_radius/ds), visibility=1;
        for (int body=0; body<2; ++body) {
            double to_body[3];
            for (int j=0; j<3; ++j) to_body[j]=(body==0 ? 0 : env[j])-r[j];
            double db=norm(to_body), radius=(body==0 ? p[3] : p[6]);
            if (db<=radius) return 1;
            if (db<ds && std::acos(clamp(dot(to_body,to_sun)/(db*ds)))+sun_angle<=std::asin(radius/db))
                visibility=0;
        }
        for (int j=0; j<3; ++j)
            result[j]-=visibility*p[5]*au*au*to_sun[j]/(ds*ds*ds);
    }
    for (int j=0; j<3; ++j) if (!std::isfinite(result[j])) return 1;
    for (int j=0; j<3; ++j) out[j]=result[j];
    return 0;
}

extern "C" int cislunar_verlet_step(double pos[3], double vel[3], double dt,
                                   const double env0[9], const double env1[9], const double p[7]) {
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE s_axilite port=pos bundle=control
    #pragma HLS INTERFACE s_axilite port=vel bundle=control
    #pragma HLS INTERFACE s_axilite port=dt bundle=control
    #pragma HLS INTERFACE s_axilite port=env0 bundle=control
    #pragma HLS INTERFACE s_axilite port=env1 bundle=control
    #pragma HLS INTERFACE s_axilite port=p bundle=control
    if (!std::isfinite(dt) || dt<=0) return 1;
    for (int j=0; j<3; ++j) if (!std::isfinite(vel[j])) return 1;
    double a0[3], a1[3], next[3], half[3], vnext[3];
    if (cislunar_acceleration(pos,env0,p,a0)) return 1;
    if (norm(pos)<=p[3]) return 1;
    for (int j=0; j<3; ++j) {
        half[j]=vel[j]+0.5*dt*a0[j];
        next[j]=pos[j]+dt*half[j];
    }
    if (cislunar_acceleration(next,env1,p,a1) || norm(next)<=p[3]) return 1;
    for (int j=0; j<3; ++j) {
        vnext[j]=half[j]+0.5*dt*a1[j];
        if (!std::isfinite(vnext[j])) return 1;
    }
    for (int j=0; j<3; ++j) { pos[j]=next[j]; vel[j]=vnext[j]; }
    return 0;
}
