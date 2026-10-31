#include "lunar_harmonics.h"
#include <cmath>

// Cartesian solid harmonics and their analytic gradients. No latitude,
// longitude or division by cos(latitude): valid at both poles.
struct D { double v, x, y, z; };
static D add(D a,D b) { return {a.v+b.v,a.x+b.x,a.y+b.y,a.z+b.z}; }
static D scale(double k,D a) { return {k*a.v,k*a.x,k*a.y,k*a.z}; }
static D mul(D a,D b) { return {a.v*b.v,a.x*b.v+a.v*b.x,a.y*b.v+a.v*b.y,a.z*b.v+a.v*b.z}; }

extern "C" int lunar_harmonics_acceleration(const double r[3], int degree, double gm, double radius,
                                           const double c[], const double s[], double acceleration[3], double *potential) {
    if (degree<0 || degree>360 || !std::isfinite(gm) || gm<=0 || !std::isfinite(radius) || radius<=0) return 1;
    for (int j=0;j<3;++j) if (!std::isfinite(r[j])) return 1;
    double r2=r[0]*r[0]+r[1]*r[1]+r[2]*r[2], rn=std::sqrt(r2);
    if (!std::isfinite(rn) || rn==0) return 1;
    double q=radius/r2;
    D X={r[0]*q,q-2*r[0]*r[0]*q/r2,-2*r[0]*r[1]*q/r2,-2*r[0]*r[2]*q/r2};
    D Y={r[1]*q,-2*r[1]*r[0]*q/r2,q-2*r[1]*r[1]*q/r2,-2*r[1]*r[2]*q/r2};
    D Z={r[2]*q,-2*r[2]*r[0]*q/r2,-2*r[2]*r[1]*q/r2,q-2*r[2]*r[2]*q/r2};
    D Q={radius*q,-2*radius*q*r[0]/r2,-2*radius*q*r[1]/r2,-2*radius*q*r[2]/r2};
    D diagonal={radius/rn,-radius*r[0]/(rn*r2),-radius*r[1]/(rn*r2),-radius*r[2]/(rn*r2)};
    D diagonal_s={0,0,0,0}, sum={0,0,0,0};
    for (int m=0;m<=degree;++m) {
        if (m>0) {
            double f=m==1 ? std::sqrt(3.0) : std::sqrt((2.0*m+1)/(2.0*m));
            D next=scale(f,add(mul(X,diagonal),scale(-1,mul(Y,diagonal_s))));
            diagonal_s=scale(f,add(mul(Y,diagonal),mul(X,diagonal_s)));
            diagonal=next;
        }
        D v2={0,0,0,0},w2={0,0,0,0},v1=diagonal,w1=diagonal_s;
        for (int n=m;n<=degree;++n) {
            D v=v1,w=w1;
            if (n>m) {
                double a=std::sqrt((4.0*n*n-1)/(double(n*n-m*m)));
                double b=n==m+1 ? 0 : std::sqrt((2.0*n+1)*((n-1.0)*(n-1.0)-m*m)/((2.0*n-3)*(n*n-m*m)));
                v=add(scale(a,mul(Z,v1)),scale(-b,mul(Q,v2)));
                w=add(scale(a,mul(Z,w1)),scale(-b,mul(Q,w2)));
                v2=v1; w2=w1; v1=v; w1=w;
            }
            int i=n*(n+1)/2+m;
            if (!std::isfinite(c[i]) || !std::isfinite(s[i])) return 1;
            sum=add(sum,add(scale(c[i],v),scale(s[i],w)));
        }
    }
    sum=scale(gm/radius,sum);
    if (!std::isfinite(sum.v) || !std::isfinite(sum.x) || !std::isfinite(sum.y) || !std::isfinite(sum.z)) return 1;
    acceleration[0]=sum.x; acceleration[1]=sum.y; acceleration[2]=sum.z;
    *potential=sum.v; // Positive gravitational potential; acceleration = grad(U).
    return 0;
}
