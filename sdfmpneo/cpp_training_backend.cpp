#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32)
#define SDF_EXPORT extern "C" __declspec(dllexport)
#else
#define SDF_EXPORT extern "C" __attribute__((visibility("default")))
#endif

static int g_threads = 1;

SDF_EXPORT const char* sdfmpneo_training_backend_version() { return "sdfmpneo_training_cpp_v1"; }
SDF_EXPORT int sdfmpneo_training_backend_abi() { return 1; }
SDF_EXPORT int sdfmpneo_training_backend_has_openmp() {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}
SDF_EXPORT void sdfmpneo_training_backend_set_threads(int n) {
    g_threads = std::max(1, n);
#ifdef _OPENMP
    omp_set_num_threads(g_threads);
#endif
}

static inline bool inv3(const double* J, double* inv, double& det) {
    const double a=J[0], b=J[1], c=J[2];
    const double d=J[3], e=J[4], f=J[5];
    const double g=J[6], h=J[7], i=J[8];
    const double A=e*i-f*h, B=c*h-b*i, C=b*f-c*e;
    const double D=f*g-d*i, E=a*i-c*g, F=c*d-a*f;
    const double G=d*h-e*g, H=b*g-a*h, I=a*e-b*d;
    det = a*A + b*D + c*G;
    if (!(std::isfinite(det)) || std::abs(det) <= 0.0) return false;
    const double s=1.0/det;
    inv[0]=A*s; inv[1]=B*s; inv[2]=C*s;
    inv[3]=D*s; inv[4]=E*s; inv[5]=F*s;
    inv[6]=G*s; inv[7]=H*s; inv[8]=I*s;
    return true;
}

static inline bool tet_geometry(const double* vertices, const int64_t* tet, double* grad, double& volume) {
    const double* x0 = vertices + 3*tet[0];
    const double* x1 = vertices + 3*tet[1];
    const double* x2 = vertices + 3*tet[2];
    const double* x3 = vertices + 3*tet[3];
    double J[9] = {
        x1[0]-x0[0], x2[0]-x0[0], x3[0]-x0[0],
        x1[1]-x0[1], x2[1]-x0[1], x3[1]-x0[1],
        x1[2]-x0[2], x2[2]-x0[2], x3[2]-x0[2]
    };
    double inv[9], det;
    if (!inv3(J, inv, det) || det <= 0.0) return false;
    volume = det / 6.0;
    // grad(lambda_1..3) are rows of J^{-1}; lambda_0 is minus their sum.
    for (int k=0;k<3;++k) {
        grad[1*3+k] = inv[0*3+k];
        grad[2*3+k] = inv[1*3+k];
        grad[3*3+k] = inv[2*3+k];
        grad[0*3+k] = -(grad[3+k] + grad[6+k] + grad[9+k]);
    }
    return true;
}

SDF_EXPORT int sdfmpneo_p1_thermal_local_f64(
    const double* vertices, const int64_t* tets, int64_t n_tet,
    const double* rho, const double* kappa,
    double* mass_out, double* stiff_out, double* volume_out) {
    int failed = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(g_threads > 1) reduction(|:failed)
#endif
    for (int64_t q=0;q<n_tet;++q) {
        double grad[12], V;
        if (!tet_geometry(vertices, tets+4*q, grad, V)) { failed |= 1; continue; }
        if (volume_out) volume_out[q]=V;
        for (int a=0;a<4;++a) for (int b=0;b<4;++b) {
            const int idx = int(q*16 + a*4+b);
            mass_out[idx] = rho[q]*V*((a==b)?0.1:0.05);
            double dot=0.0;
            for (int k=0;k<3;++k) dot += grad[3*a+k]*grad[3*b+k];
            stiff_out[idx] = kappa[q]*V*dot;
        }
    }
    return failed ? 2 : 0;
}

static inline void local_pairs(const int64_t* tet, int pairs[12]) {
    const int fixed[12]={0,1,0,2,0,3,1,2,1,3,2,3};
    for (int p=0;p<6;++p) {
        int i=fixed[2*p], j=fixed[2*p+1];
        if (tet[i] > tet[j]) std::swap(i,j);
        pairs[2*p]=i; pairs[2*p+1]=j;
    }
}

SDF_EXPORT int sdfmpneo_nedelec_local_f64(
    const double* vertices, const int64_t* tets, int64_t n_tet,
    const double* nu, const double* sigma,
    double* stiff_out, double* mass_out) {
    int failed=0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(g_threads > 1) reduction(|:failed)
#endif
    for (int64_t q=0;q<n_tet;++q) {
        double grad[12], V;
        const int64_t* tet=tets+4*q;
        if (!tet_geometry(vertices, tet, grad, V)) { failed |= 1; continue; }
        int pairs[12]; local_pairs(tet,pairs);
        double curls[18];
        for (int p=0;p<6;++p) {
            const int a=pairs[2*p], b=pairs[2*p+1];
            const double* ga=grad+3*a; const double* gb=grad+3*b;
            curls[3*p+0]=2.0*(ga[1]*gb[2]-ga[2]*gb[1]);
            curls[3*p+1]=2.0*(ga[2]*gb[0]-ga[0]*gb[2]);
            curls[3*p+2]=2.0*(ga[0]*gb[1]-ga[1]*gb[0]);
        }
        for (int p=0;p<6;++p) {
            const int i=pairs[2*p], j=pairs[2*p+1];
            const double* gi=grad+3*i; const double* gj=grad+3*j;
            for (int r=0;r<6;++r) {
                const int k=pairs[2*r], l=pairs[2*r+1];
                const double* gk=grad+3*k; const double* gl=grad+3*l;
                auto integ=[&](int aa,int bb){return V*((aa==bb)?0.1:0.05);};
                auto dot3=[](const double* aa,const double* bb){return aa[0]*bb[0]+aa[1]*bb[1]+aa[2]*bb[2];};
                const double mass = dot3(gj,gl)*integ(i,k) - dot3(gj,gk)*integ(i,l)
                                  - dot3(gi,gl)*integ(j,k) + dot3(gi,gk)*integ(j,l);
                const int idx=int(q*36+p*6+r);
                mass_out[idx]=sigma[q]*mass;
                stiff_out[idx]=nu[q]*V*dot3(curls+3*p,curls+3*r);
            }
        }
    }
    return failed ? 2 : 0;
}

static inline double factorial_ratio_moment(const int32_t* p, int first, int second) {
    int a[4]={p[0],p[1],p[2],p[3]};
    ++a[first]; ++a[second];
    const int degree=a[0]+a[1]+a[2]+a[3];
    long double logv=std::log(6.0L);
    for (int k=0;k<4;++k) logv += std::lgamma((long double)a[k]+1.0L);
    logv -= std::lgamma((long double)degree+4.0L);
    return (double)std::exp(logv);
}

static inline void moments_from_terms(
    double volume, const int64_t* offsets, int64_t poly_index,
    const int32_t* powers, const double* coeffs, double* moments) {
    for (int i=0;i<16;++i) moments[i]=0.0;
    const int64_t begin=offsets[poly_index], end=offsets[poly_index+1];
    for (int64_t s=begin;s<end;++s) {
        const int32_t* p=powers+4*s;
        const double c=coeffs[s]*volume;
        for (int i=0;i<4;++i) for (int j=0;j<4;++j)
            moments[4*i+j] += c*factorial_ratio_moment(p,i,j);
    }
}

static inline void moments_from_product_terms(
    double volume,
    const int64_t* aoff, int64_t ai, const int32_t* apow, const double* acoef,
    const int64_t* boff, int64_t bi, const int32_t* bpow, const double* bcoef,
    double* moments) {
    for (int i=0;i<16;++i) moments[i]=0.0;
    const int64_t ab=aoff[ai], ae=aoff[ai+1], bb=boff[bi], be=boff[bi+1];
    int32_t p[4];
    for (int64_t s=ab;s<ae;++s) for (int64_t t=bb;t<be;++t) {
        for (int k=0;k<4;++k) p[k]=apow[4*s+k]+bpow[4*t+k];
        const double c=acoef[s]*bcoef[t]*volume;
        for (int i=0;i<4;++i) for (int j=0;j<4;++j)
            moments[4*i+j] += c*factorial_ratio_moment(p,i,j);
    }
}

static inline void reduced_add(
    const double* coeff, const double* fields_interleaved, int nred,
    const double* moments, double* out_interleaved) {
    double local[36];
    for (int p=0;p<6;++p) for (int r=0;r<6;++r) {
        double v=0.0;
        for (int i=0;i<4;++i) for (int j=0;j<4;++j) {
            double ck=0.0;
            for (int d=0;d<3;++d) ck += coeff[(p*4+i)*3+d]*coeff[(r*4+j)*3+d];
            v += ck*moments[4*i+j];
        }
        local[p*6+r]=v;
    }
    for (int a=0;a<nred;++a) for (int b=0;b<nred;++b) {
        double re=0.0, im=0.0;
        for (int p=0;p<6;++p) {
            const double fpa_re=fields_interleaved[2*(p*nred+a)+0];
            const double fpa_im=fields_interleaved[2*(p*nred+a)+1];
            for (int r=0;r<6;++r) {
                const double l=local[p*6+r];
                const double frb_re=fields_interleaved[2*(r*nred+b)+0];
                const double frb_im=fields_interleaved[2*(r*nred+b)+1];
                re += l*(fpa_re*frb_re + fpa_im*frb_im);
                im += l*(fpa_re*frb_im - fpa_im*frb_re);
            }
        }
        out_interleaved[2*(a*nred+b)+0] += re;
        out_interleaved[2*(a*nred+b)+1] += im;
    }
}

SDF_EXPORT int sdfmpneo_reduced_assemble_many_f64(
    const double* volumes, const double* coefficients, const double* fields_interleaved,
    int64_t n_tet, int nred, int nfamily,
    const int64_t* offsets, const int32_t* powers, const double* poly_coeffs,
    double* out_interleaved) {
    const int64_t outn=(int64_t)nfamily*nred*nred*2;
    std::fill(out_interleaved,out_interleaved+outn,0.0);
    // Preserve tetrahedron accumulation order. Python point/context parallelism supplies concurrency.
    for (int64_t q=0;q<n_tet;++q) {
        const double* cq=coefficients + q*6*4*3;
        const double* fq=fields_interleaved + q*6*nred*2;
        for (int f=0;f<nfamily;++f) {
            const int64_t pi=(int64_t)f*n_tet+q;
            if (offsets[pi]==offsets[pi+1]) continue;
            double moments[16]; moments_from_terms(volumes[q],offsets,pi,powers,poly_coeffs,moments);
            reduced_add(cq,fq,nred,moments,out_interleaved+(int64_t)f*nred*nred*2);
        }
    }
    return 0;
}

SDF_EXPORT int sdfmpneo_reduced_assemble_products_f64(
    const double* volumes, const double* coefficients, const double* fields_interleaved,
    int64_t n_tet, int nred, int nfamily, int nmult,
    const int64_t* aoff, const int32_t* apow, const double* acoef,
    const int64_t* boff, const int32_t* bpow, const double* bcoef,
    double* out_interleaved) {
    const int64_t outn=(int64_t)nmult*nfamily*nred*nred*2;
    std::fill(out_interleaved,out_interleaved+outn,0.0);
    for (int64_t q=0;q<n_tet;++q) {
        const double* cq=coefficients + q*6*4*3;
        const double* fq=fields_interleaved + q*6*nred*2;
        for (int m=0;m<nmult;++m) {
            const int64_t bi=(int64_t)m*n_tet+q;
            if (boff[bi]==boff[bi+1]) continue;
            for (int f=0;f<nfamily;++f) {
                const int64_t ai=(int64_t)f*n_tet+q;
                if (aoff[ai]==aoff[ai+1]) continue;
                double moments[16];
                moments_from_product_terms(volumes[q],aoff,ai,apow,acoef,boff,bi,bpow,bcoef,moments);
                double* out=out_interleaved+((int64_t)m*nfamily+f)*nred*nred*2;
                reduced_add(cq,fq,nred,moments,out);
            }
        }
    }
    return 0;
}
