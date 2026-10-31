#pragma once
#include "backend.hpp"
#include <cmath>
#include <cstddef>
#ifdef LU_DOUBLE
using Real=double;
#else
using Real=float;
#endif
constexpr double pi=3.14159265358979323846;
template<class T> HD T sq(T x){return x*x;}
HD inline Real bounded(Real p){return p<0?0:(p>1?1:p);}
struct Vec {
    Real x,y,z;
    HD Real operator[](int a)const{return a==0?x:(a==1?y:z);}
};
HD inline Vec e15(int q) {
    // Ordering inherited from baseline Flux.cpp: axes, opposite diagonal pairs, rest.
    switch(q) {
    case 0:return {1,0,0};case 1:return {-1,0,0};
    case 2:return {0,1,0};case 3:return {0,-1,0};
    case 4:return {0,0,1};case 5:return {0,0,-1};
    case 6:return {1,1,1};case 7:return {-1,-1,-1};
    case 8:return {1,1,-1};case 9:return {-1,-1,1};
    case 10:return {1,-1,1};case 11:return {-1,1,-1};
    case 12:return {1,-1,-1};case 13:return {-1,1,1};
    default:return {0,0,0};
    }
}
HD inline Real w15(int q){return q<6?Real(1./9):q<14?Real(1./72):Real(2./9);}
HD inline Real equilibrium(int q,Real p,Real rho,Vec u) {
    Vec e=e15(q);Real eu=e.x*u.x+e.y*u.y+e.z*u.z;
    return w15(q)*(p+rho*(eu+Real(1.5)*eu*eu-Real(.5)*(sq(u.x)+sq(u.y)+sq(u.z))));
}
HD inline Real delta_cos(Real r) {
    r=::fabs(r);return r<2?Real(.25)*(1+::cos(Real(pi*.5)*r)):0;
}
HD inline Real delta_weight(Vec r) {return delta_cos(r.x)*delta_cos(r.y)*delta_cos(r.z);}
HD inline Real weno5(Real a,Real b,Real c,Real d,Real e) {
    Real b0=Real(13./12)*sq(a-2*b+c)+Real(.25)*sq(a-4*b+3*c);
    Real b1=Real(13./12)*sq(b-2*c+d)+Real(.25)*sq(b-d);
    Real b2=Real(13./12)*sq(c-2*d+e)+Real(.25)*sq(3*c-4*d+e);
    Real a0=Real(.1)/sq(Real(1.e-6)+b0),a1=Real(.6)/sq(Real(1.e-6)+b1),a2=Real(.3)/sq(Real(1.e-6)+b2);
    return (a0*(2*a-7*b+11*c)+a1*(-b+5*c+2*d)+a2*(2*c+5*d-e))/(6*(a0+a1+a2));
}
#include "metric_grid.hpp"
struct Physics {
    Real rho0,rho1,nu0,nu1,sigma,xi,mobility,gravity,level;
    int periodic;
    int pressure_product_rule=0; // 0: centered u.grad(rho); 1: compatible face product rule.
    int macro_face_density=0; // Diagnostic alternative for the ambiguous density in Lu (12).
    int transport_mode=0; // 0: legacy; 1: shared velocity; 2: shared mass flux; 3: bounded FCT mass flux.
    int clamp_density=1; // Diagnostic switch to isolate the nonlinear material remapping.
    int capillary_mode=0; // 0: -phi*D(mu); 1: conservative discrete product rule.
    Real ib_phase_scale=1; // Controlled source-sign/zero-source ablation, default Yan interpretation.
    Real smagorinsky_cs=0; // Explicit SGS uncertainty study; not claimed to be Lu's hidden model.
    int ib_phase_bound=0; // Conservative minimum-change projection of the regularized IB source.
    int momentum_advection=0; // 0: LBFS face velocity; 1: donor; 2: monotonized-central reconstruction.
    int cache_faces=0; // Cache FCT's first LBFS reconstruction; extra 22 Real/cell.
    int top_pressure_outlet=0; // p=0 at upper face, zero normal gradients of velocity/phase.
    HD Real rho(Real phi)const{return rho0+(rho1-rho0)*(clamp_density?bounded(phi):phi);}
    HD Real beta()const{return 12*sigma/xi;}
    HD Real kappa()const{return Real(1.5)*sigma*xi;}
};
HD inline int map_index(int i,int n,bool periodic) {
    if(periodic){i%=n;return i<0?i+n:i;}
    // Supports WENO halos even for small verification grids.
    while(i<0||i>=n){if(i<0)i=-i-1;else i=2*n-i-1;}return i;
}
struct Field {
    Real* a;Grid g;Physics p;const Real* gradient_cache=nullptr;
    // ALE data are expressed in the current local frame.  The frame follows
    // the blade midpoint and yaws about the vertical axis; z therefore stays
    // vertical.  A face velocity is Cdot_local + (0,0,yaw_rate)x.
    Vec ale_translation{0,0,0};
    Vec ale_origin{0,0,0};
    Real ale_yaw_rate=0;
    int ale_enabled=0;
    HD Real raw(int v,size_t s)const{return a[size_t(v)*g.n()+s];}
    HD Real value(int v,int i,int j,int k)const {
        int ii=map_index(i,g.nx,p.periodic),jj=map_index(j,g.ny,p.periodic),kk=map_index(k,g.nz,p.periodic);
        size_t s=g.id(ii,jj,kk);Real r=raw(v,s);
        if(p.periodic)return r;
        bool outlet=p.top_pressure_outlet&&i>=g.nx;
        if(v==0&&outlet)r=-r;
        else if(v==0&&i!=ii)r-=p.rho(raw(4,s))*p.gravity*(g.center(0,i)-g.center(0,ii));
        if(v>=1&&v<=3){int flips=(!outlet&&i!=ii)+(j!=jj)+(k!=kk);if(flips%2)r=-r;}
        return r;
    }
    HD Real primitive(int v,int i,int j,int k)const {
        Real a=value(v,i,j,k);return v>=1&&v<=3?3*a/p.rho(value(4,i,j,k)):a;
    }
    HD Real shifted(int v,int i,int j,int k,int axis,int step,bool prim=false)const{
        if(axis==0)i+=step;else if(axis==1)j+=step;else k+=step;
        return prim?primitive(v,i,j,k):value(v,i,j,k);
    }
};
HD inline Vec ale_mesh_velocity(Field f,Vec x) {
    if(!f.ale_enabled) return {0,0,0};
    return {f.ale_translation.x-f.ale_yaw_rate*(x.y-f.ale_origin.y),
            f.ale_translation.y+f.ale_yaw_rate*(x.x-f.ale_origin.x),
            f.ale_translation.z};
}
HD inline Vec ale_face_velocity(Field f,int i,int j,int k,int axis) {
    int index=axis==0?i:axis==1?j:k;
    Vec x{f.g.center(0,i),f.g.center(1,j),f.g.center(2,k)};
    if(axis==0)x.x=f.g.face(0,index+1); else if(axis==1)x.y=f.g.face(1,index+1); else x.z=f.g.face(2,index+1);
    return ale_mesh_velocity(f,x);
}
HD inline Real scalar_at(const Real* a,Grid g,Physics p,int i,int j,int k) {
    return a[g.id(map_index(i,g.nx,p.periodic),map_index(j,g.ny,p.periodic),map_index(k,g.nz,p.periodic))];
}
#include "metric_operators.hpp"
HD inline Real laplacian(const Real* a,Grid g,Physics p,int i,int j,int k) {
    if(g.stretched())return metric_laplacian(a,g,p,i,j,k);
    Real c=scalar_at(a,g,p,i,j,k);
    Real dx=(scalar_at(a,g,p,i+1,j,k)-c)+(scalar_at(a,g,p,i-1,j,k)-c);
    Real dy=(scalar_at(a,g,p,i,j+1,k)-c)+(scalar_at(a,g,p,i,j-1,k)-c);
    Real dz=(scalar_at(a,g,p,i,j,k+1)-c)+(scalar_at(a,g,p,i,j,k-1)-c);
    return (dx+dy+dz)/(g.h*g.h);
}
HD inline Real capillary_force(Field f,const Real* mu,int i,int j,int k,int axis) {
    if(f.g.stretched())return metric_capillary_force(f,mu,i,j,k,axis);
    int ip=i+(axis==0),jp=j+(axis==1),kp=k+(axis==2),im=i-(axis==0),jm=j-(axis==1),km=k-(axis==2);
    Real mp=scalar_at(mu,f.g,f.p,ip,jp,kp),mm=scalar_at(mu,f.g,f.p,im,jm,km);
    if(!f.p.capillary_mode)return -f.value(4,i,j,k)*(mp-mm)/(2*f.g.h);
    // -phi grad(mu) = grad(f_bulk-phi*mu)-kappa lap(phi) grad(phi).
    // Discrete potential gradient instead of a discrete chain-rule assumption.
    // For periodic Cartesian grids, symmetric L and skew D commute, hence
    // sum (L phi)*(D phi)=0: all three force integrals vanish to roundoff.
    // Adapted to Lu's pressure gauge from Huang et al. §3.2.2 (2020).
    Real pp=f.value(4,ip,jp,kp),pm=f.value(4,im,jm,km);
    Real ep=f.p.beta()*sq(pp*(1-pp))-pp*mp,em=f.p.beta()*sq(pm*(1-pm))-pm*mm;
    Real lap=laplacian(f.a+4*f.g.n(),f.g,f.p,i,j,k);
    return ((ep-em)-f.p.kappa()*lap*(pp-pm))/(2*f.g.h);
}
HD inline Real smagorinsky_viscosity(const Real gradient[3][3],Real h,Real cs) {
    Real strain2=0;
    for(int a=0;a<3;++a)for(int b=0;b<3;++b){Real s=Real(.5)*(gradient[a][b]+gradient[b][a]);strain2+=2*s*s;}
    return sq(cs*h)*::sqrt(strain2);
}
HD inline void lbfs_face(Field f,int i,int j,int k,int axis,Real out[4],Vec* velocity=nullptr) {
    if(f.g.stretched()){metric_lbfs_face(f,i,j,k,axis,out,velocity);return;}
    int ri=i+(axis==0),rj=j+(axis==1),rk=k+(axis==2);
    Real left[5],right[5],gl[5][3],gr[5][3];
    for(int v=0;v<5;++v){
        left[v]=f.primitive(v,i,j,k);right[v]=f.primitive(v,ri,rj,rk);
        for(int a=0;a<3;++a){
            gl[v][a]=(f.shifted(v,i,j,k,a,1,true)-f.shifted(v,i,j,k,a,-1,true))/2;
            gr[v][a]=(f.shifted(v,ri,rj,rk,a,1,true)-f.shifted(v,ri,rj,rk,a,-1,true))/2;
        }
    }
    Real phl=f.value(4,i,j,k),phr=f.value(4,ri,rj,rk);
    Real rl=f.p.rho(phl),rr=f.p.rho(phr),rf=(rl+rr)/2,ph=(bounded(phl)+bounded(phr))/2;
    Real departure[15],pf=0;Vec uf{0,0,0};
    for(int q=0;q<15;++q){
        Vec e=e15(q);Real v[5];
        for(int c=0;c<5;++c){
            Real vl=left[c],vr=right[c];
            for(int a=0;a<3;++a){
                vl+=gl[c][a]*(Real(.5)*(a==axis)-Real(.5)*e[a]);
                vr+=gr[c][a]*(-Real(.5)*(a==axis)-Real(.5)*e[a]);
            }
            v[c]=e[axis]>0?vl:e[axis]<0?vr:(vl+vr)/2;
        }
        // Eq. (5): density at the full 3D departure point, including transverse
        // displacement. Eq. (7) below still uses the common face density.
        Real rd=f.p.macro_face_density?rf:f.p.rho(v[4]);
        Real fe=equilibrium(q,v[0],rd,{v[1],v[2],v[3]});
        departure[q]=equilibrium(q,v[0],rf,{v[1],v[2],v[3]}); // Eq. (7): common face density.
        pf+=fe;uf.x+=fe*e.x;uf.y+=fe*e.y;uf.z+=fe*e.z;
    }
    uf.x*=3/rf;uf.y*=3/rf;uf.z*=3/rf;
    if(velocity)*velocity=uf;
    Real viscosity=f.p.nu0+(f.p.nu1-f.p.nu0)*ph;
    if(f.p.smagorinsky_cs>0){Real gradient[3][3];
        for(int a=0;a<3;++a)for(int b=0;b<3;++b)gradient[a][b]=(gl[a+1][b]+gr[a+1][b])/(2*f.g.h);
        viscosity+=smagorinsky_viscosity(gradient,f.g.h,f.p.smagorinsky_cs);
    }
    Real tau_minus_half=6*viscosity/f.g.h;
    Vec wm=ale_face_velocity(f,i,j,k,axis); Real un=uf[axis]-wm[axis];
    out[0]=rf*un/3;out[1]=out[2]=out[3]=0;
    for(int q=0;q<15;++q){
        Vec e=e15(q);Real eq=equilibrium(q,pf,rf,uf);
        Real hat=eq-tau_minus_half*(eq-departure[q]);
        out[1]+=e[axis]*e.x*hat;out[2]+=e[axis]*e.y*hat;out[3]+=e[axis]*e.z*hat;
    }
    // Reynolds transport theorem: subtract mesh advection of the conserved
    // momentum from the LBFS stress flux.
    out[1]-=wm[axis]*rf*uf.x/3; out[2]-=wm[axis]*rf*uf.y/3; out[3]-=wm[axis]*rf*uf.z/3;
}
HD inline Real phase_face(Field f,int i,int j,int k,int axis) {
    Real phi[6],u[6],alpha=0;
    for(int t=0;t<6;++t){phi[t]=f.shifted(4,i,j,k,axis,t-2);u[t]=f.shifted(axis+1,i,j,k,axis,t-2,true);Real a=::fabs(u[t]);alpha=a>alpha?a:alpha;}
    Real fp[5],fm[5];
    for(int t=0;t<5;++t){fp[t]=Real(.5)*phi[t]*(u[t]+alpha);int s=5-t;fm[t]=Real(.5)*phi[s]*(u[s]-alpha);}
    return weno5(fp[0],fp[1],fp[2],fp[3],fp[4])+weno5(fm[0],fm[1],fm[2],fm[3],fm[4]);
}
HD inline void consistent_momentum_flux(Field f,int i,int j,int k,int axis,Real out[5],Vec uf) {
    int ri=i+(axis==0),rj=j+(axis==1),rk=k+(axis==2);
    Real rf=metric_face_scalar(f.p.rho(f.value(4,i,j,k)),f.p.rho(f.value(4,ri,rj,rk)),f.g,axis,axis==0?i:axis==1?j:k);
    Vec wm=ale_face_velocity(f,i,j,k,axis); Real un=uf[axis]-wm[axis],massflux=f.p.rho0*un+(f.p.rho1-f.p.rho0)*out[4];
    for(int component=0;component<3;++component){
        Real advected=uf[component];
        if(f.p.momentum_advection){
            bool positive=massflux>=0;
            int ci=positive?i:ri,cj=positive?j:rj,ck=positive?k:rk;
            Real center=f.primitive(component+1,ci,cj,ck);advected=center;
            if(f.p.momentum_advection==2){
                int index=axis==0?ci:axis==1?cj:ck;
                Real dm=f.g.center(axis,index)-f.g.center(axis,index-1),dp=f.g.center(axis,index+1)-f.g.center(axis,index);
                Real dl=(center-f.shifted(component+1,ci,cj,ck,axis,-1,true))/dm;
                Real dr=(f.shifted(component+1,ci,cj,ck,axis,1,true)-center)/dp,slope=0;
                if(dl*dr>0){Real magnitude=::fabs((dm*dr+dp*dl)/(dm+dp));
                    magnitude=2*::fabs(dl)<magnitude?2*::fabs(dl):magnitude;
                    magnitude=2*::fabs(dr)<magnitude?2*::fabs(dr):magnitude;slope=dl>0?magnitude:-magnitude;}
                advected+=(positive?Real(.5):Real(-.5))*f.g.width(axis,index)*slope;
            }
        }
        out[component+1]+=(massflux*advected-rf*un*uf[component])/3;
    }
}
HD inline void coupled_face(Field f,const Real* mu,int i,int j,int k,int axis,Real out[5],const Real* fct=nullptr,Real* low_flux=nullptr,Vec* face_velocity=nullptr,bool defer_momentum=false) {
    Vec uf;lbfs_face(f,i,j,k,axis,out,&uf);
    if(face_velocity)*face_velocity=uf;
    Real phi[6];for(int t=0;t<6;++t)phi[t]=f.shifted(4,i,j,k,axis,t-2);
    Real left=f.g.stretched()?metric_weno(f,i,j,k,axis,true):weno5(phi[0],phi[1],phi[2],phi[3],phi[4]);
    Real right=f.g.stretched()?metric_weno(f,i,j,k,axis,false):weno5(phi[5],phi[4],phi[3],phi[2],phi[1]);
    Vec wm=ale_face_velocity(f,i,j,k,axis); Real un=uf[axis]-wm[axis];
    int ri=i+(axis==0),rj=j+(axis==1),rk=k+(axis==2);
    Real diff=f.p.mobility*(scalar_at(mu,f.g,f.p,ri,rj,rk)-scalar_at(mu,f.g,f.p,i,j,k))/(f.g.center(axis,(axis==0?i:axis==1?j:k)+1)-f.g.center(axis,axis==0?i:axis==1?j:k));
    out[4]=un*(un>=0?left:right)-diff;
    Real low=un*(un>=0?phi[2]:phi[3]);
    if(f.p.transport_mode==3){
        int index=axis==0?i:axis==1?j:k,extent=axis==0?f.g.nx:axis==1?f.g.ny:f.g.nz;
        bool outlet=f.p.top_pressure_outlet&&axis==0&&index==extent-1;
        if(!f.p.periodic&&(index<0||(index==extent-1&&!outlet))){low=0;out[4]=0;}
        else if(fct){
            size_t N=f.g.n();Real anti=scalar_at(fct+axis*N,f.g,f.p,i,j,k);
            Real donor=scalar_at(fct+(anti>=0?4:3)*N,f.g,f.p,i,j,k);
            Real receiver=scalar_at(fct+(anti>=0?3:4)*N,f.g,f.p,ri,rj,rk);
            out[4]=low+(donor<receiver?donor:receiver)*anti;
        }
    }
    if(low_flux)*low_flux=low;
    if(f.p.transport_mode>=2&&!defer_momentum){
        // Discrete consistent mass flux, specialized to phi in [0,1]:
        // J_rho = rho_g*U_face + (rho_l-rho_g)*J_phi, including CH diffusion.
        // Replace only advective momentum transport. Retain LBFS pressure and
        // non-equilibrium stress. Huang et al. (2020), arXiv:2010.01099, §3.5.
        consistent_momentum_flux(f,i,j,k,axis,out,uf);
    }
}
struct Marker {Vec x,n;Real area;};
HD inline Real support_weight(Grid g,Marker m,int i,int j,int k) {
    Vec x=g.position(i,j,k);return delta_weight({(x.x-m.x.x)/g.h,(x.y-m.x.y)/g.h,(x.z-m.x.z)/g.h});
}
HD inline void support_base(Grid g,Marker m,int& i,int& j,int& k){
    if(g.stretched()){
        i=g.lower_center(0,m.x.x)-1;j=g.lower_center(1,m.x.y)-1;k=g.lower_center(2,m.x.z)-1;
        if(i<0||j<0||k<0||i+3>=g.nx||j+3>=g.ny||k+3>=g.nz){i=j=k=-1;return;}
        for(int a=0;a<3;++a)for(int q=0;q<4;++q)if(::fabs(g.width(a,(a==0?i:a==1?j:k)+q)-g.h)>Real(.0005)*g.h){i=j=k=-1;return;}
        return;
    }

    Real x=m.x.x/g.h-Real(.5),y=m.x.y/g.h-Real(.5),z=m.x.z/g.h-Real(.5);
    // Reject nonfinite/escaped markers BEFORE float-to-int conversion. A
    // divergent body coordinate must not overflow integer halo indices.
    if(!(x>=1&&x<g.nx-2&&y>=1&&y<g.ny-2&&z>=1&&z<g.nz-2)){i=j=k=-1;return;}
    i=int(::floor(x))-1;j=int(::floor(y))-1;k=int(::floor(z))-1;
}

#include "metric_stress_vector.hpp"
