#pragma once
#include "numerics.hpp"
#include "parallel_force_reduce.hpp"
#include "ale_motion.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cstdint>
struct Body {
    Real cx,cy,z0,v0,radius,length,density;
    int enabled,free;
    int shape=0; // 0 rectangular plate, 1 vertical cylinder
    int dual=0; Real cx2=0,cy2=0,z02=0;
    Real plate_height=100,plate_thickness=4;
    Real ramp_time=0,release_time=0;
    int inertia_mode=0; // 0: rigid gas; 1: measured explicit; 2: measured simultaneous body/IB solve.
    HD double prescribed_velocity(double t)const{
        return ramp_time>0&&t<ramp_time?double(v0)*t/ramp_time:double(v0);
    }
    HD double prescribed_position(double t)const{
        double integral=ramp_time>0&&t<ramp_time?.5*t*t/ramp_time:t-.5*ramp_time;
        return double(z0)+double(v0)*integral;
    }
    HD Real volume()const{return shape==1?Real(pi)*radius*radius*length:(dual?Real(2):Real(1))*plate_height*plate_thickness*length;}
    HD Real signed_distance(Vec x,Real z)const{
        if(shape==1){Real q=::sqrt(sq(x.x-cx)+sq(x.y-cy))-radius,c=::fabs(x.z-z)-length/2;Real aq=q>0?q:0,ac=c>0?c:0,m=q>c?q:c;return ::sqrt(aq*aq+ac*ac)+(m<0?m:0);}
        Real a=::fabs(x.x-cx)-plate_height/2,b=::fabs(x.y-cy)-length/2,c=::fabs(x.z-z)-plate_thickness/2;Real ap=a>0?a:0,bp=b>0?b:0,cp=c>0?c:0,m=a>b?a:b;m=m>c?m:c;Real d=::sqrt(ap*ap+bp*bp+cp*cp)+(m<0?m:0);
        if(dual){a=::fabs(x.x-cx2)-plate_height/2;b=::fabs(x.y-cy2)-length/2;c=::fabs(x.z-(z02+z-z0))-plate_thickness/2;ap=a>0?a:0;bp=b>0?b:0;cp=c>0?c:0;m=a>b?a:b;m=m>c?m:c;Real d2=::sqrt(ap*ap+bp*bp+cp*cp)+(m<0?m:0);if(d2<d)d=d2;}return d;
    }
    HD Real volume_fraction(Vec x,Real z,Real h)const{
        if(shape==1){int inside=0;for(int a=-1;a<=1;++a)for(int bb=-1;bb<=1;++bb)for(int cc=-1;cc<=1;++cc){Real xx=x.x+Real(a)*h*.4,yy=x.y+Real(bb)*h*.4,zz=x.z+Real(cc)*h*.4;if(sq(xx-cx)+sq(yy-cy)<=sq(radius)&&::fabs(zz-z)<=length/2)++inside;}return Real(inside)/27;}
        // Exact overlap for an axis-aligned rectangular solid. Unlike a signed
        // distance ramp, its total volume is invariant under subcell translation,
        // including plates thinner than a cell. This is geometry, not filtering.
        Real c[3]={cx,cy,z},extent[3]={plate_height,length,plate_thickness},f=1;for(int a=0;a<3;++a){Real low=x[a]-h/2,high=x[a]+h/2;Real bl=c[a]-extent[a]/2,bh=c[a]+extent[a]/2;low=low>bl?low:bl;high=high<bh?high:bh;f*=bounded((high-low)/h);}if(dual){c[0]=cx2;c[1]=cy2;c[2]=z02+z-z0;Real f2=1;for(int a=0;a<3;++a){Real low=x[a]-h/2,high=x[a]+h/2;Real bl=c[a]-extent[a]/2,bh=c[a]+extent[a]/2;low=low>bl?low:bl;high=high<bh?high:bh;f2*=bounded((high-low)/h);}f+=f2;}return bounded(f);
    }

};
inline std::vector<Marker> cylinder_markers(Body b,Real spacing) {
    std::vector<Marker> ms;if(!b.enabled)return ms;
    if(b.shape==1){int nt=std::max(8,int(std::ceil(2*pi*b.radius/spacing))),nz=std::max(1,int(std::ceil(b.length/spacing)));for(int iz=0;iz<nz;++iz)for(int it=0;it<nt;++it){Real th=Real(2*pi)*(it+.5)/nt;Real zz=b.length*((iz+.5)/nz-.5);ms.push_back({{b.cx+b.radius*::cos(th),b.cy+b.radius*::sin(th),zz},{::cos(th),::sin(th),0},Real(2*pi)*b.radius*b.length/(nt*nz)});}int nr=std::max(1,int(std::ceil(b.radius/spacing)));for(int sign:{-1,1})for(int ir=0;ir<nr;++ir)for(int it=0;it<nt;++it){Real rr=b.radius*(ir+.5)/nr,th=Real(2*pi)*(it+.5)/nt;ms.push_back({{b.cx+rr*::cos(th),b.cy+rr*::sin(th),sign*b.length/2},{0,0,Real(sign)},Real(pi*b.radius*b.radius/(nr*nt))});}return ms;}
    int bodies=b.dual?2:1;
    for(int which=0;which<bodies;++which){ Body one=b; one.dual=0; if(which){one.cx=b.cx2;one.cy=b.cy2;one.z0=b.z02;}
    Real extent[3]={one.plate_height,one.length,one.plate_thickness};
    for(int axis=0;axis<3;++axis){int u=(axis+1)%3,v=(axis+2)%3;
        int nu=std::max(1,int(std::ceil(extent[u]/spacing))),nv=std::max(1,int(std::ceil(extent[v]/spacing)));
        for(int sign:{-1,1})for(int i=0;i<nu;++i)for(int j=0;j<nv;++j){
            Real x[3]={0,0,0},n[3]={0,0,0};x[axis]=sign*extent[axis]/2;n[axis]=Real(sign);
            x[u]=extent[u]*((i+Real(.5))/nu-Real(.5));x[v]=extent[v]*((j+Real(.5))/nv-Real(.5));
            ms.push_back({{one.cx+x[0],one.cy+x[1],x[2]},{n[0],n[1],n[2]},extent[u]*extent[v]/(nu*nv)});
        }
    }}return ms;
}
struct Stats {
    double phase_sum=0,mass=0,pmin=1e100,pmax=-1e100,phimin=1e100,phimax=-1e100,umax=0,nonfinite=0;
};
class Solver {
    static constexpr int partials=4096;
#ifndef LU_FORCE_PARTIALS
#define LU_FORCE_PARTIALS 65536
#endif
    static constexpr int force_partials=LU_FORCE_PARTIALS;
public:
    Grid g;Physics p;Body body;
    AleTrajectory ale_trajectory;
    Vec ale_translation{0,0,0}, ale_origin{0,0,0};
    Real ale_yaw_rate=0; int ale_enabled=0;
    double ale_time_scale=1.0;
    double ale_velocity_scale=1.0;
    Buffer<Real> state,base,work,faces,chem,gradients,coupling,boundary_flux;
    Buffer<Marker> markers;
    // 0..12 body state/force diagnostics; 13..17 projection solve;
    // 18..20 unprojected force polynomial; 21 body residual; 22..23 reserved;
    // 24 persistent projection failure flag.
    Buffer<double> reductions,body_state,force_sums;
    explicit Solver(Grid grid,Physics phys,Body b,const std::vector<Marker>& ms,const AleTrajectory& trajectory=AleTrajectory()):g(grid),p(phys),body(b),ale_trajectory(trajectory),
        state(5*g.n()),base(5*g.n()),work(5*g.n()),faces((p.cache_faces?27:p.transport_mode==3?5:4)*g.n()),chem(g.n()),gradients(15*g.n()),coupling(5*g.n()),boundary_flux(5*std::max({size_t(g.nx)*g.ny,size_t(g.nx)*g.nz,size_t(g.ny)*g.nz})),markers(std::max(size_t(1),ms.size())),
        reductions(partials*8),body_state(25),force_sums(4*force_partials+4*((force_partials+255)/256)+4),marker_count(ms.size()) {
        if(!p.cache_faces||p.transport_mode!=3||!g.stretched())throw std::runtime_error("Optimized metric solver requires cached FCT and metric mesh");
        if(b.enabled&&b.free&&b.inertia_mode==0)throw std::runtime_error("Free-body IB requires measured or implicit interior inertia");
        if(!ms.empty())markers.upload(ms);
        double initial_v=b.ramp_time>0?0:b.v0;
        std::vector<double> initial_body(25,0);initial_body[0]=initial_body[2]=b.z0;
        initial_body[1]=initial_body[3]=initial_body[11]=initial_v;body_state.upload(initial_body);
    }
    size_t marker_count;
    bool trace_stages=false;
    double elapsed_lattice=0;
    Field field_for(Real* ptr)const { Field f{ptr,g,p}; f.ale_translation=ale_translation;f.ale_origin=ale_origin;f.ale_yaw_rate=ale_yaw_rate;f.ale_enabled=ale_enabled; return f; }
    Field field()const{return field_for(state.data());}
    void update_ale(double t) {
        if(!ale_trajectory.enabled){ale_enabled=0;return;}
        AleFrameHost q=ale_trajectory.sample(t);
        // Interpolate in the world trajectory, then rotate the translational
        // velocity into the current local yaw frame.  z is unchanged.
        double c=std::cos(q.yaw),s=std::sin(q.yaw);
        double vs=ale_velocity_scale;
        ale_translation={Real((c*q.vx+s*q.vy)*vs),Real((-s*q.vx+c*q.vy)*vs),Real(q.vz*vs)};
        // Angular rate has units 1/s, so in lattice coordinates it is divided
        // by the time scale (theta is dimensionless while t_lattice=T*t_s).
        ale_origin={0,0,0}; ale_yaw_rate=Real(q.yaw_rate/ale_time_scale); ale_enabled=1;
    }
    void initialize(int mode=0) {
        Field f=field();Grid gr=g;Physics ph=p;Body b=body;size_t N=g.n();
        parallel_for(N,[=] HD(size_t s){
            int i,j,k;gr.xyz(s,i,j,k);Vec x=gr.position(i,j,k);
            Real phi=Real(.5)*(1-::tanh(2*(x.x-ph.level)/ph.xi));
            Real dist=b.signed_distance(x,b.z0);
            // Fictitious interior has the same equilibrium phase profile as exterior.
            if(mode==1)phi=Real(.5)+Real(.1)*::sin(Real(2*pi)*x.x/(gr.extent(0)));
            if(mode==2)phi=1;
            f.a[4*N+s]=phi;
            // Integral of the SAME diffuse free-surface profile, evaluated without cosh overflow.
            Real az=::fabs(2*(x.x-ph.level)/ph.xi),at=::fabs(2*(gr.extent(0)-ph.level)/ph.xi);
            Real logz=az+::log1p(::exp(-2*az)),logt=at+::log1p(::exp(-2*at));
            Real waterdepth=Real(.5)*(gr.extent(0)-x.x)-ph.xi/4*(logt-logz);
            f.a[s]=ph.gravity*(ph.rho0*(gr.extent(0)-x.x)+(ph.rho1-ph.rho0)*waterdepth);
            if(mode==2)f.a[s]=ph.rho1*ph.gravity*(gr.extent(0)-x.x);
            f.a[N+s]=f.a[2*N+s]=0;
            Real vz=b.enabled?Real(.5)*(1-::tanh(2*dist/ph.xi))*Real(b.prescribed_velocity(0)):0;
            f.a[3*N+s]=ph.rho(phi)*vz/3;
            if(mode==1)f.a[N+s]=ph.rho(phi)*Real(.02)/3;
        });
    }
    void fluid_predict(Real dt) {
        Field f=field();Field base_metric=f;base_metric.gradient_cache=nullptr;Grid gr=g;Physics ph=p;Real* mu=chem.data();Real* grad=gradients.data();Real* w=work.data();Real* flux=faces.data();size_t N=g.n();Real* boundary=boundary_flux.data();
        // Compute each metric primitive gradient once. The cached FP32 values
        // are shared by LBFS face reconstruction and capillary stress.
        parallel_for(N,[=] HD(size_t s){int i,j,k;gr.xyz(s,i,j,k);Real phi=base_metric.raw(4,s);
            mu[s]=2*ph.beta()*phi*(phi-1)*(2*phi-1)-ph.kappa()*laplacian(base_metric.a+4*N,gr,ph,i,j,k);
            for(int v=0;v<5;++v)for(int a=0;a<3;++a)grad[(size_t(v)*3+a)*N+s]=metric_gradient(base_metric,v,i,j,k,a,true);
        });
        f.gradient_cache=grad;
        // Coupling storage is unused until immersed_predict, which clears it.
        Real* scratch=coupling.data();Real* phaseflux=ph.transport_mode==3?flux+4*N:scratch;
        if(ph.transport_mode==3){
            // Unsplit Zalesak FCT (1979), Eqs. 6-13: aggregate all six faces
            // before limiting. Bounds require a bounded low-order update;
            // divergent volume velocities can violate this prerequisite.
            parallel_for(N,[=] HD(size_t s){w[4*N+s]=f.raw(4,s);});
            for(int a=0;a<3;++a){
                Real* cached=flux+size_t(9*a)*N;
                parallel_for(N,[=] HD(size_t s){int i,j,k;gr.xyz(s,i,j,k);Real out[5],low;
                    Vec uf;coupled_face(f,mu,i,j,k,a,out,nullptr,&low,&uf,true);

                        for(int v=0;v<5;++v)cached[size_t(v)*N+s]=out[v];
                        for(int v=0;v<3;++v)cached[size_t(v+5)*N+s]=uf[v];
                        cached[8*N+s]=low;
                    
                    scratch[size_t(a)*N+s]=out[4]-low;
                });
                Real* low_flux=cached+8*N;
                parallel_for(N,[=] HD(size_t s){int i,j,k;gr.xyz(s,i,j,k);int idx=a==0?i:a==1?j:k;
                    Real fm=idx||ph.periodic?scalar_at(low_flux,gr,ph,i-(a==0),j-(a==1),k-(a==2)):0;
                    w[4*N+s]-=dt*(low_flux[s]-fm)/gr.cell_width(a,i,j,k);
                });
            }
            parallel_for(N,[=] HD(size_t s){int i,j,k;gr.xyz(s,i,j,k);Real incoming=0,outgoing=0;
                for(int a=0;a<3;++a){int idx=a==0?i:a==1?j:k;
                    Real ap=dt/gr.cell_width(a,i,j,k)*scratch[size_t(a)*N+s];
                    Real am=idx||ph.periodic?dt/gr.cell_width(a,i,j,k)*scalar_at(scratch+size_t(a)*N,gr,ph,i-(a==0),j-(a==1),k-(a==2)):0;
                    incoming+=(ap<0?-ap:0)+(am>0?am:0);
                    outgoing+=(ap>0?ap:0)+(am<0?-am:0);
                }
                Real low=w[4*N+s];
                scratch[3*N+s]=incoming>0?bounded((1-low)/incoming):1;
                scratch[4*N+s]=outgoing>0?bounded(low/outgoing):1;
            });
        }
        parallel_for(N,[=] HD(size_t s){
            int i,j,k;gr.xyz(s,i,j,k);Real phi=f.raw(4,s),rho=ph.rho(phi);Real pressure_source=0;
            Vec shared_force=metric_capillary_vector(f,mu,i,j,k);
            for(int a=0;a<3;++a){
                Real up=f.primitive(a+1,i,j,k);
                Real rhop=ph.rho(f.shifted(4,i,j,k,a,1)),rhom=ph.rho(f.shifted(4,i,j,k,a,-1));
                int index=a==0?i:a==1?j:k;
                pressure_source+=up*metric_derivative(rhom,rho,rhop,gr.center(a,index)-gr.center(a,index-1),gr.center(a,index+1)-gr.center(a,index))/3;
                Real force=shared_force[a];
                w[size_t(a+1)*N+s]=f.raw(a+1,s)+dt*(force-(a==0?rho*ph.gravity:0))/3;
            }
            w[s]=f.raw(0,s)+(ph.pressure_product_rule?Real(0):dt*pressure_source);
            if(ph.transport_mode)w[4*N+s]=phi;
            else{
                Real adv=0;
                for(int a=0;a<3;++a){int im=i-(a==0),jm=j-(a==1),km=k-(a==2);
                    adv+=(phase_face(f,i,j,k,a)-phase_face(f,im,jm,km,a))/gr.h;
                }
                w[4*N+s]=phi+dt*(-adv+ph.mobility*laplacian(mu,gr,ph,i,j,k));
            }
        });
        for(int a=0;a<3;++a){
            Real* axis_flux=flux+size_t(9*a)*N;
            Real* axis_phase=ph.cache_faces?axis_flux+4*N:phaseflux;
            parallel_for(N,[=] HD(size_t s){int i,j,k;gr.xyz(s,i,j,k);Real out[5];

                    for(int v=0;v<5;++v)out[v]=axis_flux[size_t(v)*N+s];
                    Real anti=scratch[size_t(a)*N+s];int ri=i+(a==0),rj=j+(a==1),rk=k+(a==2);
                    Real donor=scratch[size_t(anti>=0?4:3)*N+s];
                    Real receiver=scalar_at(scratch+size_t(anti>=0?3:4)*N,gr,ph,ri,rj,rk);
                    out[4]=axis_flux[8*N+s]+(donor<receiver?donor:receiver)*anti;
                    Vec uf{axis_flux[5*N+s],axis_flux[6*N+s],axis_flux[7*N+s]};
                    consistent_momentum_flux(f,i,j,k,a,out,uf);axis_phase[s]=out[4];

                for(int v=0;v<4;++v)axis_flux[size_t(v)*N+s]=out[v];
            });
            size_t surface=N/gr.count(a);
            parallel_for(surface,[=] HD(size_t t){
                int i=a==0?-1:int(t%gr.nx),j=a==0?int(t%gr.ny):a==1?-1:int(t/gr.nx),k=a==2?-1:int(t/(a==0?gr.ny:gr.nx));
                Real neg[5];coupled_face(f,mu,i,j,k,a,neg,scratch);
                for(int v=0;v<5;++v)boundary[size_t(v)*surface+t]=neg[v];
            });
            parallel_for(N,[=] HD(size_t s){int i,j,k;gr.xyz(s,i,j,k);int index=a==0?i:a==1?j:k;
                int im=i-(a==0),jm=j-(a==1),km=k-(a==2);
                size_t boundary_id=a==0?size_t(k)*gr.ny+j:a==1?size_t(k)*gr.nx+i:size_t(j)*gr.nx+i;
                for(int v=0;v<4;++v){Real fn=index?axis_flux[size_t(v)*N+gr.id(im,jm,km)]:boundary[size_t(v)*surface+boundary_id];
                    Real fp=axis_flux[size_t(v)*N+s];
                    if(v==0&&ph.pressure_product_rule){
                        // Lu (8)-(10): -div(rho*u)/3 + u.grad(rho)/3.
                        // Define BOTH terms with the same reconstructed face velocity:
                        // S_p=sum_faces (rho_face-rho_cell)*u_face.n/(3*h).
                        // Their difference is exactly -rho_cell*div(u_face)/3.
                        // This is an explicit, separately validated discretization choice.
                        Real rho=ph.rho(f.raw(4,s));
                        Real rp=metric_face_scalar(rho,ph.rho(f.shifted(4,i,j,k,a,1)),gr,a,index);
                        Real rm=metric_face_scalar(ph.rho(f.shifted(4,i,j,k,a,-1)),rho,gr,a,index-1);
                        fp*=rho/rp;fn*=rho/rm;
                    }
                    w[size_t(v)*N+s]-=dt*(fp-fn)/gr.cell_width(a,i,j,k);
                }
                if(ph.transport_mode){Real fn=index?axis_phase[gr.id(im,jm,km)]:boundary[4*surface+boundary_id];
                    w[4*N+s]-=dt*(axis_phase[s]-fn)/gr.cell_width(a,i,j,k);
                }
            });
        }
    }
    void project_ib_phase(Real dt) {
        size_t N=g.n();Real* c=coupling.data();Real* response=faces.data();Real* w=work.data();
        double* bs=body_state.data();double* red=reductions.data();
        // Euclidean projection on box bounds and the prescribed total IB flux:
        // minimize sum (phi_new-phi_raw)^2, subject to bounds and unchanged sum.
        // KKT conditions give phi_new=clip(phi_raw+lambda,lower,upper).
        // Only IB-support cells participate. Existing fluid-step overshoots are
        // retained in the bounds, so this operator cannot hide fluid failures.
        parallel_for(partials,[=] HD(size_t t){double lo=1e100,hi=-1e100,capacity_low=0,capacity_high=0;
            for(size_t s=t;s<N;s+=partials)if(c[s]>0){
                double phi=w[4*N+s],raw=phi+double(dt)*(response[3*N+s]+response[N+s]*bs[11]);
                double lower=phi<0?phi:0,upper=phi>1?phi:1;
                lo=lower-raw<lo?lower-raw:lo;hi=upper-raw>hi?upper-raw:hi;
                capacity_low+=raw-lower;capacity_high+=upper-raw;
            }
            red[t]=lo;red[partials+t]=hi;red[2*partials+t]=capacity_low;red[3*partials+t]=capacity_high;
        });
        parallel_for(1,[=] HD(size_t){double lo=1e100,hi=-1e100,low=0,high=0;
            for(int t=0;t<partials;++t){lo=red[t]<lo?red[t]:lo;hi=red[partials+t]>hi?red[partials+t]:hi;low+=red[2*partials+t];high+=red[3*partials+t];}
            bs[13]=0;bs[14]=lo;bs[15]=hi;bs[16]=0;bs[17]=0;
            if(low < -1e-10||high < -1e-10){bs[24]=1;bs[17]=1;}
        });
        for(int iteration=0;iteration<32;++iteration){
            parallel_for(partials,[=] HD(size_t t){if(bs[17])return;double residual=0,active=0;
                for(size_t s=t;s<N;s+=partials)if(c[s]>0){
                    double phi=w[4*N+s],raw=phi+double(dt)*(response[3*N+s]+response[N+s]*bs[11]);
                    double lower=phi<0?phi:0,upper=phi>1?phi:1,value=raw+bs[13];
                    if(value>lower&&value<upper)active+=1;
                    value=value<lower?lower:value>upper?upper:value;residual+=value-raw;
                }
                red[t]=residual;red[partials+t]=active;
            });
            parallel_for(1,[=] HD(size_t){if(bs[17])return;double residual=0,active=0;
                for(int t=0;t<partials;++t){residual+=red[t];active+=red[partials+t];}
                bs[16]=residual;if(::fabs(residual)<1e-10){bs[17]=1;return;}
                if(residual>0)bs[15]=bs[13];else bs[14]=bs[13];
                double next=active>0?bs[13]-residual/active:.5*(bs[14]+bs[15]);
                if(next<=bs[14]||next>=bs[15])next=.5*(bs[14]+bs[15]);
                bs[13]=next;
            });
        }
        parallel_for(1,[=] HD(size_t){if(!bs[17]&&::fabs(bs[16])>1e-8)bs[24]=2;});
        parallel_for(N,[=] HD(size_t s){
            c[3*N+s]=response[2*N+s]+response[s]*Real(bs[11]);
            if(c[s]>0){double phi=w[4*N+s],raw=phi+double(dt)*(response[3*N+s]+response[N+s]*bs[11]);
                double lower=phi<0?phi:0,upper=phi>1?phi:1,value=raw+bs[13];
                value=value<lower?lower:value>upper?upper:value;c[4*N+s]=Real((value-phi)/dt);
            }else c[4*N+s]=0;
        });
    }
    void bounded_body_target(Real dt,bool release_body) {
        Grid gr=g;Physics ph=p;Body b=body;size_t N=g.n();Real* c=coupling.data();Real* response=faces.data();
        Real* mu=chem.data();double* red=reductions.data();double* bs=body_state.data();Field old=field(),pred=field_for(work.data());
        parallel_for(N,[=] HD(size_t s){response[2*N+s]=c[3*N+s];response[3*N+s]=c[4*N+s];});
        int iterations=b.free&&release_body?6:1;
        for(int iteration=0;iteration<iterations;++iteration){
            project_ib_phase(dt);
            if(!(b.free&&release_body))break;
            parallel_for(partials,[=] HD(size_t t){double force=0;
                for(size_t s=t;s<N;s+=partials){
                    int i,j,k;gr.xyz(s,i,j,k);Vec x=gr.position(i,j,k);Real w0=0,w1=0;
                    if(b.signed_distance(x,Real(bs[0]))<b.radius+gr.h+::fabs(Real(dt*bs[1]))){
                        w0=b.volume_fraction(x,Real(bs[0]),gr.h);w1=b.volume_fraction(x,Real(bs[0]+dt*bs[1]),gr.h);
                    }
                    Real r0=ph.rho(pred.raw(4,s)),r1=ph.rho(pred.raw(4,s)+dt*c[4*N+s]);
                    Real corrected=r1*(pred.raw(3,s)/r0+c[3*N+s]/3);
                    double dv=gr.volume(i,j,k);
                    force+=3*((w1-1)*double(corrected)+pred.raw(3,s)-double(w0)*old.raw(3,s))*dv/dt;
                    if(w0>0)force-=w0*(capillary_force(old,mu,i,j,k,2)-ph.rho(old.raw(4,s))*Real(0))*dv;
                }
                red[t]=force;
            });
            parallel_for(1,[=] HD(size_t){double force=0;for(int t=0;t<partials;++t)force+=red[t];
                double mass=double(b.density)*b.volume();double residual=mass*(bs[11]-bs[1])/dt-force+mass*Real(0);
                double derivative=mass/dt-bs[19]-2*bs[20]*bs[11];
                bs[21]=residual;if(::fabs(derivative)>1e-30)bs[11]-=residual/derivative;
            });
        }
        // Materialize the phase and velocity source at the final target velocity.
        if(b.free&&release_body)project_ib_phase(dt);
    }
    void implicit_body_target(Real dt,bool release_body) {
        Grid gr=g;Physics ph=p;Body b=body;size_t N=g.n();
        Real* c=coupling.data();Real* response=faces.data();Real* mu=chem.data();
        double* red=reductions.data();double* bs=body_state.data();Field old=field(),pred=field_for(work.data());
        // The discrete IB map is affine in target velocity; linear rho(phi)
        // makes its momentum correction quadratic. Integrate the three
        // coefficients once and solve the 1DOF body equation on the device.
        parallel_for(partials,[=] HD(size_t t){double f0=0,f1=0,f2=0;
            for(size_t s=t;s<N;s+=partials){
                int i,j,k;gr.xyz(s,i,j,k);Vec x=gr.position(i,j,k);
                Real w0=0,w1=0;
                if(b.signed_distance(x,Real(bs[0]))<b.radius+gr.h+::fabs(Real(dt*bs[1]))) {
                    w0=b.volume_fraction(x,Real(bs[0]),gr.h);w1=b.volume_fraction(x,Real(bs[0]+dt*bs[1]),gr.h);
                }
                double rp=ph.rho(pred.raw(4,s)),m=3*double(pred.raw(3,s));
                double ra=rp+double(ph.rho1-ph.rho0)*dt*c[4*N+s];
                double rb=double(ph.rho1-ph.rho0)*dt*response[N+s];
                double va=m/rp+c[3*N+s],vb=response[s];
                double a0=ra*va,a1=ra*vb+rb*va,a2=rb*vb,dv=double(gr.h)*gr.h*gr.h;
                f0+=((w1-1)*a0+m-3*double(w0)*old.raw(3,s))*dv/dt;
                f1+=(w1-1)*a1*dv/dt;f2+=(w1-1)*a2*dv/dt;
                if(w0>0)f0-=w0*(capillary_force(old,mu,i,j,k,2)-ph.rho(old.raw(4,s))*Real(0))*dv;
            }
            red[t]=f0;red[partials+t]=f1;red[2*partials+t]=f2;
        });
        parallel_for(1,[=] HD(size_t){double f0=0,f1=0,f2=0;
            for(int t=0;t<partials;++t){f0+=red[t];f1+=red[partials+t];f2+=red[2*partials+t];}
            double mass=double(b.density)*b.volume(),target=bs[1];
            if(b.free&&release_body){
                for(int it=0;it<12;++it){
                    double residual=mass*(target-bs[1])/dt-(f0+f1*target+f2*target*target-mass*Real(0));
                    double derivative=mass/dt-f1-2*f2*target;
                    if(::fabs(derivative)<1e-30){atomic_plus(bs+6,1.);break;}
                    target-=residual/derivative;
                }
            }
            bs[11]=target;bs[18]=f0;bs[19]=f1;bs[20]=f2;
        });
        if(ph.ib_phase_bound)bounded_body_target(dt,release_body);
        else parallel_for(N,[=] HD(size_t s){c[3*N+s]+=response[s]*Real(bs[11]);c[4*N+s]+=response[N+s]*Real(bs[11]);});
    }
    void immersed_predict(Real dt,bool release_body=true) {
        Grid gr=g;Physics ph=p;size_t N=g.n();Real* c=coupling.data();Marker* ms=markers.data();double* bs=body_state.data();
        Field pred=field_for(work.data());Body b=body;
        coupling.zero();
        if(!b.enabled)return;
        Real* response=faces.data();
        if(b.inertia_mode==2)parallel_for(2*N,[=] HD(size_t s){response[s]=0;});
        // Eq. (22) row sum: occupancy S(D); gather I(occupancy). O(64*markers).
        parallel_for(marker_count,[=] HD(size_t l){Marker m=ms[l];m.x.z+=Real(bs[0]);int i0,j0,k0;support_base(gr,m,i0,j0,k0);
            if(i0<0||j0<0||k0<0||i0+3>=gr.nx||j0+3>=gr.ny||k0+3>=gr.nz){atomic_plus(bs+6,1.);return;}
            for(int k=k0;k<k0+4;++k)for(int j=j0;j<j0+4;++j)for(int i=i0;i<i0+4;++i){
                Real w=support_weight(gr,m,i,j,k);atomic_plus(c+gr.id(i,j,k),w*m.area/(gr.h*gr.h*gr.h));
            }
        });
        parallel_for(marker_count,[=] HD(size_t l){Marker m=ms[l];m.x.z+=Real(bs[0]);int i0,j0,k0;support_base(gr,m,i0,j0,k0);
            if(i0<0||j0<0||k0<0||i0+3>=gr.nx||j0+3>=gr.ny||k0+3>=gr.nz)return;
            Real denom=0;Vec u{0,0,0};
            for(int k=k0;k<k0+4;++k)for(int j=j0;j<j0+4;++j)for(int i=i0;i<i0+4;++i){
                size_t s=gr.id(i,j,k);Real w=support_weight(gr,m,i,j,k),fac=3*w/ph.rho(pred.raw(4,s));
                denom+=w*c[s];u.x+=fac*pred.raw(1,s);u.y+=fac*pred.raw(2,s);u.z+=fac*pred.raw(3,s);
            }
            if(!(denom>0)){atomic_plus(bs+6,1.);return;}
            Real fac=m.area/(denom*gr.h*gr.h*gr.h);u.x=-u.x*fac;u.y=-u.y*fac;u.z=((b.inertia_mode==2?Real(0):Real(bs[1]))-u.z)*fac;
            for(int k=k0;k<k0+4;++k)for(int j=j0;j<j0+4;++j)for(int i=i0;i<i0+4;++i){
                size_t s=gr.id(i,j,k);Real w=support_weight(gr,m,i,j,k);
                atomic_plus(c+N+s,w*u.x);atomic_plus(c+2*N+s,w*u.y);atomic_plus(c+3*N+s,w*u.z);
                if(b.inertia_mode==2)atomic_plus(response+s,w*fac);
            }
        });
        // Reconstructed normal surface flux: Yan (2024) Eqs. 31,40,42, minus chemical term as in Lu.
        // This is an explicitly documented interpretation of Lu's incomplete Eq. (23), NOT author-confirmed.
        parallel_for(marker_count,[=] HD(size_t l){Marker m=ms[l];m.x.z+=Real(bs[0]);int i0,j0,k0;support_base(gr,m,i0,j0,k0);
            if(i0<0||j0<0||k0<0||i0+3>=gr.nx||j0+3>=gr.ny||k0+3>=gr.nz)return;
            Real q=0,qresponse=0;
            for(int k=k0;k<k0+4;++k)for(int j=j0;j<j0+4;++j)for(int i=i0;i<i0+4;++i){
                size_t s=gr.id(i,j,k);q+=2*pred.raw(4,s)*support_weight(gr,m,i,j,k)*(c[N+s]*m.n.x+c[2*N+s]*m.n.y+c[3*N+s]*m.n.z);
                if(b.inertia_mode==2)qresponse+=2*pred.raw(4,s)*support_weight(gr,m,i,j,k)*response[s]*m.n.z;
            }
            q*=-ph.ib_phase_scale*m.area/(gr.h*gr.h*gr.h);
            qresponse*=-ph.ib_phase_scale*m.area/(gr.h*gr.h*gr.h);
            for(int k=k0;k<k0+4;++k)for(int j=j0;j<j0+4;++j)for(int i=i0;i<i0+4;++i){
                atomic_plus(c+4*N+gr.id(i,j,k),q*support_weight(gr,m,i,j,k));
                if(b.inertia_mode==2)atomic_plus(response+N+gr.id(i,j,k),qresponse*support_weight(gr,m,i,j,k));
            }
        });
        if(b.inertia_mode==2)implicit_body_target(dt,release_body);
        double* red=force_sums.data();double* scratch_sum=red+4*force_partials;double* totals=scratch_sum+4*((force_partials+255)/256);Field old=field();Real* mu=chem.data();
        parallel_for(force_partials,[=] HD(size_t t){double sum=0,internal_rate=0,internal_mass=0,internal_source=0;
            for(size_t s=t;s<N;s+=force_partials){
                int vi,vj,vk;gr.xyz(s,vi,vj,vk);
                Real r0=ph.rho(pred.raw(4,s)),r1=ph.rho(pred.raw(4,s)+dt*c[4*N+s]);
                Real momentum=pred.raw(3,s);
                Real corrected=r1*(momentum/r0+c[3*N+s]/3);
                // Reaction to the ENTIRE IB substep, including density remapping.
                // This is an impulse balance, not yet the complete hydrodynamic
                // force: the interior fictitious-fluid contribution is separate.
                sum-=3*double(corrected-momentum)*gr.volume(vi,vj,vk)/dt;
                if(b.inertia_mode){
                    int i,j,k;gr.xyz(s,i,j,k);Vec x=gr.position(i,j,k);
                    // Cheap support rejection before distance/square-root evaluation.
                    if(b.signed_distance(x,Real(bs[0]))<b.radius+gr.h+::fabs(Real(dt*bs[1]))){
                        Real w0=b.volume_fraction(x,Real(bs[0]),gr.h);
                        Real w1=b.volume_fraction(x,Real(bs[0]+dt*bs[1]),gr.h);
                        double dv=gr.volume(i,j,k);
                        internal_rate+=3*(double(w1)*corrected-double(w0)*old.raw(3,s))*dv/dt;
                        Real oldrho=ph.rho(old.raw(4,s));internal_mass+=w0*oldrho*dv;
                        internal_source+=w0*(capillary_force(old,mu,i,j,k,2)-oldrho*Real(0))*dv;
                    }
                }
            }
            red[t]=sum;red[force_partials+t]=internal_rate;red[2*force_partials+t]=internal_mass;red[3*force_partials+t]=internal_source;
        });
        reduce_force_sums(red,force_partials,scratch_sum,totals);
        parallel_for(1,[=] HD(size_t){double force=totals[0],rate=totals[1],mass=totals[2],source=totals[3];
            bs[4]=force;bs[7]=rate;bs[8]=mass;bs[9]=source;bs[10]=force+rate-source;
            // Unvalidated rigid fictitious-gas approximation. Actual interior mass
            // and momentum are measured separately in physics_audit.
            // Moving-control-volume balance: remove fictitious interior momentum
            // and its non-IB volume forces. Finite slip/CH boundary mass flux is
            // still a separately measured consistency error, not assumed exact.
            bs[5]=b.inertia_mode==2?(bs[11]-bs[1])/dt:b.inertia_mode?bs[10]/(b.density*b.volume())-Real(0):
                force/((b.density-ph.rho0)*b.volume())-Real(0);
            bs[12]=b.inertia_mode==2?(b.density*b.volume())*(bs[5]+Real(0))-bs[10]:0;
        });
        parallel_for(N,[=] HD(size_t s){
            Real oldphi=pred.raw(4,s),newphi=oldphi+dt*c[4*N+s];
            Real r0=ph.rho(oldphi),r1=ph.rho(newphi);
            for(int a=1;a<=3;++a)pred.a[size_t(a)*N+s]=r1*(pred.raw(a,s)/r0+c[size_t(a)*N+s]/3);
            pred.a[4*N+s]=newphi;
        });
    }
    void step(Real dt) {
        base.copy_from(state);double* bs=body_state.data();Body b=body;
        bool startup=!b.free; // prescribed analytic motion at every RK stage
        double time=elapsed_lattice;
        parallel_for(1,[=] HD(size_t){bs[2]=bs[0];bs[3]=bs[1];});
        for(int stage=0;stage<3;++stage){
            if(startup){double stage_time=time+(stage==0?0:stage==1?double(dt):.5*dt);
                update_ale(stage_time/ale_time_scale);
                parallel_for(1,[=] HD(size_t){bs[0]=b.prescribed_position(stage_time);bs[1]=b.prescribed_velocity(stage_time);});
            } else update_ale(time/ale_time_scale);
            fluid_predict(dt);
            if(trace_stages){auto st=stats(true);std::cerr<<"stage_probe,"<<time<<','<<stage<<",fluid,"<<st.phimin<<','<<st.phimax<<','<<st.umax<<','<<st.phase_sum<<'\n';}
            immersed_predict(dt,!startup);
            if(trace_stages){auto st=stats(true);std::cerr<<"stage_probe,"<<time<<','<<stage<<",ib,"<<st.phimin<<','<<st.phimax<<','<<st.umax<<','<<st.phase_sum<<'\n';}
            Real a=stage==0?0:stage==1?Real(.75):Real(1./3),bb=1-a;
            Real* u=state.data();Real* u0=base.data();Real* w=work.data();size_t NN=5*g.n();
            parallel_for(NN,[=] HD(size_t s){u[s]=u0[s]+bb*(w[s]-u0[s]);});
            double ba=stage==0?0.:stage==1?.75:1./3,bc=1-ba;
            if(!startup)parallel_for(1,[=] HD(size_t){double z=bs[0],v=bs[1];
                bs[0]=bs[2]+bc*(z+dt*v-bs[2]);
                bs[1]=bs[3]+bc*(v+(b.free?dt*bs[5]:0)-bs[3]);
            });
        }
        elapsed_lattice+=double(dt);
        if(startup){double end_time=elapsed_lattice;
            parallel_for(1,[=] HD(size_t){bs[0]=b.prescribed_position(end_time);bs[1]=b.prescribed_velocity(end_time);});
        }
    }
    Stats stats(bool predicted=false) {
        Field f=predicted?Field{work.data(),g,p}:field();size_t N=g.n();double* r=reductions.data();
        parallel_for(partials,[=] HD(size_t t){
            double a=0,b=0,pmi=1e100,pma=-1e100,fmi=1e100,fma=-1e100,um=0,nan=0;
            for(size_t s=t;s<N;s+=partials){
                double phi=f.raw(4,s),rho=f.p.rho(Real(phi)),pr=f.raw(0,s);
                double u=3*::sqrt(sq(double(f.raw(1,s)))+sq(double(f.raw(2,s)))+sq(double(f.raw(3,s))))/rho;
                if(!(phi==phi&&pr==pr&&u==u)||::fabs(phi)>1e30||::fabs(pr)>1e30||u>1e30){nan+=1;continue;}
                int i,j,k;f.g.xyz(s,i,j,k);double volume=f.g.volume(i,j,k);a+=phi*volume;b+=rho*volume;pmi=pr<pmi?pr:pmi;pma=pr>pma?pr:pma;fmi=phi<fmi?phi:fmi;fma=phi>fma?phi:fma;um=u>um?u:um;
            }
            r[t]=a;r[partials+t]=b;r[2*partials+t]=pmi;r[3*partials+t]=pma;r[4*partials+t]=fmi;r[5*partials+t]=fma;r[6*partials+t]=um;r[7*partials+t]=nan;
        });
        auto v=reductions.download();Stats st;
        for(int t=0;t<partials;++t){st.phase_sum+=v[t];st.mass+=v[partials+t];st.pmin=std::min(st.pmin,v[2*partials+t]);st.pmax=std::max(st.pmax,v[3*partials+t]);st.phimin=std::min(st.phimin,v[4*partials+t]);st.phimax=std::max(st.phimax,v[5*partials+t]);st.umax=std::max(st.umax,v[6*partials+t]);st.nonfinite+=v[7*partials+t];}
        return st;
    }
    std::vector<double> peak_probe() {
        // Diagnostic only: distinguish physical exterior speeds from fictitious
        // interior speeds. Reuse the reduction buffer; no state is modified.
        Field f=field();Grid gr=g;Body b=body;size_t N=g.n();double* r=reductions.data();double* bs=body_state.data();
        parallel_for(partials,[=] HD(size_t t){double peak=0,index=-1,phi=0,pressure=0,inside=0,outside=0;
            for(size_t s=t;s<N;s+=partials){
                double rho=f.p.rho(f.raw(4,s));double speed=3*::sqrt(sq(double(f.raw(1,s)))+sq(double(f.raw(2,s)))+sq(double(f.raw(3,s))))/::fabs(rho);
                int i,j,k;gr.xyz(s,i,j,k);bool solid=b.enabled&&b.signed_distance(gr.position(i,j,k),Real(bs[0]))<0;
                if(solid){inside=speed>inside?speed:inside;}else outside=speed>outside?speed:outside;
                if(speed>peak){peak=speed;index=double(s);phi=f.raw(4,s);pressure=f.raw(0,s);}
            }
            r[t]=peak;r[partials+t]=index;r[2*partials+t]=phi;r[3*partials+t]=pressure;r[4*partials+t]=inside;r[5*partials+t]=outside;
        });
        auto data=reductions.download();std::vector<double> result(6,0);result[1]=-1;
        for(int t=0;t<partials;++t){if(data[t]>result[0])for(int a=0;a<4;++a)result[a]=data[a*partials+t];
            result[4]=std::max(result[4],data[4*partials+t]);result[5]=std::max(result[5],data[5*partials+t]);}
        return result;
    }
    void write_slice(const std::string& path,double length_scale,double pressure_scale) {
        // Download only a plane, not the full 3D solution.
        Buffer<Real> plane(size_t(g.nx)*g.nz*2);Real* out=plane.data();Field f=field();Grid gr=g;
        parallel_for(size_t(g.nx)*g.nz,[=] HD(size_t s){int i=int(s%gr.nx),k=int(s/gr.nx);size_t n=size_t(gr.nx)*gr.nz;
            out[s]=f.raw(4,gr.id(i,gr.ny/2,k));out[n+s]=f.raw(0,gr.id(i,gr.ny/2,k));});
        auto v=plane.download();std::ofstream o(path);o<<"x_m,z_m,phi,p_Pa\n"<<std::setprecision(9);
        for(int k=0;k<g.nz;++k)for(int i=0;i<g.nx;++i){size_t s=size_t(k)*g.nx+i;
            o<<g.center(0,i)/length_scale<<','<<g.center(2,k)/length_scale<<','<<v[s]<<','<<v[size_t(g.nx)*g.nz+s]/pressure_scale<<'\n';}
    }
    // Compact full-field checkpoint.  Each frame stores five channels
    // (pressure, phase, ux, uy, uz) as uint16 followed by five float
    // min/max pairs.  Scaling is per frame, so quantisation is explicit and
    // reversible to the reported half-bin error.
    void write_uint16_frame(const std::string& path,double pressure_scale,double velocity_scale) {
        auto v=state.download(); size_t N=g.n(); std::vector<float> lo(5,1e30f),hi(5,-1e30f);
        std::vector<float> q(5*N);
        for(size_t s=0;s<N;++s){Real phi=v[4*N+s],rho=p.rho(phi);float x[5]={float(v[s]/pressure_scale),float(phi),float(3*v[N+s]/rho*velocity_scale),float(3*v[2*N+s]/rho*velocity_scale),float(3*v[3*N+s]/rho*velocity_scale)};
            for(int c=0;c<5;++c){q[size_t(c)*N+s]=x[c];lo[c]=std::min(lo[c],x[c]);hi[c]=std::max(hi[c],x[c]);}}
        std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("cannot open uint16 output: "+path);
        for(int c=0;c<5;++c){out.write(reinterpret_cast<char*>(&lo[c]),sizeof(float));out.write(reinterpret_cast<char*>(&hi[c]),sizeof(float));}
        std::vector<uint16_t> packed(5*N);
        for(int c=0;c<5;++c){double d=double(hi[c])-lo[c];for(size_t s=0;s<N;++s){double z=d>0?(q[size_t(c)*N+s]-lo[c])/d:0;z=std::max(0.,std::min(1.,z));packed[size_t(c)*N+s]=uint16_t(std::llround(z*65535.));}}
        out.write(reinterpret_cast<const char*>(packed.data()),std::streamsize(packed.size()*sizeof(uint16_t)));
    }
};
