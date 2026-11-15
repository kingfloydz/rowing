#undef NDEBUG
#include "solver.hpp"
#include <cassert>
#include <cstring>
void close(double x,double y,double tol=2e-4){if(std::abs(x-y)>tol){std::cerr<<x<<" != "<<y<<"\n";std::abort();}}
int main(){
    Grid g{32,32,32,1};Physics p{};p.rho0=p.rho1=2;p.nu0=p.nu1=.1;p.xi=2;p.sigma=0;p.cache_faces=1;p.transport_mode=3;
    Body body{};PolicyAssemblyTrajectory tr;tr.enabled=true;PolicyFrameHost a,b;a.t=0;b.t=1;
    for(int i=0;i<3;++i){a.part[i].px=a.part[i].py=a.part[i].pz=16;b.part[i]=a.part[i];}tr.frames={a,b};
    std::vector<Marker> markers;
    for(int axis=0;axis<3;++axis)for(int sign:{-1,1}){Vec x{0,0,0},n{0,0,0};if(axis==0){x.x=2*sign;n.x=sign;}if(axis==1){x.y=2*sign;n.y=sign;}if(axis==2){x.z=2*sign;n.z=sign;}markers.push_back({x,n,16,0});}
    Solver solver(g,p,body,markers,{},tr);std::vector<Real> state(5*g.n(),0);
    for(size_t q=0;q<g.n();++q){int i,j,k;g.xyz(q,i,j,k);Vec x=g.position(i,j,k);state[q]=100-2*.3*x.z;state[4*g.n()+q]=1;state[g.n()+q]=2*.04*x.y/3;}
    solver.state.upload(state);auto baseline=solver.state.download();auto force=solver.measure_surface_force(.5);
    close(force[0],0);close(force[1],0);close(force[2],2*.3*64);for(int i=3;i<6;++i)close(force[i],0);
    auto after=solver.state.download();assert(std::memcmp(baseline.data(),after.data(),state.size()*sizeof(Real))==0);
    auto shifted=markers;for(auto& m:shifted)m.x.x+=1;solver.markers.upload(shifted);solver.surface_force_time=-1;
    auto offset=solver.measure_surface_force(.5);close(offset[2],2*.3*64);close(offset[4],-2*.3*64);
    solver.markers.upload(markers);solver.surface_force_time=-1;
    auto probe=[&](Vec normal){Buffer<SurfaceTraction> result(1);auto* ptr=result.data();Field field=solver.field();
        parallel_for(1,[=] HD(size_t){ptr[0]=surface_cell_traction(field,15,15,15,normal);});return result.download()[0];};
    auto point=probe({0,1,0});close(point.viscous.x,2*.1*.04,1e-6);
    // Constant diffuse phase: capillary stress is isotropic f_bulk, mu=0.
    p.sigma=.1;solver.p=p;for(size_t q=0;q<g.n();++q)state[4*g.n()+q]=.5;solver.state.upload(state);
    point=probe({1,0,0});close(point.capillary.x,p.beta()*.0625,1e-6);
    // Partition ownership: replicated surface quadrature is counted once.
    solver.p.sigma=0;solver.state.upload(baseline);solver.ib_partitioned=true;solver.core_i0=0;solver.core_i1=16;solver.core_j0=solver.core_k0=0;solver.core_j1=solver.core_k1=32;solver.surface_force_time=-1;
    auto left=solver.measure_surface_force(.5);solver.core_i0=16;solver.core_i1=32;solver.surface_force_time=-1;auto right=solver.measure_surface_force(.5);
    for(int i=0;i<18;++i)close(left[i]+right[i],force[i]);
    // End-time force vectors must be rotated to world basis, not mixed with
    // world-frame solid poses while retaining an ALE-frame force.
    solver.ib_partitioned=false;solver.ale_trajectory.enabled=true;solver.ale_trajectory.anchor_x=solver.ale_trajectory.anchor_y=16;
    AleFrameHost af,bf;af.cx=af.cy=16;af.yaw=pi/2;bf=af;bf.t=1;solver.ale_trajectory.frames={af,bf};
    for(size_t q=0;q<g.n();++q){int i,j,k;g.xyz(q,i,j,k);state[q]=100-2*.3*g.position(i,j,k).x;state[4*g.n()+q]=1;state[g.n()+q]=0;}
    solver.state.upload(state);solver.surface_force_time=-1;auto world=solver.measure_surface_force(.5);close(world[0],0);close(world[1],2*.3*64);close(world[2],0);
    // Invalid exterior probe must be rejected, not silently omitted.
    SurfaceTraction t;assert(!surface_exterior_traction(solver.field(),{{1,1,1},{-1,0,0},1,0},t));
    std::cout<<"PASS buoyancy, pressure sign, zero torque, Couette shear, capillary gauge, rank ownership, ALE/world rotation, read-only state, invalid probes\n";
}
