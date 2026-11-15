#undef NDEBUG
#include "solver.hpp"
#include "metric_mesh_host.hpp"
#include <cassert>
int main(){
    std::array<std::vector<Real>,3> axes;
    for(int i=0;i<=64;++i)axes[0].push_back(i*.25f);
    for(int i=0;i<=16;++i)axes[1].push_back(Real(i));
    for(int i=0;i<=32;++i)axes[2].push_back(i*.5f);
    MetricMesh full(axes,.25);Physics ph{};ph.cache_faces=1;ph.transport_mode=3;ph.rho0=ph.rho1=2;ph.nu0=ph.nu1=.1;ph.sigma=.1;ph.xi=2;ph.smagorinsky_cs=.16;
    Body body{};PolicyAssemblyTrajectory tr;tr.enabled=true;PolicyFrameHost a,b;b.t=1;
    for(int j=0;j<3;++j){a.part[j].px=a.part[j].py=a.part[j].pz=8;b.part[j]=a.part[j];}tr.frames={a,b};
    std::vector<Marker> ms={{{.05f,0,0},{-.8f,.6f,0},2,0}};
    auto fill=[](Solver& s){Grid g=s.g;std::vector<Real> u(5*g.n());
        for(size_t q=0;q<g.n();++q){int i,j,k;g.xyz(q,i,j,k);auto x=g.position(i,j,k);
            u[q]=100-.3f*x.x+.1f*x.y;u[g.n()+q]=2*.02f*x.y/3;u[2*g.n()+q]=2*.01f*x.x/3;u[4*g.n()+q]=.5f;}
        s.state.upload(u);
    };
    Solver ref(full.grid(),ph,body,ms,{},tr);fill(ref);auto expected=ref.measure_surface_force(.5);
    std::vector<double> total(73,0);
    for(int rank=0;rank<2;++rank){int low=rank==0?0:28,high=rank==0?36:64;
        auto local=axes;local[0]={axes[0].begin()+low,axes[0].begin()+high+1};MetricMesh mesh(local,.25);
        Solver s(mesh.grid(),ph,body,ms,{},tr);s.ib_partitioned=true;s.core_i0=rank==0?0:4;s.core_i1=rank==0?32:36;
        s.core_j0=s.core_k0=0;s.core_j1=16;s.core_k1=32;fill(s);
        if(rank==1){bool refused=false;try{s.measure_surface_force(.5);}catch(const std::runtime_error&){refused=true;}assert(refused);}
        s.configure_surface_probe_grid(axes);auto before=s.state.download();auto got=s.measure_surface_force(.5);auto after=s.state.download();assert(before==after);
        for(int i=0;i<73;++i)total[i]+=got[i];
    }
    for(int i=0;i<73;++i){double tol=2e-4+2e-6*std::abs(expected[i]);if(std::abs(expected[i]-total[i])>tol)throw std::runtime_error("routed stress/torque differs at component "+std::to_string(i));}
    std::cout<<"PASS anisotropic cross-rank probes: old failure, all stress components and torque, unchanged state\n";
}


