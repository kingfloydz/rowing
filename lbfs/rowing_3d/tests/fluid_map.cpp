#include "solver.hpp"
#include "metric_mesh_host.hpp"
int main(int argc,char** argv){
 if(argc!=2)return 2;
 std::ofstream out(argv[1],std::ios::binary);
 for(int metric=0;metric<2;++metric)for(int open=0;open<2;++open)for(int mode=0;mode<4;++mode){
  std::array<std::vector<Real>,3> axes;
  for(int a=0;a<3;++a){axes[a].push_back(0);for(int n=0;n<18;++n)axes[a].push_back(axes[a].back()+Real(1+.02*a*n));}
  MetricMesh mesh(axes,1);Grid g=metric?mesh.grid():Grid{18,18,18,1};
  Physics p{};p.rho0=.001;p.rho1=1;p.nu0=.01;p.nu1=.001;p.sigma=.0001;p.xi=2;p.mobility=.01;p.gravity=.00001;
  p.vertical_axis=2;p.periodic=!open;p.horizontal_open_boundary=open;p.top_pressure_outlet=open;
  p.pressure_product_rule=p.macro_face_density=p.cache_faces=p.capillary_mode=p.gravity_reconstruction=p.harmonic_acoustic_pressure=1;
  p.transport_mode=3;p.momentum_advection=2;p.mass_transport_bound=.8;p.transport_cfl_limit=.9;p.smagorinsky_cs=.16;
  Body b{};Solver sol(g,p,b,{});size_t N=g.n();std::vector<Real> values(5*N);
  for(size_t s=0;s<N;++s){int i,j,k;g.xyz(s,i,j,k);
   Real phi=mode==0?0:mode==1?1:k<6?0:k>11?1:Real(k-5)/7;
   if(mode==3&&i==8&&j==8)phi=Real(.000001);
   values[4*N+s]=phi;values[s]=Real(1e-5)*std::sin(Real(i+j+k));
   for(int a=0;a<3;++a)values[size_t(a+1)*N+s]=p.rho(phi)*Real(.001)*std::sin(Real(i+2*j+3*k+a))/3;
  }
  sol.state.upload(values);sol.ale_enabled=1;sol.ale_translation={.0002,-.0001,0};sol.ale_yaw_rate=.00001;
  sol.fluid_predict(Real(.01));sol.enforce_transport_guard();auto actual=sol.work.download();
  for(Real v:actual)if(!std::isfinite(v))throw std::runtime_error("nonfinite test state");
  out.write(reinterpret_cast<const char*>(actual.data()),std::streamsize(actual.size()*sizeof(Real)));
 }
 std::cout<<"PASS 16 fluid maps: pure/mixed phases, uniform/metric, periodic/open, ALE\n";
}
