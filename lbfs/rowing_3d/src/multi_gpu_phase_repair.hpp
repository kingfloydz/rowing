#pragma once
#include "tensor_poisson_gpu.hpp"
// Experimental conservative bounded repair. Unlike local clipping, every
// correction is represented by a common global face flux used for momentum.
// Unit-weight Neumann flux is a documented extension, not Lu's published model.
struct MultiGpuPhaseRepair {
 Grid global;size_t N,plane;int partials;TensorPoissonGpu poisson;Buffer<Real> raw;Buffer<double> mapped,reduce;
 Buffer<Real> original_momentum;
 Buffer<double> iterate_a,iterate_b;
 double seconds=0;size_t calls=0,total_iterations=0;int max_iterations=0;
 MultiGpuPhaseRepair(const MetricMesh& mesh,std::vector<MultiGpuRank>& ranks):global(mesh.grid()),N(global.n()),plane(size_t(global.nx)*global.ny),partials(N>=1000000?32768:1024),poisson(mesh.axes),raw(N),mapped(N),reduce(4*partials),original_momentum(3*N),iterate_a(3*N),iterate_b(3*N){
  cuda_check(cudaSetDevice(0));
 }
 ~MultiGpuPhaseRepair(){cudaSetDevice(0);}
 void apply(std::vector<MultiGpuRank>& ranks){
  auto start=std::chrono::steady_clock::now();multi_gpu_sync_all(ranks);cuda_check(cudaSetDevice(0));
  for(auto& rank:ranks){auto& s=*rank.solver;cuda_check(cudaMemcpyPeer(raw.data()+size_t(rank.core0)*plane,0,s.work.data()+4*s.g.n()+size_t(rank.core0-rank.halo0)*plane,rank.device,size_t(rank.core1-rank.core0)*plane*sizeof(Real)));}
  Real* phi=raw.data();double* c=mapped.data();double* red=reduce.data();size_t count=N;int nx=global.nx,ny=global.ny,P=partials;
  const double *hx=poisson.width[0]->data(),*hy=poisson.width[1]->data(),*hz=poisson.width[2]->data();
  parallel_for(count,[=] HD(size_t q){c[q]=phi[q]<0?0:phi[q]>1?1:double(phi[q]);});
  double volume=0,defect=0;bool converged=false;double rg=ranks.front().solver->p.rho0,dr=ranks.front().solver->p.rho1-rg;
  for(int it=0;it<32;++it){
   parallel_for(P,[=] HD(size_t t){double delta=0,den=0,vsum=0,bad=0;
    for(size_t q=t;q<count;q+=P){double v=hx[q%nx]*hy[(q/nx)%ny]*hz[q/(size_t(nx)*ny)];delta+=(double(phi[q])-c[q])*v;den+=4*c[q]*(1-c[q])*v;vsum+=v;if(!(::fabs(double(phi[q]))<1e30)||!(rg+dr*phi[q]>0))bad+=1;}
    red[t]=delta;red[P+t]=den;red[2*P+t]=vsum;red[3*P+t]=bad;
   });auto sums=reduce.download();defect=0;double den=0,bad=0;volume=0;for(int t=0;t<P;++t){defect+=sums[t];den+=sums[P+t];volume+=sums[2*P+t];bad+=sums[3*P+t];}
   if(bad)throw std::runtime_error("phase repair input nonfinite");
   if(std::abs(defect)<=1e-12*volume){converged=true;break;}
   if(!(den>0))throw std::runtime_error("phase repair has no admissible interface capacity");double alpha=defect/den;
   parallel_for(count,[=] HD(size_t q){double v=c[q]+alpha*4*c[q]*(1-c[q]);c[q]=v<0?0:v>1?1:v;});
  }
  if(!converged)throw std::runtime_error("phase repair volume mapping did not converge");
  double* potential=poisson.value.data();double compatibility=defect/volume;
  parallel_for(count,[=] HD(size_t q){potential[q]=double(phi[q])-c[q]-compatibility;});poisson.solve();
  // Backward-Euler donor-cell transport of the *integrated correction mass
  // flux*. The density balance is rho_new + outgoing = rho_old + incoming.
  // Consequently the velocity equation is an M-matrix system and each exact
  // solution component stays in the old global velocity bounds. A central
  // explicit correction has no such property at density ratio 1000.
  for(auto& rank:ranks){auto& s=*rank.solver;size_t n=s.g.n();for(int a=0;a<3;++a)
   cuda_check(cudaMemcpyPeer(original_momentum.data()+size_t(a)*N+size_t(rank.core0)*plane,0,s.work.data()+size_t(a+1)*n+size_t(rank.core0-rank.halo0)*plane,rank.device,size_t(rank.core1-rank.core0)*plane*sizeof(Real)));}
  Real* initial=original_momentum.data();double* u=iterate_a.data();double* next=iterate_b.data();int nz=global.nz;
  parallel_for(count,[=] HD(size_t q){double rho=rg+dr*phi[q];for(int a=0;a<3;++a)u[size_t(a)*count+q]=double(initial[size_t(a)*count+q])/rho;});
  bool solved=false;int iterations=0;
  for(;iterations<1024;++iterations){
   parallel_for(count,[=] HD(size_t q){int index[3]={int(q%nx),int((q/nx)%ny),int(q/(size_t(nx)*ny))},extent[3]={nx,ny,nz};size_t stride[3]={1,size_t(nx),size_t(nx)*ny};const double* widths[3]={hx,hy,hz};double rhs[3]={initial[q],initial[count+q],initial[2*count+q]},diagonal=rg+dr*c[q];
    for(int a=0;a<3;++a)for(int sign=-1;sign<=1;sign+=2){int j=index[a]+sign;if(j<0||j>=extent[a])continue;size_t nb=sign>0?q+stride[a]:q-stride[a];double h=widths[a][index[a]],distance=.5*(h+widths[a][j]);double flux=dr*(potential[q]-potential[nb])/(distance*h);
     if(flux>=0)diagonal+=flux;else for(int v=0;v<3;++v)rhs[v]-=flux*u[size_t(v)*count+nb];}
    for(int v=0;v<3;++v)next[size_t(v)*count+q]=rhs[v]/diagonal;
   });
   if(iterations%8==7){parallel_for(P,[=] HD(size_t t){double error=0;for(size_t q=t;q<3*count;q+=P){double e=::fabs(next[q]-u[q]);error=e>error?e:error;}red[t]=error;});auto errors=reduce.download();double error=0;for(int t=0;t<P;++t)error=std::max(error,errors[t]);if(error<1e-12)solved=true;}
   std::swap(u,next);if(solved){++iterations;break;}
  }
  if(!solved)throw std::runtime_error("phase repair implicit mass transport did not converge");
  total_iterations+=iterations;max_iterations=std::max(max_iterations,iterations);
  parallel_for(count,[=] HD(size_t q){double rho=rg+dr*c[q];for(int a=0;a<3;++a)initial[size_t(a)*count+q]=Real(rho*u[size_t(a)*count+q]);phi[q]=Real(c[q]);});synchronize();
  for(auto& rank:ranks){cuda_check(cudaSetDevice(rank.device));auto& s=*rank.solver;size_t localN=s.g.n();
   cuda_check(cudaMemcpyPeer(s.work.data()+4*localN,rank.device,phi+size_t(rank.halo0)*plane,0,localN*sizeof(Real)));
   for(int a=0;a<3;++a)cuda_check(cudaMemcpyPeer(s.work.data()+size_t(a+1)*localN,rank.device,initial+size_t(a)*N+size_t(rank.halo0)*plane,0,localN*sizeof(Real)));
  }multi_gpu_sync_all(ranks);multi_gpu_exchange(ranks,true);cuda_check(cudaSetDevice(0));
  seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();++calls;
 }
};
