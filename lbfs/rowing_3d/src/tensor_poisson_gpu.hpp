#pragma once
// Research backend: exact tensor-product finite-volume Neumann Poisson solve.
// Requires -lcublas -lcusolver. No iterative full-domain sweeps.
#include <cublas_v2.h>
#include <cusolverDn.h>
inline void blas_check(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("tensor cublas failure "+std::to_string(int(s)));}
inline void eigen_check(cusolverStatus_t s){if(s!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error("tensor cusolver failure "+std::to_string(int(s)));}
struct TensorPoissonGpu {
 int nx,ny,nz;size_t N;cublasHandle_t blas{};
 std::array<std::unique_ptr<Buffer<double>>,3> basis,eigen,width;
 Buffer<double> value,temp;
 TensorPoissonGpu(const std::array<std::vector<Real>,3>& axes):nx(int(axes[0].size())-1),ny(int(axes[1].size())-1),nz(int(axes[2].size())-1),N(size_t(nx)*ny*nz),value(N),temp(N){
  blas_check(cublasCreate(&blas));cusolverDnHandle_t solver;eigen_check(cusolverDnCreate(&solver));
  for(int a=0;a<3;++a){int n=int(axes[a].size())-1;std::vector<double> h(n),matrix(size_t(n)*n,0);for(int i=0;i<n;++i)h[i]=double(axes[a][i+1])-axes[a][i];
   for(int i=0;i<n-1;++i){double c=2/(h[i]+h[i+1]);matrix[size_t(i)*n+i]+=c/h[i];matrix[size_t(i+1)*n+i+1]+=c/h[i+1];matrix[size_t(i+1)*n+i]=matrix[size_t(i)*n+i+1]=-c/std::sqrt(h[i]*h[i+1]);}
   basis[a]=std::make_unique<Buffer<double>>(size_t(n)*n);basis[a]->upload(matrix);eigen[a]=std::make_unique<Buffer<double>>(n);width[a]=std::make_unique<Buffer<double>>(n);width[a]->upload(h);
   int size=0;eigen_check(cusolverDnDsyevd_bufferSize(solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,basis[a]->data(),n,eigen[a]->data(),&size));Buffer<double> scratch(size);Buffer<int> info(1);
   eigen_check(cusolverDnDsyevd(solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,basis[a]->data(),n,eigen[a]->data(),scratch.data(),size,info.data()));if(info.download()[0])throw std::runtime_error("tensor eigen decomposition failed");cuda_check(cudaMemset(eigen[a]->data(),0,sizeof(double)));
  }eigen_check(cusolverDnDestroy(solver));
 }
 ~TensorPoissonGpu(){cublasDestroy(blas);}
 void solve(){
  // Input and output use physical cell scalars. Operator: -div(grad(q)).
  double *v=value.data(),*t=temp.data();const double *hx=width[0]->data(),*hy=width[1]->data(),*hz=width[2]->data();int x=nx,y=ny;size_t count=N;
  parallel_for(count,[=] HD(size_t i){v[i]*=::sqrt(hx[i%x]*hy[(i/x)%y]*hz[i/(size_t(x)*y)]);});
  double one=1,zero=0;
  auto transform=[&](int a,bool inverse){const double* Q=basis[a]->data();
   if(a==0)blas_check(cublasDgemm(blas,inverse?CUBLAS_OP_N:CUBLAS_OP_T,CUBLAS_OP_N,nx,ny*nz,nx,&one,Q,nx,v,nx,&zero,t,nx));
   if(a==1)blas_check(cublasDgemmStridedBatched(blas,CUBLAS_OP_N,inverse?CUBLAS_OP_T:CUBLAS_OP_N,nx,ny,ny,&one,v,nx,size_t(nx)*ny,Q,ny,0,&zero,t,nx,size_t(nx)*ny,nz));
   if(a==2)blas_check(cublasDgemm(blas,CUBLAS_OP_N,inverse?CUBLAS_OP_T:CUBLAS_OP_N,nx*ny,nz,nz,&one,v,nx*ny,Q,nz,&zero,t,nx*ny));
   std::swap(v,t);
  };
  for(int a=0;a<3;++a)transform(a,false);
  const double *ex=eigen[0]->data(),*ey=eigen[1]->data(),*ez=eigen[2]->data();
  parallel_for(count,[=] HD(size_t i){v[i]=i?v[i]/(ex[i%x]+ey[(i/x)%y]+ez[i/(size_t(x)*y)]):0;});
  for(int a=2;a>=0;--a)transform(a,true);
  if(v!=value.data())throw std::runtime_error("tensor buffer permutation");
  parallel_for(count,[=] HD(size_t i){v[i]/=::sqrt(hx[i%x]*hy[(i/x)%y]*hz[i/(size_t(x)*y)]);});
 }
};
