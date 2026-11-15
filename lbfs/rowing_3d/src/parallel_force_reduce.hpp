#pragma once
// Four independent double sums, two reduction levels, no floating atomics.
// n <= 65536 ensures the second level fits into a single 256-thread block.
#ifdef __CUDACC__
__global__ void force_sum_kernel(const double* input,double* output,int n){
 __shared__ double shared[4][256];int lane=threadIdx.x,id=blockIdx.x*256+lane;
 #pragma unroll
 for(int c=0;c<4;++c)shared[c][lane]=id<n?input[size_t(c)*n+id]:0;
 __syncthreads();
 for(int stride=128;stride;stride/=2){if(lane<stride){
  #pragma unroll
  for(int c=0;c<4;++c)shared[c][lane]+=shared[c][lane+stride];
 }__syncthreads();}
 if(lane==0){
  #pragma unroll
  for(int c=0;c<4;++c)output[size_t(c)*gridDim.x+blockIdx.x]=shared[c][0];
 }
}
inline void reduce_force_sums(double* input,int n,double* scratch,double* output){
 int blocks=(n+255)/256;force_sum_kernel<<<blocks,256>>>(input,scratch,n);cuda_check(cudaGetLastError());
 force_sum_kernel<<<1,256>>>(scratch,output,blocks);cuda_check(cudaGetLastError());
}
#else
inline void reduce_force_sums(double* input,int n,double*,double* output){
 for(int c=0;c<4;++c){double sum=0;for(int i=0;i<n;++i)sum+=input[size_t(c)*n+i];output[c]=sum;}
}
#endif
