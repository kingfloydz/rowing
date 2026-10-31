#pragma once
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HD __host__ __device__
inline void cuda_check(cudaError_t e) {
    if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<class F> __global__ void parallel_kernel(size_t n, F f) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) f(i);
}
#ifndef LU_GPU_BLOCK_SIZE
#define LU_GPU_BLOCK_SIZE 64
#endif
template<class F> void parallel_for(size_t n,F f) {
    parallel_kernel<<<unsigned((n+LU_GPU_BLOCK_SIZE-1)/LU_GPU_BLOCK_SIZE),LU_GPU_BLOCK_SIZE>>>(n,f);
    cuda_check(cudaGetLastError());
}
inline void synchronize() { cuda_check(cudaDeviceSynchronize()); }
constexpr const char* backend_name="cuda";
#else
#define HD
template<class F> void parallel_for(size_t n,F f) {
    #pragma omp parallel for if(n>4096)
    for(long long i=0;i<static_cast<long long>(n);++i) f(size_t(i));
}
inline void synchronize() {}
constexpr const char* backend_name="cpu";
#endif
template<class T> HD void atomic_plus(T* p,T v) {
#ifdef __CUDA_ARCH__
    atomicAdd(p,v);
#else
    #pragma omp atomic update
    *p+=v;
#endif
}
inline size_t& allocated_bytes() { static size_t n=0;return n; }
inline size_t& peak_bytes() { static size_t n=0;return n; }
template<class T> class Buffer {
    T* p_=nullptr; size_t n_=0;
public:
    explicit Buffer(size_t n):n_(n) {
#ifdef __CUDACC__
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&p_),n*sizeof(T)));
#else
        p_=new T[n];
#endif
        allocated_bytes()+=n*sizeof(T);peak_bytes()=std::max(peak_bytes(),allocated_bytes());
        zero();
    }
    ~Buffer() {
#ifdef __CUDACC__
        cudaFree(p_);
#else
        delete[] p_;
#endif
        allocated_bytes()-=n_*sizeof(T);
    }
    Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
    T* data() const {return p_;} size_t size()const{return n_;}
    void zero() {
#ifdef __CUDACC__
        cuda_check(cudaMemset(p_,0,n_*sizeof(T)));
#else
        std::fill(p_,p_+n_,T{});
#endif
    }
    void copy_from(const Buffer& b) {
        if(b.n_!=n_)throw std::runtime_error("buffer shape mismatch");
#ifdef __CUDACC__
        cuda_check(cudaMemcpy(p_,b.p_,n_*sizeof(T),cudaMemcpyDeviceToDevice));
#else
        std::copy(b.p_,b.p_+n_,p_);
#endif
    }
    void upload(const std::vector<T>& v) {
        if(v.size()!=n_)throw std::runtime_error("upload shape mismatch");
#ifdef __CUDACC__
        cuda_check(cudaMemcpy(p_,v.data(),n_*sizeof(T),cudaMemcpyHostToDevice));
#else
        std::copy(v.begin(),v.end(),p_);
#endif
    }
    std::vector<T> download()const {
        std::vector<T> v(n_);
#ifdef __CUDACC__
        cuda_check(cudaMemcpy(v.data(),p_,n_*sizeof(T),cudaMemcpyDeviceToHost));
#else
        std::copy(p_,p_+n_,v.begin());
#endif
        return v;
    }
};
struct DeviceTimer {
#ifdef __CUDACC__
    cudaEvent_t a,b;
    DeviceTimer(){cuda_check(cudaEventCreate(&a));cuda_check(cudaEventCreate(&b));}
    ~DeviceTimer(){cudaEventDestroy(a);cudaEventDestroy(b);}
    void start(){cuda_check(cudaEventRecord(a));}
    double stop(){cuda_check(cudaEventRecord(b));cuda_check(cudaEventSynchronize(b));float ms;cuda_check(cudaEventElapsedTime(&ms,a,b));return ms/1000.;}
#else
    std::chrono::steady_clock::time_point a;
    void start(){a=std::chrono::steady_clock::now();}
    double stop(){return std::chrono::duration<double>(std::chrono::steady_clock::now()-a).count();}
#endif
};
