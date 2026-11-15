#pragma once
// Diagnostic build only. Enqueue events around individual kernels without
// synchronizing every launch. Never use its instrumented wall time as a speedup.
#include <map>
#include <fstream>
#include <iomanip>
struct KernelTiming {
 struct Sample {int device;const char* name;cudaEvent_t begin,end;};
 std::vector<Sample> samples;std::map<int,size_t> counts;
 size_t begin(const char* name){
  int device;cuda_check(cudaGetDevice(&device));
  if(counts[device]>=512)return size_t(-1);
  Sample s{device,name,nullptr,nullptr};
  cuda_check(cudaEventCreate(&s.begin));cuda_check(cudaEventCreate(&s.end));
  cuda_check(cudaEventRecord(s.begin));samples.push_back(s);++counts[device];return samples.size()-1;
 }
 void end(size_t index){if(index!=size_t(-1))cuda_check(cudaEventRecord(samples[index].end));}
 void write(const std::string& path){
  struct Summary {size_t calls=0;double total_ms=0,max_ms=0;};
  std::map<std::pair<int,std::string>,Summary> sums;
  int original;cuda_check(cudaGetDevice(&original));
  for(auto& s:samples){cuda_check(cudaSetDevice(s.device));cuda_check(cudaEventSynchronize(s.end));
   float ms=0;cuda_check(cudaEventElapsedTime(&ms,s.begin,s.end));auto& sum=sums[{s.device,s.name}];
   ++sum.calls;sum.total_ms+=ms;sum.max_ms=std::max(sum.max_ms,double(ms));
   cuda_check(cudaEventDestroy(s.begin));cuda_check(cudaEventDestroy(s.end));
  }
  samples.clear();cuda_check(cudaSetDevice(original));
  std::ofstream f(path);if(!f)throw std::runtime_error("cannot write kernel timings");
  f<<"device\tcalls\ttotal_ms\tmax_ms\tkernel\n"<<std::setprecision(12);
  for(const auto& entry:sums)f<<entry.first.first<<'\t'<<entry.second.calls<<'\t'<<entry.second.total_ms<<'\t'<<entry.second.max_ms<<'\t'<<entry.first.second<<'\n';
 }
};
inline KernelTiming& kernel_timing(){static thread_local KernelTiming timing;return timing;}
