#pragma once
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

// Minimize squared modeled layer costs, subject to core >= physical halo.
// compute_padding defaults to the physical halo for whole-resident updates;
// cropped fluid kernels use a separately measured compute padding.
// The incremental cost is 2*resident+1, so allocating each remaining layer
// to the smallest resident count is optimal for this cell-count objective.
// This is NOT dynamic per-step/IB balancing. Complexity O(nz*nr), O(nr)
// storage, instead of the previous O(nz^2*nr) dynamic program.
inline std::vector<std::pair<int,int>> multi_gpu_balanced_slabs(int nz,int nr,int halo,int compute_padding=-1) {
    if(nr<2||halo<1||nz<nr*halo||compute_padding< -1||compute_padding>halo)
        throw std::runtime_error("each GPU core must contain at least halo layers");
    int padding=compute_padding<0?halo:compute_padding;
    std::vector<int> core(nr,halo),resident(nr);
    for(int r=0;r<nr;++r)resident[r]=core[r]+(r>0?padding:0)+(r+1<nr?padding:0);
    for(int left=nz-nr*halo;left>0;--left){
        auto r=size_t(std::min_element(resident.begin(),resident.end())-resident.begin());
        ++core[r];++resident[r];
    }
    std::vector<std::pair<int,int>> result;
    int z=0;
    for(int n:core){result.emplace_back(z,z+n);z+=n;}
    return result;
}
