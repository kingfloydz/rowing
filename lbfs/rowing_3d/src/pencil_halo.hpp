#pragma once

// Included after MultiGpuRank. Core indices are LOCAL; halo_lo is the
// GLOBAL origin of the allocation. Never subtract that origin from a core.
// Buffers are allocated once. Each directed face needs one peer transfer,
// including all five variables, instead of thousands of row-sized transfers.
class PencilHalo {
    struct Link {
        int source, destination, axis;
        Grid src_grid, dst_grid;
        int src_start, dst_start, width, nx, ny, nz;
        std::unique_ptr<Buffer<Real>> send, receive;
        size_t cells() const {return size_t(nx)*ny*nz;}
    };
    std::vector<Link> links;
public:
    PencilHalo(const std::vector<MultiGpuRank>& ranks,int halo) {
        if(ranks.size()!=8)throw std::runtime_error("PencilHalo requires 4x2 ranks");
        auto add=[&](const MultiGpuRank& src,const MultiGpuRank& dst,int axis,bool high){
            Grid gs=src.solver->g,gd=dst.solver->g;
            if(src.core_hi[axis]-src.core_lo[axis]<halo)
                throw std::runtime_error("pencil core is narrower than halo");
            int ss=high?src.core_hi[axis]-halo:src.core_lo[axis];
            int ds=high?dst.core_lo[axis]-halo:dst.core_hi[axis];
            if(ss<0||ds<0||ss+halo>gs.count(axis)||ds+halo>gd.count(axis))
                throw std::runtime_error("pencil halo region exceeds allocation");
            for(int a=0;a<3;++a)if(a!=axis&&(gs.count(a)!=gd.count(a)||gs.offset(a)!=gd.offset(a)))
                throw std::runtime_error("pencil transverse extents do not match");
            Link l{src.device,dst.device,axis,gs,gd,ss,ds,halo,
                   axis==0?halo:gs.nx,axis==1?halo:gs.ny,gs.nz,nullptr,nullptr};
            cuda_check(cudaSetDevice(src.device));l.send=std::make_unique<Buffer<Real>>(5*l.cells());
            cuda_check(cudaSetDevice(dst.device));l.receive=std::make_unique<Buffer<Real>>(5*l.cells());
            links.push_back(std::move(l));
        };
        for(const auto& r:ranks){
            if(r.px<3){const auto& q=ranks[size_t(r.py)*4+r.px+1];add(r,q,0,true);add(q,r,0,false);}
            if(r.py<1){const auto& q=ranks[size_t(r.py+1)*4+r.px];add(r,q,1,true);add(q,r,1,false);}
        }
    }
    PencilHalo(const PencilHalo&)=delete;
    PencilHalo& operator=(const PencilHalo&)=delete;
    ~PencilHalo(){for(auto& l:links){cudaSetDevice(l.source);l.send.reset();cudaSetDevice(l.destination);l.receive.reset();}}

    void exchange(const std::vector<MultiGpuRank>& ranks,bool predicted) {
        // The driver synchronizes producers immediately before this exchange.
        // Keep the per-axis barriers below: y packing consumes x corner halos.
        for(int axis=0;axis<2;++axis){
            for(auto& l:links)if(l.axis==axis){
                cuda_check(cudaSetDevice(l.source));
                const auto& s=*ranks[l.source].solver;
                const Real* src=(predicted?s.work:s.state).data();Real* packed=l.send->data();
                Grid g=l.src_grid;size_t n=l.cells();int nx=l.nx,ny=l.ny,start=l.src_start;
                parallel_for(5*n,[=] HD(size_t q){size_t v=q/n,t=q%n;
                    int i=int(t%nx),j=int((t/nx)%ny),k=int(t/(size_t(nx)*ny));
                    if(axis==0)i+=start;else j+=start;
                    packed[q]=src[v*g.n()+g.id(i,j,k)];
                });
            }
            multi_gpu_sync_all(ranks);
            for(auto& l:links)if(l.axis==axis){
                cuda_check(cudaSetDevice(l.destination));
                cuda_check(cudaMemcpyPeerAsync(l.receive->data(),l.destination,l.send->data(),l.source,5*l.cells()*sizeof(Real),0));
            }
            for(auto& l:links)if(l.axis==axis){
                cuda_check(cudaSetDevice(l.destination));
                auto& s=*ranks[l.destination].solver;
                Real* dst=(predicted?s.work:s.state).data();const Real* packed=l.receive->data();
                Grid g=l.dst_grid;size_t n=l.cells();int nx=l.nx,ny=l.ny,start=l.dst_start;
                parallel_for(5*n,[=] HD(size_t q){size_t v=q/n,t=q%n;
                    int i=int(t%nx),j=int((t/nx)%ny),k=int(t/(size_t(nx)*ny));
                    if(axis==0)i+=start;else j+=start;
                    dst[v*g.n()+g.id(i,j,k)]=packed[q];
                });
            }
            // Complete x before packing y: the y faces carry updated corner
            // halos, including the diagonal rank's data needed by IB/WENO.
            multi_gpu_sync_all(ranks);
        }
    }
};
