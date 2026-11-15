#pragma once

// Single-process multi-GPU z-slab driver.  Each rank owns a local metric mesh
// with twelve stencil cells of halo on either side.  The state and predicted work
// fields are exchanged with cudaMemcpyPeer after every RK stage.  Policy
// assemblies use the same halo path while their local markers are transformed
// from prescribed hull and oar 6DOF poses at every RK stage.
#include "solver.hpp"
#include "step_time.hpp"
#include "slab_partition.hpp"
#include <map>
#include "phase_audit.hpp"

#ifdef __CUDACC__
struct MultiGpuRank {
    int device=0;
    int core0=0,core1=0,halo0=0,halo1=0;
    std::unique_ptr<MetricMesh> mesh;
    std::unique_ptr<Solver> solver;
    int core_lo[3]={0,0,0},core_hi[3]={0,0,0};
    int halo_lo[3]={0,0,0},halo_hi[3]={0,0,0};
    int px=0,py=0;
};

#include "checkpoint.hpp"

inline void multi_gpu_sync_all(const std::vector<MultiGpuRank>& ranks) {
    for(const auto& r:ranks){cuda_check(cudaSetDevice(r.device));try{synchronize();}catch(const std::exception& e){throw std::runtime_error(std::string("device synchronize ")+std::to_string(r.device)+": "+e.what());}}
}

inline void multi_gpu_exchange(const std::vector<MultiGpuRank>& ranks,bool predicted) {
    if(ranks.size()<2)return;
    const int nx=ranks.front().solver->g.nx,ny=ranks.front().solver->g.ny;
    const size_t plane=size_t(nx)*ny, halo=12, bytes=halo*plane*sizeof(Real);
    for(size_t q=1;q<ranks.size();++q){
        const auto& left=ranks[q-1];const auto& right=ranks[q];
        const Buffer<Real>& src_upper=predicted?left.solver->work:left.solver->state;
        Buffer<Real>& dst_lower=predicted?right.solver->work:right.solver->state;
        size_t src_layer=size_t(left.core1-left.halo0)-halo;
        size_t dst_layer=0;
        const size_t src_n=left.solver->g.n(),dst_n=right.solver->g.n();
        cuda_check(cudaSetDevice(right.device));
        for(int v=0;v<5;++v){
            const Real* src=src_upper.data()+size_t(v)*src_n+src_layer*plane;
            Real* dst=dst_lower.data()+size_t(v)*dst_n+dst_layer*plane;
            try{cuda_check(cudaMemcpyPeerAsync(dst,right.device,src,left.device,bytes,0));}
            catch(const std::exception& e){throw std::runtime_error(std::string("lower halo peer copy ")+std::to_string(left.device)+"->"+std::to_string(right.device)+" v="+std::to_string(v)+": "+e.what());}
        }
        cuda_check(cudaSetDevice(left.device));
        const Buffer<Real>& src_lower=right.solver->work;
        Buffer<Real>& dst_upper=left.solver->work;
        if(!predicted){
            // Rebind the references for the state exchange.  Keeping the
            // copies separate avoids packing a temporary buffer per rank.
            const Buffer<Real>& s=right.solver->state;Buffer<Real>& d=left.solver->state;
            size_t slayer=size_t(right.core0-right.halo0),dlayer=size_t(left.core1-left.halo0);
            for(int v=0;v<5;++v){
                const Real* p=s.data()+size_t(v)*dst_n+slayer*plane;
                Real* q=d.data()+size_t(v)*src_n+dlayer*plane;
                try{cuda_check(cudaMemcpyPeerAsync(q,left.device,p,right.device,bytes,0));}
                catch(const std::exception& e){throw std::runtime_error(std::string("upper halo peer copy ")+std::to_string(right.device)+"->"+std::to_string(left.device)+" v="+std::to_string(v)+": "+e.what());}
            }
        } else {
            size_t slayer=size_t(right.core0-right.halo0),dlayer=size_t(left.core1-left.halo0);
            for(int v=0;v<5;++v){
                const Real* p=src_lower.data()+size_t(v)*dst_n+slayer*plane;
                Real* q=dst_upper.data()+size_t(v)*src_n+dlayer*plane;
                try{cuda_check(cudaMemcpyPeerAsync(q,left.device,p,right.device,bytes,0));}
                catch(const std::exception& e){throw std::runtime_error(std::string("upper work peer copy ")+std::to_string(right.device)+"->"+std::to_string(left.device)+" v="+std::to_string(v)+": "+e.what());}
            }
        }
    }
    multi_gpu_sync_all(ranks);
}

#ifdef LU_EXPERIMENTAL_PHASE_REPAIR
#include "multi_gpu_phase_repair.hpp"
#endif

inline std::array<std::vector<Real>,3> multi_gpu_local_axes(const MetricMesh& global_mesh,int k0,int k1) {
    std::array<std::vector<Real>,3> a=global_mesh.axes;
    a[2]=std::vector<Real>(global_mesh.axes[2].begin()+k0,global_mesh.axes[2].begin()+k1+1);
    return a;
}

inline std::array<std::vector<Real>,3> multi_gpu_local_axes_xy(const MetricMesh& global_mesh,const int lo[3],const int hi[3]) {
    std::array<std::vector<Real>,3> a;
    for(int axis=0;axis<3;++axis)a[axis]=std::vector<Real>(global_mesh.axes[axis].begin()+lo[axis],global_mesh.axes[axis].begin()+hi[axis]+1);
    return a;
}

inline std::vector<std::pair<int,int>> multi_gpu_split_axis(int n,int parts) {
    std::vector<std::pair<int,int>> out;int base=n/parts,rem=n%parts,at=0;
    for(int p=0;p<parts;++p){int q=base+(p<rem);out.emplace_back(at,at+q);at+=q;}return out;
}

#include "pencil_halo.hpp"
#include "column_phase_repair.hpp"

// Write one global, halo-free frame.  Each rank contributes its core z-range
// at a fixed file offset, so the file layout is identical to a single-GPU
// frame: ten float32 min/max values followed by five channel-major uint16
// arrays over the global nx*ny*nz core domain.
inline void multi_gpu_write_global_uint16(const std::string& path,
                                          const std::vector<MultiGpuRank>& ranks,
                                          int nx,int ny,int nz,double pressure_scale,
                                          double velocity_scale) {
    const size_t global_n=size_t(nx)*ny*nz;
    std::array<float,5> lo{1e30f,1e30f,1e30f,1e30f,1e30f};
    std::array<float,5> hi{-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
    std::vector<std::vector<Real>> host(ranks.size());
    for(size_t q=0;q<ranks.size();++q){
        const auto& r=ranks[q];cuda_check(cudaSetDevice(r.device));host[q]=r.solver->state.download();
        size_t n=r.solver->g.n();int lk0=r.core_lo[2],lk1=r.core_hi[2];
        for(int k=lk0;k<lk1;++k)for(int j=r.core_lo[1];j<r.core_hi[1];++j)for(int i=r.core_lo[0];i<r.core_hi[0];++i){
            size_t s=r.solver->g.id(i,j,k);Real phi=host[q][4*n+s],rho=r.solver->p.rho(phi);
            float x[5]={float(host[q][s]/pressure_scale),float(phi),float(3*host[q][n+s]/rho*velocity_scale),float(3*host[q][2*n+s]/rho*velocity_scale),float(3*host[q][3*n+s]/rho*velocity_scale)};
            for(int c=0;c<5;++c){lo[c]=std::min(lo[c],x[c]);hi[c]=std::max(hi[c],x[c]);}
        }
    }
    std::fstream out(path,std::ios::binary|std::ios::in|std::ios::out|std::ios::trunc);
    if(!out)throw std::runtime_error("cannot open global uint16 output: "+path);
    for(int c=0;c<5;++c){out.write(reinterpret_cast<const char*>(&lo[c]),sizeof(float));out.write(reinterpret_cast<const char*>(&hi[c]),sizeof(float));}
    const std::streamoff header=std::streamoff(5*2*sizeof(float));
    for(size_t q=0;q<ranks.size();++q){
        const auto& r=ranks[q];size_t n=r.solver->g.n();int lk0=r.core_lo[2],lk1=r.core_hi[2];size_t ni=r.core_hi[0]-r.core_lo[0],nj=r.core_hi[1]-r.core_lo[1],core_n=size_t(lk1-lk0)*ni*nj;
        std::vector<uint16_t> packed(core_n);
        for(int c=0;c<5;++c){
            for(int k=lk0;k<lk1;++k)for(int j=r.core_lo[1];j<r.core_hi[1];++j)for(int i=r.core_lo[0];i<r.core_hi[0];++i){
                size_t src=r.solver->g.id(i,j,k),dst=(size_t(k-lk0)*nj+j-r.core_lo[1])*ni+i-r.core_lo[0];Real phi=host[q][4*n+src],rho=r.solver->p.rho(phi);
                float x=c==0?float(host[q][src]/pressure_scale):c==1?float(phi):c==2?float(3*host[q][n+src]/rho*velocity_scale):c==3?float(3*host[q][2*n+src]/rho*velocity_scale):float(3*host[q][3*n+src]/rho*velocity_scale);
                double d=double(hi[c])-lo[c];double z=d>0?(double(x)-lo[c])/d:0.;z=std::max(0.,std::min(1.,z));packed[dst]=uint16_t(std::llround(z*65535.));
            }
            for(int k=lk0;k<lk1;++k)for(int j=r.core_lo[1];j<r.core_hi[1];++j){
                size_t global_index=(size_t(k+r.halo_lo[2])*ny+j+r.halo_lo[1])*nx+r.core_lo[0]+r.halo_lo[0];
                std::streamoff off=header+std::streamoff(c*global_n+global_index)*sizeof(uint16_t);
                size_t local=(size_t(k-lk0)*nj+j-r.core_lo[1])*ni;
                out.seekp(off);out.write(reinterpret_cast<const char*>(packed.data()+local),std::streamsize(ni*sizeof(uint16_t)));
            }
        }
    }
    out.flush();
}

inline int run_multi_gpu(const std::string& output,
                         const std::string& name,double L,double T,double mass_scale,
                         double dtsec,double end,int limit,int logevery,
                         int mode,double phi_tol,double umax_limit,double pressure_scale,
                         Physics p,Body b,const std::vector<Marker>& all_markers,
                         const AleTrajectory& trajectory,const PolicyAssemblyTrajectory& policy_trajectory,const MetricMesh* global_mesh,
                         int requested_devices,bool write_fields,double output_interval,double force_output_interval,double video_output_interval,
                         const std::string& decomp="z",bool four_stage=false) {
    const RkScheme rk{four_stage};
    if(four_stage&&(!policy_trajectory.enabled||b.free))throw std::runtime_error("SSPRK43 requires prescribed policy solids");
    if(!global_mesh)throw std::runtime_error("multi-GPU mode requires axis-file metric mesh");
    int available=0;cuda_check(cudaGetDeviceCount(&available));
    if(requested_devices<2||available<requested_devices)throw std::runtime_error("requested GPU count is not available; refusing a smaller run");
    const int nr=requested_devices;
    const bool pencil=decomp=="xy";
    const bool column_repair=std::getenv("LU_COLUMN_PHASE_REPAIR")&&std::atoi(std::getenv("LU_COLUMN_PHASE_REPAIR"))==1;
    if(column_repair&&!pencil)throw std::runtime_error("column phase repair requires complete columns: select --decomp xy");
    if(column_repair&&std::getenv("LU_CONSERVATIVE_PHASE_REPAIR")&&std::atoi(std::getenv("LU_CONSERVATIVE_PHASE_REPAIR")))
        throw std::runtime_error("select one phase repair algorithm, not both");
    if(decomp!="z"&&!pencil)throw std::runtime_error("unknown decomposition");
    if(pencil&&nr!=8)throw std::runtime_error("4x2 pencil requires eight GPUs");
    for(int dst=0;dst<nr;++dst)for(int src=0;src<nr;++src)if(src!=dst){
        bool adjacent=pencil?((src/4==dst/4&&std::abs(src-dst)==1)||std::abs(src-dst)==4):std::abs(src-dst)==1;
        if(!adjacent)continue;
        int access=0;cuda_check(cudaDeviceCanAccessPeer(&access,dst,src));
        if(!access)throw std::runtime_error("adjacent GPU peer access unavailable");
        cuda_check(cudaSetDevice(dst));auto e=cudaDeviceEnablePeerAccess(src,0);
        if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else cuda_check(e);
    }
    const int nz=global_mesh->description.nz,halo=12,nx=global_mesh->description.nx,ny=global_mesh->description.ny;
    p.hydrostatic_top=global_mesh->description.extent(p.vertical_axis);p.global_hydrostatic_top=true;
    std::vector<MultiGpuRank> ranks;ranks.reserve(nr);
    int compute_padding=2;
#ifdef LU_FULL_FLUID_HALO
    compute_padding=halo;
#endif
    // The same cost model in each decomposed direction compensates for the
    // extra face work on interior ranks. Timings remain the acceptance metric.
    auto xs=pencil?multi_gpu_balanced_slabs(nx,4,halo,compute_padding):multi_gpu_split_axis(nx,1);
    // Optional measured-work partition; preserve the global mesh and stencil.
    if(const char* raw=std::getenv("LU_PENCIL_X_CUTS")){
        if(!pencil)throw std::runtime_error("custom x cuts require xy decomposition");
        std::stringstream input(raw);std::string token;std::vector<int> cuts{0};
        while(std::getline(input,token,',')){size_t used=0;int cut=std::stoi(token,&used);if(used!=token.size())throw std::runtime_error("invalid x cut");cuts.push_back(cut);}
        cuts.push_back(nx);if(cuts.size()!=5)throw std::runtime_error("exactly three interior x cuts required");
        for(int a=0;a<4;++a){if(cuts[a+1]-cuts[a]<halo)throw std::runtime_error("x cut produces a core narrower than halo");xs[a]={cuts[a],cuts[a+1]};}
        std::cout<<"pencil_x_cuts="<<raw<<'\n';
    }
    auto ys=pencil?multi_gpu_balanced_slabs(ny,2,halo,compute_padding):multi_gpu_split_axis(ny,1);
    auto zs=pencil?multi_gpu_split_axis(nz,1):multi_gpu_balanced_slabs(nz,nr,halo,compute_padding);
    for(int r=0;r<nr;++r){
        int px=pencil?r%4:0,py=pencil?r/4:0,pz=pencil?0:r;
        int lo[3]={xs[px].first,ys[py].first,zs[pz].first},hi[3]={xs[px].second,ys[py].second,zs[pz].second};
        int hl[3],hh[3],counts[3]={nx,ny,nz};
        for(int a=0;a<3;++a){hl[a]=std::max(0,lo[a]-halo);hh[a]=std::min(counts[a],hi[a]+halo);}
        cuda_check(cudaSetDevice(r));
        auto axes=multi_gpu_local_axes_xy(*global_mesh,hl,hh);
        auto mm=std::make_unique<MetricMesh>(axes,global_mesh->description.h);
        auto& g=mm->description;g.global_nx=nx;g.global_ny=ny;g.global_nz=nz;
        g.global_i0=hl[0];g.global_j0=hl[1];g.global_k0=hl[2];
        std::vector<Marker> local;
        Real zlo=axes[2].front()-Real(4)*global_mesh->description.h;
        Real zhi=axes[2].back()+Real(4)*global_mesh->description.h;
        if(policy_trajectory.enabled||pencil)local=all_markers;
        else for(const Marker& m:all_markers)if(m.x.z>=zlo&&m.x.z<=zhi)local.push_back(m);
        size_t free_bytes=0,total_bytes=0;cuda_check(cudaMemGetInfo(&free_bytes,&total_bytes));
        // Leave room for persistent packed halos and compact surface buffers.
        size_t exchange_bytes=pencil?size_t(20)*halo*(size_t(g.nx)+g.ny)*g.nz*sizeof(Real):0;
        if(Solver::allocation_bytes(mm->grid(),p,local.size())+exchange_bytes+(column_repair?40*g.n():0)>free_bytes*0.9)
            throw std::runtime_error("insufficient local solver memory on GPU "+std::to_string(r));
        auto sol=std::make_unique<Solver>(mm->grid(),p,b,local,trajectory,policy_trajectory);
        sol->async_video_output=true;sol->ib_partitioned=true;
        sol->core_i0=lo[0]-hl[0];sol->core_i1=hi[0]-hl[0];sol->core_j0=lo[1]-hl[1];sol->core_j1=hi[1]-hl[1];
        sol->core_k0=lo[2]-hl[2];sol->core_k1=hi[2]-hl[2];
        sol->ale_time_scale=T;sol->ale_velocity_scale=L/T;sol->update_ale(0);sol->update_policy_pose(0);sol->initialize(mode);
        ranks.push_back({r,lo[2],hi[2],hl[2],hh[2],std::move(mm),std::move(sol),
                        {lo[0]-hl[0],lo[1]-hl[1],lo[2]-hl[2]},{hi[0]-hl[0],hi[1]-hl[1],hi[2]-hl[2]},
                        {hl[0],hl[1],hl[2]},{hh[0],hh[1],hh[2]},px,py});
    }
    std::unique_ptr<PencilHalo> pencil_halo;
    if(pencil)pencil_halo=std::make_unique<PencilHalo>(ranks,halo);
    double halo_seconds=0;
    auto exchange=[&](bool predicted){
        multi_gpu_sync_all(ranks);auto start=std::chrono::steady_clock::now();
        if(pencil)pencil_halo->exchange(ranks,predicted);else multi_gpu_exchange(ranks,predicted);
        halo_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    };
    auto core_cells=[](const MultiGpuRank& r){return size_t(r.core_hi[0]-r.core_lo[0])*(r.core_hi[1]-r.core_lo[1])*(r.core_hi[2]-r.core_lo[2]);};
    std::vector<std::unique_ptr<ColumnPhaseRepair>> column;
    double column_seconds=0,column_roundoff=0;size_t corrected_columns=0;
    if(column_repair)for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));column.emplace_back(std::make_unique<ColumnPhaseRepair>(r.solver->g));}
    auto repair_columns=[&](bool predicted){
        if(!column_repair)return;multi_gpu_sync_all(ranks);auto start=std::chrono::steady_clock::now();
        for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));auto& sol=*r.solver;
            column[r.device]->launch((predicted?sol.work:sol.state).data(),p,sol.core_i0,sol.core_i1,sol.core_j0,sol.core_j1);}
        for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));auto result=column[r.device]->check();corrected_columns+=size_t(result[1]);column_roundoff+=result[3];}
        column_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    };
    if(const char* restart=std::getenv("LU_RESTART_CHECKPOINT"))multi_gpu_checkpoint(restart,ranks,true);
    repair_columns(false);multi_gpu_sync_all(ranks);exchange(false);
#ifdef LU_EXPERIMENTAL_PHASE_REPAIR
    std::unique_ptr<MultiGpuPhaseRepair> phase_repair;
    if(std::getenv("LU_CONSERVATIVE_PHASE_REPAIR")&&std::atoi(std::getenv("LU_CONSERVATIVE_PHASE_REPAIR"))==1){
        if(pencil)throw std::runtime_error("global legacy phase repair only supports z slabs; pencil requires validated local transport");
        cuda_check(cudaSetDevice(0));phase_repair=std::make_unique<MultiGpuPhaseRepair>(*global_mesh,ranks);
        // A restart can already contain overshoots. Repair once before base is
        // formed, otherwise convex RK blending would retain that old defect.
        for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));r.solver->work.copy_from(r.solver->state);}
        phase_repair->apply(ranks);
        for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));r.solver->state.copy_from(r.solver->work);}
        exchange(false);
    }
#endif
    double start_seconds=ranks.front().solver->elapsed_lattice/T;
    int start_step=int(std::llround(start_seconds/dtsec));
    double schedule_origin=start_seconds-start_step*dtsec;
    if(std::abs(schedule_origin)<1e-10)schedule_origin=0;
    int steps=int(std::ceil((end-schedule_origin)/dtsec-1e-9));if(limit>0)steps=std::min(steps,limit);
    make_directory(output);
    std::ofstream summary(output+"/multi_gpu_rank_timing.csv");summary<<"step,rank,core_k0,core_k1,core_cells,local_cells,compute_s\n";
    const int audit_from=std::getenv("LU_PHASE_AUDIT_FROM")?std::atoi(std::getenv("LU_PHASE_AUDIT_FROM")):0;
    const int audit_until=std::getenv("LU_PHASE_AUDIT_UNTIL")?std::atoi(std::getenv("LU_PHASE_AUDIT_UNTIL")):0;
    const bool audit_cfl=std::getenv("LU_PHASE_AUDIT_CFL")&&std::atoi(std::getenv("LU_PHASE_AUDIT_CFL"))!=0;
    std::ofstream audit;
    if(audit_from){audit.open(output+"/phase_audit.csv");audit<<"step,stage,rank,operation,phi_min,phi_max,bad_cells,x_m,y_m,z_m,p,mx,my,mz,phi,rho,umax_lattice,outgoing_advective_cfl,ux_m,uy_m,uz_m,up,umx,umy,umz,uphi\n";}
    Real audit_dt=Real(dtsec*T);
    auto audit_all=[&](int it,int stage,const char* label,bool predicted){if(!audit_from||it<audit_from||(audit_until>0&&it>audit_until))return;
        // Queue all independent card scans before the first blocking download.
        for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));launch_phase_audit(*r.solver,predicted,audit_dt,audit_cfl);}
        bool bad=false;for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));bad=collect_phase_audit(*r.solver,audit,it,stage,r.device,label,predicted,L,audit_cfl)||bad;}
        if(bad)throw std::runtime_error(std::string("stage audit: invalid state after ")+label+" step="+std::to_string(it)+" stage="+std::to_string(stage));};
    std::vector<double> rank_seconds(nr,0);
    double field_output_seconds=0, force_output_seconds=0,video_output_seconds=0;
    std::vector<cudaEvent_t> ev_start(nr),ev_stop(nr);
    for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));cuda_check(cudaEventCreate(&ev_start[r.device]));cuda_check(cudaEventCreate(&ev_stop[r.device]));}
    auto wall0=std::chrono::steady_clock::now();
    double next_field_output=output_interval>0?output_interval:1e300;
    double next_force_output=force_output_interval>0?force_output_interval:1e300;
    double next_video_output=video_output_interval>0?video_output_interval:1e300;
    if(output_interval>0)next_field_output=(std::floor(start_seconds/output_interval+1e-9)+1)*output_interval;
    if(force_output_interval>0)next_force_output=(std::floor(start_seconds/force_output_interval+1e-9)+1)*force_output_interval;
    if(video_output_interval>0)next_video_output=(std::floor(start_seconds/video_output_interval+1e-9)+1)*video_output_interval;
    if(video_output_interval>0){
        auto start=std::chrono::steady_clock::now();auto af=trajectory.sample(start_seconds);
        for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));r.solver->prescribed_force.zero();r.solver->write_video_frame(output+"/video_surface_rank_"+std::to_string(r.device)+".bin",start_seconds,L,T,mass_scale,af,r.core0-r.halo0,r.core1-r.halo0);}
        video_output_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    }
    int checkpoint_from=std::getenv("LU_CHECKPOINT_FROM")?std::atoi(std::getenv("LU_CHECKPOINT_FROM")):0;
    int checkpoint_every=std::getenv("LU_CHECKPOINT_EVERY")?std::atoi(std::getenv("LU_CHECKPOINT_EVERY")):0;
    // Opt-in global rollback/subcycling. No additional full-domain scans or
    // state arrays: base already holds the complete pre-RK state on each GPU.
    const double retry_min=std::getenv("LU_TRANSPORT_MIN_DT_S")?std::atof(std::getenv("LU_TRANSPORT_MIN_DT_S")):0;
    if(!std::isfinite(retry_min)||retry_min<0||retry_min>dtsec||(retry_min>0&&b.free))
        throw std::runtime_error("transport retry requires prescribed solids and 0 < minimum dt <= nominal dt");
    double retry_cap=dtsec;
    std::ofstream retries;
    if(retry_min>0){retries.open(output+"/transport_retries.log");if(!retries)throw std::runtime_error("cannot open retry log");}
    for(int it=start_step+1;it<=steps;++it){
        const auto clock=step_time(it,dtsec,end,T,schedule_origin);
        Real d=clock.lattice_dt;
        const double output_time=clock.end_s;
        double accepted_time=clock.start_s;
        while(accepted_time<output_time-1e-13){
        const double trial_seconds=std::min(retry_cap,output_time-accepted_time);
        d=Real(trial_seconds*T);audit_dt=d;
        int active_stage=-1,active_rank=-1;
        // Re-anchor to scheduled physical time; rejected stages never advance it.
        for(auto& r:ranks)r.solver->elapsed_lattice=accepted_time*T;
        try {
        for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));r.solver->begin_step();}
        for(int stage=0;stage<rk.stages();++stage){
            active_stage=stage;
            const Real stage_dt=Real(double(d)*rk.euler_fraction());audit_dt=stage_dt;
            for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));cuda_check(cudaEventRecord(ev_start[r.device]));
                if(four_stage){double t=accepted_time+rk.abscissa(stage)*trial_seconds;r.solver->update_policy_pose(t);r.solver->update_ale(t);}
                else r.solver->begin_stage(d,stage);
                r.solver->fluid_predict(stage_dt);cuda_check(cudaEventRecord(ev_stop[r.device]));}
            for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));active_rank=r.device;r.solver->enforce_transport_guard();}
            repair_columns(true);
            audit_all(it,stage,"fluid",true);
            exchange(true);
            for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));float ms=0;cuda_check(cudaEventElapsedTime(&ms,ev_start[r.device],ev_stop[r.device]));rank_seconds[r.device]+=double(ms)/1000.;}
#ifdef LU_EXPERIMENTAL_PHASE_REPAIR
            if(phase_repair)phase_repair->apply(ranks);
#endif
            for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));cuda_check(cudaEventRecord(ev_start[r.device]));r.solver->accumulate_policy_force=(stage==rk.stages()-1)&&
                    ((force_output_interval>0&&output_time+1e-12>=next_force_output)||
                     (video_output_interval>0&&output_time+1e-12>=next_video_output));
                r.solver->immersed_predict(stage_dt,true,true);}
            if(policy_trajectory.enabled&&p.ib_phase_bound&&b.enabled){
                std::array<double,4> budget{};
                // Independent reductions run concurrently; collect in the same
                // rank/partial order as before to preserve the global budget.
                for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));r.solver->launch_policy_phase_budget(stage_dt);}
                for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));auto local=r.solver->collect_policy_phase_budget();for(int a=0;a<4;++a)budget[a]+=local[a];}
                for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));r.solver->apply_policy_phase_budget(stage_dt,budget);}
            }
            audit_all(it,stage,"ib",true);
            for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));if(four_stage)r.solver->finish_prescribed_stage(rk.base_weight(stage));else r.solver->finish_stage(d,stage);cuda_check(cudaEventRecord(ev_stop[r.device]));}
            multi_gpu_sync_all(ranks);
            for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));float ms=0;cuda_check(cudaEventElapsedTime(&ms,ev_start[r.device],ev_stop[r.device]));rank_seconds[r.device]+=double(ms)/1000.;}
            exchange(false);
        }
        for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));r.solver->finish_step(d);}
        multi_gpu_sync_all(ranks);
        accepted_time+=trial_seconds;
        for(auto& r:ranks)r.solver->elapsed_lattice=accepted_time*T;
        } catch(const TransportCourantError& failure) {
            // A failure on any rank rejects the SAME whole RK trial everywhere.
            // Restore even ranks whose stage has completed; work/forces are
            // regenerated before any output. No failing field is accepted.
            multi_gpu_sync_all(ranks);
            for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));r.solver->state.copy_from(r.solver->base);r.solver->elapsed_lattice=accepted_time*T;}
            multi_gpu_sync_all(ranks);
            std::ostringstream context;
            context<<std::setprecision(16)<<"step="<<it<<" stage="<<active_stage<<" rank="<<active_rank
                   <<" accepted_time_s="<<accepted_time<<" attempted_dt_s="<<trial_seconds<<"\n"<<failure.what();
            if(retry_min<=0||trial_seconds*.5<retry_min*(1-1e-10)){
                multi_gpu_checkpoint(output+"/last_accepted_checkpoint",ranks,false);
                throw TransportCourantError(context.str()+"\nretry disabled or minimum dt reached; preserved last accepted state");
            }
            retry_cap=trial_seconds*.5;
            retries<<context.str()<<"\nretry_dt_s="<<retry_cap<<std::endl;
            std::cout<<"transport_retry step="<<it<<" stage="<<active_stage<<" rank="<<active_rank
                     <<" time_s="<<accepted_time<<" dt_s="<<retry_cap<<std::endl;
        }
        } // accepted substeps reach the unchanged macro/output time
        if(checkpoint_from>0&&checkpoint_every>0&&it>=checkpoint_from&&it%checkpoint_every==0)
            multi_gpu_checkpoint(output+"/checkpoint_"+std::to_string(it),ranks,false);
        if(it%logevery==0||it==steps){
            for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));Stats st=r.solver->stats();
                auto bs=r.solver->body_state.download();
                if(st.nonfinite||!std::isfinite(st.umax)||st.phimin < -phi_tol||st.phimax>1+phi_tol||
                   (umax_limit>0&&st.umax>umax_limit)||p.rho(Real(st.phimin))<=0||bs[6]||bs[24]){
                    // Keep the actual failed state, distinct from an accepted
                    // restart. Previously health-check failures lost this data.
                    multi_gpu_checkpoint(output+"/failed_health_checkpoint",ranks,false);
                    throw std::runtime_error("multi-GPU health check failed at step "+std::to_string(it)+" rank "+std::to_string(r.device)+" phi_min="+std::to_string(st.phimin)+" phi_max="+std::to_string(st.phimax)+" umax="+std::to_string(st.umax)+" nonfinite="+std::to_string(st.nonfinite)+" ib_errors="+std::to_string(bs[6])+" projection_errors="+std::to_string(bs[24]));
                }
                summary<<it<<','<<r.device<<','<<r.core0<<','<<r.core1<<','<<core_cells(r)<<','<<r.solver->g.n()<<','<<rank_seconds[r.device]<<'\n';
                std::cout<<"rank="<<r.device<<" step="<<it<<" phi=["<<st.phimin<<','<<st.phimax<<"] umax="<<st.umax<<"\n";}
            summary.flush();std::cout.flush();
        }
        bool field_due=write_fields&&output_interval>0&&output_time+1e-12>=next_field_output;
        bool force_due=force_output_interval>0&&output_time+1e-12>=next_force_output;
        if(field_due){
            auto output_start=std::chrono::steady_clock::now();
            int frame=int(std::floor((output_time)/output_interval+1e-9))-1;
            multi_gpu_write_global_uint16(output+"/fields_"+std::to_string(frame)+".u16",ranks,nx,ny,nz,pressure_scale,L/T);
            std::ofstream om(output+"/fields_"+std::to_string(frame)+".json");
            om<<"{\"time_s\":"<<std::setprecision(12)<<output_time<<",\"channels\":[\"pressure_Pa\",\"phi\",\"ux_ms\",\"uy_ms\",\"uz_ms\"],\"dtype\":\"uint16\",\"nx\":"<<nx<<",\"ny\":"<<ny<<",\"nz\":"<<nz<<",\"halo_cells\":0,\"layout\":\"header_10_float32_then_5_channel_major_uint16_core\",\"lagrangian_force_model\":\"exterior_stress_world_body_origin\"}\n";
            while(next_field_output<=output_time+1e-12)next_field_output+=output_interval;
            field_output_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-output_start).count();
        }
        if(force_due){
            auto output_start=std::chrono::steady_clock::now();
            int frame=int(std::floor((output_time)/force_output_interval+1e-9))-1;
            std::vector<std::string> force_paths;
            for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));std::string dpath=output+"/rank_"+std::to_string(r.device);make_directory(dpath);
                std::string fp=dpath+"/lagrangian_forces_"+std::to_string(frame)+".csv";
                r.solver->write_lagrangian_forces(fp,L,mass_scale*L/(T*T),double(d));force_paths.push_back(fp);
            }
            std::ofstream combined(output+"/lagrangian_forces_"+std::to_string(frame)+".csv");
            combined<<std::setprecision(16)<<"kind,id,x_m,y_m,z_m,fx_N,fy_N,fz_N,mx_Nm,my_Nm,mz_Nm\n";
            struct Aggregate {double x=0,y=0,z=0,f[6]={0,0,0,0,0,0};int count=0;};std::map<std::pair<std::string,int>,Aggregate> aggregate;
            for(const auto& fp:force_paths){std::ifstream in(fp);std::string line;std::getline(in,line);while(std::getline(in,line)){if(line.empty())continue;std::stringstream q(line);std::string tok,kind;int id;double x,y,z;std::array<double,6> f{};std::getline(q,kind,',');std::getline(q,tok,',');id=std::stoi(tok);std::getline(q,tok,',');x=std::stod(tok);std::getline(q,tok,',');y=std::stod(tok);std::getline(q,tok,',');z=std::stod(tok);for(int k=0;k<6;++k){std::getline(q,tok,',');f[k]=std::stod(tok);}auto& a=aggregate[{kind,id}];a.x+=x;a.y+=y;a.z+=z;for(int k=0;k<6;++k)a.f[k]+=f[k];a.count++;}}
            for(const auto& kv:aggregate){const auto& k=kv.first;const auto& a=kv.second;double div=std::max(1,a.count);combined<<k.first<<','<<k.second<<','<<a.x/div<<','<<a.y/div<<','<<a.z/div;for(int n=0;n<6;++n)combined<<','<<a.f[n];combined<<'\n';}
            while(next_force_output<=output_time+1e-12)next_force_output+=force_output_interval;
            force_output_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-output_start).count();
        }
        if(video_output_interval>0&&output_time+1e-12>=next_video_output){
            auto video_start=std::chrono::steady_clock::now();
            AleFrameHost af=trajectory.enabled?trajectory.sample(output_time):AleFrameHost{};
            for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));r.solver->write_video_frame(output+"/video_surface_rank_"+std::to_string(r.device)+".bin",output_time,L,T,mass_scale,af,r.core0-r.halo0,r.core1-r.halo0);}
            while(next_video_output<=output_time+1e-12)next_video_output+=video_output_interval;
            video_output_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-video_start).count();
        }
    }
    auto drain_start=std::chrono::steady_clock::now();
    for(auto& r:ranks)r.solver->finish_video_output();
    video_output_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-drain_start).count();
    double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall0).count();
    std::ofstream hm(output+"/halo_metrics.json");hm<<"{\"decomposition\":\""<<(pencil?"4x2x1":"z_slab")<<"\",\"exchange_seconds\":"<<halo_seconds<<",\"wall_seconds\":"<<wall<<"}\n";
    size_t core_min=std::numeric_limits<size_t>::max(),core_max=0,res_min=std::numeric_limits<size_t>::max(),res_max=0;
    for(const auto& r:ranks){size_t core=core_cells(r),res=r.solver->g.n();core_min=std::min(core_min,core);core_max=std::max(core_max,core);res_min=std::min(res_min,res);res_max=std::max(res_max,res);}
    double compute_max_rank=*std::max_element(rank_seconds.begin(),rank_seconds.end());
    std::ofstream meta(output+"/multi_gpu_metrics.json");meta<<"{\n  \"case\":\""<<name<<"\",\n  \"devices\":"<<nr<<",\n  \"steps\":"<<steps<<",\n  \"wall_seconds\":"<<wall<<",\n  \"compute_seconds_max_rank\":"<<compute_max_rank<<",\n  \"field_output_seconds\":"<<field_output_seconds<<",\n  \"force_output_seconds\":"<<force_output_seconds<<",\n  \"output_seconds_total\":"<<field_output_seconds+force_output_seconds+video_output_seconds<<",\n  \"output_wall_fraction\":"<<(wall>0?(field_output_seconds+force_output_seconds+video_output_seconds)/wall:0)<<",\n  \"halo_cells\":"<<halo<<",\n  \"core_cells_min\":"<<core_min<<",\n  \"core_cells_max\":"<<core_max<<",\n  \"resident_cells_min\":"<<res_min<<",\n  \"resident_cells_max\":"<<res_max<<",\n  \"field_output\":\"one_global_uint16_file_per_frame\",\n  \"force_output\":\"one_combined_csv_per_frame_plus_rank_diagnostics\"\n}\n";
    for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));cuda_check(cudaEventDestroy(ev_start[r.device]));cuda_check(cudaEventDestroy(ev_stop[r.device]));}
#ifdef LU_EXPERIMENTAL_PHASE_REPAIR
    if(phase_repair){std::ofstream repair_metrics(output+"/phase_repair_metrics.json");repair_metrics<<"{\"calls\":"<<phase_repair->calls<<",\"seconds\":"<<phase_repair->seconds<<",\"total_iterations\":"<<phase_repair->total_iterations<<",\"max_iterations\":"<<phase_repair->max_iterations<<"}\n";phase_repair.reset();}
#endif
    if(column_repair){std::ofstream cm(output+"/column_repair_metrics.json");cm<<"{\"seconds\":"<<column_seconds<<",\"corrected_columns\":"<<corrected_columns<<",\"accumulated_abs_phase_volume_roundoff_lattice\":"<<column_roundoff<<"}\n";
        for(auto& r:ranks){cuda_check(cudaSetDevice(r.device));column[r.device].reset();}}
    pencil_halo.reset();
#ifdef LU_KERNEL_TIMING
    kernel_timing().write(output+"/kernel_timings.tsv");
#endif
    for(auto it=ranks.rbegin();it!=ranks.rend();++it){cuda_check(cudaSetDevice(it->device));it->solver.reset();it->mesh.reset();}
    return 0;
}
#endif
