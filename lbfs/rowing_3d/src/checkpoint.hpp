#pragma once
#include "checkpoint_store.hpp"
// Diagnostic restart at a completed RK step. Same Real type, axes, rank
// decomposition and prescribed-motion inputs are required. Not a remesher.
inline void multi_gpu_checkpoint(const std::string& folder,const std::vector<MultiGpuRank>& ranks,bool reading) {
    std::string resolved=reading?checkpoint_resolve(folder):folder;
    if(!reading)make_directory(resolved);
    for(const auto& r:ranks){cuda_check(cudaSetDevice(r.device));auto& s=*r.solver;
        std::string path=resolved+"/rank_"+std::to_string(r.device)+".checkpoint";
        std::vector<uint64_t> shape={0x4c55435030303031ull,sizeof(Real),uint64_t(ranks.size()),uint64_t(s.g.nx),uint64_t(s.g.ny),uint64_t(s.g.nz),uint64_t(r.core0),uint64_t(r.core1),uint64_t(r.halo0)};
        bool pencil=s.core_i0>0||s.core_j0>0||(s.core_i1>=0&&s.core_i1<s.g.nx)||(s.core_j1>=0&&s.core_j1<s.g.ny);
        if(pencil){shape[0]=0x4c55435030303032ull;
            shape.insert(shape.end(),{uint64_t(s.core_i0),uint64_t(s.core_i1),uint64_t(s.core_j0),uint64_t(s.core_j1),uint64_t(s.g.global_i0),uint64_t(s.g.global_j0)});}
        if(reading){
            std::ifstream in(path,std::ios::binary);std::vector<uint64_t> actual(shape.size());
            in.read(reinterpret_cast<char*>(actual.data()),actual.size()*sizeof(uint64_t));
            if(!in||shape!=actual)throw std::runtime_error("checkpoint shape/type mismatch");
            in.read(reinterpret_cast<char*>(&s.elapsed_lattice),sizeof(double));
            for(const auto& axis:r.mesh->axes){std::vector<Real> values(axis.size());in.read(reinterpret_cast<char*>(values.data()),values.size()*sizeof(Real));if(values!=axis)throw std::runtime_error("checkpoint axes differ");}
            std::vector<Real> state(s.state.size());std::vector<double> body(s.body_state.size());
            in.read(reinterpret_cast<char*>(state.data()),state.size()*sizeof(Real));
            in.read(reinterpret_cast<char*>(body.data()),body.size()*sizeof(double));
            if(!in||!std::isfinite(s.elapsed_lattice))throw std::runtime_error("checkpoint truncated/invalid");
            s.state.upload(state);s.body_state.upload(body);s.update_ale(s.elapsed_lattice/s.ale_time_scale);s.update_policy_pose(s.elapsed_lattice/s.ale_time_scale);
        }else{
            std::ofstream out(path,std::ios::binary|std::ios::trunc);
            out.write(reinterpret_cast<const char*>(shape.data()),shape.size()*sizeof(uint64_t));
            out.write(reinterpret_cast<const char*>(&s.elapsed_lattice),sizeof(double));
            for(const auto& axis:r.mesh->axes)out.write(reinterpret_cast<const char*>(axis.data()),axis.size()*sizeof(Real));
            auto state=s.state.download();auto body=s.body_state.download();
            out.write(reinterpret_cast<const char*>(state.data()),state.size()*sizeof(Real));
            out.write(reinterpret_cast<const char*>(body.data()),body.size()*sizeof(double));
            out.flush();if(!out)throw std::runtime_error("checkpoint write failed");
        }
    }
    if(reading){double time=ranks.front().solver->elapsed_lattice;for(const auto& r:ranks)if(r.solver->elapsed_lattice!=time)throw std::runtime_error("checkpoint rank times differ");}
}
