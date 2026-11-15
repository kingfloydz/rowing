#undef NDEBUG
#include "numerics.hpp"
#include "ale_motion.hpp"
#include <iostream>
#include <cassert>
int main(int argc,char**argv){
    assert(argc==2);PolicyAssemblyTrajectory t;t.load_csv(argv[1]);double maxjump=0,maxstep=0,maxnorm=0;
    for(size_t k=1;k+1<t.frames.size();++k){double time=t.frames[k].t;auto l=t.sample(time-1e-10),r=t.sample(time+1e-10);
        for(int p=0;p<3;++p){auto a=l.part[p],b=r.part[p];double jump=std::sqrt((a.px-b.px)*(a.px-b.px)+(a.py-b.py)*(a.py-b.py)+(a.pz-b.pz)*(a.pz-b.pz));maxjump=std::max(maxjump,jump);
            double dot=a.qw*b.qw+a.qx*b.qx+a.qy*b.qy+a.qz*b.qz;assert(std::abs(dot)>1-1e-10);}}
    auto last=t.sample(0);
    for(int step=1;step<=464000;++step){auto next=t.sample(step*.000025);
        for(int p=0;p<3;++p){const auto&a=last.part[p];const auto&b=next.part[p];double d=std::sqrt((a.px-b.px)*(a.px-b.px)+(a.py-b.py)*(a.py-b.py)+(a.pz-b.pz)*(a.pz-b.pz));maxstep=std::max(maxstep,d);maxnorm=std::max(maxnorm,std::abs(b.qw*b.qw+b.qx*b.qx+b.qy*b.qy+b.qz*b.qz-1));}
        last=next;}
    assert(maxjump<1e-7&&maxnorm<1e-10&&maxstep<.001);
    std::cout<<"PASS actual 11.6s trajectory: knot_jump_m="<<maxjump<<" max_25us_displacement_m="<<maxstep<<" quaternion_norm_error="<<maxnorm<<"\n";
    // Antipodal quaternion samples represent the same orientation.
    for(auto& f:t.frames)for(auto& p:f.part){p.qw=-p.qw;p.qx=-p.qx;p.qy=-p.qy;p.qz=-p.qz;}
    auto q=t.sample(.001);assert(std::isfinite(q.part[0].qw));
}
