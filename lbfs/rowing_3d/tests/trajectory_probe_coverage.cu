// Geometry-only full-trajectory test: no fluid state allocation or advancement.
#include "solver.hpp"
#include <memory>
struct ProbeBox {int lo[3],hi[3];};
int main(int argc,char**argv){try{
    if(argc!=2)throw std::runtime_error("usage: trajectory_probe_coverage case_folder");
    std::string root=argv[1];std::array<std::vector<Real>,3> axes;
    for(int a=0;a<3;++a){std::ifstream in(root+"/axis_"+std::string(1,"xyz"[a])+".txt");double x;while(in>>x)axes[a].push_back(Real(x*1000));if(axes[a].size()<5)throw std::runtime_error("missing axes");}
    auto markers=load_policy_markers_csv(root+"/policy_markers_exterior.csv");for(auto& m:markers){m.x.x=Real(double(m.x.x)*1000);m.x.y=Real(double(m.x.y)*1000);m.x.z=Real(double(m.x.z)*1000);}
    PolicyAssemblyTrajectory pt;pt.load_csv(root+"/replay/policy_trajectory.csv");for(auto& f:pt.frames)for(auto& p:f.part){p.px*=1000;p.py*=1000;p.pz*=1000;}
    AleTrajectory at;at.load_csv(root+"/replay/ale_motion.csv");
    std::vector<std::unique_ptr<Buffer<Real>>> storage;
    auto make_grid=[&](int* low,int* high){Grid g{high[0]-low[0],high[1]-low[1],high[2]-low[2],3.5f};std::vector<Real> v;
        for(int a=0;a<3;++a){g.face_offset[a]=int(v.size());v.insert(v.end(),axes[a].begin()+low[a],axes[a].begin()+high[a]+1);}
        storage.emplace_back(new Buffer<Real>(v.size()));storage.back()->upload(v);g.metric_device=storage.back()->data();g.metric_host=storage.back()->data();return g;};
    int zero[3]={0,0,0},end[3]={1245,1277,383};Grid global=make_grid(zero,end);
    std::vector<Grid> grids;std::vector<ProbeBox> boxes;int xc[]={0,303,625,952,1245},yc[]={0,639,1277};
    for(int r=0;r<8;++r){int low[3]={xc[r%4],yc[r/4],0},high[3]={xc[r%4+1],yc[r/4+1],383},hl[3],hh[3];ProbeBox box;
        for(int a=0;a<3;++a){hl[a]=std::max(0,low[a]-12);hh[a]=std::min(end[a],high[a]+12);box.lo[a]=low[a]-hl[a];box.hi[a]=high[a]-hl[a];}
        grids.push_back(make_grid(hl,hh));boxes.push_back(box);
    }
    Buffer<Grid> gd(8);gd.upload(grids);Buffer<ProbeBox> bd(8);bd.upload(boxes);Buffer<Marker> md(markers.size());md.upload(markers);Buffer<PolicyPoseDevice> poses(3);Buffer<double> errors(1);
    auto gp=gd.data();auto bp=bd.data();auto mp=md.data();auto pp=poses.data();auto ep=errors.data();size_t n=markers.size();
    int checked=0;
    for(int it=0;it<=5800;++it){double t=it*.002;auto frame=pt.sample(t);std::vector<PolicyPoseDevice> pv(3);
        for(int j=0;j<3;++j){auto p=pose_in_ale(frame.part[j],at.sample(t),0,0,1000);pv[j].position={Real(p.px),Real(p.py),Real(p.pz)};pv[j].qw=Real(p.qw);pv[j].qx=Real(p.qx);pv[j].qy=Real(p.qy);pv[j].qz=Real(p.qz);}
        poses.upload(pv);errors.zero();
        parallel_for(n,[=] HD(size_t k){Marker m=policy_world_marker(mp[k],pp,true,0);Vec x[2];bool bad=!surface_probe_locations(global,m,x[0],x[1]);
            if(!bad)for(int probe=0;probe<2;++probe){int owners=0;for(int r=0;r<8;++r)if(surface_inside_box(gp[r],x[probe],bp[r].lo,bp[r].hi)){++owners;if(!surface_probe_covered(gp[r],x[probe]))bad=true;}if(owners!=1)bad=true;}
            if(bad)atomic_plus(ep,1.);
        });
        auto e=errors.download()[0];if(e){std::cerr<<"FAIL t="<<t<<" invalid_markers="<<e<<"\n";return 2;}++checked;
        if(it%1000==0)std::cout<<"coverage t="<<t<<" passed\n"<<std::flush;
    }
    std::cout<<"PASS "<<checked<<" output times, "<<n<<" markers, global 2h/3h offsets, unique owner and full stress stencil\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}


