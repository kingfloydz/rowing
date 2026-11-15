#pragma once

// Prescribed ALE kinematics.  The trajectory is deliberately kept outside the
// CFD state: it is sampled on the host once per RK stage and the resulting
// three velocity components and yaw rate are copied into the small Field POD.
// This keeps the hot kernels free of interpolation branches.
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>

struct AleFrameHost {
    double t=0, cx=0, cy=0, cz=0, yaw=0;
    double vx=0, vy=0, vz=0, yaw_rate=0;
};

struct AleTrajectory {
    // Fixed coordinates of the tracked center in the computational mesh, m.
    double anchor_x=0,anchor_y=0;
    std::vector<AleFrameHost> frames;
    bool enabled=false;
    void load_csv(const std::string& path) {
        std::ifstream in(path);
        if(!in) throw std::runtime_error("cannot open ALE motion csv: "+path);
        std::string line; std::getline(in,line);
        while(std::getline(in,line)) {
            if(line.empty()) continue;
            std::replace(line.begin(),line.end(),',',' ');
            std::istringstream s(line); AleFrameHost f;
            if(!(s>>f.t>>f.cx>>f.cy>>f.cz>>f.yaw>>f.vx>>f.vy>>f.vz>>f.yaw_rate))
                throw std::runtime_error("invalid ALE motion row: "+line);
            if(!frames.empty() && f.t<=frames.back().t) throw std::runtime_error("ALE motion times must increase");
            frames.push_back(f);
        }
        if(frames.size()<2) throw std::runtime_error("ALE motion csv needs at least two rows");
        enabled=true;
    }
    AleFrameHost sample(double t) const {
        if(!enabled) return {};
        if(t<=frames.front().t) return frames.front();
        if(t>=frames.back().t) return frames.back();
        auto it=std::upper_bound(frames.begin(),frames.end(),t,[](double x,const AleFrameHost& f){return x<f.t;});
        const auto& b=*it; const auto& a=*(it-1); double q=(t-a.t)/(b.t-a.t);
        AleFrameHost r; r.t=t;
        auto lerp=[&](double x,double y){return x+q*(y-x);};
        r.cx=lerp(a.cx,b.cx);r.cy=lerp(a.cy,b.cy);r.cz=lerp(a.cz,b.cz);
        r.yaw=lerp(a.yaw,b.yaw);r.vx=lerp(a.vx,b.vx);r.vy=lerp(a.vy,b.vy);
        r.vz=lerp(a.vz,b.vz);r.yaw_rate=lerp(a.yaw_rate,b.yaw_rate); return r;
    }
};

// Full prescribed rigid motion exported from the policy replay.  The policy
// controls the boat free joint and the oar joints; the exporter resolves those
// joints with MuJoCo and writes world 6DOF poses for the hull and both oars.
// LBFS consumes these poses directly and never integrates a solid equation.
struct PolicyPoseHost {
    double px=0,py=0,pz=0,qw=1,qx=0,qy=0,qz=0,vx=0,vy=0,vz=0,wx=0,wy=0,wz=0;
};
// Input/output pose uses lattice positions/velocities; ALE center and anchor
// use metres. Velocity remains inertial velocity in the rotated basis: mesh
// velocity is subtracted by ALE fluxes, not by this boundary target.
inline PolicyPoseHost pose_in_ale(PolicyPoseHost p,const AleFrameHost& frame,
                                  double anchor_x,double anchor_y,double length_scale){
    const double c=std::cos(frame.yaw),s=std::sin(frame.yaw);
    double x=p.px-frame.cx*length_scale,y=p.py-frame.cy*length_scale;
    p.px=anchor_x*length_scale+c*x+s*y;p.py=anchor_y*length_scale-s*x+c*y;
    double vx=p.vx,vy=p.vy,wx=p.wx,wy=p.wy;
    p.vx=c*vx+s*vy;p.vy=-s*vx+c*vy;p.wx=c*wx+s*wy;p.wy=-s*wx+c*wy;
    double a=std::cos(frame.yaw/2),b=-std::sin(frame.yaw/2);
    double qw=p.qw,qx=p.qx,qy=p.qy,qz=p.qz;
    p.qw=a*qw-b*qz;p.qx=a*qx-b*qy;p.qy=a*qy+b*qx;p.qz=a*qz+b*qw;
    return p;
}
struct PolicyFrameHost { double t=0; PolicyPoseHost part[3]{}; };
struct PolicyAssemblyTrajectory {
    std::vector<PolicyFrameHost> frames;
    bool enabled=false;
    void load_csv(const std::string& path) {
        std::ifstream in(path); if(!in) throw std::runtime_error("cannot open policy trajectory csv: "+path);
        std::string line; std::getline(in,line);
        while(std::getline(in,line)) {
            if(line.empty()) continue; std::replace(line.begin(),line.end(),',',' ');
            std::istringstream s(line); PolicyFrameHost f;
            if(!(s>>f.t)) throw std::runtime_error("invalid policy trajectory row: "+line);
            for(int part=0;part<3;++part) {
                auto& p=f.part[part];
                if(!(s>>p.px>>p.py>>p.pz>>p.qw>>p.qx>>p.qy>>p.qz>>p.vx>>p.vy>>p.vz>>p.wx>>p.wy>>p.wz))
                    throw std::runtime_error("policy trajectory requires 40 columns: "+line);
            }
            if(!frames.empty()&&f.t<=frames.back().t) throw std::runtime_error("policy trajectory times must increase");
            frames.push_back(f);
        }
        if(frames.size()<2) throw std::runtime_error("policy trajectory needs at least two rows");
        enabled=true;
    }
    PolicyFrameHost sample(double t) const {
        if(!enabled)return {};
        if(t<=frames.front().t)return frames.front();
        if(t>=frames.back().t)return frames.back();
        auto it=std::upper_bound(frames.begin(),frames.end(),t,[](double x,const PolicyFrameHost& f){return x<f.t;});
        const auto& a=*(it-1);const auto& b=*it;double q=(t-a.t)/(b.t-a.t);PolicyFrameHost r;r.t=t;
        for(int part=0;part<3;++part){const auto& x=a.part[part];const auto& y=b.part[part];auto l=[&](double u,double v){return u+q*(v-u);};auto& z=r.part[part];
            z.px=l(x.px,y.px);z.py=l(x.py,y.py);z.pz=l(x.pz,y.pz);z.vx=l(x.vx,y.vx);z.vy=l(x.vy,y.vy);z.vz=l(x.vz,y.vz);z.wx=l(x.wx,y.wx);z.wy=l(x.wy,y.wy);z.wz=l(x.wz,y.wz);
            double sign=(x.qw*y.qw+x.qx*y.qx+x.qy*y.qy+x.qz*y.qz)<0?-1.:1.;double qw=l(x.qw,sign*y.qw),qx=l(x.qx,sign*y.qx),qy=l(x.qy,sign*y.qy),qz=l(x.qz,sign*y.qz),n=std::sqrt(qw*qw+qx*qx+qy*qy+qz*qz);z.qw=qw/n;z.qx=qx/n;z.qy=qy/n;z.qz=qz/n;
        }
        return r;
    }
};

inline std::vector<Marker> load_policy_markers_csv(const std::string& path) {
    std::ifstream in(path); if(!in) throw std::runtime_error("cannot open policy marker csv: "+path);
    std::string line; std::getline(in,line); std::vector<Marker> out;
    while(std::getline(in,line)) { if(line.empty()) continue; std::replace(line.begin(),line.end(),',',' '); std::istringstream s(line); int part; double x,y,z,nx,ny,nz,area;
        if(!(s>>part>>x>>y>>z>>nx>>ny>>nz>>area)||part<0||part>2||area<=0) throw std::runtime_error("invalid policy marker row: "+line);
        out.push_back({{Real(x),Real(y),Real(z)},{Real(nx),Real(ny),Real(nz)},Real(area),part});
    }
    if(out.empty()) throw std::runtime_error("policy marker csv is empty: "+path); return out;
}
