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
