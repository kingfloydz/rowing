#pragma once
#include <algorithm>

// Shared by both drivers: scheduling uses seconds, all Solver hooks use
// lattice time. This small host-only value is computed once per step.
struct StepTime {
    double start_s, duration_s, end_s;
    Real lattice_dt;
};
inline StepTime step_time(int step,double dt_s,double end_s,double time_scale,double origin_s=0) {
    double start=origin_s+(step-1)*dt_s;
    double stop=std::min(origin_s+step*dt_s,end_s);
    return {start,stop-start,stop,Real((stop-start)*time_scale)};
}
