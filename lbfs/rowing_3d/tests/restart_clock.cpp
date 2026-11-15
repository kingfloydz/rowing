#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using Real=float;
#include "step_time.hpp"
int main(){
 for(double start:{0.,3.09,3.76515,3.90015}){
  double dt=1e-4,scale=40000,end=start+.135;
  int first=int(std::llround(start/dt));double origin=start-first*dt;
  if(std::abs(origin)<1e-10)origin=0;
  double previous=start;
  for(int step=first+1;step<=first+1350;++step){
   auto t=step_time(step,dt,end,scale,origin);
   if(std::abs(t.start_s-previous)>1e-12||std::abs(t.duration_s-dt)>1e-12||t.lattice_dt!=4.f)throw std::runtime_error("restart changed time or step duration");
   previous=t.end_s;
  }
  if(std::abs(previous-end)>1e-12)throw std::runtime_error("wrong endpoint");
 }
 std::cout<<"PASS fixed 100 us steps from aligned and unaligned real checkpoint timestamps\n";
}
