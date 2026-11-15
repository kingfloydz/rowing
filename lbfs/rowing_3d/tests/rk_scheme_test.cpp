#include "rk_scheme.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
template<class F>double step(double y,double t,double dt,F f){RkScheme rk{true};double base=y;for(int s=0;s<rk.stages();++s){double work=y+dt*rk.euler_fraction()*f(t+rk.abscissa(s)*dt,y);y=rk.base_weight(s)*base+(1-rk.base_weight(s))*work;}return y;}
int main(){
 RkScheme rk{true};double clock=0;for(int s=0;s<4;++s){if(std::abs(clock-rk.abscissa(s))>1e-14)throw std::runtime_error("nonautonomous stage time");clock=(1-rk.base_weight(s))*(clock+.5);}if(std::abs(clock-1)>1e-14)throw std::runtime_error("end time");
 double previous=0;for(int n:{20,40,80}){double y=1,h=1./n;for(int i=0;i<n;++i)y=step(y,i*h,h,[](double t,double u){return t+u;});double error=std::abs(y-(2*std::exp(1.)-2));if(previous&&previous/error<7.7)throw std::runtime_error("third order nonautonomous convergence");previous=error;std::cout<<"n="<<n<<" error="<<error<<'\n';}
 for(double z:{-.5,-1.,-2.986169215,-3.59560332}){double expected=1+z+z*z/2+z*z*z/6+z*z*z*z/48;double got=step(1,0,1,[=](double,double u){return z*u;});if(std::abs(got-expected)>1e-13)throw std::runtime_error("amplification polynomial");}
 double x=.9,y=.1,x0=x,y0=y;for(int s=0;s<4;++s){double dx=y-x,xx=x+dx,yy=y-dx;x=rk.base_weight(s)*x0+(1-rk.base_weight(s))*xx;y=rk.base_weight(s)*y0+(1-rk.base_weight(s))*yy;if(x<0||y<0||std::abs(x+y-1)>1e-14)throw std::runtime_error("SSP mass positivity");}
 std::cout<<"SSPRK43 stage times, order, polynomial, positive conservative exchange: PASS\n";
}
