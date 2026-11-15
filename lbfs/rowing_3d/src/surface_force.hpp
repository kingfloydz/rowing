#pragma once
// Exterior Cauchy traction. n points OUT of solid into fluid.
// This measures surface load directly: no fictitious-fluid inertia term is
// added to a stress integral. IB reaction is a different diagnostic.
struct SurfaceTraction { Vec pressure{0,0,0},viscous{0,0,0},capillary{0,0,0}; };
HD inline SurfaceTraction surface_cell_traction(Field f,int i,int j,int k,Vec n){
    SurfaceTraction t;
    Real phi=f.value(4,i,j,k),rho=f.p.rho(phi),pressure=f.value(0,i,j,k);
    Real grad[3][3],gp[3],norm2=0;
    for(int a=0;a<3;++a){gp[a]=metric_gradient(f,4,i,j,k,a);norm2+=gp[a]*gp[a];
        for(int b=0;b<3;++b)grad[a][b]=metric_gradient(f,a+1,i,j,k,b,true);}
    Real nu=f.p.nu0+(f.p.nu1-f.p.nu0)*bounded(phi);
    if(f.p.smagorinsky_cs>0)nu+=smagorinsky_viscosity(grad,::cbrt(f.g.volume(i,j,k)),f.p.smagorinsky_cs);
    Real lap=laplacian(f.a+4*f.g.n(),f.g,f.p,i,j,k);
    Real chemical=2*f.p.beta()*phi*(phi-1)*(2*phi-1)-f.p.kappa()*lap;
    // Same pressure gauge as the conservative capillary stress in the solver:
    // K=(f_bulk-phi*mu+kappa|grad(phi)|^2/2)I-kappa grad(phi) tensor grad(phi).
    Real iso=f.p.beta()*sq(phi*(1-phi))-phi*chemical+f.p.kappa()*norm2/2;
    Real gn=gp[0]*n.x+gp[1]*n.y+gp[2]*n.z;
    Real visc[3]={0,0,0},cap[3];
    for(int a=0;a<3;++a){for(int b=0;b<3;++b)visc[a]+=rho*nu*(grad[a][b]+grad[b][a])*n[b];cap[a]=iso*n[a]-f.p.kappa()*gp[a]*gn;}
    t.pressure={-pressure*n.x,-pressure*n.y,-pressure*n.z};
    t.viscous={visc[0],visc[1],visc[2]};t.capillary={cap[0],cap[1],cap[2]};return t;
}
HD inline void traction_add(SurfaceTraction& a,const SurfaceTraction& b,Real w){
    a.pressure.x+=w*b.pressure.x;a.pressure.y+=w*b.pressure.y;a.pressure.z+=w*b.pressure.z;
    a.viscous.x+=w*b.viscous.x;a.viscous.y+=w*b.viscous.y;a.viscous.z+=w*b.viscous.z;
    a.capillary.x+=w*b.capillary.x;a.capillary.y+=w*b.capillary.y;a.capillary.z+=w*b.capillary.z;
}
HD inline bool surface_interpolate_traction(Field f,Vec x,Vec n,SurfaceTraction& t){
    int b[3];Real q[3];
    for(int a=0;a<3;++a){b[a]=f.g.lower_center(a,x[a]);if(b[a]<1||b[a]+2>=f.g.count(a))return false;
        q[a]=(x[a]-f.g.center(a,b[a]))/(f.g.center(a,b[a]+1)-f.g.center(a,b[a]));}
    for(int k=0;k<2;++k)for(int j=0;j<2;++j)for(int i=0;i<2;++i)
        traction_add(t,surface_cell_traction(f,b[0]+i,b[1]+j,b[2]+k,n),(i?q[0]:1-q[0])*(j?q[1]:1-q[1])*(k?q[2]:1-q[2]));
    return true;
}
// Probe spacing is defined at the marker on the GLOBAL mesh. It must not
// depend on the rank selected to evaluate either probe.
HD inline bool surface_probe_locations(Grid g,Marker m,Vec& x2,Vec& x3){
    Real h=0;
    for(int a=0;a<3;++a){int b=g.lower_center(a,m.x[a]);if(b<0)return false;Real w=g.width(a,b);h=w>h?w:h;}
    x2={m.x.x+2*h*m.n.x,m.x.y+2*h*m.n.y,m.x.z+2*h*m.n.z};
    x3={m.x.x+3*h*m.n.x,m.x.y+3*h*m.n.y,m.x.z+3*h*m.n.z};return true;
}
HD inline bool surface_inside_box(Grid g,Vec x,const int* lo,const int* hi){
    for(int a=0;a<3;++a)if(x[a]<g.face(a,lo[a])||x[a]>=g.face(a,hi[a]))return false;
    return true;
}
HD inline bool surface_probe_covered(Grid g,Vec x){
    for(int a=0;a<3;++a){int b=g.lower_center(a,x[a]);if(b<1||b+2>=g.count(a))return false;}
    return true;
}
HD inline bool surface_exterior_traction(Field f,Marker m,SurfaceTraction& t){
    Vec x2,x3;if(!surface_probe_locations(f.g,m,x2,x3))return false;
    SurfaceTraction a,b;
    if(!surface_interpolate_traction(f,x2,m.n,a)||!surface_interpolate_traction(f,x3,m.n,b))return false;
    traction_add(t,a,3);traction_add(t,b,-2);return true;
}
