HD inline Real smagorinsky_viscosity(const Real gradient[3][3],Real h,Real cs);
HD inline Real metric_derivative(Real minus,Real center,Real plus,Real dm,Real dp){
 return dm/(dp*(dm+dp))*(plus-center)+dp/(dm*(dm+dp))*(center-minus);
}
HD inline Real metric_gradient(Field f,int v,int i,int j,int k,int a,bool primitive=false){
 if(f.gradient_cache&&(primitive||v==4))return f.gradient_cache[(size_t(v)*3+a)*f.g.n()+f.g.id(map_index(i,f.g.nx,f.p.periodic),map_index(j,f.g.ny,f.p.periodic),map_index(k,f.g.nz,f.p.periodic))];
 int index=a==0?i:a==1?j:k;Real dm=f.g.center(a,index)-f.g.center(a,index-1),dp=f.g.center(a,index+1)-f.g.center(a,index);
 return metric_derivative(f.shifted(v,i,j,k,a,-1,primitive),primitive?f.primitive(v,i,j,k):f.value(v,i,j,k),f.shifted(v,i,j,k,a,1,primitive),dm,dp);
}
HD inline Real metric_face_scalar(Real l,Real r,Grid g,int a,int index){
 Real wl=g.width(a,index),wr=g.width(a,index+1);return (wr*l+wl*r)/(wl+wr);
}
HD inline Real metric_weno(Field f,int i,int j,int k,int a,bool positive){
 int index=a==0?i:a==1?j:k,donor=positive?index:index+1,sign=positive?1:-1;
 const Real* c=f.g.metric()+f.g.weno_offset[a]+60*(index+1)+(positive?0:30);
 Real values[5];for(int n=0;n<5;++n){int idx=donor+sign*(n-2);values[n]=f.value(4,a==0?idx:i,a==1?idx:j,a==2?idx:k);}
 Real numerator=0,denominator=0;
 for(int s=0;s<3;++s){Real p=0,d1=0,d2=0;for(int n=0;n<3;++n){p+=c[3+9*s+n]*values[s+n];d1+=c[6+9*s+n]*values[s+n];d2+=c[9+9*s+n]*values[s+n];}
  Real beta=d1*d1+Real(13./3)*d2*d2,alpha=c[s]/sq(Real(1e-6)+beta);numerator+=alpha*p;denominator+=alpha;
 }return numerator/denominator;
}
HD inline Real metric_laplacian(const Real* a,Grid g,Physics p,int i,int j,int k){
 Real c=scalar_at(a,g,p,i,j,k),result=0;
 for(int axis=0;axis<3;++axis){int index=axis==0?i:axis==1?j:k;
  Real plus=scalar_at(a,g,p,i+(axis==0),j+(axis==1),k+(axis==2));
  Real minus=scalar_at(a,g,p,i-(axis==0),j-(axis==1),k-(axis==2));
  Real dp=g.center(axis,index+1)-g.center(axis,index),dm=g.center(axis,index)-g.center(axis,index-1);
  result+=((plus-c)/dp-(c-minus)/dm)/g.width(axis,index);
 }return result;
}
HD inline Real metric_stress_face(Field f,const Real* mu,int i,int j,int k,int normal,int component){
 int ri=i+(normal==0),rj=j+(normal==1),rk=k+(normal==2),index=normal==0?i:normal==1?j:k;
 Real gradient[3],norm2=0,pl=f.value(4,i,j,k),pr=f.value(4,ri,rj,rk);
 for(int a=0;a<3;++a){gradient[a]=a==normal?(pr-pl)/(f.g.center(a,index+1)-f.g.center(a,index)):
  metric_face_scalar(metric_gradient(f,4,i,j,k,a),metric_gradient(f,4,ri,rj,rk,a),f.g,normal,index);norm2+=sq(gradient[a]);}
 Real ml=scalar_at(mu,f.g,f.p,i,j,k),mr=scalar_at(mu,f.g,f.p,ri,rj,rk);
 Real potential=metric_face_scalar(f.p.beta()*sq(pl*(1-pl))-pl*ml,f.p.beta()*sq(pr*(1-pr))-pr*mr,f.g,normal,index);
 return (component==normal?potential+f.p.kappa()*norm2/2:0)-f.p.kappa()*gradient[normal]*gradient[component];
}
HD inline Real metric_capillary_force(Field f,const Real* mu,int i,int j,int k,int component){
 if(!f.p.capillary_mode){int index=component==0?i:component==1?j:k;
  Real dm=f.g.center(component,index)-f.g.center(component,index-1),dp=f.g.center(component,index+1)-f.g.center(component,index);
  return -f.value(4,i,j,k)*metric_derivative(scalar_at(mu,f.g,f.p,i-(component==0),j-(component==1),k-(component==2)),scalar_at(mu,f.g,f.p,i,j,k),scalar_at(mu,f.g,f.p,i+(component==0),j+(component==1),k+(component==2)),dm,dp);
 }
 Real sum=0;for(int a=0;a<3;++a)sum+=(metric_stress_face(f,mu,i,j,k,a,component)-metric_stress_face(f,mu,i-(a==0),j-(a==1),k-(a==2),a,component))/f.g.cell_width(a,i,j,k);
 return sum;
}
HD inline void metric_lbfs_face(Field f,int i,int j,int k,int axis,Real out[4],Vec* velocity){
 int ri=i+(axis==0),rj=j+(axis==1),rk=k+(axis==2),index=axis==0?i:axis==1?j:k;
 Real wl=f.g.width(axis,index),wr=f.g.width(axis,index+1),delta=wl<wr?wl:wr;
 #pragma unroll
 for(int a=0;a<3;++a){Real width=f.g.cell_width(a,i,j,k);delta=width<delta?width:delta;}delta/=2;
 Real left[5],right[5],gl[5][3],gr[5][3];
 #pragma unroll
 for(int v=0;v<5;++v){left[v]=f.primitive(v,i,j,k);right[v]=f.primitive(v,ri,rj,rk);
  #pragma unroll
  for(int a=0;a<3;++a){gl[v][a]=metric_gradient(f,v,i,j,k,a,true);gr[v][a]=metric_gradient(f,v,ri,rj,rk,a,true);}}
 Real rl=f.p.rho(f.value(4,i,j,k)),rr=f.p.rho(f.value(4,ri,rj,rk));
 Real rf=metric_face_scalar(rl,rr,f.g,axis,index),phase=metric_face_scalar(bounded(f.value(4,i,j,k)),bounded(f.value(4,ri,rj,rk)),f.g,axis,index);
 Real departure[15],pf=0;Vec uf{0,0,0};
 #pragma unroll
 for(int q=0;q<15;++q){Vec e=e15(q);Real v[5];
  #pragma unroll
  for(int c=0;c<5;++c){Real vl=left[c],vr=right[c];
   #pragma unroll
   for(int a=0;a<3;++a){vl+=gl[c][a]*((a==axis?wl/2:0)-delta*e[a]);vr+=gr[c][a]*((a==axis?-wr/2:0)-delta*e[a]);}
   // Reconstructed states share one departure point: original Flux.cpp average.
   v[c]=e[axis]>0?vl:e[axis]<0?vr:(vl+vr)/2;}
  Real rd=f.p.macro_face_density?rf:f.p.rho(v[4]),fe=equilibrium(q,v[0],rd,{v[1],v[2],v[3]});
  departure[q]=equilibrium(q,v[0],rf,{v[1],v[2],v[3]});pf+=fe;uf.x+=fe*e.x;uf.y+=fe*e.y;uf.z+=fe*e.z;
 }
 uf.x*=3/rf;uf.y*=3/rf;uf.z*=3/rf;if(velocity)*velocity=uf;
 Real viscosity=f.p.nu0+(f.p.nu1-f.p.nu0)*phase;
 if(f.p.smagorinsky_cs>0){Real gradient[3][3];
  #pragma unroll
  for(int a=0;a<3;++a){
   #pragma unroll
   for(int b=0;b<3;++b)gradient[a][b]=metric_face_scalar(gl[a+1][b],gr[a+1][b],f.g,axis,index);}
  // LES filter: local volume-equivalent width, independent of streaming delta.
  Real width=::cbrt(f.g.volume(i,j,k));viscosity+=smagorinsky_viscosity(gradient,width,f.p.smagorinsky_cs);}
 Real tau_minus_half=3*viscosity/delta;Vec wm=ale_face_velocity(f,i,j,k,axis);
 Real un=uf[axis]-wm[axis];out[0]=rf*un/3;out[1]=out[2]=out[3]=0;
 #pragma unroll
 for(int q=0;q<15;++q){Vec e=e15(q);Real eq=equilibrium(q,pf,rf,uf),hat=eq-tau_minus_half*(eq-departure[q]);out[1]+=e[axis]*e.x*hat;out[2]+=e[axis]*e.y*hat;out[3]+=e[axis]*e.z*hat;}
 out[1]-=wm[axis]*rf*uf.x/3;out[2]-=wm[axis]*rf*uf.y/3;out[3]-=wm[axis]*rf*uf.z/3;
}
