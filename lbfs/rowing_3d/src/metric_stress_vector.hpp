// Same face stress as metric_stress_face, all components share its geometry.
template<bool CachedGradients=false>
HD inline Vec metric_stress_vector(Field f,const Real* mu,int i,int j,int k,int normal){
 int ri=i+(normal==0),rj=j+(normal==1),rk=k+(normal==2),index=normal==0?i:normal==1?j:k;
 Real gradient[3],norm2=0,pl=f.value(4,i,j,k),pr=f.value(4,ri,rj,rk);
 size_t left=0,right=0;
 if constexpr(CachedGradients){
  left=f.g.id(map_index(i,f.g.nx,f.p.periodic),map_index(j,f.g.ny,f.p.periodic),map_index(k,f.g.nz,f.p.periodic));
  right=f.g.id(map_index(ri,f.g.nx,f.p.periodic),map_index(rj,f.g.ny,f.p.periodic),map_index(rk,f.g.nz,f.p.periodic));
 }
 for(int a=0;a<3;++a){
  if(a==normal)gradient[a]=(pr-pl)/(f.g.center(a,index+1)-f.g.center(a,index));
  else if constexpr(CachedGradients){size_t offset=size_t(12+a)*f.g.n();
   gradient[a]=metric_face_scalar(f.gradient_cache[offset+left],f.gradient_cache[offset+right],f.g,normal,index);
  }else gradient[a]=metric_face_scalar(metric_gradient(f,4,i,j,k,a),metric_gradient(f,4,ri,rj,rk,a),f.g,normal,index);
  norm2+=sq(gradient[a]);
 }
 Real ml=scalar_at(mu,f.g,f.p,i,j,k),mr=scalar_at(mu,f.g,f.p,ri,rj,rk);
 Real potential=metric_face_scalar(f.p.beta()*sq(pl*(1-pl))-pl*ml,f.p.beta()*sq(pr*(1-pr))-pr*mr,f.g,normal,index);
 Real iso=potential+f.p.kappa()*norm2/2,normal_part=f.p.kappa()*gradient[normal];
 return {(normal==0?iso:0)-normal_part*gradient[0],(normal==1?iso:0)-normal_part*gradient[1],(normal==2?iso:0)-normal_part*gradient[2]};
}
template<bool CachedStress=false>
HD inline Vec metric_capillary_vector(Field f,const Real* mu,int i,int j,int k){
 if constexpr(CachedStress)if(f.capillary_face_cache){
  size_t N=f.g.n(),s=f.g.id(i,j,k);Vec sum{0,0,0};
  for(int a=0;a<3;++a){const Real* cache=f.capillary_face_cache+size_t(3*a)*N;
   Vec pos{cache[s],cache[N+s],cache[2*N+s]},neg;
   int index=a==0?i:a==1?j:k;
   if(index>0){size_t t=f.g.id(i-(a==0),j-(a==1),k-(a==2));neg={cache[t],cache[N+t],cache[2*N+t]};}
   else neg=metric_stress_vector<true>(f,mu,i-(a==0),j-(a==1),k-(a==2),a);
   Real width=f.g.cell_width(a,i,j,k);
   sum.x+=(pos.x-neg.x)/width;sum.y+=(pos.y-neg.y)/width;sum.z+=(pos.z-neg.z)/width;
  }
  return sum;
 }
 if constexpr(!CachedStress)if(!f.p.capillary_mode)return {metric_capillary_force(f,mu,i,j,k,0),metric_capillary_force(f,mu,i,j,k,1),metric_capillary_force(f,mu,i,j,k,2)};
 Vec sum{0,0,0};for(int a=0;a<3;++a){Vec pos=metric_stress_vector<CachedStress>(f,mu,i,j,k,a),neg=metric_stress_vector<CachedStress>(f,mu,i-(a==0),j-(a==1),k-(a==2),a);Real width=f.g.cell_width(a,i,j,k);
  sum.x+=(pos.x-neg.x)/width;sum.y+=(pos.y-neg.y)/width;sum.z+=(pos.z-neg.z)/width;
 }return sum;
}
