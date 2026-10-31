// Same face stress as metric_stress_face, all components share its geometry.
HD inline Vec metric_stress_vector(Field f,const Real* mu,int i,int j,int k,int normal){
 int ri=i+(normal==0),rj=j+(normal==1),rk=k+(normal==2),index=normal==0?i:normal==1?j:k;
 Real gradient[3],norm2=0,pl=f.value(4,i,j,k),pr=f.value(4,ri,rj,rk);
 for(int a=0;a<3;++a){gradient[a]=a==normal?(pr-pl)/(f.g.center(a,index+1)-f.g.center(a,index)):
  metric_face_scalar(metric_gradient(f,4,i,j,k,a),metric_gradient(f,4,ri,rj,rk,a),f.g,normal,index);norm2+=sq(gradient[a]);}
 Real ml=scalar_at(mu,f.g,f.p,i,j,k),mr=scalar_at(mu,f.g,f.p,ri,rj,rk);
 Real potential=metric_face_scalar(f.p.beta()*sq(pl*(1-pl))-pl*ml,f.p.beta()*sq(pr*(1-pr))-pr*mr,f.g,normal,index);
 Real iso=potential+f.p.kappa()*norm2/2,normal_part=f.p.kappa()*gradient[normal];
 return {(normal==0?iso:0)-normal_part*gradient[0],(normal==1?iso:0)-normal_part*gradient[1],(normal==2?iso:0)-normal_part*gradient[2]};
}
HD inline Vec metric_capillary_vector(Field f,const Real* mu,int i,int j,int k){
 if(!f.p.capillary_mode)return {metric_capillary_force(f,mu,i,j,k,0),metric_capillary_force(f,mu,i,j,k,1),metric_capillary_force(f,mu,i,j,k,2)};
 Vec sum{0,0,0};for(int a=0;a<3;++a){Vec pos=metric_stress_vector(f,mu,i,j,k,a),neg=metric_stress_vector(f,mu,i-(a==0),j-(a==1),k-(a==2),a);Real width=f.g.cell_width(a,i,j,k);
  sum.x+=(pos.x-neg.x)/width;sum.y+=(pos.y-neg.y)/width;sum.z+=(pos.z-neg.z)/width;
 }return sum;
}
