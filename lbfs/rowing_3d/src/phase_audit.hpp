#pragma once
// Opt-in diagnostic. No state mutation; core ownership excludes stale outer halos.
inline void launch_phase_audit(Solver& s,bool predicted,Real dt=0,bool scan_cfl=false) {
 Field f=s.field_for(predicted?s.work.data():s.state.data());Grid g=s.g;
 size_t N=g.n();
 int i0=s.ib_partitioned?s.core_i0:0,j0=s.ib_partitioned?s.core_j0:0,k0=s.ib_partitioned?s.core_k0:0;
 int ni=(s.ib_partitioned&&s.core_i1>=0?s.core_i1:g.nx)-i0;
 int nj=(s.ib_partitioned&&s.core_j1>=0?s.core_j1:g.ny)-j0;
 int nk=(s.ib_partitioned&&s.core_k1>=0?s.core_k1:g.nz)-k0;
 size_t owned=size_t(ni)*nj*nk;
 double* red=s.reductions.data();const int P=int(s.reductions.size()/8);
 const Real* faces=s.faces.data();
 parallel_for(P,[=] HD(size_t t){double lo=1e100,hi=-1e100,li=-1,bi=-1,bad=0,um=-1,ui=-1,cflmax=scan_cfl?0:-1;
  for(size_t c=t;c<owned;c+=P){
   size_t q=g.id(int(c%ni)+i0,int((c/ni)%nj)+j0,int(c/(size_t(ni)*nj))+k0);bool finite=true;
   for(int v=0;v<5;++v){double x=f.raw(v,q);if(!(x==x)||::fabs(x)>1e30)finite=false;}
   double phi=f.raw(4,q),rho=f.p.rho(Real(phi));
   if(!finite||!(rho>0)){++bad;if(bi<0)bi=double(q);}
   if(phi<lo){lo=phi;li=double(q);}if(phi>hi)hi=phi;
   if(rho>0&&finite){double u=3*::sqrt(sq(double(f.raw(1,q)))+sq(double(f.raw(2,q)))+sq(double(f.raw(3,q))))/rho;
    if(u>um){um=u;ui=double(q);}}
   if(scan_cfl){int i,j,k;g.xyz(q,i,j,k);double cfl=0;
   for(int a=0;a<3;++a){int index=a==0?i:a==1?j:k;
    if(index==0)continue; // Boundary faces are not in this diagnostic cache.
    size_t m=g.id(i-(a==0),j-(a==1),k-(a==2));
    double up=faces[size_t(9*a+5+a)*N+q]-ale_face_velocity(f,i,j,k,a)[a];
    double un=faces[size_t(9*a+5+a)*N+m]-ale_face_velocity(f,i-(a==0),j-(a==1),k-(a==2),a)[a];
    cfl+=dt*((up>0?up:0)+(un<0?-un:0))/g.cell_width(a,i,j,k);
   }
   if(cfl>cflmax)cflmax=cfl;}
  }
  red[t]=lo;red[P+t]=hi;red[2*P+t]=li;red[3*P+t]=bi;red[4*P+t]=bad;
  red[5*P+t]=um;red[6*P+t]=ui;red[7*P+t]=cflmax;
 });
 // Finish the reduction and gather both selected cells on the device.
 // Previously each rank copied 32768 doubles and performed ten synchronous
 // scalar transfers. A 22-double packet retains the same ordering and tie rules.
 double* packet=s.audit_packet.data();
 parallel_for(1,[=] HD(size_t){
  double lo=1e100,hi=-1e100,li=-1,bi=-1,bad=0,um=-1,ui=-1,cfl=scan_cfl?0:-1;
  for(int t=0;t<P;++t){
   if(red[t]<lo){lo=red[t];li=red[2*P+t];}
   hi=hi<red[P+t]?red[P+t]:hi;bad+=red[4*P+t];
   if(red[3*P+t]>=0&&(bi<0||red[3*P+t]<bi))bi=red[3*P+t];
   if(red[5*P+t]>um){um=red[5*P+t];ui=red[6*P+t];}
   cfl=cfl<red[7*P+t]?red[7*P+t]:cfl;
  }
  size_t q=size_t(bi>=0?bi:li);int i,j,k;g.xyz(q,i,j,k);Vec x=g.position(i,j,k);
  packet[0]=lo;packet[1]=hi;packet[2]=bad;packet[3]=x.x;packet[4]=x.y;packet[5]=x.z;
  for(int v=0;v<5;++v)packet[6+v]=f.raw(v,q);
  packet[11]=f.p.rho(f.raw(4,q));packet[12]=um;packet[13]=cfl;
  q=size_t(ui>=0?ui:0);g.xyz(q,i,j,k);x=g.position(i,j,k);
  packet[14]=x.x;packet[15]=x.y;packet[16]=x.z;
  for(int v=0;v<5;++v)packet[17+v]=f.raw(v,q);
 });
}
inline bool collect_phase_audit(Solver& s,std::ostream& log,int step,int stage,int rank,
                               const char* label,bool predicted,double L,bool scan_cfl=false) {
 (void)predicted;(void)scan_cfl;
 auto p=s.audit_packet.download();
 log<<std::setprecision(12)<<step<<','<<stage<<','<<rank<<','<<label;
 for(int v=0;v<22;++v){double value=p[v];if((v>=3&&v<=5)||(v>=14&&v<=16))value/=L;log<<','<<value;}
 log<<'\n';log.flush();return p[2]>0;
}
