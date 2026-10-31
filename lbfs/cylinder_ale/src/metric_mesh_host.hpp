#pragma once
#include <array>
#include <memory>
// Exact cell-average polynomial reconstructions, precomputed per face.
inline std::vector<double> metric_solve(std::vector<std::vector<double>> a,std::vector<double> b){
 int n=int(b.size());for(int k=0;k<n;++k){int pivot=k;for(int j=k+1;j<n;++j)if(std::abs(a[j][k])>std::abs(a[pivot][k]))pivot=j;
  if(std::abs(a[pivot][k])<1e-20)throw std::runtime_error("singular metric polynomial");
  std::swap(a[pivot],a[k]);std::swap(b[pivot],b[k]);double d=a[k][k];for(int j=k;j<n;++j)a[k][j]/=d;b[k]/=d;
  for(int i=0;i<n;++i)if(i!=k){double q=a[i][k];for(int j=k;j<n;++j)a[i][j]-=q*a[k][j];b[i]-=q*b[k];}
 }return b;
}
inline std::vector<double> metric_tail(double length,double h,double growth,double maxratio){
 if(length<1e-10*h)return {};
 int n=1;auto sum=[&](int count,double q){double s=0,w=h;for(int k=0;k<count;++k){w=std::min(w*q,h*maxratio);s+=w;}return s;};
 while(sum(n,growth)<length-1e-12*h)++n;
 if(n*h>length+1e-8*h)throw std::runtime_error("fine region leaves less than one fine cell in tail");
 double lo=1,hi=growth;for(int k=0;k<70;++k){double m=(lo+hi)/2;if(sum(n,m)>length)hi=m;else lo=m;}
 std::vector<double> widths;double w=h;for(int k=0;k<n;++k){w=std::min(w*(lo+hi)/2,h*maxratio);widths.push_back(w);}return widths;
}
inline std::vector<Real> metric_axis(double extent,double h,double lo,double hi,double growth,double maxratio){
 if(!(std::isfinite(extent)&&std::isfinite(h)&&h>0&&extent/h<=100000&&0<=lo&&lo<hi&&hi<=extent&&growth>=1&&growth<=1.2&&std::isfinite(maxratio)&&maxratio>=1))throw std::runtime_error("invalid stretched grid bounds/growth");
 double n=(hi-lo)/h;if(std::abs(n-std::round(n))>1e-6)throw std::runtime_error("fine-region width must be integer multiples of h");
 auto left=metric_tail(lo,h,growth,maxratio),right=metric_tail(extent-hi,h,growth,maxratio);std::reverse(left.begin(),left.end());
 std::vector<Real> edges{0};double x=0;for(auto w:left){x+=w;edges.push_back(Real(x));}
 for(int i=1;i<=int(std::round(n));++i)edges.push_back(Real(lo+i*h));
 x=hi;for(auto w:right){x+=w;edges.push_back(Real(x));}edges.back()=Real(extent);
 if(edges.size()<5)throw std::runtime_error("metric WENO requires at least four cells per axis");
 for(size_t i=1;i<edges.size();++i)if(edges[i]<=edges[i-1])throw std::runtime_error("nonmonotone mesh");
 return edges;
}
struct MetricMesh {
 std::vector<Real> storage;std::unique_ptr<Buffer<Real>> device;Grid description;
 std::array<std::vector<Real>,3> axes;
 MetricMesh(double h,const std::array<double,3>& extent,const std::array<double,3>& lo,const std::array<double,3>& hi,double growth,double maxratio):description{0,0,0,Real(h)}{
  for(int a=0;a<3;++a){axes[a]=metric_axis(extent[a],h,lo[a],hi[a],growth,maxratio);description.face_offset[a]=int(storage.size());storage.insert(storage.end(),axes[a].begin(),axes[a].end());}
  description.nx=int(axes[0].size())-1;description.ny=int(axes[1].size())-1;description.nz=int(axes[2].size())-1;
  // Use the rounded Real face coordinates which the runtime actually uses.
  for(int a=0;a<3;++a){description.weno_offset[a]=int(storage.size());const auto& edges=axes[a];int nn=int(edges.size())-1;
   auto face=[&](int i)->double{if(i<0)return -double(edges[-i]);if(i>nn)return 2*double(edges[nn])-edges[2*nn-i];return edges[i];};
   for(int fi=0;fi<=nn;++fi)for(int direction:{1,-1}){
    int donor=direction==1?fi-1:fi;double center=.5*(face(donor)+face(donor+1)),width=face(donor+1)-face(donor),target=(face(fi)-center)/width;
    auto matrix=[&](const std::vector<int>& ids){int m=int(ids.size());std::vector<std::vector<double>> mat(m,std::vector<double>(m));
     for(int r=0;r<m;++r){double l=(face(ids[r])-center)/width,u=(face(ids[r]+1)-center)/width;
      for(int c=0;c<m;++c)mat[r][c]=(std::pow(u,c+1)-std::pow(l,c+1))/((c+1)*(u-l));}return mat;};
    std::vector<int> ids(5);for(int k=0;k<5;++k)ids[k]=donor+direction*(k-2);
    auto full=matrix(ids);std::vector<double> optimal(5);
    for(int c=0;c<5;++c){std::vector<double> e(5);e[c]=1;auto coeff=metric_solve(full,e);for(int j=0;j<5;++j)optimal[c]+=coeff[j]*std::pow(target,j);}
    double maps[3][3][3]={};
    for(int k=0;k<3;++k){std::vector<int> sub{ids[k],ids[k+1],ids[k+2]};auto mat=matrix(sub);
     for(int c=0;c<3;++c){std::vector<double> e(3);e[c]=1;auto coeff=metric_solve(mat,e);maps[k][0][c]=coeff[0]+target*coeff[1]+target*target*coeff[2];maps[k][1][c]=coeff[1];maps[k][2][c]=coeff[2];}}
    double weights[3]={optimal[0]/maps[0][0][0],0,optimal[4]/maps[2][0][2]};weights[1]=1-weights[0]-weights[2];
    for(int c=0;c<5;++c){double actual=0;for(int k=0;k<3;++k)if(c>=k&&c<k+3)actual+=weights[k]*maps[k][0][c-k];if(std::abs(actual-optimal[c])>1e-8)throw std::runtime_error("WENO optimal reconstruction mismatch");}
    for(auto w:weights){if(w<=0)throw std::runtime_error("nonpositive WENO linear weight on this mesh");storage.push_back(Real(w));}
    for(int k=0;k<3;++k)for(int derivative=0;derivative<3;++derivative)for(int c=0;c<3;++c)storage.push_back(Real(maps[k][derivative][c]));
   }
  }
  device.reset(new Buffer<Real>(storage.size()));device->upload(storage);description.metric_host=storage.data();description.metric_device=device->data();
 }
 Grid grid()const{return description;}
 void write(const std::string& output,double scale)const{std::ofstream f(output+"/mesh_faces.csv");f<<"axis,index,coordinate_m\n"<<std::setprecision(12);for(int a=0;a<3;++a)for(size_t i=0;i<axes[a].size();++i)f<<a<<','<<i<<','<<axes[a][i]/scale<<'\n';}
};
