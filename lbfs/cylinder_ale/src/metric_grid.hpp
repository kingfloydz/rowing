// Included after Real and Vec. Tensor-product finite-volume geometry.
struct Grid {
    int nx,ny,nz;Real h;
    const Real* metric_host=nullptr;const Real* metric_device=nullptr;
    int face_offset[3]={0,0,0},weno_offset[3]={0,0,0};
    HD bool stretched()const{return metric_host!=nullptr;}
    HD const Real* metric()const{
#ifdef __CUDA_ARCH__
        return metric_device;
#else
        return metric_host;
#endif
    }
    HD int count(int a)const{return a==0?nx:a==1?ny:nz;}
    HD size_t n()const{return size_t(nx)*ny*nz;}
    HD size_t id(int i,int j,int k)const{return (size_t(k)*ny+j)*nx+i;}
    HD void xyz(size_t s,int& i,int& j,int& k)const{i=int(s%nx);s/=nx;j=int(s%ny);k=int(s/ny);}
    HD Real face(int a,int i)const{
        if(!stretched())return i*h;
        const Real* f=metric()+face_offset[a];int nn=count(a);
        if(i<0)return 2*f[0]-f[-i];
        if(i>nn)return 2*f[nn]-f[2*nn-i];
        return f[i];
    }
    HD Real width(int a,int i)const{return face(a,i+1)-face(a,i);}
    HD Real center(int a,int i)const{return (face(a,i)+face(a,i+1))/2;}
    HD Real extent(int a)const{return face(a,count(a));}
    HD Real cell_width(int a,int i,int j,int k)const{return width(a,a==0?i:a==1?j:k);}
    HD Real volume(int i,int j,int k)const{return width(0,i)*width(1,j)*width(2,k);}
    HD Vec position(int i,int j,int k)const{return {center(0,i),center(1,j),center(2,k)};}
    HD int lower_center(int a,Real x)const{
        int lo=0,hi=count(a)-1;
        if(!(x>=center(a,1)&&x<center(a,hi-1)))return -2;
        while(lo+1<hi){int mid=(lo+hi)/2;if(center(a,mid)<=x)lo=mid;else hi=mid;}
        return lo;
    }
};
