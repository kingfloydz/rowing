#pragma once
#include <memory>
#include <limits>

struct SurfaceVertex {float x,y,z;};
static_assert(sizeof(SurfaceVertex)==12,"surface vertex layout");

// Six conforming tetrahedra share the 0--7 cube diagonal. The same split on
// every cell makes shared face diagonals agree, including rank boundaries.
HD inline unsigned surface_cube(Grid g,const Real* phi,size_t index,
                                 SurfaceVertex* out,double scale) {
    int i,j,k;g.xyz(index,i,j,k);
    Vec x[8];Real v[8];
    for(int b=0;b<8;++b){int ii=i+(b&1),jj=j+((b>>1)&1),kk=k+((b>>2)&1);
        x[b]=g.position(ii,jj,kk);v[b]=phi[g.id(ii,jj,kk)];}
    const int tet[6][4]={{0,1,3,7},{0,3,2,7},{0,2,6,7},{0,6,4,7},{0,4,5,7},{0,5,1,7}};
    unsigned count=0;
    for(int t=0;t<6;++t){
        int inside[4],outside[4],ni=0,no=0;
        for(int a=0;a<4;++a){int b=tet[t][a];if(v[b]>=Real(.5))inside[ni++]=b;else outside[no++]=b;}
        if(ni==0||ni==4)continue;
        int edges[4][2],num=0;
        if(ni==1||ni==3){int a=ni==1?inside[0]:outside[0];
            for(int b=0;b<3;++b){edges[b][0]=a;edges[b][1]=ni==1?outside[b]:inside[b];}num=3;
        }else{int e[4][2]={{inside[0],outside[0]},{inside[0],outside[1]},{inside[1],outside[1]},{inside[1],outside[0]}};
            for(int b=0;b<4;++b){edges[b][0]=e[b][0];edges[b][1]=e[b][1];}num=4;}
        Vec p[4];
        for(int b=0;b<num;++b){int a=edges[b][0],c=edges[b][1];Real q=(Real(.5)-v[a])/(v[c]-v[a]);
            p[b]={x[a].x+q*(x[c].x-x[a].x),x[a].y+q*(x[c].y-x[a].y),x[a].z+q*(x[c].z-x[a].z)};}
        // Orient normals from liquid (>= .5) toward gas.
        Vec direction{x[outside[0]].x-x[inside[0]].x,x[outside[0]].y-x[inside[0]].y,x[outside[0]].z-x[inside[0]].z};
        for(int b=1;b<num-1;++b){Vec a=p[0],c=p[b],d=p[b+1];
            Vec u{c.x-a.x,c.y-a.y,c.z-a.z},w{d.x-a.x,d.y-a.y,d.z-a.z};
            Vec n{u.y*w.z-u.z*w.y,u.z*w.x-u.x*w.z,u.x*w.y-u.y*w.x};
            if(n.x*n.x+n.y*n.y+n.z*n.z==0)continue;
            if(n.x*direction.x+n.y*direction.y+n.z*direction.z<0){Vec tmp=c;c=d;d=tmp;}
            if(out){out[3*count]={float(a.x/scale),float(a.y/scale),float(a.z/scale)};
                out[3*count+1]={float(c.x/scale),float(c.y/scale),float(c.z/scale)};
                out[3*count+2]={float(d.x/scale),float(d.y/scale),float(d.z/scale)};}
            ++count;
        }
    }
    return count;
}

HD inline unsigned surface_reserve(unsigned* counter,unsigned n){
#ifdef __CUDA_ARCH__
    return atomicAdd(counter,n);
#else
    unsigned old;
    #pragma omp critical(surface_reserve)
    {old=*counter;*counter+=n;}
    return old;
#endif
}

// Reused per rank. Host receives only compact triangles, never full state.
// Half-open owned boxes refer to cube lower corners; the positive
// neighbour is supplied by halo. This also owns pencil seam cubes once.
class SurfaceExtractor {
    std::unique_ptr<Buffer<unsigned>> active_,offset_;
    std::unique_ptr<Buffer<SurfaceVertex>> vertices_;
    Buffer<unsigned> counters_{2};
    size_t cells_=0,capacity_=0;
public:
    std::vector<SurfaceVertex> extract(Grid g,const Real* phi,double scale,int k0=0,int k1=-1,int i0=0,int i1=-1,int j0=0,int j1=-1){
        if(k1<0)k1=g.nz-1;
        k1=std::min(k1,g.nz-1);
        if(i1<0)i1=g.nx-1;if(j1<0)j1=g.ny-1;
        i1=std::min(i1,g.nx-1);j1=std::min(j1,g.ny-1);
        if(k0<0||k1<k0||i0<0||i1<i0||j0<0||j1<j0||!(scale>0))throw std::runtime_error("invalid surface ownership/scale");
        if(g.n()>std::numeric_limits<unsigned>::max()/36)throw std::runtime_error("surface index capacity exceeded");
        int ni=i1-i0,nj=j1-j0;
        size_t cells=size_t(ni)*nj*(k1-k0);
        if(cells>cells_){active_.reset(new Buffer<unsigned>(cells));offset_.reset(new Buffer<unsigned>(cells));cells_=cells;}
        if(!cells)return {};
        counters_.zero();unsigned* counts=counters_.data();unsigned* active=active_->data();unsigned* offset=offset_->data();
        parallel_for(cells,[=] HD(size_t q){int i=int(q%ni)+i0;q/=ni;int j=int(q%nj)+j0,k=int(q/nj)+k0;
            bool below=false,above=false;
            for(int b=0;b<8;++b){Real v=phi[g.id(i+(b&1),j+((b>>1)&1),k+((b>>2)&1))];below|=v<Real(.5);above|=v>=Real(.5);}
            if(below&&above)active[surface_reserve(counts,1)]=unsigned(g.id(i,j,k));
        });
        unsigned n=counters_.download()[0];
        parallel_for(n,[=] HD(size_t q){unsigned triangles=surface_cube(g,phi,active[q],nullptr,scale);offset[q]=surface_reserve(counts+1,triangles);});
        size_t total=counters_.download()[1],needed=3*total;
        if(needed>capacity_){vertices_.reset(new Buffer<SurfaceVertex>(needed));capacity_=needed;}
        if(!needed)return {};
        SurfaceVertex* dst=vertices_->data();
        parallel_for(n,[=] HD(size_t q){surface_cube(g,phi,active[q],dst+3*offset[q],scale);});
        std::vector<SurfaceVertex> result(needed);
#ifdef __CUDACC__
        cuda_check(cudaMemcpy(result.data(),dst,needed*sizeof(SurfaceVertex),cudaMemcpyDeviceToHost));
#else
        std::copy(dst,dst+needed,result.begin());
#endif
        return result;
    }
};
