#pragma once
#include "numerics.hpp"
#include <memory>

// Experimental bounded conservative remap on complete vertical columns.
// NOT a six-neighbour FCT limiter and NOT an equation attributed to Lu.
// Each column preserves its phase integral. The cumulative defect defines
// a shared integrated z-face mass flux. Backward donor momentum transport
// with that flux is a tridiagonal M-matrix, solved directly (no Jacobi/Poisson).
// Only complete columns are supported; the x-y pencil decomposition owns them.
class ColumnPhaseRepair {
    Grid grid;
    Buffer<double> mapped,upper,rhs,diagnostic;
public:
    explicit ColumnPhaseRepair(Grid g):grid(g),mapped(g.n()),upper(g.n()),rhs(3*g.n()),diagnostic(5) {
        if(g.global_k0!=0||g.nz!=g.global_count(2))throw std::runtime_error("column repair requires complete z columns");
    }
    void launch(Real* state,Physics ph,int i0,int i1,int j0,int j1) {
        if(ph.clamp_density||ph.vertical_axis!=2||ph.periodic)
            throw std::runtime_error("column repair requires linear density, vertical z and nonperiodic boundaries");
        Grid g=grid;size_t N=g.n(),plane=size_t(g.nx)*g.ny;
        double* c=mapped.data();double* up=upper.data();double* d=rhs.data();double* report=diagnostic.data();
        diagnostic.zero();int ni=i1-i0,nj=j1-j0;
        parallel_for(size_t(ni)*nj,[=] HD(size_t col){
            int i=int(col%ni)+i0,j=int(col/ni)+j0;size_t bottom=g.id(i,j,0);
            double target=0,height=0,clipped=0;bool active=false;
            for(int k=0;k<g.nz;++k){size_t q=bottom+size_t(k)*plane;double raw=state[4*N+q],h=g.width(2,k);
                if(!(raw==raw)||::fabs(raw)>1e20||!(ph.rho(Real(raw))>0)){atomic_plus(report,1.);return;}
                c[q]=raw<0?0:raw>1?1:raw;active|=raw<0||raw>1;
                target+=raw*h;clipped+=c[q]*h;height+=h;
            }
            if(!active)return; // Exact identity on every already-bounded column.
            atomic_plus(report+1,1.);
            if(target<0||target>height){atomic_plus(report,1.);return;}
            // Interface-weighted redistribution. The initial neighbouring
            // phases define capacity; homogeneous regions are not seeded.
            bool converged=false;
            for(int iteration=0;iteration<32;++iteration){
                double defect=target-clipped;
                if(::fabs(defect)<=1e-12*height){converged=true;break;}
                double capacity=0;
                for(int k=0;k<g.nz;++k){size_t q=bottom+size_t(k)*plane;
                    double mean=(double(bounded(state[4*N+q]))+bounded(state[4*N+bottom+size_t(k>0?k-1:k)*plane])+bounded(state[4*N+bottom+size_t(k+1<g.nz?k+1:k)*plane]))/3;
                    capacity+=4*mean*(1-mean)*(defect>0?1-c[q]:c[q])*g.width(2,k);
                }
                if(!(capacity>0)){atomic_plus(report,1.);return;}
                double alpha=::fabs(defect)/capacity;alpha=alpha>1?1:alpha;clipped=0;
                for(int k=0;k<g.nz;++k){size_t q=bottom+size_t(k)*plane;
                    double mean=(double(bounded(state[4*N+q]))+bounded(state[4*N+bottom+size_t(k>0?k-1:k)*plane])+bounded(state[4*N+bottom+size_t(k+1<g.nz?k+1:k)*plane]))/3;
                    c[q]+=(defect>0?1:-1)*alpha*4*mean*(1-mean)*(defect>0?1-c[q]:c[q]);
                    clipped+=c[q]*g.width(2,k);
                }
            }
            if(!converged){atomic_plus(report,1.);return;}
            // Store representable phase values, then reduce rounding residual
            // using the cell with the smallest admissible value (finest ULP).
            double residual=target;
            for(int k=0;k<g.nz;++k){size_t q=bottom+size_t(k)*plane;c[q]=Real(c[q]);residual-=c[q]*g.width(2,k);}
            for(int pass=0;pass<4&&::fabs(residual)>1e-13*height;++pass){
                int selected=-1;double smallest=2;
                for(int k=0;k<g.nz;++k){size_t q=bottom+size_t(k)*plane;double v=c[q]+residual/g.width(2,k);
                    if(c[q]>0&&c[q]<1&&v>=0&&v<=1&&c[q]<smallest){selected=k;smallest=c[q];}}
                if(selected<0)break;size_t q=bottom+size_t(selected)*plane;double next=Real(c[q]+residual/g.width(2,selected));
                residual-=(next-c[q])*g.width(2,selected);c[q]=next;
            }
            atomic_plus(report+2,residual*g.width(0,i)*g.width(1,j));
            atomic_plus(report+3,::fabs(residual)*g.width(0,i)*g.width(1,j));
            // Integrated face phase flux G(k+1/2)=sum_{l<=k} h_l(raw-c).
            // Momentum uses EXACTLY (rho_l-rho_g)*G. At both physical
            // boundaries correction flux is zero (up to reported rounding).
            double fm=0,previous_upper=0,previous_rhs[3]={0,0,0};
            for(int k=0;k<g.nz;++k){size_t q=bottom+size_t(k)*plane;double h=g.width(2,k);
                double fp=fm+double(ph.rho1-ph.rho0)*(double(state[4*N+q])-c[q])*h;
                if(k==g.nz-1)fp=0;
                double lower=-(fm>0?fm:0)/h,high=(fp<0?fp:0)/h;
                double diagonal=double(ph.rho(Real(c[q])))+((fp>0?fp:0)-(fm<0?fm:0))/h;
                double pivot=diagonal-lower*previous_upper;
                if(!(pivot>0)){atomic_plus(report,1.);return;}
                up[q]=high/pivot;previous_upper=up[q];
                for(int a=0;a<3;++a){double v=(double(state[size_t(a+1)*N+q])-lower*previous_rhs[a])/pivot;
                    d[size_t(a)*N+q]=v;previous_rhs[a]=v;}
                fm=fp;
            }
            double next[3]={0,0,0};
            for(int k=g.nz-1;k>=0;--k){size_t q=bottom+size_t(k)*plane;Real rho=ph.rho(Real(c[q]));
                for(int a=0;a<3;++a){next[a]=d[size_t(a)*N+q]-up[q]*next[a];state[size_t(a+1)*N+q]=Real(rho*next[a]);}
                state[4*N+q]=Real(c[q]);
            }
        });
    }
    std::vector<double> check() {
        auto result=diagnostic.download();
        if(result[0])throw std::runtime_error("column phase repair: infeasible/nonfinite columns="+std::to_string(result[0]));
        return result;
    }
};
