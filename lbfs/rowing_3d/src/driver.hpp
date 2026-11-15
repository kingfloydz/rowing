#pragma once
#include "solver.hpp"
#include "step_time.hpp"
#include "metric_mesh_host.hpp"
#include <cerrno>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif
#include <iostream>
#include <map>
#include <sstream>
#include <set>
#include <cstdlib>
#include <fstream>
inline void make_directory(const std::string& path) {
    if(path.empty())throw std::runtime_error("empty output path");
    for(size_t i=1;i<=path.size();++i){
        if(i<path.size()&&path[i]!='/'&&path[i]!='\\')continue;
        auto part=path.substr(0,i);if(part.back()==':')continue;
#ifdef _WIN32
        int e=_mkdir(part.c_str());
#else
        int e=mkdir(part.c_str(),0775);
#endif
        if(e!=0&&errno!=EEXIST)throw std::runtime_error("cannot create output directory: "+part);
    }
}
inline std::vector<Real> read_axis_file(const std::string& path,double scale=1) {
    std::ifstream f(path);if(!f)throw std::runtime_error("cannot open mesh axis: "+path);
    std::vector<Real> a;double x;while(f>>x){if(!std::isfinite(x))throw std::runtime_error("nonfinite mesh axis");a.push_back(Real(x*scale));}
    return a;
}
struct Config {
    std::map<std::string,std::string> kv;
    explicit Config(const std::string& path){
        std::ifstream f(path);if(!f)throw std::runtime_error("cannot open case: "+path);
        std::string line;
        while(std::getline(f,line)){
            line=line.substr(0,line.find('#'));std::istringstream in(line);std::string key,val,extra;
            if(!(in>>key))continue;
            if(!(in>>val)||in>>extra)throw std::runtime_error("expected key value: "+line);
            if(!kv.emplace(key,val).second)throw std::runtime_error("duplicate key: "+key);
        }
    }
    std::string str(const std::string& k){auto it=kv.find(k);if(it==kv.end())throw std::runtime_error("missing key: "+k);auto s=it->second;kv.erase(it);return s;}
    double num(const std::string& k){auto s=str(k);size_t n;double v=std::stod(s,&n);if(n!=s.size()||!std::isfinite(v))throw std::runtime_error("invalid number: "+k);return v;}
    int integer(const std::string& k){double x=num(k);if(x!=std::floor(x)||x<0||x>2147483647.)throw std::runtime_error("invalid integer: "+k);return int(x);}
    void finish(){if(!kv.empty())throw std::runtime_error("unknown case key: "+kv.begin()->first);}
};
#include "multi_gpu_driver.hpp"
inline int run(int argc,char** argv) {
    std::string output;
    try {
        if(argc<2){std::cerr<<"Usage: lu2025 CASE [--output DIR] [--steps N] [--inspect] [--require-paper]\n";return 2;}
        std::string path=argv[1];output="run";int limit=-1;int multi_gpu=0;bool inspect=false,strict=false;std::string decomp="z";
        for(int i=2;i<argc;++i){std::string s=argv[i];
            if(s=="--output"&&i+1<argc)output=argv[++i];
            else if(s=="--steps"&&i+1<argc){std::string v=argv[++i];size_t n;limit=std::stoi(v,&n);if(n!=v.size()||limit<1)throw std::runtime_error("invalid step limit");}
            else if(s=="--inspect")inspect=true;
            else if(s=="--multi-gpu"&&i+1<argc){multi_gpu=std::stoi(argv[++i]);if(multi_gpu<2||multi_gpu>16)throw std::runtime_error("invalid multi-GPU count");}
            else if(s=="--decomp"&&i+1<argc){decomp=argv[++i];if(decomp!="z"&&decomp!="xy")throw std::runtime_error("decomp must be z or xy");}
            else if(s=="--require-paper")strict=true;
            else throw std::runtime_error("unknown option: "+s);
        }
        Config c(path);auto name=c.str("name"),status=c.str("validation_status"),closure=c.str("phase_ib"),inertia=c.str("interior_inertia"),sgs=c.str("subgrid_model");
        if(status!="research"||(closure!="yan2024_normal_flux"&&closure!="yan2022_solid_outward_flux")||(inertia!="rigid_gas"&&inertia!="measured_interior"&&inertia!="implicit_interior")||(sgs!="none"&&sgs!="smagorinsky"))throw std::runtime_error("unsupported model choice");
        if(strict)throw std::runtime_error("Paper reproduction blocked: Eq.23/24 interpretations, mobility, interface width, SGS and startup history are not author-confirmed. See reports/ALGORITHM.md.");
        double L=c.num("length_scale"),T=c.num("time_scale"),mass_scale=c.num("mass_scale");
        double h=c.num("h_m"),lx=c.num("lx_m"),ly=c.num("ly_m"),lz=c.num("lz_m");
        double dtsec=c.num("dt_s"),end=c.num("end_s");
        if(!(L>0&&T>0&&mass_scale>0&&h>0&&lx>0&&ly>0&&lz>0&&dtsec>0&&end>0))throw std::runtime_error("nonpositive scales or dimensions");
        auto cells=[&](double len){double n=len/h;if(std::fabs(n-std::round(n))>1e-5||n<4||n>100000)throw std::runtime_error("domain length must contain an integer number of >=4 cells");return int(std::round(n));};
        Grid g{4,4,4,Real(h*L)};
        std::unique_ptr<MetricMesh> metric_mesh;
        bool stretch=c.kv.count("stretched_grid")?c.integer("stretched_grid")!=0:false;
        bool axis_files=c.kv.count("mesh_axis_x")&&c.kv.count("mesh_axis_y")&&c.kv.count("mesh_axis_z");
        if(!stretch&&!axis_files)g=Grid{cells(lx),cells(ly),cells(lz),Real(h*L)};
        std::array<double,3> fine_lo{},fine_hi{};
        if(axis_files){
            if(c.kv.count("mesh_growth"))c.str("mesh_growth"); if(c.kv.count("mesh_max_ratio"))c.str("mesh_max_ratio");
            std::array<std::vector<Real>,3> axes={read_axis_file(c.str("mesh_axis_x"),L),read_axis_file(c.str("mesh_axis_y"),L),read_axis_file(c.str("mesh_axis_z"),L)};
            metric_mesh.reset(new MetricMesh(axes,h*L));g=metric_mesh->grid();stretch=true;
        } else if(stretch){
            const char* names[3]={"x","y","z"};
            for(int a=0;a<3;++a){fine_lo[a]=c.num(std::string("fine_")+names[a]+"_lo_m");fine_hi[a]=c.num(std::string("fine_")+names[a]+"_hi_m");}
            double growth=c.num("mesh_growth"),maxratio=c.num("mesh_max_ratio");
            metric_mesh.reset(new MetricMesh(h*L,{lx*L,ly*L,lz*L},{fine_lo[0]*L,fine_lo[1]*L,fine_lo[2]*L},{fine_hi[0]*L,fine_hi[1]*L,fine_hi[2]*L},growth,maxratio));g=metric_mesh->grid();
        }

        double density_scale=mass_scale/(L*L*L),pressure_scale=mass_scale/(L*T*T);
        Physics p;
        if(c.kv.count("rk_stages")&&c.integer("rk_stages")!=3)throw std::runtime_error("Only validated SSPRK(3,3) is available; the four-stage experiment is archived");
        p.rho0=Real(c.num("rho_g_kgm3")*density_scale);p.rho1=Real(c.num("rho_l_kgm3")*density_scale);
        p.nu0=Real(c.num("nu_g_m2s")*L*L/T);p.nu1=Real(c.num("nu_l_m2s")*L*L/T);
        p.sigma=Real(c.num("sigma_Nm")*mass_scale/(T*T));p.xi=Real(c.num("xi_m")*L);p.mobility=Real(c.num("mobility_lattice"));
        p.gravity=Real(c.num("gravity_ms2")*L/(T*T));p.level=Real(c.num("water_level_m")*L);p.periodic=c.integer("periodic");
        p.gravity_reconstruction=c.kv.count("gravity_reconstruction")?c.integer("gravity_reconstruction"):0;
        p.ale_pressure_transport=c.kv.count("ale_pressure_transport")?c.integer("ale_pressure_transport"):1;
        p.policy_ib_mass_weighted=c.kv.count("policy_ib_mass_weighted")?c.integer("policy_ib_mass_weighted"):0;
        p.capillary_reconstruction=c.kv.count("capillary_reconstruction")?c.integer("capillary_reconstruction"):0;
        p.transport_cfl_limit=Real(c.kv.count("transport_cfl_limit")?c.num("transport_cfl_limit"):0);
        if(p.transport_cfl_limit<0||p.transport_cfl_limit>1)throw std::runtime_error("transport_cfl_limit must be in [0,1]");
        p.mass_transport_bound=Real(c.kv.count("mass_transport_bound")?c.num("mass_transport_bound"):0);
        if(p.mass_transport_bound<0||p.mass_transport_bound>1||
           (p.mass_transport_bound>0&&p.transport_cfl_limit>0&&p.mass_transport_bound>=p.transport_cfl_limit))
            throw std::runtime_error("mass_transport_bound must be in [0,1] and below the enabled transport guard");
        p.vertical_axis=c.kv.count("vertical_axis")?c.integer("vertical_axis"):(c.kv.count("policy_motion_csv")?2:0);
        if(p.vertical_axis>2)throw std::runtime_error("vertical_axis must be 0, 1 or 2");
        p.pressure_product_rule=c.kv.count("pressure_product_rule")?c.integer("pressure_product_rule"):0;
        if(p.pressure_product_rule>1)throw std::runtime_error("invalid pressure_product_rule");
        p.macro_face_density=c.kv.count("macro_face_density")?c.integer("macro_face_density"):0;
        if(p.macro_face_density>1)throw std::runtime_error("invalid macro_face_density");
        p.harmonic_acoustic_pressure=c.kv.count("harmonic_acoustic_pressure")?c.integer("harmonic_acoustic_pressure"):0;
        if(p.harmonic_acoustic_pressure<0||p.harmonic_acoustic_pressure>1||(p.harmonic_acoustic_pressure&&!p.macro_face_density))
            throw std::runtime_error("harmonic_acoustic_pressure requires 0/1 and macro_face_density=1");
        p.transport_mode=c.kv.count("transport_mode")?c.integer("transport_mode"):0;
        p.clamp_density=c.kv.count("clamp_density")?c.integer("clamp_density"):1;
        p.capillary_mode=c.kv.count("capillary_mode")?c.integer("capillary_mode"):0;
        // CAD normals point out of the solid. The exterior fluid has the
        // opposite normal, so -delta(q.n_fluid) is +phi*delta(u).n_solid.
        // The old interpretation remains explicit for immutable controls.
        p.ib_phase_scale=Real(c.kv.count("ib_phase_scale")?c.num("ib_phase_scale"):
            closure=="yan2022_solid_outward_flux"?-1:1);
        p.ib_phase_bound=c.kv.count("ib_phase_bound")?c.integer("ib_phase_bound"):0;
        if(p.ib_phase_bound<0||p.ib_phase_bound>1)throw std::runtime_error("invalid IB phase projection");
        p.momentum_advection=c.kv.count("momentum_advection")?c.integer("momentum_advection"):0;
        if(p.momentum_advection<0||p.momentum_advection>2||(p.momentum_advection&&p.transport_mode<2))throw std::runtime_error("invalid consistent momentum reconstruction");
        p.cache_faces=c.kv.count("cache_faces")?c.integer("cache_faces"):0;
        if(p.cache_faces<0||p.cache_faces>1||(p.cache_faces&&p.transport_mode!=3))throw std::runtime_error("face cache requires FCT transport");
        p.top_pressure_outlet=c.kv.count("top_pressure_outlet")?c.integer("top_pressure_outlet"):0;
        p.horizontal_open_boundary=c.kv.count("horizontal_open_boundary")?c.integer("horizontal_open_boundary"):0;
        if(p.horizontal_open_boundary<0||p.horizontal_open_boundary>1||(p.horizontal_open_boundary&&p.periodic))throw std::runtime_error("invalid horizontal open boundary");
        if(p.top_pressure_outlet<0||p.top_pressure_outlet>1||(p.top_pressure_outlet&&p.periodic))throw std::runtime_error("invalid top pressure outlet");
        p.smagorinsky_cs=Real(c.kv.count("smagorinsky_cs")?c.num("smagorinsky_cs"):0);
        if(p.smagorinsky_cs<0||(sgs=="smagorinsky"&&p.smagorinsky_cs==0)||(sgs=="none"&&p.smagorinsky_cs!=0))throw std::runtime_error("SGS requires an explicit positive coefficient, or zero for none");
        if(p.capillary_mode<0||p.capillary_mode>1)throw std::runtime_error("invalid capillary discretization");
        if(p.transport_mode<0||p.transport_mode>3||p.clamp_density<0||p.clamp_density>1)throw std::runtime_error("invalid transport or density mapping");
        if(p.mass_transport_bound>0&&p.clamp_density)throw std::runtime_error("mass transport budget requires the linear, unclamped density law");
        Body b;
        b.cx=Real(c.num("body_x_m")*L);b.cy=Real(c.num("body_y_m")*L);b.z0=Real(c.num("body_z_m")*L);b.v0=Real(c.num("body_vz_ms")*L/T);
        b.radius=Real(c.num("body_radius_m")*L);b.length=Real(c.num("body_length_m")*L);b.density=Real(c.num("body_rho_kgm3")*density_scale);
        b.plate_height=Real(c.num("plate_height_m")*L);b.plate_thickness=Real(c.num("plate_thickness_m")*L);
        if(c.kv.count("body_dual")){b.dual=c.integer("body_dual");if(b.dual){b.cx2=Real(c.num("body2_x_m")*L);b.cy2=Real(c.num("body2_y_m")*L);b.z02=Real(c.kv.count("body2_z_m")?c.num("body2_z_m")*L:b.z0);}}
        b.enabled=c.integer("body_enabled");b.free=c.integer("body_free");b.inertia_mode=inertia=="implicit_interior"?2:inertia=="measured_interior"?1:0;
        if(c.kv.count("body_shape"))b.shape=c.integer("body_shape");
        if(c.kv.count("body_dual")){b.dual=c.integer("body_dual");if(b.dual){b.cx2=Real(c.num("body2_x_m")*L);b.cy2=Real(c.num("body2_y_m")*L);b.z02=Real((c.kv.count("body2_z_m")?c.num("body2_z_m"):c.num("body_z_m"))*L);}}
        if(b.inertia_mode==2&&p.clamp_density)throw std::runtime_error("implicit interior coupling requires linear, unclamped density");
        b.ramp_time=Real(c.kv.count("startup_ramp_s")?c.num("startup_ramp_s")*T:0);
        b.release_time=Real(c.kv.count("startup_release_s")?c.num("startup_release_s")*T:0);
        if(b.ramp_time<0||b.release_time<b.ramp_time||(b.ramp_time==0&&b.release_time!=0))throw std::runtime_error("invalid startup times");
        int logevery=c.integer("log_every"),sliceevery=c.integer("slice_every"),warmup=c.integer("warmup_steps"),mode=c.integer("initialization");
        bool trace=c.kv.count("trace_stages")?c.integer("trace_stages")!=0:false;
        bool diagnose_peak=c.kv.count("diagnose_peak")?c.integer("diagnose_peak")!=0:false;
        double phi_tol=c.num("phi_tolerance"),umax_limit=c.num("max_lattice_speed");
        std::string ale_csv; bool ale_enabled_cfg=false; std::string policy_csv,policy_markers_csv; bool policy_enabled_cfg=false; double output_interval=0,force_output_interval=0,video_output_interval=0; bool uint16_output=false;
        if(c.kv.count("ale_motion_csv")){ale_csv=c.str("ale_motion_csv");ale_enabled_cfg=true;}
        if(c.kv.count("policy_motion_csv")){policy_csv=c.str("policy_motion_csv");policy_enabled_cfg=true;}
        if(p.ib_phase_bound&&!policy_enabled_cfg&&b.inertia_mode!=2)throw std::runtime_error("legacy bounded IB source requires implicit_interior coupling");
        if(c.kv.count("policy_markers_csv")){policy_markers_csv=c.str("policy_markers_csv");if(!policy_enabled_cfg)throw std::runtime_error("policy_markers_csv requires policy_motion_csv");}
        if(c.kv.count("ale_output_interval_s"))output_interval=c.num("ale_output_interval_s");
        if(c.kv.count("ale_force_output_interval_s"))force_output_interval=c.num("ale_force_output_interval_s");
        if(c.kv.count("ale_uint16_output"))uint16_output=c.integer("ale_uint16_output")!=0;
        if(c.kv.count("video_output_interval_s"))video_output_interval=c.num("video_output_interval_s");
        double anchor_x=0,anchor_y=0;
        if(ale_enabled_cfg&&policy_enabled_cfg){
            anchor_x=c.num("ale_anchor_x_m");anchor_y=c.num("ale_anchor_y_m");
        }
        if(video_output_interval>0){
            if(!policy_enabled_cfg)throw std::runtime_error("video output requires a full policy assembly");
            uint16_output=false;output_interval=0;sliceevery=0;
        }
        std::string time_integrator=c.kv.count("time_integrator")?c.str("time_integrator"):"ssprk33";
        if(time_integrator!="ssprk33"&&time_integrator!="ssprk43")throw std::runtime_error("unknown time_integrator");
        RkScheme rk{time_integrator=="ssprk43"};
        if(rk.four_stage&&(multi_gpu<2||!policy_enabled_cfg))throw std::runtime_error("SSPRK43 currently requires multi-GPU prescribed policy motion");
        c.finish();
        if(video_output_interval<0||!(p.rho0>0&&p.rho1>=p.rho0&&p.nu0>=0&&p.nu1>=0&&p.xi>0&&p.sigma>=0&&p.mobility>=0&&logevery>0&&phi_tol>=0&&umax_limit>=0&&(umax_limit>0||p.transport_cfl_limit>0))||mode>2||p.periodic>1||b.enabled>1||b.free>1||b.dual<0||b.dual>1)
            throw std::runtime_error("invalid material or output parameters");
        if(b.enabled&&(!(b.radius>0&&b.length>0&&b.density>p.rho0)||p.periodic))throw std::runtime_error("invalid body or unsupported periodic IB");
        if(!policy_enabled_cfg&&(b.free||b.inertia_mode!=1))throw std::runtime_error("Drive experiment requires prescribed motion and measured interior force");
        if(b.plate_height<=0||b.plate_thickness<=0||b.radius<std::max(b.plate_height,b.plate_thickness)/2)throw std::runtime_error("Invalid plate or bounding radius");

        if(stretch&&!axis_files){
            if(p.periodic||p.transport_mode!=3||(p.ib_phase_bound&&!policy_enabled_cfg))throw std::runtime_error("Stretched branch requires nonperiodic FCT; bounded source requires policy motion");
            double finish=b.prescribed_position(end*T)/L;
            double low[3]={b.cx/L-b.plate_height/(2*L),b.cy/L-b.length/(2*L),std::min(double(b.z0)/L,finish)-b.plate_thickness/(2*L)};
            double high[3]={b.cx/L+b.plate_height/(2*L),b.cy/L+b.length/(2*L),std::max(double(b.z0)/L,finish)+b.plate_thickness/(2*L)};
            if(b.enabled)for(int a=0;a<3;++a)if(low[a]-3*h<fine_lo[a]||high[a]+3*h>fine_hi[a])throw std::runtime_error("Entire moving IB support must stay within uniform fine region");
            if(b.enabled&&b.dual){double lo2[3]={b.cx2/L-b.plate_height/(2*L),b.cy2/L-b.length/(2*L),b.z02/L-b.plate_thickness/(2*L)};double hi2[3]={b.cx2/L+b.plate_height/(2*L),b.cy2/L+b.length/(2*L),b.z02/L+b.plate_thickness/(2*L)};for(int a=0;a<3;++a)if(lo2[a]-3*h<fine_lo[a]||hi2[a]+3*h>fine_hi[a])throw std::runtime_error("Second moving IB support must stay within uniform fine region");}
        }
        Real dt=Real(dtsec*T);
        // Conservative explicit stability screens, not a guarantee of nonlinear stability.
        double cfl=dt/(std::sqrt(3.)*g.h),ch=dt*p.mobility*p.kappa()*144/std::pow(g.h,4);
        if(cfl*rk.euler_fraction()>.5||ch*rk.euler_fraction()>2.)throw std::runtime_error("Euler stage exceeds acoustic or fourth-order CH stability screen");
        double nsteps=std::ceil(end/dtsec-1e-9);if(nsteps>2147483647.)throw std::runtime_error("too many steps");
        int steps=int(nsteps);if(limit>0)steps=std::min(steps,limit);
        auto ms=cylinder_markers(b,g.h);
        make_directory(output);
        if(std::ifstream(output+"/history.csv"))throw std::runtime_error("output already contains history.csv; use a new directory");
        if(metric_mesh)metric_mesh->write(output,L);
        {std::ifstream src(path,std::ios::binary);std::ofstream dst(output+"/case.cfg",std::ios::binary);dst<<src.rdbuf();if(!dst)throw std::runtime_error("cannot save case");}
        std::string device="CPU";size_t free_before=0,total_device=0;
#ifdef __CUDACC__
        cuda_check(cudaSetDevice(0));cudaDeviceProp prop;cuda_check(cudaGetDeviceProperties(&prop,0));device=prop.name;
        if(prop.major<6)throw std::runtime_error("CUDA compute capability >=6.0 required for FP64 reduction atomics");
        cuda_check(cudaMemGetInfo(&free_before,&total_device));
#endif
        AleTrajectory trajectory;trajectory.anchor_x=anchor_x;trajectory.anchor_y=anchor_y;if(ale_enabled_cfg)trajectory.load_csv(ale_csv);
        PolicyAssemblyTrajectory policy_trajectory;
        if(policy_enabled_cfg){
            if(p.vertical_axis!=2)throw std::runtime_error("full policy assembly requires z vertical");
            if(policy_markers_csv.empty())throw std::runtime_error("policy_motion_csv requires policy_markers_csv");
            policy_trajectory.load_csv(policy_csv);ms=load_policy_markers_csv(policy_markers_csv);
            if(policy_trajectory.frames.front().t>0||policy_trajectory.frames.back().t<end)
                throw std::runtime_error("policy trajectory must cover [0,end_s] without extrapolation");
            if(trajectory.enabled&&(trajectory.frames.front().t>0||trajectory.frames.back().t<end))
                throw std::runtime_error("ALE trajectory must cover [0,end_s] without extrapolation");
            for(auto& m:ms){m.x.x=Real(double(m.x.x)*L);m.x.y=Real(double(m.x.y)*L);m.x.z=Real(double(m.x.z)*L);m.area=Real(double(m.area)*L*L);}
            for(auto& f:policy_trajectory.frames)for(auto& s:f.part){s.px*=L;s.py*=L;s.pz*=L;s.vx*=L/T;s.vy*=L/T;s.vz*=L/T;s.wx/=T;s.wy/=T;s.wz/=T;}
            b.enabled=1;b.free=0;b.inertia_mode=1;
        }
        size_t predicted=Solver::allocation_bytes(g,p,ms.size());
        if(metric_mesh)predicted+=metric_mesh->storage.size()*sizeof(Real);
        std::cout<<"case="<<name<<" backend="<<backend_name<<" precision="<<8*sizeof(Real)<<" status=research_unvalidated\n";
        std::cout<<"grid="<<g.nx<<'x'<<g.ny<<'x'<<g.nz<<" cells="<<g.n()<<" markers="<<ms.size()<<" allocation_estimate_bytes="<<predicted<<" steps="<<steps<<" acoustic_cfl="<<cfl<<" ch_stiffness="<<ch<<'\n';
        std::cout<<"time_integrator="<<time_integrator<<" stages="<<rk.stages()<<" euler_stage_fraction="<<rk.euler_fraction()<<" stage_acoustic_cfl="<<cfl*rk.euler_fraction()<<'\n';
        if(inspect)return 0;
#ifdef __CUDACC__
        // A decomposed run checks its local allocation on each device instead.
        if(multi_gpu<2&&Solver::allocation_bytes(g,p,ms.size())>free_before*0.9)throw std::runtime_error("insufficient device memory (10% headroom required)");
        if(multi_gpu>1)return run_multi_gpu(output,name,L,T,mass_scale,dtsec,end,limit,logevery,mode,phi_tol,umax_limit,pressure_scale,p,b,ms,trajectory,policy_trajectory,metric_mesh.get(),multi_gpu,uint16_output,output_interval,force_output_interval,video_output_interval,decomp,rk.four_stage);
#else
        if(multi_gpu>1)throw std::runtime_error("--multi-gpu requires CUDA");
#endif
        Solver solver(g,p,b,ms,trajectory,policy_trajectory);solver.trace_stages=trace;solver.ale_time_scale=T;solver.ale_velocity_scale=L/T;solver.update_ale(0);solver.update_policy_pose(0);solver.initialize(mode);synchronize();
        std::ofstream history(output+"/history.csv");history<<"step,time_s,drive_position_m,unused_coordinate_m,negative_drive_displacement_m,drive_velocity_ms,reaction_N,phi_sum,phi_mass_relative_change,phi_min,phi_max,max_lattice_speed,p_min_Pa,p_max_Pa,geometry_errors,nonfinite,interior_momentum_rate_N,interior_mass_kg,corrected_hydrodynamic_force_N,body_force_balance_residual_N\n"<<std::setprecision(12);
        double initial_mass=solver.stats().phase_sum;double compute_seconds=0,timed_seconds=0;int timed_steps=0,completed=0;bool valid=true;std::string failure;double next_video=video_output_interval>0?video_output_interval:1e300;int video_index=0;
        size_t observed_peak=0;
        auto record=[&](int step,double time){Stats st=solver.stats();auto bs=solver.body_state.download();
            double force_scale=mass_scale*L/(T*T);
            history<<step<<','<<time<<','<<bs[0]/L<<','<<(bs[0]-p.level)/L<<','<<(b.z0-bs[0])/L<<','<<bs[1]*T/L<<','<<bs[4]/force_scale<<','<<st.phase_sum<<','<<(st.phase_sum-initial_mass)/(std::abs(initial_mass)+1e-100)<<','<<st.phimin<<','<<st.phimax<<','<<st.umax<<','<<st.pmin/pressure_scale<<','<<st.pmax/pressure_scale<<','<<bs[6]<<','<<st.nonfinite;
            if(b.inertia_mode)history<<','<<bs[7]/force_scale<<','<<bs[8]/mass_scale<<','<<bs[10]/force_scale<<','<<bs[12]/force_scale;
            else history<<",,,,";
            history<<'\n';history.flush();
#ifdef __CUDACC__
            size_t now,tot;cuda_check(cudaMemGetInfo(&now,&tot));observed_peak=std::max(observed_peak,free_before>now?free_before-now:size_t(0));
#endif
            if(st.nonfinite||!std::isfinite(bs[0])||!std::isfinite(bs[1])){valid=false;failure="nonfinite state";}
            else if(p.rho(Real(st.phimin))<=0){valid=false;failure="nonpositive material density";}
            else if(bs[6]){valid=false;failure="marker support outside computational domain";}
            else if(bs[24]){valid=false;failure="IB phase projection infeasible or failed to converge";}
            else if(st.phimin < -phi_tol||st.phimax>1+phi_tol){valid=false;failure="phase overshoot exceeded configured tolerance";}
            else if(umax_limit>0&&st.umax>umax_limit){valid=false;failure="maximum lattice speed exceeded configured limit";}
            std::cout<<"step="<<step<<" t="<<time<<" phi=["<<st.phimin<<','<<st.phimax<<"] umax="<<st.umax<<" descent_m="<<(b.z0-bs[0])/L<<'\n';
            if(diagnose_peak&&(st.umax>.1||!valid)){
                auto peak=solver.peak_probe();int i=0,j=0,k=0;
                if(peak[1]>=0)g.xyz(size_t(peak[1]),i,j,k);
                Vec point=g.position(i,j,k);
                std::cout<<"peak_probe step="<<step<<" x_m="<<point.x/L<<" y_m="<<point.y/L<<" z_m="<<point.z/L
                         <<" phi="<<peak[2]<<" p_Pa="<<peak[3]/pressure_scale<<" distance_from_solid_m="<<b.signed_distance(point,Real(bs[0]))/L
                         <<" interior_umax="<<peak[4]<<" exterior_umax="<<peak[5]<<'\n';
            }
            std::cout.flush();
        };
        record(0,0);double next_output=output_interval>0?output_interval:1e300,next_force_output=force_output_interval>0?force_output_interval:1e300;int output_index=0;double field_output_seconds=0,force_output_seconds=0,video_output_seconds=0;auto start=std::chrono::steady_clock::now();
        DeviceTimer timer;
        for(int it=1;it<=steps&&valid;++it){
            const auto clock=step_time(it,dtsec,end,T);
            double t0=clock.start_s,ds=clock.duration_s;Real stepdt=clock.lattice_dt;
            double seconds=0;
            try{timer.start();solver.step(stepdt);seconds=timer.stop();}
            catch(const std::exception& error){valid=false;failure=std::string("solver execution error: ")+error.what();break;}
            compute_seconds+=seconds;
            if(it>warmup){timed_seconds+=seconds;++timed_steps;}completed=it;
            if(it%logevery==0||it==steps)record(it,t0+ds);
            if(uint16_output&&output_interval>0&&t0+ds+1e-12>=next_output){
                auto output_start=std::chrono::steady_clock::now();
                std::string stem=output+"/fields_"+std::to_string(output_index)+".u16";
                solver.write_uint16_frame(stem,pressure_scale,L/T);
                std::ofstream om(output+"/fields_"+std::to_string(output_index)+".json");
                om<<"{\"time_s\":"<<std::setprecision(12)<<t0+ds<<",\"channels\":[\"pressure_Pa\",\"phi\",\"ux_ms\",\"uy_ms\",\"uz_ms\"],\"dtype\":\"uint16\",\"nx\":"<<g.nx<<",\"ny\":"<<g.ny<<",\"nz\":"<<g.nz<<",\"lagrangian_force_model\":\"ship_aggregate_plus_paddle_markers\"}\n";
                ++output_index;next_output+=output_interval;
                field_output_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-output_start).count();
            }
            if(force_output_interval>0&&t0+ds+1e-12>=next_force_output){
                auto output_start=std::chrono::steady_clock::now();
                int force_index=int(std::floor((t0+ds)/force_output_interval+1e-9))-1;
                solver.write_lagrangian_forces(output+"/lagrangian_forces_"+std::to_string(force_index)+".csv",L,mass_scale*L/(T*T),double(stepdt));
                while(next_force_output<=t0+ds+1e-12)next_force_output+=force_output_interval;
                force_output_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-output_start).count();
            }
            if(video_output_interval>0&&t0+ds+1e-12>=next_video){
                auto video_start=std::chrono::steady_clock::now();
                AleFrameHost af=trajectory.enabled?trajectory.sample(t0+ds):AleFrameHost{};
                solver.write_video_frame(output+"/video_surface.bin",t0+ds,L,T,mass_scale,af);
                video_output_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-video_start).count();
                ++video_index;while(next_video<=t0+ds+1e-12)next_video+=video_output_interval;
            }
            if(sliceevery>0&&(it%sliceevery==0||it==steps))solver.write_slice(output+"/slice_"+std::to_string(it)+".csv",L,pressure_scale);
        }
        double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::ofstream meta(output+"/metrics.json");meta<<std::setprecision(12)<<"{\n  \"case\": \""<<name<<"\",\n  \"backend\": \""<<backend_name<<"\",\n  \"device\": \""<<device<<"\",\n  \"precision_bits\": "<<8*sizeof(Real)<<",\n  \"validation_status\": \"research_unvalidated\",\n  \"run_passed_health_checks\": "<<(valid?"true":"false")<<",\n  \"completed_steps\": "<<completed<<",\n  \"requested_steps\": "<<steps<<",\n  \"full_case_duration_reached\": "<<((completed*dtsec>=end-1e-10)&&valid?"true":"false")<<",\n  \"compute_seconds\": "<<compute_seconds<<",\n  \"timed_steps_after_warmup\": "<<timed_steps<<",\n  \"seconds_per_100_steps\": ";
        if(timed_steps)meta<<100*timed_seconds/timed_steps;else meta<<"null";
        meta<<",\n  \"wall_seconds_including_diagnostics_io\": "<<wall<<",\n  \"field_output_seconds\": "<<field_output_seconds<<",\n  \"force_output_seconds\": "<<force_output_seconds<<",\n  \"output_seconds_total\": "<<field_output_seconds+force_output_seconds+video_output_seconds<<",\n  \"output_wall_fraction\": "<<(wall>0?(field_output_seconds+force_output_seconds+video_output_seconds)/wall:0)<<",\n  \"peak_owned_allocation_bytes\": "<<peak_bytes()<<",\n  \"device_free_memory_delta_peak_bytes\": ";
#ifdef __CUDACC__
        meta<<observed_peak;
#else
        (void)free_before;(void)total_device;(void)observed_peak;meta<<"null";
#endif
        meta<<",\n  \"gpu_elapsed_seconds\": ";
#ifdef __CUDACC__
        meta<<compute_seconds;
#else
        meta<<"null";
#endif
        meta<<",\n  \"eulerian_cells\": "<<g.n()<<",\n  \"lagrangian_markers\": "<<ms.size()<<",\n  \"prescribed_policy_assembly\": "<<(policy_trajectory.enabled?"true":"false")<<",\n  \"prescribed_parts\": "<<(policy_trajectory.enabled?3:0)<<",\n  \"bytes_per_eulerian_cell_fields\": "<<(p.cache_faces?48:p.transport_mode==3?26:25)*sizeof(Real)<<",\n  \"failure\": \""<<failure<<"\"\n}\n";
        return valid?0:3;
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
}
