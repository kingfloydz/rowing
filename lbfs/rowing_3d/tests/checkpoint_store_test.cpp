#undef NDEBUG
#include "checkpoint_store.hpp"
#include <cassert>
#include <iostream>
int run(int argc,char**argv){
    std::string root=argv[1];int count=2;
    auto write=[&](const std::string& dir){for(int r=0;r<count;++r){std::ofstream f(dir+"/rank_"+std::to_string(r)+".checkpoint");f<<"state";}};
    checkpoint_publish_latest(root,10000,.5,count,write);assert(checkpoint_current(root)=="step_10000");
    bool failed=false;try{checkpoint_publish_latest(root,20000,1,count,[&](const std::string& dir){std::ofstream f(dir+"/rank_0.checkpoint");f<<"partial";throw std::runtime_error("injected failure");});}catch(const std::runtime_error&){failed=true;}
    assert(failed&&checkpoint_current(root)=="step_10000");assert(std::ifstream(root+"/step_10000/rank_1.checkpoint").good());
    checkpoint_publish_latest(root,20000,1,count,write);assert(checkpoint_resolve(root)==root+"/step_20000");assert(!std::ifstream(root+"/step_10000/rank_0.checkpoint").good());
    checkpoint_remove_generation(root+"/step_20000",count);std::remove((root+"/CURRENT").c_str());
#ifdef _WIN32
    _rmdir(root.c_str());
#else
    rmdir(root.c_str());
#endif
    return 0;
}

int main(int argc,char**argv){try{return run(argc,argv);}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}}
