#pragma once
#include <cstdio>
#include <cerrno>
#include <fstream>
#include <stdexcept>
#include <string>
#include <chrono>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <direct.h>
#include <io.h>
#include <fcntl.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#endif

inline void checkpoint_mkdir(const std::string& p){
#ifdef _WIN32
    int rc=_mkdir(p.c_str());
#else
    int rc=mkdir(p.c_str(),0775);
#endif
    if(rc&&errno!=EEXIST)throw std::runtime_error("cannot create checkpoint directory: "+p);
}
inline void checkpoint_durable(const std::string& p){
#ifdef _WIN32
    int fd=_open(p.c_str(),_O_RDWR|_O_BINARY);if(fd<0)throw std::runtime_error("checkpoint sync open failed");int rc=_commit(fd);_close(fd);
#else
    int fd=open(p.c_str(),O_RDONLY);if(fd<0)throw std::runtime_error("checkpoint sync open failed");int rc=fsync(fd);close(fd);
#endif
    if(rc)throw std::runtime_error("checkpoint sync failed: "+p);
}
inline bool checkpoint_generation(const std::string& s){
    return s.rfind("step_",0)==0&&s.size()>5&&s.find_first_not_of("0123456789",5)==std::string::npos;
}
inline std::string checkpoint_current(const std::string& root){
    std::ifstream in(root+"/CURRENT");std::string s;if(!in)return {};
    if(!(in>>s)||!checkpoint_generation(s))throw std::runtime_error("invalid checkpoint CURRENT");return s;
}
inline std::string checkpoint_resolve(const std::string& root){
    auto s=checkpoint_current(root);return s.empty()?root:root+"/"+s;
}
inline void checkpoint_remove_generation(const std::string& folder,int ranks){
    // Only files produced by this writer; never recursive shell removal.
    for(int r=0;r<ranks;++r)std::remove((folder+"/rank_"+std::to_string(r)+".checkpoint").c_str());
    std::remove((folder+"/complete.json").c_str());
#ifdef _WIN32
    int rc=_rmdir(folder.c_str());
#else
    int rc=rmdir(folder.c_str());
#endif
    if(rc&&errno!=ENOENT)throw std::runtime_error("cannot remove previous checkpoint generation: "+folder);
}
template<class Writer> inline void checkpoint_publish_latest(const std::string& root,int step,double time,int ranks,Writer writer){
    checkpoint_mkdir(root);auto previous=checkpoint_current(root);std::string name="step_"+std::to_string(step),next=root+"/"+name;
    if(previous==name)throw std::runtime_error("refusing to overwrite published checkpoint");
    // An incomplete generation at this step must be investigated, not overwritten.
#ifdef _WIN32
    int rc=_mkdir(next.c_str());
#else
    int rc=mkdir(next.c_str(),0775);
#endif
    if(rc)throw std::runtime_error("checkpoint generation already exists or cannot be created: "+next);
    try{
        writer(next);
        for(int r=0;r<ranks;++r)checkpoint_durable(next+"/rank_"+std::to_string(r)+".checkpoint");
        {std::ofstream meta(next+"/complete.json");meta.precision(17);meta<<"{\"step\":"<<step<<",\"time_s\":"<<time<<",\"ranks\":"<<ranks<<"}\n";meta.close();if(!meta)throw std::runtime_error("checkpoint manifest write failed");}
        checkpoint_durable(next+"/complete.json");
#ifndef _WIN32
        checkpoint_durable(next);
#endif
        {std::ofstream f(root+"/CURRENT.tmp",std::ios::trunc);f<<name<<'\n';f.close();if(!f)throw std::runtime_error("checkpoint CURRENT write failed");}
        checkpoint_durable(root+"/CURRENT.tmp");
#ifdef _WIN32
        if(!MoveFileExA((root+"/CURRENT.tmp").c_str(),(root+"/CURRENT").c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))throw std::runtime_error("checkpoint publish failed");
#else
        if(std::rename((root+"/CURRENT.tmp").c_str(),(root+"/CURRENT").c_str()))throw std::runtime_error("checkpoint publish failed");
        checkpoint_durable(root);
#endif
    }catch(...){if(checkpoint_current(root)!=name)checkpoint_remove_generation(next,ranks);throw;}
    if(!previous.empty())checkpoint_remove_generation(root+"/"+previous,ranks);
}
