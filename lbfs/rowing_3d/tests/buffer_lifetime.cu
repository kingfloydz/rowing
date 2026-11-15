#include "backend.hpp"
#include <iostream>
int main(){
    Buffer<float> state(4096),base(4096),work(4096);
    std::vector<float> values(4096);for(size_t i=0;i<values.size();++i)values[i]=float(i);
    state.upload(values);base.copy_from_async(state);
    float* p=work.data();parallel_for(work.size(),[=] HD(size_t i){p[i]=2*float(i);});
    auto old=state.data();state.swap_storage(work);
    if(work.data()!=old)throw std::runtime_error("buffer ownership was not swapped");
    auto s=state.download(),b=base.download();
    for(size_t i=0;i<s.size();++i)if(s[i]!=2*values[i]||b[i]!=values[i])throw std::runtime_error("stream order / snapshot corrupted");
    for(int n=0;n<100;++n)state.swap_storage(work);
    if(state.download()!=s)throw std::runtime_error("repeated swap corrupted storage");
    bool rejected=false;try{Buffer<float> bad(1);state.swap_storage(bad);}catch(const std::runtime_error&){rejected=true;}
    if(!rejected)throw std::runtime_error("shape mismatch accepted");
    std::cout<<"GPU buffer swap, async snapshot, ownership lifetime: PASS\n";
}
