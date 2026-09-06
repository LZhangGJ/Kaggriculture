// Only the generated copy of rl_common.hpp changes its KEEP score constant.
#define economic_policy_api frozen_economic_policy_api
#include "build/f3_policy.cpp"
#undef economic_policy_api

namespace keep_ablation {
constexpr float values[]{3.58351893846f,2.f,1.f,0.f};
struct Controlled {
    Policy policy;
    float bonus;
    Controlled(const char*path,uint64_t seed,int mode):policy(path,seed,mode==0?0:1),bonus(3.58351893846f){
        if(mode!=0){
            if(mode<100||mode>103)throw std::runtime_error("KEEP ablation mode");
            bonus=values[mode-100];
        }
    }
    fastkag::PlayerAction act(const econrl::Input&i){
        econrl::inference_keep_bonus=bonus;
        return policy.act(i);
    }
};
void*create(const char*p,uint64_t s,int m){return new Controlled(p,s,m);}
void destroy(void*p){delete static_cast<Controlled*>(p);}
fastkag::PlayerAction act(void*p,const econrl::Input&i){return static_cast<Controlled*>(p)->act(i);}
econrl::Session*session(void*p){return &static_cast<Controlled*>(p)->policy.rl;}
}
extern "C" econrl::Api economic_policy_api(){return {keep_ablation::create,keep_ablation::destroy,keep_ablation::act,keep_ablation::session};}
