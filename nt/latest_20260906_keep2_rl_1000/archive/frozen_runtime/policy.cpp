#define economic_policy_api frozen_economic_policy_api
#include "../economic_rl_keep_bias_20260906/build/f3_policy.cpp"
#undef economic_policy_api

namespace keep2_training {
struct Controlled {
    Policy policy;
    Controlled(const char*path,uint64_t seed,int mode):policy(path,seed,mode){
        if(mode<0||mode>3)throw std::runtime_error("KEEP2 mode");
    }
    fastkag::PlayerAction act(const econrl::Input&i){
        econrl::inference_keep_bonus=2.f;
        return policy.act(i);
    }
};
void*create(const char*p,uint64_t s,int m){return new Controlled(p,s,m);}
void destroy(void*p){delete static_cast<Controlled*>(p);}
fastkag::PlayerAction act(void*p,const econrl::Input&i){return static_cast<Controlled*>(p)->act(i);}
econrl::Session*session(void*p){return &static_cast<Controlled*>(p)->policy.rl;}
}
extern "C" econrl::Api economic_policy_api(){return {keep2_training::create,keep2_training::destroy,keep2_training::act,keep2_training::session};}
