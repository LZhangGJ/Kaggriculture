// Handoff-only adapter: a day-indexed candidate choice, NOT a raw action tape.
// All economic construction, validity masks and low-level execution stay frozen.
#define economic_policy_api frozen_policy_api
#ifdef HANDOFF_F3
#include "f3_policy.cpp"
#else
#include "c3_policy.cpp"
#endif
#undef economic_policy_api

namespace dayplan {
struct Controlled {
    Policy policy;
    std::array<int,29> choices{};
    Controlled(const char*path,uint64_t seed,int mode):policy("",seed,0){
        if(mode!=0)throw std::runtime_error("day-plan adapter only supports mode 0");
        if(!path||!*path)return;
        std::ifstream file(path);
        if(!file)throw std::runtime_error("day-plan file missing");
        for(int&choice:choices){
            if(!(file>>choice)||choice<0||choice>=econrl::A)
                throw std::runtime_error("day plan requires 29 integers in 0..9");
        }
        std::string extra;
        if(file>>extra)throw std::runtime_error("extra day-plan data");
    }
    fastkag::PlayerAction act(const econrl::Input&i){
        if(i.day<0||i.day>29)throw std::runtime_error("day out of range");
        // Session implements illegal-candidate -> KEEP, with mask and actual
        // choice both returned to callers. Day 29 retains terminal management.
        policy.rl.mode=i.day<29?10+choices[i.day]:0;
        return policy.act(i);
    }
};
void*create(const char*f,uint64_t seed,int mode){return new Controlled(f,seed,mode);}
void destroy(void*p){delete static_cast<Controlled*>(p);}
fastkag::PlayerAction act(void*p,const econrl::Input&i){return static_cast<Controlled*>(p)->act(i);}
econrl::Session*session(void*p){return &static_cast<Controlled*>(p)->policy.rl;}
}
extern "C" econrl::Api economic_policy_api(){return {dayplan::create,dayplan::destroy,dayplan::act,dayplan::session};}
