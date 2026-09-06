// Audit-only wrapper. The accepted F3 policy and its action/feature code are unchanged.
#define economic_policy_api frozen_f3_policy_api
#include "../economic_rl_f3_ledger_fix_20260906/f3_policy.cpp"
#undef economic_policy_api

namespace candidate_audit {
struct Controlled {
    Policy policy;
    int normal_mode=2, target=-1, forced=-1, suffix=2;
    Controlled(const char*path,uint64_t seed,int mode):policy(path,seed,mode<1000?mode:2){
        if(mode<1000){normal_mode=mode;return;}
        int code=mode-1000;
        suffix=code/300;target=(code%300)/10;forced=code%10;
        if((suffix!=0&&suffix!=2)||target<0||target>28||forced<0||forced>=econrl::A)
            throw std::runtime_error("audit intervention code");
    }
    fastkag::PlayerAction act(const econrl::Input&i){
        bool daily=i.day!=policy.ctx.agent.m.day;
        policy.rl.mode=target<0?normal_mode:(i.day<target?2:(i.day==target?10+forced:suffix));
        auto action=policy.act(i);
        if(daily&&i.day==target){
            auto&d=policy.rl.records.back();
            if(!d.mask[forced]||d.choice!=forced)throw std::runtime_error("forced unavailable audit option");
            // Same consumption as Session::select(mode=2), even though its choice
            // is overridden. Post-target common random numbers remain aligned.
            std::array<double,econrl::A>w{};
            for(int j=0;j<econrl::A;j++)w[j]=d.probability[j];
            (void)std::discrete_distribution<int>(w.begin(),w.end())(policy.rl.rng);
        }
        return action;
    }
};
void*create(const char*p,uint64_t seed,int mode){return new Controlled(p,seed,mode);}
void destroy(void*p){delete static_cast<Controlled*>(p);}
fastkag::PlayerAction act(void*p,const econrl::Input&i){return static_cast<Controlled*>(p)->act(i);}
econrl::Session*session(void*p){return &static_cast<Controlled*>(p)->policy.rl;}
}
extern "C" econrl::Api economic_policy_api(){
    return {candidate_audit::create,candidate_audit::destroy,candidate_audit::act,candidate_audit::session};
}
