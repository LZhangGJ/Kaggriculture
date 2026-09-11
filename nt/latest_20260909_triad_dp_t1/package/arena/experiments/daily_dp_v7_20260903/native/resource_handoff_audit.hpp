#pragma once
// Read-only frozen unit-phase verification of public-state exchange proposals.
#include "production_audit.hpp"
#include "resource_exchange.hpp"
namespace dp7audit::handoff {
using namespace dp7;
using namespace dp7::exchange;
inline auto tile_key(const Tile&t){
 return std::tuple(int(t.kind),int(t.crop),int(t.animal),t.planted_day,t.placed_day,t.yield_units,t.consecutive_unwatered,t.consecutive_unfed,t.fertilized_until_day,t.pending_care_bonus,t.max_lifespan_step,t.watered_today,t.fed_today,t.cared_today,t.fertilizer_available);
}
struct Projection {Simulator env;ProductionDay effects;int noop=0;};
inline Projection project(const Simulator&env,int seat,const std::vector<Plan>&ps){
 Projection out{env,{}};std::vector<NoEffect>examples;int span=Controller::plan_load(ps).first;
 for(int k=0;k<span;k++){PlayerAction a;a.units.resize(ps.size());for(size_t u=0;u<ps.size();u++)if(k<int(ps[u].a.size()))a.units[u]=ps[u].a[k];out.env=units(out.env,seat,a,out.effects,examples);}
 for(int n:out.effects.no_effect)out.noop+=n;return out;
}
inline bool equal_effect(const Projection&a,const Projection&b,int seat){
 auto&af=a.env.farms()[seat];auto&bf=b.env.farms()[seat];
 if(af.money!=bf.money||af.unlocked_mask!=bf.unlocked_mask||af.tiles.size()!=bf.tiles.size())return false;
 for(size_t k=0;k<af.tiles.size();k++)if(tile_key(af.tiles[k])!=tile_key(bf.tiles[k]))return false;
 return a.env.privates()[seat].seeds==b.env.privates()[seat].seeds&&private_stock(a.env,seat)==private_stock(b.env,seat)&&a.effects.drop_loss==b.effects.drop_loss;
}
struct Sample {int step=0,day=0,hour=0,compatible=0,feasible=0,shorter=0,total_saved=0,peak_saved=0,unit_a=-1,unit_b=-1,pos_a=-1,pos_b=-1,base_noop=-1,trial_noop=-1;bool base_resources=false,base_deadline=false,same_effect=false;};
inline Sample inspect(const Controller&ctl,const Simulator&env,int seat){
 View v{env.step_count(),env.day(),env.hour(),env.farms()[seat],env.farms()[1-seat],env.privates()[seat],env.market(),env.shops()};auto c=propose(ctl,v);Sample s;
 s.step=v.step;s.day=v.day;s.hour=v.hour;s.compatible=c.compatible;s.feasible=c.feasible;s.shorter=c.shorter;s.base_resources=c.base_resources;s.base_deadline=c.base_deadline;
 s.total_saved=c.total_saved;s.peak_saved=c.peak_saved;s.unit_a=c.unit_a;s.unit_b=c.unit_b;s.pos_a=c.pos_a;s.pos_b=c.pos_b;
 if(c.total_saved>0){auto before=project(env,seat,c.base),after=project(env,seat,c.best);s.base_noop=before.noop;s.trial_noop=after.noop;s.same_effect=equal_effect(before,after,seat);}
 return s;
}
}
