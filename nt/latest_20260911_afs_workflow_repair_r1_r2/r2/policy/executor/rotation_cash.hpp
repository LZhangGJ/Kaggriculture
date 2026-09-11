#pragma once
// Online current-observation experiment, not an Oracle or replay selector.
namespace dp7::rotationcash {
using namespace rotationproto;
struct Edit {std::vector<int>plots;int kind,when;double cheap=0;};
inline void edit(Controller&c,const Edit&e){
 for(int pos:e.plots){
  auto it=std::find_if(c.target.begin(),c.target.end(),[&](auto x){return x.first==pos;});
  if(it==c.target.end())c.target.emplace_back(pos,e.kind);else it->second=e.kind;
  c.plant_not_before[pos]=e.when;c.deferred_kind[pos]=e.kind;c.deferred_since[pos]=c.day;
 }
}
inline Works planned_works(const Controller&c,const View&o){
 auto works=physical_works(c,o);std::array<bool,100>seen{};for(auto&w:works)seen[w.pos]=true;
 for(auto[pos,k]:c.target){
  if(k<0)continue;auto&t=o.own.tiles[pos];
  if(plant(t)&&k<5){
   if(k!=int(t.crop)||c.plant_not_before[pos]>o.day)works=replace(c,o,works,pos,k,std::max(o.day,c.plant_not_before[pos]));
  }else if(!seen[pos]){
   if(k<5){auto f=crop(c,o,k,t,std::max(o.day,c.plant_not_before[pos]));works.push_back({pos,k,f.cal,f.seeds,0});}
   else works.push_back({pos,k,portfolio::delivery(c,k,o.day),{},double(animal_price[k-9])});
   seen[pos]=true;
  }
 }
 return works;
}
struct Consequence {double score=-INFINITY;bool known=false;std::array<int,12>lost{};int ticks=0;};
inline Consequence consequence(const Controller&source,const View&o){
 ObservedDayScenario scenario(o);auto c=source;c.admission_inspection=nullptr;c.workflow_runtime=false;
 // Avoid recursively searching while comparing a single current-day plan.
 // Keep the real resource/worker/market executor, and its per-step repairs.
 c.p.portfolio_rotation=false;c.p.day_consequence_compare=false;c.p.day_value_replan_next_day=false;
 // Bounded leaf for outer investment search: do not start another search over
 // alternative first schedules inside each hypothetical macro candidate.
 // Real post-fill compilation still uses the selected first-schedule policy.
 if(c.p.compile_bounded_rollout)c.p.compile_consequence=c.p.compile_replant_choices=false;
 c.day=o.day;c.last_step=o.step-1;Consequence out;
 while(!scenario.finished())scenario.advance(c.act(scenario.view()));
 out.ticks=scenario.ticks();out.known=true;
 // Existing assets cannot be silently destroyed to make the residual look
 // attractive. Planned harvest/replacement is distinguished from loss below.
 const auto&after=scenario.own();
 for(int pos=0;pos<100;pos++){
  auto&a=o.own.tiles[pos];auto&b=after.tiles[pos];
  if(animal(a)&&!animal(b))out.lost[int(a.animal)]++;
  if(plant(a)&&!plant(b)&&b.kind==TileKind::WEED)out.lost[int(a.crop)]++;
 }
 if(scenario.step_count()>=719){out.score=after.money;return out;}
 View end{scenario.step_count(),scenario.step_count()/24,0,after,o.opponent,scenario.inventory(),scenario.market(),scenario.shops()};
 c.day=end.day;auto works=physical_works(c,end);
 // Only actual assets and deliberately deferred projects survive into the
 // residual. Unfilled purchases/unstarted ordinary targets earn no income.
 for(int pos=0;pos<100;pos++)if(c.deferred_kind[pos]>=0&&!plant(after.tiles[pos])&&!animal(after.tiles[pos])){
  int kind=c.deferred_kind[pos];auto f=crop(c,end,kind,after.tiles[pos],std::max(end.day,c.plant_not_before[pos]));
  works.push_back({pos,kind,f.cal,f.seeds,0});
 }
 auto v=value(c,end,works);out.known=v.schedulable&&v.funding_gap<1e-6;out.score=v.cash;
 return out;
}
inline std::vector<Edit> proposals(const Controller&c,const View&o){
 std::array<std::vector<int>,5>groups;auto live=c.counts(o);
 const std::array<int,5>limits{75,75,c.p.max_tomato,c.p.max_strawberry,c.p.max_melon};
 for(auto[pos,k]:c.target){auto&t=o.own.tiles[pos];
  if(plant(t)&&!ongoing(int(t.crop))){int old=int(t.crop);bool expiry=t.yield_units>0&&t.max_lifespan_step>=0&&t.max_lifespan_step<=(o.day+1)*24;
   if(o.day-t.planted_day>=c.h_age(old)||expiry)groups[old].push_back(pos);
  }else if(!plant(t)&&!animal(t)&&c.deferred_kind[pos]>=0)groups[c.deferred_kind[pos]].push_back(pos);
 }
 std::vector<Edit> result;
 for(int old=0;old<5;old++){
  auto&g=groups[old];if(g.empty())continue;
  std::sort(g.begin(),g.end(),[](int a,int b){return std::tuple(near(a),snake(a))<std::tuple(near(b),snake(b));});
  std::vector<int>sizes;for(int n=1;n<int(g.size());n*=2)sizes.push_back(n);sizes.push_back(int(g.size()));
  for(int kind=0;kind<5;kind++)for(int n:sizes){
   if(kind!=old&&live[kind]+n>limits[kind])continue;
   std::vector<int>times{o.day};if(c.p.rotation_timing){if(o.day+1<30)times.push_back(o.day+1);if(o.day+2<30)times.push_back(o.day+2);if(kind==old)times.push_back(30);}
   for(int when:times){if(kind==old&&when==o.day)continue;if(when<30&&when+c.h_age(kind)>29)continue;
    result.push_back({std::vector<int>(g.begin(),g.begin()+n),kind,when});
   }
  }
 }
 return result;
}
} // namespace dp7::rotationcash
namespace dp7 {
inline void Controller::compare_rotation(const View&planning,const View&actual,int released){
 if(!p.portfolio_rotation||day>=29)return;
 auto candidates=rotationcash::proposals(*this,planning);if(candidates.empty())return;
 rotation_generated+=int(candidates.size());
 auto context=rotationcash::planned_works(*this,planning);
 // Cheap cash-only screen is a shortlist, not the final decision. Preserve
 // immediate, waiting and stopping representatives when timing is enabled.
 for(auto&e:candidates){auto work=context;for(int pos:e.plots)work=rotationproto::replace(*this,planning,work,pos,e.kind,e.when);e.cheap=rotationproto::value(*this,planning,work,false).cash;}
 std::stable_sort(candidates.begin(),candidates.end(),[](const auto&a,const auto&b){return std::tuple(-a.cheap,a.when,a.kind,a.plots)<std::tuple(-b.cheap,b.when,b.kind,b.plots);});
 std::vector<rotationcash::Edit> shortlist;
 auto take=[&](const auto&e){for(const auto&x:shortlist)if(x.kind==e.kind&&x.when==e.when&&x.plots==e.plots)return;shortlist.push_back(e);};
 // Compute budget, not a crop quota: one best per timing family, then fill.
 for(int category=0;category<(p.rotation_timing?3:1);category++)for(auto&e:candidates){int type=e.when==30?2:e.when>day?1:0;if(type==category){take(e);break;}}
 for(auto&e:candidates){if(shortlist.size()>=4)break;take(e);}
 auto reference=rotationcash::consequence(*this,actual);rotation_evaluated++;
 if(!reference.known){rotation_unknown++;return;}
 auto best=*this;double best_value=reference.score;bool changed=false;
 for(auto&e:shortlist){
  auto trial=*this;rotationcash::edit(trial,e);trial.prepare_orders(planning,actual,released);
  auto next=rotationcash::consequence(trial,actual);rotation_evaluated++;
  if(!next.known){rotation_unknown++;continue;}
  bool safe=true;for(int i=0;i<12;i++)if(next.lost[i]>reference.lost[i])safe=false;
  if(safe&&next.score>best_value+1e-6){best_value=next.score;best=std::move(trial);changed=true;}
 }
 if(changed){target=std::move(best.target);plant_not_before=best.plant_not_before;deferred_kind=best.deferred_kind;deferred_since=best.deferred_since;
  prepare_orders(planning,actual,released);rotation_applied++;for(int pos=0;pos<100;pos++)rotation_waits+=plant_not_before[pos]>day;
 }
}
}
