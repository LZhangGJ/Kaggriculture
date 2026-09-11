#pragma once
// Public-state, live idle-capacity admission. No Simulator, opponent identity,
// future RNG, replay route, hindsight labels, or counterfactual game suffix.
#include "policy.hpp"
namespace dp7::portfolio {double future_wages(const Controller&,const View&,const std::vector<std::pair<int,int>>&);}
namespace dp7::intraday {
struct Reservations {Counts shed{},seeds{};std::array<bool,100>tiles{};};
inline Reservations reserved(const Controller&c){
 Reservations r;for(auto&pl:c.plans)for(size_t k=pl.index;k<pl.a.size();k++){
  auto a=pl.a[k];int i=int(a.item),pos=pl.target[k];
  if(pos>=0&&!Controller::movement(a.op)&&a.op!=Op::PICKUP&&a.op!=Op::DROP)r.tiles[pos]=true;
  if(a.op==Op::PICKUP&&i>=0&&i<12)r.shed[i]+=a.quantity;
  if(a.op==Op::PLANT&&i>=0&&i<5)r.seeds[i]++;
 }return r;
}
inline Job project_job(const Controller&c,const View&o,int pos,int kind){
 Controller probe(c.p);probe.day=c.day;probe.target={{pos,kind}};auto js=probe.jobs(o);
 return js.size()==1?js[0]:Job{pos,{}};
}
inline Plan route(const View&o,int unit,const Job&j){
 Plan pl;int pos=cell(unit?o.own.hands.at(unit-1):o.own.farmer);Counts pickup{};
 for(int i=0;i<12;i++)pickup[i]=std::max(0,j.needs[i]-o.priv.inventories.at(unit)[i]);
 if(sum(pickup)){
  int dep=depot[0];for(int d:depot)if(dist(pos,d)<dist(pos,dep))dep=d;Controller::walk(pl,pos,dep);
  std::vector<int>items;for(int i=0;i<12;i++)if(pickup[i])items.push_back(i);
  std::sort(items.begin(),items.end(),[](int a,int b){return name(a)<name(b);});
  for(int i:items){pl.a.push_back(action(Op::PICKUP,i,pickup[i]));pl.target.push_back(dep);}
 }
 Controller::walk(pl,pos,j.pos);for(auto a:j.actions){pl.a.push_back(a);pl.target.push_back(j.pos);}return pl;
}
inline bool available(const View&o,int unit,const Job&j,const Reservations&r){
 for(int i=0;i<12;i++)if(j.needs[i]>o.priv.inventories.at(unit)[i]+std::max(0,o.priv.shed[i]-r.shed[i]))return false;
 for(int i=0;i<5;i++)if(j.seeds[i]>std::max(0,o.priv.seeds[i]-r.seeds[i]))return false;
 return true;
}
inline bool vacant(const View&o,int pos){return pos>=0&&pos<100&&(o.own.unlocked_mask&(1<<quad(pos)))&&!plant(o.own.tiles[pos])&&!animal(o.own.tiles[pos])&&o.own.tiles[pos].kind!=TileKind::LOCKED;}
inline void complete_pending(Controller&c,const View&o){
 for(auto it=c.intraday_units.begin();it!=c.intraday_units.end();){
  if(*it>=int(c.plans.size())||c.plans[*it].index>=c.plans[*it].a.size())it=c.intraday_units.erase(it);else ++it;
 }
 if(!c.p.intraday_admission||!c.pending_admission.active)return;
 auto pending=c.pending_admission;c.pending_admission.active=false;
 if(c.phase!=3||pending.day!=o.day||o.day>=29||pending.unit>=int(c.plans.size())||!vacant(o,pending.pos)||c.plant_not_before[pending.pos]>o.day){
  c.intraday_cancelled++;return;
 }
 if(c.plans[pending.unit].index<c.plans[pending.unit].a.size()){c.intraday_cancelled++;return;}
 auto r=reserved(c);auto j=project_job(c,o,pending.pos,pending.kind);
 if(r.tiles[pending.pos]||j.actions.empty()||!available(o,pending.unit,j,r)){c.intraday_unfilled++;return;}
 auto pl=route(o,pending.unit,j);
 if(o.hour+int(pl.a.size())>24){c.intraday_cancelled++;return;}
 c.plans[pending.unit]=std::move(pl);
 c.intraday_units.insert(pending.unit);
 auto it=std::find_if(c.target.begin(),c.target.end(),[&](auto x){return x.first==pending.pos;});
 if(it==c.target.end())c.target.emplace_back(pending.pos,pending.kind);else it->second=pending.kind;
 c.intraday_activated++;c.intraday_activated_by_kind[pending.kind]++;
}
struct Proposal {bool valid=false;int unit=-1,pos=-1,kind=-1;double value=0,expense=0;Job job;Acts buy;};
// Prospective work, not owned resources. Only scheduled, unplaced projects on
// empty plots are added: already visible crops include their own renewal path.
inline std::vector<std::pair<int,int>> committed_new_targets(const Controller&c,const View&o,const PlayerAction&a){
 std::array<bool,100>seen{};std::vector<std::pair<int,int>>r;
 auto take=[&](const Action&x,int pos,int offset){int k=int(x.item);
  if(!((x.op==Op::PLANT&&k>=0&&k<5)||(x.op==Op::PLACE&&k>=9&&k<12)))return;
  if(pos<0||pos>=100||o.hour+offset>=24||seen[pos]||!vacant(o,pos))return;
  seen[pos]=true;r.emplace_back(pos,k);
 };
 for(size_t u=0;u<a.units.size();u++)take(a.units[u],cell(u?o.own.hands.at(u-1):o.own.farmer),0);
 for(const auto&pl:c.plans)for(size_t i=pl.index;i<pl.a.size();i++)take(pl.a[i],pl.target[i],1+int(i-pl.index));
 return r;
}
inline double workforce_score(const Controller&c,const Proposal&q,double before,double after){
 if(!std::isfinite(before)||!std::isfinite(after))return q.value;
 auto pr=q.kind>=9?c.animal_project(q.kind):c.crop_project(q.kind);
 // Replace the old future-work proxy, don't charge it on top of real wages.
 // Current hired workers are already paid. Today-only walking opportunity
 // cost is retained in q.value. Bias remains unchanged for a clean ablation.
 return q.value+(c.p.action_shadow*pr.actions-std::max(0.,after-before))*c.p.bias[q.kind];
}
inline bool better(const View&o,const Proposal&a,const Proposal&b){
 return a.valid&&(!b.valid||std::tuple(a.value,-int(route(o,a.unit,a.job).a.size()),-a.expense,-a.unit,-a.pos)>
                          std::tuple(b.value,-int(route(o,b.unit,b.job).a.size()),-b.expense,-b.unit,-b.pos));
}
inline Proposal compare_workforce(const Controller&c,const View&o,const PlayerAction&a,const std::array<Proposal,12>&representatives,const Proposal&legacy){
 if(!c.p.intraday_future_workforce)return legacy;
 bool any=false;for(const auto&q:representatives)any|=q.valid;if(!any)return legacy;
 c.intraday_workforce_checks++;auto forecast=c;forecast.p.portfolio_crop_calendar=true;
 auto commitments=committed_new_targets(c,o,a);double before=portfolio::future_wages(forecast,o,commitments);
 Proposal best;
 for(auto q:representatives)if(q.valid){
  auto added=commitments;added.emplace_back(q.pos,q.kind);
  double after=portfolio::future_wages(forecast,o,added);c.intraday_workforce_candidates++;
  if(!std::isfinite(before)||!std::isfinite(after))c.intraday_workforce_unknown++;
  q.value=workforce_score(c,q,before,after);
  if(q.value>0&&better(o,q,best))best=std::move(q);
 }
 c.intraday_workforce_switches+=best.valid!=legacy.valid||(best.valid&&std::tuple(best.kind,best.pos,best.unit)!=std::tuple(legacy.kind,legacy.pos,legacy.unit));
 return best;
}
struct Declared {Counts planned{},selected{};int feed=0,duplicates=0,late=0;};
inline Declared declared(const Controller&c,const View&o,const PlayerAction&a){
 Declared r;std::array<bool,100>seen{};
 auto take=[&](const Action&x,int pos,int offset){
  int kind=int(x.item);if(!((x.op==Op::PLANT&&kind>=0&&kind<5)||(x.op==Op::PLACE&&kind>=9&&kind<12)))return;
  if(pos<0||pos>=100||!(o.own.unlocked_mask&(1<<quad(pos))))return;
  if(o.hour+offset>=24){r.late++;return;}
  if(seen[pos]){r.duplicates++;return;}seen[pos]=true;
  // Existing animals cannot be replaced by an animal PLACE. A crop PLANT
  // behind HARVEST/DIG is a new lifecycle, not the already visible crop.
  if(x.op==Op::PLACE&&animal(o.own.tiles[pos]))return;
  auto pr=kind>=9?c.animal_project(kind):c.crop_project(kind);r.planned[kind]++;
  r.selected[pr.item]+=pr.out;r.selected[F]+=kind>=9?pr.fert:-pr.fert;r.feed+=pr.feed;
 };
 for(size_t u=0;u<a.units.size();u++)take(a.units[u],cell(u?o.own.hands.at(u-1):o.own.farmer),0);
 for(const auto&pl:c.plans)for(size_t i=pl.index;i<pl.a.size();i++)take(pl.a[i],pl.target[i],1+int(i-pl.index));
 return r;
}
inline double committed_market_cost(const Controller&c,const View&o,const PlayerAction&a){
 double used=0;int hires=o.own.hires_today;
 for(auto x:a.market){
  if(x.op==Op::BUY_SEED||x.op==Op::BUY_ANIMAL||x.op==Op::BUY_PRODUCT||x.op==Op::BUY_LAND)used+=c.cost(o,x);
  if(x.op==Op::HIRE){if(hires>=int(fib.size()))return INFINITY;used+=fib[hires++];}
 }return used; // No credit for an unexecuted sale, harvest or deposit.
}
inline Proposal propose(const Controller&c,const View&o,const PlayerAction&a,const Declared*known=nullptr){
 Proposal best;if(c.phase!=3||o.day>=29||o.hour>=23||c.plans.size()!=o.priv.inventories.size())return best;
 std::array<Proposal,12> representatives;
 std::vector<int>idle;for(size_t u=0;u<c.plans.size();u++)if(c.plans[u].index>=c.plans[u].a.size()&&u<a.units.size()&&a.units[u].op==Op::PASS)idle.push_back(int(u));
 if(idle.empty())return best;
 auto reservations=reserved(c);auto live=c.counts(o);
 // Current actions have already advanced their plan cursors but have NOT
 // settled yet. Reserve them too; never allocate the same seed/tile twice.
 for(size_t u=0;u<a.units.size();u++){
  auto x=a.units[u];int i=int(x.item);int pos=cell(u?o.own.hands[u-1]:o.own.farmer);
  if(x.op==Op::PICKUP&&i>=0&&i<12)reservations.shed[i]+=x.quantity;
  if(x.op==Op::PLANT&&i>=0&&i<5)reservations.seeds[i]++;
  if(x.op!=Op::PASS&&!Controller::movement(x.op)&&x.op!=Op::DROP&&x.op!=Op::PICKUP)reservations.tiles[pos]=true;
 }
 for(auto[pos,kind]:c.target)if(kind>=0&&reservations.tiles[pos]&&!plant(o.own.tiles[pos])&&!animal(o.own.tiles[pos]))live[kind]++;
 int animals=live[G]+live[CO]+live[SH],land=std::popcount(unsigned(o.own.unlocked_mask));
 double cash=std::min(c.investment_budget(o,land,animals),o.own.money-committed_market_cost(c,o,a));
 if(cash<0)return best;
 Counts limits{75,75,c.p.max_tomato,c.p.max_strawberry,c.p.max_melon,0,0,0,0,c.p.max_geese,c.p.max_cows,c.p.max_sheep};
 Declared declared_value;
 if(!known&&c.p.intraday_declared_value){declared_value=declared(c,o,a);known=&declared_value;}
 auto vs=c.values(o,c.existing(o),known?known->selected:Counts{},known?known->feed:0);
 for(auto[value,pr]:vs){int kind=pr.kind;
  if(pr.out<=0||(!c.p.intraday_future_workforce&&value<=0)||live[kind]>=limits[kind])continue;
  if(kind>=9&&(o.day>c.p.latest_animal_day||animals>=c.p.max_animals))continue;
  for(int pos=0;pos<100;pos++)if(vacant(o,pos)&&!reservations.tiles[pos]&&c.plant_not_before[pos]<=o.day){
   auto j=project_job(c,o,pos,kind);if(j.actions.empty())continue;
   for(int u:idle){
    auto pl=route(o,u,j);if(o.hour+1+int(pl.a.size())>24)continue;
    Acts buy;double expense=0;int volume=0;
    for(int i=0;i<5;i++){int n=std::max(0,j.seeds[i]-std::max(0,o.priv.seeds[i]-reservations.seeds[i]));if(n)buy.push_back(action(Op::BUY_SEED,i,n));}
    for(int i=0;i<12;i++){
     int n=std::max(0,j.needs[i]-o.priv.inventories[u][i]-std::max(0,o.priv.shed[i]-reservations.shed[i]));
     if(n){buy.push_back(action(i>=9?Op::BUY_ANIMAL:Op::BUY_PRODUCT,i,n));volume+=n;}
    }
    if(!c.p.intraday_procurement&&!buy.empty())continue;
    if(buy.size()+a.market.size()>10||volume>100-sum(o.priv.shed))continue;
    for(auto x:buy)expense+=c.cost(o,x);if(expense>cash)continue;
    // Reuse the incumbent economic estimate. Added walking is explicit; this
    // does not claim a calibrated terminal-profit estimate or free future labour.
    double score=value-c.p.action_shadow*std::max(0,int(pl.a.size())-int(j.actions.size()));
    if(c.p.intraday_future_workforce){Proposal q{true,u,pos,kind,score,expense,j,buy};if(better(o,q,representatives[kind]))representatives[kind]=std::move(q);}
    if(score<=0)continue;
    if(!best.valid||std::tuple(score,-int(pl.a.size()),-expense,-u,-pos)>std::tuple(best.value,-int(route(o,best.unit,best.job).a.size()),-best.expense,-best.unit,-best.pos)){
     best={true,u,pos,kind,score,expense,j,std::move(buy)};
    }
   }
  }
 }
 return compare_workforce(c,o,a,representatives,best);
}
inline AdmissionInspection inspect(const Controller&c,const View&o,const PlayerAction&a,const Proposal&base){
 AdmissionInspection r;auto known=declared(c,o,a);
 if(!base.valid&&sum(known.planned)==0)return r;
 r.seen=true;r.step=o.step;r.planned=known.planned;r.selected=known.selected;
 r.duplicates=known.duplicates;r.late=known.late;r.feed=known.feed;
 auto alternative=sum(known.planned)?propose(c,o,a,&known):base;
 r.base_valid=base.valid;r.alternative_valid=alternative.valid;
 r.base_kind=base.kind;r.alternative_kind=alternative.kind;
 r.base_pos=base.pos;r.alternative_pos=alternative.pos;
 r.base_unit=base.unit;r.alternative_unit=alternative.unit;
 r.base_value=base.value;r.alternative_value=alternative.value;
 if(base.valid){
  for(auto[v,pr]:c.values(o,c.existing(o),{}))if(pr.kind==base.kind)r.same_kind_before=v;
  for(auto[v,pr]:c.values(o,c.existing(o),known.selected,known.feed))if(pr.kind==base.kind)r.same_kind_after=v;
 }return r;
}
inline void protect_admitted_inputs(const Controller&c,const View&o,PlayerAction&a){
 Counts protect{};
 for(int u:c.intraday_units)if(u<int(c.plans.size())){
  auto&pl=c.plans[u];for(size_t k=pl.index;k<pl.a.size();k++){auto x=pl.a[k];int i=int(x.item);if(x.op==Op::PICKUP&&i>=0&&i<12)protect[i]+=x.quantity;}
  if(u<int(a.units.size())){auto x=a.units[u];if(x.op==Op::PICKUP&&int(x.item)>=0)protect[int(x.item)]+=x.quantity;}
 }
 if(c.pending_admission.active){auto&pending=c.pending_admission;auto j=project_job(c,o,pending.pos,pending.kind);
  for(int i=0;i<12;i++)protect[i]+=std::max(0,j.needs[i]-o.priv.inventories.at(pending.unit)[i]);
 }
 for(auto&x:a.market)if(x.op==Op::SELL&&int(x.item)>=0&&protect[int(x.item)]>0)x.quantity=std::min(x.quantity,std::max(0,o.priv.shed[int(x.item)]-protect[int(x.item)]));
 a.market.erase(std::remove_if(a.market.begin(),a.market.end(),[](auto x){return x.quantity<=0;}),a.market.end());
}
inline void consider(Controller&c,const View&o,PlayerAction&a){
 if(!c.p.intraday_admission||c.phase!=3)return;
 c.intraday_checks++;
 if(!c.pending_admission.active){auto choice=propose(c,o,a);
  if(c.admission_inspection)*c.admission_inspection=inspect(c,o,a,choice);
  if(choice.valid){
  c.pending_admission={true,o.day,o.step,choice.unit,choice.pos,choice.kind};c.intraday_proposals++;
  a.market.insert(a.market.end(),choice.buy.begin(),choice.buy.end());c.intraday_purchase_orders+=int(choice.buy.size());
 }}
 protect_admitted_inputs(c,o,a);
}
}
