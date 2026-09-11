#pragma once
// Shared by real execution and the outer economic scenarios. Leaf probes keep
// the same executor/market guards, but do not recursively search alternatives.
#include <map>
namespace triad::route_detail {
using namespace dp7;
using Plans=std::vector<Plan>;
using Work=std::map<dp7::Controller::TaskKey,int>;
inline bool field(Action a){return !dp7::Controller::movement(a.op)&&a.op!=Op::PASS&&
 a.op!=Op::PICKUP&&a.op!=Op::DROP&&!(a.op==Op::PLACE&&int(a.item)<9);}
inline int wanted(Action a){return a.op==Op::HIRE||a.op==Op::BUY_LAND?1:std::max(0,a.quantity);}
inline bool required_order(Action a){return a.op==Op::HIRE||a.op==Op::BUY_PRODUCT||a.op==Op::BUY_SEED;}
inline bool procurement(Action a){return required_order(a)||a.op==Op::BUY_ANIMAL||a.op==Op::BUY_LAND;}
inline int deferred(const triad::Controller&c,const dp7::Controller::TaskKey&key){auto it=c.logistics.deferred_work.find(key);return it==c.logistics.deferred_work.end()?0:it->second;}
inline int purchase_priority(Action a){return a.op==Op::SELL?0:required_order(a)?1:2;}
inline int unfilled(const Acts&orders,const std::vector<int32_t>&fills,bool all=false){
 int n=0;for(size_t k=0;k<orders.size();k++)if(all?procurement(orders[k]):required_order(orders[k]))
  n+=std::max(0,wanted(orders[k])-(k<fills.size()?fills[k]:0));return n;
}
inline Counts reserves(const triad::Controller&c,const View&v){
 Counts q=intraday::reserved(c.core).shed;
 if(c.core.phase!=3){for(const auto&j:c.core.jobs(v))add(q,j.needs);
  for(const auto&bag:v.priv.inventories)for(int i=0;i<12;i++)q[i]=std::max(0,q[i]-bag[i]);}
 Counts promised{};
 for(auto[key,n]:c.logistics.required){auto[pos,op,item]=key;int done=0;
  if(auto it=c.logistics.completed.find(key);it!=c.logistics.completed.end())done=it->second;
  if(Op(op)==Op::FEED)promised[W]+=std::max(0,n-done-deferred(c,key));
  if(Op(op)==Op::FERTILIZE)promised[F]+=std::max(0,n-done-deferred(c,key));
 }
 for(const auto&bag:v.priv.inventories)for(int i:{W,F})promised[i]=std::max(0,promised[i]-bag[i]);
 for(int i:{W,F})q[i]=std::max(q[i],promised[i]);
 if(v.day<29&&v.hour<23){int feed=0;for(const auto&t:v.own.tiles)feed+=animal(t);
  for(const auto&bag:v.priv.inventories)feed-=bag[W];q[W]=std::max(q[W],feed);}
 if(v.day>=29)q={};return q;
}
inline int effect(const fastkag::Simulator&before,const fastkag::Simulator&after,int u,Action a){
 int pos=cell(u?before.farms()[0].hands[u-1]:before.farms()[0].farmer),i=int(a.item);
 const auto&t=before.farms()[0].tiles[pos];const auto&z=after.farms()[0].tiles[pos];
 const auto&b=before.privates()[0].inventories[u];const auto&e=after.privates()[0].inventories[u];
 switch(a.op){
  case Op::HARVEST:{int n=0;for(int k=0;k<8;k++)n+=std::max(0,e[k]-b[k]);return n;}
  case Op::COLLECT_FERTILIZER:return std::max(0,e[F]-b[F]);
  case Op::WATER:return plant(t)&&!t.watered_today&&z.watered_today;
  case Op::CARE:return animal(t)&&!t.cared_today&&z.cared_today;
  case Op::FEED:return animal(t)&&!t.fed_today&&z.fed_today&&b[W]>e[W];
  case Op::FERTILIZE:return b[F]>e[F];
  case Op::PLANT:return !plant(t)&&plant(z)&&int(z.crop)==i;
  case Op::DIG:return t.kind!=TileKind::EMPTY&&z.kind==TileKind::EMPTY;
  case Op::BUILD_COOP:return t.kind==TileKind::EMPTY&&z.kind==TileKind::COOP;
  case Op::BUILD_PASTURE:return t.kind==TileKind::EMPTY&&z.kind==TileKind::PASTURE;
  case Op::PLACE:return i>=9&&!animal(t)&&animal(z)&&int(z.animal)==i;
  default:return 0;
 }
}
inline bool preserves(const triad::Controller::RouteValue&next,const triad::Controller::RouteValue&keep){
 for(const auto&[key,n]:keep.completed){auto it=next.completed.find(key);if(it==next.completed.end()||it->second<n)return false;}
 for(const auto&[key,n]:keep.harvested){auto it=next.harvested.find(key);if(it==next.harvested.end()||it->second<n)return false;}
 return next.unfilled<=keep.unfilled&&next.overflow<=keep.overflow;
}
inline bool better(const triad::Controller::RouteValue&a,const triad::Controller::RouteValue&b,bool economic){
 if(a.unfilled!=b.unfilled)return a.unfilled<b.unfilled;
 if(a.overflow!=b.overflow)return a.overflow<b.overflow;
 if(economic){
  // Next morning is a declared forecast, not an observed obligation.
  if(a.next_unfilled!=b.next_unfilled)return a.next_unfilled<b.next_unfilled;
  if(std::abs(a.score-b.score)>1e-6)return a.score>b.score;
 }else if(a.assets<b.assets-1e-6)return false;
 if(a.moves!=b.moves)return a.moves<b.moves;
 return a.cash_area>b.cash_area+1e-6;
}
inline Plans remaining(const dp7::Controller&c){Plans ps;for(int u=0;u<int(c.plans.size());u++)ps.push_back(service::remaining(c,u));return ps;}
inline Work committed(const triad::Controller&c,const View&o){
 Work work;
 if(c.core.phase==3){for(const auto&p:c.core.plans)for(size_t k=p.index;k<p.a.size();k++)
  if(field(p.a[k]))work[dp7::Controller::task_key(p.target[k],p.a[k])]++;}
 else for(const auto&j:c.core.jobs(o))for(auto a:j.actions)if(field(a))work[dp7::Controller::task_key(j.pos,a)]++;
 return work;
}
inline void restrict_work(triad::Controller::RouteValue&v,const Work&work){
 for(auto it=v.completed.begin();it!=v.completed.end();)if(!work.count(it->first))it=v.completed.erase(it);
  else {it->second=std::min(it->second,work.at(it->first));++it;}
 for(auto it=v.harvested.begin();it!=v.harvested.end();)if(!work.count(it->first))it=v.harvested.erase(it);else ++it;
}
inline bool outstanding(const triad::Controller&c,const triad::Controller::RouteValue&v){
 for(auto[key,n]:c.logistics.required){int done=0,future=0;
  if(auto it=c.logistics.completed.find(key);it!=c.logistics.completed.end())done=it->second;
  if(auto it=v.completed.find(key);it!=v.completed.end())future=it->second;
  if(done+future+deferred(c,key)<n)return false;
 }return true;
}
inline void promise(triad::Controller&c,const triad::Controller::RouteValue&v){
 for(auto[key,n]:v.completed)c.logistics.required[key]=std::max(c.logistics.required[key],c.logistics.completed[key]+n);
}
inline bool prepared_complete(const triad::Controller&source,const View&o){
 auto prepared=source.core.project_preparation(o);
 if(o.hour+prepared.elapsed>=24)return false;
 View v{o.step+prepared.elapsed,o.day,o.hour+prepared.elapsed,prepared.farm,o.opponent,prepared.priv,prepared.market,o.shops};
 auto c=source.core;Work required;
 for(const auto&j:c.jobs(v))for(auto a:j.actions)if(field(a))required[dp7::Controller::task_key(j.pos,a)]++;
 c.compile(v);Work scheduled;
 for(const auto&p:c.plans)for(size_t k=0;k<p.a.size();k++)if(field(p.a[k]))scheduled[dp7::Controller::task_key(p.target[k],p.a[k])]++;
 if(c.actual_drop!=0||c.resource_degraded!=0)return false;
 for(auto[key,n]:required)if(scheduled[key]<n)return false;
 return true;
}
inline bool inputs(const View&o,const Plans&ps){
 Counts pickups{},seeds{};std::set<int>harvested,collected;
 for(size_t u=0;u<ps.size();u++){
  auto held=o.priv.inventories[u];
  for(size_t k=0;k<ps[u].a.size();k++){
   auto a=ps[u].a[k];int i=int(a.item),pos=ps[u].target[k];
   if(a.op==Op::PICKUP){pickups[i]+=a.quantity;held[i]+=a.quantity;}
   if(a.op==Op::DROP)held={};
   if(a.op==Op::FEED&&--held[W]<0)return false;
   if(a.op==Op::FERTILIZE&&--held[F]<0)return false;
   if(a.op==Op::PLACE&&(held[i]-=a.quantity)<0)return false;
   if(a.op==Op::PLANT)seeds[i]++;
   if(pos>=0&&a.op==Op::COLLECT_FERTILIZER&&collected.insert(pos).second&&o.own.tiles[pos].fertilizer_available)held[F]++;
   if(pos>=0&&a.op==Op::HARVEST&&harvested.insert(pos).second){const auto&t=o.own.tiles[pos];
    int item=animal(t)?product[int(t.animal)-9]:plant(t)?int(t.crop):-1;if(item>=0)held[item]+=t.yield_units;}
  }
 }
 for(int i=0;i<12;i++)if(pickups[i]>o.priv.shed[i])return false;
 for(int i=0;i<5;i++)if(seeds[i]>o.priv.seeds[i])return false;
 return true;
}
inline auto order(const Plans&ps){
 auto copy=ps;for(auto&p:copy)for(size_t k=0;k<p.a.size();k++)if(!field(p.a[k]))p.target[k]=-1;
 return exchange::order(copy);
}
inline bool fits(const View&o,const Plans&ps){
 for(const auto&p:ps)if(int(p.a.size())>std::min(24-o.hour,719-o.step))return false;
 return true;
}
struct Proof {Work completed,harvested;bool valid=true;};
inline bool covers(const Work&done,const Work&required){
 for(auto[key,n]:required){auto it=done.find(key);if(it==done.end()||it->second<n)return false;}return true;
}
// Execute the fixed schedule with the existing simulator. This is a work proof,
// not a price forecast: it includes unit ordering, inventory, crop clocks and
// the actual last action deadline. No hidden state or opponent action is used.
inline Proof verify(const View&o,const Plans&ps){
 Proof result;if(ps.size()!=o.priv.inventories.size()||!fits(o,ps)){result.valid=false;return result;}
 fastkag::ObservedDayScenario world(o);
 for(size_t tick=0;!world.finished();tick++){
  PlayerAction out;out.units.resize(ps.size());bool any=false;
  for(size_t u=0;u<ps.size();u++)if(tick<ps[u].a.size()){out.units[u]=ps[u].a[tick];any=true;}
  if(!any)break;
  auto before=world.project_units(out.units,0);
  for(size_t u=0;u<ps.size();u++){
   auto a=out.units[u];auto after=world.project_units(out.units,int(u)+1);
   if(field(a)){
    int pos=cell(u?before.farms()[0].hands[u-1]:before.farms()[0].farmer);
    int n=effect(before,after,int(u),a);
    if(n<=0||pos!=ps[u].target[tick])result.valid=false;
    if(n>0){auto key=dp7::Controller::task_key(pos,a);result.completed[key]++;if(a.op==Op::HARVEST)result.harvested[key]+=n;}
   }
   before=std::move(after);
  }
  world.advance(out);
 }return result;
}
inline Work debt(const triad::Controller&c){
 Work work;for(auto[key,n]:c.logistics.required){auto it=c.logistics.completed.find(key);
  int left=n-(it==c.logistics.completed.end()?0:it->second)-deferred(c,key);if(left>0)work[key]=left;}return work;
}
// Add only missing route inputs. Later native verification checks the causal
// harvest/collection credits and aggregate warehouse contention exactly.
inline Plan supplied(const View&o,int u,const Plan&base){
 auto held=o.priv.inventories[u];Counts missing{};std::set<int>harvested,collected;
 for(auto[a,pos]:dp7::Controller::plan_semantics(base)){
  int i=int(a.item);if(a.op==Op::PICKUP)held[i]+=a.quantity;
  if(a.op==Op::DROP)held={};
  int resource=a.op==Op::FEED?W:a.op==Op::FERTILIZE?F:a.op==Op::PLACE?i:-1;
  if(resource>=0){int q=a.op==Op::PLACE?a.quantity:1;int deficit=std::max(0,q-held[resource]);missing[resource]+=deficit;held[resource]+=deficit-q;}
  const auto&t=o.own.tiles[pos];
  if(a.op==Op::COLLECT_FERTILIZER&&collected.insert(pos).second&&t.fertilizer_available)held[F]++;
  if(a.op==Op::HARVEST&&harvested.insert(pos).second){int item=animal(t)?product[int(t.animal)-9]:plant(t)?int(t.crop):-1;if(item>=0)held[item]+=t.yield_units;}
 }
 if(!sum(missing))return base;
 auto sem=dp7::Controller::plan_semantics(base);Plan out;int pos=cell(u?o.own.hands[u-1]:o.own.farmer),dest=depot[0];
 int next=sem.empty()?pos:sem.front().second;
 for(int d:depot)if(dist(pos,d)+dist(d,next)<dist(pos,dest)+dist(dest,next))dest=d;
 dp7::Controller::walk(out,pos,dest);
 for(int i=0;i<12;i++)if(missing[i]){out.a.push_back(action(Op::PICKUP,i,missing[i]));out.target.push_back(dest);}
 for(auto[a,target]:sem){dp7::Controller::walk(out,pos,target);out.a.push_back(a);out.target.push_back(target);}return out;
}
inline int travel(const Plans&ps){int n=0;for(const auto&p:ps)for(auto a:p.a)n+=dp7::Controller::movement(a.op);return n;}
inline Plan rebuild(const View&o,int u,const exchange::Groups&gs){
 Plan p;int pos=cell(u?o.own.hands[u-1]:o.own.farmer);
 for(const auto&g:gs){dp7::Controller::walk(p,pos,g.pos);for(auto a:g.actions){p.a.push_back(a);p.target.push_back(g.pos);}}return p;
}
inline bool movable(const exchange::Group&g){return std::all_of(g.actions.begin(),g.actions.end(),[](auto a){
 return a.op==Op::HARVEST||a.op==Op::COLLECT_FERTILIZER||a.op==Op::WATER||a.op==Op::CARE;});}
inline Plans shorter(const View&o,const Plans&base){
 auto best=base;auto causal=order(base);int bestcost=travel(base);
 std::vector<exchange::Groups>groups;for(const auto&p:base)groups.push_back(exchange::groups(p));
 // ponytail: 256 deterministic relocation trials; raise only if profiling and
 // paired games justify a larger local-search budget.
 uint64_t rng=0x9e3779b97f4a7c15ULL+o.step;
 auto pick=[&](int n){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return int(rng%unsigned(n));};
 for(int t=0;t<256;t++){
  int u=pick(groups.size()),v=pick(groups.size());if(groups[u].empty())continue;
  int k=pick(groups[u].size());if(!movable(groups[u][k]))continue;
  auto trial_groups=groups;auto job=trial_groups[u][k];trial_groups[u].erase(trial_groups[u].begin()+k);
  int at=pick(trial_groups[v].size()+1);trial_groups[v].insert(trial_groups[v].begin()+at,job);
  auto ps=best;ps[u]=route_detail::rebuild(o,u,trial_groups[u]);ps[v]=route_detail::rebuild(o,v,trial_groups[v]);
  int cost=travel(ps);if(cost>=bestcost||!fits(o,ps)||order(ps)!=causal)continue;
  best=std::move(ps);groups=std::move(trial_groups);bestcost=cost;
 }return best;
}
struct Delivery {Plan plan;int unit=-1;double priority=0;};
inline std::vector<Delivery> deliveries(const View&o,const Plans&base){
 std::vector<Delivery>out;
 for(int u=0;u<int(base.size());u++){
  auto sem=dp7::Controller::plan_semantics(base[u]);Counts held=o.priv.inventories[u];std::set<int>harvested,collected;
  for(int at=0;at<=int(sem.size());at++){
   Counts need{};for(int k=at;k<int(sem.size());k++){
    auto a=sem[k].first;if(a.op==Op::FEED)need[W]++;if(a.op==Op::FERTILIZE)need[F]++;
    if(a.op==Op::PLACE&&int(a.item)>=9)need[int(a.item)]+=a.quantity;
   }
   Counts surplus{};for(int i=0;i<9;i++)surplus[i]=std::max(0,held[i]-need[i]);
   if(sum(surplus)>0&&(at==0||sem[at-1].first.op==Op::HARVEST||sem[at-1].first.op==Op::COLLECT_FERTILIZER||at==int(sem.size()))){
    Plan p;int pos=cell(u?o.own.hands[u-1]:o.own.farmer);
    for(int k=0;k<at;k++){dp7::Controller::walk(p,pos,sem[k].second);p.a.push_back(sem[k].first);p.target.push_back(pos);}
    int next=at<int(sem.size())?sem[at].second:pos,dep=depot[0];
    for(int d:depot)if(dist(pos,d)+dist(d,next)<dist(pos,dep)+dist(dep,next))dep=d;
    dp7::Controller::walk(p,pos,dep);double value=0;
    if(sum(surplus)==sum(held)){
     p.a.push_back(action(Op::DROP));p.target.push_back(dep);
     for(int i=0;i<9;i++)value+=revenue(i,o.market.inventory[i],surplus[i]);
    }else{
     int item=-1;for(int i=0;i<9;i++)if(surplus[i]>0&&(item<0||surplus[i]*o.market.prices[i]>surplus[item]*o.market.prices[item]))item=i;
     p.a.push_back(action(Op::PLACE,item,surplus[item]));p.target.push_back(dep);
     value=revenue(item,o.market.inventory[item],surplus[item]);
    }
    for(int k=at;k<int(sem.size());k++){dp7::Controller::walk(p,pos,sem[k].second);p.a.push_back(sem[k].first);p.target.push_back(pos);}
    if(int(p.a.size())<=std::min(24-o.hour,719-o.step)){
     int extra=std::max(1,int(p.a.size())-int(base[u].a.size()));
     out.push_back({std::move(p),u,value/extra});
    }
   }
   if(at==int(sem.size()))break;
   auto a=sem[at].first;int pos=sem[at].second,i=int(a.item);const auto&t=o.own.tiles[pos];
   if(a.op==Op::PICKUP&&i>=0)held[i]+=a.quantity;
   if(a.op==Op::DROP)held={};
   if(a.op==Op::PLACE&&i>=0)held[i]=std::max(0,held[i]-a.quantity);
   if(a.op==Op::FEED)held[W]=std::max(0,held[W]-1);
   if(a.op==Op::FERTILIZE)held[F]=std::max(0,held[F]-1);
   if(a.op==Op::HARVEST&&harvested.insert(pos).second){int product_id=animal(t)?product[int(t.animal)-9]:plant(t)?int(t.crop):-1;
    if(product_id>=0)held[product_id]+=t.yield_units;}
   if(a.op==Op::COLLECT_FERTILIZER&&animal(t)&&t.fertilizer_available&&collected.insert(pos).second)held[F]++;
  }
 }
 std::stable_sort(out.begin(),out.end(),[](const auto&a,const auto&b){return a.priority>b.priority;});return out;
}
} // namespace triad::route_detail

namespace triad {
inline void Controller::route_reconcile(const View&o){
 auto&l=logistics;
 if(l.promised_day!=o.day){l.promised_day=o.day;l.required.clear();l.completed.clear();l.witness.clear();l.witness_step=-1;l.deferred_work.clear();l.recovery_pending=false;l.recovery_waiting=false;}
 if(!l.probe&&l.recovery_allowed&&l.expected_step==o.step&&l.expected_day==o.day){
  bool missing=l.short_order;
  if(l.purchased){
   missing|=l.expected_hands>int(o.own.hands.size())||(l.expected_land&~o.own.unlocked_mask)!=0;
   for(int i=0;i<12;i++)missing|=l.expected_shed[i]>o.priv.shed[i];
   for(int i=0;i<5;i++)missing|=l.expected_seeds[i]>o.priv.seeds[i];
  }
  // Final route_finance checks the complete unit -> sale -> purchase batch.
  // A lower cash balance alone does not imply missing inputs or a failed order.
  if(missing)l.recovery_waiting=false;
  if(missing&&!l.recovery_pending){l.recovery_pending=true;l.recovery_events++;}
 }
 if(l.expected_step==o.step&&l.expected_day==o.day&&
    (l.expected_hands!=int(o.own.hands.size())||l.expected_shed!=o.priv.shed||l.expected_seeds!=o.priv.seeds)){
  l.fill_mismatches++;l.dirty=true;l.replans++;
  // A new worker gets an empty route; existing workers retain their cargo and
  // unfinished chains. Recoordination then uses the authoritative observation.
  if(core.phase==3){core.plans.resize(o.priv.inventories.size());core.recoordinate(o);}
 }
 if(core.phase==3&&core.plans.size()!=o.priv.inventories.size()){
  core.plans.resize(o.priv.inventories.size());core.recoordinate(o);l.dirty=true;l.replans++;
 }
}
inline void Controller::route_restore(const View&o){
 using namespace route_detail;
 if(core.plans.size()!=o.priv.inventories.size()||logistics.required.empty())return;
 auto base=remaining(core);auto due=debt(*this);
 if(logistics.witness_step>=0){
  // Reuse the constructive proof when market-dependent heuristic ordering
  // cannot reproduce it. Recheck against observed positions and resources.
  auto proof=verify(o,base);
  if(!proof.valid||!covers(proof.completed,due)){
   auto fallback=logistics.witness;
   if(fallback.size()==base.size()){
    for(size_t u=0;u<fallback.size();u++){int stale=0;fallback[u]=core.repair_plan(o,int(u),fallback[u],stale);}
    auto checked=verify(o,fallback);
    if(checked.valid&&covers(checked.completed,due)){base=std::move(fallback);core.plans=base;logistics.witness_used++;logistics.dirty=true;}
    else logistics.proof_rejected++;
   }
  }
  logistics.witness.clear();logistics.witness_step=-1;
 }
 Work covered;
 for(const auto&p:base)for(size_t k=0;k<p.a.size();k++)if(field(p.a[k]))covered[dp7::Controller::task_key(p.target[k],p.a[k])]++;
 std::set<int>missing;
 for(auto[key,n]:due)if(covered[key]<n)missing.insert(std::get<0>(key));
 if(missing.empty())return;
 auto raw=core;raw.p.shared_task_atoms_v2=raw.p.split_service_jobs=false;
 auto jobs=raw.jobs(o);
 for(int pos:missing){
  budget::check();
  Work wanted;std::set<dp7::Controller::TaskKey>remove;
  for(auto[key,n]:covered)if(std::get<0>(key)==pos&&n>0){wanted[key]=n;remove.insert(key);}
  for(auto[key,n]:due)if(std::get<0>(key)==pos)wanted[key]=std::max(wanted[key],n);
  Job chain;chain.pos=pos;auto left=wanted;
  for(const auto&j:jobs)if(j.pos==pos)for(auto a:j.actions){auto key=dp7::Controller::task_key(pos,a);if(left[key]>0){chain.actions.push_back(a);left[key]--;}}
  // Standalone obligations can outlive the economic target list. Preserve
  // them if currently valid; dependent chains require their canonical order.
  bool known=std::all_of(left.begin(),left.end(),[](const auto&x){return x.second==0;});
  if(!known&&wanted.size()==1){auto[key,n]=*wanted.begin();auto[p,op,item]=key;auto a=action(Op(op),item);
   if(core.semantic_needed(o,0,a,p)){chain.actions.assign(n,a);known=true;}}
  if(!known){logistics.unresolved++;continue;}
  auto stripped=base;
  for(size_t u=0;u<base.size();u++)stripped[u]=core.strip_tasks(o,int(u),base[u],remove);
  auto keep=verify(o,base);Plans chosen;int best=100000;
  for(int u=0;u<int(base.size());u++)for(int at=0;at<=int(dp7::Controller::plan_semantics(stripped[u]).size());at++){
   budget::check();
   auto trial=stripped;trial[u]=supplied(o,u,dp7::Controller::insert_service(o,u,stripped[u],chain,at));
   int cost=0;for(const auto&p:trial)cost+=int(p.a.size());
   if(cost>=best||!fits(o,trial)||!inputs(o,trial))continue;
   auto checked=verify(o,trial);
   if(!checked.valid||!covers(checked.completed,keep.completed)||!covers(checked.harvested,keep.harvested)||!covers(checked.completed,wanted))continue;
   chosen=std::move(trial);best=cost;
  }
  if(chosen.empty()){logistics.unresolved++;continue;}
  for(auto[key,n]:wanted){logistics.restored+=std::max(0,n-covered[key]);covered[key]=n;}
  base=std::move(chosen);core.plans=base;logistics.chain_repairs++;logistics.dirty=true;
 }
}
inline void Controller::route_capacity(const View&o,PlayerAction&out){
 fastkag::ObservedDayScenario world(o);
 for(int u=0;u<int(out.units.size());u++)if(out.units[u].op==Op::DROP||(out.units[u].op==Op::PLACE&&int(out.units[u].item)>=0&&int(out.units[u].item)<9)){
  auto prefix=world.project_units(out.units,u);int pos=cell(u?prefix.farms()[0].hands[u-1]:prefix.farms()[0].farmer);
  if(!at_depot(pos))continue;
  const auto&bag=prefix.privates()[0].inventories[u];int room=std::max(0,100-sum(prefix.privates()[0].shed));
  auto requested=out.units[u];int quantity=requested.op==Op::DROP?sum(bag):std::min(bag[int(requested.item)],requested.quantity);
  if(quantity<=room)continue;
  // DROP destroys overflow. Quantitative PLACE keeps everything that did not
  // fit in the backpack. Retain the original DROP for the following step.
  int item=requested.op==Op::PLACE?int(requested.item):-1;
  if(item<0)for(int i=0;i<9;i++)if(bag[i]>0&&(item<0||bag[i]*o.market.prices[i]>bag[item]*o.market.prices[item]))item=i;
  out.units[u]=room>0&&item>=0?action(Op::PLACE,item,std::min(room,quantity)):action(Op::PASS);
  if(requested.op==Op::DROP&&room>0&&item>=0)out.units[u].quantity=std::min(room,bag[item]);
  if(u<int(core.plans.size())){auto&pl=core.plans[u];if(pl.index>0&&dp7::Controller::same_action(pl.a[pl.index-1],requested))pl.index--;}
  logistics.capacity_repairs++;logistics.dirty=true;
 }
}
inline void Controller::route_finance(const View&o,PlayerAction&out,bool fund_all){
 using namespace route_detail;
 fastkag::ObservedDayScenario world(o);auto post=world.project_units(out.units,-1);
 View v{o.step,o.day,o.hour,post.farms()[0],post.farms()[1],post.privates()[0],post.market(),o.shops};
 auto reserve=reserves(*this,v);Counts available{};
 for(int i=0;i<9;i++)available[i]=std::max(0,v.priv.shed[i]-reserve[i]);
 auto projected=post.project_own_market(0,out.market);
 bool shortage=unfilled(out.market,projected.last_market_fills()[0],fund_all)>0;
 if(shortage&&out.market.size()==10&&core.phase!=3&&o.hour<22&&sum(available)>0){
  // Make an actual order slot for funding; retain the displaced purchase for
  // the next preparation step instead of silently losing it to truncation.
  core.queue.insert(core.queue.begin(),out.market.back());out.market.pop_back();core.phase=1;
  projected=post.project_own_market(0,out.market);
 }
 // Clear room before the next unit phase, including the night auto-deposit.
 int carried=0;
 for(size_t u=0;u<v.priv.inventories.size();u++){
  const auto&bag=v.priv.inventories[u];int due=o.hour==23?sum(bag):0;
  if(o.hour!=23&&u<core.plans.size()){const auto&p=core.plans[u];if(p.index<p.a.size()){
   auto a=p.a[p.index];int item=int(a.item);
   if(a.op==Op::DROP)due=sum(bag);
   else if(a.op==Op::PLACE&&item>=0&&item<9)due=std::min(bag[item],std::max(0,a.quantity));
  }}
  carried+=due;
 }
 bool capacity=sum(v.priv.shed)+carried>100;
 if(shortage||capacity){
  Acts buys;Counts sell{};for(auto a:out.market){if(a.op==Op::SELL&&int(a.item)>=0&&int(a.item)<9)sell[int(a.item)]+=a.quantity;else buys.push_back(a);}
  if(shortage)std::stable_sort(buys.begin(),buys.end(),[](auto a,auto b){return purchase_priority(a)<purchase_priority(b);});
  auto assemble=[&](Counts quantities){
   auto sales=core.sales_sorted(o,quantities);auto chosen=buys;
   while(!fund_all&&sales.size()+chosen.size()>10&&!chosen.empty()&&!required_order(chosen.back()))chosen.pop_back();
   if(sales.size()+chosen.size()>10)sales.resize(10-chosen.size());
   sales.insert(sales.end(),chosen.begin(),chosen.end());return sales;
  };
  auto candidate=assemble(sell);
  auto sufficient=[&](const Acts&orders){auto p=post.project_own_market(0,orders);
   return unfilled(orders,p.last_market_fills()[0],fund_all)==0&&sum(p.privates()[0].shed)+carried<=100;};
  for(auto offer:core.sales_sorted(o,available)){
   if(sufficient(candidate))break;
   int item=int(offer.item),limit=std::max(0,available[item]-sell[item]);
   int low=1,high=limit,chosen=limit;
   while(low<=high){int mid=(low+high)/2;auto q=sell;q[item]+=mid;
    if(sufficient(assemble(q))){chosen=mid;high=mid-1;}else low=mid+1;}
   sell[item]+=chosen;candidate=assemble(sell);
  }
  auto funded=post.project_own_market(0,candidate);
  if(unfilled(candidate,funded.last_market_fills()[0],fund_all)<=unfilled(out.market,projected.last_market_fills()[0],fund_all)){
   bool changed=candidate.size()!=out.market.size();for(size_t k=0;k<candidate.size()&&!changed;k++)changed=!dp7::Controller::same_action(candidate[k],out.market[k]);
   if(changed){out.market=std::move(candidate);projected=std::move(funded);logistics.finance_repairs++;}
  }
 }
 logistics.expected_step=o.step+1;logistics.expected_day=o.day;
 logistics.expected_hands=projected.farms()[0].hands.size();logistics.expected_shed=projected.privates()[0].shed;
 logistics.expected_seeds=projected.privates()[0].seeds;
 logistics.expected_cash=projected.farms()[0].money;logistics.expected_land=projected.farms()[0].unlocked_mask;
 logistics.purchased=false;logistics.short_order=false;
 for(size_t k=0;k<out.market.size();k++)if(procurement(out.market[k])){
  logistics.purchased=true;logistics.short_order|=wanted(out.market[k])>projected.last_market_fills()[0][k];
 }
 // Own unit effects precede all market orders and are deterministic from the
 // observation. Account once, so later searches cannot forget earlier work.
 if(!logistics.required.empty()){
  auto before=world.project_units(out.units,0);
  for(int u=0;u<int(out.units.size());u++){
   auto after=world.project_units(out.units,u+1);auto a=out.units[u];
   if(field(a)&&effect(before,after,u,a)>0){int pos=cell(u?before.farms()[0].hands[u-1]:before.farms()[0].farmer);
    logistics.completed[dp7::Controller::task_key(pos,a)]++;}
   before=std::move(after);
  }
 }
}
inline Controller::RouteValue Controller::route_evaluate(const View&o,bool continuation,const PlayerAction*first)const{
 using namespace route_detail;
 auto c=*this;c.logistics.probe=true;c.previous_step=o.step-1;c.core.last_step=o.step-1;
 c.core.resume_compiled_tick=c.core.phase==3;c.core.admission_inspection=nullptr;
 fastkag::PublicFlowScenario world(o,model.rival,s.supply);RouteValue value;
 while(!world.done()&&world.view().day==o.day){
  budget::check();auto v=world.view();auto a=first&&v.step==o.step?*first:c.act(v);
  if(first&&v.step==o.step){c.previous_step=o.step;c.core.last_step=o.step;}
  if(value.witness_step<0&&core.phase!=3&&c.core.phase==3){
   value.witness=c.core.plans;for(auto&p:value.witness)p.index=0;value.witness_step=v.step;
   auto proof=verify(v,value.witness);value.certificate=proof.valid;
   value.certified=value.completed;value.certified_harvest=value.harvested;
   for(auto[key,n]:proof.completed)value.certified[key]+=n;
   for(auto[key,n]:proof.harvested)value.certified_harvest[key]+=n;
  }
  auto before=world.project_units(a.units,0);
  for(int u=0;u<int(a.units.size());u++){
   auto after=world.project_units(a.units,u+1);auto action=a.units[u];
   if(field(action)){int n=effect(before,after,u,action);if(n>0){
    int pos=cell(u?before.farms()[0].hands[u-1]:before.farms()[0].farmer);auto key=dp7::Controller::task_key(pos,action);
    value.completed[key]++;if(action.op==Op::HARVEST)value.harvested[key]+=n;
   }}
   if(action.op==Op::DROP){int total_before=sum(before.privates()[0].shed)+sum(before.privates()[0].inventories[u]);
    int total_after=sum(after.privates()[0].shed)+sum(after.privates()[0].inventories[u]);value.overflow+=std::max(0,total_before-total_after);}
   value.moves+=dp7::Controller::movement(action.op);before=std::move(after);
  }
  world.advance(a);value.unfilled+=unfilled(a.market,world.fills());value.overflow+=world.overflow();
  value.cash_area+=world.own_cash();value.ticks++;
 }
 value.cash=world.own_cash();value.assets=value.cash+world.liquidation_quote();
 value.score=value.assets-s.competition*world.rival_cash();
 if(continuation&&!world.done()){
  budget::check();
  // Fixed next-day planning boundary. PublicFlowScenario keeps known shops and
  // expected unknown demand; neither a replay suffix nor the game seed is used.
  auto endpoint=world.view();c.plan(endpoint);
  value.score=value.cash-s.competition*world.rival_cash()+c.predicted;
  for(int t=0;t<4&&!world.done()&&c.core.phase==1;t++){
   auto a=c.act(world.view());world.advance(a);value.next_unfilled+=unfilled(a.market,world.fills());
  }
 }
 return value;
}
inline void Controller::route_prepare(const View&o){
 using namespace route_detail;
 if(core.phase!=1)return;
 // Necessary staff/inputs precede optional expansion only when the complete
 // preparation sequence is not funded in the own-order projection.
 auto projected=core.project_preparation(o);int hires=0;for(auto a:core.queue)hires+=a.op==Op::HIRE;
 if(int(projected.farm.hands.size())<int(o.own.hands.size())+hires){
  std::stable_sort(core.queue.begin(),core.queue.end(),[](auto a,auto b){return purchase_priority(a)<purchase_priority(b);});
 }
 if constexpr(R2_ROUTE_ECONOMICS<3)return;
 if(logistics.probe||logistics.prepared_day==o.day||hires==0)return;
 budget::check();
 logistics.prepared_day=o.day;logistics.hire_checks++;
 auto trial=*this;
 auto last=std::find_if(trial.core.queue.rbegin(),trial.core.queue.rend(),[](auto a){return a.op==Op::HIRE;});
 trial.core.queue.erase(std::next(last).base());
 if(!prepared_complete(trial,o)){logistics.unsafe++;return;}
 auto work=committed(*this,o);auto keep=route_evaluate(o,true);restrict_work(keep,work);promise(trial,keep);
 auto next=trial.route_evaluate(o,true);logistics.evaluations+=2;
 // Never buy a wage saving by losing work or relying on an unfilled order.
 bool certified=next.certificate&&covers(next.certified,keep.completed)&&covers(next.certified_harvest,keep.harvested);
 if(!certified)logistics.proof_rejected++;
 if(certified&&preserves(next,keep)&&outstanding(*this,next)&&better(next,keep,true)){
  core.queue=std::move(trial.core.queue);logistics.hire_changes++;
  promise(*this,keep);
  logistics.witness=std::move(next.witness);logistics.witness_step=next.witness_step;
 }
}
inline void Controller::route_choose(const View&o){
 using namespace route_detail;
 budget::check();
 logistics.checks++;if(core.plans.empty()||core.plans.size()!=o.priv.inventories.size())return;
 int cargo=0;for(const auto&bag:o.priv.inventories)cargo+=sum(bag);
 bool first=logistics.searched_day!=o.day;
 bool pressure=o.day>=29||o.own.money<s.reserve||core.expected_auto_deposit(o)+sum(o.priv.shed)>100;
 if(!first&&!logistics.dirty&&(!pressure||cargo==0||o.step-logistics.last_search<6))return;
 if(!first&&o.step-logistics.last_search<2)return;
 logistics.searched_day=o.day;logistics.last_search=o.step;logistics.dirty=false;
 auto base=remaining(core);auto causal=order(base);std::vector<Plans>candidates{base};
 auto add=[&](Plans ps){
  if(candidates.size()>=6||!fits(o,ps)||order(ps)!=causal)return;
  for(const auto&prior:candidates){bool same=true;for(size_t u=0;u<ps.size();u++)same&=dp7::Controller::same_remaining(ps[u],prior[u]);if(same)return;}
  candidates.push_back(std::move(ps));
 };
 auto compact=shorter(o,base);add(compact);
 auto deliveries_list=deliveries(o,base);auto combined=base;std::set<int>changed;
 for(const auto&d:deliveries_list){if(changed.insert(d.unit).second)combined[d.unit]=d.plan;}
 add(std::move(combined));
 for(const auto&d:deliveries_list){auto ps=base;ps[d.unit]=d.plan;add(std::move(ps));if(candidates.size()>=6)break;}
 if(candidates.size()<2)return;
 auto work=committed(*this,o);auto keep=route_evaluate(o,false),best=keep;restrict_work(keep,work);logistics.evaluations++;int selected=0;
 std::vector<std::tuple<int,int,double,double,int>>shortlist;
 for(int i=1;i<int(candidates.size());i++){
  auto trial=*this;trial.core.plans=candidates[i];promise(trial,keep);
  auto v=trial.route_evaluate(o,false);logistics.evaluations++;
  if(!preserves(v,keep)||!outstanding(*this,v)){logistics.unsafe++;continue;}
  if constexpr(R2_ROUTE_ECONOMICS>=3)shortlist.emplace_back(v.unfilled,v.overflow,-v.assets,-v.cash_area,i);
  else if(better(v,best,false)){best=std::move(v);selected=i;}
 }
 if constexpr(R2_ROUTE_ECONOMICS>=3)if(!shortlist.empty()){
  std::stable_sort(shortlist.begin(),shortlist.end());
  best=route_evaluate(o,true);logistics.evaluations++;
  for(size_t k=0;k<std::min<size_t>(2,shortlist.size());k++){
   int i=std::get<4>(shortlist[k]);auto trial=*this;trial.core.plans=candidates[i];promise(trial,keep);
   auto v=trial.route_evaluate(o,true);logistics.evaluations++;
   if(preserves(v,keep)&&outstanding(*this,v)&&better(v,best,true)){best=std::move(v);selected=i;}
  }
 }
 if(selected){core.plans=std::move(candidates[selected]);logistics.changes++;
  promise(*this,keep);
 }
}
} // namespace triad
