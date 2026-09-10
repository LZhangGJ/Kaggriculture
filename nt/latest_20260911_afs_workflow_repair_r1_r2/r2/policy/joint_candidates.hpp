#pragma once
// Structural exchanges, not an opponent-specific opening. The prefilter below
// is only a conditional stream estimate. Real admission is the common event
// rollout, including sourcing feed, wages, delivery and successor PLANT/WATER.
#ifndef P16_AFS_LOCAL_EXCHANGE
#define P16_AFS_LOCAL_EXCHANGE 0
#endif
namespace triad {
struct JointCandidate {int id=16;Controller policy;int removed=0,feeders=0,age=0,next=-1;double hint=0;};
inline std::vector<JointCandidate> joint_candidates(const Controller&live,const Controller&keep,const View&o){
 std::vector<JointCandidate> out;
 if constexpr(P16_JOINT_BUNDLES==0)return out;
 if(live.joint.active||o.day+4>=29)return out;
 std::vector<int>removable,empty;int total_animals=0,actual_animals=0;
 for(const auto&t:o.own.tiles)actual_animals+=animal(t);
 for(auto[pos,k]:keep.core.target){
  const auto&t=o.own.tiles[pos];if(k>=9)total_animals++;
  bool vacant=!animal(t)&&!plant(t)&&t.kind!=TileKind::LOCKED;
  if(!vacant)continue;
  if(k<0)empty.push_back(pos);
  if(k>=9&&!keep.book[pos].funded&&keep.paths[pos].first_cost>0){
   int paid=o.priv.shed[k];for(const auto&iv:o.priv.inventories)paid+=iv[k];
   if(paid==0)removable.push_back(pos);
  }
 }
 if(removable.empty())return out;
 // Least valuable unpurchased animal removed first, ties are geometric only.
 std::stable_sort(removable.begin(),removable.end(),[&](int a,int b){
  auto aa=keep.portfolio,bb=keep.portfolio;add(aa,keep.paths[a],-1);add(bb,keep.paths[b],-1);
  return std::tuple(keep.model.value(o,aa),-near(a),-a)>std::tuple(keep.model.value(o,bb),-near(b),-b);
 });
 std::stable_sort(empty.begin(),empty.end(),[](int a,int b){return std::tuple(near(a),a)<std::tuple(near(b),b);});
 std::set<std::tuple<int,int,int,int>>seen;
 for(auto[remove,nfeed]:std::array<std::pair<int,int>,3>{{{1,1},{1,2},{2,2}}}){
  if(int(removable.size())<remove||total_animals-remove<actual_animals)continue;
  std::vector<int>positions(removable.begin(),removable.begin()+std::min(remove,nfeed));
  for(int p:empty)if(int(positions.size())<nfeed)positions.push_back(p);
  if(int(positions.size())<nfeed)continue;
  // Enumerate feeder duration and a SINGLE successor cycle. No infinite
  // repeat credit; staggered arrival is next dawn, never same-day financing.
  struct Option{double value=-1e100;int age=-1,kind=-1;};Option best;
  for(int age=2;age<=4;age++)for(int k=0;k<5;k++){
   int next=o.day+age+1;if(next>=29)continue;
   auto portfolio=keep.portfolio;for(int i=0;i<remove;i++)add(portfolio,keep.paths[removable[i]],-1);
   bool okay=true;
   for(int pos:positions){
    auto first=keep.crop(W,o.day,pos,o.day+age,keep.prices);
    // For the hint only: feeder receipts cannot fund an earlier purchase.
    first.f[o.day+age+1][W]+=first.f[o.day+age][W];first.f[o.day+age][W]=0;
    add(portfolio,first);
    auto path=keep.choose_crop(k,next,pos,keep.prices,keep.rotations_dp(pos,keep.prices,next));
    if(path.kind<0){okay=false;break;}
    add(portfolio,path.a);
   }
   if(!okay)continue;
   double value=keep.model.value(o,portfolio);
   if(value>best.value+1e-6)best={value,age,k};
  }
  if(best.kind<0||!seen.insert({remove,nfeed,best.age,best.kind}).second)continue;
  Controller p=live;p.configure(keep.s);
  auto history=live.joint;p.joint=JointBundleState{};
  p.joint.source_started=history.source_started;p.joint.source_harvested=history.source_harvested;
  p.joint.successor_started=history.successor_started;p.joint.cancelled=history.cancelled;p.joint.completed=history.completed;
  p.joint.active=true;p.joint.count=nfeed;p.joint.created_day=o.day;p.joint.animal_limit=total_animals-remove;p.joint.variant=16+int(out.size());
  for(int i=0;i<nfeed;i++){auto&b=p.joint.slots[i];b.pos=positions[i];b.source_age=best.age;b.next_kind=best.kind;}
  if constexpr(P16_AFS_LOCAL_EXCHANGE>0){
   // Surgical exchange: keep all unrelated targets, maintenance and paid
   // commitments. Do not turn an animal cap into a whole-farm re-optimization.
   auto state=p.joint;p=keep;p.joint=state;
   auto settarget=[&](int pos,int k){for(auto&x:p.core.target)if(x.first==pos){x.second=k;return;}};
   for(int j=0;j<remove;j++){
    int pos=removable[j];add(p.portfolio,p.paths[pos],-1);p.paths[pos]={};
    settarget(pos,-1);p.book[pos]={};p.successor[pos]=-1;p.length[pos]=0;
   }
   int wheat_seed=o.priv.seeds[W];
   for(auto[pos,k]:p.core.target)if(k==W&&p.successor[pos]==W)wheat_seed=std::max(0,wheat_seed-1);
   for(int pos:positions){
    auto path=p.crop(W,o.day,pos,o.day+best.age,p.prices);bool paid=wheat_seed>0;
    if(paid){wheat_seed--;path.fixed[o.day]+=path.first_cost;path.first_cost=0;}
    p.paths[pos]=path;add(p.portfolio,path);settarget(pos,W);
    p.book[pos]={W,-1,o.day,best.age,-1,paid};p.successor[pos]=W;p.length[pos]=best.age;
    p.core.triad_crop_age[pos]=best.age;p.core.plant_not_before[pos]=0;
   }
   p.core.p.max_animals=p.joint.animal_limit;
   p.core.p.operating_reserve=p.s.reserve+p.joint.seed_reserve(o);
   auto farm=o.own;int expiry=dp7::Controller::project_zero_expiry(farm,o.step);
   View actual{o.step,o.day,o.hour,farm,o.opponent,o.priv,o.market,o.shops};
   p.model.value(o,p.portfolio,&p.prices);p.model.shadow=p.prices;
   p.core.prepare_orders(actual,o,expiry);p.predicted=p.model.value(o,p.portfolio);
  }else p.plan(o);
  bool all=true;
  for(int pos:positions){auto it=std::find_if(p.core.target.begin(),p.core.target.end(),[&](auto x){return x.first==pos;});all&=it!=p.core.target.end()&&it->second==W;}
  int now_animals=0;for(auto[pos,k]:p.core.target)now_animals+=k>=9;
  all&=now_animals<=p.joint.animal_limit;
  if(all)out.push_back({16+int(out.size()),std::move(p),remove,nfeed,best.age,best.kind,best.value});
 }
 return out;
}
}
