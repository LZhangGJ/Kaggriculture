#pragma once
// Included inside bridge, AFTER AuditOpponent is defined. This offline evaluator
// intentionally has full simulator snapshots. The public generator has none.
namespace investment_branch_audit {
constexpr uint64_t initial=1469598103934665603ULL;
inline void mix(uint64_t&h,uint64_t x){h^=x;h*=1099511628211ULL;}
inline uint64_t state_hash(const Simulator&s){
 uint64_t h=initial;mix(h,s.step_count());mix(h,s.seed());mix(h,s.done());
 for(auto&f:s.farms()){
  mix(h,std::bit_cast<uint64_t>(f.money));mix(h,f.unlocked_mask);mix(h,f.hires_today);
  mix(h,f.farmer.x);mix(h,f.farmer.y);mix(h,f.hands.size());for(auto p:f.hands){mix(h,p.x);mix(h,p.y);}
  for(auto&t:f.tiles){mix(h,int(t.kind));mix(h,int(t.crop));mix(h,int(t.animal));mix(h,t.planted_day);mix(h,t.placed_day);
   mix(h,t.yield_units);mix(h,t.consecutive_unwatered);mix(h,t.consecutive_unfed);mix(h,t.fertilized_until_day);
   mix(h,t.pending_care_bonus);mix(h,t.max_lifespan_step);mix(h,t.watered_today);mix(h,t.fed_today);mix(h,t.cared_today);mix(h,t.fertilizer_available);}
 }
 for(auto&p:s.privates()){
  for(int x:p.shed)mix(h,x);for(int x:p.seeds)mix(h,x);mix(h,p.inventories.size());
  for(auto&inv:p.inventories)for(int x:inv)mix(h,x);
  for(auto&order:p.inventory_order){mix(h,order.size());for(int x:order)mix(h,x);}
 }
 for(auto x:s.market().inventory)mix(h,x);for(auto x:s.market().prices)mix(h,std::bit_cast<uint64_t>(double(x)));
 mix(h,s.shops().size());for(auto x:s.shops())mix(h,x);
 for(auto&xs:s.last_market_fills()){mix(h,xs.size());for(int x:xs)mix(h,x);}
 for(int x:s.last_end_of_day_overflow())mix(h,x);return h;
}
inline uint64_t frame_hash(const std::array<PlayerAction,2>&actions,const Simulator&after){
 uint64_t h=state_hash(after);for(auto&p:actions){for(auto*xs:{&p.units,&p.market}){mix(h,xs->size());for(auto&a:*xs){mix(h,int(a.op));mix(h,int(a.item));mix(h,a.quantity);}}}return h;
}
struct Tail {double cash=0,rival_cash=0;uint64_t hash=initial;int steps=0;};
inline Tail suffix(const Simulator&source,const dp7::Controller&policy,const AuditOpponent&opponent,int seat){
 auto env=source;auto own=policy;auto rival=opponent;Tail result;
 while(!env.done()){
  std::array<PlayerAction,2>a;a[seat]=own.act(view(env,seat));a[1-seat]=rival.act(env,1-seat);
  env.step(a);mix(result.hash,frame_hash(a,env));result.steps++;
 }
 result.cash=env.farms()[seat].money;result.rival_cash=env.farms()[1-seat].money;return result;
}
inline bool same(const Tail&a,const Tail&b){return a.cash==b.cash&&a.rival_cash==b.rival_cash&&a.hash==b.hash&&a.steps==b.steps;}
struct Snapshot {
 Simulator env;dp7::Controller own;AuditOpponent rival;
 Snapshot(const Simulator&e,const dp7::Controller&c,const AuditOpponent&r):env(e),own(c),rival(r){}
};
struct Choice {
 std::string family;int kind=-1,amount=0;double score=0;int proposed=0,started=0;
 std::vector<std::pair<int,int>>edits;Tail result;
};
struct Node {int day=0;uint64_t state_hash=0;double source_cash=0,source_rival_cash=0;std::vector<Choice>choices;bool keep_before_after=false,reverse_equal=false;};
struct Match {uint64_t seed=0;int seat=0;double cash=0,rival_cash=0,seconds=0;std::vector<Node>nodes;int continuations=0;uint64_t transitions=0;};
inline Match run(uint64_t seed,int seat,dp7::Params params,AuditOpponentSpec spec,const std::vector<int>&days,bool reverse_check){
 auto tic=std::chrono::steady_clock::now();Simulator env(Config{},seed);dp7::Controller own(params);AuditOpponent rival(spec);
 std::vector<std::unique_ptr<Snapshot>>snapshots;std::vector<uint64_t>frames;Match result;result.seed=seed;result.seat=seat;
 while(!env.done()){
  if(env.hour()==0&&std::find(days.begin(),days.end(),env.day())!=days.end())snapshots.push_back(std::make_unique<Snapshot>(env,own,rival));
  std::array<PlayerAction,2>a;a[seat]=own.act(view(env,seat));a[1-seat]=rival.act(env,1-seat);env.step(a);frames.push_back(frame_hash(a,env));
 }
 result.cash=env.farms()[seat].money;result.rival_cash=env.farms()[1-seat].money;
 for(auto&ptr:snapshots){auto&s=*ptr;Node node;node.day=s.env.day();node.state_hash=state_hash(s.env);node.source_cash=s.env.farms()[seat].money;node.source_rival_cash=s.env.farms()[1-seat].money;
  auto candidates=dp7branch::generate(s.own,view(s.env,seat));if(candidates.empty()||candidates[0].family!="KEEP")throw std::runtime_error("missing KEEP branch");
  Tail expected{result.cash,result.rival_cash,initial,719-s.env.step_count()};for(size_t t=s.env.step_count();t<frames.size();t++)mix(expected.hash,frames[t]);
  for(auto&c:candidates){Choice choice;choice.family=c.family;choice.kind=c.kind;choice.amount=c.amount;choice.score=c.executable_score;
   choice.proposed=c.proposed;choice.started=c.conditionally_started;choice.edits=c.edits;choice.result=suffix(s.env,c.controller,s.rival,seat);
   result.continuations++;result.transitions+=choice.result.steps;node.choices.push_back(std::move(choice));
  }
  if(!same(node.choices[0].result,expected))throw std::runtime_error("KEEP did not replay full original suffix");
  auto again=suffix(s.env,candidates[0].controller,s.rival,seat);result.continuations++;result.transitions+=again.steps;
  if(!same(again,expected)||state_hash(s.env)!=node.state_hash)throw std::runtime_error("branch polluted original snapshot");node.keep_before_after=true;
  if(reverse_check){for(size_t i=candidates.size();i-->0;){auto tail=suffix(s.env,candidates[i].controller,s.rival,seat);result.continuations++;result.transitions+=tail.steps;if(!same(tail,node.choices[i].result))throw std::runtime_error("reverse-order branch changed result");}node.reverse_equal=true;}
  result.nodes.push_back(std::move(node));
 }
 result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-tic).count();return result;
}
}
