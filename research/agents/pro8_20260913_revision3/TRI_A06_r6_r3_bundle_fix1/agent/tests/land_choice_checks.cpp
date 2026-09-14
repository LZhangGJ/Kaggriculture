// Focused offline tests of the actual production classes. Not a submitted agent.
#include "bridge.cpp"
#include <set>

namespace {
void need(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
bool land_order(const triad::Proposal&p){return std::any_of(p.policy.core.queue.begin(),p.policy.core.queue.end(),[](auto a){return a.op==fastkag::Op::BUY_LAND;});}
std::vector<double> own_signature(const fastkag::Simulator&s){
 const auto&f=s.farms()[0];const auto&p=s.privates()[0];const auto&m=s.market();
 std::vector<double>z{f.money,double(f.farmer.x),double(f.farmer.y),double(f.hands.size())};
 for(auto xy:f.hands){z.push_back(xy.x);z.push_back(xy.y);}z.push_back(f.unlocked_mask);z.push_back(f.hires_today);
 for(const auto&t:f.tiles){for(double x:{double(t.kind),double(t.crop),double(t.animal),double(t.planted_day),double(t.placed_day),double(t.yield_units),double(t.consecutive_unwatered),double(t.consecutive_unfed),double(t.fertilized_until_day),double(t.pending_care_bonus),double(t.max_lifespan_step),double(t.watered_today),double(t.fed_today),double(t.cared_today),double(t.fertilizer_available)})z.push_back(x);}
 for(auto x:p.shed)z.push_back(x);for(auto x:p.seeds)z.push_back(x);z.push_back(p.inventories.size());
 for(size_t i=0;i<p.inventories.size();i++){for(auto x:p.inventories[i])z.push_back(x);z.push_back(p.inventory_order[i].size());for(auto x:p.inventory_order[i])z.push_back(x);}
 for(auto x:m.inventory)z.push_back(x);for(auto x:m.prices)z.push_back(x);return z;
}
bool same_book(const triad::Controller&a,const triad::Controller&b){
 for(int i=0;i<100;i++){auto x=a.book[i],y=b.book[i];
  if(std::tie(x.kind,x.birth,x.chosen_day,x.length,x.successor,x.funded)!=std::tie(y.kind,y.birth,y.chosen_day,y.length,y.successor,y.funded))return false;
 }return true;
}
}
extern "C" const char*td_r2_choice_checks(void*ptr,const double*input,size_t count,int exercise){
 auto&h=*static_cast<Handle*>(ptr);
 try{
  Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
  auto a=r.farm(),b=r.farm();auto priv=r.priv();fastkag::Market market;
  for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();
  std::vector<int8_t>shops;int n=r.count();for(int i=0;i<n;i++)shops.push_back(r.integer());
  need(r.i==count&&hour==0,"test clock or input");
  dp7::View v{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};
  const int owned=std::popcount(unsigned(v.own.unlocked_mask));
  const auto incoming_key=triad::proposal_key(h.policy.live);const auto incoming_book=h.policy.live.book;
  auto original=triad::generate_proposals(h.policy.live,h.policy.base,v);
  auto choices=h.policy.prepare(v);std::set<int>ids;int deferred=0,expanded=0,paid=0;
  need(triad::proposal_key(h.policy.live)==incoming_key,"prepare mutated incoming policy");
  for(int i=0;i<100;i++)need(h.policy.live.book[i].kind==incoming_book[i].kind&&h.policy.live.book[i].successor==incoming_book[i].successor,"prepare mutated incoming book");
  const bool paired=h.policy.base.a06_execution_candidates>0;
  for(const auto&p:original){
   for(int variant=0;variant<(paired?2:1);variant++){
    int id=p.id+(variant?16:0);auto it=std::find_if(choices.begin(),choices.end(),[&](const auto&q){return q.id==id;});
    need(it!=choices.end(),"original positive-investment candidate removed");
    need(triad::proposal_key(it->policy)==triad::proposal_key(p.policy),"original candidate plan changed");
    need(same_book(it->policy,p.policy),"original commitments changed");
   }
  }
  for(const auto&p:choices){
   need(p.id>=0&&p.id<64&&ids.insert(p.id).second,"candidate id alias or out-of-bounds");
   if(!(p.id&32)){expanded+=land_order(p);continue;}
   deferred++;
   need(p.policy.s.max_land==owned&&p.policy.core.p.max_land==owned&&p.policy.core.planned_land==owned,"defer cap not installed into real controller");
   need(!land_order(p),"defer branch still purchases land");
   for(auto[pos,k]:p.policy.core.target)need(k<0||v.own.tiles[pos].kind!=fastkag::TileKind::LOCKED,"defer targets locked land");
   const int source=p.id&15;auto it=std::find_if(original.begin(),original.end(),[&](auto&q){return q.id==source;});
   need(it!=original.end()&&land_order(*it),"defer source did not have land transaction");
   triad::Controller expected=h.policy.live;auto settings=it->policy.s;settings.max_land=owned;expected.configure(settings);expected.plan(v);
   need(triad::proposal_key(expected)==triad::proposal_key(p.policy),"defer did not independently replan from incoming state");
   need(same_book(expected,p.policy),"defer inherited expanded unfunded commitments");
   for(auto[pos,k]:p.policy.core.target)if(k>=0&&p.policy.book[pos].funded)paid++;
  }
  if(owned==4||day==29||std::any_of(original.begin(),original.end(),[](const auto&p){return !land_order(p);}))need(deferred==0,"unnecessary defer branches at full land/terminal day");
  std::ostringstream j;j.precision(17);j<<"{\"step\":"<<step<<",\"owned\":"<<owned<<",\"original\":"<<original.size()<<",\"choices\":"<<choices.size()<<",\"expanded\":"<<expanded<<",\"deferred\":"<<deferred<<",\"funded_commitments_checked\":"<<paid<<",\"branch_exercises\":[";
  bool comma=false;
  if(exercise){
   // Exercise one wait branch and one retained, funded expansion. This is a
   // bounded declared public-flow scenario, NOT an opponent or a match.
   for(bool want_wait:{true,false}){
    auto it=std::find_if(choices.begin(),choices.end(),[&](const auto&p){
     if(want_wait)return (p.id&32)!=0;
     return !(p.id&32)&&land_order(p)&&p.policy.core.project_preparation(v).farm.unlocked_mask!=v.own.unlocked_mask;
    });
    if(it==choices.end())continue;
    auto test=h.policy;test.install(*it,day);
    // Isolate restoration from tomorrow's *new* independent search. All real
    // Controller action, procurement, route and maintenance code still runs.
    test.base.scenario=0;
    fastkag::PublicFlowScenario world(v,it->policy.model.rival,h.policy.base.supply);
    int ticks=0,land_actions=0,new_projects=0;std::vector<fastkag::PlayerAction>acts;std::vector<double>prefix;
    for(int t=0;t<24&&!world.done();t++){
     auto vv=world.view();need(vv.day==day,"current-day scenario horizon");
     auto act=test.act(vv);if(t==0){fastkag::ObservedDayScenario observed(vv);auto after_units=observed.project_units(act.units,-1);prefix=own_signature(after_units.project_own_market(0,act.market));}need(act.market.size()<=10&&act.units.size()==vv.own.hands.size()+1,"actual entry shape");
     for(auto x:act.market)if(x.op==fastkag::Op::BUY_LAND)land_actions++;
     if(want_wait)need(test.live.s.max_land==owned&&test.live.core.p.max_land==owned,"day cap lost in real act");
     acts.push_back(act);world.advance(act);ticks++;
     if(want_wait)need(world.view().own.unlocked_mask==v.own.unlocked_mask,"wait execution unlocked land");
    }
    auto next=world.view();
    if(!want_wait)need(next.own.unlocked_mask!=v.own.unlocked_mask,"funded expansion did not execute");
    for(int pos=0;pos<100;pos++)if(v.own.tiles[pos].kind==fastkag::TileKind::LOCKED&&(dp7::plant(next.own.tiles[pos])||dp7::animal(next.own.tiles[pos])))new_projects++;
    if(!world.done()){
     need(next.day==day+1,"next-day clock");test.act(next);
     need(test.restore_day==-1&&test.live.s.max_land==test.base.max_land&&test.live.core.p.max_land==int(test.base.max_land),"land eligibility not restored tomorrow");
    }
    if(comma)j<<",";comma=true;
    j<<"{\"id\":"<<it->id<<",\"wait\":"<<(want_wait?"true":"false")<<",\"ticks\":"<<ticks<<",\"land_actions\":"<<land_actions<<",\"new_land_projects_at_dawn\":"<<new_projects<<",\"own_cash_conditional\":"<<world.own_cash()<<",\"actions\":[";
    for(size_t t=0;t<acts.size();t++){if(t)j<<",";j<<"{\"units\":[";for(size_t u=0;u<acts[t].units.size();u++){if(u)j<<",";auto x=acts[t].units[u];j<<"["<<int(x.op)<<","<<int(x.item)<<","<<x.quantity<<"]";}j<<"],\"market\":[";for(size_t u=0;u<acts[t].market.size();u++){if(u)j<<",";auto x=acts[t].market[u];j<<"["<<int(x.op)<<","<<int(x.item)<<","<<x.quantity<<"]";}j<<"]}";}
    j<<"],\"first_prefix_signature\":[";for(size_t i=0;i<prefix.size();i++){if(i)j<<",";j<<prefix[i];}j<<"]}";
   }
  }
  j<<"]}";h.text=j.str();return h.text.c_str();
 }catch(const std::exception&e){h.text=std::string("ERROR: ")+e.what();return h.text.c_str();}
}
