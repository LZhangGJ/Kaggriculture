// Actual production classes; no scenario is a game or an observed future.
#include "../land_choice_checks.cpp"
namespace {
std::string contract(const triad::Controller&c){Handle h(c.s);h.policy.live=c;return td_contract_json(&h);}
bool same_asset(const competitive::Asset&a,const competitive::Asset&b){return a.f==b.f&&a.fixed==b.fixed&&a.labor==b.labor&&a.end==b.end&&a.kind==b.kind&&a.first_cost==b.first_cost;}
void same_whole(const triad::Controller&a,const triad::Controller&b){
 need(triad::proposal_key(a)==triad::proposal_key(b),"whole plan key");need(contract(a)==contract(b),"whole execution and commitment state");
 need(a.predicted==b.predicted&&a.prices==b.prices&&same_asset(a.portfolio,b.portfolio),"whole portfolio valuation");
 for(int i=0;i<100;i++)need(same_asset(a.paths[i],b.paths[i]),"project path copy");
 need(a.forecast_service==b.forecast_service&&a.core.daily_need==b.core.daily_need&&a.core.prepared_seed_need==b.core.prepared_seed_need&&a.core.labor_target==b.core.labor_target,"whole resource service labor state");
 need(a.release==b.release&&a.successor==b.successor&&a.length==b.length,"release successor length copy");
 need(a.core.crop_water==b.core.crop_water&&a.core.crop_fertilize==b.core.crop_fertilize&&a.core.crop_birth==b.core.crop_birth&&a.core.crop_kind==b.core.crop_kind&&a.core.service_feed==b.core.service_feed&&a.core.service_care==b.core.service_care,"service instructions copy");
#if R2_FINITE_FERTILIZER
 need(a.finite_first_fertilizer==b.finite_first_fertilizer,"finite fertilizer copy");
#endif

}
void action_json(std::ostream&j,const fastkag::PlayerAction&a){
 j<<"{\"units\":[";for(size_t i=0;i<a.units.size();i++){if(i)j<<",";auto x=a.units[i];j<<"["<<int(x.op)<<","<<int(x.item)<<","<<x.quantity<<"]";}j<<"],\"market\":[";for(size_t i=0;i<a.market.size();i++){if(i)j<<",";auto x=a.market[i];j<<"["<<int(x.op)<<","<<int(x.item)<<","<<x.quantity<<"]";}j<<"]}";
}
template<class F>std::string with_view(void*ptr,const double*in,size_t n,F f){auto&h=*static_cast<Handle*>(ptr);try{
 Cursor r{in,n};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();auto a=r.farm(),b=r.farm();auto pr=r.priv();fastkag::Market m;for(auto&x:m.inventory)x=r.integer();for(auto&x:m.prices)x=r.integer();std::vector<int8_t>sh;int ns=r.count();for(int i=0;i<ns;i++)sh.push_back(r.integer());need(r.i==n,"input complete");
 dp7::View v{step,day,hour,seat==0?a:b,seat==0?b:a,pr,m,sh};return f(h,v);
 }catch(const std::exception&e){return std::string("ERROR: ")+e.what();}}
int risk(const fastkag::Farm&f){int n=0;for(auto&t:f.tiles)n+=dp7::plant(t)&&!t.watered_today&&t.consecutive_unwatered>=1;return n;}
}
extern "C" const char*td_r3_state(void*ptr,const double*in,size_t n){auto&h=*static_cast<Handle*>(ptr);h.text=with_view(ptr,in,n,[](Handle&h,const dp7::View&v){
 triad::Controller before=h.policy.live;auto key=contract(before);int owned=std::popcount(unsigned(v.own.unlocked_mask));
 auto expand=before;expand.plan_impl(v,int(expand.s.max_land));auto wait=before;wait.plan_impl(v,owned);auto guarded=before;guarded.plan(v);
 bool compare=owned<int(before.s.max_land)&&v.day<=20&&expand.core.planned_land>owned;
 bool accepted=!compare||(std::isfinite(expand.predicted)&&expand.predicted>wait.predicted+1e-6);
 same_whole(guarded,accepted?expand:wait);need(contract(before)==key,"incoming state immutable");
 // Compare procurement on the exact SAME chosen targets/commitments. Reordering
 // must preserve every order and its quantity, including today feed and animals.
 auto legacy=guarded.core,funded=guarded.core;legacy.p.funded_labor=false;funded.p.funded_labor=true;
 legacy.prepare_orders(v,v,0);funded.prepare_orders(v,v,0);
 auto orderkey=[](const auto&q){std::vector<std::tuple<int,int,int>>k;for(auto a:q)k.emplace_back(int(a.op),int(a.item),a.quantity);std::sort(k.begin(),k.end());return k;};
 need(orderkey(legacy.queue)==orderkey(funded.queue),"funding changed order multiset");need(legacy.labor_target==funded.labor_target,"selected labor target changed");
 bool saw_hire=false,saw_capital=false;int hires=0;for(auto a:funded.queue){if(a.op==fastkag::Op::HIRE){need(!saw_capital,"capital preempts wage");saw_hire=true;hires++;}else if(a.op==fastkag::Op::BUY_LAND||a.op==fastkag::Op::BUY_ANIMAL||a.op==fastkag::Op::BUY_SEED)saw_capital=true;}
 bool changed=false;for(size_t i=0;i<legacy.queue.size();i++)if(int(legacy.queue[i].op)!=int(funded.queue[i].op)||legacy.queue[i].item!=funded.queue[i].item||legacy.queue[i].quantity!=funded.queue[i].quantity)changed=true;
 std::ostringstream j;j.precision(17);j<<"{\"step\":"<<v.step<<",\"compare\":"<<compare<<",\"retained_expansion\":"<<(compare&&accepted)<<",\"deferred_expansion\":"<<(compare&&!accepted)<<",\"expand_score\":"<<expand.predicted<<",\"wait_score\":"<<wait.predicted<<",\"funding_reordered\":"<<changed<<",\"queue_size\":"<<funded.queue.size()<<",\"hire_orders\":"<<hires<<",\"desired_hands\":"<<funded.labor_target<<",\"new_land\":"<<guarded.core.planned_land-owned<<",\"pass\":true}";return j.str();});return h.text.c_str();}

// mode=0: genuine production SearchController from a cold own-visible state.
// mode=1/2: isolated maintenance fixture with actual current crew, no new projects;
//          bounded staffing recovery enabled/disabled respectively.
extern "C" const char*td_r3_branch(void*ptr,const double*in,size_t n,int mode,int desired){auto&h=*static_cast<Handle*>(ptr);h.text=with_view(ptr,in,n,[=](Handle&h,const dp7::View&v){
 auto selector=h.policy;dp7::Controller fixture=selector.live.core;
 if(mode==1||mode==2){fixture.day=v.day;fixture.phase=2;fixture.queue.clear();fixture.target.clear();fixture.plans.clear();fixture.pending_admission={};fixture.labor_target=desired;fixture.crop_service_day=-1;fixture.animal_service_day=-1;fixture.p.funded_labor=mode==1;fixture.p.intraday_admission=false;fixture.p.procure_service_inputs=false;fixture.p.recover_service_inputs=false;fixture.p.triad_delivery_pressure=0;
 for(int pos=0;pos<100;pos++){auto&t=v.own.tiles[pos];if(dp7::plant(t))fixture.target.emplace_back(pos,int(t.crop));if(dp7::animal(t))fixture.target.emplace_back(pos,int(t.animal));}}
 if(mode==3||mode==4){need(fixture.day==v.day&&fixture.phase==3&&fixture.plans.size()==v.priv.inventories.size(),"imported fixture roster/clock");fixture.p.funded_labor=mode==3;fixture.labor_target=desired;fixture.p.intraday_admission=false;fixture.p.intraday_procurement=false;fixture.seed_reconcile_checked=true;}
 fastkag::ObservedDayScenario world(v);int start_risk=risk(v.own),watered=0,hired=0,land=0,fed=0;std::set<int>new_projects;std::ostringstream j;j.precision(17);
 j<<"{\"mode\":"<<mode<<",\"step\":"<<v.step<<",\"initial_risk\":"<<start_risk<<",\"ticks\":[";bool comma=false;
 // Excludes midnight and any unobserved random future. Only today's known shops.
 while(!world.finished()&&world.view().hour<23){auto o=world.view();fastkag::PlayerAction a;if(mode){a=fixture.act(o);if(mode==1||mode==3)dp7::labor::procure(fixture,o,a);}else a=selector.act(o);
 need(a.units.size()==o.priv.inventories.size()&&a.market.size()<=10,"real execution output shape");
 auto prefix=world.project_units(a.units,-1).project_own_market(0,a.market);auto sig=own_signature(prefix);
 if(comma)j<<",";comma=true;j<<"{\"step\":"<<o.step<<",\"cash\":"<<o.own.money<<",\"hands\":"<<o.own.hands.size()<<",\"risk\":"<<risk(o.own)<<",\"action\":";action_json(j,a);j<<",\"prefix_signature\":[";for(size_t i=0;i<sig.size();i++){if(i)j<<",";j<<sig[i];}j<<"]";
 int nh=prefix.farms()[0].hands.size()-o.own.hands.size();hired+=nh;
 need(nh>=0,"unexpected loss of worker within day");for(auto x:a.market)land+=x.op==fastkag::Op::BUY_LAND;
 for(int pos=0;pos<100;pos++)watered+=dp7::plant(o.own.tiles[pos])&&!o.own.tiles[pos].watered_today&&prefix.farms()[0].tiles[pos].watered_today;
 for(int pos=0;pos<100;pos++){const auto&t=prefix.farms()[0].tiles[pos];if(v.own.tiles[pos].kind==fastkag::TileKind::LOCKED&&(dp7::plant(t)||dp7::animal(t)))new_projects.insert(pos);fed+=dp7::animal(o.own.tiles[pos])&&!o.own.tiles[pos].fed_today&&t.fed_today;}
 world.advance(a);j<<",\"cash_after\":"<<world.own().money<<",\"hands_after\":"<<world.own().hands.size()<<",\"risk_after\":"<<risk(world.own())<<",\"fills\":[";for(size_t i=0;i<world.fills().size();i++){if(i)j<<",";j<<world.fills()[i];}j<<"]}";
 }
 j<<"],\"final_risk\":"<<risk(world.own())<<",\"watered\":"<<watered<<",\"hired\":"<<hired<<",\"land_orders\":"<<land<<",\"land_purchases_filled\":"<<(std::popcount(unsigned(world.own().unlocked_mask))-std::popcount(unsigned(v.own.unlocked_mask)))<<",\"new_land_projects_started\":"<<new_projects.size()<<",\"existing_animal_feeds_completed\":"<<fed<<",\"labor_offers\":"<<(mode?fixture.labor_offers:selector.live.core.labor_offers)<<",\"labor_arrivals\":"<<(mode?fixture.labor_arrivals:selector.live.core.labor_arrivals)<<",\"pass\":true}";return j.str();});return h.text.c_str();}

extern "C" const char*td_r3_synthetic(void*ptr,const double*in,size_t n){auto&h=*static_cast<Handle*>(ptr);h.text=with_view(ptr,in,n,[](Handle&h,const dp7::View&original){
 // Shape-valid explicit synthetic variants of this public/own snapshot.
 auto f=original.own;for(auto&t:f.tiles)t=fastkag::Tile{};f.unlocked_mask=15;f.hands.clear();f.hires_today=0;f.farmer={4,4};f.money=100;
 auto pr=original.priv;pr.shed={};pr.seeds={};pr.inventories.assign(1,{});pr.inventory_order.assign(1,{});
 for(int pos:{34,35,24,25}){auto&t=f.tiles[pos];t.kind=fastkag::TileKind::PLANT;t.crop=fastkag::Item::STRAWBERRY;t.planted_day=7;t.consecutive_unwatered=1;t.watered_today=false;t.yield_units=0;}
 auto m=original.market;for(int i=0;i<9;i++){m.inventory[i]=10000;m.prices[i]=dp7::price(i,10000);}
 std::vector<std::string>passed;std::ostringstream details;bool dc=false;
 for(int variant=0;variant<21;variant++){
 auto ff=f;auto pp=pr;auto mm=m;int hour=3;dp7::Controller c=h.policy.live.core;c.day=9;c.phase=3;c.plans.assign(1,{});c.pending_admission={};c.target.clear();c.crop_service_day=-1;c.animal_service_day=-1;c.labor_target=2;c.p.funded_labor=true;
 for(int p:{34,35,24,25})c.target.emplace_back(p,dp7::S);
 fastkag::PlayerAction act;act.units.assign(1,dp7::action(fastkag::Op::PASS));
 if(variant==1)ff.money=0;
 if(variant==2){ff.money=0;pp.shed[dp7::MI]=1;act.market.push_back(dp7::action(fastkag::Op::SELL,dp7::MI,1));}
 if(variant==3)for(int pos:{34,35,24,25})ff.tiles[pos].watered_today=true;
 if(variant==4)c.labor_target=0;
 if(variant==5)hour=23;
 if(variant==6)for(int k=0;k<10;k++)act.market.push_back(dp7::action(fastkag::Op::PASS));
 if(variant==7){for(int pos:{34,35,24,25}){c.plans[0].a.push_back(dp7::action(fastkag::Op::WATER));c.plans[0].target.push_back(pos);}}
 if(variant==8){ff.money=0;pp.inventories[0][dp7::MI]=1;pp.inventory_order[0].push_back(dp7::MI);} // carried goods are not sale receipts
 if(variant==9){ff.money=1;ff.hires_today=10;}
 if(variant>=10&&variant<=15){
  for(auto&t:ff.tiles)t=fastkag::Tile{};auto&t=ff.tiles[34];t.kind=fastkag::TileKind::ANIMAL;t.animal=fastkag::Item::COW;t.placed_day=0;t.consecutive_unfed=1;t.fed_today=false;t.cared_today=true;
  c.target={{34,dp7::CO}};c.animal_service_day=9;c.service_feed.fill(1);c.service_care.fill(0);
  if(variant>=11&&variant<=13)pp.shed[dp7::W]=1;
  if(variant==12){c.plans[0].a={dp7::action(fastkag::Op::PICKUP,dp7::W,1)};c.plans[0].target={44};}
  if(variant==13){c.pending_admission={true,9,218,0,33,dp7::CO};}
  if(variant==14)act.market.push_back(dp7::action(fastkag::Op::BUY_PRODUCT,dp7::W,1));
  if(variant==15){ff.money=0;pp.shed[dp7::MI]=1;act.market.push_back(dp7::action(fastkag::Op::SELL,dp7::MI,1));act.market.push_back(dp7::action(fastkag::Op::BUY_PRODUCT,dp7::W,1));}
 }
 if(variant==16)for(int k=0;k<9;k++)act.market.push_back(dp7::action(fastkag::Op::PASS));
 if(variant==17)c.p.max_hands=0;
 if(variant==18)hour=22;
 if(variant==19){ff.farmer={4,3};for(int pos:{35,24,25})ff.tiles[pos].watered_today=true;act.units[0]=dp7::action(fastkag::Op::WATER);}
 if(variant==20)act.market.push_back(dp7::action(fastkag::Op::HIRE));
 dp7::View v{216+hour,9,hour,ff,original.opponent,pp,mm,original.shops};
 auto oldplans=c.plans;dp7::labor::procure(c,v,act);int count=std::count_if(act.market.begin(),act.market.end(),[](auto a){return a.op==fastkag::Op::HIRE;});
 bool positive=variant==0||variant==2||variant==11||variant==14||variant==15||variant==16;need(((count-(variant==20))>0)==positive,"recovery positive/negative admission");need(c.plans.size()==oldplans.size(),"hypothetical worker installed before observation");need(act.market.size()<=10,"order page overflow");
 if(positive){fastkag::ObservedDayScenario w(v);auto after=w.project_units(act.units,-1).project_own_market(0,act.market);need(after.farms()[0].hands.size()==size_t(count),"offered crew not paid by actual own prefix");
 dp7::View next{v.step+1,9,hour+1,after.farms()[0],original.opponent,after.privates()[0],after.market(),original.shops};dp7::labor::observe(c,next);need(c.plans.size()==1+count&&c.labor_arrivals==count&&c.labor_jobs>0,"observed crew has no service assignment");need(c.plans[0].a.size()==oldplans[0].a.size()&&c.plans[0].index==oldplans[0].index,"incumbent plan changed");std::set<int>plots;for(size_t u=1;u<c.plans.size();u++){need(c.plans[u].a.size()<=size_t(24-next.hour),"service route beyond time window");for(size_t k=0;k<c.plans[u].a.size();k++)if(c.plans[u].a[k].op==fastkag::Op::WATER)need(plots.insert(c.plans[u].target[k]).second,"double assigned service");}}
 if(dc)details<<",";dc=true;details<<"{\"variant\":"<<variant<<",\"hire_orders\":"<<count<<",\"pass\":true}";
 }
 int floor_checks=0;
 for(int i=0;i<9;i++)for(int offset:{-2,-1,0,1})for(int q:{1,2,3,10,100}){
  double inv=competitive::ConditionalMarket::saturation(i)+offset;
  if(inv+q<=competitive::ConditionalMarket::saturation(i))continue;
  double at=inv,expected=0;for(int k=0;k<q;k++){int price=dp7::price(i,at);expected+=price;if(price>1)at++;}
  double cash=competitive::Planner::trade(i,inv,q);need(cash==expected&&inv==at,"r1 floor cash/inventory regression");floor_checks++;
 }
 double wool_inventory=10056;double wool_cash=competitive::Planner::trade(dp7::WO,wool_inventory,10);need(wool_cash==41&&wool_inventory==10059,"reported wool floor regression");
 return std::string("{\"cases\":[")+details.str()+"],\"floor_crossing_checks\":"+std::to_string(floor_checks)+",\"reported_wool_cash\":41,\"reported_wool_inventory\":10059,\"pass\":true}";
 });return h.text.c_str();}

// Explicit test-only serialized parent service fields, not a production input.
extern "C" int td_r3_loadfixture(void*ptr,const double*in,size_t n){auto&h=*static_cast<Handle*>(ptr);try{
 size_t at=0;auto get=[&](){need(at<n,"short parent fixture");return int(in[at++]);};
 triad::Settings s;double*ss=reinterpret_cast<double*>(&s);for(int i=0;i<triad::SETTINGS_COUNT;i++){need(at<n,"short settings");ss[i]=in[at++];}h.policy.live.configure(s);auto&c=h.policy.live.core;
 c.day=get();c.phase=get();c.last_step=get();c.planned_land=get();c.feed_stock_target=get();c.crop_service_day=get();c.animal_service_day=get();
 auto arr=[&](auto&out){for(auto&x:out)x=get();};arr(c.daily_need);arr(c.prepared_seed_need);arr(c.crop_birth);arr(c.crop_kind);arr(c.crop_water);arr(c.crop_fertilize);arr(c.triad_crop_age);arr(c.plant_not_before);arr(c.service_feed);arr(c.service_care);
 c.target.clear();int nt=get();for(int i=0;i<nt;i++){int pos=get(),kind=get();c.target.emplace_back(pos,kind);}c.plans.clear();int np=get();for(int i=0;i<np;i++){dp7::Plan p;int na=get();for(int k=0;k<na;k++){int op=get(),item=get(),q=get(),pos=get();p.a.push_back(dp7::action(fastkag::Op(op),item,q));p.target.push_back(pos);}c.plans.push_back(std::move(p));}need(at==n,"trailing parent fixture");c.queue.clear();c.pending_admission={};return 0;
 }catch(const std::exception&e){h.text=std::string("ERROR: ")+e.what();return -1;}}
