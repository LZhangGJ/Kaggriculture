#include "search.hpp"
#include "observation_codec.hpp"
#include <cstring>
struct Handle {triad::SearchController policy;std::vector<triad::Proposal>prepared;std::vector<double>prepared_scores;std::vector<std::array<double,5>>prepared_horizons;std::vector<triad::SearchController::ScoreAudit>prepared_audits;std::vector<std::string>prepared_names;std::vector<bool>prepared_key_unique;int prepared_day=-1;std::string text;explicit Handle(triad::Settings s):policy(s){}};
extern "C" void*td_new(const double*x,size_t n){try{triad::Settings s;if(n){if(n!=triad::SETTINGS_COUNT&&n!=triad::SETTINGS_COUNT-1)return nullptr;for(size_t i=0;i<n;i++)if(!std::isfinite(x[i]))return nullptr;std::memcpy(&s,x,n*sizeof(double));}if((s.marginal_value!=0&&s.marginal_value!=1)||s.max_hands<0||s.max_hands>15||s.max_land<1||s.max_land>4||s.max_animals<0||s.max_animals>75||s.labor_hours<1||s.labor_hours>24||s.portfolio_swaps<0||s.portfolio_swaps>2||s.portfolio_swaps!=std::floor(s.portfolio_swaps)||s.portfolio_swap_min_gain<0)return nullptr;return new Handle(s);}catch(...){return nullptr;}}
extern "C" void td_delete(void*p){delete static_cast<Handle*>(p);}
extern "C" const char*td_debug(void*p){auto&h=*static_cast<Handle*>(p);if(h.text.rfind("ERROR",0)!=0)h.text=h.policy.debug();return h.text.c_str();}
extern "C" int td_act(void*p,const dp7::View*v,fastkag::PlayerAction*out){auto&h=*static_cast<Handle*>(p);try{*out=h.policy.act(*v);return 0;}catch(const std::exception&e){h.text=std::string("ERROR: ")+e.what();return -1;}}
extern "C" int td_observe(void*p,const double*input,size_t count,int32_t*out,size_t cap){try{
 Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
 if(step<0||step>=719||day!=step/24||hour!=step%24||seat<0||seat>1)throw std::runtime_error("clock/seat");
 auto a=r.farm(),b=r.farm();auto priv=r.priv();fastkag::Market market;for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();std::vector<int8_t>shops;int n=r.count();if(n>8)throw std::runtime_error("shops");for(int i=0;i<n;i++)shops.push_back(r.integer());if(r.i!=count)throw std::runtime_error("trailing input");
 if(priv.inventories.size()!=(seat==0?a:b).hands.size()+1)throw std::runtime_error("inventory size");
 dp7::View v{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};fastkag::PlayerAction action;if(td_act(p,&v,&action))return -1;
 size_t size=2+3*(action.units.size()+action.market.size());if(size>cap)throw std::runtime_error("output capacity");
 out[0]=action.units.size();out[1]=action.market.size();int at=2;auto put=[&](const fastkag::Action&a){out[at++]=int(a.op);out[at++]=int(a.item);out[at++]=a.quantity;};for(auto&a:action.units)put(a);for(auto&a:action.market)put(a);return size;
 }catch(const std::exception&e){static_cast<Handle*>(p)->text=std::string("ERROR: ")+e.what();return -1;}}
extern "C" int td_observe_external(void*p,const double*input,size_t count,const int32_t*actions,size_t action_count){try{
 Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
 auto a=r.farm(),b=r.farm();auto priv=r.priv();fastkag::Market market;for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();std::vector<int8_t>shops;int n=r.count();for(int i=0;i<n;i++)shops.push_back(r.integer());
 if(r.i!=count||seat<0||seat>1||action_count<2)throw std::runtime_error("external observation");
 dp7::View v{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};fastkag::PlayerAction out;
 size_t at=2,units=actions[0],orders=actions[1];if(action_count!=2+3*(units+orders))throw std::runtime_error("external action");
 auto take=[&](){auto op=actions[at++],item=actions[at++],quantity=actions[at++];if(op<0||op>=24||item<-1||item>=12)throw std::runtime_error("external atom");return fastkag::Action{fastkag::Op(op),fastkag::Item(item),quantity};};
 for(size_t i=0;i<units;i++)out.units.push_back(take());for(size_t i=0;i<orders;i++)out.market.push_back(take());
 auto&c=static_cast<Handle*>(p)->policy;c.live.joint.observe(v);
#if R2_CROP_CLOCK_MODE >= 1
 c.crop_clock.observe(v);c.live.model.rival_harvest_weights=c.crop_clock.weights();
#endif
 if constexpr(triad::PublicTradeLedger::enabled){c.ledger.observe(v);c.sale_clock.observe(c.ledger);c.ledger.record(v,out);}
 c.live.sale_memory=c.sale_clock;c.live.joint.record(v,out);return 0;
 }catch(const std::exception&e){static_cast<Handle*>(p)->text=std::string("ERROR: ")+e.what();return -1;}}
extern "C" int td_activate_external(void*p,const double*input,size_t count){try{
 Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
 auto a=r.farm(),b=r.farm();auto priv=r.priv();fastkag::Market market;for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();std::vector<int8_t>shops;int n=r.count();for(int i=0;i<n;i++)shops.push_back(r.integer());
 if(r.i!=count||seat<0||seat>1)throw std::runtime_error("external activation");
 auto&c=static_cast<Handle*>(p)->policy;c.live.previous_step=step-1;c.live.sale_memory=c.sale_clock;return 0;
 }catch(const std::exception&e){static_cast<Handle*>(p)->text=std::string("ERROR: ")+e.what();return -1;}}

// Offline experiment control. No Simulator, seed, opponent ID or hidden private
// state is accepted even by the feature/candidate API.
extern "C" void*td_clone(void*p){try{return new Handle(*static_cast<Handle*>(p));}catch(...){return nullptr;}}
extern "C" int td_prepare(void*p,const dp7::View*v){try{auto&h=*static_cast<Handle*>(p);h.policy.begin_observation(*v);h.prepared=h.policy.prepare(*v,true);h.prepared_scores.clear();h.prepared_horizons.clear();h.prepared_audits.clear();for(const auto&x:h.prepared){triad::SearchController::ScoreAudit audit;h.prepared_scores.push_back(h.policy.score(*v,x,0,&audit));h.prepared_audits.push_back(std::move(audit));std::array<double,5>a;for(int d=1;d<=5;d++)a[d-1]=h.policy.score(*v,x,d);h.prepared_horizons.push_back(a);}h.prepared_names.assign(h.prepared.size(),"");h.prepared_key_unique.assign(h.prepared.size(),true);h.prepared_day=v->day;return h.prepared.size();}catch(const std::exception&e){static_cast<Handle*>(p)->text=std::string("ERROR: ")+e.what();return -1;}}
extern "C" int td_prepare_observation(void*p,const double*input,size_t count){try{
 Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
 auto a=r.farm(),b=r.farm();auto priv=r.priv();fastkag::Market market;for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();std::vector<int8_t>shops;int n=r.count();for(int i=0;i<n;i++)shops.push_back(r.integer());
 if(r.i!=count||day!=step/24||hour!=step%24||seat<0||seat>1)throw std::runtime_error("prepare observation");
 auto&h=*static_cast<Handle*>(p);dp7::View v{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};if(h.policy.begun_step!=step&&td_activate_external(p,input,count))return -1;int count=td_prepare(p,&v);if(count<0)return count;
 auto add=[&](triad::Settings s,int id,const char*name){triad::Controller policy=h.policy.live;policy;policy.configure(s);policy.plan(v);auto key=triad::proposal_key(policy);bool unique=true;for(const auto&x:h.prepared)unique&=triad::proposal_key(x.policy)!=key;auto features=triad::portfolio_features(v,policy,h.prepared[0].policy,id,true);h.prepared.push_back({id,std::move(policy),std::move(features)});const auto&proposal=h.prepared.back();triad::SearchController::ScoreAudit audit;h.prepared_scores.push_back(h.policy.score(v,proposal,0,&audit));h.prepared_audits.push_back(std::move(audit));std::array<double,5>scores;for(int d=1;d<=5;d++)scores[d-1]=h.policy.score(v,proposal,d);h.prepared_horizons.push_back(scores);h.prepared_names.push_back(name);h.prepared_key_unique.push_back(unique);};
 auto winner=std::max_element(h.prepared_scores.begin(),h.prepared_scores.end())-h.prepared_scores.begin();
#if R2_OUTER_NEIGHBOR_AUDIT
 auto winner_settings=h.prepared[winner].policy.s;auto slots=h.prepared[winner].policy.greedy_audit;
 std::set<std::vector<int>>outer_seen;for(const auto&x:h.prepared)outer_seen.insert(triad::proposal_key(x.policy));
 struct OuterBranch{int pos,kind;double score;};std::vector<OuterBranch>branches;
 auto append_outer=[&](triad::Controller policy,int id,std::string name){auto features=triad::portfolio_features(v,policy,h.prepared[0].policy,id,true);triad::Proposal proposal{id,std::move(policy),std::move(features)};
  triad::SearchController::ScoreAudit audit;double score=h.policy.score(v,proposal,0,&audit);
  h.prepared.push_back(std::move(proposal));h.prepared_scores.push_back(score);h.prepared_horizons.push_back({score,score,score,score,score});
  h.prepared_audits.push_back(std::move(audit));h.prepared_names.push_back(std::move(name));h.prepared_key_unique.push_back(true);return score;};
 for(const auto&slot:slots)for(int k:{0,1,2,3,4,9,10,11}){
  if(k==slot.picked_kind)continue;
  triad::Controller policy=h.policy.live;policy.configure(winner_settings);policy.forced_kind.fill(-2);policy.forced_kind[slot.pos]=k;policy.plan(v);
  auto target=std::find_if(policy.core.target.begin(),policy.core.target.end(),[&](auto x){return x.first==slot.pos;});
  if(target==policy.core.target.end()||target->second!=k)continue;
  auto key=triad::proposal_key(policy);if(!outer_seen.insert(key).second)continue;
  double score=append_outer(std::move(policy),1000+slot.pos*16+k,"outer_p"+std::to_string(slot.pos)+"_k"+std::to_string(k));
  branches.push_back({slot.pos,k,score});
 }
#if R2_OUTER_BEAM_WIDTH > 0
 std::stable_sort(branches.begin(),branches.end(),[](const auto&a,const auto&b){return a.score>b.score;});
 if(branches.size()>R2_OUTER_BEAM_WIDTH)branches.resize(R2_OUTER_BEAM_WIDTH);
 for(const auto&branch:branches)for(const auto&slot:slots){
  if(slot.pos==branch.pos)continue;
  for(int k:{0,1,2,3,4,9,10,11}){
   if(k==slot.picked_kind)continue;
   triad::Controller policy=h.policy.live;policy.configure(winner_settings);policy.forced_kind.fill(-2);
   policy.forced_kind[branch.pos]=branch.kind;policy.forced_kind[slot.pos]=k;policy.plan(v);
   auto has=[&](int pos,int kind){auto target=std::find_if(policy.core.target.begin(),policy.core.target.end(),[&](auto x){return x.first==pos;});return target!=policy.core.target.end()&&target->second==kind;};
   if(!has(branch.pos,branch.kind)||!has(slot.pos,k))continue;
   auto key=triad::proposal_key(policy);if(!outer_seen.insert(key).second)continue;
   int id=100000+(((branch.pos*12+branch.kind)*100+slot.pos)*12+k);
   append_outer(std::move(policy),id,"outer2_p"+std::to_string(branch.pos)+"_k"+std::to_string(branch.kind)+"_p"+std::to_string(slot.pos)+"_k"+std::to_string(k));
  }
 }
#endif
#endif
 auto succession=h.prepared[winner].policy.s;succession.rotation=succession.repeat=1;add(succession,100,"crop_succession");auto sale=h.prepared[winner].policy.s;sale.delay_sale=1;add(sale,101,"competitive_sale");return h.prepared.size();
 }catch(const std::exception&e){static_cast<Handle*>(p)->text=std::string("ERROR: ")+e.what();return -1;}}
extern "C" double td_candidate_score_horizon_observation(void*p,int index,const double*input,size_t count,int horizon){try{
 auto&h=*static_cast<Handle*>(p);if(index<0||index>=int(h.prepared.size())||horizon<1||horizon>30)return NAN;
 Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
 auto a=r.farm(),b=r.farm();auto priv=r.priv();fastkag::Market market;
 for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();
 std::vector<int8_t>shops;int n=r.count();for(int i=0;i<n;i++)shops.push_back(r.integer());
 if(r.i!=count||day!=step/24||hour!=step%24||seat<0||seat>1)return NAN;
 dp7::View v{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};
 return h.policy.score(v,h.prepared[index],horizon);
 }catch(...){return NAN;}}

#if R2_SHOP_BRANCH_AUDIT
// Diagnostic Bellman boundary table. The public-flow scenario and online
// SearchController are copied; neither the prepared handle nor production ABI
// changes. A short horizon prevents accidental full-game use.
extern "C" const char*td_candidate_online_boundaries_json(void*p,int index,const double*input,size_t count,int days,int shift_item,int shift_units){try{
 auto&h=*static_cast<Handle*>(p);if(index<0||index>=int(h.prepared.size())||days<2||days>6||
  shift_item<-1||shift_item>=9||shift_units<0||(shift_item<0&&shift_units))return nullptr;
 Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
 auto a=r.farm(),b=r.farm();auto priv=r.priv();fastkag::Market market;
 for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();
 std::vector<int8_t>shops;int n=r.count();for(int i=0;i<n;i++)shops.push_back(r.integer());
 if(r.i!=count||day!=step/24||hour!=step%24||seat<0||seat>1||hour!=0)return nullptr;
 dp7::View start{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};
 const auto&proposal=h.prepared[index];for(int d=0;d<30;d++)for(int i=0;i<9;i++)
  if(proposal.policy.model.rival[d][i]!=h.prepared[0].policy.model.rival[d][i])
   throw std::runtime_error("online boundaries require common rival flow");
 auto online=h.policy;online.install(proposal,day);
 fastkag::PublicFlowScenario world(start,proposal.policy.model.rival,online.base.supply,
#if R2_SALE_CLOCK_MODE >= 1
  &online.sale_clock
#else
  nullptr
#endif
 );
 triad::Settings common=online.base;common.scenario=0;
 double beta=1+online.base.discount*std::max(0.,1-start.own.money/20000.);
 double realized=start.own.money-online.base.competition*start.opponent.money;
 double realized_margin=start.own.money-start.opponent.money;
 std::array<int,9> sold{},bought{};
 competitive::Flow prior_rival{};bool have_prior_rival=false;
 std::array<int,9> prior_market_prediction{};bool have_prior_market=false;
 competitive::Asset prior_portfolio{};bool have_prior_portfolio=false;
 std::ostringstream s;s.precision(17);s<<"{\"id\":"<<proposal.id<<",\"start_day\":"<<day
  <<",\"competition\":"<<online.base.competition<<",\"beta\":"<<beta
  <<",\"formal_h1\":"<<h.prepared_scores[index]<<",\"boundaries\":[";
 bool first=true;
 while(!world.done()&&world.view().day<=day+days){
  auto v=world.view();
  if(v.hour==0){
   online.begin_observation(v);
   if(v.day>day){
    triad::Controller tail=online.live;tail.configure(common);
    tail.model.use_value_basis=true;tail.model.value_basis={day,start.own.money};tail.plan(v);
    competitive::ValueBreakdown parts;
    competitive::ValueBasis basis{day,start.own.money};
    double checked=tail.model.value(v,tail.portfolio,nullptr,-1,&parts,&basis);
    if(std::abs(checked-tail.predicted)>1e-6)throw std::runtime_error("boundary tail breakdown mismatch");
    auto margin_model=tail.model;margin_model.cfg.competition=1;
    double margin_tail=margin_model.value(v,tail.portfolio,nullptr,-1,nullptr,&basis);
    double carried_rival_tail=0;
    if(have_prior_rival){auto carried=tail.model;carried.rival=prior_rival;
     carried_rival_tail=carried.value(v,tail.portfolio,nullptr,-1,nullptr,&basis);}
    double prior_market_tail=0;
    if(have_prior_market){auto market_prior=v.market;market_prior.inventory=prior_market_prediction;
     dp7::View old_market_view{v.step,v.day,v.hour,v.own,v.opponent,v.priv,market_prior,v.shops};
     prior_market_tail=tail.model.value(old_market_view,tail.portfolio,nullptr,-1,nullptr,&basis);}
    double prior_both_tail=0;
    if(have_prior_market&&have_prior_rival){auto market_prior=v.market;market_prior.inventory=prior_market_prediction;
     dp7::View old_market_view{v.step,v.day,v.hour,v.own,v.opponent,v.priv,market_prior,v.shops};
     auto carried=tail.model;carried.rival=prior_rival;
     prior_both_tail=carried.value(old_market_view,tail.portfolio,nullptr,-1,nullptr,&basis);}
    double prior_portfolio_tail=0,prior_portfolio_stock_tail=0;
    std::array<double,9> prior_stock_item_delta{},prior_stock_own_delta{},prior_stock_rival_delta{};
    if(have_prior_portfolio){
     competitive::ValueBreakdown old_parts;
     prior_portfolio_tail=tail.model.value(v,prior_portfolio,nullptr,-1,&old_parts,&basis);
     auto with_stock=prior_portfolio;
     for(int i=0;i<9;i++){int qty=v.priv.shed[i];for(const auto&bag:v.priv.inventories)qty+=bag[i];
      with_stock.f[v.day][i]+=qty;
      auto one=prior_portfolio;one.f[v.day][i]+=qty;
      competitive::ValueBreakdown one_parts;
      prior_stock_item_delta[i]=tail.model.value(v,one,nullptr,-1,&one_parts,&basis)-prior_portfolio_tail;
      prior_stock_own_delta[i]=one_parts.own_trade-old_parts.own_trade;
      prior_stock_rival_delta[i]=one_parts.rival_penalty-old_parts.rival_penalty;}
     prior_portfolio_stock_tail=tail.model.value(v,with_stock,nullptr,-1,nullptr,&basis);
    }
    if(!first)s<<",";first=false;
    s<<"{\"day\":"<<v.day<<",\"step\":"<<v.step
     <<",\"own_cash\":"<<world.own_cash()<<",\"rival_cash\":"<<world.rival_cash()
     <<",\"realized\":"<<realized<<",\"tail\":"<<tail.predicted
     <<",\"boundary\":"<<realized+tail.predicted
     <<",\"realized_margin\":"<<realized_margin<<",\"margin_tail\":"<<margin_tail
     <<",\"margin_boundary\":"<<realized_margin+margin_tail
     <<",\"searches\":"<<online.searches<<",\"search_changes\":"<<online.changes
     <<",\"tail_parts\":{\"fixed\":"<<parts.fixed<<",\"own_trade\":"<<parts.own_trade
     <<",\"wages\":"<<parts.wages<<",\"actions\":"<<parts.actions
     <<",\"rival_penalty\":"<<parts.rival_penalty
     <<",\"liquidity_penalty\":"<<parts.liquidity_penalty<<"}"
     <<",\"carried_rival_available\":"<<(have_prior_rival?"true":"false");
    if(have_prior_rival)s<<",\"carried_rival_tail\":"<<carried_rival_tail
     <<",\"carried_rival_delta\":"<<carried_rival_tail-tail.predicted;
    s<<",\"prior_market_available\":"<<(have_prior_market?"true":"false");
    if(have_prior_market)s<<",\"prior_market_tail\":"<<prior_market_tail
     <<",\"prior_market_delta\":"<<prior_market_tail-tail.predicted;
    if(have_prior_market&&have_prior_rival)s<<",\"prior_both_tail\":"<<prior_both_tail
     <<",\"prior_both_delta\":"<<prior_both_tail-tail.predicted;
    s<<",\"prior_portfolio_available\":"<<(have_prior_portfolio?"true":"false");
    if(have_prior_portfolio)s<<",\"prior_portfolio_tail\":"<<prior_portfolio_tail
     <<",\"prior_portfolio_stock_tail\":"<<prior_portfolio_stock_tail
     <<",\"prior_portfolio_stock_delta\":"<<prior_portfolio_stock_tail-tail.predicted;
    s<<",\"prior_stock_item_delta\":[";
    for(int i=0;i<9;i++){if(i)s<<",";s<<prior_stock_item_delta[i];}s<<"]";
    s<<",\"prior_stock_own_delta\":[";
    for(int i=0;i<9;i++){if(i)s<<",";s<<prior_stock_own_delta[i];}s<<"]";
    s<<",\"prior_stock_rival_delta\":[";
    for(int i=0;i<9;i++){if(i)s<<",";s<<prior_stock_rival_delta[i];}s<<"]";
    s
     <<",\"shift_item\":"<<shift_item<<",\"shift_units\":"<<shift_units;
    if(shift_item>=0&&v.day==day+days&&v.day+1<30){
     if(tail.portfolio.f[v.day][shift_item]<shift_units)
      throw std::runtime_error("boundary tail shift exceeds flow");
     competitive::Asset shifted=tail.portfolio;
     shifted.f[v.day][shift_item]-=shift_units;shifted.f[v.day+1][shift_item]+=shift_units;
     double shifted_value=tail.model.value(v,shifted,nullptr,-1,nullptr,&basis);
     s<<",\"shifted_tail\":"<<shifted_value
      <<",\"shift_delta\":"<<shifted_value-tail.predicted;
    }
    s
     <<",\"sold_previous_day\":[";
    for(int i=0;i<9;i++){if(i)s<<",";s<<sold[i];}s<<"],\"bought_previous_day\":[";
    for(int i=0;i<9;i++){if(i)s<<",";s<<bought[i];}s<<"],\"market_inventory\":[";
    for(int i=0;i<9;i++){if(i)s<<",";s<<v.market.inventory[i];}
    s<<"],\"prior_market_inventory\":[";
    for(int i=0;i<9;i++){if(i)s<<",";s<<prior_market_prediction[i];}
    s<<"],\"private_stock\":[";
    for(int i=0;i<9;i++){if(i)s<<",";int qty=v.priv.shed[i];
     for(const auto&bag:v.priv.inventories)qty+=bag[i];s<<qty;}
    s<<"],\"tail_own_by_item\":[";
    for(int i=0;i<9;i++){if(i)s<<",";s<<parts.own_trade_by_item[i];}
    s<<"],\"tail_rival_by_item\":[";
    for(int i=0;i<9;i++){if(i)s<<",";s<<parts.rival_penalty_by_item[i];}
    s<<"],\"tail_supply_by_item\":[";
    for(int i=0;i<9;i++){if(i)s<<",";double qty=0;for(int d=v.day;d<30;d++)qty+=tail.portfolio.f[d][i];s<<qty;}
    s<<"],\"tail_daily_flow\":[";
    for(int d=v.day;d<30;d++){if(d!=v.day)s<<",";s<<"[";
     for(int i=0;i<9;i++){if(i)s<<",";s<<tail.portfolio.f[d][i];}s<<"]";}
    s<<"],\"tail_rival_forecast_by_item\":[";
    for(int i=0;i<9;i++){if(i)s<<",";double qty=0;for(int d=v.day;d<30;d++)qty+=tail.model.rival[d][i];s<<qty;}
    s<<"],\"tail_rival_daily_flow\":[";
    for(int d=v.day;d<30;d++){if(d!=v.day)s<<",";s<<"[";
     for(int i=0;i<9;i++){if(i)s<<",";s<<tail.model.rival[d][i];}s<<"]";}
    s<<"],\"tail_demand_daily_flow\":[";
    for(int d=v.day;d<30;d++){if(d!=v.day)s<<",";s<<"[";
     for(int i=0;i<9;i++){if(i)s<<",";s<<tail.model.dem[d][i];}s<<"]";}
    s<<"]}";
    for(int i=0;i<9;i++){
     double stock=v.market.inventory[i],dem=tail.model.dem[v.day][i];
     double rival=tail.model.cfg.supply*tail.model.rival[v.day][i]*.5;
     stock-=dem*.5;competitive::Planner::trade(i,stock,rival);
     competitive::Planner::trade(i,stock,tail.portfolio.f[v.day][i]);
     competitive::Planner::trade(i,stock,rival);stock-=dem*.5;
     prior_market_prediction[i]=int(std::llround(stock));
    }
    have_prior_market=true;
    prior_rival=tail.model.rival;have_prior_rival=true;
    prior_portfolio=tail.portfolio;have_prior_portfolio=true;
    sold.fill(0);bought.fill(0);
   }
  }
  if(v.day==day+days)break;
  double own0=world.own_cash(),rival0=world.rival_cash();auto act=online.act(v);world.advance(act);
  const auto&fills=world.own_market_fills();
  if(fills.size()!=act.market.size())throw std::runtime_error("boundary market fill length");
  for(size_t j=0;j<fills.size();j++){int i=int(act.market[j].item);if(i<0||i>=9)continue;
   if(act.market[j].op==fastkag::Op::SELL)sold[i]+=fills[j];
   if(act.market[j].op==fastkag::Op::BUY_PRODUCT)bought[i]+=fills[j];}
  realized+=((world.own_cash()-own0)-online.base.competition*(world.rival_cash()-rival0))
   /std::pow(beta,v.day-day);
  realized_margin+=((world.own_cash()-own0)-(world.rival_cash()-rival0))
   /std::pow(beta,v.day-day);
 }
 s<<"],\"end_step\":"<<world.view().step<<",\"done\":"<<(world.done()?"true":"false")<<"}";
 h.text=s.str();return h.text.c_str();
 }catch(const std::exception&e){static_cast<Handle*>(p)->text=std::string("ERROR: ")+e.what();return nullptr;}}
extern "C" const char*td_candidate_transition_json(void*p,int index){
 auto&h=*static_cast<Handle*>(p);if(index<0||index>=int(h.prepared_audits.size()))return nullptr;
 const auto&a=h.prepared_audits[index];std::ostringstream s;s.precision(17);
 auto vec=[&](const char*name,const auto&v){s<<",\""<<name<<"\":[";for(size_t i=0;i<v.size();i++){if(i)s<<",";s<<v[i];}s<<"]";};
 s<<"{\"own_cash\":"<<a.rollout_own_cash<<",\"rival_cash\":"<<a.rollout_rival_cash;
 vec("market_inventory",a.transition_inventory);vec("market_prices",a.transition_prices);
 vec("own_assets",a.transition_own_assets);vec("rival_assets",a.transition_rival_assets);
 s<<"}";h.text=s.str();return h.text.c_str();
}
extern "C" const char*td_candidate_tail_items_json(void*p,int index){
 auto&h=*static_cast<Handle*>(p);if(index<0||index>=int(h.prepared_audits.size()))return nullptr;
 const auto&a=h.prepared_audits[index];std::ostringstream s;s.precision(17);
 auto vec=[&](const char*name,const auto&v){s<<",\""<<name<<"\":[";for(size_t i=0;i<v.size();i++){if(i)s<<",";s<<v[i];}s<<"]";};
 s<<"{\"tail\":"<<a.tail<<",\"tail_exact_market\":"<<a.static_tail_exact_market;
 vec("own_trade_by_item",a.tail_parts.own_trade_by_item);
 vec("rival_penalty_by_item",a.tail_parts.rival_penalty_by_item);
 vec("exact_own_trade_by_item",a.static_tail_exact_market_parts.own_trade_by_item);
 vec("exact_rival_penalty_by_item",a.static_tail_exact_market_parts.rival_penalty_by_item);
 vec("tail_flow_total",a.static_tail_flow_total);
 s<<"}";h.text=s.str();return h.text.c_str();
}
// Same-plan, single-incumbent harvest-age replacements. Diagnostic only:
// no resource calendar or executor is rerun, so gains are local value regret.
extern "C" const char*td_candidate_harvest_age_json(void*p,int index,const double*input,size_t count){try{
 auto&h=*static_cast<Handle*>(p);if(index<0||index>=int(h.prepared.size()))return nullptr;
 Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
 auto a=r.farm(),b=r.farm();auto priv=r.priv();fastkag::Market market;
 for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();
 std::vector<int8_t>shops;int n=r.count();for(int i=0;i<n;i++)shops.push_back(r.integer());
 if(r.i!=count||day!=step/24||hour!=step%24||seat<0||seat>1)return nullptr;
 dp7::View v{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};
 auto c=h.prepared[index].policy;competitive::ValueBreakdown base_parts;
 double baseline=c.model.value(v,c.portfolio,nullptr,-1,&base_parts);
 std::ostringstream s;s.precision(17);s<<"{\"baseline\":"<<baseline<<",\"rows\":[";
 int eligible=0,positive=0,mismatch=0;double max_gain=0;bool first_row=true;
 for(int pos=0;pos<100;pos++){
  const auto&t=v.own.tiles[pos];if(!dp7::plant(t)||dp7::ongoing(int(t.crop))||
    c.joint.index(pos)>=0||c.book[pos].successor>=0)continue;
  int k=int(t.crop),chosen=c.release[pos],earliest=std::max(day,int(t.planted_day)+dp7::first[k]);
  int last=std::min(29,int(t.planted_day)+(k==dp7::W?4:k==dp7::C?3:12));
  if(chosen<earliest||chosen>last)continue;
  eligible++;double best=baseline,chosen_rebuilt=0;int best_day=chosen;
  auto best_asset=c.paths[pos];auto best_parts=base_parts;
  for(int d=earliest;d<=last;d++){
   auto alt=c.crop(k,t.planted_day,pos,d,c.path_prices(),&t,nullptr,-1,-1,
      c.s.marginal_value>0?&c.forecast_crop_service[pos]:nullptr);
   auto portfolio=c.portfolio;competitive::add(portfolio,c.paths[pos],-1);competitive::add(portfolio,alt);
   competitive::ValueBreakdown parts;double val=c.model.value(v,portfolio,nullptr,-1,&parts);
   if(d==chosen)chosen_rebuilt=val-baseline;
   if(val>best+1e-9){best=val;best_day=d;best_asset=alt;best_parts=parts;}
  }
  mismatch+=std::abs(chosen_rebuilt)>1e-6;double gain=best-baseline;
  positive+=gain>1e-6;max_gain=std::max(max_gain,gain);
  if(!first_row)s<<',';first_row=false;
  double labor_delta=0;for(int d=day;d<30;d++)labor_delta+=best_asset.labor[d]-c.paths[pos].labor[d];
  double scalar_delta=c.scalar(best_asset,c.path_prices(),day)-c.scalar(c.paths[pos],c.path_prices(),day);
  s<<"{\"pos\":"<<pos<<",\"crop\":"<<k<<",\"chosen\":"<<chosen
   <<",\"best\":"<<best_day<<",\"gain\":"<<gain
   <<",\"scalar_delta\":"<<scalar_delta<<",\"labor_delta\":"<<labor_delta
   <<",\"own_trade_delta\":"<<best_parts.own_trade-base_parts.own_trade
   <<",\"rival_penalty_delta\":"<<best_parts.rival_penalty-base_parts.rival_penalty
   <<",\"wages_delta\":"<<best_parts.wages-base_parts.wages
   <<",\"actions_delta\":"<<best_parts.actions-base_parts.actions
   <<",\"fixed_delta\":"<<best_parts.fixed-base_parts.fixed
   <<",\"chosen_rebuild_delta\":"<<chosen_rebuilt<<"}";
 }
 s<<"],\"eligible\":"<<eligible<<",\"positive\":"<<positive
  <<",\"mismatch\":"<<mismatch<<",\"max_gain\":"<<max_gain<<"}";
 h.text=s.str();return h.text.c_str();
 }catch(const std::exception&e){static_cast<Handle*>(p)->text=std::string("ERROR: ")+e.what();return nullptr;}}
extern "C" const char*td_candidate_shop_branch_json(void*p,int index,const double*input,size_t count,int shop){try{
 auto&h=*static_cast<Handle*>(p);if(index<0||index>=int(h.prepared.size())||shop<-1||shop>7)return nullptr;
 for(int d=0;d<30;d++)for(int i=0;i<9;i++)
  if(h.prepared[index].policy.model.rival[d][i]!=h.prepared[0].policy.model.rival[d][i])
   throw std::runtime_error("shop branch candidates have different rival forecasts");
 Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
 auto a=r.farm(),b=r.farm();auto priv=r.priv();fastkag::Market market;
 for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();
 std::vector<int8_t>shops;int n=r.count();for(int i=0;i<n;i++)shops.push_back(r.integer());
 if(r.i!=count||day!=step/24||hour!=step%24||seat<0||seat>1)return nullptr;
 dp7::View v{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};
 triad::SearchController::ScoreAudit audit;
 double q=h.policy.score(v,h.prepared[index],30-day,&audit,shop);
 if(audit.continuation_first_key.empty()||(shop>=0&&audit.shop_branch_day15_key.empty()))return nullptr;
 std::ostringstream s;s.precision(17);s<<"{\"shop\":"<<shop<<",\"q\":"<<q;
 auto vec=[&](const char*name,const auto&v){s<<",\""<<name<<"\":[";for(size_t i=0;i<v.size();i++){if(i)s<<",";s<<v[i];}s<<"]";};
 vec("static_tail_key",h.prepared_audits[index].static_tail_key);
 vec("continuation_first_key",audit.continuation_first_key);
 vec("next_shop_plan_key",audit.shop_branch_day15_key);
 s<<",\"static_tail_predicted\":"<<h.prepared_audits[index].static_tail_predicted
  <<",\"continuation_first_predicted\":"<<audit.continuation_first_predicted;
 vec("static_tail_flow_day",h.prepared_audits[index].static_tail_flow_day);
 vec("continuation_first_flow_day",audit.continuation_first_flow_day);
 s<<"}";h.text=s.str();return h.text.c_str();
 }catch(const std::exception&e){static_cast<Handle*>(p)->text=std::string("ERROR: ")+e.what();return nullptr;}}
#endif

extern "C" int td_candidate_features(void*p,int index,double*out,size_t cap){auto&h=*static_cast<Handle*>(p);if(index<0||index>=int(h.prepared.size()))return -1;const auto&f=h.prepared[index].features.x;if(cap<f.size())return -2;std::copy(f.begin(),f.end(),out);return f.size();}
extern "C" int td_candidate_id(void*p,int index){auto&h=*static_cast<Handle*>(p);if(index<0||index>=int(h.prepared.size()))return -1;return h.prepared[index].id;}
extern "C" const char*td_candidate_json(void*p,int index){auto&h=*static_cast<Handle*>(p);if(index<0||index>=int(h.prepared.size()))return nullptr;const auto&q=h.prepared[index];const auto&a=h.prepared_audits[index];int crops=0,animals=0,hires=0;for(auto[pos,k]:q.policy.core.target){crops+=k>=0&&k<5;animals+=k>=9;}for(auto x:q.policy.core.queue)hires+=x.op==fastkag::Op::HIRE;double work1=q.policy.portfolio.labor[h.prepared_day],work3=0;for(int d=h.prepared_day;d<std::min(30,h.prepared_day+3);d++)work3+=q.policy.portfolio.labor[d];std::ostringstream s;s.precision(17);auto parts=[&](const char*name,const competitive::ValueBreakdown&b){s<<",\""<<name<<"\":{\"fixed\":"<<b.fixed<<",\"own_trade\":"<<b.own_trade<<",\"wages\":"<<b.wages<<",\"actions\":"<<b.actions<<",\"rival_penalty\":"<<b.rival_penalty<<",\"liquidity_penalty\":"<<b.liquidity_penalty<<",\"total\":"<<b.total<<"}";};s<<"{\"index\":"<<index<<",\"id\":"<<q.id<<",\"diagnostic\":\""<<h.prepared_names[index]<<"\",\"key_unique\":"<<(h.prepared_key_unique[index]?"true":"false")<<",\"score\":"<<h.prepared_scores[index]<<",\"scores_horizon\":[";for(int d=0;d<5;d++){if(d)s<<",";s<<h.prepared_horizons[index][d];}s<<"],\"predicted\":"<<q.policy.predicted<<",\"discount\":"<<q.policy.s.discount<<",\"capital_power\":"<<q.policy.s.capital_power<<",\"delay_sale\":"<<q.policy.s.delay_sale<<",\"rotation\":"<<q.policy.s.rotation<<",\"repeat\":"<<q.policy.s.repeat<<",\"target_crops\":"<<crops<<",\"target_animals\":"<<animals<<",\"hires\":"<<hires<<",\"forecast_work_1\":"<<work1<<",\"forecast_work_3\":"<<work3<<",\"scenario_rollout_own_cash\":"<<a.rollout_own_cash<<",\"scenario_rollout_rival_cash\":"<<a.rollout_rival_cash<<",\"scenario_rollout_objective\":"<<a.rollout_objective<<",\"scenario_tail\":"<<a.tail<<",\"scenario_frozen_clock_tail\":"<<a.frozen_clock_tail<<",\"scenario_frozen_clock_delta\":"<<a.frozen_clock_tail-a.tail<<",\"scenario_carried_tail\":"<<a.carried_tail<<",\"scenario_carried_delta\":"<<a.carried_tail-a.tail<<",\"scenario_carried_replan_tail\":"<<a.carried_replan_tail<<",\"scenario_carried_replan_delta\":"<<a.carried_replan_tail-a.tail<<",\"scenario_replan_changed\":"<<(a.replan_changed?"true":"false");parts("scenario_tail_parts",a.tail_parts);parts("scenario_frozen_clock_parts",a.frozen_clock_parts);parts("scenario_carried_tail_parts",a.carried_parts);parts("scenario_carried_replan_parts",a.carried_replan_parts);s<<",\"optimizer\":"<<q.policy.optimizer_json()<<",\"rival_flow_today\":[";for(int i=0;i<9;i++){if(i)s<<",";s<<a.initial[h.prepared_day][i];}s<<"],\"rival_future_mismatch\":[";for(int i=0;i<9;i++){if(i)s<<",";double x=0;for(int d=h.prepared_day+1;d<30;d++)x+=a.recomputed[d][i]-a.initial[d][i];s<<x;}s<<"]}";h.text=s.str();return h.text.c_str();}
extern "C" int td_install(void*p,int index){auto&h=*static_cast<Handle*>(p);if(index<0||index>=int(h.prepared.size()))return -1;h.policy.install(h.prepared[index],h.prepared_day);h.prepared.clear();h.prepared_scores.clear();h.prepared_horizons.clear();h.prepared_audits.clear();h.prepared_names.clear();h.prepared_key_unique.clear();return 0;}

// Offline diagnostic only: requested rival flow versus what the conditional
// scenario actually filled. It does not alter candidate scoring or execution.
extern "C" const char*td_candidate_flow_json(void*p,int index){auto&h=*static_cast<Handle*>(p);if(index<0||index>=int(h.prepared_audits.size()))return nullptr;const auto&a=h.prepared_audits[index];std::ostringstream s;s.precision(17);s<<"{";bool first=true;auto flow=[&](const char*name,const auto&x){if(!first)s<<",";first=false;s<<"\""<<name<<"\":[";for(int i=0;i<9;i++){if(i)s<<",";s<<x[i];}s<<"]";};flow("requested",a.rival_flow.requested);flow("rounded",a.rival_flow.rounded);flow("filled",a.rival_flow.filled);flow("market_delta",a.rival_flow.market_delta);flow("cash_delta",a.rival_flow.cash_delta);flow("shed_residual",a.rival_shed);s<<"}";h.text=s.str();return h.text.c_str();}
extern "C" const char*td_candidate_clock_json(void*p,int index){
 auto&h=*static_cast<Handle*>(p);if(index<0||index>=int(h.prepared_audits.size()))return nullptr;
 const auto&a=h.prepared_audits[index];std::ostringstream s;s.precision(17);
 s<<"{\"old_tail\":"<<a.tail<<",\"frozen_same_plan_tail\":"<<a.frozen_clock_tail
  <<",\"consistent_replan_tail\":"<<a.clock_consistent_tail
  <<",\"consistent_score\":"<<a.clock_consistent_score
  <<",\"rival_stock_delta\":"<<a.rival_stock_delta
  <<",\"rival_stock_score\":"<<a.rival_stock_score
  <<",\"replan_changed\":"<<(a.clock_replan_changed?"true":"false")<<"}";
 h.text=s.str();return h.text.c_str();
}

extern "C" size_t td_settings_count(){return triad::SETTINGS_COUNT;}
extern "C" const char*td_crop_clock_json(void*p){
 auto&h=*static_cast<Handle*>(p);auto&c=h.policy;std::ostringstream out;out.precision(17);
 out<<"{\"step\":"<<c.crop_clock.previous_step<<",\"counts\":[";
 for(int i=0;i<5;i++){if(i)out<<",";out<<c.crop_clock.count[i];}out<<"]";
 auto flow=[&](const char*name,const auto&f){out<<",\""<<name<<"\":[";for(size_t d=0;d<f.size();d++){if(d)out<<",";out<<"[";for(size_t i=0;i<f[d].size();i++){if(i)out<<",";out<<f[d][i];}out<<"]";}out<<"]";};
 flow("weights",c.crop_clock.weights());flow("observed_forecast",c.removal_forecast);flow("live_forecast",c.live.model.rival);out<<"}";h.text=out.str();return h.text.c_str();
}
extern "C" const char*td_clock_json(void*p){
 auto&h=*static_cast<Handle*>(p);auto&c=h.policy.sale_clock;auto&l=h.policy.ledger;
 std::ostringstream out;out<<"{\"step\":"<<l.observation_step<<",\"source_step\":"<<l.source_step<<",\"invalid_market\":"<<l.invalid_market;
 auto ints=[&](const char*name,const auto&a){out<<",\""<<name<<"\":[";for(int i=0;i<9;i++){if(i)out<<",";out<<a[i];}out<<"]";};
 ints("own_sales",l.own_sales);ints("own_valid",l.own_valid);ints("rival_sales",l.rival_net);ints("rival_valid",l.valid);
 auto reals=[&](const char*name,const auto&a){out<<",\""<<name<<"\":[";for(int i=0;i<9;i++){if(i)out<<",";out<<a[i];}out<<"]";};
 reals("rival_pending_lower",l.lower);reals("rival_pending_upper",l.upper);reals("rival_pending_added",l.added);reals("rival_pending_certain_added",l.certain_added);
 auto profiles=[&](const char*name,const auto&a){out<<",\""<<name<<"\":[";for(int i=0;i<9;i++){if(i)out<<",";out<<"{\"days\":"<<a[i].count()<<",\"density\":[";auto d=a[i].density();for(int t=0;t<24;t++){if(t)out<<",";out<<d[t];}out<<"]}";}out<<"]";};
 profiles("own",c.own);profiles("rival",c.rival);out<<"}";h.text=out.str();return h.text.c_str();
}

extern "C" const char*td_repair_debug(void*p){
 auto&h=*static_cast<Handle*>(p);const auto&s=h.policy.live.core.t3;
 std::ostringstream o;o<<"{\"obligation_checks\":"<<s.obligation_checks
 <<",\"water_insertions\":"<<s.water_insertions<<",\"water_unresolved_checks\":"<<s.water_unresolved
 <<",\"capacity_checks\":"<<s.capacity_checks<<",\"capacity_rollouts\":"<<s.capacity_rollouts
 <<",\"delivery_insertions\":"<<s.delivery_insertions<<",\"projected_overflow_saved\":"<<s.projected_overflow_saved
 <<",\"receipt_checks\":"<<s.receipt_checks<<",\"receipt_hires\":"<<s.receipt_hires<<",\"receipt_seeds\":"<<s.receipt_seeds
 <<",\"transport_admissions\":"<<s.transport_admissions<<"}";h.text=o.str();return h.text.c_str();}

// Read-only diagnosis for the CURRENT legal observation. No hidden state input.
extern "C" const char* td_obligations_json(void*p,const double*input,size_t count){
 auto&h=*static_cast<Handle*>(p);
 try {
  Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
  auto a=r.farm(),b=r.farm();auto priv=r.priv();fastkag::Market market;
  for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();std::vector<int8_t>shops;int n=r.count();for(int i=0;i<n;i++)shops.push_back(r.integer());
  if(r.i!=count||seat<0||seat>1)throw std::runtime_error("inspect observation");
  dp7::View v{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};
  const auto&live=h.policy.live;const auto&c=live.core;
  std::ostringstream s;s<<"{\"step\":"<<step<<",\"phase\":"<<c.phase<<",\"feed_shortfall\":"<<c.feed_shortfall(v)<<",\"plan_dropped\":"<<c.actual_drop<<",\"resource_degraded\":"<<c.resource_degraded<<",\"input_rejected_atoms\":"<<c.input_rejected_atoms<<",\"input_rejected_feed\":"<<c.input_rejected_feed<<",\"service\":[";
  bool comma=false;for(int pos=0;pos<100;pos++)if(dp7::animal(v.own.tiles[pos])){if(comma)s<<",";comma=true;s<<"["<<pos<<","<<int(c.service_feed[pos])<<","<<int(c.service_care[pos])<<","<<int(v.own.tiles[pos].consecutive_unfed)<<"]";}s<<"],\"queued\":[";
  comma=false;for(auto x:c.queue){if(comma)s<<",";comma=true;s<<"["<<int(x.op)<<","<<int(x.item)<<","<<x.quantity<<"]";}s<<"],\"raw_jobs\":[";
  comma=false;for(const auto&j:c.jobs(v)){if(comma)s<<",";comma=true;s<<"{\"pos\":"<<j.pos<<",\"actions\":[";bool sep=false;for(auto x:j.actions){if(sep)s<<",";sep=true;s<<int(x.op);}s<<"],\"wheat\":"<<j.needs[0]<<"}";}s<<"],\"remaining\":[";
  comma=false;for(const auto&pl:c.plans){if(comma)s<<",";comma=true;s<<"[";bool sep=false;for(size_t k=pl.index;k<pl.a.size();k++){if(sep)s<<",";sep=true;s<<"["<<int(pl.a[k].op)<<","<<int(pl.a[k].item)<<","<<pl.a[k].quantity<<","<<pl.target[k]<<"]";}s<<"]";}s<<"]}";
  h.text=s.str();return h.text.c_str();
 }catch(const std::exception&e){h.text=std::string("ERROR: ")+e.what();return h.text.c_str();}
}
