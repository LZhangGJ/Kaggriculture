#pragma once
#include "triad.hpp"
namespace triad {
struct Features {
 std::vector<double> x;std::vector<std::string>names;bool naming=false;
 void add(const std::string&name,double value){if(!std::isfinite(value))value=0;x.push_back(value);if(naming)names.push_back(name);}
};
inline std::vector<Settings> alternative_settings(Settings base){
 int width=std::clamp(int(base.portfolio_passes),2,9);std::vector<Settings>s(width,base);
 for(auto&v:s)v.scenario=0;
 if(width>1){s[1].discount=std::max(.08,base.discount*2);s[1].capital_power=std::max(.5,base.capital_power);}
 if(width>2)s[2].animal_bias*=.6;
 if(width>3){s[3].animal_bias*=1.6;s[3].capital_power=std::max(0.,base.capital_power-.2);}
 if(width>4)s[4].competition*=.5;
 if(width>5)s[5].competition*=1.5;
 if(width>6)s[6].labor_hours=std::max(6.,base.labor_hours*.8);
 if(width>7)s[7].land_rent*=2;
 if(width>8)s[8].max_land=4;
 if(base.candidate_extra>0){
  auto slow=base;slow.scenario=0;slow.capital_power=0;slow.discount=0; s.push_back(slow);
  auto liquid=base;liquid.scenario=0;liquid.reserve=std::max(300.,base.reserve*3);liquid.feed_cover=2;s.push_back(liquid);
  auto crops=slow;crops.animal_bias*=.3;s.push_back(crops);
  if(base.candidate_extra>=2){
   auto affordable=base;affordable.scenario=0;affordable.work_price*=.5;affordable.labor_hours=std::min(20.,base.labor_hours+3);s.push_back(affordable);
   auto rent=slow;rent.land_rent=std::max(8.,base.land_rent*3);s.push_back(rent);
   auto noexpand=base;noexpand.scenario=0;noexpand.max_land=2;s.push_back(noexpand);
  }
 }
 return s;
}
inline std::vector<int> proposal_key(const Controller&p){
 std::vector<int>key;for(auto[pos,k]:p.core.target){key.push_back(pos);key.push_back(k);key.push_back(p.core.triad_crop_age[pos]);}
 key.push_back(p.core.planned_land);for(auto a:p.core.queue){key.push_back(int(a.op));key.push_back(int(a.item));key.push_back(a.quantity);}
 for(int pos=0;pos<100;pos++){key.push_back(p.core.service_feed[pos]);key.push_back(p.core.service_care[pos]);key.push_back(p.core.crop_water[pos]);key.push_back(p.core.crop_fertilize[pos]);}
 return key;
}
inline Features portfolio_features(const View&o,const Controller&c,const Controller&keep,int id,bool naming=false){
 Features f;f.naming=naming;auto add=[&](std::string n,double v){f.add(n,v);};
 add("day",o.day);add("hour",o.hour);add("own_cash",o.own.money);add("rival_public_cash",o.opponent.money);add("public_cash_margin",o.own.money-o.opponent.money);
 std::array<int,8>shops{};for(int s:o.shops)shops[s]++;
 for(int i=0;i<8;i++)add("known_shop_"+std::to_string(i),shops[i]);
 for(int i=0;i<9;i++){add("market_inventory_"+std::to_string(i),o.market.inventory[i]-10000);add("market_price_"+std::to_string(i),o.market.prices[i]);}
 for(int i=0;i<12;i++)add("own_shed_"+std::to_string(i),o.priv.shed[i]);
 for(int i=0;i<5;i++)add("own_seed_"+std::to_string(i),o.priv.seeds[i]);
 for(int side=0;side<2;side++){
  const Farm&farm=side?o.opponent:o.own;std::string p=side?"rival_public_":"own_";
  add(p+"land",std::popcount(unsigned(farm.unlocked_mask)));add(p+"workers",farm.hands.size()+1);
  std::array<int,12>count{},age{},recent{},young{},dry{},bonus{};std::array<int,9>yield{};std::array<double,12>travel{};std::array<int,4>free{},weed{};
  for(int pos=0;pos<100;pos++){
   const auto&t=farm.tiles[pos];int k=animal(t)?int(t.animal):plant(t)?int(t.crop):-1;
   if(k<0){free[quad(pos)]+=t.kind==TileKind::EMPTY;weed[quad(pos)]+=t.kind==TileKind::WEED;continue;}
   int a=o.day-(k>=9?t.placed_day:t.planted_day);count[k]++;age[k]+=a;recent[k]+=a<=1;young[k]+=a<=3;travel[k]+=near(pos);
   dry[k]+=k>=9?t.consecutive_unfed:t.consecutive_unwatered;bonus[k]+=k>=9?t.pending_care_bonus:(t.fertilized_until_day>=o.day);
   yield[k>=9?product[k-9]:k]+=t.yield_units;
  }
  for(int k:{0,1,2,3,4,9,10,11}){std::string n=p+std::to_string(k)+"_";add(n+"count",count[k]);add(n+"mean_age",count[k]?double(age[k])/count[k]:0);add(n+"new_1d",recent[k]);add(n+"new_3d",young[k]);add(n+"dry",dry[k]);add(n+"bonus",bonus[k]);add(n+"depot_distance",travel[k]);}
  for(int i=0;i<9;i++)add(p+"held_yield_"+std::to_string(i),yield[i]);
  for(int q=0;q<4;q++){add(p+"free_q"+std::to_string(q),free[q]);add(p+"weed_q"+std::to_string(q),weed[q]);}
 }
 add("proposal_variant",id);add("candidate_competition",c.s.competition);add("candidate_discount",c.s.discount);add("candidate_capital_power",c.s.capital_power);add("candidate_animal_bias",c.s.animal_bias);add("candidate_labor_hours",c.s.labor_hours);add("candidate_land_rent",c.s.land_rent);
 add("candidate_predicted_value",c.predicted);add("predicted_difference_from_keep",c.predicted-keep.predicted);add("candidate_land_target",c.core.planned_land);
 std::array<int,12>targets{},newtargets{},keeptargets{};for(auto[p,k]:keep.core.target)if(k>=0)keeptargets[k]++;
 for(auto[p,k]:c.core.target)if(k>=0){targets[k]++;const auto&t=o.own.tiles[p];int current=animal(t)?int(t.animal):plant(t)?int(t.crop):-1;newtargets[k]+=current!=k;}
 for(int k:{0,1,2,3,4,9,10,11}){add("target_"+std::to_string(k),targets[k]);add("target_delta_"+std::to_string(k),targets[k]-keeptargets[k]);add("new_target_"+std::to_string(k),newtargets[k]);}
 std::array<int,24>orders{};std::array<int,12>buy{},sale{};double spending=0;int staff=0;
 for(auto a:c.core.queue){orders[int(a.op)]+=a.quantity;if(a.op==Op::HIRE)staff++;if(a.op==Op::BUY_SEED||a.op==Op::BUY_ANIMAL||a.op==Op::BUY_PRODUCT){buy[int(a.item)]+=a.quantity;spending+=c.core.cost(o,a);}if(a.op==Op::SELL)sale[int(a.item)]+=a.quantity;}
 add("immediate_spending",spending);add("hire_target",staff);add("market_orders",c.core.queue.size());
 for(int i=0;i<12;i++)add("buy_"+std::to_string(i),buy[i]);for(int i=0;i<9;i++)add("sell_"+std::to_string(i),sale[i]);
 for(int h:{1,3,7,14,30}){
  double work=0,peak=0;std::array<double,9>flow{},diff{};
  for(int d=o.day;d<std::min(30,o.day+h);d++){work+=c.portfolio.labor[d];peak=std::max(peak,c.portfolio.labor[d]);for(int i=0;i<9;i++){flow[i]+=c.portfolio.f[d][i];diff[i]+=c.portfolio.f[d][i]-keep.portfolio.f[d][i];}}
  add("forecast_work_"+std::to_string(h),work);add("forecast_peak_work_"+std::to_string(h),peak);
  for(int i=0;i<9;i++){add("forecast_flow_"+std::to_string(h)+"_"+std::to_string(i),flow[i]);add("flow_difference_"+std::to_string(h)+"_"+std::to_string(i),diff[i]);}
 }
 return f;
}
struct Proposal {int id=0;Controller policy;Features features;};
inline std::vector<Proposal> generate_proposals(const Controller&live,const Settings&base,const View&o,bool naming=false){
 std::vector<Proposal>out;std::set<std::vector<int>>seen;auto settings=alternative_settings(base);
 for(int i=0;i<int(settings.size());i++){
  Controller p=live;p.configure(settings[i]);p.plan(o);
  if(!seen.insert(proposal_key(p)).second)continue;
  out.push_back({i,std::move(p),{}});
 }
 if(naming)for(auto&p:out)p.features=portfolio_features(o,p.policy,out[0].policy,p.id,naming);
 return out;
}
}
