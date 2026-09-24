// Included inside triad::Controller. A06 r3's bounded WAIT/NOW recurrence.
// Inputs: live public observation, our private inventory, unexecuted contracts.
// Never reads seed, replay, realized future shops, or another farm's inventory.
// Today must pass the incumbent exact input/route/remaining-hour compiler.
// Days after today remain CONDITIONAL lifecycle/workforce forecasts, not actual
// future execution. No extra beam levels or capital penalties are introduced.
std::array<double,12> a06_remaining_values(const dp7::Controller&c,const View&o,const PlayerAction&out,bool unit_phase_settled=false){
 a06_live_checks++;
 std::array<double,12> result;result.fill(-1e90);a06_last_now.fill(-1e90);
 auto reservations=dp7::intraday::reserved(c);
 for(size_t u=0;u<out.units.size();u++){
  auto x=out.units[u];int i=int(x.item),p=cell(u?o.own.hands[u-1]:o.own.farmer);
  if(x.op==Op::PLANT&&i>=0&&i<5)reservations.seeds[i]++;
  if(x.op==Op::PICKUP&&i>=0&&i<12)reservations.shed[i]+=x.quantity;
  if(x.op!=Op::PASS&&!dp7::Controller::movement(x.op)&&x.op!=Op::DROP&&x.op!=Op::PICKUP)reservations.tiles[p]=true;
 }
 // Use an actually vacant, unpromised plot, not the first stale daily target.
 int representative=-1;
 for(int p=0;p<100;p++)if(dp7::intraday::vacant(o,p)&&!reservations.tiles[p]&&c.plant_not_before[p]<=o.day)
  if(representative<0||std::tuple(near(p),p)<std::tuple(near(representative),representative))representative=p;
 if(representative<0)return result;

 // Known town consumption remaining today only. Unknown future shops keep the
 // inherited expectation from current public information in BOTH alternatives.
 Planner forecast=model;forecast.day=o.day;forecast.demand(o);forecast.public_rival(o);
 // A06 cash objective: no opponent-cash reward, no invented per-action cash
 // expense, no additional liquidity penalty. Existing future wage model kept.
 // r6: retain the same rival-market externality as the chosen daily plan.
 // It is a conditional payoff DIFFERENCE, not observed enemy cash or a claim
 // that rival supply is known. Wages, sunk input accounting and admission
 // constraints are unchanged in this ablation.
 forecast.cfg.competition=(int(s.a06_competitive_contract)&1)?s.competition:0;forecast.cfg.discount=0;forecast.cfg.action_cost=0;forecast.cfg.risk=0;
 static const std::array<std::vector<int>,8> shops{{{E,W},{E,W,S},{W,C,T,S},{S,MI,W},{C,C},{MI,T,W},{S,MI},{WO,WO}}};
 forecast.dem[o.day].fill(0);
 for(int tick=o.step;tick<std::min(719,(o.day+1)*24);tick++){
  if(tick%24==0)for(int i=0;i<8;i++)forecast.dem[o.day][i]+=1;
  if(tick%4==0)for(int sh:o.shops)for(int i:shops[sh])forecast.dem[o.day][i]+=1;
 }

 // Delivery labels use only our already-issued current action and remaining
 // route. Unscheduled/overnight automatic deposits are not sold before tomorrow.
 std::array<int,100> delivery;delivery.fill(std::min(29,o.day+1));
 std::array<int,100> established;established.fill(-1);
 std::vector<int> cargo_day(o.priv.inventories.size(),std::min(29,o.day+1));
 for(size_t u=0;u<c.plans.size();u++){
  std::vector<std::pair<Action,int>> route;
  if(u<out.units.size())route.push_back({out.units[u],cell(u?o.own.hands[u-1]:o.own.farmer)});
  auto&pl=c.plans[u];for(size_t i=pl.index;i<pl.a.size();i++)route.push_back({pl.a[i],pl.target[i]});
  bool deposit=false;
  for(int i=int(route.size())-1;i>=0;i--){
   if(o.hour+i+(unit_phase_settled?1:0)>=24)continue;auto [x,p]=route[i];int k=int(x.item);
   if(p<0||p>=100)continue;
   if(at_depot(p)&&(x.op==Op::DROP||(x.op==Op::PLACE&&k>=0&&k<9)))deposit=true;
   if(x.op==Op::HARVEST||x.op==Op::COLLECT_FERTILIZER)
    delivery[p]=deposit?o.day:std::min(29,o.day+1);
   if((x.op==Op::PLANT&&k>=0&&k<5)||(x.op==Op::PLACE&&k>=9&&k<12))established[p]=k;
  }
  if(deposit)cargo_day[u]=o.day;
 }
 Asset base;Flow px=prices;
 for(int i=0;i<9;i++){
  base.f[o.day][i]+=o.priv.shed[i];
  for(size_t u=0;u<o.priv.inventories.size();u++)base.f[cargo_day[u]][i]+=o.priv.inventories[u][i];
 }
 forecast.value(o,base,&px);
 for(int p=0;p<100;p++){
  const auto&t=o.own.tiles[p];Asset a;
  if(animal(t))a=animal_path(int(t.animal),t.placed_day,p,px,&t);
  else if(plant(t)){
   int k=int(t.crop),age=c.triad_crop_age[p]>=0?c.triad_crop_age[p]:(ongoing(k)?first[k]+3*interval[k]:c.h_age(k));
   int finish=std::clamp(int(t.planted_day)+age,o.day,29);
   a=crop(k,t.planted_day,p,finish,px,&t);
  }
  if(animal(t)||plant(t)){
   // Shift currently available positive output only. This is not another copy
   // of the output that was harvested/sold earlier in the day.
   if(delivery[p]>o.day)for(int i=0;i<9;i++)if(a.f[o.day][i]>0){a.f[delivery[p]][i]+=a.f[o.day][i];a.f[o.day][i]=0;}
   add(base,a);
  }
  int k=established[p];if(k<0)continue;
  // Explicit not-yet-executed PLANT/PLACE is a binding establishment contract,
  // including a successor behind a HARVEST. Its seed/animal is already funded.
  auto dp=rotations_dp(p,px,o.day);
  Asset committed=k>=9?animal_path(k,o.day,p,px):choose_crop(k,o.day,p,px,dp,&o,&base,&forecast,true,true).a;
  if(committed.first_cost>0){committed.fixed[o.day]+=committed.first_cost;committed.first_cost=0;add(base,committed);}
 }
 // All currently hired workers have already been paid. No hypothetical second
 // wage bill is charged today; exact remaining capacity is checked by propose.
 base.labor[o.day]=0;
 forecast.value(o,base,&px);
 double base_value=forecast.value(o,base),wait=0;
 auto today_dp=rotations_dp(representative,px,o.day);
 auto tomorrow_dp=rotations_dp(representative,px,std::min(29,o.day+1));
 auto live=c.counts(o);int animals=live[G]+live[CO]+live[SH];
 for(int k:{0,1,2,3,4,9,10,11}){
  if(k>=9&&(animals>=c.p.max_animals||o.day>c.p.latest_animal_day))continue;
  // Sunk, unreserved inputs are credited identically in every calendar label.
  int spare=k<5?o.priv.seeds[k]-reservations.seeds[k]:o.priv.shed[k]-reservations.shed[k];
  Asset now=k>=9?animal_path(k,o.day,representative,px):choose_crop(k,o.day,representative,px,today_dp,&o,&base,&forecast,spare>0,true).a;
  if(now.first_cost<=0)continue;
  if(spare>0){now.fixed[o.day]+=now.first_cost;now.first_cost=0;}
  now.labor[o.day]=0;
  auto with=base;add(with,now);a06_last_now[k]=forecast.value(o,with)-base_value;
  result[k]=a06_last_now[k];
  if(o.day+1<29){
   Asset later=k>=9?animal_path(k,o.day+1,representative,px):choose_crop(k,o.day+1,representative,px,tomorrow_dp,&o,&base,&forecast,spare>0,false).a;
   if(later.first_cost>0){
    // WAIT does not get to borrow unseen receipts. Its upfront seed/animal
    // must fit CURRENT uncommitted cash (or already paid spare stock).
    double budget=o.own.money-dp7::intraday::committed_market_cost(c,o,out)-s.reserve;
    if(spare>0){later.fixed[o.day+1]+=later.first_cost;later.first_cost=0;}
    if(later.first_cost<=budget){auto deferred=base;add(deferred,later);wait=std::max(wait,forecast.value(o,deferred)-base_value);}
   }
  }
 }
 // One-step Bellman stop/wait option. No alternate future observation is read.
 // V = max(STOP, WAIT-one-day, feasible NOW[k]); the route compiler retains
 // current stock/cash/time feasibility. Reporting is a forecast, not cash.
 a06_last_wait=wait;a06_last_best_now=*std::max_element(result.begin(),result.end());
 if(s.a06_reinvest==3||s.a06_reinvest==5)for(int k:{0,1,2,3,4,9,10,11})if(result[k]<=wait){if(result[k]>0)a06_wait_rejections++;result[k]=-1e90;}
 return result;
}

// Reconcile the unit-phase boundary using ONLY known current own actions.
// Plan cursors have already advanced. Apply those actions exactly once, omit
// them from remaining input reservations, and start remaining routes at +1.
// No market order, clock tick, rival action or random event is executed here.
// The original admission compiler still checks pre-sale cash and input fills.
std::array<double,12> a06_live_intraday_values(const dp7::Controller&c,const View&o,const PlayerAction&out){
 if(s.a06_state_contract<=0)return a06_remaining_values(c,o,out);
 fastkag::ObservedDayScenario scenario(o);auto post=scenario.project_units(out.units,-1);
 View v{o.step,o.day,o.hour,post.farms()[0],o.opponent,post.privates()[0],o.market,o.shops};
 PlayerAction rest=out;rest.units.clear();a06_contract_calls++;
 auto values=a06_remaining_values(c,v,rest,true);
 std::ostringstream a;a.precision(17);a<<"{\"step\":"<<o.step<<",\"prediction\":"<<prediction()<<",\"cash\":"<<o.own.money<<",\"projected_unit_cash\":"<<v.own.money<<",\"unsettled_order_cost\":"<<dp7::intraday::committed_market_cost(c,o,out);
 auto counts=[&](const char*name,const auto&z){a<<",\""<<name<<"\":[";for(size_t i=0;i<z.size();i++){if(i)a<<",";a<<z[i];}a<<"]";};
 counts("shed_before",o.priv.shed);counts("shed_after_units",v.priv.shed);counts("seeds_before",o.priv.seeds);counts("seeds_after_units",v.priv.seeds);
 auto reserved=dp7::intraday::reserved(c);counts("remaining_pickup_reserve",reserved.shed);counts("remaining_seed_reserve",reserved.seeds);counts("remaining_flow_marginals",values);
 a<<",\"pending\":"<<c.pending_admission.active<<",\"remaining_labor\":[";
 for(size_t u=0;u<c.plans.size();u++){if(u)a<<",";a<<c.plans[u].a.size()-c.plans[u].index;}a<<"]}";a06_contract_last=a.str();return values;
}
