// Included inside triad::Controller. Comparison shares current plans, real
// inventories and r6's public-flow assumption; it never consumes a replay suffix.
 struct R11Value {double own=0,rival=0;fastkag::Farm farm;int ticks=0;};
 R11Value r11_value(const View&o,const PlayerAction&first,const std::optional<r11::Choice>&choice){
  auto probe=*this;probe.r11_evaluating=true;probe.labor11.active=bool(choice);if(choice)probe.labor11.pending=*choice;
  fastkag::PublicFlowScenario world(o,model.rival,s.supply,nullptr);world.advance(first);int ticks=1;
  while(!world.done()&&world.view().day==o.day){auto v=world.view();auto a=probe.act(v);world.advance(a);ticks++;}
  R11Value out;out.own=world.own_cash();out.rival=world.rival_cash();out.farm=world.view().own;out.ticks=ticks;return out;
 }
 void r11_consider(const View&o,PlayerAction&out){
  if(s.a06_r11_recovery<=0||r11_evaluating||labor11.active||core.phase!=3||core.pending_admission.active||!core.queue.empty())return;
  if(o.hour>=22||out.market.size()>=10||core.plans.size()!=o.priv.inventories.size())return;
  // Current investment orders take precedence. A new wage is never justified
  // by an unfilled SALE/BUY, and cannot steal its current transaction budget.
  if(std::any_of(out.market.begin(),out.market.end(),[](auto a){return a.op!=Op::SELL;}))return;
  auto jobs=r11::uncovered(core,o,out);if(jobs.empty())return;labor11.checks++;
  fastkag::ObservedDayScenario current(o);auto post=current.project_units(out.units,-1);auto&pr=post.privates()[0];auto&farm=post.farms()[0];
  auto reserved=dp7::intraday::reserved(core).shed;
  // Establishment inputs already paid for but temporarily unassigned are not
  // spare material for a rescue. Preserve their feed/fertilizer needs too.
  for(const auto&j:core.jobs(o))if(std::any_of(j.actions.begin(),j.actions.end(),[](auto a){return a.op==Op::PLANT||a.op==Op::PLACE;})){
   bool assigned=dp7::intraday::reserved(core).tiles[j.pos];if(!assigned&&book[j.pos].funded)add(reserved,j.needs);
  }
  Counts free{};for(int i=0;i<12;i++)free[i]=std::max(0,pr.shed[i]-reserved[i]);
  int ticks=(o.day==29?23:24)-o.hour-1,n=int(o.own.hands.size())+1;std::vector<r11::Choice>choices;
  for(int u=0;u<n;u++)if(out.units[u].op==Op::PASS&&core.plans[u].index>=core.plans[u].a.size()&&sum(pr.inventories[u])==0){
   auto q=r11::solve(jobs,free,o.market,u,cell(u?farm.hands[u-1]:farm.farmer),ticks,false,&labor11);if(q)choices.push_back(std::move(*q));
  }
  // At most two no-cost worker alternatives plus one extra worker are probed.
  std::stable_sort(choices.begin(),choices.end(),[](auto&a,auto&b){return std::tuple(a.rank,-int(a.route.a.size()),-a.unit)>std::tuple(b.rank,-int(b.route.a.size()),-b.unit);});if(choices.size()>2)choices.resize(2);
  int hires=o.own.hires_today;bool affordable=hires>=0&&hires<int(fib.size())&&int(o.own.hands.size())<int(s.max_hands)&&o.own.money>=fib[hires]+s.reserve;
  if(affordable){auto orders=out.market;orders.push_back(action(Op::HIRE));auto hired=post.project_own_market(0,orders);
   if(hired.farms()[0].hands.size()==farm.hands.size()+1){auto q=r11::solve(jobs,free,o.market,n,cell(hired.farms()[0].hands.back()),ticks,true,&labor11);if(q)choices.push_back(std::move(*q));}
  }
  if(choices.empty())return;
  auto keep=r11_value(o,out,{});labor11.rollouts++;double best_gain=0;std::optional<r11::Choice>best;PlayerAction best_out;
  std::ostringstream audit;audit.precision(17);audit<<"{\"step\":"<<o.step<<",\"own_cash_observed\":"<<o.own.money<<",\"actual_workers\":"<<n<<",\"actual_hires_today\":"<<hires<<",\"keep_cash\":"<<keep.own<<",\"keep_rival_cash\":"<<keep.rival<<",\"scenarios\":[";
  for(size_t i=0;i<choices.size();i++){
   auto q=choices[i];q.issue_step=o.step;q.day=o.day;q.before_workers=n;q.before_hires=hires;
   auto a=out;r11::protect_orders(a,pr,reserved,q.needs);if(q.hire)a.market.push_back(action(Op::HIRE));
   auto value=r11_value(o,a,q);labor11.rollouts++;double gain=value.own-keep.own,margin_gain=gain-(value.rival-keep.rival);
   bool safe=r11::protected_assets(o.own,keep.farm,value.farm);
   // Retain already-paid same-day establishment fulfilled by KEEP.
   for(int pos=0;pos<100;pos++)if(book[pos].funded&&!plant(o.own.tiles[pos])&&!animal(o.own.tiles[pos])&&(plant(keep.farm.tiles[pos])||animal(keep.farm.tiles[pos])))if(!r11::same_asset(keep.farm.tiles[pos],value.farm.tiles[pos]))safe=false;
   if(i)audit<<",";audit<<"{\"worker\":"<<q.unit<<",\"hire\":"<<q.hire<<",\"wage\":"<<(q.hire?fib[hires]:0)<<",\"steps\":"<<q.route.a.size()<<",\"cash\":"<<value.own<<",\"rival_cash\":"<<value.rival<<",\"own_gain\":"<<gain<<",\"margin_gain\":"<<margin_gain<<",\"service_guard\":"<<safe<<",\"plots\":[";for(size_t j=0;j<q.jobs.size();j++){if(j)audit<<",";audit<<q.jobs[j].pos;}audit<<"]}";
   if(safe&&gain>best_gain+1e-6&&margin_gain>=-1e-6){best_gain=gain;best=q;best_out=std::move(a);labor11.last_gain=gain;labor11.last_margin_gain=margin_gain;}
  }
  audit<<"],\"selected_worker\":"<<(best?best->unit:-1)<<",\"selected_hire\":"<<(best&&best->hire)<<",\"scope\":\"same-day conditional simulation, not actual earned cash\"}";labor11.last=audit.str();
  if(best){labor11.pending=*best;labor11.active=true;labor11.offered++;labor11.hire_requests+=best->hire;out=std::move(best_out);}
 }
