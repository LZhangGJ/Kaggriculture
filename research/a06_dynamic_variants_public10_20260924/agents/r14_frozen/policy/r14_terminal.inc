// Inside SearchController: finite late-game policy alternatives evaluated to
// the terminal cash boundary. Unknown rival inventory and future supply remain
// scenario assumptions; no live referee, random seed or rival code is accepted.
int terminal_searches=0,terminal_rollouts=0,terminal_changes=0;
double terminal_last_gain=0;
void terminal_choose(const View&o){
 const int mode=int(base.r14_terminal_mode);
 std::vector<Settings> alternatives;
 auto add=[&](Settings x){x.scenario=0;x.r14_terminal_mode=0;alternatives.push_back(x);};
 add(base);
 auto s=base;s.work_price=2;add(s);
 s=base;s.work_price=8;add(s);
 s=base;s.harvest_threshold=4;add(s);
 s=base;s.intraday=0;add(s);
 s=base;s.a06_r12_calendar=0;add(s);
 s=base;s.max_hands=std::max(4.,base.max_hands-2);add(s);
 if(mode>=2){s=base;s.competition=.5;add(s);s=base;s.competition=3;add(s);s=base;s.rotation=1;add(s);}
 auto forecast=live.model;forecast.day=o.day;forecast.demand(o);forecast.public_rival(o);
 auto flows=forecast.rival;
 // A conservative observable upper bound is optional, not hidden stock.
 if(mode==3)for(int i=1;i<=7;i++)flows[o.day][i]+=std::min(40.,ledger.upper[i]);
 double best=-1e100,first=0;int chosen=0;Controller winner=live;
 std::set<std::vector<int>> seen;
 for(int j=0;j<int(alternatives.size());j++){
  Controller prepared=live;prepared.configure(alternatives[j]);prepared.plan(o);
  // Executor/service settings can differ even with identical target keys.
  auto roll=prepared;roll.call_origin=Controller::CallOrigin::PublicPrediction;roll.previous_step=o.step-1;
  fastkag::PublicFlowScenario world(o,flows,base.supply,base.r13_clock>=1?&sale_clock:nullptr);
  while(!world.done()){auto v=world.view();auto action=roll.act(v);world.advance(action);}
  const double value=world.own_cash()-world.rival_cash();
  terminal_rollouts++;if(j==0)first=value;
  if(value>best+1e-6){best=value;chosen=j;winner=std::move(prepared);}
 }
 terminal_searches++;terminal_changes+=chosen!=0;terminal_last_gain=best-first;
 live=std::move(winner);restore_day=o.day+1;
}
