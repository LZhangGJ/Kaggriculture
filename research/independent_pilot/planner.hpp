// Independent feedback executor. The 5x14 plan describes six-day resource targets:
// five crops, three animals, hired hands, land, reserve, sale delay, care, planting cutoff.
// No reference-policy actions, future RNG, hidden inventories or opponent identity.
using Plan = vector<vector<int>>;
void check_plan(const Plan& p) {
  if(p.size()!=5) throw std::runtime_error("plan needs five stages");
  for(auto& r:p) {
    if(r.size()!=14) throw std::runtime_error("plan needs fourteen fields per stage");
    int total=0;
    for(int i=0;i<8;i++) { if(r[i]<0 || r[i]>80) throw std::runtime_error("production target"); total+=r[i]; }
    if(total>100 || r[8]<0 || r[8]>16 || r[9]<1 || r[9]>4 || r[10]<0 || r[10]>3000 ||
       r[11]<0 || r[11]>23 || r[12]<0 || r[12]>1 || r[13]<0 || r[13]>29)
      throw std::runtime_error("plan bounds");
  }
}
Turn plan_action(const Farm& own,const array<int,9>& market,const array<Price,9>& params,
                 int step,const Plan& p) {
  const int day=step/24,hour=step%24;
  const auto& target=p[std::min(4,day/6)];
  State visible; visible.step=step; visible.market=market; visible.params=params; visible.farms[0]=own;
  Tasks tasks; Ledger l; l.full_actions=true; l.begin(visible,0,tasks);
  const int workers=own.positions.size();
  array<int,100> desired; desired.fill(-1);
  array<int,8> counts{};
  vector<int> empty;
  for(int k=0;k<100;k++) {
    const auto& t=own.tiles[k];
    if(t.item>=0) counts[t.item<5?t.item:t.item-4]++;
    else if(t.kind!=-1) empty.push_back(k);
  }
  std::stable_sort(empty.begin(),empty.end(),[](int a,int b){
    Pos pa{a%10,a/10},pb{b%10,b/10};return distance(pa,nearest(pa))<distance(pb,nearest(pb));
  });
  size_t slot=0;
  for(int it=0;it<8;it++) {
    int kind=it<5?it:it+4;
    if(day>target[13] || day+(it<5?firstday[it]:afirst[it-5])>=30) continue;
    for(int n=counts[it];n<target[it] && slot<empty.size();n++) desired[empty[slot++]]=kind;
  }
  array<bool,100> claimed{};
  for(int w=0;w<workers;w++) {
    Pos pos=l.f.positions[w]; const auto& inv=l.f.invs[w];
    struct Job {int verb=WAIT,item=-1,q=1,tile=-1;Pos target;double score=-1;};
    Job best;
    auto offer=[&](int v,int it,Pos dest,double value,int q=1,int tile=-1){
      int d=distance(pos,dest);
      // Overnight delivery allows productive work until the last turn of a day.
      if(d+1>24-hour && v!=DROP) return;
      double score=value/(1.+d);
      if(score>best.score) best=Job{v,it,q,tile,dest,score};
    };
    int produce=0; for(int it=0;it<9;it++) if(it!=0 && it!=8) produce+=inv.q[it];
    if(produce || (inv.sum()>0 && (hour>=21 || step>=715)))
      offer(DROP,-1,nearest(pos),step>=715?1e6:double(produce+1)*25);
    for(int k=0;k<100;k++) {
      const auto& t=l.f.tiles[k]; if(t.kind==-1 || claimed[k]) continue;
      Pos dest{k%10,k/10}; int it=t.item;
      if(t.yield>0 && (it>=9 || (t.kind==2 && day-t.day>=firstday[it])))
        offer(HARVEST,-1,dest,100.+t.yield*quote(it<5?it:it-4,market[it<5?it:it-4],params),1,k);
      if(t.kind==2 && !t.watered) offer(WATER,-1,dest,t.unwatered?600:170,1,k);
      if(t.kind==2 && inv.q[8]>0 && t.fertuntil<day && day<27)
        offer(FERTILIZE,-1,dest,90,1,k);
      if(it>=9) {
        if(!t.fed) {
          if(inv.q[0]>0) offer(FEED,-1,dest,t.unfed?800:250,1,k);
          else if(l.f.shed[0]>0) offer(PICKUP,0,nearest(pos),200,std::min(8,l.f.shed[0]));
        }
        if(target[12] && !t.cared && t.fed) offer(CARE,-1,dest,100,1,k);
        if(t.fertilizer) offer(FERT,-1,dest,60,1,k);
      }
      int want=desired[k]; if(want<0) continue;
      if(t.kind==1 || (t.item<0 && t.kind>0 && (want<5 || t.kind!=astructure[want-9])))
        offer(DIG,-1,dest,70,1,k);
      else if(t.kind==0 && want<5 && l.f.seeds[want]>0)
        offer(PLANT,want,dest,100,1,k);
      else if(want>=9) {
        if(t.kind==0) offer(astructure[want-9]==3?COOP:PASTURE,-1,dest,80,1,k);
        else if(t.item<0 && t.kind==astructure[want-9]) {
          if(inv.q[want]>0) offer(PLACE,want,dest,120,1,k);
          else if(l.f.shed[want]>0) offer(PICKUP,want,nearest(pos),130,1);
        }
      }
    }
    Action a;
    if(best.score>=0) {
      if(pos==best.target) a={best.verb,best.item,best.q};
      else {int dx=best.target.x-pos.x,dy=best.target.y-pos.y;
        a={dx>0?EAST:dx<0?WEST:dy>0?SOUTH:NORTH};}
      if(best.tile>=0) claimed[best.tile]=true;
    }
    // Exact menu repairs commands invalidated by another worker earlier this turn.
    vector<Choice> menu; l.menu(w,menu); Choice c;
    for(auto opt:menu) if(opt.verb==a.verb && opt.item==a.item) {c=opt;c.q=std::min(a.q,opt.q);break;}
    l.apply(c,w);
  }
  auto order=[&](int verb,int item=-1,int q=1){
    if(l.action.market.size()>=10) return;
    vector<Choice> menu; l.menu(-1,menu);
    for(auto c:menu) if(c.verb==verb && c.item==item) {c.q=std::min(q,c.q);if(c.q>0)l.apply(c,-1);return;}
  };
  // Conditional sales use observed quotes; expected sale proceeds never fund
  // a same-turn purchase. Revisit commitments after actual trade receipts arrive.
  for(int it=0;it<9;it++) {
    int keep=it==0 ? (counts[5]+counts[6]+counts[7])*2 : 0;
    int q=l.f.shed[it]-keep;
    if(q>0 && (hour>=target[11] || l.f.stock()>80 || step>=715)) order(SELL,it,q);
  }
  if(l.uncertain) return l.action;
  if(hour<12 && step<716) while(int(l.f.positions.size())-1<target[8] &&
      l.f.cash-hirecost(l.f.hires)>=target[10] && l.action.market.size()<10) order(HIRE);
  const int cost[3]={1000,2000,4000};
  if(day<22 && l.f.unlocked<target[9] && l.f.cash-cost[l.f.unlocked-1]>=target[10]) order(LAND);
  array<int,8> need{};
  for(int k=0;k<100;k++) if(desired[k]>=0) need[desired[k]<5?desired[k]:desired[k]-4]++;
  for(int it=0;it<8;it++) {
    int actual=it<5?it:it+4,held=it<5?l.f.seeds[it]:l.f.shed[actual];
    if(it>=5) for(auto& inv:l.f.invs) held+=inv.q[actual];
    int price=it<5?seedcost[it]:acost[it-5];
    int q=std::min(need[it]-held,int(std::max(0.,l.f.cash-target[10])/price));
    if(q>0) order(it<5?SEED:ANIMAL,actual,q);
  }
  int animals=counts[5]+counts[6]+counts[7];
  if(animals && l.f.shed[0]<2*animals && l.f.cash>target[10])
    order(PRODUCT,0,std::min(2*animals-l.f.shed[0],int((l.f.cash-target[10])/quote(0,market[0]-1,params))));
  return l.action;
}

py::dict act_plan(py::dict obs,const Plan& p) {
  check_plan(p); auto s=public_state(obs); int seat=getint(obs,"player");
  return turndict(plan_action(s.farms[seat],s.market,s.params,s.step,p));
}

py::array_t<double> plan_games(const Plan& p,const vector<uint64_t>& seeds,const vector<Plan>& rivals) {
  check_plan(p); for(auto& r:rivals)check_plan(r);
  py::array_t<double> result({int(seeds.size()*rivals.size()*2),3}); auto* out=result.mutable_data();
  py::gil_scoped_release release; int row=0;
  for(auto seed:seeds) for(auto& rival:rivals) for(int seat=0;seat<2;seat++) {
    State s(seed);
    while(!s.done()) {
      array<Turn,2> actions;
      for(int who=0;who<2;who++) actions[who]=plan_action(s.farms[who],s.market,s.params,s.step,who==seat?p:rival);
      s.advance(actions);
    }
    out[row*3]=s.farms[seat].cash;out[row*3+1]=s.farms[1-seat].cash;out[row*3+2]=s.step;row++;
  }
  return result;
}
