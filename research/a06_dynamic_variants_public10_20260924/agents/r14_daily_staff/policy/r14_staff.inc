// R14: daily headcount search over the already selected A06 portfolio.
// Only this player's observation, public production, and known rules are used.
// No labels, learned weights, opposing program, seed or replay is supplied.
int staff_calls14=0,staff_changes14=0,staff_removed14=0;
std::string staff_last14="null";
static void staff_install14(Controller&c,int h){
 c.core.p.max_hands=h;
 Acts q;for(auto a:c.core.queue)if(a.op!=Op::HIRE)q.push_back(a);
 for(int j=0;j<h;j++)q.push_back(dp7::action(Op::HIRE));
 c.core.queue=std::move(q);
}
// Liquidation of actual inventory plus continuation of existing physical
// projects. No unfunded hypothetical expansion is credited at this boundary.
double staff_tail14(const View&v,Controller&c){
 c.model.day=v.day;c.model.demand(v);c.model.public_rival(v);
 Asset a;Flow px{};
 for(int i=0;i<9;i++){a.f[v.day][i]+=v.priv.shed[i];for(auto&b:v.priv.inventories)a.f[v.day][i]+=b[i];}
 c.model.value(v,a,&px);
 for(int p=0;p<100;p++){
  const auto&t=v.own.tiles[p];
  if(animal(t))add(a,c.animal_path(int(t.animal),t.placed_day,p,px,&t));
  else if(plant(t)){
   int k=int(t.crop),len=ongoing(k)?first[k]+3*interval[k]:c.core.h_age(k);
   if(c.core.triad_crop_age[p]>=0)len=c.core.triad_crop_age[p];
   int end=std::clamp(int(t.planted_day)+len,v.day,29);
   add(a,c.crop(k,t.planted_day,p,end,px,&t));
  }
 }
 return c.model.value(v,a);
}
void daily_staff(const View&o){
 if(o.hour!=0||o.day<3||o.day>=29)return;
 if(live.core.day!=o.day)live.plan(o);
 int original=0;for(auto a:live.core.queue)original+=a.op==Op::HIRE;
 if(original<8)return;
 staff_calls14++;
 int selected=original;double best=-1e100,baseline_score=0;
 std::array<Farm,2>reference;
 std::ostringstream audit;audit.precision(17);audit<<"{\"day\":"<<o.day<<",\"original\":"<<original<<",\"candidates\":[";
 int depth=std::clamp(int(base.r14_staff_depth),1,3);
 int floor=std::max(6,original-int(base.r14_staff_range));
 for(int h=original;h>=floor;--h){
  double robust=1e100;bool safe=true;
  int worlds=base.r14_staff>=3?2:1;
  for(int z=0;z<worlds;z++){
   Controller c=live;staff_install14(c,h);c.previous_step=o.step-1;
   c.call_origin=Controller::CallOrigin::PublicPrediction;
   fastkag::PublicFlowScenario world(o,c.model.rival,z?base.supply*1.5:base.supply);
   int stop=std::min(719,o.step+depth*24);
   while(!world.done()&&world.view().step<stop){
    auto v=world.view();if(v.day!=o.day)c.configure(base);
    world.advance(c.act(v));
   }
   auto v=world.view();
   if(h==original)reference[z]=v.own;
   else if(base.r14_staff_guard>0){
    for(int p=0;p<100;p++){
     const auto&t=reference[z].tiles[p];const auto&u=v.own.tiles[p];
     // A real established asset cannot be sacrificed for apparent wage savings.
     if(animal(t)&&(!animal(u)||u.animal!=t.animal||u.placed_day!=t.placed_day))safe=false;
     if(plant(t)&&(!plant(u)||u.crop!=t.crop||u.planted_day!=t.planted_day))safe=false;
     if(animal(t)&&animal(u)&&u.consecutive_unfed>t.consecutive_unfed)safe=false;
     if(plant(t)&&plant(u)&&u.consecutive_unwatered>t.consecutive_unwatered)safe=false;
    }
   }
   double score=world.own_cash()-base.competition*world.rival_cash();
   if(!world.done())score+=staff_tail14(v,c);
   robust=std::min(robust,score);
  }
  if(h==original)baseline_score=robust;
  if(h!=original)audit<<",";
  audit<<"["<<h<<","<<robust<<","<<(safe?1:0)<<"]";
  if(safe&&robust>best+1e-6){best=robust;selected=h;}
 }
 staff_install14(live,selected);staff_changes14+=selected!=original;staff_removed14+=original-selected;
 audit<<"],\"selected\":"<<selected<<",\"conditional_gain\":"<<best-baseline_score<<"}";staff_last14=audit.str();
}
