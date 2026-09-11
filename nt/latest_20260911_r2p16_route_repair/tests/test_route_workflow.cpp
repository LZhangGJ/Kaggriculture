#define main existing_checks
#include "test_route_economics.cpp"
#undef main
#ifndef R2_WORKFLOW_ENTRY
#define R2_WORKFLOW_ENTRY main
#endif
int R2_WORKFLOW_ENTRY(){
 using namespace triad::route_detail;
 existing_checks();int cases=0;
 // Rotate both geometry and crop kind. A three-action window fits exactly;
 // a two-action window must not admit an unfinished planting chain.
 for(int crop=0;crop<5;crop++)for(int pos:{34,43,45,54})for(int hour:{21,22}){
  State s;s.step=48+hour;s.priv.seeds[crop]=1;auto c=controller(s);
  c.core.target={{pos,crop}};
  c.logistics.required[dp7::Controller::task_key(pos,action(Op::PLANT,crop))]=1;
  c.logistics.required[dp7::Controller::task_key(pos,action(Op::WATER))]=1;
  c.route_restore(s.view());
  if(hour==21){auto p=verify(s.view(),c.core.plans);assert(p.valid&&covers(p.completed,debt(c)));}
  else assert(c.core.plans[0].a.empty());
  cases++;
 }
 for(int pos:{34,43,45,54}){
  State s;s.step=48+19;s.priv.shed[W]=1;auto&t=s.own.tiles[pos];
  t.kind=TileKind::ANIMAL;t.animal=Item::GOOSE;t.placed_day=0;t.yield_units=2;
  auto c=controller(s);c.core.target={{pos,G}};
  int at=44;dp7::Controller::walk(c.core.plans[0],at,pos);
  c.core.plans[0].a.push_back(action(Op::HARVEST));c.core.plans[0].target.push_back(pos);
  for(auto op:{Op::HARVEST,Op::FEED,Op::CARE})c.logistics.required[dp7::Controller::task_key(pos,action(op))]=1;
  c.route_restore(s.view());auto p=verify(s.view(),c.core.plans);
  assert(p.valid&&covers(p.completed,debt(c)));
  assert(p.harvested.at(dp7::Controller::task_key(pos,action(Op::HARVEST)))==2);
  assert(c.core.plans[0].a.front().op==Op::PICKUP);cases++;
 }
 for(int pos:{34,43,45,54})for(int milk_price:{1,100,300}){
  State s;s.step=48+21;s.priv.seeds[T]=1;s.market.prices[MI]=milk_price;
  auto c=controller(s);c.core.target={{pos,T}};
  auto plant=dp7::Controller::task_key(pos,action(Op::PLANT,T));
  auto water=dp7::Controller::task_key(pos,action(Op::WATER));
  c.logistics.required={{plant,1},{water,1}};
  Plan witness;int at=44;dp7::Controller::walk(witness,at,pos);
  witness.a.push_back(action(Op::PLANT,T));witness.target.push_back(pos);
  witness.a.push_back(action(Op::WATER));witness.target.push_back(pos);
  c.logistics.witness={witness};c.logistics.witness_step=s.step;
  c.route_restore(s.view());auto p=verify(s.view(),c.core.plans);
  assert(p.valid&&covers(p.completed,debt(c))&&c.logistics.witness_used==1);cases++;
 }
 {
  State s;s.priv.seeds[T]=1;Plan p{{action(Op::WATER),action(Op::PLANT,T)},{44,44},0};
  assert(!verify(s.view(),{p}).valid);cases++;
 }
 {
  State s;s.step=718;s.priv.seeds[T]=1;Plan p{{action(Op::PLANT,T),action(Op::WATER)},{44,44},0};
  assert(!verify(s.view(),{p}).valid);cases++;
 }
 std::cout<<"PASS "<<cases<<" generalized workflow cases\n";return 0;
}
