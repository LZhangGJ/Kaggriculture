#define R2_WORKFLOW_ENTRY workflow_checks
#include "test_route_workflow.cpp"
#undef R2_WORKFLOW_ENTRY
int main(){
 using namespace triad::recovery_detail;
 workflow_checks();int cases=0;
 {
  State s;auto c=controller(s);c.core.queue={action(Op::HIRE)};
  assert(!c.route_recover(s.view()));assert(c.core.queue.size()==1);cases++;
 }
 for(int crop=0;crop<5;crop++){
  State s;s.step=49;s.own.money=1000;s.own.tiles[43].kind=TileKind::EMPTY;
  auto c=controller(s);c.core.target={{43,crop}};c.logistics.recovery_pending=true;
  c.core.queue={action(Op::HIRE),action(Op::HIRE)};
  auto a=c.route_recover(s.view());assert(a&&a->market.size()==1&&a->market[0].op==Op::BUY_SEED);
  fastkag::ObservedDayScenario w(s.view());w.advance(*a);
  assert(!c.route_recover(w.view()));assert(c.logistics.recovery_installs==1);
  auto proof=verify(w.view(),c.core.plans);assert(proof.valid&&covers(proof.completed,debt(c)));cases++;
 }
 {
  State s;s.step=49;s.own.money=0;s.priv.shed[WO]=3;auto c=controller(s);
  c.core.target={{43,M}};c.logistics.recovery_pending=true;
  auto sale=c.route_recover(s.view());assert(sale&&!sale->market.empty());
  for(auto a:sale->market)assert(a.op==Op::SELL);
  fastkag::ObservedDayScenario w(s.view());w.advance(*sale);
  auto buy=c.route_recover(w.view());assert(buy&&buy->market[0].op==Op::BUY_SEED);cases++;
 }
 {
  State s;s.step=49;s.own.money=0;auto c=controller(s);c.core.target={{43,M}};
  c.logistics.recovery_pending=true;assert(!c.route_recover(s.view()));
  assert(c.logistics.required.size()==2&&std::all_of(c.logistics.completed.begin(),c.logistics.completed.end(),[](auto x){return x.second==0;}));
  assert(c.logistics.recovery_deferred==2&&c.core.target.empty());cases++;
 }
 {
  State s;s.step=49;s.own.money=0;s.bag(0,W,1);auto&t=s.own.tiles[43];
  t.kind=TileKind::ANIMAL;t.animal=Item::GOOSE;
  auto c=controller(s);c.core.target={{43,G}};c.logistics.recovery_pending=true;
  assert(deficits(s.view(),jobs(c,s.view())).empty());assert(!c.route_recover(s.view()));
  auto p=verify(s.view(),c.core.plans);assert(p.valid&&covers(p.completed,debt(c)));cases++;
 }
 {
  State s;s.step=70;s.own.money=3000;auto c=controller(s);c.core.target={{33,M}};
  c.logistics.recovery_pending=true;assert(!c.route_recover(s.view()));
  assert(c.core.plans[0].a.empty()&&c.logistics.recovery_deferred==2);cases++;
 }
 for(auto op:{Op::BUY_SEED,Op::BUY_ANIMAL,Op::HIRE,Op::BUY_LAND}){
  State s;s.own.money=0;auto c=controller(s);c.logistics.promised_day=s.view().day;
  PlayerAction order;order.units.resize(1);order.market={action(op,op==Op::BUY_SEED?M:op==Op::BUY_ANIMAL?G:-1)};
  c.route_finance(s.view(),order);s.step++;c.route_reconcile(s.view());
  assert(c.logistics.recovery_pending&&c.logistics.recovery_events==1);cases++;
 }
 {
  State s;s.step=49;s.own.money=1;auto c=controller(s);
  c.logistics.promised_day=s.view().day;c.logistics.expected_day=s.view().day;
  c.logistics.expected_step=s.step;c.logistics.expected_cash=100;
  c.core.queue={action(Op::BUY_SEED,M)};c.route_reconcile(s.view());
  assert(c.logistics.recovery_pending);cases++;
 }
 {
  State s;s.step=49;s.own.money=0;s.priv.shed[WO]=100;auto c=controller(s);
  c.core.target={{43,G}};c.logistics.recovery_pending=true;
  auto a=c.route_recover(s.view());assert(a);
  for(auto x:a->market)assert(x.op==Op::SELL);cases++;
 }
 {
  State s(2);s.step=69;s.own.hires_today=9;s.own.money=1;
  auto c=controller(s);c.core.target={{11,M}};
  auto q=choose(c,s.view());assert(!q.valid); // Cheap first-hire cost cannot be reused.
  cases++;
 }
 {
  State s;s.step=49;auto c=controller(s);c.logistics.recovery_pending=true;c.logistics.recovery_allowed=false;
  assert(!c.route_recover(s.view())&&c.logistics.recovery_checks==0);cases++;
 }
 {
  State s;s.step=(first[T]+3*interval[T]+1)*24+1;auto&t=s.own.tiles[43];
  t.kind=TileKind::PLANT;t.crop=Item::TOMATO;t.planted_day=0;t.yield_units=2;
  auto c=controller(s);c.core.target={{43,T}};
  auto original=jobs(c,s.view());bool renewal=false;
  for(auto&j:original)for(auto a:j.actions)renewal|=a.op==Op::DIG;assert(renewal);
  defer(c,s.view(),43);auto kept=jobs(c,s.view());assert(!kept.empty());
  for(auto&j:kept)for(auto a:j.actions)assert(a.op!=Op::DIG&&a.op!=Op::PLANT);
  assert(c.core.recovery_not_before[43]==s.view().day+1);cases++;
 }
 std::cout<<"PASS "<<cases<<" procurement recovery cases\n";
}
