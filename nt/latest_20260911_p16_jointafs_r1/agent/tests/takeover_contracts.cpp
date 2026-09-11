#include "policy/search.hpp"
#include "policy/observation_codec.hpp"
#include <fstream>
#include <cstring>
#include <iostream>
int checks=0;
void check(bool ok,const char*msg){++checks;if(!ok)throw std::runtime_error(msg);}
std::vector<double> load(const char*path){std::ifstream f(path);if(!f)throw std::runtime_error("missing fixture");std::vector<double>v;double x;while(f>>x)v.push_back(x);return v;}
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("fixture and settings required");auto input=load(argv[1]),cfg=load(argv[2]);
 check(cfg.size()==38&&triad::SETTINGS_COUNT==38,"ABI38");triad::Settings settings;std::memcpy(&settings,cfg.data(),sizeof(settings));
 Cursor r{input.data(),input.size()};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();auto a=r.farm(),b=r.farm();auto pr=r.priv();fastkag::Market market;for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();std::vector<int8_t>shops;int n=r.count();for(int i=0;i<n;i++)shops.push_back(r.integer());check(r.i==input.size(),"fixture parsed");
 auto own=seat?b:a,other=seat?a:b;dp7::View v{step,day,hour,own,other,pr,market,shops};
 check(step==24&&day==1&&own.money==26&&pr.shed[0]==3,"real unedited handoff");
 auto prior=triad::startup_supply_prior(v,settings);double prior_sum=0;for(const auto&d:prior)for(double x:d)prior_sum+=std::abs(x);check(prior_sum==0,"no startup prior on foreign step24");
 triad::Controller c(settings);c.plan(v);int feeds=0;for(const auto&j:c.core.jobs(v))for(auto x:j.actions)feeds+=x.op==fastkag::Op::FEED;check(feeds==4,"four feed obligations rebuilt from facts");
 auto ready=c.core.reserve(v,c.core.jobs(v));int accepted=0;for(const auto&j:ready)for(auto x:j.actions)accepted+=x.op==fastkag::Op::FEED;
 check(accepted==3,"real unfunded fourth job diagnostic");check(c.core.input_rejected_feed==1,"no silent dropped feed");check(c.core.input_rejected_atoms>=1,"count wholly discarded atomic job");
 check(c.core.feed_shortfall(v)==1,"shortage not erased by planning");
 c.core.phase=2;double cash=own.money;auto wheat=pr.shed[0];bool manure=own.tiles[44].fertilizer_available;
 auto first=c.working_capital_gate(v);check(bool(first),"gate materialises reachable collateral");check(first->units.size()==1&&first->units[0].op==fastkag::Op::COLLECT_FERTILIZER,"collect current depot manure");check(first->market.empty(),"no sale of unbanked manure");check(own.money==cash&&pr.shed[0]==wheat&&own.tiles[44].fertilizer_available==manure,"no optimistic mutation of truth");
 fastkag::ObservedDayScenario world(v);world.advance(*first);auto next=world.view();check(next.priv.inventories[0][8]==1,"actual collected manure in bag");
 auto second=c.act(next);bool buy=false,sell=false;for(auto x:second.market){buy|=x.op==fastkag::Op::BUY_PRODUCT&&int(x.item)==0&&x.quantity==1;sell|=x.op==fastkag::Op::SELL&&int(x.item)==8&&x.quantity==1;}
 check(buy&&sell,"sell collateral and fund missing feed");check(second.units[0].op==fastkag::Op::PLACE,"same-turn legal deposit precedes market");world.advance(second);auto funded=world.view();int total=funded.priv.shed[0];for(auto&iv:funded.priv.inventories)total+=iv[0];check(total==4,"actual fourth feed acquired");check(funded.own.money>=0,"no overdraft");
 // Revalue the exact same visible state without consuming a stale morning flow.
 triad::Controller residual(settings);residual.plan(v);competitive::Planner p1,p2;competitive::Flow px1,px2;
 auto f1=residual.remaining_portfolio(v,p1,px1);residual.portfolio.f[day][4]+=999;residual.portfolio.fixed[day]-=999999;auto f2=residual.remaining_portfolio(v,p2,px2);
 check(f1.f==f2.f&&f1.fixed==f2.fixed&&f1.labor==f2.labor,"morning phantom flows ignored");
 auto more=pr;more.shed[4]+=7;dp7::View vmore{step,day,hour,own,other,more,market,shops};auto f3=residual.remaining_portfolio(vmore,p2,px2);check(std::abs(f3.f[day][4]-f1.f[day][4]-7)<1e-9,"visible stock counted exactly once");
 competitive::Flow expensive{};for(auto&d:expensive){d.fill(1.);d[0]=10000.;}
 std::array<int8_t,2> adopted{{1,0}};auto cow=own.tiles[34];residual.model.day=day;auto stream=residual.animal_path(10,cow.placed_day,34,expensive,&cow,nullptr,&adopted);check(stream.f[day][0]==-1,"current adopted feed not repriced away");
 cow.fed_today=true;auto already=residual.animal_path(10,cow.placed_day,34,expensive,&cow,nullptr,&adopted);check(already.f[day][0]==0,"already fed not charged twice");
 fastkag::Tile plant;plant.kind=fastkag::TileKind::PLANT;plant.crop=fastkag::Item::WHEAT;plant.planted_day=0;plant.yield_units=1;plant.consecutive_unwatered=0;residual.model.day=3;competitive::Flow cropprices{};for(auto&d:cropprices){d.fill(100);d[8]=1;}
 auto nofert=residual.crop(0,0,50,4,cropprices,&plant,nullptr,1,0);auto usefert=residual.crop(0,0,50,4,cropprices,&plant,nullptr,1,1);check(nofert.f[3][8]==0&&usefert.f[3][8]==-1,"adopted fertilizer action preserved");
 // No critical need -> no rescue; no physical collateral -> no fabricated funds.
 auto enough=pr;enough.shed[0]=4;dp7::View ve{step,day,hour,own,other,enough,market,shops};triad::Controller normal(settings);normal.plan(ve);check(!normal.working_capital_gate(ve),"no gate when already funded");
 auto bare=own;bare.tiles[44].fertilizer_available=false;dp7::View vb{step,day,hour,bare,other,pr,market,shops};triad::Controller nofund(settings);nofund.plan(vb);check(!nofund.working_capital_gate(vb),"no imaginary collateral");
 // A resource promised to an existing crop is not sellable collateral.
 auto fertile=own;auto stock=pr;stock.inventories[0][8]=1;stock.inventory_order[0].push_back(8);
 int crop_pos=-1;for(int pos=0;pos<100;pos++)if(dp7::plant(fertile.tiles[pos])&&fertile.tiles[pos].crop==fastkag::Item::WHEAT){crop_pos=pos;break;}
 check(crop_pos>=0,"existing legal finite crop fixture");
 dp7::View vf{step,day,hour,fertile,other,stock,market,shops};triad::Controller protect(settings);protect.plan(vf);
 protect.core.crop_service_day=day;protect.core.crop_birth[crop_pos]=fertile.tiles[crop_pos].planted_day;protect.core.crop_kind[crop_pos]=0;protect.core.crop_fertilize[crop_pos]=1;protect.core.crop_water[crop_pos]=1;
 int promised=0;for(const auto&job:protect.core.jobs(vf))promised+=job.needs[8];check(promised>=1,"fixture has an actual fertilizer resource obligation");
 auto act=protect.working_capital_gate(vf);bool sold_reserved=false;if(act)for(auto x:act->market)sold_reserved|=x.op==fastkag::Op::SELL&&x.item==fastkag::Item::FERTILIZER;check(!sold_reserved,"fertilizer promise protected");

 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<"}\n";return 0;
 }catch(const std::exception&e){std::cerr<<"CONTRACT FAIL after "<<checks<<": "<<e.what()<<"\n";return 1;}}
