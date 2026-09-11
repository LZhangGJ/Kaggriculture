// Read-only replay of the CURRENT greedy proposal calculation. This is not a
// candidate policy rollout and does not alter the live controller.
#include "policy.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
PYBIND11_MODULE(_dp7_investment_choice_probe,m){m.def("inspect",[](const dp7::Controller&source,const fastkag::Simulator&env,int seat){
 using namespace dp7;
 View actual{env.step_count(),env.day(),env.hour(),env.farms()[seat],env.farms()[1-seat],env.privates()[seat],env.market(),env.shops()};
 if(actual.hour!=0||source.day!=actual.day)throw std::invalid_argument("inspect AFTER current new-day decision, BEFORE environment step");
 auto farm=actual.own;int released=0;if(source.p.plan_zero_expiry)released=Controller::project_zero_expiry(farm,actual.step);
 View o{actual.step,actual.day,actual.hour,farm,actual.opponent,actual.priv,actual.market,actual.shops};
 auto c=source;c.admission_inspection=nullptr;c.schedule_cache.reset();
 // Trace choose_impl at the FINAL selected land capacity. Later rotation and
 // aggregate portfolio selection can differ: both actual targets and this
 // counterfactually held-capacity calculation are reported separately.
 const int planned=source.planned_land;auto count=c.counts(o);
 int free=0;for(int pos=0;pos<100;pos++)if(quad(pos)<planned&&!plant(farm.tiles[pos])&&!animal(farm.tiles[pos]))free++;
 const int animals=count[G]+count[CO]+count[SH];const double budget=c.investment_budget(o,planned,animals);
 Counts base=c.existing(o),selected{},newcount{},limits{75,75,c.p.max_tomato,c.p.max_strawberry,c.p.max_melon,0,0,0,0,c.p.max_geese,c.p.max_cows,c.p.max_sheep},floor{};
 floor[CO]=c.p.force_min_cows;floor[SH]=c.p.force_min_sheep;floor[S]=c.p.force_min_strawberry;floor[M]=c.p.force_min_melon;
 double spent=0;int feed=0;py::list rounds;std::vector<int>choices;
 bool prescribed=actual.day==0&&animals==0&&!c.p.autonomous_start;
 if(!prescribed)while(int(choices.size())<free){
  auto values=c.values(o,base,selected,feed);auto ranks=c.investment_ranks(o,values,selected,budget-spent);
  std::optional<std::pair<double,Project>>best;py::list options;
  for(auto[value,pr]:values){
   int k=pr.kind;double rank=ranks[k];if(count[k]+newcount[k]<floor[k]){double bonus=k>=9?3500:1800;value+=bonus;rank+=bonus;}
   std::string reject;
   if(count[k]+newcount[k]>=limits[k])reject="scale_limit";
   else if(spent+pr.capital>budget)reject="capital_budget";
   else if(k>=9&&(actual.day>c.p.latest_animal_day||animals+newcount[G]+newcount[CO]+newcount[SH]>=c.p.max_animals))reject="animal_limit_or_time";
   else if(value<=0)reject="nonpositive_value";
   if(reject.empty()&&(!best||std::tuple(rank,-pr.capital,name(k))>std::tuple(best->first,-best->second.capital,name(best->second.kind))))best={{rank,pr}};
   py::dict v;v["kind"]=k;v["value"]=value;v["rank"]=rank;v["reason"]=reject;v["capital"]=pr.capital;v["output"]=pr.out;v["actions"]=pr.actions;v["feed"]=pr.feed;v["fertilizer"]=pr.fert;v["seed_cost"]=pr.seed;options.append(v);
  }
  py::dict row;row["remaining_budget"]=budget-spent;row["options"]=options;row["chosen"]=best?best->second.kind:-1;rounds.append(row);
  if(!best)break;auto pr=best->second;choices.push_back(pr.kind);newcount[pr.kind]++;spent+=pr.capital;selected[pr.item]+=pr.out;selected[F]+=(!c.p.fix_values||pr.kind>=9?pr.fert:-pr.fert);feed+=pr.feed;
 }
 if(!prescribed){auto expected=c;expected.choose_impl(o,planned);Counts actual_counts{};for(auto[pos,k]:expected.target)if(k>=0&&!plant(farm.tiles[pos])&&!animal(farm.tiles[pos]))actual_counts[k]++;
  if(actual_counts!=newcount)throw std::runtime_error("greedy choice trace mismatch");}
 // act() has already removed this step's market packet from source.queue.
 // Reconstruct preparation for the selected targets from the SAME unchanged
 // physical observation, rather than accidentally omitting those purchases.
 c.prepare_orders(o,actual,released);
 auto preview=c.preview_bundle(o,false);Counts intended{},started{};
 for(auto[pos,k]:source.target)if(k>=0&&!plant(farm.tiles[pos])&&!animal(farm.tiles[pos]))intended[k]++;
 for(auto[pos,k]:preview.started_targets)started[k]++;
 auto prepared=c.project_preparation(o);py::dict out;
 out["day"]=o.day;out["cash"]=o.own.money;out["shed"]=o.priv.shed;out["seed_stock"]=o.priv.seeds;out["current_land"]=std::popcount(unsigned(o.own.unlocked_mask));out["planned_land"]=planned;out["live"]=count;out["free"]=free;out["budget"]=budget;
 out["prescribed_opening"]=prescribed;out["rounds"]=rounds;out["raw_proposals"]=newcount;out["intended"]=intended;out["projected_started"]=started;out["prepared_hands"]=prepared.farm.hands.size();out["preparation_ticks"]=prepared.elapsed;out["prepared_cash"]=prepared.farm.money;
 return out;
 });}
