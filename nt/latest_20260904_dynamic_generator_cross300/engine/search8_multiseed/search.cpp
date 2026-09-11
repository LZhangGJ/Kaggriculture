// Isolated experiment. Reuse the complete frozen production bridge without
// editing/rebuilding its binary or registering two copies of the same C++ type.
#include <pybind11/pybind11.h>
#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable, ...) void register_production(pybind11::module_ &variable)
#include "../native/module.cpp"
#pragma pop_macro("PYBIND11_MODULE")
#include <sstream>

namespace search8 {
using namespace bridge;
struct Recipe {
 std::string family="KEEP";int kind=-1,amount=0;
 auto key()const{return std::tuple(family,kind,amount);}
 bool operator<(const Recipe&b)const{return key()<b.key();}
 bool operator==(const Recipe&b)const{return key()==b.key();}
};
Recipe recipe(const dp7branch::Candidate&c){return {c.family,c.kind,c.amount};}
py::dict pack_recipe(const Recipe&r){py::dict d;d["family"]=r.family;d["kind"]=r.kind;d["amount"]=r.amount;return d;}
Recipe parse_recipe(py::dict d){return {py::cast<std::string>(d["family"]),py::cast<int>(d["kind"]),py::cast<int>(d["amount"])};}
struct Pool {
 NativeTeammateExecutor executor;int size;
 Pool(py::dict d):executor(load(d)),size(py::len(d["routes"])){}
 static NativeTapeLibrary load(py::dict d){auto data=load_g001(d);data.library.raw_passthrough=py::cast<bool>(d["raw_passthrough"]);return std::move(data.library);}
};
struct World {
 Simulator env;dp7::Controller own;NativeAgentState rival;uint64_t seed;int seat;
 int missing=0,changed=0;
 World(uint64_t s,int p,dp7::Params pars):env(Config{},s),own(pars),seed(s),seat(p){}
};
struct Outcome {double cash=0,other=0;int missing=0,changed=0;uint64_t steps=0;};
struct Score {
 int wins=0;double margin=0,worst=INFINITY,cash=0;
 auto key()const{return std::tuple(wins,margin,worst,cash);}
 bool operator<(const Score&b)const{return key()<b.key();}
};
Score score(const std::vector<Outcome>&v){Score s;for(auto&r:v){s.wins+=r.cash>r.other;s.margin+=r.cash-r.other;s.worst=std::min(s.worst,r.cash-r.other);s.cash+=r.cash;}return s;}
std::array<PlayerAction,2> tick(World&w,const Pool&p,int route){
 std::array<PlayerAction,2>a;a[w.seat]=w.own.act(view(w.env,w.seat));a[1-w.seat]=p.executor.action_for(w.env,1-w.seat,route,w.rival);w.env.step(a);return a;
}
uint64_t advance(World&w,const Pool&p,int route,int day){uint64_t n=0;while(!w.env.done()&&w.env.day()<day){tick(w,p,route);n++;}return n;}
Outcome finish(World w,const Pool&p,int route){auto n=advance(w,p,route,30);return {w.env.farms()[w.seat].money,w.env.farms()[1-w.seat].money,w.missing,w.changed,n};}
using Menu=std::vector<dp7branch::Candidate>;
int locate(const Menu&v,const Recipe&r){for(int i=0;i<int(v.size());i++)if(recipe(v[i])==r)return i;return -1;}
void install(World&w,const Menu&menu,const Recipe&r){
 int index=locate(menu,r);if(index<0){index=0;w.missing++;}
 if(menu.empty()||menu[0].family!="KEEP")throw std::runtime_error("KEEP absent");
 w.changed+=index!=0;w.own=menu[index].controller;
}
py::list pack_outcomes(const std::vector<Outcome>&out,const std::vector<World>&worlds){py::list rows;
 for(size_t i=0;i<out.size();i++){auto&r=out[i];py::dict d;d["seed"]=worlds[i].seed;d["seat"]=worlds[i].seat;d["cash"]=r.cash;d["opponent_cash"]=r.other;d["win"]=r.cash>r.other;d["margin"]=r.cash-r.other;d["missing_recipe"]=r.missing;d["changed_nodes"]=r.changed;rows.append(d);}return rows;
}
struct Node {std::vector<Recipe>plan;std::vector<World>worlds;std::vector<Outcome>outcomes;Score value;};
struct Proposal {int parent;Recipe move;Node node;};
class Search {
 public:
 std::shared_ptr<Pool> pool;int route,width,threads,depth=0;dp7::Params pars;
 std::vector<int>days;std::vector<World>initial;std::vector<Node>beam;std::vector<Outcome>baseline;
 uint64_t total_steps=0,total_suffixes=0;
 Search(std::shared_ptr<Pool>p,int r,py::dict config,std::vector<uint64_t>seeds,std::vector<int>seats,std::vector<int>d,int b,int t):pool(p),route(r),width(b),threads(t),pars(params(py::dict(config.attr("copy")()))),days(d){
  if(r<0||r>=pool->size||seeds.size()!=seats.size()||seeds.empty()||t<1||t>16||b<1||days.empty()||days[0]!=0)throw std::invalid_argument("search shape");
  if(!std::is_sorted(days.begin(),days.end())||std::adjacent_find(days.begin(),days.end())!=days.end()||days.back()>28)throw std::invalid_argument("days");
  for(size_t i=0;i<seeds.size();i++){if(seats[i]<0||seats[i]>1)throw std::invalid_argument("seat");initial.emplace_back(seeds[i],seats[i],pars);}
  baseline.resize(initial.size());std::vector<std::string>errors(initial.size());
  {py::gil_scoped_release release;
   #pragma omp parallel for schedule(dynamic) num_threads(threads)
   for(int i=0;i<int(initial.size());i++)try{baseline[i]=finish(initial[i],*pool,route);}catch(const std::exception&e){errors[i]=e.what();}
  }
  for(auto&e:errors)if(!e.empty())throw std::runtime_error(e);
  beam.push_back(Node{{},initial,baseline,score(baseline)});
 }
 py::dict expand(){
  if(depth>=int(days.size()))throw std::runtime_error("search complete");
  auto start=std::chrono::steady_clock::now();int worlds=int(initial.size()),day=days[depth],next=depth+1<int(days.size())?days[depth+1]:30;
  std::vector<std::vector<Menu>>menus(beam.size(),std::vector<Menu>(worlds));
  std::vector<Proposal>proposals;std::vector<std::string>errors(beam.size()*worlds);uint64_t steps=0;
  {py::gil_scoped_release release;
   #pragma omp parallel for schedule(dynamic) num_threads(threads)
   for(int job=0;job<int(beam.size())*worlds;job++)try{
    int b=job/worlds,i=job%worlds;auto&w=beam[b].worlds[i];
    if(w.env.day()!=day||w.env.hour()!=0)throw std::runtime_error("wrong checkpoint");
    menus[b][i]=dp7branch::generate(w.own,view(w.env,w.seat));
   }catch(const std::exception&e){errors[job]=e.what();}
  }
  for(auto&e:errors)if(!e.empty())throw std::runtime_error(e);
  for(int b=0;b<int(beam.size());b++){
   std::set<Recipe>catalog;for(auto&menu:menus[b])for(auto&c:menu)catalog.insert(recipe(c));
   // KEEP first. The rest is canonical semantic order, never a world-specific index.
   std::vector<Recipe>order{Recipe{}};for(auto&r:catalog)if(r.family!="KEEP")order.push_back(r);
   for(auto&r:order){Node n=beam[b];n.plan.push_back(r);n.outcomes.resize(worlds);proposals.push_back({b,r,std::move(n)});}
  }
  errors.assign(proposals.size()*worlds,{});
  {py::gil_scoped_release release;
   #pragma omp parallel for schedule(dynamic) num_threads(threads) reduction(+:steps)
   for(int job=0;job<int(proposals.size())*worlds;job++)try{
    auto&p=proposals[job/worlds];int i=job%worlds;auto&w=p.node.worlds[i];
    install(w,menus[p.parent][i],p.move);
    auto used=advance(w,*pool,route,next);p.node.outcomes[i]=finish(w,*pool,route);steps+=used+p.node.outcomes[i].steps;
   }catch(const std::exception&e){errors[job]=e.what();}
  }
  for(auto&e:errors)if(!e.empty())throw std::runtime_error(e);
  for(auto&p:proposals)p.node.value=score(p.node.outcomes);
  std::stable_sort(proposals.begin(),proposals.end(),[](const auto&a,const auto&b){return b.node.value<a.node.value;});
  if(proposals.empty()||proposals.front().node.value<beam.front().value)throw std::runtime_error("lost incumbent KEEP continuation");
  py::list frontier;for(auto&p:proposals){py::dict row;py::list plan;for(auto&r:p.node.plan)plan.append(pack_recipe(r));row["plan"]=plan;row["wins"]=p.node.value.wins;row["mean_margin"]=p.node.value.margin/worlds;row["mean_cash"]=p.node.value.cash/worlds;frontier.append(row);}
  beam.clear();for(int b=0;b<std::min(width,int(proposals.size()));b++)beam.push_back(std::move(proposals[b].node));
  depth++;total_steps+=steps;total_suffixes+=proposals.size()*worlds;
  py::dict d=state();d["day"]=day;d["proposals"]=proposals.size();d["suffixes"]=proposals.size()*worlds;d["transitions"]=steps;d["frontier"]=frontier;d["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return d;
 }
 py::dict state(){py::dict d;d["depth"]=depth;d["total_transitions"]=total_steps;d["total_suffixes"]=total_suffixes;d["baseline"]=pack_outcomes(baseline,initial);py::list nodes;
  for(auto&n:beam){py::dict item;py::list plan;for(auto&r:n.plan)plan.append(pack_recipe(r));item["plan"]=plan;item["outcomes"]=pack_outcomes(n.outcomes,initial);nodes.append(item);}d["beam"]=nodes;return d;
 }
};
py::dict replay(std::shared_ptr<Pool>pool,int route,py::dict config,uint64_t seed,int seat,std::vector<int>days,py::list plan,bool trace){
 auto pars=params(py::dict(config.attr("copy")()));std::vector<Recipe>moves;for(auto d:plan)moves.push_back(parse_recipe(py::cast<py::dict>(d)));
 if(days.size()!=moves.size()||seat<0||seat>1||route<0||route>=pool->size)throw std::invalid_argument("replay shape");
 World w(seed,seat,pars);std::vector<std::array<PlayerAction,2>>frames;std::vector<std::array<double,2>>cash;std::vector<std::tuple<int,bool,std::vector<std::pair<int,int>>>>effects;
 uint64_t h=investment_branch_audit::initial;auto tic=std::chrono::steady_clock::now();
 {py::gil_scoped_release release;
  while(!w.env.done()){
   auto found=std::find(days.begin(),days.end(),w.env.day());
   if(w.env.hour()==0&&found!=days.end()){
    auto menu=dp7branch::generate(w.own,view(w.env,seat));const auto&r=moves[found-days.begin()];bool missing=locate(menu,r)<0;install(w,menu,r);effects.emplace_back(w.env.day(),missing,w.own.target);
   }
   auto a=tick(w,*pool,route);investment_branch_audit::mix(h,investment_branch_audit::frame_hash(a,w.env));
   if(trace){frames.push_back(a);cash.push_back({w.env.farms()[0].money,w.env.farms()[1].money});}
  }
 }
 Outcome result{w.env.farms()[seat].money,w.env.farms()[1-seat].money,w.missing,w.changed,719};py::dict d=pack_outcomes({result},{w})[0].cast<py::dict>();
 d["steps"]=w.env.step_count();d["state_action_hash"]=std::to_string(h);d["node_effects"]=effects;d["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-tic).count();
 if(trace){py::list out;for(auto&a:frames){py::list pair;pair.append(pack_player(a[0]));pair.append(pack_player(a[1]));out.append(pair);}d["trace"]=out;d["cash_by_step"]=cash;}return d;
}
}
PYBIND11_MODULE(_dp7_search8,m){
 register_production(m);
 py::class_<search8::Pool,std::shared_ptr<search8::Pool>>(m,"RoutePool").def(py::init<py::dict>());
 py::class_<search8::Search>(m,"Search8").def(py::init<std::shared_ptr<search8::Pool>,int,py::dict,std::vector<uint64_t>,std::vector<int>,std::vector<int>,int,int>()).def("expand",&search8::Search::expand).def("state",&search8::Search::state);
 m.def("replay8",&search8::replay,py::arg("pool"),py::arg("route"),py::arg("config"),py::arg("seed"),py::arg("seat"),py::arg("days"),py::arg("plan"),py::arg("trace")=false);
}
