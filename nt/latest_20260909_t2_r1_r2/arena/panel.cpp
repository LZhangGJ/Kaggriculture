// Isolated extension. Original simulator/native executor sources remain frozen.
#include "policy.hpp"
#include "cash_audit.hpp"
#include "production_audit.hpp"
#include "admission_audit.hpp"
#include "investment_candidates.hpp"
#include "resource_handoff_audit.hpp"
#include "recoordination_inspection.hpp"
#include "fieldbook_adapter.hpp"
#include "boatlee_v29.hpp"
#include "kaito_v58.hpp"
#include "lynn_v5.hpp"
#include "three_day_adapter.hpp"
#include "ecobot_v7.hpp"
// Same translation unit exposes the original, unchanged 147-feature builder.
#include "vendor/native_teammate.cpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>
#include <omp.h>
namespace py=pybind11;
using namespace fastkag;
namespace bridge {
constexpr const char* ops[]{"PASS","NORTH","SOUTH","EAST","WEST","DROP","PICKUP","PLACE","PLANT","WATER","HARVEST","FERTILIZE","DIG","BUILD_COOP","BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE","HIRE","BUY_LAND","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL"};
Action parse(py::handle h){Action a;auto s=py::reinterpret_borrow<py::sequence>(h);if(!s.size())return a;std::string op=py::cast<std::string>(s[0]);for(int i=0;i<24;i++)if(op==ops[i])a.op=Op(i);if(s.size()>1&&py::isinstance<py::str>(s[1])){std::string n=py::cast<std::string>(s[1]);for(int i=0;i<12;i++)if(n==item_name(i))a.item=Item(i);}if(s.size()>2&&py::isinstance<py::int_>(s[2]))a.quantity=py::cast<int>(s[2]);return a;}
PlayerAction parse_player(py::handle h){auto d=py::reinterpret_borrow<py::dict>(h);PlayerAction a;a.units.push_back(d.contains("farmer")?parse(d["farmer"]):Action{});if(d.contains("hands"))for(auto x:d["hands"])a.units.push_back(parse(x));if(d.contains("market"))for(auto x:d["market"])a.market.push_back(parse(x));return a;}
std::vector<PlayerAction>tape(py::handle h){std::vector<PlayerAction>x;for(auto a:py::reinterpret_borrow<py::sequence>(h))x.push_back(parse_player(a));return x;}
py::list pack(const Action&a){py::list x;x.append(ops[int(a.op)]);if(a.item!=Item::NONE){x.append(item_name(int(a.item)));if(a.op==Op::PLACE||a.op==Op::PICKUP||a.op==Op::BUY_SEED||a.op==Op::BUY_ANIMAL||a.op==Op::BUY_PRODUCT||a.op==Op::SELL)x.append(a.quantity);}else if(a.quantity!=1)x.append(a.quantity);return x;}
py::dict pack_player(const PlayerAction&a){py::dict d;d["farmer"]=pack(a.units.empty()?Action{}:a.units[0]);py::list hands,market;for(size_t u=1;u<a.units.size();u++)hands.append(pack(a.units[u]));for(auto x:a.market)market.append(pack(x));d["hands"]=hands;d["market"]=market;return d;}
py::object tile(const Tile&t){if(t.kind==TileKind::EMPTY)return py::none();if(t.kind==TileKind::LOCKED)return py::str("LOCKED");py::dict d;if(t.kind==TileKind::WEED){d["kind"]="WEED";return d;}if(t.kind==TileKind::COOP){d["kind"]="COOP";return d;}if(t.kind==TileKind::PASTURE){d["kind"]="PASTURE";return d;}if(t.kind==TileKind::PLANT){d["kind"]="PLANT";d["crop"]=item_name(int(t.crop));d["planted_day"]=t.planted_day;d["watered_today"]=t.watered_today;d["consecutive_unwatered"]=t.consecutive_unwatered;d["yield_units"]=t.yield_units;d["max_lifespan_step"]=t.max_lifespan_step;d["fertilized_until_day"]=t.fertilized_until_day;return d;}d["kind"]=t.animal==Item::GOOSE?"COOP":"PASTURE";d["animal"]=item_name(int(t.animal));d["placed_day"]=t.placed_day;d["yield_units"]=t.yield_units;d["consecutive_unfed"]=t.consecutive_unfed;d["fed_today"]=t.fed_today;d["cared_today"]=t.cared_today;d["fertilizer_available"]=t.fertilizer_available;d["pending_care_bonus"]=t.pending_care_bonus;return d;}
py::dict observation(const Simulator&s,int seat){py::dict o;o["player"]=seat;o["step"]=s.step_count();o["day"]=s.day();o["hour"]=s.hour();py::list farms;
 for(auto&f:s.farms()){py::dict d;d["money"]=f.money;d["farmer"]=py::make_tuple(f.farmer.x,f.farmer.y);py::list hands,tiles,quads;for(auto p:f.hands)hands.append(py::make_tuple(p.x,p.y));for(int y=0;y<10;y++){py::list row;for(int x=0;x<10;x++)row.append(tile(f.tiles[y*10+x]));tiles.append(row);}const char*q[]{"NW","NE","SW","SE"};for(int i=0;i<4;i++)if(f.unlocked_mask&(1<<i))quads.append(q[i]);d["hands"]=hands;d["tiles"]=tiles;d["unlocked_quadrants"]=quads;d["hires_today"]=f.hires_today;farms.append(d);}o["farms"]=farms;
 auto&p=s.privates()[seat];py::dict priv,shed,seeds;for(int i=0;i<12;i++)shed[item_name(i)]=p.shed[i];for(int i=0;i<5;i++)seeds[item_name(i)]=p.seeds[i];py::list inv;for(size_t u=0;u<p.inventories.size();u++){py::dict d;for(int i:p.inventory_order[u])if(p.inventories[u][i])d[item_name(i)]=p.inventories[u][i];inv.append(d);}priv["shed"]=shed;priv["seeds"]=seeds;priv["inventories"]=inv;o["private"]=priv;py::dict market,mi,mp,town;for(int i=0;i<9;i++){mi[item_name(i)]=s.market().inventory[i];mp[item_name(i)]=s.market().prices[i];}market["inventory"]=mi;market["prices"]=mp;o["market"]=market;py::list shops;for(int sh:s.shops())shops.append(shop_name(sh));town["unlocked_shops"]=shops;o["town"]=town;return o;
}
dp7::View view(const Simulator&s,int p){return {s.step_count(),s.day(),s.hour(),s.farms()[p],s.farms()[1-p],s.privates()[p],s.market(),s.shops()};}
dp7::Params params(py::dict d){dp7::Params p;
#define FIELD(n) if(d.contains(#n)){p.n=py::cast<decltype(p.n)>(d[#n]);d.attr("pop")(#n);}
 FIELD(exact_schedule_cache) FIELD(incremental_regret_cost)
 FIELD(continuous_market_execution) FIELD(insert_missing_feed)
 FIELD(fix_resources) FIELD(fix_expiry) FIELD(fix_values) FIELD(fix_calendar) FIELD(fix_liquidity) FIELD(fix_logistics)
 FIELD(max_land) FIELD(max_animals) FIELD(max_cows) FIELD(max_sheep) FIELD(max_geese) FIELD(max_strawberry) FIELD(max_tomato) FIELD(max_melon)
 FIELD(compile_consequence) FIELD(compile_replant_choices) FIELD(compile_bounded_rollout)
 FIELD(capacity_hauling)
 FIELD(insertion_hire_estimate)
 FIELD(reconcile_seed_drift)
 FIELD(plan_zero_expiry)
 FIELD(cashflow_value_mode)
 FIELD(funded_bundle_mode)
 FIELD(preparation_work)
 FIELD(split_service_jobs) FIELD(regret_schedule) FIELD(regret_compile) FIELD(regret_hire_estimate)
 FIELD(terminal_deposit_schedule)
 FIELD(overflow_idle_dispatch)
 FIELD(capital_time_rank)
 FIELD(shared_task_atoms_v2)
 FIELD(stepwise_recoordination)
 FIELD(preparation_pipeline_v2)
 FIELD(schedule_value_compare)
 FIELD(preparation_spawn_guard) FIELD(net_feed_buffer)
 FIELD(midroute_delivery)
 FIELD(resource_aware_exchange)
 FIELD(intraday_admission) FIELD(intraday_procurement) FIELD(intraday_declared_value)
 FIELD(autonomous_start)
 FIELD(joint_investment_portfolio)
 FIELD(portfolio_crop_calendar)
 FIELD(recover_service_inputs) FIELD(procure_service_inputs) FIELD(finance_service_inputs)
 FIELD(incremental_pickup_repair)
 FIELD(shared_service_insertions)
 FIELD(intraday_future_workforce)
 FIELD(day_consequence_compare) FIELD(day_value_public_supply)
 FIELD(day_value_replan_next_day)
 FIELD(idle_task_handoff)
 FIELD(opening_melon) FIELD(opening_strawberry) FIELD(latest_animal_day) FIELD(timing_discount) FIELD(action_shadow) FIELD(feed_price_mult) FIELD(capital_fraction) FIELD(future_shop_weight)
 FIELD(opponent_supply_weight) FIELD(own_feed_demand_weight) FIELD(temporal_value_weight)
 FIELD(committed_feed_weight)
 FIELD(existing_replant_weight)
 FIELD(hold_opponent_supply_weight)
 FIELD(competition_objective_weight)
 FIELD(competitive_sell_weight) FIELD(sell_horizon_days) FIELD(sell_advantage)
 FIELD(opening_crops) FIELD(opening_crop_order)
 FIELD(rotate_finite) FIELD(rotation_margin)
 FIELD(portfolio_rotation) FIELD(rotation_timing)
 FIELD(economic_land) FIELD(operating_reserve) FIELD(land_margin) FIELD(crop_harvest_age) FIELD(finite_fertilizer) FIELD(fix_finite_projection)
 FIELD(early_deposit) FIELD(sell_deposits)
 FIELD(efficient_water) FIELD(efficient_care)
 FIELD(renew_ongoing)
 FIELD(hold_capacity) FIELD(shed_safety) FIELD(max_hands) FIELD(feed_cover_days) FIELD(feed_stock_cap) FIELD(force_min_cows) FIELD(force_min_sheep) FIELD(force_min_strawberry) FIELD(force_min_melon)
 FIELD(feed_forecast_days)
#undef FIELD
 if(d.contains("project_bias")){auto b=py::cast<py::dict>(d["project_bias"]);for(int i=0;i<12;i++)if(b.contains(item_name(i)))p.bias[i]=py::cast<double>(b[item_name(i)]);d.attr("pop")("project_bias");}
 if(d.contains("opening_animals")){p.opening_animals.clear();for(auto s:d["opening_animals"]){auto n=py::cast<std::string>(s);int i=9;for(;i<12;i++)if(n==item_name(i))break;if(i==12)throw std::invalid_argument("opening animal");p.opening_animals.push_back(i);}d.attr("pop")("opening_animals");}
 if(d.size())throw std::invalid_argument("Unknown native policy parameter");if(p.max_hands<0||p.max_hands>15||p.max_land<1||p.max_land>4)throw std::invalid_argument("parameter bounds");
 if(p.procure_service_inputs&&!p.recover_service_inputs)throw std::invalid_argument("service procurement requires recovery");
 if(p.finance_service_inputs&&!p.procure_service_inputs)throw std::invalid_argument("service financing requires procurement");
 if(p.rotation_timing&&!p.portfolio_rotation)throw std::invalid_argument("rotation timing requires portfolio rotation");
 auto crop_order=p.opening_crop_order;std::sort(crop_order.begin(),crop_order.end());if(crop_order!=std::array<int,5>{0,1,2,3,4})throw std::invalid_argument("opening_crop_order must be a permutation of 0..4");
 if(!std::isfinite(p.hold_opponent_supply_weight)||p.hold_opponent_supply_weight<0||p.hold_opponent_supply_weight>2)throw std::invalid_argument("hold supply weight must be in [0,2]");
 if(p.feed_forecast_days<0||p.feed_forecast_days>29)throw std::invalid_argument("feed forecast days");
 if(!std::isfinite(p.committed_feed_weight)||p.committed_feed_weight<0||p.committed_feed_weight>1)throw std::invalid_argument("committed feed weight in [0,1]");
 if(!std::isfinite(p.existing_replant_weight)||p.existing_replant_weight<0||p.existing_replant_weight>1)throw std::invalid_argument("existing replant weight in [0,1]");
 if(p.cashflow_value_mode<0||p.cashflow_value_mode>3)throw std::invalid_argument("cashflow_value_mode in 0..3");
 if(p.funded_bundle_mode<0||p.funded_bundle_mode>2)throw std::invalid_argument("funded_bundle_mode in 0..2");return p;
}
struct Tree{int step;std::vector<int>left,right,feature,targets;std::vector<float>threshold;std::vector<std::vector<float>>value;int predict(const std::array<float,147>&x)const{int node=0;while(left.at(node)>=0)node=x.at(feature.at(node))<=threshold.at(node)?left.at(node):right.at(node);auto&v=value.at(node);return targets.at(std::max_element(v.begin(),v.end())-v.begin());}};
struct G001Data{NativeTapeLibrary library;int opening;std::vector<Tree>trees;};
G001Data load_g001(py::dict d){G001Data x;x.opening=py::cast<int>(d["opening"]);for(auto r:d["routes"])x.library.routes.push_back(tape(r));x.library.r5_reference=tape(d["r5"]);x.library.md_reference=tape(d["md"]);for(int i=0;i<5;i++){x.library.moon[i]=tape(d["moon"][py::int_(i)]);x.library.moon_legacy[i]=tape(d["moon_legacy"][py::int_(i)]);}for(auto n:d["nodes"]){auto r=py::cast<py::dict>(n);Tree t;t.step=py::cast<int>(r["checkpoint"]);t.left=py::cast<std::vector<int>>(r["left"]);t.right=py::cast<std::vector<int>>(r["right"]);t.feature=py::cast<std::vector<int>>(r["feature"]);t.targets=py::cast<std::vector<int>>(r["targets"]);t.threshold=py::cast<std::vector<float>>(r["threshold"]);t.value=py::cast<std::vector<std::vector<float>>>(r["value"]);x.trees.push_back(std::move(t));}return x;}
struct G001State{NativeAgentState state;fastkag::FeatureHistory history;int current=-1,last=-1;bool switched=false;};
class G001{public:G001Data data;NativeTeammateExecutor executor;explicit G001(G001Data d):data(std::move(d)),executor(data.library){}PlayerAction act(const Simulator&s,int p,G001State&st)const{if(s.step_count()==0||s.step_count()<st.last)st=G001State{};if(st.current<0)st.current=data.opening;st.history.update(s);for(auto&t:data.trees)if(!st.switched&&t.step==s.step_count()){auto features=fastkag::build_features(s,p,st.history,data.library.routes[data.opening]);int next=t.predict(features);if(next!=st.current){st.current=next;st.switched=true;}}st.last=s.step_count();return executor.action_for(s,p,st.current,st.state);}};
struct Result{uint64_t seed;int seat,steps;double cash,other,seconds;bool switched;std::array<int,2>overflow{};};
Result run(uint64_t seed,int seat,dp7::Params pars,const G001*g){auto start=std::chrono::steady_clock::now();Simulator env(Config{},seed);dp7::Controller c(pars);G001State st;std::array<int,2>overflow{};while(!env.done()){std::array<PlayerAction,2>a;a[seat]=c.act(view(env,seat));if(g)a[1-seat]=g->act(env,1-seat,st);env.step(a);for(int p=0;p<2;p++)overflow[p]+=env.last_end_of_day_overflow()[p];}return {seed,seat,env.step_count(),env.farms()[seat].money,env.farms()[1-seat].money,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),st.switched,overflow};}
struct DayAudit {int shared_insertion_checks=0,shared_insertion_applied=0,shared_insertion_peak_saved=0,shared_insertion_steps_saved=0;int service_checks=0,service_recovered_feed=0,service_recovered_fertilize=0,service_buys=0,service_financed=0,service_deposits=0;int portfolio_generated=0,portfolio_evaluated=0,portfolio_switches=0,portfolio_rejected=0;int day=0,compile_drop=0,degraded=0,max_hands=0,unfed=0,unwatered=0,overflow=0;double start_money=0,end_money=0,opponent_money=0;dp7::Counts target{},live{},opponent_live{},bought{},sold{},shed{},carried{};int hired=0,land=0;int bundle_evaluations=0,bundle_switches=0,bundle_removed_targets=0,preparation_actions=0,shared_service_plots=0,regret_trials=0,regret_improvements=0;int terminal_schedule_evaluations=0,terminal_schedule_switches=0;double terminal_expected_cash_gain=0.;int overflow_dispatch_checks=0,overflow_dispatch_units=0,overflow_dispatch_quantity=0;int resource_exchange_checks=0,resource_exchange_pairs=0,resource_exchange_applied=0,resource_exchange_local_steps_saved=0;int intraday_checks=0,intraday_proposals=0,intraday_activated=0,intraday_cancelled=0,intraday_unfilled=0,intraday_purchase_orders=0;dp7::Counts intraday_activated_by_kind{};};
struct FieldbookResult {Result match;std::array<int,5>segments;};
boatlee29::Data load_boatlee(py::dict d){
 if(py::cast<std::string>(d["source_sha256"])!="c4a6964cec3c1c99207c32bb1fd91e53c3ec01e6890da5734331cbeab1cc1267")throw std::invalid_argument("Boatlee version");
 if(py::len(d["_FR_ITEMS"]))throw std::invalid_argument("Different enabled Boatlee front-run branch");
 boatlee29::Data x;x.actions=tape(d["actions"]);if(x.actions.size()!=720)throw std::invalid_argument("Boatlee action count");
 x.config=py::cast<std::map<std::string,double>>(d["_AM_CONFIG"]);x.weed_replay_steps=py::cast<int>(d["_WEED_REPLAY_STEPS"]);
 auto item=[](py::handle h){auto s=py::cast<std::string>(h);for(int i=0;i<12;i++)if(s==item_name(i))return i;throw std::invalid_argument("Boatlee item");};
 for(auto name:d["_AM_ITEMS"])x.items.push_back(item(name));
 for(auto kv:py::reinterpret_borrow<py::dict>(d["_AM_BASE_PRICE"]))x.base_price[item(kv.first)]=py::cast<double>(kv.second);
 const char*shops[]{"BAKERY","BRUNCH_SPOT","FARMERS_MARKET","ICE_CREAM_SHOP","PET_CAFE","PIZZA_SHOP","SMOOTHIE_SHOP","YARN_STORE"};
 auto sd=py::reinterpret_borrow<py::dict>(d["_SHOP_PRODUCTS"]);
 for(int s=0;s<8;s++){auto v=sd[shops[s]];int n=py::len(v)==1?2:1;for(auto name:v)x.shop_demand[s][item(name)]=n;}
 return x;
}
py::dict boatlee_debug(const boatlee29::State&s){py::dict d,pressure,added,lastinv,lastsold,active;
 for(int i:{dp7::S,dp7::MI,dp7::WO}){pressure[item_name(i)]=s.pressure[i];added[item_name(i)]=s.added[i];lastinv[item_name(i)]=s.last_inventory[i];lastsold[item_name(i)]=s.last_sold[i];}
 for(auto [u,r]:s.active){py::dict v;v["start"]=r.start;v["intended"]=pack(r.intended);active[py::int_(u)]=v;}
 d["last_step"]=s.am_last;d["weed_last"]=s.weed_last;d["near_mirror"]=s.near_mirror;d["pressure"]=pressure;d["added"]=added;d["last_inventory"]=lastinv;d["last_sold"]=lastsold;d["active"]=active;d["last_shops"]=s.last_shops;return d;
}
kaito58::Data load_kaito(py::dict d){
 if(py::cast<std::string>(d["source_sha256"])!="b041058ec187a8d0a01edc0eab8de068b53deca3e6c1973faf74ace6916ddcb9")throw std::invalid_argument("Kaito version");
 auto ktape=[](py::handle h){auto result=tape(h);size_t step=0;auto rows=py::reinterpret_borrow<py::sequence>(h);auto integer=py::module_::import("builtins").attr("int");
   auto quantity=[&](Action& a,py::handle raw){auto seq=py::reinterpret_borrow<py::sequence>(raw);if(seq.size()>2&&(a.op==Op::PICKUP||a.op==Op::PLACE||a.op==Op::BUY_SEED||a.op==Op::BUY_PRODUCT||a.op==Op::BUY_ANIMAL||a.op==Op::SELL))a.quantity=py::cast<int>(integer(seq[2]));};
   for(auto raw:rows){auto row=py::reinterpret_borrow<py::dict>(raw);auto&a=result[step++];if(row.contains("farmer"))quantity(a.units[0],row["farmer"]);size_t u=1;if(row.contains("hands"))for(auto v:row["hands"])quantity(a.units[u++],v);size_t m=0;if(row.contains("market"))for(auto v:row["market"])quantity(a.market[m++],v);}return result;};
 kaito58::Data x;int n=0;for(auto h:d["routes"]){if(n>=10)throw std::invalid_argument("Kaito route count");auto r=py::cast<py::dict>(h);x.names[n]=py::cast<std::string>(r["name"]);x.routes[n]=ktape(r["actions"]);if(x.routes[n].size()!=719)throw std::invalid_argument("Kaito incomplete route");n++;}
 const std::array<std::string,10>names{"base_backbone","base_yarn","base_pet","recovery","smoothie","clone","known_yarn","ice_minimax","bakery_yarn","pizza_recovery"};
 if(n!=10||x.names!=names)throw std::invalid_argument("Kaito route order");
 auto cfg=py::cast<py::dict>(d["config"]);for(auto kv:cfg)if(py::isinstance<py::bool_>(kv.second)||py::isinstance<py::int_>(kv.second)||py::isinstance<py::float_>(kv.second))x.config[py::cast<std::string>(kv.first)]=py::cast<double>(kv.second);
 if(x.c("defer_enabled")||x.c("market_maker_enabled")||!x.c("preempt_enabled")||py::cast<std::string>(cfg["terminal_rule"])!="collision")throw std::invalid_argument("Kaito unsupported configuration");
 const std::vector<std::string>items{"CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL"};
 if(py::cast<std::vector<std::string>>(cfg["front_items"])!=items||py::cast<std::vector<std::string>>(cfg["preempt_items"])!=items)throw std::invalid_argument("Kaito item configuration");
 if(py::cast<std::vector<std::string>>(d["pet_second_shops"])!=std::vector<std::string>{"PET_CAFE","YARN_STORE"})throw std::invalid_argument("Kaito pet router");return x;
}
py::dict kaito_debug(const kaito58::State&s){
 py::dict d;d["last_step"]=s.last;d["mode"]=s.mode.empty()?py::none():py::object(py::str(s.mode));d["mirror_streak"]=s.mirror_streak;d["clone_selected"]=s.clone;d["known_yarn"]=s.known_yarn;d["selected"]=s.selected;
 py::list policies;for(auto&r:s.policies){py::dict row,iv,net,supply,due,active;
   for(int i=0;i<9;i++){iv[item_name(i)]=r.inventory[i];net[item_name(i)]=r.net[i];supply[item_name(i)]=r.supply[i];}
   for(auto [t,items]:r.due){py::dict v;for(int i=0;i<9;i++)if(items[i])v[item_name(i)]=items[i];due[py::int_(t)]=v;}
   for(auto [u,rec]:r.weed){py::dict v;v["start"]=rec.start;v["intended"]=pack(rec.intended);active[py::int_(u)]=v;}
   row["last_step"]=r.last;row["near_streak"]=r.near_streak;row["near_latched"]=r.near;row["last_inventory"]=iv;row["last_market_net"]=net;row["opponent_supply"]=supply;row["last_shops"]=r.shops;row["mirror_confidence"]=r.confidence;row["mirror_evidence_turns"]=r.evidence;row["due"]=due;row["active"]=active;row["emitted"]=pack_player(r.emitted);row["collisions"]=r.collisions;row["preempt_turns"]=r.preempt_turns;row["preempt_units"]=r.preempt_units;row["repaid_units"]=r.repaid;policies.append(row);
 }d["policies"]=policies;return d;
}
lynn5::Data load_lynn(py::dict d){
 if(py::cast<std::string>(d["source_sha256"])!="e8498c67914ecc607ae69fde25a728361eb5acea94c00ecc85deffbdafe50413")throw std::invalid_argument("Lynn version");
 lynn5::Data x;x.actions=tape(d["actions"]);if(x.actions.size()!=719&&x.actions.size()!=720)throw std::invalid_argument("Lynn incomplete tape");return x;
}
py::dict lynn_debug(const lynn5::State&s){
 const char* names[]{"cow88","cow150","cow169","cow176","sheep313"};
 auto animal=[](int i)->py::object{return i<0?py::object(py::none()):py::object(py::str(item_name(i)));};
 auto shops=[](const std::vector<int8_t>& values){py::list x;for(int i:values)x.append(shop_name(i));return x;};
 py::dict d,assignments,weed;for(int b=0;b<5;b++)if(s.assignments[b]>=0)assignments[names[b]]=animal(s.assignments[b]);
 for(auto [u,r]:s.weed){py::dict x;x["start"]=r.start;x["intended"]=pack(r.intended);weed[py::int_(u)]=x;}
 d["last_step"]=s.last;d["base_last"]=s.base_last;d["assignments"]=assignments;d["weed_transactions"]=weed;
 d["cow_to_sheep"]=s.cow_to_sheep;d["sheep_to_cow"]=s.sheep_to_cow;d["cow_window_target"]=animal(s.window);d["cow_window_shops"]=shops(s.window_shops);
 d["deferred_cow_signal"]=py::none();if(s.deferred_bundle>=0){py::dict r;r["bundle"]=names[s.deferred_bundle];r["cow_signal"]=s.deferred_signal;r["shops"]=shops(s.deferred_shops);d["deferred_cow_signal"]=r;}
 py::list decisions;for(auto&r:s.decisions){py::dict x,p;
   p["wool"]=r.wool;p["dairy"]=r.dairy;p["gap"]=r.wool-r.dairy;p["milk_price"]=r.milk_price;p["wool_price"]=r.wool_price;
   x["bundle"]=names[r.bundle];x["source"]=r.bundle==4?"SHEEP":"COW";x["target"]=animal(r.target);x["quantity"]=r.bundle==1?2:1;x["pressure"]=p;x["opponent_supply_balance"]=r.balance;x["cow_signal"]=r.signal;x["unlocked_shops"]=shops(r.shops);x["evidence_mature"]=r.mature;x["deferred"]=r.deferred;x["window_target"]=animal(r.window);x["window_reused"]=r.reused;decisions.append(x);
 }d["decision_rows"]=decisions;py::list delivery,stages;
 for(int i=0;i<3;i++){auto&r=s.delivery[i];py::dict x;x["last_step"]=r.last;x["active"]=r.active;x["cancelled"]=r.cancelled;x["cancel_reason"]=r.reason.empty()?py::object(py::none()):py::object(py::str(r.reason));x["intervention_steps"]=r.interventions;if(i<2)x["animal"]=animal(r.animal);delivery.append(x);}
 for(auto&a:s.stages)stages.append(pack_player(a));d["delivery"]=delivery;d["stages"]=stages;return d;
}
struct LynnResult{Result match;lynn5::State state;int weed_frames=0;};
LynnResult run_lynn(uint64_t seed,int seat,dp7::Params pars,const lynn5::Agent&opponent){
 auto start=std::chrono::steady_clock::now();Simulator env(Config{},seed);dp7::Controller c(pars);lynn5::State st;std::array<int,2>overflow{};int weeds=0;
 while(!env.done()){std::array<PlayerAction,2>a;a[seat]=c.act(view(env,seat));a[1-seat]=opponent.act(view(env,1-seat),1-seat,st);weeds+=!st.weed.empty();env.step(a);for(int p=0;p<2;p++)overflow[p]+=env.last_end_of_day_overflow()[p];}
 return {{seed,seat,env.step_count(),env.farms()[seat].money,env.farms()[1-seat].money,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),false,overflow},st,weeds};
}
struct KaitoResult{Result match;std::array<int,10>selected{};std::array<int,10>preempt{},weed{};};
KaitoResult run_kaito(uint64_t seed,int seat,dp7::Params pars,const kaito58::Agent&opponent){
 auto start=std::chrono::steady_clock::now();Simulator env(Config{},seed);dp7::Controller c(pars);kaito58::State st;KaitoResult result;std::array<int,2>overflow{};
 while(!env.done()){std::array<PlayerAction,2>a;a[seat]=c.act(view(env,seat));a[1-seat]=opponent.act(view(env,1-seat),1-seat,st);result.selected[st.selected]++;env.step(a);for(int p=0;p<2;p++)overflow[p]+=env.last_end_of_day_overflow()[p];}
 result.match={seed,seat,env.step_count(),env.farms()[seat].money,env.farms()[1-seat].money,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),false,overflow};
 for(int i=0;i<10;i++){result.preempt[i]=st.policies[i].preempt_units;result.weed[i]=st.policies[i].collisions;}return result;
}
struct BoatleeResult{Result match;boatlee29::State state;int weed_repair_frames=0;};
BoatleeResult run_boatlee(uint64_t seed,int seat,dp7::Params pars,const boatlee29::Agent&opponent){
 auto start=std::chrono::steady_clock::now();Simulator env(Config{},seed);dp7::Controller c(pars);boatlee29::State st;std::array<int,2>overflow{};int weedframes=0;
 while(!env.done()){std::array<PlayerAction,2>a;a[seat]=c.act(view(env,seat));a[1-seat]=opponent.act(view(env,1-seat),1-seat,st);weedframes+=!st.active.empty();env.step(a);for(int p=0;p<2;p++)overflow[p]+=env.last_end_of_day_overflow()[p];}
 return {{seed,seat,env.step_count(),env.farms()[seat].money,env.farms()[1-seat].money,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),false,overflow},st,weedframes};
}
FieldbookResult run_fieldbook(uint64_t seed,int seat,dp7::Params pars){
 auto start=std::chrono::steady_clock::now();Simulator env(Config{},seed);dp7::Controller c(pars);fieldbook::Controller rival;std::array<int,2>overflow{};
 while(!env.done()) {std::array<PlayerAction,2>a;a[seat]=c.act(view(env,seat));a[1-seat]=rival.act(view(env,1-seat),1-seat);env.step(a);for(int p=0;p<2;p++)overflow[p]+=env.last_end_of_day_overflow()[p];}
 return {{seed,seat,env.step_count(),env.farms()[seat].money,env.farms()[1-seat].money,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),false,overflow},rival.segments()};
}
struct ThreeDayResult{Result match;int route=-1;};
ThreeDayResult run_threeday(uint64_t seed,int seat,dp7::Params pars){
 auto start=std::chrono::steady_clock::now();Simulator env(Config{},seed);dp7::Controller c(pars);threeday::Controller rival;std::array<int,2>overflow{};
 while(!env.done()){std::array<PlayerAction,2>a;a[seat]=c.act(view(env,seat));a[1-seat]=rival.act(view(env,1-seat),1-seat);env.step(a);for(int p=0;p<2;p++)overflow[p]+=env.last_end_of_day_overflow()[p];}
 return {{seed,seat,env.step_count(),env.farms()[seat].money,env.farms()[1-seat].money,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),false,overflow},rival.route()};
}
using CashSeason=std::array<std::array<dp7audit::CashDay,2>,30>;
struct EcoBotNative{
 eco7::Controller controller;
 PlayerAction act(const Simulator&s,int p){return controller.act({s.step_count(),s.day(),s.hour(),s.farms()[p],s.privates()[p],s.market(),s.shops()});}
};
Result run_ecobot(uint64_t seed,int seat,dp7::Params pars){
 auto start=std::chrono::steady_clock::now();Simulator env(Config{},seed);dp7::Controller c(pars);EcoBotNative rival;std::array<int,2>overflow{};
 while(!env.done()){std::array<PlayerAction,2>a;a[seat]=c.act(view(env,seat));a[1-seat]=rival.act(env,1-seat);env.step(a);for(int p=0;p<2;p++)overflow[p]+=env.last_end_of_day_overflow()[p];}
 return {seed,seat,env.step_count(),env.farms()[seat].money,env.farms()[1-seat].money,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),false,overflow};
}
using ProductionSeason=std::array<std::array<dp7audit::ProductionDay,2>,30>;
struct ProductionResult {ProductionSeason days{};std::vector<dp7audit::NoEffect>examples;std::array<double,2>money{};};
// Offline audit routing only. These opponent handles/counters never enter View.
struct AuditOpponentSpec {
 int kind=0;const G001* route=nullptr;const boatlee29::Agent* boatlee=nullptr;
 const kaito58::Agent* kaito=nullptr;const lynn5::Agent* lynn=nullptr;
};
struct AuditOpponent {
 AuditOpponentSpec spec;G001State route;boatlee29::State boatlee;kaito58::State kaito;lynn5::State lynn;
 fieldbook::Controller fieldbook;threeday::Controller three_day;EcoBotNative eco;
 explicit AuditOpponent(AuditOpponentSpec s):spec(s){}
 PlayerAction act(const Simulator&s,int p){switch(spec.kind){
  case 0:return {};case 1:return spec.route->act(s,p,route);
  case 2:return spec.boatlee->act(view(s,p),p,boatlee);
  case 3:return spec.kaito->act(view(s,p),p,kaito);
  case 4:return spec.lynn->act(view(s,p),p,lynn);
  case 5:return fieldbook.act(view(s,p),p);case 6:return three_day.act(view(s,p),p);
  case 7:return eco.act(s,p);default:throw std::invalid_argument("audit opponent kind");
 }}
};

} // namespace bridge

#include <dlfcn.h>
#include <iomanip>
#include <sstream>
struct DynamicPolicy {
 std::string path_;
 void* handle=nullptr;void* ctx=nullptr;
 void*(*create)(const double*,size_t)=nullptr;
 int(*step)(void*,const dp7::View*,fastkag::PlayerAction*)=nullptr;
 void(*destroy)(void*)=nullptr;const char*(*inspect)(void*)=nullptr;
 void*(*clone_ctx)(void*)=nullptr;
 int(*prepare)(void*,const dp7::View*)=nullptr;
 int(*features)(void*,int,double*,size_t)=nullptr;
 int(*candidate_id)(void*,int)=nullptr;
 int(*install)(void*,int)=nullptr;
 DynamicPolicy(const std::string&path,const std::vector<double>&c):path_(path){
  handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);if(!handle)throw std::runtime_error(dlerror());
  create=(decltype(create))dlsym(handle,"td_new");step=(decltype(step))dlsym(handle,"td_act");destroy=(decltype(destroy))dlsym(handle,"td_delete");inspect=(decltype(inspect))dlsym(handle,"td_debug");
  clone_ctx=(decltype(clone_ctx))dlsym(handle,"td_clone");prepare=(decltype(prepare))dlsym(handle,"td_prepare");features=(decltype(features))dlsym(handle,"td_candidate_features");candidate_id=(decltype(candidate_id))dlsym(handle,"td_candidate_id");install=(decltype(install))dlsym(handle,"td_install");
  if(!create||!step||!destroy)throw std::runtime_error("invalid policy ABI");ctx=create(c.data(),c.size());if(!ctx)throw std::runtime_error("policy configuration rejected");
 }
 DynamicPolicy(const DynamicPolicy&o):path_(o.path_),create(o.create),step(o.step),destroy(o.destroy),inspect(o.inspect),clone_ctx(o.clone_ctx),prepare(o.prepare),features(o.features),candidate_id(o.candidate_id),install(o.install){
  if(!clone_ctx)throw std::runtime_error("clone unsupported");handle=dlopen(path_.c_str(),RTLD_NOW|RTLD_LOCAL);if(!handle)throw std::runtime_error(dlerror());ctx=clone_ctx(o.ctx);if(!ctx)throw std::runtime_error("clone failed");
 }
 DynamicPolicy&operator=(const DynamicPolicy&)=delete;
 ~DynamicPolicy(){if(ctx)destroy(ctx);if(handle)dlclose(handle);}
 fastkag::PlayerAction act(const dp7::View&v){fastkag::PlayerAction a;if(step(ctx,&v,&a))throw std::runtime_error(inspect?inspect(ctx):"policy error");return a;}
};
struct MatchResult {
 uint64_t seed=0,hash=1469598103934665603ULL;int seat=0,opponent=0,steps=0,overflow=0;
 double cash=0,other=0,elapsed=0,policy_seconds=0,max_ms=0;std::string error,debug;
 std::array<int,24> ops{};std::vector<std::array<fastkag::PlayerAction,2>>actions;
 std::vector<fastkag::Simulator>days;
 std::array<dp7audit::ProductionDay,30>production;std::vector<dp7audit::NoEffect>noeffects;
};
struct LabelRow {
 uint64_t seed=0;int seat=0,opponent=0,day=0,candidate=0,index=0;
 double cash=0,other=0;std::vector<double>x;std::string error;
};
struct Collection {std::vector<LabelRow> rows;double cash=0,other=0;std::string error;};
class Pool{
 std::unique_ptr<bridge::G001>g1,g3;std::unique_ptr<boatlee29::Agent>boat;
 std::unique_ptr<kaito58::Agent>kaito;std::unique_ptr<lynn5::Agent>lynn;
 public:
 Pool(py::dict assets){g1=std::make_unique<bridge::G001>(bridge::load_g001(assets["g001"]));g3=std::make_unique<bridge::G001>(bridge::load_g001(assets["g003"]));boat=std::make_unique<boatlee29::Agent>(bridge::load_boatlee(assets["boatlee_v29"]));kaito=std::make_unique<kaito58::Agent>(bridge::load_kaito(assets["kaito_v58"]));lynn=std::make_unique<lynn5::Agent>(bridge::load_lynn(assets["lynn_v5"]));}
 MatchResult runone(const std::string&lib,const std::vector<double>&cfg,uint64_t seed,int seat,int opponent,bool trace){
  using namespace bridge;MatchResult r;r.seed=seed;r.seat=seat;r.opponent=opponent;
  auto start=std::chrono::steady_clock::now();
  try {
   DynamicPolicy own(lib,cfg);AuditOpponentSpec spec;
   if(opponent==0){spec.kind=1;spec.route=g1.get();}else if(opponent==1){spec.kind=1;spec.route=g3.get();}
   else if(opponent==2){spec.kind=2;spec.boatlee=boat.get();}else if(opponent==3){spec.kind=3;spec.kaito=kaito.get();}
   else if(opponent==4){spec.kind=4;spec.lynn=lynn.get();}else if(opponent==5)spec.kind=5;else if(opponent==6)spec.kind=6;else spec.kind=0;
   AuditOpponent rival(spec);fastkag::Simulator env(fastkag::Config{},seed);
   while(!env.done()){
    if(trace&&env.hour()==0)r.days.push_back(env);
    auto t=std::chrono::steady_clock::now();std::array<fastkag::PlayerAction,2>a;a[seat]=own.act(view(env,seat));
    double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();r.policy_seconds+=sec;r.max_ms=std::max(r.max_ms,sec*1000);
    a[1-seat]=rival.act(env,1-seat);
    for(auto&x:a[seat].units)r.ops[int(x.op)]++;
    for(auto&x:a[seat].market)r.ops[int(x.op)]++;
    auto push=[&](uint64_t x){r.hash^=x;r.hash*=1099511628211ULL;};
    for(auto&x:a){push(x.units.size());for(auto&t:x.units){push(int(t.op));push(int(t.item)+1);push(t.quantity);}push(x.market.size());for(auto&t:x.market){push(int(t.op));push(int(t.item)+1);push(t.quantity);}}
    if(trace){r.actions.push_back(a);dp7audit::units(env,seat,a[seat],r.production[env.day()],r.noeffects);}
    env.step(a);r.overflow+=env.last_end_of_day_overflow()[seat];r.steps=env.step_count();
   }
   r.cash=env.farms()[seat].money;r.other=env.farms()[1-seat].money;
   if(own.inspect)r.debug=own.inspect(own.ctx);if(trace)r.days.push_back(env);
  }catch(const std::exception&e){r.error=e.what();}
  r.elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return r;
 }
 bridge::AuditOpponentSpec spec_for(int opponent){
  bridge::AuditOpponentSpec spec;
  if(opponent==0){spec.kind=1;spec.route=g1.get();}else if(opponent==1){spec.kind=1;spec.route=g3.get();}
  else if(opponent==2){spec.kind=2;spec.boatlee=boat.get();}else if(opponent==3){spec.kind=3;spec.kaito=kaito.get();}
  else if(opponent==4){spec.kind=4;spec.lynn=lynn.get();}else if(opponent==5)spec.kind=5;else if(opponent==6)spec.kind=6;
  else throw std::invalid_argument("seven-opponent index");return spec;
 }
 Collection collectone(const std::string&lib,const std::vector<double>&cfg,uint64_t seed,int seat,int opponent,const std::vector<int>&days,bool reverse){
  Collection out;
  try{
   DynamicPolicy own(lib,cfg);if(!own.prepare||!own.features||!own.install)throw std::runtime_error("counterfactual ABI missing");
   bridge::AuditOpponent rival(spec_for(opponent));Simulator env(Config{},seed);
   while(!env.done()){
    if(env.hour()==0&&std::find(days.begin(),days.end(),env.day())!=days.end()){
     auto v=bridge::view(env,seat);int n=own.prepare(own.ctx,&v);if(n<1)throw std::runtime_error("empty portfolio candidates");
     // Freeze all feature vectors BEFORE running any label rollout.
     std::vector<LabelRow> rows(n);
     for(int k=0;k<n;k++){
      auto&r=rows[k];r.seed=seed;r.seat=seat;r.opponent=opponent;r.day=env.day();r.index=k;r.candidate=own.candidate_id(own.ctx,k);r.x.resize(4096);int m=own.features(own.ctx,k,r.x.data(),r.x.size());if(m<=0)throw std::runtime_error("features failed");r.x.resize(m);
     }
     for(int z=0;z<n;z++){
      int k=reverse?n-1-z:z;auto&r=rows[k];
      // These hidden-state copies exist exclusively in the OFFLINE labeller.
      // Candidate inference and feature generation receive only public View.
      DynamicPolicy policy(own);bridge::AuditOpponent enemy(rival);Simulator branch(env);
      if(policy.install(policy.ctx,k))throw std::runtime_error("candidate installation failed");
      while(!branch.done()){
       std::array<PlayerAction,2>a;a[seat]=policy.act(bridge::view(branch,seat));a[1-seat]=enemy.act(branch,1-seat);branch.step(a);
      }
      r.cash=branch.farms()[seat].money;r.other=branch.farms()[1-seat].money;
     }
     out.rows.insert(out.rows.end(),rows.begin(),rows.end());
    }
    std::array<PlayerAction,2>a;a[seat]=own.act(bridge::view(env,seat));a[1-seat]=rival.act(env,1-seat);env.step(a);
   }
   out.cash=env.farms()[seat].money;out.other=env.farms()[1-seat].money;
   // KEEP must exactly reproduce the uninterrupted reference trajectory.
   // This checks hidden opponent carry, own carry and shared continuation.
   for(auto&r:out.rows)if(r.candidate==0&&(r.cash!=out.cash||r.other!=out.other))throw std::runtime_error("KEEP continuation not identical");
  }catch(const std::exception&e){out.error=e.what();}
  return out;
 }
 py::list collect(const std::string&lib,const std::vector<double>&cfg,const std::vector<uint64_t>&seeds,const std::vector<int>&opponents,const std::vector<int>&days,int threads,bool reverse){
  if(threads<1||threads>16||days.empty())throw std::invalid_argument("collector dimensions");
  size_t n=seeds.size()*opponents.size()*2;std::vector<Collection> rr(n);
  {py::gil_scoped_release release;
   #pragma omp parallel for schedule(dynamic) num_threads(threads)
   for(size_t i=0;i<n;i++)rr[i]=collectone(lib,cfg,seeds[i/(opponents.size()*2)],i%2,opponents[(i/2)%opponents.size()],days,reverse);
  }
  py::list out;for(auto&c:rr){if(!c.error.empty())throw std::runtime_error(c.error);for(auto&r:c.rows){py::dict d;d["seed"]=r.seed;d["opponent"]=r.opponent;d["seat"]=r.seat;d["day"]=r.day;d["candidate"]=r.candidate;d["index"]=r.index;d["features"]=r.x;d["cash"]=r.cash;d["opponent_cash"]=r.other;d["margin"]=r.cash-r.other;d["reference_cash"]=c.cash;d["reference_opponent_cash"]=c.other;out.append(d);}}return out;
 }
 py::list run(const std::string&lib,const std::vector<double>&cfg,const std::vector<uint64_t>&seeds,const std::vector<int>&opponents,int threads,bool trace){
  if(threads<1||threads>16||seeds.empty()||opponents.empty())throw std::invalid_argument("invalid panel dimensions");
  const size_t n=seeds.size()*opponents.size()*2;std::vector<MatchResult>rr(n);
  {py::gil_scoped_release release;
   #pragma omp parallel for schedule(dynamic) num_threads(threads)
   for(size_t i=0;i<n;i++)rr[i]=runone(lib,cfg,seeds[i/(opponents.size()*2)],i%2,opponents[(i/2)%opponents.size()],trace);
  }
  py::list result;for(auto&r:rr){py::dict d;d["seed"]=r.seed;d["seat"]=r.seat;d["opponent"]=r.opponent;d["steps"]=r.steps;d["cash"]=r.cash;d["opponent_cash"]=r.other;d["win"]=r.error.empty()&&r.cash>r.other;d["margin"]=r.cash-r.other;d["error"]=r.error;d["seconds"]=r.elapsed;d["policy_seconds"]=r.policy_seconds;d["max_action_ms"]=r.max_ms;d["overflow"]=r.overflow;d["action_hash_fnv64"]=r.hash;d["ops"]=r.ops;d["debug"]=r.debug;
   if(trace){py::list a,days;for(auto&pair:r.actions){py::list row;row.append(bridge::pack_player(pair[0]));row.append(bridge::pack_player(pair[1]));a.append(row);}for(auto&e:r.days)days.append(bridge::observation(e,r.seat));d["actions"]=a;d["days"]=days;
    py::list prod,fail;for(int day=0;day<30;day++){auto&x=r.production[day];py::dict q;q["day"]=day;q["attempts"]=x.attempts;q["no_effect"]=x.no_effect;q["drop_loss"]=x.drop_loss;prod.append(q);}for(auto&x:r.noeffects){py::dict q;q["step"]=x.step;q["unit"]=x.unit;q["op"]=x.op;q["position"]=x.position;q["item"]=x.item;fail.append(q);}d["production_audit"]=prod;d["no_effect_examples"]=fail;}
   result.append(d);
  }return result;
 }
};
PYBIND11_MODULE(_triad_panel,m){py::class_<Pool>(m,"Pool").def(py::init<py::dict>()).def("run",&Pool::run,py::arg("library"),py::arg("config"),py::arg("seeds"),py::arg("opponents"),py::arg("threads")=4,py::arg("trace")=false).def("collect",&Pool::collect,py::arg("library"),py::arg("config"),py::arg("seeds"),py::arg("opponents"),py::arg("days"),py::arg("threads")=4,py::arg("reverse")=false);}
