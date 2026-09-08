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
#include "investment_branch_audit.hpp"
struct PoolAuditResult {
 ProductionResult production;CashSeason cash{};std::array<DayAudit,30>planning{};
 dp7audit::InvestmentAudit investments;
 std::array<dp7audit::Admission,30>admissions{};
 std::array<int,2>overflow{};std::vector<int8_t>shops;
};
PoolAuditResult pool_audit(uint64_t seed,int seat,dp7::Params pars,AuditOpponentSpec spec){
 Simulator env(Config{},seed);dp7::Controller own(pars);AuditOpponent rival(spec);PoolAuditResult result;
 while(!env.done()){
  int day=env.day(),hour=env.hour();auto&plan=result.planning[day];plan.day=day;
  if(hour==0)for(int p=0;p<2;p++)result.cash[day][p].start=env.farms()[p].money;
  std::array<PlayerAction,2>a;a[seat]=own.act(view(env,seat));a[1-seat]=rival.act(env,1-seat);
  if(hour==0)result.investments.propose(own,view(env,seat));
  if(own.phase==3&&!result.admissions[day].captured)result.admissions[day]=dp7audit::admission(own,view(env,seat));
  if(hour==0)for(auto[pos,k]:own.target)if(k>=0)plan.target[k]++;
  plan.compile_drop=std::max(plan.compile_drop,own.actual_drop);plan.degraded=std::max(plan.degraded,own.resource_degraded);
  plan.max_hands=std::max(plan.max_hands,int(env.farms()[seat].hands.size()));
  plan.bundle_evaluations=own.bundle_evaluations;plan.bundle_switches=own.bundle_switches;plan.bundle_removed_targets=own.bundle_removed_targets;
  plan.portfolio_generated=own.portfolio_generated;plan.portfolio_evaluated=own.portfolio_evaluated;
  plan.portfolio_switches=own.portfolio_switches;plan.portfolio_rejected=own.portfolio_rejected;
  plan.preparation_actions=own.preparation_actions;
  plan.service_checks=own.service_checks;plan.service_recovered_feed=own.service_recovered_feed;plan.service_recovered_fertilize=own.service_recovered_fertilize;
  plan.service_buys=own.service_buys;plan.service_financed=own.service_financed;plan.service_deposits=own.service_deposits;
  plan.resource_exchange_checks=own.resource_exchange_checks;plan.resource_exchange_pairs=own.resource_exchange_pairs;
  plan.shared_insertion_checks=own.shared_insertion_checks;plan.shared_insertion_applied=own.shared_insertion_applied;
  plan.shared_insertion_peak_saved=own.shared_insertion_peak_saved;plan.shared_insertion_steps_saved=own.shared_insertion_steps_saved;
  plan.resource_exchange_applied=own.resource_exchange_applied;plan.resource_exchange_local_steps_saved=own.resource_exchange_local_steps_saved;
  plan.intraday_checks=own.intraday_checks;plan.intraday_proposals=own.intraday_proposals;plan.intraday_activated=own.intraday_activated;
  plan.intraday_cancelled=own.intraday_cancelled;plan.intraday_unfilled=own.intraday_unfilled;
  plan.intraday_purchase_orders=own.intraday_purchase_orders;plan.intraday_activated_by_kind=own.intraday_activated_by_kind;
   plan.shared_service_plots=own.shared_service_plots;plan.regret_trials=own.regret_trials;plan.regret_improvements=own.regret_improvements;
   plan.terminal_schedule_evaluations=own.terminal_schedule_evaluations;plan.terminal_schedule_switches=own.terminal_schedule_switches;plan.terminal_expected_cash_gain=own.terminal_expected_cash_gain;
   plan.overflow_dispatch_checks=own.overflow_dispatch_checks;plan.overflow_dispatch_units=own.overflow_dispatch_units;plan.overflow_dispatch_quantity=own.overflow_dispatch_quantity;
  std::array<Simulator,2>projected{dp7audit::units(env,0,a[0],result.production.days[day][0],result.production.examples,seat==0?&result.investments:nullptr),dp7audit::units(env,1,a[1],result.production.days[day][1],result.production.examples,seat==1?&result.investments:nullptr)};
  Simulator before=env;dp7audit::BeforeMarket market_before(env);env.step(a);
  dp7audit::finish(before,projected,env,a,result.production.days[day]);dp7audit::accumulate(market_before,env,a,result.cash[day]);
  result.investments.finish(before,env,seat,a[seat]);
  for(int p=0;p<2;p++)result.overflow[p]+=env.last_end_of_day_overflow()[p];
  if(hour==23||env.done()){
   plan.live=own.counts(view(env,seat));plan.opponent_live=own.counts(view(env,1-seat));
   plan.land=std::popcount(unsigned(env.farms()[seat].unlocked_mask));
   for(auto&t:projected[seat].farms()[seat].tiles){plan.unfed+=dp7::animal(t)&&!t.fed_today;plan.unwatered+=dp7::plant(t)&&!t.watered_today;}
  }
 }
 for(int p=0;p<2;p++)result.production.money[p]=env.farms()[p].money;
 for(auto&day:result.cash)for(auto&d:day)if(d.start+d.net()!=d.end)throw std::runtime_error("pool cash reconciliation");
 result.shops=env.shops();return result;
}
ProductionResult production_audit(uint64_t seed,int seat,dp7::Params pars,const G001*g){
 Simulator env(Config{},seed);dp7::Controller c(pars);G001State st;ProductionResult result;
 while(!env.done()){
  int day=env.day();std::array<PlayerAction,2>a;a[seat]=c.act(view(env,seat));if(g)a[1-seat]=g->act(env,1-seat,st);
  std::array<Simulator,2>projected{dp7audit::units(env,0,a[0],result.days[day][0],result.examples),dp7audit::units(env,1,a[1],result.days[day][1],result.examples)};
  Simulator before=env;env.step(a);dp7audit::finish(before,projected,env,a,result.days[day]);
 }
 for(int p=0;p<2;p++)result.money[p]=env.farms()[p].money;return result;
}
CashSeason cash_audit(uint64_t seed,int seat,dp7::Params pars,const G001*g){
 Simulator env(Config{},seed);dp7::Controller c(pars);G001State st;CashSeason days{};
 while(!env.done()){
  int day=env.day();if(env.hour()==0)for(int p=0;p<2;p++)days[day][p].start=env.farms()[p].money;
  std::array<PlayerAction,2>a;a[seat]=c.act(view(env,seat));if(g)a[1-seat]=g->act(env,1-seat,st);
  dp7audit::BeforeMarket before(env);env.step(a);dp7audit::accumulate(before,env,a,days[day]);
 }
 for(auto&day:days)for(auto&d:day)if(d.start+d.net()!=d.end)throw std::runtime_error("daily cash reconciliation");
 return days;
}
std::vector<DayAudit>audit(uint64_t seed,int seat,dp7::Params pars,const G001*g){Simulator env(Config{},seed);dp7::Controller c(pars);G001State st;std::vector<DayAudit>days(30);while(!env.done()){int day=env.day(),hour=env.hour();auto&d=days[day];d.day=day;if(hour==0)d.start_money=env.farms()[seat].money;std::array<PlayerAction,2>a;a[seat]=c.act(view(env,seat));if(g)a[1-seat]=g->act(env,1-seat,st);if(hour==0)for(auto[pos,k]:c.target)if(k>=0)d.target[k]++;d.compile_drop=std::max(d.compile_drop,c.actual_drop);d.degraded=std::max(d.degraded,c.resource_degraded);d.max_hands=std::max(d.max_hands,int(env.farms()[seat].hands.size()));if(hour==23||env.step_count()==718){auto projected=env.project_unit_phase(seat,a[seat].units);for(auto&t:projected.farms()[seat].tiles){d.unfed+=dp7::animal(t)&&!t.fed_today;d.unwatered+=dp7::plant(t)&&!t.watered_today;}for(auto&inv:projected.privates()[seat].inventories)dp7::add(d.carried,inv);}
 env.step(a);d.overflow+=env.last_end_of_day_overflow()[seat];auto&fills=env.last_market_fills()[seat];for(size_t i=0;i<fills.size()&&i<a[seat].market.size();i++){auto&x=a[seat].market[i];if(x.op==Op::BUY_PRODUCT||x.op==Op::BUY_ANIMAL||x.op==Op::BUY_SEED)d.bought[int(x.item)]+=fills[i];if(x.op==Op::SELL)d.sold[int(x.item)]+=fills[i];if(x.op==Op::HIRE)d.hired+=fills[i];}if(hour==23||env.done()){d.end_money=env.farms()[seat].money;d.opponent_money=env.farms()[1-seat].money;d.live=c.counts(view(env,seat));d.opponent_live=c.counts(view(env,1-seat));d.shed=env.privates()[seat].shed;d.land=std::popcount(unsigned(env.farms()[seat].unlocked_mask));}}
 return days;}
}
PYBIND11_MODULE(_dp7_native,m){using namespace bridge;
 m.def("rotation_stats",[](const dp7::Controller&c){py::dict d;d["generated"]=c.rotation_generated;d["evaluated"]=c.rotation_evaluated;d["applied"]=c.rotation_applied;d["waits"]=c.rotation_waits;d["unknown"]=c.rotation_unknown;d["plant_not_before"]=c.plant_not_before;d["deferred_kind"]=c.deferred_kind;return d;});
 m.def("compile_choice_stats",[](const dp7::Controller&c){py::dict d;d["calls"]=c.compile_choice_calls;d["candidates"]=c.compile_choice_candidates;d["evaluations"]=c.compile_choice_evaluations;d["changes"]=c.compile_choice_changes;d["unknown"]=c.compile_choice_unknown;d["unsafe"]=c.compile_choice_unsafe;return d;});
 m.def("schedule_cache_stats",[](const dp7::Controller&c){py::dict d;d["day"]=c.schedule_cache_day;d["active_context"]=dp7::packmemo::active!=nullptr;d["present"]=bool(c.schedule_cache);if(c.schedule_cache){auto&x=*c.schedule_cache;d["calls"]=x.calls;d["hits"]=x.hits;d["misses"]=x.misses;d["entries"]=x.entries.size();d["accounted_bytes"]=x.accounted_bytes;d["not_stored"]=x.not_stored;}return d;});
 m.def("shared_insertion_stats",[](const dp7::Controller&c){py::dict d;d["checks"]=c.shared_insertion_checks;d["applied"]=c.shared_insertion_applied;d["peak_saved"]=c.shared_insertion_peak_saved;d["steps_saved"]=c.shared_insertion_steps_saved;return d;});
 m.def("intraday_workforce_stats",[](const dp7::Controller&c){py::dict d;d["checks"]=c.intraday_workforce_checks;d["candidates"]=c.intraday_workforce_candidates;d["switches"]=c.intraday_workforce_switches;d["unknown"]=c.intraday_workforce_unknown;return d;});
 m.def("inspect_recoordination",[](const dp7::Controller&c,const Simulator&s,int seat){
  auto r=dp7audit::inspect_recoord(c,view(s,seat));py::dict d;d["eligible"]=r.eligible;d["legacy_accepts"]=r.legacy_accepts;
  d["movable"]=r.movable;d["covered"]=r.covered;d["assigned"]=r.assigned;d["remaining"]=r.remaining;py::list extra;
  d["insertion_improves"]=r.insertion_improves;d["insertion_assigned"]=r.insertion_assigned;d["peak_saved"]=r.peak_saved;d["total_saved"]=r.total_saved;
  for(const auto&e:r.extra){py::dict x;x["pos"]=e.pos;x["op"]=e.op;x["unit"]=e.unit;x["finish"]=e.finish;x["extra"]=e.extra;x["nominal_value"]=e.nominal_value;extra.append(x);}d["extra"]=extra;return d;
 });
 m.def("portfolio_stats",[](const dp7::Controller&c){py::dict d;d["generated"]=c.portfolio_generated;d["evaluated"]=c.portfolio_evaluated;d["rejected"]=c.portfolio_rejected;d["switches"]=c.portfolio_switches;return d;});
 m.def("service_stats",[](const dp7::Controller&c){py::dict d;d["checks"]=c.service_checks;d["feed"]=c.service_recovered_feed;d["fertilize"]=c.service_recovered_fertilize;d["buys"]=c.service_buys;d["financed"]=c.service_financed;d["deposits"]=c.service_deposits;return d;});
 m.def("live_repair_stats",[](const dp7::Controller&c){py::dict d;d["market_checks"]=c.live_market_checks;d["market_added"]=c.live_market_added;d["market_slots_blocked"]=c.live_market_slots_blocked;d["feed_checks"]=c.feed_insert_checks;d["feed_applied"]=c.feed_insert_applied;d["feed_unresolved"]=c.feed_insert_unresolved;return d;});
 m.def("day_value_stats",[](const dp7::Controller&c){py::dict d;d["checks"]=c.day_value_checks;d["rejected"]=c.day_value_rejected;d["unknown"]=c.day_value_unknown;return d;});
 m.def("idle_handoff_stats",[](const dp7::Controller&c){py::dict d;d["checks"]=c.idle_handoff_checks;d["applied"]=c.idle_handoff_applied;d["prefix"]=c.idle_handoff_prefix;return d;});
 py::class_<Simulator>(m,"Env").def(py::init([](uint64_t seed){return Simulator(Config{},seed);})).def("step",[](Simulator&s,py::sequence a){s.step({parse_player(a[0]),parse_player(a[1])});}).def("observation",&observation).def_property_readonly("done",&Simulator::done).def_property_readonly("step_count",&Simulator::step_count);
 py::class_<dp7::Controller>(m,"Controller").def(py::init([](py::dict d){return dp7::Controller(params(py::dict(d.attr("copy")())));} ),py::arg("params")=py::dict()).def("act",[](dp7::Controller&c,const Simulator&s,int p){return pack_player(c.act(view(s,p)));}).def("debug",[](dp7::Controller&c){py::dict d;d["phase"]=c.phase;d["target"]=c.target;d["daily_need"]=c.daily_need;d["expected_output"]=c.expected;d["actual_drop"]=c.actual_drop;d["feed_stock_target"]=c.feed_stock_target;d["resource_degraded"]=c.resource_degraded;d["prepared_seed_need"]=c.prepared_seed_need;d["seed_reconcile_checked"]=c.seed_reconcile_checked;d["seed_reconciliations"]=c.seed_reconciliations;d["seed_reconciled_units"]=c.seed_reconciled_units;d["anticipated_releases"]=c.anticipated_releases;d["step_recoord_checks"]=c.step_recoord_checks;d["step_recoord_rebuilds"]=c.step_recoord_rebuilds;d["step_recoord_stale_groups"]=c.step_recoord_stale_groups;d["step_recoord_added_groups"]=c.step_recoord_added_groups;d["step_recoord_reassigned_groups"]=c.step_recoord_reassigned_groups;d["preparation_pipeline_checks"]=c.preparation_pipeline_checks;d["preparation_pipeline_actions"]=c.preparation_pipeline_actions;d["preparation_pipeline_moves"]=c.preparation_pipeline_moves;d["schedule_value_evaluations"]=c.schedule_value_evaluations;d["schedule_value_switches"]=c.schedule_value_switches;d["schedule_value_gain"]=c.schedule_value_gain;return d;});
 py::class_<G001State>(m,"G001State").def(py::init<>()).def_readonly("current",&G001State::current).def_readonly("switched",&G001State::switched);
 py::class_<G001>(m,"G001").def(py::init([](py::dict d){return G001(load_g001(d));})).def("act",[](const G001&g,const Simulator&s,int p,G001State&st){return pack_player(g.act(s,p,st));});
 py::class_<fieldbook::Controller>(m,"Fieldbook").def(py::init<>()).def("act",[](fieldbook::Controller&c,const Simulator&s,int p){return pack_player(c.act(view(s,p),p));}).def("segments",&fieldbook::Controller::segments);
 py::class_<threeday::Controller>(m,"ThreeDay").def(py::init<>()).def("act",[](threeday::Controller&c,const Simulator&s,int p){return pack_player(c.act(view(s,p),p));}).def("route",&threeday::Controller::route);
 py::class_<EcoBotNative>(m,"EcoBotV7").def(py::init<>()).def("act",[](EcoBotNative&c,const Simulator&s,int p){return pack_player(c.act(s,p));});
 m.def("batch_ecobot",[](std::vector<uint64_t>seeds,std::vector<int>seats,py::dict pd,int threads){
   if(seeds.size()!=seats.size()||threads<1||threads>16)throw std::invalid_argument("EcoBot batch shape/threads");for(int p:seats)if(p<0||p>1)throw std::invalid_argument("seat");
   auto pars=params(py::dict(pd.attr("copy")()));std::vector<Result>results(seeds.size());std::vector<std::string>errors(seeds.size());
   {py::gil_scoped_release release;
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for(int i=0;i<int(seeds.size());i++)try{results[i]=run_ecobot(seeds[i],seats[i],pars);}catch(const std::exception&e){errors[i]=e.what();}
   }
   py::list out;for(size_t i=0;i<results.size();i++){if(!errors[i].empty())throw std::runtime_error(errors[i]);auto&x=results[i];py::dict d;d["seed"]=x.seed;d["seat"]=x.seat;d["steps"]=x.steps;d["cash"]=x.cash;d["opponent_cash"]=x.other;d["margin"]=x.cash-x.other;d["win"]=x.cash>x.other;d["seconds"]=x.seconds;d["overflow"]=x.overflow;d["opponent_switched"]=py::none();out.append(d);}return out;
 });
 m.def("three_day_input_fields",[](const Simulator&s,int p){return threeday::input_fields(view(s,p),p);});
 m.def("batch_three_day",[](std::vector<uint64_t>seeds,std::vector<int>seats,py::dict pd,int threads){
   if(seeds.size()!=seats.size()||threads<1||threads>16)throw std::invalid_argument("Three-Day batch shape/threads");for(int p:seats)if(p<0||p>1)throw std::invalid_argument("seat");
   auto pars=params(py::dict(pd.attr("copy")()));std::vector<ThreeDayResult>results(seeds.size());std::vector<std::string>errors(seeds.size());
   {py::gil_scoped_release release;
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for(int i=0;i<int(seeds.size());i++)try{results[i]=run_threeday(seeds[i],seats[i],pars);}catch(const std::exception&e){errors[i]=e.what();}
   }
   py::list out;for(size_t i=0;i<results.size();i++){if(!errors[i].empty())throw std::runtime_error(errors[i]);auto&r=results[i];auto&x=r.match;py::dict d;d["seed"]=x.seed;d["seat"]=x.seat;d["steps"]=x.steps;d["cash"]=x.cash;d["opponent_cash"]=x.other;d["margin"]=x.cash-x.other;d["win"]=x.cash>x.other;d["seconds"]=x.seconds;d["overflow"]=x.overflow;d["opponent_switched"]=py::none();d["selected_route"]=r.route;out.append(d);}return out;
 });
 py::class_<boatlee29::State>(m,"BoatleeState").def(py::init<>()).def("debug",&boatlee_debug);
 py::class_<lynn5::State>(m,"LynnState").def(py::init<>()).def("debug",&lynn_debug);
 py::class_<lynn5::Agent>(m,"LynnV5").def(py::init([](py::dict d){return lynn5::Agent(load_lynn(d));})).def("act",[](const lynn5::Agent&a,const Simulator&s,int p,lynn5::State&st){return pack_player(a.act(view(s,p),p,st));})
 .def("probe_observation",[](const lynn5::Agent&a,py::dict obs,lynn5::State&st){
   // Test-only public/own-observation adapter. Never injects into the simulator
   // and never supplies opponent private state or a seed to either policy.
   auto position=[](py::handle h){auto x=py::cast<py::sequence>(h);return Position{py::cast<int16_t>(x[0]),py::cast<int16_t>(x[1])};};
   auto item=[](py::handle h){auto s=py::cast<std::string>(h);for(int i=0;i<12;i++)if(s==item_name(i))return i;throw std::invalid_argument("Lynn observation item");};
   std::array<Farm,2>farms;auto rawfarms=py::cast<py::sequence>(obs["farms"]);
   for(int p=0;p<2;p++){auto d=py::cast<py::dict>(rawfarms[p]);auto&f=farms[p];f.money=py::cast<double>(d["money"]);f.farmer=position(d["farmer"]);for(auto h:d["hands"])f.hands.push_back(position(h));f.hires_today=py::cast<int>(d["hires_today"]);f.unlocked_mask=(1<<py::len(d["unlocked_quadrants"]))-1;
     for(auto row:d["tiles"])for(auto raw:row){Tile t;if(raw.is_none())t.kind=TileKind::EMPTY;else if(py::isinstance<py::str>(raw))t.kind=TileKind::LOCKED;else {auto tile=py::cast<py::dict>(raw);auto k=py::cast<std::string>(tile["kind"]);if(k=="WEED")t.kind=TileKind::WEED;else if(k=="PLANT")t.kind=TileKind::PLANT;else if(k=="COOP")t.kind=TileKind::COOP;else if(k=="PASTURE")t.kind=TileKind::PASTURE;else throw std::invalid_argument("Lynn tile");if(tile.contains("animal")){t.kind=TileKind::ANIMAL;t.animal=Item(item(tile["animal"]));}}f.tiles.push_back(t);}
     if(f.tiles.size()!=100)throw std::invalid_argument("Lynn board");
   }
   PrivateState priv;auto pr=py::cast<py::dict>(obs["private"]);for(auto kv:py::cast<py::dict>(pr["shed"]))priv.shed[item(kv.first)]=py::cast<int>(kv.second);
   for(auto raw:pr["inventories"]){std::array<int32_t,12>counts{};std::vector<int8_t>order;for(auto kv:py::cast<py::dict>(raw)){int i=item(kv.first);counts[i]=py::cast<int>(kv.second);if(counts[i])order.push_back(i);}priv.inventories.push_back(counts);priv.inventory_order.push_back(order);}
   Market market;auto mr=py::cast<py::dict>(obs["market"]);for(auto kv:py::cast<py::dict>(mr["inventory"]))market.inventory[item(kv.first)]=py::cast<int>(kv.second);for(auto kv:py::cast<py::dict>(mr["prices"]))market.prices[item(kv.first)]=py::cast<int>(kv.second);
   std::vector<int8_t>shops;auto town=py::cast<py::dict>(obs["town"]);for(auto name:town["unlocked_shops"]){auto n=py::cast<std::string>(name);int i=0;for(;i<8;i++)if(n==shop_name(i))break;if(i==8)throw std::invalid_argument("Lynn shop");shops.push_back(i);}
   int p=py::cast<int>(obs["player"]),step=py::cast<int>(obs["step"]);if(p<0||p>1)throw std::invalid_argument("Lynn seat");dp7::View v{step,step/24,step%24,farms[p],farms[1-p],priv,market,shops};return pack_player(a.act(v,p,st));
 },"Test-only observation differential probe; not a match runner.");
 m.def("batch_lynn",[](std::vector<uint64_t>seeds,std::vector<int>seats,py::dict pd,const lynn5::Agent&opponent,int threads){
   if(seeds.size()!=seats.size()||threads<1||threads>16)throw std::invalid_argument("Lynn batch shape/threads");for(int p:seats)if(p<0||p>1)throw std::invalid_argument("seat");
   auto pars=params(py::dict(pd.attr("copy")()));std::vector<LynnResult> results(seeds.size());std::vector<std::string> errors(seeds.size());
   {py::gil_scoped_release release;
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for(int i=0;i<int(seeds.size());i++)try{results[i]=run_lynn(seeds[i],seats[i],pars,opponent);}catch(const std::exception&e){errors[i]=e.what();}
   }
   py::list out;for(size_t i=0;i<results.size();i++){if(!errors[i].empty())throw std::runtime_error(errors[i]);auto&r=results[i];auto&x=r.match;py::dict d;d["seed"]=x.seed;d["seat"]=x.seat;d["steps"]=x.steps;d["cash"]=x.cash;d["opponent_cash"]=x.other;d["margin"]=x.cash-x.other;d["win"]=x.cash>x.other;d["seconds"]=x.seconds;d["overflow"]=x.overflow;d["opponent_switched"]=py::none();d["opponent_state"]=lynn_debug(r.state);d["weed_repair_frames"]=r.weed_frames;out.append(d);}return out;
 });
 py::class_<kaito58::State>(m,"KaitoState").def(py::init<>()).def("debug",&kaito_debug);
 py::class_<kaito58::Agent>(m,"KaitoV58").def(py::init([](py::dict d){return kaito58::Agent(load_kaito(d));})).def("act",[](const kaito58::Agent&a,const Simulator&s,int p,kaito58::State&st){return pack_player(a.act(view(s,p),p,st));});
 // Synthetic PUBLIC router fixtures only, never used by a match/controller.
 m.def("kaito_router_probe",[](const kaito58::Agent&a,py::dict spec){
   Simulator env(Config{},0);auto own=env.farms()[0],other=env.farms()[1];auto market=env.market();auto sh=py::cast<std::vector<int8_t>>(spec["shops"]);int step=py::cast<int>(spec["step"]);
   if(spec.contains("signature")){auto s=py::cast<std::array<int,7>>(spec["signature"]);own.money=s[0];other.money=s[1];market.inventory[0]=s[2];other.tiles.assign(100,Tile{});int at=0;
     for(auto [item,n]:std::array<std::pair<int,int>,4>{{{10,s[3]},{11,s[4]},{0,s[5]},{4,s[6]}}})for(int j=0;j<n;j++){if(at>=100)throw std::invalid_argument("router fixture capacity");auto&t=other.tiles[at++];if(item>=9){t.kind=TileKind::ANIMAL;t.animal=Item(item);}else{t.kind=TileKind::PLANT;t.crop=Item(item);}}
   }
   if(spec.contains("mirror")&&py::cast<bool>(spec["mirror"]))other=own;
   kaito58::State st;st.seat=0;st.last=step;auto state=spec.contains("state")?py::cast<py::dict>(spec["state"]):py::dict();
   if(state.contains("mode")&&!state["mode"].is_none())st.mode=py::cast<std::string>(state["mode"]);
   if(state.contains("mirror_streak"))st.mirror_streak=py::cast<int>(state["mirror_streak"]);if(state.contains("clone_selected"))st.clone=py::cast<bool>(state["clone_selected"]);if(state.contains("known_yarn"))st.known_yarn=py::cast<bool>(state["known_yarn"]);
   dp7::View v{step,step/24,step%24,own,other,env.privates()[0],market,sh};st.selected=a.select(v,st);
   auto obs=observation(env,0);obs["step"]=step;obs["day"]=step/24;obs["hour"]=step%24;
   auto farms=py::cast<py::list>(obs["farms"]);for(int p=0;p<2;p++){auto f=py::cast<py::dict>(farms[p]);auto&src=p?other:own;f["money"]=src.money;py::list tiles;for(int y=0;y<10;y++){py::list row;for(int x=0;x<10;x++)row.append(tile(src.tiles[y*10+x]));tiles.append(row);}f["tiles"]=tiles;}
   auto mkt=py::cast<py::dict>(obs["market"]);auto inv=py::cast<py::dict>(mkt["inventory"]);inv["WHEAT"]=market.inventory[0];py::list shops;for(int s:sh)shops.append(shop_name(s));auto town=py::cast<py::dict>(obs["town"]);town["unlocked_shops"]=shops;
   py::dict result;result["observation"]=obs;result["state"]=kaito_debug(st);return result;
 });
 m.def("batch_kaito",[](std::vector<uint64_t>seeds,std::vector<int>seats,py::dict pd,const kaito58::Agent&opponent,int threads){
   if(seeds.size()!=seats.size())throw std::invalid_argument("batch shapes");for(int s:seats)if(s<0||s>1)throw std::invalid_argument("batch seats");
   auto p=params(py::dict(pd.attr("copy")()));std::vector<KaitoResult>results(seeds.size());std::vector<std::string>errors(seeds.size());
   {py::gil_scoped_release release;omp_set_num_threads(std::clamp(threads,1,16));
    #pragma omp parallel for schedule(dynamic)
    for(int i=0;i<int(seeds.size());i++)try{results[i]=run_kaito(seeds[i],seats[i],p,opponent);}catch(const std::exception&e){errors[i]=e.what();}
   }
   py::list out;for(size_t i=0;i<results.size();i++){if(!errors[i].empty())throw std::runtime_error(errors[i]);auto&r=results[i];auto&x=r.match;py::dict d;d["seed"]=x.seed;d["seat"]=x.seat;d["steps"]=x.steps;d["cash"]=x.cash;d["opponent_cash"]=x.other;d["margin"]=x.cash-x.other;d["win"]=x.cash>x.other;d["seconds"]=x.seconds;d["overflow"]=x.overflow;d["selected_route_frames"]=r.selected;d["preempt_units_by_route"]=r.preempt;d["weed_collisions_by_route"]=r.weed;d["opponent_switched"]=py::none();out.append(d);}return out;
 },py::arg("seeds"),py::arg("seats"),py::arg("params"),py::arg("opponent"),py::arg("threads")=16);
 py::class_<boatlee29::Agent>(m,"BoatleeV29").def(py::init([](py::dict d){return boatlee29::Agent(load_boatlee(d));})).def("act",[](const boatlee29::Agent&a,const Simulator&s,int p,boatlee29::State&st){return pack_player(a.act(view(s,p),p,st));});
 m.def("batch_boatlee",[](std::vector<uint64_t>seeds,std::vector<int>seats,py::dict pd,const boatlee29::Agent&opponent,int threads){
  if(seeds.size()!=seats.size()||threads<1||threads>16)throw std::invalid_argument("Boatlee batch shape/threads");for(int seat:seats)if(seat<0||seat>1)throw std::invalid_argument("Boatlee seat");
  auto p=params(py::dict(pd.attr("copy")()));std::vector<BoatleeResult>results(seeds.size());std::vector<std::string>errors(seeds.size());
  {py::gil_scoped_release release;
   #pragma omp parallel for schedule(dynamic) num_threads(threads)
   for(int i=0;i<int(seeds.size());i++)try{results[i]=run_boatlee(seeds[i],seats[i],p,opponent);}catch(const std::exception&e){errors[i]=e.what();}
  }
  py::list out;for(size_t i=0;i<results.size();i++){if(!errors[i].empty())throw std::runtime_error(errors[i]);auto&x=results[i].match;py::dict d;d["seed"]=x.seed;d["seat"]=x.seat;d["steps"]=x.steps;d["cash"]=x.cash;d["opponent_cash"]=x.other;d["margin"]=x.cash-x.other;d["win"]=x.cash>x.other;d["seconds"]=x.seconds;d["overflow"]=x.overflow;d["opponent_switched"]=py::none();d["opponent_state"]=boatlee_debug(results[i].state);d["weed_repair_frames"]=results[i].weed_repair_frames;out.append(d);}return out;
 },py::arg("seeds"),py::arg("seats"),py::arg("params"),py::arg("opponent"),py::arg("threads")=16);
 m.def("fieldbook_input_fields",[](const Simulator&s,int p){return fieldbook::input_fields(view(s,p),p);});
 m.def("batch_fieldbook",[](std::vector<uint64_t>seeds,std::vector<int>seats,py::dict pd,int threads){
  if(seeds.size()!=seats.size()||threads<1||threads>16)throw std::invalid_argument("Fieldbook batch shape/threads");for(int seat:seats)if(seat<0||seat>1)throw std::invalid_argument("Fieldbook seat");
  auto p=params(py::dict(pd.attr("copy")()));std::vector<FieldbookResult>results(seeds.size());std::vector<std::string>errors(seeds.size());
  {py::gil_scoped_release release;
   #pragma omp parallel for schedule(dynamic) num_threads(threads)
   for(int i=0;i<int(seeds.size());i++)try{results[i]=run_fieldbook(seeds[i],seats[i],p);}catch(const std::exception&e){errors[i]=e.what();}
  }
  py::list out;for(size_t i=0;i<results.size();i++){if(!errors[i].empty())throw std::runtime_error(errors[i]);auto&x=results[i].match;py::dict d;d["seed"]=x.seed;d["seat"]=x.seat;d["steps"]=x.steps;d["cash"]=x.cash;d["opponent_cash"]=x.other;d["margin"]=x.cash-x.other;d["win"]=x.cash>x.other;d["seconds"]=x.seconds;d["overflow"]=x.overflow;d["opponent_switched"]=py::none();d["selected_segments"]=results[i].segments;out.append(d);}return out;
 },py::arg("seeds"),py::arg("seats"),py::arg("params")=py::dict(),py::arg("threads")=16);
 m.def("batch",[](std::vector<uint64_t>seeds,std::vector<int>seats,py::dict pd,const G001*g,int threads){if(seeds.size()!=seats.size()||threads<1||threads>16)throw std::invalid_argument("batch shape/threads");for(int s:seats)if(s<0||s>1)throw std::invalid_argument("seat");auto p=params(py::dict(pd.attr("copy")()));std::vector<Result>r(seeds.size());std::vector<std::string>errors(seeds.size());{py::gil_scoped_release release;
 #pragma omp parallel for schedule(dynamic) num_threads(threads)
 for(int i=0;i<int(seeds.size());i++){try{r[i]=run(seeds[i],seats[i],p,g);}catch(const std::exception&e){errors[i]=e.what();}}
 }py::list out;for(size_t i=0;i<r.size();i++){if(!errors[i].empty())throw std::runtime_error(errors[i]);auto&x=r[i];py::dict d;d["seed"]=x.seed;d["seat"]=x.seat;d["steps"]=x.steps;d["cash"]=x.cash;d["opponent_cash"]=x.other;d["margin"]=x.cash-x.other;d["win"]=x.cash>x.other;d["seconds"]=x.seconds;d["g001_switched"]=x.switched;d["overflow"]=x.overflow;out.append(d);}return out;},py::arg("seeds"),py::arg("seats"),py::arg("params")=py::dict(),py::arg("g001")=nullptr,py::arg("threads")=16);
 m.def("price",&dp7::price);
 m.def("investment_branch_batch",[](std::vector<uint64_t>seeds,std::vector<int>seats,py::dict pd,int kind,py::object opponent,std::vector<int>days,int threads,bool reverse_check){
  if(seeds.size()!=seats.size()||threads<1||threads>16)throw std::invalid_argument("branch batch shape/threads");
  for(int d:days)if(d<0||d>29)throw std::invalid_argument("branch day");
  for(int s:seats)if(s<0||s>1)throw std::invalid_argument("branch seat");
  AuditOpponentSpec spec;spec.kind=kind;
  switch(kind){case 0:case 5:case 6:case 7:if(!opponent.is_none())throw std::invalid_argument("unexpected branch asset");break;
   case 1:spec.route=opponent.cast<const G001*>();break;case 2:spec.boatlee=opponent.cast<const boatlee29::Agent*>();break;
   case 3:spec.kaito=opponent.cast<const kaito58::Agent*>();break;case 4:spec.lynn=opponent.cast<const lynn5::Agent*>();break;
   default:throw std::invalid_argument("unknown branch opponent");}
  auto p=params(py::dict(pd.attr("copy")()));std::vector<investment_branch_audit::Match>results(seeds.size());std::vector<std::string>errors(seeds.size());
  {py::gil_scoped_release release;
   #pragma omp parallel for schedule(dynamic) num_threads(threads)
   for(int i=0;i<int(seeds.size());i++)try{results[i]=investment_branch_audit::run(seeds[i],seats[i],p,spec,days,reverse_check);}catch(const std::exception&e){errors[i]=e.what();}
  }
  py::list out;for(size_t i=0;i<results.size();i++){
   if(!errors[i].empty())throw std::runtime_error(errors[i]);auto&r=results[i];py::dict row;py::list nodes;
   row["seed"]=r.seed;row["seat"]=r.seat;row["cash"]=r.cash;row["opponent_cash"]=r.rival_cash;row["seconds"]=r.seconds;row["continuations"]=r.continuations;row["transitions"]=r.transitions;
   for(auto&n:r.nodes){py::dict node;py::list choices;node["day"]=n.day;node["full_state_hash_OFFLINE_ONLY"]=n.state_hash;
    node["source_cash"]=n.source_cash;node["source_opponent_cash"]=n.source_rival_cash;node["keep_before_after"]=n.keep_before_after;node["reverse_equal"]=n.reverse_equal;
    for(auto&c:n.choices){py::dict x;x["family"]=c.family;x["kind"]=c.kind;x["amount"]=c.amount;x["score"]=c.score;x["proposed"]=c.proposed;x["conditionally_started"]=c.started;x["edits"]=c.edits;
     x["cash"]=c.result.cash;x["opponent_cash"]=c.result.rival_cash;x["suffix_hash_OFFLINE_ONLY"]=c.result.hash;x["steps"]=c.result.steps;choices.append(x);}
    node["choices"]=choices;nodes.append(node);
   }row["nodes"]=nodes;out.append(row);
  }return out;
 },py::arg("seeds"),py::arg("seats"),py::arg("params"),py::arg("kind"),py::arg("opponent")=py::none(),py::arg("days")=std::vector<int>{0,1,3,6,9,12,18,24},py::arg("threads")=16,py::arg("reverse_check")=false);
 m.def("declared_commitment_audit_batch",[](std::vector<uint64_t>seeds,std::vector<int>seats,py::dict pd,int kind,py::object opponent,int threads){
  if(seeds.size()!=seats.size()||threads<1||threads>16)throw std::invalid_argument("commitment audit shape/threads");
  for(int seat:seats)if(seat<0||seat>1)throw std::invalid_argument("commitment audit seat");
  AuditOpponentSpec spec;spec.kind=kind;
  switch(kind){case 0:case 5:case 6:case 7:if(!opponent.is_none())throw std::invalid_argument("unexpected audit asset");break;
   case 1:spec.route=opponent.cast<const G001*>();break;case 2:spec.boatlee=opponent.cast<const boatlee29::Agent*>();break;
   case 3:spec.kaito=opponent.cast<const kaito58::Agent*>();break;case 4:spec.lynn=opponent.cast<const lynn5::Agent*>();break;
   default:throw std::invalid_argument("unknown audit opponent");}
  struct Result{std::array<double,2>money{};std::array<int,2>overflow{};int checked_actions=0;std::vector<dp7::AdmissionInspection>samples;};
  auto pars=params(py::dict(pd.attr("copy")()));std::vector<Result>results(seeds.size());std::vector<std::string>errors(seeds.size());
  {py::gil_scoped_release release;
   #pragma omp parallel for schedule(dynamic) num_threads(threads)
   for(int i=0;i<int(seeds.size());i++)try{
    Simulator env(Config{},seeds[i]);dp7::Controller own(pars),shadow(pars);AuditOpponent rival(spec);
    auto same=[](const std::vector<Action>&a,const std::vector<Action>&b){return a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin(),dp7::Controller::same_action);};
    while(!env.done()){
     dp7::AdmissionInspection sample;own.admission_inspection=&sample;
     std::array<PlayerAction,2>a;a[seats[i]]=own.act(view(env,seats[i]));own.admission_inspection=nullptr;
     auto baseline=shadow.act(view(env,seats[i]));
     if(!same(a[seats[i]].units,baseline.units)||!same(a[seats[i]].market,baseline.market))throw std::logic_error("read-only audit changed live action");
     results[i].checked_actions++;if(sample.seen)results[i].samples.push_back(sample);
     a[1-seats[i]]=rival.act(env,1-seats[i]);env.step(a);
     for(int s=0;s<2;s++)results[i].overflow[s]+=env.last_end_of_day_overflow()[s];
    }for(int s=0;s<2;s++)results[i].money[s]=env.farms()[s].money;
   }catch(const std::exception&e){errors[i]=e.what();}
  }
  py::list out;for(size_t i=0;i<results.size();i++){
   if(!errors[i].empty())throw std::runtime_error(errors[i]);auto&r=results[i];py::dict row;py::list samples;
   row["seed"]=seeds[i];row["seat"]=seats[i];row["money"]=r.money;row["overflow"]=r.overflow;row["checked_actions"]=r.checked_actions;
   for(auto&s:r.samples){py::dict x;
    #define CS(n) x[#n]=s.n;
    CS(step) CS(base_valid) CS(alternative_valid) CS(base_kind) CS(alternative_kind) CS(base_pos) CS(alternative_pos) CS(base_unit) CS(alternative_unit)
    CS(planned) CS(selected) CS(feed) CS(duplicates) CS(late) CS(base_value) CS(alternative_value) CS(same_kind_before) CS(same_kind_after)
    #undef CS
    samples.append(x);
   }row["samples"]=samples;out.append(row);
  }return out;
 },py::arg("seeds"),py::arg("seats"),py::arg("params"),py::arg("kind"),py::arg("opponent")=py::none(),py::arg("threads")=16);
 m.def("resource_handoff_audit_batch",[](std::vector<uint64_t>seeds,std::vector<int>seats,py::dict pd,int kind,py::object opponent,int threads){
  if(seeds.size()!=seats.size()||threads<1||threads>16)throw std::invalid_argument("handoff audit shape/threads");
  for(int seat:seats)if(seat<0||seat>1)throw std::invalid_argument("handoff seat");
  AuditOpponentSpec spec;spec.kind=kind;
  switch(kind){case 0:case 5:case 6:case 7:if(!opponent.is_none())throw std::invalid_argument("unexpected audit asset");break;
   case 1:spec.route=opponent.cast<const G001*>();break;case 2:spec.boatlee=opponent.cast<const boatlee29::Agent*>();break;
   case 3:spec.kaito=opponent.cast<const kaito58::Agent*>();break;case 4:spec.lynn=opponent.cast<const lynn5::Agent*>();break;
   default:throw std::invalid_argument("unknown audit opponent");}
  struct Result{std::array<double,2>money{};std::vector<dp7audit::handoff::Sample>samples;};
  auto p=params(py::dict(pd.attr("copy")()));std::vector<Result>result(seeds.size());std::vector<std::string>errors(seeds.size());
  {py::gil_scoped_release release;
   #pragma omp parallel for schedule(dynamic) num_threads(threads)
   for(int i=0;i<int(seeds.size());i++)try{
    Simulator env(Config{},seeds[i]);dp7::Controller own(p);AuditOpponent rival(spec);std::array<bool,30>captured{};
    while(!env.done()){
     if(own.phase==3&&own.day==env.day()&&(!captured[env.day()]||env.hour()==6||env.hour()==12||env.hour()==18)){
      result[i].samples.push_back(dp7audit::handoff::inspect(own,env,seats[i]));captured[env.day()]=true;
     }
     std::array<PlayerAction,2>a;a[seats[i]]=own.act(view(env,seats[i]));a[1-seats[i]]=rival.act(env,1-seats[i]);env.step(a);
    }for(int s=0;s<2;s++)result[i].money[s]=env.farms()[s].money;
   }catch(const std::exception&e){errors[i]=e.what();}
  }
  py::list out;for(size_t i=0;i<result.size();i++){
   if(!errors[i].empty())throw std::runtime_error(errors[i]);py::dict row;py::list samples;
   row["seed"]=seeds[i];row["seat"]=seats[i];row["money"]=result[i].money;
   for(auto&s:result[i].samples){py::dict x;
    #define HS(n) x[#n]=s.n;
    HS(step) HS(day) HS(hour) HS(compatible) HS(feasible) HS(shorter) HS(total_saved) HS(peak_saved)
    HS(unit_a) HS(unit_b) HS(pos_a) HS(pos_b) HS(base_noop) HS(trial_noop) HS(base_resources) HS(base_deadline) HS(same_effect)
    #undef HS
    samples.append(x);
   }row["samples"]=samples;out.append(row);
  }return out;
 },py::arg("seeds"),py::arg("seats"),py::arg("params"),py::arg("kind"),py::arg("opponent")=py::none(),py::arg("threads")=16);
 m.def("pool_audit_batch",[](std::vector<uint64_t>seeds,std::vector<int>seats,py::dict pd,int kind,py::object opponent,int threads){
  if(seeds.size()!=seats.size()||threads<1||threads>16)throw std::invalid_argument("pool audit shape/threads");
  for(int seat:seats)if(seat<0||seat>1)throw std::invalid_argument("pool audit seat");
  AuditOpponentSpec spec;spec.kind=kind;
  switch(kind){case 0:case 5:case 6:case 7:if(!opponent.is_none())throw std::invalid_argument("unexpected audit asset");break;
   case 1:spec.route=opponent.cast<const G001*>();break;case 2:spec.boatlee=opponent.cast<const boatlee29::Agent*>();break;
   case 3:spec.kaito=opponent.cast<const kaito58::Agent*>();break;case 4:spec.lynn=opponent.cast<const lynn5::Agent*>();break;
   default:throw std::invalid_argument("unknown audit opponent");}
  auto p=params(py::dict(pd.attr("copy")()));std::vector<PoolAuditResult>result(seeds.size());std::vector<std::string>errors(seeds.size());
  {py::gil_scoped_release release;
   #pragma omp parallel for schedule(dynamic) num_threads(threads)
   for(int i=0;i<int(seeds.size());i++)try{result[i]=pool_audit(seeds[i],seats[i],p,spec);}catch(const std::exception&e){errors[i]=e.what();}
  }
  py::list out;for(size_t i=0;i<result.size();i++){
   if(!errors[i].empty())throw std::runtime_error(errors[i]);auto&r=result[i];py::dict row;
   row["seed"]=seeds[i];row["seat"]=seats[i];row["money"]=r.production.money;row["overflow"]=r.overflow;row["shops"]=r.shops;
   py::list production,cash,planning,admissions;
   for(int day=0;day<30;day++){py::list pp,cp;for(int side=0;side<2;side++){
    {auto&d=r.production.days[day][side];py::dict x;
     #define PROD(n) x[#n]=d.n;
     PROD(generated) PROD(acquired) PROD(used) PROD(unit_discard) PROD(environment_loss) PROD(drop_loss) PROD(eod_loss) PROD(bought) PROD(sold) PROD(end_private) PROD(end_field) PROD(planted) PROD(fertilized) PROD(watered) PROD(seed_bought) PROD(end_seeds) PROD(placed) PROD(fed) PROD(cared) PROD(escaped) PROD(attempts) PROD(no_effect) PROD(checks)
     #undef PROD
     pp.append(x);}
    {auto&d=r.cash[day][side];py::dict x;
     #define CASH(n) x[#n]=d.n;
     CASH(sales) CASH(products) CASH(seeds) CASH(animals) CASH(sold) CASH(bought_products) CASH(bought_seeds) CASH(bought_animals) CASH(hired) CASH(land) CASH(start) CASH(end) CASH(hires) CASH(lands) CASH(checks)
     #undef CASH
     cp.append(x);}
   }production.append(pp);cash.append(cp);auto&d=r.planning[day];py::dict x;
    #define PLAN(n) x[#n]=d.n;
    PLAN(day) PLAN(target) PLAN(live) PLAN(opponent_live) PLAN(land) PLAN(max_hands) PLAN(compile_drop) PLAN(degraded) PLAN(unfed) PLAN(unwatered)
    PLAN(bundle_evaluations) PLAN(bundle_switches) PLAN(bundle_removed_targets)
    PLAN(portfolio_generated) PLAN(portfolio_evaluated) PLAN(portfolio_switches) PLAN(portfolio_rejected)
    PLAN(preparation_actions)
    PLAN(shared_insertion_checks) PLAN(shared_insertion_applied) PLAN(shared_insertion_peak_saved) PLAN(shared_insertion_steps_saved)
    PLAN(resource_exchange_checks) PLAN(resource_exchange_pairs) PLAN(resource_exchange_applied) PLAN(resource_exchange_local_steps_saved)
    PLAN(intraday_checks) PLAN(intraday_proposals) PLAN(intraday_activated) PLAN(intraday_cancelled) PLAN(intraday_unfilled) PLAN(intraday_purchase_orders) PLAN(intraday_activated_by_kind)
    PLAN(service_checks) PLAN(service_recovered_feed) PLAN(service_recovered_fertilize) PLAN(service_buys) PLAN(service_financed) PLAN(service_deposits)
     PLAN(shared_service_plots) PLAN(regret_trials) PLAN(regret_improvements)
     PLAN(terminal_schedule_evaluations) PLAN(terminal_schedule_switches) PLAN(terminal_expected_cash_gain)
     PLAN(overflow_dispatch_checks) PLAN(overflow_dispatch_units) PLAN(overflow_dispatch_quantity)
    #undef PLAN
    planning.append(x);auto&ad=r.admissions[day];py::dict adict;
    #define ADMIT(n) adict[#n]=ad.n;
    ADMIT(captured) ADMIT(all_ready_jobs_fit) ADMIT(day) ADMIT(hour) ADMIT(planned_jobs) ADMIT(ready_jobs) ADMIT(resource_dropped_jobs)
    ADMIT(seed_shortage) ADMIT(feed_shortage) ADMIT(fertilizer_shortage) ADMIT(animal_shortage) ADMIT(hands) ADMIT(min_hands)
    ADMIT(unplaced_animals_in_shed) ADMIT(cash) ADMIT(conditional_wage_saving)
    #undef ADMIT
    admissions.append(adict);
   }row["production"]=production;row["cash_ledger"]=cash;row["planning"]=planning;row["admissions"]=admissions;py::list examples;
   for(auto&x:r.production.examples)examples.append(py::make_tuple(x.step,x.player,x.unit,x.op,x.position,x.item));row["no_effect_examples"]=examples;
   py::dict inv;py::list proposals,events,cohorts;
   for(auto&e:r.investments.proposals){py::dict x;
    #define IV(n) x[#n]=e.n;
    IV(step) IV(position) IV(kind) IV(current_kind) IV(first_output_day) IV(project_output) IV(capital) IV(standalone_quoted_value)
    #undef IV
    proposals.append(x);}
   for(auto&e:r.investments.events)events.append(py::make_tuple(e.step,int(e.op),e.item,e.quantity,e.unit,e.position));
   for(auto&e:r.investments.cohorts){py::dict x;
    #define IV(n) x[#n]=e.n;
    IV(kind) IV(position) IV(start_step) IV(first_output_state_step) IV(first_harvest_step) IV(end_state_step) IV(harvested) IV(output_seen)
    #undef IV
    cohorts.append(x);}
   inv["proposals"]=proposals;inv["events"]=events;inv["cohorts"]=cohorts;row["investments"]=inv;out.append(row);
  }return out;
 },py::arg("seeds"),py::arg("seats"),py::arg("params"),py::arg("kind"),py::arg("opponent")=py::none(),py::arg("threads")=16);
 m.def("production_audit_batch",[](std::vector<uint64_t>seeds,std::vector<int>seats,py::dict pd,const G001*g,int threads){
  if(seeds.size()!=seats.size()||threads<1||threads>16)throw std::invalid_argument("production batch shape/threads");for(int seat:seats)if(seat<0||seat>1)throw std::invalid_argument("production seat");
  auto p=params(py::dict(pd.attr("copy")()));std::vector<ProductionResult>result(seeds.size());std::vector<std::string>errors(seeds.size());
  {py::gil_scoped_release release;
   #pragma omp parallel for schedule(dynamic) num_threads(threads)
   for(int i=0;i<int(seeds.size());i++)try{result[i]=production_audit(seeds[i],seats[i],p,g);}catch(const std::exception&e){errors[i]=e.what();}
  }
  py::list out;for(size_t i=0;i<result.size();i++){
   if(!errors[i].empty())throw std::runtime_error(errors[i]);py::dict row;row["seed"]=seeds[i];row["seat"]=seats[i];row["money"]=result[i].money;py::list days,examples;
   for(int day=0;day<30;day++){py::list players;for(int side=0;side<2;side++){auto&d=result[i].days[day][side];py::dict x;
    #define PROD(n) x[#n]=d.n;
    PROD(generated) PROD(acquired) PROD(used) PROD(unit_discard) PROD(environment_loss) PROD(drop_loss) PROD(eod_loss) PROD(bought) PROD(sold) PROD(end_private) PROD(end_field) PROD(planted) PROD(fertilized) PROD(watered) PROD(seed_bought) PROD(end_seeds) PROD(placed) PROD(fed) PROD(cared) PROD(escaped) PROD(attempts) PROD(no_effect) PROD(checks)
    #undef PROD
    players.append(x);
   }days.append(players);}row["days"]=days;
   for(auto&x:result[i].examples)examples.append(py::make_tuple(x.step,x.player,x.unit,x.op,x.position,x.item));row["no_effect_examples"]=examples;out.append(row);
  }return out;
 },py::arg("seeds"),py::arg("seats"),py::arg("params")=py::dict(),py::arg("g001")=nullptr,py::arg("threads")=16);
 m.def("cash_audit_batch",[](std::vector<uint64_t>seeds,std::vector<int>seats,py::dict pd,const G001*g,int threads){
  if(seeds.size()!=seats.size()||threads<1||threads>16)throw std::invalid_argument("audit batch shape/threads");for(int seat:seats)if(seat<0||seat>1)throw std::invalid_argument("audit seat");
  auto p=params(py::dict(pd.attr("copy")()));std::vector<CashSeason>result(seeds.size());std::vector<std::string>errors(seeds.size());
  {py::gil_scoped_release release;
   #pragma omp parallel for schedule(dynamic) num_threads(threads)
   for(int i=0;i<int(seeds.size());i++)try{result[i]=cash_audit(seeds[i],seats[i],p,g);}catch(const std::exception&e){errors[i]=e.what();}
  }
  py::list out;for(size_t i=0;i<result.size();i++){
   if(!errors[i].empty())throw std::runtime_error(errors[i]);py::dict row;row["seed"]=seeds[i];row["seat"]=seats[i];py::list days;
   for(int day=0;day<30;day++){py::list players;for(int side=0;side<2;side++){auto&d=result[i][day][side];py::dict x;
    #define CASH(n) x[#n]=d.n;
    CASH(sales) CASH(products) CASH(seeds) CASH(animals) CASH(sold) CASH(bought_products) CASH(bought_seeds) CASH(bought_animals) CASH(hired) CASH(land) CASH(start) CASH(end) CASH(hires) CASH(lands) CASH(checks)
    #undef CASH
    players.append(x);
   }days.append(players);}row["days"]=days;out.append(row);
  }return out;
 },py::arg("seeds"),py::arg("seats"),py::arg("params")=py::dict(),py::arg("g001")=nullptr,py::arg("threads")=16);
 m.def("audit",[](uint64_t seed,int seat,py::dict pd,const G001*g){auto p=params(py::dict(pd.attr("copy")()));std::vector<DayAudit>days;{py::gil_scoped_release release;days=audit(seed,seat,p,g);}py::list out;for(auto&x:days){py::dict d;
 #define AUD(n) d[#n]=x.n;
 AUD(day) AUD(start_money) AUD(end_money) AUD(opponent_money) AUD(target) AUD(live) AUD(opponent_live) AUD(bought) AUD(sold) AUD(hired) AUD(max_hands) AUD(land) AUD(compile_drop) AUD(degraded) AUD(unfed) AUD(unwatered) AUD(overflow) AUD(shed) AUD(carried)
 #undef AUD
 out.append(d);}return out;},py::arg("seed"),py::arg("seat"),py::arg("params")=py::dict(),py::arg("g001")=nullptr);
}
