// Differential-test bridge only; never called from a C++ match loop.
#include "ecobot_v7_core.hpp"
#include "ecobot_v7.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
namespace py=pybind11;
using namespace eco7;
namespace {
const std::vector<std::string> names{"WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL","FERTILIZER","GOOSE","COW","SHEEP"};
const std::vector<std::string> ops{"PASS","NORTH","SOUTH","EAST","WEST","DROP","PICKUP","PLACE","PLANT","WATER","HARVEST","FERTILIZE","DIG","BUILD_COOP","BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE","HIRE","BUY_LAND","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL"};
template<class T>T get(py::dict d,const char*key,T def){return d.contains(key)?py::cast<T>(d[key]):def;}
int item(std::string s){auto it=std::find(names.begin(),names.end(),s);if(it==names.end())throw std::runtime_error("EcoBot unknown item "+s);return int(it-names.begin());}
Stock stock(py::dict d){Stock out{};for(auto kv:d)out[item(py::cast<std::string>(kv.first))]=py::cast<int>(kv.second);return out;}
Seeds seeds(py::dict d){Seeds out{};for(auto kv:d){int i=item(py::cast<std::string>(kv.first));if(i>=5)throw std::runtime_error("EcoBot nonseed");out[i]=py::cast<int>(kv.second);}return out;}
py::dict invdict(const Stock&s){py::dict d;for(int i=0;i<12;i++)if(s[i]>0)d[py::str(names[i])]=s[i];return d;}
py::list action(const Action&a){py::list x;x.append(ops.at(int(a.op)));if(a.item!=Item::NONE)x.append(names.at(int(a.item)));if(a.op==Op::PICKUP||a.op==Op::BUY_PRODUCT||a.op==Op::BUY_SEED||a.op==Op::BUY_ANIMAL||a.op==Op::SELL)x.append(a.quantity);return x;}
py::list actions(const std::vector<Action>&as){py::list r;for(auto&a:as)r.append(action(a));return r;}
py::object produces(Produce p){return p?py::object(py::make_tuple(names[p->first],p->second)):py::none();}
py::dict plant(const Plant&p){py::dict d;d["pos"]=p.pos;d["crop"]=names[p.crop];d["age"]=p.age;d["watered_today"]=p.watered;d["water_needed"]=p.water_needed;d["yield_units"]=p.yield;d["fertilize_due"]=p.fert_due;d["fert_until_day"]=p.fert_until;d["is_expired"]=p.expired;d["is_ongoing"]=ongoing(p.crop);d["first_yield_day"]=FIRST[p.crop];d["max_yield_day"]=MAXAGE[p.crop];return d;}
py::dict farm_state(const FarmState&f){py::dict d;py::list ps,as;for(auto&p:f.plants)ps.append(plant(p));for(auto&a:f.animals){py::dict t;t["pos"]=a.pos;t["animal"]=names[a.species];t["fed_today"]=a.fed;t["consecutive_unfed"]=a.unfed;t["cared_today"]=a.cared;t["fertilizer_available"]=a.fertilizer;t["yield_units"]=a.yield;as.append(t);}d["plants"]=ps;d["animals"]=as;d["empty_pastures"]=f.pastures;d["empty_coops"]=f.coops;d["weeds"]=f.weeds;d["empty_tiles"]=f.empty;d["unlocked_count"]=f.unlocked_count;return d;}
py::dict task(const Task&t){py::dict d;d["urgency"]=t.urgency;d["pos"]=t.pos;d["actions"]=actions(t.actions);py::list ns;for(int i:t.need)ns.append(names[i]);d["need"]=ns;d["produces"]=produces(t.produces);return d;}
py::dict route(const Route&r){py::dict d;d["start"]=r.start;d["cost"]=r.cost;d["carried"]=invdict(r.carried);py::dict pickups;for(int i=0;i<12;i++)if(r.pickup[i]>=0)pickups[py::str(names[i])]=r.pickup[i];d["pickup_of"]=pickups;d["insertable_upto"]=r.upto<0?py::none():py::cast(r.upto);d["insertable_from"]=r.from;py::list ss;for(auto&s:r.stops){py::dict x;x["pos"]=s.pos;x["actions"]=actions(s.actions);x["produces"]=produces(s.produces);ss.append(x);}d["stops"]=ss;return d;}
fastkag::Farm input_farm(py::dict me,int day){fastkag::Farm f;f.tiles.resize(100);f.money=py::cast<double>(me["money"]);auto raw=py::cast<py::list>(me["tiles"]);
  using fastkag::TileKind;for(int y=0;y<10;y++)for(int x=0;x<10;x++){py::object v=raw[y].cast<py::list>()[x];auto&t=f.tiles[y*10+x];if(v.is_none())continue;if(py::isinstance<py::str>(v)){if(v.cast<std::string>()!="LOCKED")throw std::runtime_error("EcoBot invalid tile string");t.kind=TileKind::LOCKED;continue;}
    auto d=v.cast<py::dict>();auto kind=py::cast<std::string>(d["kind"]);
    if(kind=="WEED")t.kind=TileKind::WEED;else if(kind=="COOP")t.kind=TileKind::COOP;else if(kind=="PASTURE")t.kind=TileKind::PASTURE;else if(kind=="PLANT")t.kind=TileKind::PLANT;else throw std::runtime_error("EcoBot unknown kind");
    if(d.contains("animal")&&!d["animal"].is_none()){t.animal=Item(item(py::cast<std::string>(d["animal"])));t.kind=TileKind::ANIMAL;}
    if(d.contains("crop")&&!d["crop"].is_none())t.crop=Item(item(py::cast<std::string>(d["crop"])));
    t.fed_today=get(d,"fed_today",false);t.consecutive_unfed=get(d,"consecutive_unfed",0);t.cared_today=get(d,"cared_today",false);t.fertilizer_available=get(d,"fertilizer_available",false);t.yield_units=get(d,"yield_units",0);t.planted_day=get(d,"planted_day",day);t.watered_today=get(d,"watered_today",false);t.fertilized_until_day=get(d,"fertilized_until_day",-1);
    if(t.kind==TileKind::PLANT&&!d.contains("consecutive_unwatered"))throw std::runtime_error("EcoBot missing dry counter");t.consecutive_unwatered=get(d,"consecutive_unwatered",0);
  }return f;
}
struct ProbeInput{
  int step,day,hour;fastkag::Farm farm;fastkag::PrivateState priv;fastkag::Market market;std::vector<int8_t>shops;
  explicit ProbeInput(py::dict obs){
    int seat=py::cast<int>(obs["player"]);step=py::cast<int>(obs["step"]);day=py::cast<int>(obs["day"]);hour=py::cast<int>(obs["hour"]);
    auto me=obs["farms"].cast<py::list>()[seat].cast<py::dict>();farm=input_farm(me,day);auto pos=py::cast<Pos>(me["farmer"]);farm.farmer={int16_t(pos.first),int16_t(pos.second)};
    for(auto h:me["hands"].cast<py::list>()){auto p=py::cast<Pos>(h);farm.hands.push_back({int16_t(p.first),int16_t(p.second)});}farm.hires_today=py::cast<int>(me["hires_today"]);farm.unlocked_mask=0;
    for(auto h:me["unlocked_quadrants"].cast<py::list>()){auto s=py::cast<std::string>(h);farm.unlocked_mask|=1<<(s=="NW"?0:s=="NE"?1:s=="SW"?2:3);}
    auto p=obs["private"].cast<py::dict>();priv.shed=stock(p["shed"].cast<py::dict>());priv.seeds=seeds(p["seeds"].cast<py::dict>());for(auto inv:p["inventories"].cast<py::list>())priv.inventories.push_back(stock(inv.cast<py::dict>()));
    auto m=obs["market"].cast<py::dict>();auto mi=stock(m["inventory"].cast<py::dict>()),mp=stock(m["prices"].cast<py::dict>());for(int i=0;i<9;i++){market.inventory[i]=mi[i];market.prices[i]=mp[i];}
    const std::vector<std::string> shopnames{"BAKERY","BRUNCH_SPOT","FARMERS_MARKET","ICE_CREAM_SHOP","PET_CAFE","PIZZA_SHOP","SMOOTHIE_SHOP","YARN_STORE"};
    for(auto h:obs["town"].cast<py::dict>()["unlocked_shops"].cast<py::list>()){auto s=py::cast<std::string>(h);auto it=std::find(shopnames.begin(),shopnames.end(),s);if(it==shopnames.end())throw std::runtime_error("EcoBot unknown shop");shops.push_back(int8_t(it-shopnames.begin()));}
  }
  Input view()const{return {step,day,hour,farm,priv,market,shops};}
};
py::dict decision(const Decision&d){py::dict out,h,crops,caps;for(int i=0;i<5;i++)if(d.hints.crops[i]>0)crops[py::str(names[i])]=d.hints.crops[i];h["reserved_grazer_slots"]=d.hints.grazers;h["reserved_geese_slots"]=d.hints.geese;h["crop_limits"]=crops;for(int sp:{COW,SHEEP,G})caps[py::str(names[sp])]=d.caps[sp];out["market_orders"]=actions(d.orders);out["hints"]=h;out["retained_caps"]=caps;out["needed_pastures"]=d.pastures;out["needed_coops"]=d.coops;return out;}
py::dict memory(const Controller&c){py::dict out,drift,cull,prev,observed,negative,retained;py::list downsized;
  for(int i=0;i<9;i++){prev[py::str(names[i])]=c.state.prev_inv[i];if(c.state.observed[i]!=0)observed[py::str(names[i])]=c.state.observed[i];}
  for(int sp:{COW,SHEEP,G}){negative[py::str(names[sp])]=c.state.negative[sp];retained[py::str(names[sp])]=c.state.retained[sp];if(c.state.downsized[sp])downsized.append(names[sp]);}
  drift["prev_inv"]=prev;drift["prev_day"]=c.state.prev_day;drift["observed"]=observed;cull["last_day"]=c.state.cull_day;cull["negative_days"]=negative;cull["downsized"]=downsized;cull["retained_caps"]=retained;out["drift"]=drift;out["cull"]=cull;out["plan_day"]=c.plan_day;py::dict qs;for(int i=0;i<int(c.queues.size());i++)qs[py::int_(i)]=actions(c.queues[i]);out["queues"]=qs;out["decision"]=decision(c.decision);return out;
}
py::dict kernel(py::dict obs,py::dict hint_input,py::dict cap_input,int pending){
  int seat=py::cast<int>(obs["player"]),day=py::cast<int>(obs["day"]),hour=py::cast<int>(obs["hour"]);auto me=obs["farms"].cast<py::list>()[seat].cast<py::dict>();auto priv=obs["private"].cast<py::dict>();auto sd=stock(priv["shed"].cast<py::dict>());auto seed=seeds(priv["seeds"].cast<py::dict>());auto inventories=priv["inventories"].cast<py::list>();std::vector<Unit>units;
  units.push_back({py::cast<Pos>(me["farmer"]),stock(inventories[0].cast<py::dict>())});auto hs=me["hands"].cast<py::list>();for(int i=0;i<int(hs.size());i++)units.push_back({py::cast<Pos>(hs[i]),stock(inventories[i+1].cast<py::dict>())});
  int quads=0;for(auto q:me["unlocked_quadrants"].cast<py::list>()){auto s=py::cast<std::string>(q);quads|=1<<(s=="NW"?0:s=="NE"?1:s=="SW"?2:3);}
  Hints h;h.grazers=py::cast<int>(hint_input["reserved_grazer_slots"]);h.geese=py::cast<int>(hint_input["reserved_geese_slots"]);h.crops=seeds(hint_input["crop_limits"].cast<py::dict>());auto caps=all_caps();for(auto kv:cap_input)caps[item(py::cast<std::string>(kv.first))]=py::cast<int>(kv.second);
  auto f=parse(input_farm(me,day),day);auto cc=census(f,sd,units);auto ps=needed_pastures(f,cc),cs=needed_coops(f,cc,ps);auto ts=catalog(f,sd,seed,ps,cs,units,day,hour,h,quads,caps);auto rp=ready_premium(f,day);
  int budget=std::max(0,24-std::max(1,hour)-(day==29?2:0));auto rs=solve(ts,rp,units,sd,std::vector<int>(units.size(),budget),day);
  py::dict d;d["farm_state"]=farm_state(f);py::dict c;
  for(auto [sp,label]:std::vector<std::pair<int,std::string>>{{COW,"cows"},{SHEEP,"sheep"},{G,"geese"}}){c[py::str("field_"+label)]=cc.field[sp];c[py::str("shed_"+label)]=cc.shed[sp];c[py::str("carried_"+label)]=cc.carried[sp];}
  d["census"]=c;d["needed_pastures"]=ps;d["needed_coops"]=cs;d["reserved"]=reserved(f,quads,h,day);d["feed_thresholds"]=feed_thresholds(cc,caps,day);d["fert_due_tomorrow"]=fert_due_tomorrow(f,day);d["wheat_maturing"]=maturing_tomorrow(f,W);d["carrot_maturing"]=maturing_tomorrow(f,C);py::list tt,pp,rr;
  for(auto&t:ts)tt.append(task(t));for(auto&p:rp)pp.append(plant(p));for(auto&r:rs)rr.append(route(r));d["catalog"]=tt;d["ready_premium"]=pp;d["routes"]=rr;
  d["required_hands"]=required_hands(ts,rp,units,sd,std::min(13,std::popcount(unsigned(quads))<3?11:13),day);
  auto plan=day_plan(units,f,sd,seed,day,hour,ps,cs,h,quads,caps,pending);py::dict qs;for(int i=0;i<int(plan.size());i++)qs[py::int_(i)]=actions(plan[i]);d["day_plan"]=qs;return d;
}
}
PYBIND11_MODULE(_eco7_probe,m){m.def("kernel",&kernel);m.def("market_price",&eco7::market_price);
  py::class_<Controller>(m,"EcoBot").def(py::init<>()).def("act",[](Controller&c,py::dict obs){ProbeInput input(obs);auto a=c.act(input.view());py::dict d;d["farmer"]=action(a.units.at(0));py::list hs;for(size_t i=1;i<a.units.size();i++)hs.append(action(a.units[i]));d["hands"]=hs;d["market"]=actions(a.market);return d;}).def("evaluate",[](Controller&c,py::dict obs){ProbeInput input(obs);c.decision=evaluate(input.view(),c.state);return decision(c.decision);}).def("debug",&memory);
}
