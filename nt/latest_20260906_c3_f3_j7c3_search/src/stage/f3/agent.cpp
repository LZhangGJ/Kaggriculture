// Independent Daily DP agent: faithful native port followed by isolated improvements.
// Original Python source SHA256: 7dc2b0fc4439f38343d3912e98182f6b504ab6badce210de2923ab34c66be2a6
// C++17, Boost.JSON (Boost Software License 1.0). No opponent policies or replay data.
#include <boost/json.hpp>
#include <boost/json/src.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <unordered_map>
namespace fusionledger {
#include "ledger_core.cpp"
}
namespace bj=boost::json;
using namespace std;
using J=bj::value;
using JA=bj::array;
using JO=bj::object;
constexpr int NP=9,NI=12,ND=30,NT=100;
const array<string,NI> names={
    "WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL","FERTILIZER","GOOSE","COW","SHEEP"
};
const array<int,NI> costs={
    10,20,50,100,80,0,0,0,0,300,400,500
};
const array<double,9> BASE={
    25,35,60,120,250,50,160,200,100
},THR={
    400,450,200,100,300,332,122,105,200
},LT={
    .8,1,.4,.7,.2,.4,.6,.2,.4
},HT={
    .2,.7,.6,1.6,3.6,.2,1.6,3.2,.4
};
const array<string,9> LO={
    "sqrt","hinge","hinge","sqrt","log","hinge","sqrt","log","linear"
},HI={
    "log","sqrt","sqrt","linear","sq","log","linear","sq","linear"
};
const array<int,20> FIB={
    1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765
};
const vector<pair<string,vector<int>>> SHOPS={
    {
        "BAKERY",{
            5,0
        }
    },{
        "PIZZA_SHOP",{
            6,2,0
        }
    },{
        "BRUNCH_SPOT",{
            5,0,3
        }
    },{
        "YARN_STORE",{
            7,7
        }
    },{
        "ICE_CREAM_SHOP",{
            3,6,0
        }
    },{
        "PET_CAFE",{
            1,1
        }
    },{
        "SMOOTHIE_SHOP",{
            3,6
        }
    },{
        "FARMERS_MARKET",{
            0,1,2,3
        }
    }
};
struct Animal{
    int prod,first,interval,cap,kind;
};
const array<Animal,3> ANI={
    Animal{
        5,4,1,4,3
    },Animal{
        6,8,2,6,4
    },Animal{
        7,6,3,6,4
    }
};
using Stock=array<int,NI>;
using Flow=array<array<double,NP>,ND>;
int itemid(const string&s){
    for(int i=0;i<NI;i++)if(names[i]==s)return i;
    return -1;
}
double num(const J&v){
    if(v.is_int64())return v.as_int64();
    if(v.is_uint64())return v.as_uint64();
    if(v.is_double())return v.as_double();
    if(v.is_bool())return v.as_bool();
    return 0;
}
const J& get(const J&j,const char*k){
    static const J empty;
    if(!j.is_object())return empty;
    auto p=j.as_object().if_contains(k);
    return p?*p:empty;
}
double number(const J&j,const char*k,double def=0){
    const J&v=get(j,k);
    return v.is_null()?def:num(v);
}
string text(const J&j,const char*k,string def=""){
    const J&v=get(j,k);
    return v.is_string()?string(v.as_string()):def;
}
Stock stock(const J&v){
    Stock s{
    };
    if(v.is_object())for(auto&kv:v.as_object()){
        int i=itemid(string(kv.key()));
        if(i>=0)s[i]=int(num(kv.value()));
    }
    return s;
}
int sumstock(const Stock&s){
    return accumulate(s.begin(),s.end(),0);
}
int xy(int x,int y){
    return y*10+x;
}
int X(int p){
    return p%10;
}
int Y(int p){
    return p/10;
}
int dist(int a,int b){
    return abs(X(a)-X(b))+abs(Y(a)-Y(b));
}
int access(int p){
    return xy(clamp(X(p),4,5),clamp(Y(p),4,5));
}
int pos(const J&v){
    return xy(int(num(v.as_array()[0])),int(num(v.as_array()[1])));
}
JA jpos(int p){
    return JA{
        X(p),Y(p)
    };
}
JA moveact(int p,int q){
    if(X(p)<X(q))return JA{
        "EAST"
    };
    if(X(p)>X(q))return JA{
        "WEST"
    };
    if(Y(p)<Y(q))return JA{
        "SOUTH"
    };
    if(Y(p)>Y(q))return JA{
        "NORTH"
    };
    return JA{
        "PASS"
    };
}
struct Tile{
    int kind=0,item=-1,placed=0,yield=0,unwatered=0,unfed=0,pending=0,fertilized=-1,lifespan=-1;
    bool water=false,feed=false,care=false,manure=false;
    bool animal()const{
        return item>=9;
    }
    bool plant()const{
        return kind==2;
    }
    bool active()const{
        return plant()||animal();
    }
};
struct PriceParam{
    double b=0,t=0,z=10000,lt=0,ht=0;
    string lo,hi;
};
struct Obs{
    int day=0,hour=0,step=0,seat=0,unlocked=1;
    double money=0;
    array<Tile,NT> tiles;
    array<Tile,NT> opponent_tiles;
    vector<int> positions;
    vector<Stock> invs;
    Stock shed{
    },seeds{
    };
    array<double,NP> market{
    },prices{
    };
    array<PriceParam,NP> params;
    vector<string> shops;
};
struct Config{
    int turns=24,shopinterval=4,unlockinterval=3,handmult=1,orderlimit=10,shedcap=100;
};
struct Settings{
    double capital_power=.6,labor_price=3.,land_rent=5.,future_shop_weight=1.,opening_reserve=80.,advance_revenue=.7,crop_work_mult=1.,animal_work_mult=1.,labor_hours=17.;
    int max_hands=15;
    bool crop_repeat=true,fertilize=true,animal_care=true,labor_capacity=true,delivery_planning=true;
    // Raw Agent defaults preserve port semantics for isolated tests.
    // The public Context below installs the frozen release defaults.
    int version=0,animal_dp=0,market_wait=0,storage_cap=80,route_nodes=9,route_budget=20,route_terminal=19,anchor_mode=0,replant_hour=-1,cycle_gap=1,portfolio_passes=0,team_repair=0,team_slack=0,terminal_extra=0,rescue_hour=24,rescue_water=0;
    double route_value=0,early_discount=1,discount_wealth=10000;
    double animal_labor=3,expand_limit=20,feed_weight=1.,care_weight=1.,manure_weight=1.;
    double opponent_supply_weight=0.,competition_weight=0.,opponent_feed_weight=0.;
    double sale_competition_weight=0.;
    int opponent_delivery_delay=1,competition_horizon=30;
    int portfolio_dp=0,portfolio_budget=1200;
    int dynamic_replant=0;
    // Independent F3 economic/execution switches. Defaults preserve F2 exactly.
    int harvest_early=0,harvest_first=0;
    double rival_lead=0.,courier_value=0.;
    int sale_order=0,retire_spent=0,capacity_guard=0,late_hire=0;
    double late_hire_margin=1.;
    double harvest_cash=1000.,harvest_ratio=1.05;
    double rival_renewal=0.,rival_decay=1.,manure_floor=0.;
    int c2_service=0,c2_service_refresh=0,c2_interleaved=0;
    void update(const J&j){
        if(!j.is_object())return;
#define SET(n) if(!get(j,#n).is_null())n=number(j,#n)
        SET(capital_power);
        SET(labor_price);
        SET(land_rent);
        SET(future_shop_weight);
        SET(opening_reserve);
        SET(advance_revenue);
        SET(crop_work_mult);
        SET(animal_work_mult);
        SET(labor_hours);
        SET(max_hands);
        SET(crop_repeat);
        SET(fertilize);
        SET(animal_care);
        SET(labor_capacity);
        SET(delivery_planning);
        SET(version);
        SET(animal_dp);
        SET(market_wait);
        SET(storage_cap);
        SET(route_nodes);
        SET(route_budget);
        SET(route_terminal);
        SET(route_value);
        SET(anchor_mode);
        SET(early_discount);
        SET(discount_wealth);
        SET(replant_hour);
        SET(cycle_gap);
        SET(portfolio_passes);
        SET(team_repair);
        SET(team_slack);
        SET(terminal_extra);
        SET(rescue_hour);
        SET(rescue_water);
        SET(animal_labor);
        SET(expand_limit);
        SET(feed_weight);
        SET(care_weight);
        SET(manure_weight);
        SET(opponent_supply_weight);SET(competition_weight);SET(opponent_feed_weight);
        SET(sale_competition_weight);SET(opponent_delivery_delay);SET(competition_horizon);
        SET(portfolio_dp);SET(portfolio_budget);
        SET(dynamic_replant);
        SET(harvest_early);SET(harvest_first);SET(rival_lead);SET(courier_value);
        SET(sale_order);SET(retire_spent);SET(harvest_cash);SET(harvest_ratio);
        SET(rival_renewal);SET(rival_decay);SET(manure_floor);
        SET(capacity_guard);
        SET(late_hire);SET(late_hire_margin);
        SET(c2_service);SET(c2_service_refresh);SET(c2_interleaved);
#undef SET
    }
};
Obs parseobs(const J&j){
    Obs o;
    o.day=number(j,"day");
    o.hour=number(j,"hour");
    o.step=number(j,"step",o.day*24+o.hour);
    o.seat=number(j,"player");
    auto&f=get(j,"farms").as_array().at(o.seat);
    o.money=number(f,"money");
    o.unlocked=get(f,"unlocked_quadrants").as_array().size();
    o.positions.push_back(pos(get(f,"farmer")));
    for(auto&p:get(f,"hands").as_array())o.positions.push_back(pos(p));
    auto&tt=get(f,"tiles").as_array();
    if(tt.size()!=10)throw runtime_error("Only the default 10x10 board is supported");
    for(int y=0;y<10;y++)for(int x=0;x<10;x++){
        auto&v=tt[y].as_array()[x];
        auto&t=o.tiles[xy(x,y)];
        if(v.is_string()){
            t.kind=-1;
            continue;
        }
        if(v.is_null())continue;
        string k=text(v,"kind");
        t.kind=k=="PLANT"?2:k=="COOP"?3:k=="PASTURE"?4:1;
        t.item=itemid(text(v,t.kind==2?"crop":"animal"));
        t.placed=number(v,t.kind==2?"planted_day":"placed_day");
        t.yield=number(v,"yield_units");
        t.unwatered=number(v,"consecutive_unwatered");
        t.unfed=number(v,"consecutive_unfed");
        t.pending=number(v,"pending_care_bonus");
        t.fertilized=number(v,"fertilized_until_day",-1);
        t.lifespan=number(v,"max_lifespan_step",-1);
        t.water=number(v,"watered_today");
        t.feed=number(v,"fed_today");
        t.care=number(v,"cared_today");
        t.manure=number(v,"fertilizer_available");
    }
    auto&ot=get(get(j,"farms").as_array().at(1-o.seat),"tiles").as_array();
    for(int y=0;y<10;y++)for(int x=0;x<10;x++){
        auto&v=ot[y].as_array()[x];auto&t=o.opponent_tiles[xy(x,y)];
        if(v.is_string()){t.kind=-1;continue;}if(v.is_null())continue;
        string k=text(v,"kind");t.kind=k=="PLANT"?2:k=="COOP"?3:k=="PASTURE"?4:1;
        t.item=itemid(text(v,t.kind==2?"crop":"animal"));t.placed=number(v,t.kind==2?"planted_day":"placed_day");
        t.yield=number(v,"yield_units");t.unwatered=number(v,"consecutive_unwatered");t.unfed=number(v,"consecutive_unfed");
        t.pending=number(v,"pending_care_bonus");t.fertilized=number(v,"fertilized_until_day",-1);
        t.lifespan=number(v,"max_lifespan_step",-1);t.water=number(v,"watered_today");t.feed=number(v,"fed_today");
        t.care=number(v,"cared_today");t.manure=number(v,"fertilizer_available");
    }
    auto&p=get(j,"private");
    o.shed=stock(get(p,"shed"));
    o.seeds=stock(get(p,"seeds"));
    for(auto&i:get(p,"inventories").as_array())o.invs.push_back(stock(i));
    auto&m=get(j,"market");
    for(int i=0;i<NP;i++){
        o.market[i]=number(get(m,"inventory"),names[i].c_str());
        o.prices[i]=number(get(m,"prices"),names[i].c_str());
        auto&pp=get(get(m,"params"),names[i].c_str());
        o.params[i]={
            number(pp,"base",BASE[i]),number(pp,"T",THR[i]),number(pp,"I0",10000),number(pp,"below_target",LT[i]),number(pp,"above_target",HT[i]),text(pp,"below_func",LO[i]),text(pp,"above_func",HI[i])
        };
    }
    for(auto&s:get(get(j,"town"),"unlocked_shops").as_array())o.shops.push_back(string(s.as_string()));
    return o;
}
Config parsecfg(const J&j){
    Config c;
    c.turns=number(j,"turnsPerDay",24);
    c.shopinterval=number(j,"townShopSellInterval",4);
    c.unlockinterval=number(j,"townShopUnlockInterval",3);
    c.handmult=number(j,"farmHandCostMult",1);
    c.orderlimit=number(j,"maxMarketOrdersPerTurn",10);
    c.shedcap=number(j,"shedCapacity",100);
    return c;
}
double shape(const string&f,double x,double t){
    if(f=="sqrt")return sqrt(x);
    if(f=="sq")return x*x;
    if(f=="log")return log1p(x);
    if(f=="log10")return log10(1+x);
    if(f=="hinge"){
        double u=x/t;
        return u+8*pow(max(0.,u-1),2);
    }
    return x;
}
int price(int i,double inv,const Obs&o){
    auto&p=o.params[i];
    bool below=inv<p.z;
    auto&fn=below?p.lo:p.hi;
    double fac=below?p.lt:p.ht;
    return max(1,int(nearbyint(p.b+(below?1:-1)*p.b*fac*shape(fn,abs(inv-p.z),p.t)/shape(fn,p.t,p.t))));
}
void addflow(Flow&a,const Flow&b,double scale=1){
    for(int d=0;d<ND;d++)for(int i=0;i<NP;i++)a[d][i]+=scale*b[d][i];
}
struct Event{
    int age,item,q;
};
struct Variant{
    int length=0;
    vector<Event> events;
    double work=0;
    int nferts=0;
};
vector<Variant> cycles(int crop,bool fertilize=true){
    vector<Variant> out;
    if(crop==0||crop==1){
        for(int len=2;len<=(crop==0?4:3);len++)for(int f=0;f<=(fertilize?1:0);f++){
            int q=min(crop==0?6:4,1+(len-1)*(f?2:1));
            Variant v{
                len,{
                    {
                        len,crop,q
                    }
                },double(2+len+f),f
            };
            if(f)v.events.push_back({
                2,8,-1
            });
            out.push_back(v);
        }
    }
    else if(crop==4)out.push_back({
        10,{
            {
                10,4,6
            }
        },10,0
    });
    else {
        vector<int> ages=crop==2?vector<int>{
            8,9,10,11
        }
        :vector<int>{
            10,12,14,16
        };
        vector<vector<int>> sched={
            {
            }
        };
        if(fertilize){
            sched.push_back({
                ages[0]-1
            });
            sched.push_back(crop==2?vector<int>{
                7,10
            }
            :vector<int>{
                9,13
            });
        }
        for(auto&fs:sched){
            Variant v{
                ages.back(),{
                },2+ages.back()*.6+fs.size()+4,int(fs.size())
            };
            for(int f:fs)v.events.push_back({
                f,8,-1
            });
            for(int a:ages){
                bool bonus=false;
                for(int f:fs)if(f<=a-1&&a-1<=f+2)bonus=true;
                v.events.push_back({
                    a,crop,1+int(bonus)
                });
            }
            out.push_back(v);
        }
    }
    return out;
}
Flow animalflow(int item,int placed,int day,const Tile*t,bool care){
    Flow f{
    };
    auto a=ANI[item-9];
    int pending=t?t->pending:0;
    if(t){
        f[day][a.prod]+=t->yield;
        f[day][8]+=t->manure;
    }
    for(int d=day;d<29;d++)if(d>=placed){
        f[d][0]-=1;
        if(d>day||!t)f[d][8]+=d>placed;
        int age=d+1-placed;
        if(age>=a.first&&(age-a.first)%a.interval==0){
            f[d+1][a.prod]+=min(a.cap,1+pending);
            pending=0;
        }
        if(care)pending++;
    }
    if(29>max(day,placed))f[29][8]+=1;
    return f;
}
Flow cropflow(int crop,int placed,int day,const Variant&v,const Tile*t=nullptr){
    Flow f{
    };
    if(t){
        int y=t->yield;
        if(crop==2||crop==3)f[day][crop]+=y;
        for(auto e:v.events){
            int d=placed+e.age,q=e.q;
            if(d<day||d>29)continue;
            if(q<0&&t->fertilized>=d)continue;
            if(e.item==crop&&(crop==0||crop==1||crop==4)){
                int age=day-placed;
                if(age>e.age)continue;
                q=y;
                for(int dd=day;dd<=d;dd++){
                    int aa=dd-placed;
                    bool window=(crop==0&&2<=aa&&aa<=4)||(crop==1&&2<=aa&&aa<=3)||(crop==4&&6<=aa&&aa<=12);
                    if(window){
                        if(dd==day&&t->water)continue;
                        bool fert=t->fertilized>=dd;
                        for(auto ev:v.events)if(ev.item==8&&ev.q<0&&ev.age<=aa&&aa<=ev.age+2)fert=true;
                        q=min(crop==1?4:6,q+1+int(fert));
                    }
                }
            }
            if(e.item==crop&&(crop==2||crop==3)&&d==day)continue;
            f[d][e.item]+=q;
        }
    }
    else for(auto e:v.events){
        int d=placed+e.age;
        if(day<=d&&d<=29)f[d][e.item]+=e.q;
    }
    return f;
}
struct CropPlan{
    Flow flow{
    };
    double cost=0,work=0;
    Variant variant;
    bool valid=false;
};
// Conditional public production calendar, not an opponent replay or private
// inventory forecast. No renewal, unseen assets, future shops or actual seed.
Flow public_rival_flow(const Obs&o,const Settings&s){
    Flow result{};
    for(const auto&t:o.opponent_tiles){
        Flow f{};
        if(t.animal()){
            f=animalflow(t.item,t.placed,o.day,&t,true);
            for(int d=o.day;d<ND;d++)f[d][0]*=s.opponent_feed_weight;
        }else if(t.plant()){
            auto choices=cycles(t.item,false);
            auto v=choices.back();f=cropflow(t.item,t.placed,o.day,v,&t);
            // Conditional renewal, not knowledge of the opponent's next plan.
            if(s.rival_renewal>0){
                int next=max(o.day+1,t.placed+v.length+1);
                for(int start=next;start<ND;start+=v.length+1){
                    auto nf=cropflow(t.item,start,o.day,v);
                    for(int d=start;d<ND;d++)for(int i=0;i<NP;i++)if(nf[d][i]>0)f[d][i]+=s.rival_renewal*nf[d][i];
                }
            }
        }else continue;
        for(int d=o.day;d<min(ND,o.day+s.competition_horizon);d++)for(int i=0;i<NP;i++){
            int when=d+(f[d][i]>0?s.opponent_delivery_delay:0);
            if(when<ND)result[when][i]+=f[d][i]*pow(clamp(s.rival_decay,0.,1.),d-o.day);
        }
    }
    return result;
}
struct Economy{
    const Obs&o;
    const Config&c;
    const Settings&s;
    Flow dem{
    };
    Flow rival{};
    array<double,NP> stocks{
    };
    Economy(const Obs&oo,const Config&cc,const Settings&ss):o(oo),c(cc),s(ss){
        if(s.opponent_supply_weight>0||s.competition_weight>0)rival=public_rival_flow(o,s);
        array<double,NP> known{
        },avg{
        };
        double daily=double(c.turns)/c.shopinterval;
        for(auto&sh:o.shops)for(auto&kv:SHOPS)if(kv.first==sh)for(int p:kv.second)known[p]+=daily;
        for(auto&kv:SHOPS)for(int p:kv.second)avg[p]+=daily/SHOPS.size();
        for(int d=0;d<ND;d++){
            int nn=max(0,min(8,d/c.unlockinterval)-int(o.shops.size()));
            for(int i=0;i<NP;i++)dem[d][i]=int(i<8)+known[i]+nn*avg[i]*s.future_shop_weight;
        }
        for(int i=0;i<NP;i++){
            stocks[i]=o.shed[i];
            for(auto&iv:o.invs)stocks[i]+=iv[i];
        }
    }
    pair<double,Flow> value(const Flow&f)const{
        auto inv=o.market;
        double cash=0;
        Flow px{
        };
        for(int d=o.day;d<ND;d++)for(int i=0;i<NP;i++){
            double q=f[d][i]+(d==o.day?stocks[i]:0);
            if(s.c2_interleaved){
                // C2's conditional market model: half of the visible opponent's
                // estimated supply before us, half after. Not their actual order.
                inv[i]-=dem[d][i]*.5;
                auto trade=[&](double quantity){
                    double pp=price(i,inv[i]+quantity*.5,o);
                    if(abs(quantity)>2)pp=(5*price(i,inv[i]+quantity*.1127016654,o)+8*pp+5*price(i,inv[i]+quantity*.8872983346,o))/18.;
                    if(quantity<0||pp>1.001)inv[i]+=quantity;
                    return quantity*pp;
                };
                double r=s.opponent_supply_weight*rival[d][i]*.5;
                double other=trade(r),ours=trade(q);other+=trade(r);
                cash+=(ours-s.competition_weight*other)*pow(s.early_discount+(1-s.early_discount)*min(1.,o.money/s.discount_wealth),d-o.day);
                inv[i]-=dem[d][i]*.5;px[d][i]=price(i,inv[i],o);
                continue;
            }
            inv[i]-=dem[d][i]*.5;
            // Do not assume all our supply always reaches the market first.
            // Expected rival flow is derived only from currently visible assets.
            double lead=s.opponent_supply_weight>0?max(0.,rival[d][i]*s.opponent_supply_weight)*clamp(s.rival_lead,0.,1.):0.;
            if(lead>0&&price(i,inv[i]+lead*.5,o)>1.001)inv[i]+=lead;
            else lead=0;
            double mid=inv[i]+q*.5,pp=price(i,mid,o);
            if(abs(q)>2)pp=(price(i,inv[i]+q*.1127017,o)*5+price(i,mid,o)*8+price(i,inv[i]+q*.8872983,o)*5)/18.;
            cash+=q*pp*pow(s.early_discount+(1-s.early_discount)*min(1.,o.money/s.discount_wealth),d-o.day);
            inv[i]+=q;
            if(q>0&&pp<=1.001)inv[i]-=q;
            if(s.opponent_supply_weight>0){
                double r=rival[d][i]*s.opponent_supply_weight;
                double rival_price=price(i,inv[i]+r*.5,o);
                cash-=s.competition_weight*r*rival_price*pow(s.early_discount+(1-s.early_discount)*min(1.,o.money/s.discount_wealth),d-o.day);
                if(r<0||rival_price>1.001)inv[i]+=r-lead;
                else inv[i]-=lead;
            }
            inv[i]-=dem[d][i]*.5;
            px[d][i]=price(i,inv[i],o);
        }
        return {
            cash,px
        };
    }
    CropPlan newcrop(int crop,const Flow&px,int start=-1)const{
        if(start<0)start=o.day;
        array<double,32> values{
        };
        array<int,30> choices;
        choices.fill(-1);
        auto vs=cycles(crop,s.fertilize);
        for(int d=29;d>=start;d--){
            double best=0;
            for(int k=0;k<(int)vs.size();k++){
                auto&v=vs[k];
                bool sell=false;
                for(auto e:v.events)if(e.q>0&&d+e.age<=29)sell=true;
                if(!sell)continue;
                int end=min(29,d+v.length);
                double val=-costs[crop]-s.labor_price*v.work-s.land_rent*(end-d);
                for(auto e:v.events)if(d+e.age<=29)val+=e.q*px[d+e.age][e.item];
                if(s.crop_repeat)val+=values[end+s.cycle_gap];
                if(val>best){
                    best=val;
                    choices[d]=k;
                }
            }
            values[d]=best;
        }
        CropPlan p;
        if(choices[start]<0)return p;
        p.valid=true;
        p.variant=vs[choices[start]];
        p.flow=cropflow(crop,start,o.day,p.variant);
        p.cost=costs[crop];
        p.work=p.variant.work;
        if(s.crop_repeat){
            int d=start+p.variant.length+s.cycle_gap;
            while(d<=29&&choices[d]>=0){
                auto&v=vs[choices[d]];
                addflow(p.flow,cropflow(crop,d,o.day,v));
                p.cost+=costs[crop];
                p.work+=v.work;
                d+=v.length+s.cycle_gap;
            }
        }
        return p;
    }
};
#include "project_forecast_ledger.hpp"
struct Memory{
    int day=-1,hire_target=3;
    int fertilizer_deficit=0;
    vector<pair<int,int>> projects;
    array<int,NT> newitem,oldcrop;
    array<Variant,NT> policy;
    array<bool,NT> haspolicy{
    },animalfeed{
    },animalcare{
    };
    Flow forecast{
    },prices{
    };
    ProjectForecastLedger forecast_ledger;
    bool land=false,scheduled=false;
    vector<vector<int>> routes;
    vector<int> backlog;
    set<int> couriers;
    JA daily_records;
    int c2_service_calls=0,c2_refresh_changes=0;
    Memory(){
        newitem.fill(-1);
        oldcrop.fill(-1);
    }
    void setprojects(vector<pair<int,int>> p){
        projects=std::move(p);
        newitem.fill(-1);
        for(auto [p,i]:projects)newitem[p]=i;
    }
};
struct Op{
    string name;
    int res=-1;
    double value=0;
    bool mandatory=false;
};
struct Agent{
    Settings s;
    Memory m;
    struct AnimalPlan {
        Flow flow{
        };
        bool feed=true,care=true;
        double meanwork=0;
    };
    AnimalPlan c2animalplan(int item,int placed,int day,const Tile*t,const Flow&px)const{
        // Same finite-state recurrence as C2 AnimalServiceDP, adapted to this
        // planner's Flow and explicit projected output. No simulator access.
        AnimalPlan out;auto a=ANI[item-9];
        double values[31][2][7]{};int choices[30][2][7]{};
        for(int d=28;d>=day;--d){
            int age=d+1-placed;bool tick=age>=a.first&&(age-a.first)%a.interval==0;
            for(int h=0;h<2;h++)for(int p=0;p<a.cap;p++){
                double best=-1e100;int chosen=0;
                for(int act=0;act<=2;act++){
                    bool feed=act>=1,care=act==2;if(care&&!s.animal_care)continue;
                    int nh=feed?0:h+1;
                    if(nh>=2){if(0>best){best=0;chosen=0;}continue;}
                    if(care&&p>=a.cap-1&&!tick)continue;
                    int qty=tick?1+(feed?p:0):0,np=min(a.cap-1,(tick?0:p)+int(care));
                    double v=-int(feed)*(px[d][0]+s.animal_labor)-int(care)*s.animal_labor
                        +qty*px[d+1][a.prod]+max(0.,px[d+1][8]-s.animal_labor)+values[d+1][nh][np];
                    if(v>best+1e-9){best=v;chosen=act;}
                }
                values[d][h][p]=best;choices[d][h][p]=chosen;
            }
        }
        int h=t?min(1,t->unfed):0,p=t?min(a.cap-1,t->pending):0;
        if(t){out.flow[day][a.prod]+=t->yield;out.flow[day][8]+=t->manure;}
        for(int d=day;d<29;d++){
            int act=choices[d][h][p];bool feed=act>=1,care=act==2;
            if(d==day){out.feed=feed;out.care=care;}
            int nh=feed?0:h+1;if(nh>=2)break; // Planned exit; do not credit future output.
            if(feed)out.flow[d][0]-=1;
            out.flow[d+1][8]+=1;
            int age=d+1-placed;bool tick=age>=a.first&&(age-a.first)%a.interval==0;
            if(tick){out.flow[d+1][a.prod]+=1+(feed?p:0);p=0;}
            if(care)p=min(a.cap-1,p+1);h=nh;
            out.meanwork+=int(feed)+int(care)+1+1./a.interval;
        }
        if(day>=29)out.feed=out.care=false;
        out.meanwork/=max(1,29-day);return out;
    }
    AnimalPlan animalplan(int item,int placed,int day,const Tile*t,const Flow&px)const{
        if(s.c2_service)return c2animalplan(item,placed,day,t,px);
        AnimalPlan out;
        if(!s.animal_dp){
            out.flow=animalflow(item,placed,day,t,s.animal_care);
            return out;
        }
        const auto a=ANI[item-9];
        // Finite-horizon Bellman DP: state = morning, days unfed, pending care.
        // Production consumes yesterday's banked care BEFORE today's care is added.
        // No future random shop sequence, seed, opponent private state or replay is used.
        double dp[31][2][7]{
        };
        int choice[30][2][7]{
        };
        for(int d=28;d>=day;--d){
            int age=d+1-placed;
            bool tick=age>=a.first&&(age-a.first)%a.interval==0;
            for(int u=0;u<2;u++)for(int pending=0;pending<=a.cap-1;pending++){
                double best=-1e100;
                int ba=0;
                for(int act=0;act<=2;act++){
                    bool feed=act>=1,care=act==2;
                    if(care&&!s.animal_care)continue;
                    if(!feed&&u==1)continue;
                    int unext=feed?0:u+1,pnext=tick?0:pending;
                    int prod=tick?min(a.cap,1+(feed?pending:0)):0;
                    if(care)pnext=min(a.cap-1,pnext+1);
                    double val=prod*px[d+1][a.prod]-int(feed)*(px[d][0]+s.animal_labor)-int(care)*s.animal_labor+dp[d+1][unext][pnext];
                    if(val>best+1e-9){
                        best=val;
                        ba=act;
                    }
                }
                dp[d][u][pending]=best;
                choice[d][u][pending]=ba;
            }
        }
        int u=t?min(1,t->unfed):0,pending=t?min(a.cap-1,t->pending):0;
        if(t){
            out.flow[day][a.prod]+=t->yield;
            out.flow[day][8]+=t->manure;
        }
        for(int d=day;d<29;d++){
            int act=choice[d][u][pending];
            bool feed=act>=1,care=act==2;
            if(d==day){
                out.feed=feed;
                out.care=care;
            }
            if(feed)out.flow[d][0]-=1;
            if(d>day||!t)out.flow[d][8]+=d>placed;
            int age=d+1-placed;
            bool tick=age>=a.first&&(age-a.first)%a.interval==0;
            if(tick){
                out.flow[d+1][a.prod]+=min(a.cap,1+(feed?pending:0));
                pending=0;
            }
            if(care)pending=min(a.cap-1,pending+1);
            u=feed?0:u+1;
            out.meanwork+=int(feed)+int(care)+1+1./a.interval;
        }
        if(day==29)out.feed=out.care=false;
        if(29>max(day,placed))out.flow[29][8]+=1;
        out.meanwork/=max(1,29-day);
        return out;
    }
    double assetwork(int item,int p,const Variant&v=Variant{
    })const{
        int d=dist(p,access(p));
        if(item>=9)return (4.7+.35*d)*s.animal_work_mult;
        return (v.work/(v.length+s.cycle_gap)+.60+.04*d)*s.crop_work_mult;
    }
    void plan(const Obs&o,const Config&c){
        int day=o.day;
        Economy eco(o,c,s);
        Flow flow{
        };
        vector<int> assets,free;
        vector<pair<int,int>> pr;
        array<Variant,NT> policy;
        array<bool,NT> has{
        },af{
        },ac{
        };
        int nanim=0;
        auto roughpx=eco.value(flow).second;
        array<Flow,NT> projectflow{
        };
        array<double,NT> projectcost{
        };
        array<array<double,ND>,NT> projectwork{
        };
        array<bool,NT> reversible{
        };
        for(int p=0;p<NT;p++){
            auto&t=o.tiles[p];
            if(t.kind==-1)continue;
            if(t.animal()){
                nanim++;
                auto ap=animalplan(t.item,t.placed,day,&t,roughpx);
                addflow(flow,ap.flow);
                af[p]=ap.feed;
                ac[p]=ap.care;
                assets.push_back(p);
            }
            else if(t.plant()){
                int age=day-t.placed;
                auto vs=cycles(t.item,s.fertilize);
                vector<int> legal;
                for(int k=0;k<(int)vs.size();k++)if(vs[k].length>=age)legal.push_back(k);
                if(legal.empty())legal.push_back(vs.size()-1);
                double best=-1e100;
                int bv=legal[0];
                for(int k:legal){
                    auto f=cropflow(t.item,t.placed,day,vs[k],&t);
                    double val=0;
                    for(int d=day;d<ND;d++)for(int i=0;i<NP;i++)val+=f[d][i]*roughpx[d][i];
                    val-=s.labor_price*vs[k].work;
                    if(val>best){
                        best=val;
                        bv=k;
                    }
                }
                policy[p]=vs[bv];
                has[p]=true;
                addflow(flow,cropflow(t.item,t.placed,day,policy[p],&t));
                assets.push_back(p);
                int nxt=t.placed+policy[p].length+s.cycle_gap;
                if(s.crop_repeat&&day<nxt&&nxt<29){
                    auto cp=eco.newcrop(t.item,roughpx,nxt);
                    if(cp.valid)addflow(flow,cp.flow);
                }
            }
            else free.push_back(p);
        }
        // Keep live-asset forecasts separate from editable fresh investments.
        // All later project streams are captured when accepted, not reconstructed
        // from a possibly different future price vector during an RL edit.
        Flow committed_flow=flow;
        auto nearcmp=[](int a,int b){
            return tuple{
                dist(a,access(a)),Y(a),X(a)
            }
            <tuple{
                dist(b,access(b)),Y(b),X(b)
            };
        };
        sort(free.begin(),free.end(),nearcmp);
        auto [baseline,px]=eco.value(flow);
        array<double,ND> workload{
        };
        for(int p:assets){
            auto&t=o.tiles[p];
            if(t.animal()){
                for(int d=day;d<ND;d++)workload[d]+=assetwork(t.item,p);
            }
            else{
                int end=s.crop_repeat?29:min(29,t.placed+policy[p].length);
                for(int d=day;d<=end;d++)workload[d]+=assetwork(t.item,p,policy[p]);
            }
        }
        double reserve=s.opening_reserve+nanim*max(25.,o.prices[0])*1.5,liquid=0;
        for(int i=0;i<NP;i++)liquid+=o.shed[i]*o.prices[i];
        for(int p:assets)liquid+=o.tiles[p].manure*o.prices[8];
        double budget=max(0.,o.money+s.advance_revenue*liquid-reserve);
        Stock animals{
        };
        for(int i=9;i<12;i++){
            animals[i]=o.shed[i];
            for(auto&iv:o.invs)animals[i]+=iv[i];
        }
        for(auto [p,i]:m.projects){
            auto it=find(free.begin(),free.end(),p);
            if(it==free.end())continue;
            if(i>=9&&animals[i]>0){
                pr.emplace_back(p,i);
                free.erase(it);
                auto ap=animalplan(i,day,day,nullptr,roughpx);
                addflow(flow,ap.flow);
                projectflow[p]=ap.flow;
                af[p]=ap.feed;
                ac[p]=ap.care;
                animals[i]--;
                nanim++;
            }
        }
        for(int i=9;i<12;i++)for(int k=0;k<animals[i];k++){
            if(free.empty())break;
            int p=free.front();
            free.erase(free.begin());
            pr.emplace_back(p,i);
            auto ap=animalplan(i,day,day,nullptr,roughpx);
            addflow(flow,ap.flow);
            projectflow[p]=ap.flow;
            af[p]=ap.feed;
            ac[p]=ap.care;
            nanim++;
        }
        tie(baseline,px)=eco.value(flow);
        int limit=min(int(free.size()),int(s.expand_limit));
        // V2 grouped budget/space DP proposes the investment mixture. V1 keeps
        // ownership of crop variants, exact animal decisions and actual execution.
        array<int,12> dp_counts{};bool use_dp=false;
        if(s.portfolio_dp&&limit>0&&budget>=10){
            vector<int>items,cu,mx;vector<double>capital,streams;
            auto pack=[](const Flow&f,const array<double,ND>&work){
                array<double,390> a{};
                for(int d=0;d<ND;d++){for(int i=0;i<NP;i++){
                    double q=f[d][i];if(q<0&&(i==0||i==8))a[(i==0?270:300)+d]-=q;else a[d*NP+i]=q;
                }a[330+d]=work[d];}return a;
            };
            int B=min(5000,int(budget)/10);
            for(int item:{0,1,2,3,4,9,10,11}){
                Flow f{};Variant v;double cap=costs[item];
                if(item>=9){if(day+ANI[item-9].first>26)continue;f=animalplan(item,day,day,nullptr,px).flow;}
                else {auto cp=eco.newcrop(item,px);if(!cp.valid)continue;f=cp.flow;v=cp.variant;cap=cp.cost;}
                int cunit=int(ceil((costs[item]+(item>=9?o.prices[0]*1.5:0))/10.));
                if(cunit>B)continue;
                array<double,ND>w{};int end=item>=9||s.crop_repeat?29:min(29,day+v.length);
                for(int d=day;d<=end;d++)w[d]=assetwork(item,free.front(),v);
                auto st=pack(f,w);items.push_back(item);cu.push_back(cunit);mx.push_back(min(limit,B/cunit));capital.push_back(cap);streams.insert(streams.end(),st.begin(),st.end());
            }
            if(!items.empty()){
                auto base=pack(flow,workload);for(int i=0;i<NP;i++)base[day*NP+i]+=eco.stocks[i];
                array<double,270>dem{};for(int d=0;d<ND;d++)for(int i=0;i<NP;i++)dem[d*NP+i]=eco.dem[d][i]-eco.rival[d][i]*s.opponent_supply_weight;
                double discount=-log(s.early_discount+(1-s.early_discount)*min(1.,o.money/s.discount_wealth));
                double cfg[16]={discount,0,s.labor_hours,1,s.labor_price,10,1,1,0,double(s.portfolio_budget),o.money,0,300,12,0,4};
                int answer[9]{};double meta[2]{};
                int rc=fusionledger::ledger_plan(day,int(items.size()),limit,B,int(free.size()),o.unlocked,2,8,base.data(),streams.data(),cu.data(),mx.data(),capital.data(),dem.data(),o.market.data(),cfg,answer,meta);
                if(rc==0){use_dp=true;for(size_t k=0;k<items.size();k++)dp_counts[items[k]]=answer[k];}
            }
        }
        for(int it=0;it<limit;it++){
            double best=-1e100;
            int bi=-1;
            Flow bf{
            },bpx{
            };
            Variant bv;
            double bval=0,bcost=0;
            array<double,ND> bwork{
            };
            bool bfeed=true,bcare=true;
            for(int item:{
                0,1,2,3,4,9,10,11
            }){
                if(use_dp&&dp_counts[item]<=0)continue;
                if(costs[item]>budget)continue;
                Flow f{
                };
                double cost,work;
                Variant variant;
                bool feed=true,care=true;
                if(item>=9){
                    if(day+ANI[item-9].first>26)continue;
                    auto ap=animalplan(item,day,day,nullptr,px);
                    f=ap.flow;
                    feed=ap.feed;
                    care=ap.care;
                    cost=costs[item];
                    work=(29-day)*3.5+5;
                }
                else{
                    auto cp=eco.newcrop(item,px);
                    if(!cp.valid)continue;
                    f=cp.flow;
                    cost=cp.cost;
                    work=cp.work;
                    variant=cp.variant;
                }
                addflow(flow,f);
                auto [val,nextpx]=eco.value(flow);
                addflow(flow,f,-1);
                array<double,ND> dw{
                };
                if(item>=9){
                    for(int d=day;d<ND;d++)dw[d]=assetwork(item,free[0]);
                }
                else{
                    int end=s.crop_repeat?29:min(29,day+variant.length);
                    for(int d=day;d<=end;d++)dw[d]=assetwork(item,free[0],variant);
                }
                auto wages=[&](double w){
                    double h=max(0.,w/s.labor_hours-1);
                    return max(0.,(pow(1.618034,h+2)-1)/2.236068-1);
                };
                double wcost=0;
                for(int d=day;d<ND;d++)wcost+=wages(workload[d]+dw[d])-wages(workload[d]);
                double marginal=val-baseline-cost-(s.labor_capacity?wcost:s.labor_price*work);
                int duration=item>=9?29-day:min(29-day,variant.length);
                marginal-=s.land_rent*duration;
                if(marginal<=0)continue;
                double financing=costs[item]+(item>=9?100:0),exponent=s.capital_power*max(0.,1-o.money/16000),score=marginal/pow(financing,exponent);
                if(score>best){
                    best=score;
                    bi=item;
                    bf=f;
                    bv=variant;
                    bval=val;
                    bpx=nextpx;
                    bwork=dw;
                    bfeed=feed;
                    bcare=care;
                    bcost=cost;
                }
            }
            if(bi<0)break;
            if(use_dp)dp_counts[bi]--;
            for(int d=day;d<ND;d++)workload[d]+=bwork[d];
            int p=free.front();
            free.erase(free.begin());
            pr.emplace_back(p,bi);
            projectflow[p]=bf;
            projectcost[p]=bcost;
            projectwork[p]=bwork;
            reversible[p]=true;
            af[p]=bfeed;
            ac[p]=bcare;
            if(bi<5){
                policy[p]=bv;
                has[p]=true;
            }
            addflow(flow,bf);
            budget-=costs[bi];
            if(bi>=9){
                budget-=o.prices[0]*1.5;
                nanim++;
            }
            baseline=bval;
            px=bpx;
        }
        // Coordinate search over the COMPLETE chosen portfolio, not independent
        // project scores. Only unpurchased new projects are reversible. Every trial
        // updates joint market supply and the nonlinear staffing cost before comparing.
        int nswaps=0;
        for(int pass=0;pass<s.portfolio_passes;pass++){
            double best=1e-6,bval=0,bcost=0;
            int bj=-1,bi=-1;
            Flow bf{
            },bpx{
            };
            array<double,ND> bw{
            };
            Variant bv;
            bool bfeed=true,bcare=true;
            for(int j=0;j<int(pr.size());j++){
                auto [p,old]=pr[j];
                if(!reversible[p])continue;
                Flow residual=flow;
                addflow(residual,projectflow[p],-1);
                auto rpx=eco.value(residual).second;
                double oldfront=costs[old]+(old>=9?o.prices[0]*1.5:0);
                for(int item:{
                    0,1,2,3,4,9,10,11
                }){
                    if(item==old)continue;
                    double front=costs[item]+(item>=9?o.prices[0]*1.5:0);
                    if(front>budget+oldfront)continue;
                    Flow f{
                    };
                    Variant variant;
                    double cost;
                    bool feed=true,care=true;
                    if(item>=9){
                        if(day+ANI[item-9].first>26)continue;
                        auto ap=animalplan(item,day,day,nullptr,rpx);
                        f=ap.flow;
                        feed=ap.feed;
                        care=ap.care;
                        cost=costs[item];
                    }
                    else{
                        auto cp=eco.newcrop(item,rpx);
                        if(!cp.valid)continue;
                        f=cp.flow;
                        variant=cp.variant;
                        cost=cp.cost;
                    }
                    auto candidate=residual;
                    addflow(candidate,f);
                    auto [val,nextpx]=eco.value(candidate);
                    array<double,ND> dw{
                    };
                    int end=item>=9||s.crop_repeat?29:min(29,day+variant.length);
                    for(int d=day;d<=end;d++)dw[d]=assetwork(item,p,variant);
                    auto wages=[&](double w){
                        double h=max(0.,w/s.labor_hours-1);
                        return max(0.,(pow(1.618034,h+2)-1)/2.236068-1);
                    };
                    double wagechange=0;
                    for(int d=day;d<ND;d++)wagechange+=wages(workload[d]-projectwork[p][d]+dw[d])-wages(workload[d]);
                    int olddur=old>=9?29-day:min(29-day,policy[p].length),duration=item>=9?29-day:min(29-day,variant.length);
                    double delta=val-baseline+projectcost[p]-cost-wagechange-s.land_rent*(duration-olddur);
                    if(delta>best){
                        best=delta;
                        bj=j;
                        bi=item;
                        bf=f;
                        bpx=nextpx;
                        bv=variant;
                        bval=val;
                        bcost=cost;
                        bw=dw;
                        bfeed=feed;
                        bcare=care;
                    }
                }
            }
            if(bj<0)break;
            int p=pr[bj].first,old=pr[bj].second;
            budget+=costs[old]+(old>=9?o.prices[0]*1.5:0)-costs[bi]-(bi>=9?o.prices[0]*1.5:0);
            addflow(flow,projectflow[p],-1);
            addflow(flow,bf);
            for(int d=day;d<ND;d++)workload[d]+=bw[d]-projectwork[p][d];
            nanim+=int(bi>=9)-int(old>=9);
            pr[bj].second=bi;
            projectflow[p]=bf;
            projectcost[p]=bcost;
            projectwork[p]=bw;
            policy[p]=bv;
            has[p]=bi<5;
            af[p]=bfeed;
            ac[p]=bcare;
            baseline=bval;
            px=bpx;
            nswaps++;
        }
        vector<int> coords;
        for(auto [p,i]:pr)coords.push_back(p);
        sort(coords.begin(),coords.end(),nearcmp);
        stable_sort(pr.begin(),pr.end(),[](auto a,auto b){
            return pair{
                a.second>=9?0:1,a.second==0?0:1
            }
            <pair{
                b.second>=9?0:1,b.second==0?0:1
            };
        });
        auto oldpolicy=policy;
        auto oldaf=af,oldac=ac;
        // Project identity follows the spatial assignment, not the old coordinate.
        auto oldprojectflow=projectflow;
        auto oldreversible=reversible;
        for(int j=0;j<(int)pr.size();j++){
            auto [oldp,item]=pr[j];
            int p=coords[j];
            pr[j].first=p;
            af[p]=oldaf[oldp];
            ac[p]=oldac[oldp];
            // Ownership follows the assigned position even when capacity_guard
            // is disabled. This metadata alone does not alter the KEEP plan.
            projectflow[p]=oldprojectflow[oldp];
            if(s.capacity_guard){
                reversible[p]=oldreversible[oldp];
            }
            if(item<5){
                policy[p]=oldpolicy[oldp];
                has[p]=true;
            }
        }
        bool land=false;
        if(free.empty()&&o.unlocked<4&&day<20){
            int lp=array<int,3>{
                1000,2000,4000
            }
            [o.unlocked-1];
            if(budget>lp+700)land=true;
        }
        if(s.c2_service)m.c2_service_calls++;
        if(s.c2_service_refresh){
            // Refresh existing assets under the chosen portfolio, not the empty
            // farm price used at the start of planning. Keep the original asset
            // and its current harvest; change only the future service contract.
            auto fixed_prices=px;
            for(int p:assets){auto&t=o.tiles[p];if(!t.animal())continue;
                auto before=animalplan(t.item,t.placed,day,&t,roughpx);
                auto after=animalplan(t.item,t.placed,day,&t,fixed_prices);
                m.c2_refresh_changes+=af[p]!=after.feed||ac[p]!=after.care;
                addflow(flow,before.flow,-1);addflow(flow,after.flow);
                addflow(committed_flow,before.flow,-1);addflow(committed_flow,after.flow);
                af[p]=after.feed;ac[p]=after.care;
            }
            px=eco.value(flow).second;
        }
        for(int p=0;p<NT;p++)m.oldcrop[p]=o.tiles[p].plant()?o.tiles[p].item:-1;
        m.setprojects(pr);
        m.policy=policy;
        m.haspolicy=has;
        m.animalfeed=af;
        m.animalcare=ac;
        m.forecast=flow;
        m.prices=px;
        m.land=land;
        m.scheduled=false;
        m.routes.clear();
        m.day=day;
        int capacity_deferred=0;
        if(s.capacity_guard){
            // A conservative feasibility PREVIEW, not a new game restriction.
            // Only defer fresh, unpurchased investments the normal day scheduler
            // cannot place on any worker route. Existing assets and purchases
            // carried from the previous day are never cancelled by this switch.
            schedule(o);
            set<int> deferred;
            for(int p:m.backlog)if(reversible[p])deferred.insert(p);
            vector<pair<int,int>> kept;
            for(auto [p,item]:pr){
                if(!deferred.count(p)){kept.emplace_back(p,item);continue;}
                addflow(flow,projectflow[p],-1);
                if(item>=9)nanim--;
                m.haspolicy[p]=false;
                capacity_deferred++;
            }
            pr=std::move(kept);
            m.setprojects(pr);
            m.forecast=flow;
            m.prices=eco.value(flow).second;
            // Procurement and real unit actions now consume the same kept plan.
            // Regenerate routes from the actual state; do not retain stale claims.
            m.scheduled=false;
            m.routes.clear();
            m.backlog.clear();
            m.couriers.clear();
        }
        m.forecast_ledger.capture(day,committed_flow,pr,projectflow,m.forecast);
        JA jp;
        for(auto [p,i]:pr)jp.push_back(JA{
            X(p),Y(p),names[i]
        });
        m.daily_records.push_back(JO{
            {
                "day",day
            },{
                "money",o.money
            },{
                "new",jp
            },{
                "animals",nanim
            },{
                "land",land
            },{
                "capacity_deferred",capacity_deferred
            }
        });
    }
    vector<Op> needs(const Obs&o,int p)const{
        auto&t=o.tiles[p];
        int day=o.day;
        bool terminal=day==29;
        vector<Op> out;
        int fresh=m.newitem[p];
        auto add=[&](string op,int res,double val,bool mand){
            out.push_back({
                op,res,val,mand
            });
        };
        auto expedite=[&](int product,int q,int wait){
            if(!s.harvest_early||q<=0)return false;
            if(s.harvest_early==1)return true;
            // Compare public-model delivery prices, not a realized future.
            int next=min(29,day+max(1,wait));
            return o.money<s.harvest_cash||o.prices[product]>m.prices[next][product]*s.harvest_ratio;
        };
        if(t.animal()){
            auto a=ANI[t.item-9];
            int age=day-t.placed,q=t.yield;
            bool nexttick=age+1>=a.first&&(age+1-a.first)%a.interval==0;
            bool planned_exit=s.c2_service&&t.unfed>=1&&!m.animalfeed[p];
            if(q&&(q>=a.cap-1||terminal||planned_exit||(nexttick&&q+1+t.pending>a.cap)||expedite(a.prod,q,a.interval)))add("HARVEST",-1,300+q*o.prices[a.prod],true);
            if(t.manure&&(o.prices[8]>=s.manure_floor||m.fertilizer_deficit>0))add("COLLECT_FERTILIZER",-1,max(15.,o.prices[8]),true);
            if(!terminal&&!t.feed&&((!s.animal_dp&&!s.c2_service)||m.animalfeed[p]||(!s.c2_service&&t.unfed>=1)))add("FEED",0,350+100*t.unfed,true);
            int nextage=age<a.first?a.first:a.first+((age-a.first)/a.interval+1)*a.interval;
            if(s.animal_care&&!t.care&&t.placed+nextage<=29&&(!s.animal_dp||m.animalcare[p]))add("CARE",-1,max(20.,o.prices[a.prod]),false);
            stable_sort(out.begin(),out.end(),[&](const Op&a,const Op&b){
                return int(!(a.name=="FEED"&&t.unfed>=1))<int(!(b.name=="FEED"&&t.unfed>=1));
            });
            return out;
        }
        if(t.plant()){
            int crop=t.item,age=day-t.placed,q=t.yield;
            if(s.retire_spent&&(crop==2||crop==3)&&age>=(crop==2?11:16)&&q==0){
                // All four production ticks are over. DIG is legal; no future
                // production is sacrificed. Replant still checks live capacity.
                if(day<28)add("DIG",-1,150,false);
                return out;
            }
            Variant v=m.haspolicy[p]?m.policy[p]:cycles(crop).back();
            bool ready,water;
            if(crop==2||crop==3){
                int first=crop==2?8:10,interval=crop==2?1:2;
                bool tick=age+1>=first&&(age+1-first)%interval==0&&age+1<=first+3*interval;
                ready=q>0&&(q>=3||terminal||t.lifespan>=0||(tick&&q>=2)||expedite(crop,q,interval));
                bool ev=false;
                for(auto e:v.events)if(e.age==age&&e.item==8)ev=true;
                water=!terminal&&(t.unwatered>=1||age==0||(tick&&(t.fertilized>=day||ev)));
            }
            else{
                int cap=crop==1?4:6,first=crop==4?10:2;
                bool window=crop==0?(2<=age&&age<=4):crop==1?(2<=age&&age<=3):(6<=age&&age<=12);
                ready=q>0&&age>=first&&(age>=v.length||terminal||q>=cap);
                water=(t.unwatered>=1||age==0||(window&&q<cap))&&!(ready&&!window)&&!(terminal&&!ready);
            }
            bool fert=false;
            for(auto e:v.events)if(e.age==age&&e.item==8&&e.q<0)fert=true;
            fert=fert&&s.fertilize&&!terminal&&t.fertilized<day;
            if(fert)add("FERTILIZE",8,max(80.,o.prices[crop]*1.2),false);
            if(water&&!t.water)add("WATER",-1,age==0||t.unwatered>=1?450:110,true);
            if(ready)add("HARVEST",-1,300+q*o.prices[crop],true);
            if(s.harvest_first&&ready)stable_sort(out.begin(),out.end(),[](const Op&a,const Op&b){return (a.name=="HARVEST")>(b.name=="HARVEST");});
            return out;
        }
        if(fresh>=0){
            if(t.kind!=0&&!((t.kind==3||t.kind==4)&&fresh>=9&&t.kind==ANI[fresh-9].kind))add("DIG",-1,150,false);
            else if(fresh>=9){
                if(t.kind==0)add(fresh==9?"BUILD_COOP":"BUILD_PASTURE",-1,180,false);
                else add("PLACE",fresh,250,false);
            }
            else add("PLANT",fresh,160,false);
        }
        return out;
    }
    pair<vector<pair<int,int>>,int> plannednodes(const Obs&o)const{
        vector<pair<int,int>> out;
        int goods=0;
        for(int p=0;p<NT;p++){
            auto ops=needs(o,p);
            if(ops.empty())continue;
            int work=ops.size(),item=m.newitem[p];
            auto&t=o.tiles[p];
            if(item>=0&&!t.active()){
                if(item>=9)work+=ops[0].name.rfind("BUILD",0)==0?3:ops[0].name=="PLACE"?2:4;
                else work+=ops[0].name=="PLANT"?1:2;
            }
            for(auto&op:ops){
                if(op.name=="HARVEST")goods+=t.yield;
                if(op.name=="COLLECT_FERTILIZER")goods++;
            }
            out.emplace_back(p,work);
        }
        return {
            out,goods
        };
    }
    vector<int> coverroute(const Obs&o,int start,const vector<pair<int,int>>&nodes,bool last,bool bank)const{
        auto priority=[&](int p){
            if(s.anchor_mode==1){
                auto&t=o.tiles[p];
                if(t.animal()&&t.unfed>=1)return 5;
                if(t.plant()&&(t.unwatered>=1||o.day==t.placed))return 4;
            }
            if(s.anchor_mode==2){
                double value=0;
                for(auto&op:needs(o,p))value+=op.value;
                return int(value/200);
            }
            return o.tiles[p].animal()?3:o.tiles[p].plant()?2:1;
        };
        int anchor=nodes.front().first;
        for(auto [p,w]:nodes)if(pair{
            priority(p),dist(p,access(p))
        }
        >pair{
            priority(anchor),dist(anchor,access(anchor))
        })anchor=p;
        vector<int> near;
        array<int,NT> nw{
        };
        for(auto [p,w]:nodes){
            nw[p]=w;
            if(p!=anchor)near.push_back(p);
        }
        stable_sort(near.begin(),near.end(),[&](int a,int b){
            return dist(a,anchor)+.05*dist(a,access(a))<dist(b,anchor)+.05*dist(b,access(b));
        });
        int nn=clamp(s.route_nodes,3,14)-1;
        if(int(near.size())>nn)near.resize(nn);
        vector<int> pts={
            anchor
        };
        pts.insert(pts.end(),near.begin(),near.end());
        int n=pts.size(),size=1<<n;
        vector<double> values(size);
        vector<int> work(n);
        vector<double> weights(n);
        for(int i=0;i<n;i++){
            work[i]=nw[pts[i]];
            weights[i]=work[i]+2;
            for(auto&op:needs(o,pts[i]))weights[i]+=s.route_value*op.value/200.;
        }
        int distance[14][14];
        for(int j=0;j<n;j++)for(int k=0;k<n;k++)distance[j][k]=dist(pts[j],pts[k]);
        vector<int> dp(size*n,99),parent(size*n,-1);
        for(int j=0;j<n;j++)dp[(1<<j)*n+j]=dist(start,pts[j])+work[j];
        int bm=-1,bj=-1;
        double score=-1;
        for(int mask=1;mask<size;mask++){
            int bit=mask&-mask,j=__builtin_ctz(unsigned(bit));
            values[mask]=values[mask^bit]+weights[j];
            for(j=0;j<n;j++){
                int t=dp[mask*n+j];
                if(t>s.route_budget)continue;
                int end=t+((last||bank)?dist(pts[j],access(pts[j]))+1:0);
                if((mask&1)&&end<=(last?s.route_terminal:s.route_budget)){
                    double val=values[mask]-.015*end;
                    if(val>score){
                        score=val;
                        bm=mask;
                        bj=j;
                    }
                }
                int avail=(size-1)^mask;
                while(avail){
                    int low=avail&-avail,k=__builtin_ctz(unsigned(low));
                    avail^=low;
                    int tt=t+distance[j][k]+work[k],mm=mask|low;
                    if(tt<dp[mm*n+k]&&tt<=s.route_budget){
                        dp[mm*n+k]=tt;
                        parent[mm*n+k]=j;
                    }
                }
            }
        }
        if(bm<0)return {
            anchor
        };
        vector<int> route;
        int mask=bm,j=bj;
        while(j>=0){
            route.push_back(pts[j]);
            int prev=parent[mask*n+j];
            mask^=1<<j;
            j=prev;
        }
        reverse(route.begin(),route.end());
        return route;
    }
    // Exact shortest tour of a prescribed task set. The outer team repair is a
    // heuristic, but each tentative absorption uses an exact subset DP rather
    // than a nearest-neighbor distance estimate.
    pair<int,vector<int>> shortesttour(int start,const vector<int>&pts,const array<int,NT>&work,bool bank)const{
        int n=pts.size();
        if(n==0)return {
            0,{
            }
        };
        if(n>12)return {
            999,{
            }
        };
        int size=1<<n;
        vector<int> dp(size*n,999),parent(size*n,-1);
        for(int j=0;j<n;j++)dp[(1<<j)*n+j]=dist(start,pts[j])+work[pts[j]];
        for(int mask=1;mask<size;mask++)for(int j=0;j<n;j++)if(mask&(1<<j)){
            int t=dp[mask*n+j];
            if(t>24)continue;
            for(int k=0;k<n;k++)if(!(mask&(1<<k))){
                int next=mask|(1<<k),nt=t+dist(pts[j],pts[k])+work[pts[k]];
                if(nt<dp[next*n+k]){
                    dp[next*n+k]=nt;
                    parent[next*n+k]=j;
                }
            }
        }
        int bj=-1,best=999;
        for(int j=0;j<n;j++){
            int t=dp[(size-1)*n+j]+(bank?dist(pts[j],access(pts[j]))+1:0);
            if(t<best){
                best=t;
                bj=j;
            }
        }
        if(bj<0)return {
            999,{
            }
        };
        vector<int> rr;
        int mask=size-1,j=bj;
        while(j>=0){
            rr.push_back(pts[j]);
            int prev=parent[mask*n+j];
            mask^=1<<j;
            j=prev;
        }
        reverse(rr.begin(),rr.end());
        return {
            best,rr
        };
    }
    void repairteam(const Obs&o,vector<vector<int>>&routes,const array<int,NT>&work,bool bank)const{
        if(!s.team_repair)return;
        int budget=(o.day==29?s.route_terminal:s.route_budget)-s.team_slack;
        for(int pass=0;pass<s.team_repair&&routes.size()>3;pass++){
            auto candidate=routes;
            auto todo=candidate.back();
            candidate.pop_back();
            bool feasible=true;
            stable_sort(todo.begin(),todo.end(),[&](int a,int b){
                return dist(a,access(a))>dist(b,access(b));
            });
            for(int p:todo){
                int bu=-1,best=999;
                vector<int> br;
                for(int u=0;u<int(candidate.size());u++){
                    int start=u<int(o.positions.size())?o.positions[u]:xy(4+u%2,4+(u/2)%2);
                    auto pts=candidate[u];
                    pts.push_back(p);
                    auto fit=shortesttour(start,pts,work,bank);
                    if(fit.first>budget)continue;
                    auto current=shortesttour(start,candidate[u],work,bank);
                    int delta=fit.first-current.first;
                    if(delta<best){
                        best=delta;
                        bu=u;
                        br=std::move(fit.second);
                    }
                }
                if(bu<0){
                    feasible=false;
                    break;
                }
                candidate[bu]=std::move(br);
            }
            if(!feasible)break;
            routes=std::move(candidate);
        }
    }
    void schedule(const Obs&o){
        if(m.scheduled)return;
        auto [nodes,goods]=plannednodes(o);
        bool last=o.day==29;
        vector<vector<int>> routes;
        array<int,NT> service{
        };
        for(auto [p,w]:nodes)service[p]=w;
        bool repairbank=last||(s.delivery_planning&&goods>80);
        while(!nodes.empty()&&int(routes.size())<=s.max_hands+(last?s.terminal_extra:0)){
            int u=routes.size(),start=u<int(o.positions.size())?o.positions[u]:xy(4+u%2,4+(u/2)%2);
            bool bank=s.delivery_planning&&goods>80;
            auto rr=coverroute(o,start,nodes,last,bank);
            for(int p:rr)for(auto&op:needs(o,p)){
                if(op.name=="HARVEST")goods-=o.tiles[p].yield;
                if(op.name=="COLLECT_FERTILIZER")goods--;
            }
            routes.push_back(rr);
            for(int p:rr)nodes.erase(remove_if(nodes.begin(),nodes.end(),[&](auto a){
                return a.first==p;
            }),nodes.end());
        }
        repairteam(o,routes,service,repairbank);
        m.backlog.clear();
        for(auto [p,w]:nodes)m.backlog.push_back(p);
        m.hire_target=max(2,int(routes.size())-1);
        while(int(routes.size())<m.hire_target+1)routes.push_back({
        });
        m.routes=std::move(routes);
        m.scheduled=true;
        m.couriers.clear();
    }
    Stock routeinputs(const Obs&o,const vector<int>&route)const{
        Stock req{
        };
        for(int p:route){
            auto&t=o.tiles[p];
            int item=m.newitem[p];
            if(item>=9&&!t.animal()){
                req[item]++;
                if(o.day<29&&(!s.animal_dp||m.animalfeed[p]))req[0]++;
            }
            else for(auto&op:needs(o,p))if(op.name=="FEED"||op.name=="FERTILIZE"||op.name=="PLACE")req[op.res]++;
        }
        return req;
    }
    Stock holdplan(const Obs&o,const Config&c,const Stock&shed,const Stock&added,int keepfeed,int keepfert)const{
        Stock keep{
        };
        if(s.market_wait<=o.hour||o.money<5000)return keep;
        // Exact grouped knapsack for the currently visible inventory and known town
        // consumption until a fixed intraday clearing window. New deliveries are
        // handled by replanning; this is not a clairvoyant season sales policy.
        int incoming=0;
        for(auto&iv:o.invs)incoming+=sumstock(iv);
        incoming=max(0,incoming-sumstock(added));
        int capacity=max(0,min(s.storage_cap,c.shedcap-10-incoming-keepfeed-keepfert));
        if(capacity<=0)return keep;
        array<int,NP> demand{
        };
        int end=min(22,s.market_wait);
        for(int h=o.hour;h<end;h++){
            if(h%c.shopinterval==0)for(auto&sh:o.shops)for(auto&kv:SHOPS)if(kv.first==sh)for(int i:kv.second)demand[i]++;
            if(h==0)for(int i=0;i<8;i++)demand[i]++;
        }
        vector<vector<double>> gains(NP);
        Flow rival{};
        if(s.sale_competition_weight>0)rival=public_rival_flow(o,s);
        for(int i=1;i<8;i++){
            int avail=shed[i],limit=min(capacity,avail);
            gains[i].resize(limit+1);
            if(!avail||!demand[i])continue;
            vector<double> revenue(avail+1),inventory(avail+1);
            inventory[0]=o.market[i];
            for(int q=1;q<=avail;q++){
                int pp=price(i,inventory[q-1],o);
                revenue[q]=revenue[q-1]+pp;
                inventory[q]=inventory[q-1]+int(pp>1);
            }
            for(int hold=1;hold<=limit;hold++){
                double expected=0;
                if(s.sale_competition_weight>0){
                    for(int d=o.day;d<=min(29,o.day+1);d++)expected+=max(0.,rival[d][i]);
                    expected*=s.sale_competition_weight*max(0,end-o.hour)/24.;
                }
                double inv=inventory[avail-hold]-demand[i]+expected,later=0;
                for(int q=0;q<hold;q++){
                    int pp=price(i,inv,o);
                    later+=pp;
                    inv+=int(pp>1);
                }
                gains[i][hold]=revenue[avail-hold]+later-revenue[avail];
            }
        }
        vector<double> dp(capacity+1,-1e100);
        dp[0]=0;
        int parent[9][101]{
        };
        for(int i=1;i<8;i++){
            vector<double> next(capacity+1,-1e100);
            for(int used=0;used<=capacity;used++)if(dp[used]>-1e90)for(int q=0;q<int(gains[i].size())&&used+q<=capacity;q++){
                double v=dp[used]+gains[i][q];
                if(v>next[used+q]+1e-8){
                    next[used+q]=v;
                    parent[i][used+q]=q;
                }
            }
            dp=std::move(next);
        }
        int used=max_element(dp.begin(),dp.end())-dp.begin();
        for(int i=7;i>=1;i--){
            keep[i]=parent[i][used];
            used-=keep[i];
        }
        return keep;
    }
    JA marketorders(const Obs&o,const Config&c,const Stock&added){
        Stock shed=o.shed;
        for(int i=0;i<NI;i++)shed[i]+=added[i];
        int day=o.day,hour=o.hour,ntarget=0;
        JA orders;
        double cash=o.money;
        for(auto [p,i]:m.projects)if(i>=9&&!o.tiles[p].animal()&&(!s.animal_dp||m.animalfeed[p]))ntarget++;
        int nf=0,nfeed=0;
        for(int p=0;p<NT;p++)for(auto&op:needs(o,p)){
            if(op.name=="FERTILIZE")nf++;
            if(op.name=="FEED")nfeed++;
        }
        int cf=0,cfeed=0;
        for(auto&iv:o.invs){
            cfeed+=iv[0];
            cf+=iv[8];
        }
        int keepfeed=day<29?max(0,nfeed+ntarget-cfeed):0,keepfert=max(0,nf-cf);
        if(hour>0&&!m.routes.empty()){
            keepfeed=keepfert=0;
            for(int u=0;u<int(m.routes.size());u++){
                auto req=routeinputs(o,m.routes[u]);
                Stock inv{
                };
                if(u<int(o.invs.size()))inv=o.invs[u];
                keepfeed+=max(0,req[0]-inv[0]);
                keepfert+=max(0,req[8]-inv[8]);
            }
        }
        if(hour>=18&&day<29){
            int tomorrow=0;
            for(int p=0;p<NT;p++)if(m.haspolicy[p]&&o.tiles[p].plant()){
                int age=day+1-o.tiles[p].placed;
                for(auto e:m.policy[p].events)if(e.age==age&&e.item==8&&e.q<0)tomorrow++;
            }
            keepfert=max(keepfert,tomorrow);
        }
        auto retention=holdplan(o,c,shed,added,keepfeed,keepfert);
        array<int,NP> sellseq{};iota(sellseq.begin(),sellseq.end(),0);
        if(s.sale_order){
            array<double,NP>scores{},rivalqty{};
            for(const auto&t:o.opponent_tiles){
                int product=t.animal()?ANI[t.item-9].prod:t.plant()?t.item:-1;
                if(product>=0&&product<NP)rivalqty[product]+=max(0,t.yield);
                if(t.animal()&&t.manure)rivalqty[8]++;
            }
            for(int i=0;i<NP;i++){
                int keep=i==0?keepfeed:i==8?keepfert:retention[i],q=max(0,shed[i]-keep);
                double now=0,risk=0;
                for(int k=0;k<q;k++){
                    double px=price(i,o.market[i]+k,o);now+=px;
                    risk+=max(0.,px-price(i,o.market[i]+k+rivalqty[i],o));
                }
                scores[i]=s.sale_order==1?now:s.sale_order==2?risk: risk+now*.05;
            }
            stable_sort(sellseq.begin(),sellseq.end(),[&](int a,int b){return scores[a]>scores[b];});
        }
        for(int i:sellseq){
            int keep=i==0?keepfeed:i==8?keepfert:retention[i],q=max(0,shed[i]-keep);
            if(q){
                orders.push_back(JA{
                    "SELL",names[i],q
                });
                for(int k=0;k<q;k++)cash+=price(i,o.market[i]+k,o);
                shed[i]-=q;
            }
        }
        if(hour<=3||(s.late_hire&&hour<22&&m.routes.size()>o.positions.size())){
            int nhired=0;
            while(int(o.positions.size())-1+nhired<m.hire_target&&int(orders.size())<c.orderlimit){
                int n=o.positions.size()-1+nhired,cc=FIB[min(n,19)]*c.handmult;
                if(cash<cc+30)break;
                orders.push_back(JA{
                    "HIRE"
                });
                cash-=cc;
                nhired++;
            }
        }
        int lack=max(0,keepfeed-shed[0]);
        if(lack&&day<29&&orders.size()<10){
            int n=min(lack,int(cash/max(1.,o.prices[0]+1)));
            if(n>0){
                orders.push_back(JA{
                    "BUY_PRODUCT","WHEAT",n
                });
                cash-=n*(o.prices[0]+1);
            }
        }
        lack=max(0,keepfert-shed[8]);
        if(lack&&hour<17&&orders.size()<10){
            int n=min(lack,int(max(0.,cash-60)/max(1.,o.prices[8]+1)));
            if(n>0){
                orders.push_back(JA{
                    "BUY_PRODUCT","FERTILIZER",n
                });
                cash-=n*(o.prices[8]+1);
            }
        }
        Stock demand{
        };
        vector<int> orderitems;
        for(auto [p,i]:m.projects){
            if(o.tiles[p].active())continue;
            if(!demand[i])orderitems.push_back(i);
            demand[i]++;
        }
        for(int item:orderitems){
            int have=item<5?o.seeds[item]:shed[item];
            if(item>=9)for(auto&iv:o.invs)have+=iv[item];
            int q=min(max(0,demand[item]-have),int(max(0.,cash-35)/costs[item]));
            if(q&&orders.size()<10){
                orders.push_back(JA{
                    item<5?"BUY_SEED":"BUY_ANIMAL",names[item],q
                });
                cash-=costs[item]*q;
            }
        }
        if(m.land&&hour<4&&orders.size()<10){
            int cc=o.unlocked<4?array<int,3>{
                1000,2000,4000
            }
            [o.unlocked-1]:999999;
            if(cash>=cc+200){
                orders.push_back(JA{
                    "BUY_LAND"
                });
                m.land=false;
            }
        }
        if(int(orders.size())>c.orderlimit)orders.resize(c.orderlimit);
        return orders;
    }
    int pickop(const vector<Op>&ops,const Stock&inv,const Stock&seeds)const{
        for(int j=0;j<int(ops.size());j++){
            auto&op=ops[j];
            if(op.name=="PLANT"&&seeds[op.res]<1)continue;
            if((op.name=="FEED"||op.name=="FERTILIZE"||op.name=="PLACE")&&inv[op.res]<1)continue;
            return j;
        }
        return -1;
    }
    void replant(const Obs&o,const Config&c){
        if(s.replant_hour<0||o.hour>s.replant_hour||o.day>=28||!m.scheduled)return;
        Economy eco(o,c,s);
        double cash=o.money;
        Flow live{};
        if(s.dynamic_replant){
            auto px=eco.value(live).second;
            for(int q=0;q<NT;q++){
                const auto&t=o.tiles[q];
                if(t.animal())addflow(live,animalplan(t.item,t.placed,o.day,&t,px).flow);
                else if(t.plant()){
                    auto variants=cycles(t.item,s.fertilize);auto v=m.haspolicy[q]?m.policy[q]:variants.back();
                    addflow(live,cropflow(t.item,t.placed,o.day,v,&t));
                }
            }
        }
        // Newly cleared plots can be planted and watered on the SAME day if the
        // remaining visit has room. No worker owns the crop lifecycle; every future
        // operation still comes from the live observation and shared seed ledger.
        for(int p=0;p<NT;p++){
            int crop=m.oldcrop[p];
            if(crop<0||m.newitem[p]>=0||o.tiles[p].kind!=0)continue;
            if(cash<(s.dynamic_replant?10:costs[crop])+120)continue;
            bool feasible=false;
            for(int u=0;u<int(o.positions.size())&&u<int(m.routes.size());u++)if(o.positions[u]==p){
                int time=o.hour+(o.seeds[crop]?2:3),prev=p;
                for(int q:m.routes[u])if(q!=p){
                    time+=dist(prev,q)+int(needs(o,q).size());
                    prev=q;
                }
                if(time<=23)feasible=true;
            }
            if(!feasible)continue;
            auto cp=eco.newcrop(crop,m.prices);
            if(s.dynamic_replant){
                auto [baseval,px]=eco.value(live);double best=0;int selected=-1;CropPlan bestplan;
                for(int k=0;k<5;k++){
                    if(cash<costs[k]+120)continue;
                    bool fits=false;
                    for(int u=0;u<int(o.positions.size())&&u<int(m.routes.size());u++)if(o.positions[u]==p){
                        int finish=o.hour+(o.seeds[k]?2:3),prev=p;
                        for(int q:m.routes[u])if(q!=p){finish+=dist(prev,q)+int(needs(o,q).size());prev=q;}
                        if(finish<=23)fits=true;
                    }
                    if(!fits)continue;
                    auto trial=eco.newcrop(k,px);if(!trial.valid)continue;
                    auto candidate=live;addflow(candidate,trial.flow);
                    double gain=eco.value(candidate).first-baseval-trial.cost-s.labor_price*trial.work;
                    gain-=s.land_rent*min(29-o.day,trial.variant.length);
                    if(gain>best){best=gain;selected=k;bestplan=trial;}
                }
                if(selected<0)continue;crop=selected;cp=bestplan;addflow(live,cp.flow);
            }
            if(!cp.valid)continue;
            m.projects.emplace_back(p,crop);
            m.newitem[p]=crop;
            m.oldcrop[p]=-1;
            m.policy[p]=cp.variant;
            m.haspolicy[p]=true;
            bool assigned=false;
            for(auto&rr:m.routes)if(find(rr.begin(),rr.end(),p)!=rr.end())assigned=true;
            if(!assigned)m.backlog.push_back(p);
            cash-=costs[crop];
        }
    }
    // Late-day rescue: ownership belongs to the task, not the original worker.
    // Match urgent live-state tasks to currently feasible workers; reserve shared
    // wheat before ordinary pickups. Greedy matching is deliberately NOT claimed
    // to be a globally optimal multi-worker scheduler.
    vector<pair<int,int>> rescueassign(const Obs&o){
        int n=o.positions.size();
        vector<pair<int,int>> assignment(n,{
            -1,0
        });
        if(o.day==29||o.hour<s.rescue_hour)return assignment;
        vector<pair<int,int>> tasks;
        for(int p=0;p<NT;p++){
            auto&t=o.tiles[p];
            if(t.animal()&&!t.feed&&t.unfed>=1&&(!s.c2_service||m.animalfeed[p]))tasks.emplace_back(p,1);
            else if(s.rescue_water&&t.plant()&&!t.water&&t.unwatered>=1){
                bool want=false;
                for(auto&op:needs(o,p))want|=op.name=="WATER";
                if(want)tasks.emplace_back(p,2);
            }
        }
        int wheat=o.shed[0];
        while(!tasks.empty()){
            tuple<int,int,int,int> best{
                999,999,999,999
            };
            int bu=-1,bj=-1;
            for(int j=0;j<int(tasks.size());j++){
                auto [p,type]=tasks[j];
                for(int u=0;u<n;u++){
                    if(assignment[u].first>=0)continue;
                    int pos=o.positions[u],eta=dist(pos,p)+1;
                    if(type==1&&o.invs[u][0]<1){
                        if(wheat<=0)continue;
                        eta=dist(pos,access(pos))+1+dist(access(pos),p)+1;
                    }
                    if(o.hour+eta>24)continue;
                    auto key=tuple{
                        type==1?0:1,eta,int(m.routes[u].empty()?0:1),u
                    };
                    if(key<best){
                        best=key;
                        bu=u;
                        bj=j;
                    }
                }
            }
            if(bu<0)break;
            auto [p,type]=tasks[bj];
            assignment[bu]={
                p,type
            };
            if(type==1&&o.invs[bu][0]<1)wheat--;
            for(auto&rr:m.routes)rr.erase(remove(rr.begin(),rr.end(),p),rr.end());
            m.backlog.erase(remove(m.backlog.begin(),m.backlog.end(),p),m.backlog.end());
            m.routes[bu].insert(m.routes[bu].begin(),p);
            tasks.erase(tasks.begin()+bj);
        }
        return assignment;
    }
    void latestaff(const Obs&o,const Config&c){
        if(!s.late_hire||o.hour<=3)return;
        if(o.hour>=22){
            for(size_t u=o.positions.size();u<m.routes.size();u++)for(int p:m.routes[u])m.backlog.push_back(p);
            if(m.routes.size()>o.positions.size())m.routes.resize(o.positions.size());
            m.hire_target=int(o.positions.size())-1;
            return;
        }
        if(int(o.positions.size())-1>=s.max_hands)return;
        // At most one pending hire; failed orders are retried from actual state.
        if(m.routes.size()>o.positions.size())return;
        int wage=FIB[min(19,int(o.positions.size())-1)]*c.handmult;
        if(o.money<wage+50||m.backlog.empty())return;
        array<int,4> spawns={xy(4,4),xy(5,4),xy(4,5),xy(5,5)};
        int start=spawns[0],occupancy=999;
        for(int p:spawns){
            int n=count(o.positions.begin(),o.positions.end(),p);
            if(n<occupancy){occupancy=n;start=p;}
        }
        auto [all,goods]=plannednodes(o);
        array<int,NT> work{};vector<pair<int,int>> nodes;
        for(auto [p,w]:all)work[p]=w;
        set<int> owned;
        for(const auto&rr:m.routes)for(int p:rr)owned.insert(p);
        set<int> included;
        for(int p:m.backlog)if(work[p]&&!owned.count(p)&&included.insert(p).second)nodes.emplace_back(p,work[p]);
        if(nodes.empty())return;
        int saved=s.route_budget;
        // Hire settles this step; pickup and operations start no earlier than next.
        s.route_budget=min(saved,23-o.hour-2);
        auto rr=coverroute(o,start,nodes,false,true);
        s.route_budget=saved;
        int duration=1,prev=start;double value=0,inputs=0;
        // Unit movement precedes the HIRE market order. Actual spawn can differ
        // from its current-occupancy estimate: budget the worst of four corners.
        if(!rr.empty()){
            int worst=0;for(int p:spawns)worst=max(worst,dist(p,rr.front()));
            duration+=worst-dist(start,rr.front());
        }
        for(int p:rr){
            duration+=dist(prev,p)+work[p];prev=p;
            for(auto op:needs(o,p))value+=op.value;
            int item=m.newitem[p];
            if(item>=0&&item<5&&!o.tiles[p].active()&&!o.seeds[item])inputs+=costs[item];
        }
        duration+=dist(prev,access(prev))+1;
        auto req=routeinputs(o,rr);
        for(int i=0;i<NI;i++)if(req[i]){
            duration++;
            inputs+=max(0,req[i]-o.shed[i])*(i<NP?o.prices[i]+1:costs[i]);
        }
        if(o.hour+duration>23||o.money<wage+inputs+50||value<s.late_hire_margin*wage)return;
        for(int p:rr)m.backlog.erase(remove(m.backlog.begin(),m.backlog.end(),p),m.backlog.end());
        m.routes.push_back(rr);
        m.hire_target=int(o.positions.size());
    }
    JO execute(const Obs&o,const Config&c){
        if(s.manure_floor>0){
            int need=0,available=o.shed[8];
            for(const auto&iv:o.invs)available+=iv[8];
            for(int p=0;p<NT;p++)if(o.tiles[p].plant()&&m.haspolicy[p]){
                int age=o.day-o.tiles[p].placed;
                for(auto e:m.policy[p].events)if(e.item==8&&e.q<0&&e.age>=age&&e.age<=age+1&&o.tiles[p].fertilized<o.day+e.age-age)need-=e.q;
            }
            m.fertilizer_deficit=max(0,need-available);
        }
        replant(o,c);
        schedule(o);
        if(o.hour==4){
            int count=o.positions.size();
            for(int u=count;u<int(m.routes.size());u++)m.backlog.insert(m.backlog.end(),m.routes[u].begin(),m.routes[u].end());
            if(int(m.routes.size())>count)m.routes.resize(count);
            if(s.late_hire)m.hire_target=count-1;
        }
        latestaff(o,c);
        auto rescue=rescueassign(o);
        int feedreserve=0;
        for(int u=0;u<int(rescue.size());u++)if(rescue[u].second==1&&o.invs[u][0]<1)feedreserve++;
        auto&routes=m.routes;
        int hour=o.hour;
        bool last=o.day==29;
        Stock seeds=o.seeds,shed=o.shed,deposit{
        };
        JA actions;
        set<int> claimed;
        for(auto [p,type]:rescue)if(p>=0)claimed.insert(p);
        for(auto&rr:routes)rr.erase(remove_if(rr.begin(),rr.end(),[&](int p){
            return needs(o,p).empty();
        }),rr.end());
        int totalcarry=0;
        for(auto&iv:o.invs)totalcarry+=sumstock(iv);
        auto&courier=m.couriers;
        for(auto it=courier.begin();it!=courier.end();)if(*it>=int(o.positions.size()))it=courier.erase(it);
        else ++it;
        vector<int> payload(o.positions.size());
        for(int j=0;j<int(o.positions.size());j++){
            auto rj=routeinputs(o,routes.at(j));
            for(int i=0;i<NP;i++)payload[j]+=max(0,o.invs[j][i]-rj[i]);
            if(payload[j]==0)courier.erase(j);
        }
        int uncollected=plannednodes(o).second,excess=max(0,totalcarry+uncollected-90),promised=0;
        for(int j:courier)promised+=payload[j];
        if(!last&&hour>=7&&excess>promised){
            vector<pair<double,int>> options;
            for(int j=0;j<int(o.positions.size());j++){
                int pj=o.positions[j];
                if(courier.count(j)||payload[j]<3)continue;
                int dh=dist(pj,access(pj)),remaining=0,walking=0,pp=access(pj);
                for(int p:routes[j]){
                    remaining+=needs(o,p).size();
                    walking+=dist(pp,p);
                    pp=p;
                }
                if(hour+dh+1+remaining+walking>24)continue;
                options.emplace_back(double(payload[j])/(1+2*dh),j);
            }
            sort(options.rbegin(),options.rend());
            for(auto [score,j]:options){
                if(promised>=excess)break;
                courier.insert(j);
                promised+=payload[j];
            }
        }
        for(int u=0;u<int(o.positions.size());u++){
            int pos=o.positions[u],depot=access(pos),dhome=dist(pos,depot);
            auto inv=o.invs[u];
            auto&route=routes[u];
            auto req=routeinputs(o,route);
            Stock surplus{
            };
            int goods=0;
            for(int i=0;i<NP;i++){
                surplus[i]=max(0,inv[i]-req[i]);
                goods+=surplus[i];
            }
            if(rescue[u].first>=0){
                auto [target,type]=rescue[u];
                JA act;
                if(type==1&&inv[0]<1){
                    if(dhome)act=moveact(pos,depot);
                    else if(shed[0]>0){
                        act=JA{
                            "PICKUP","WHEAT",1
                        };
                        shed[0]--;
                        feedreserve--;
                    }
                    else act=JA{
                        "PASS"
                    };
                }
                else if(pos!=target)act=moveact(pos,target);
                else act=JA{
                    type==1?"FEED":"WATER"
                };
                actions.push_back(act);
                continue;
            }
            double delivery_value=0;
            if(s.courier_value>0)for(int i=0;i<NP;i++)delivery_value+=surplus[i]*o.prices[i];
            int remainingwork=0,pp=depot;
            if(s.courier_value>0)for(int p:route){remainingwork+=dist(pp,p)+needs(o,p).size();pp=p;}
            bool early_delivery=s.courier_value>0&&delivery_value>=s.courier_value*(1+2*dhome)&&hour+dhome+1+remainingwork<=23;
            bool must_return=goods&&(early_delivery||courier.count(u)||(last&&hour+dhome>=21)||(route.empty()&&hour+dhome<=22)||(o.money<100&&hour>5&&dhome<4));
            if(must_return&&hour+dhome<=22){
                JA act;
                if(dhome)act=moveact(pos,depot);
                else{
                    int room=c.shedcap-sumstock(shed);
                    bool hasreq=false;
                    for(int i=0;i<NI;i++)if(inv[i]&&req[i])hasreq=true;
                    if(sumstock(inv)<=room&&!hasreq){
                        act=JA{
                            "DROP"
                        };
                        for(int i=0;i<NI;i++){
                            deposit[i]+=inv[i];
                            shed[i]+=inv[i];
                        }
                    }
                    else{
                        int best=-1;
                        for(int i=0;i<NP;i++)if(surplus[i]>0&&(best<0||o.prices[i]>o.prices[best]))best=i;
                        if(best>=0&&room>0){
                            int q=min(surplus[best],room);
                            act=JA{
                                "PLACE",names[best],q
                            };
                            deposit[best]+=q;
                            shed[best]+=q;
                        }
                        else act=JA{
                            "PASS"
                        };
                    }
                }
                actions.push_back(act);
                continue;
            }
            if(route.empty()&&!m.backlog.empty()){
                int bp=-1,bd=999;
                for(int p:m.backlog){
                    auto ops=needs(o,p);
                    if(!ops.empty()&&hour+dist(pos,p)+int(ops.size())<(last?22:24)&&dist(pos,p)<bd){
                        bp=p;
                        bd=dist(pos,p);
                    }
                }
                if(bp>=0){
                    route.push_back(bp);
                    m.backlog.erase(find(m.backlog.begin(),m.backlog.end(),bp));
                }
            }
            if(route.empty()){
                tuple<int,int,int,int,int> best{
                    999,999,999,999,999
                };
                int bp=-1,bv=-1,bj=-1;
                for(int v=0;v<int(routes.size());v++){
                    if(v==u||routes[v].empty()||v>=int(o.positions.size()))continue;
                    for(int j=0;j<int(routes[v].size());j++){
                        int p=routes[v][j];
                        if(claimed.count(p))continue;
                        if(j==0&&dist(o.positions[v],p)<=dist(pos,p))continue;
                        if(hour+dist(pos,p)+2<(last?22:24)){
                            auto key=tuple{
                                dist(pos,p),v,j,X(p),Y(p)
                            };
                            if(key<best){
                                best=key;
                                bp=p;
                                bv=v;
                                bj=j;
                            }
                        }
                    }
                }
                if(bp>=0){
                    routes[bv].erase(routes[bv].begin()+bj);
                    route.push_back(bp);
                }
            }
            req=routeinputs(o,route);
            Stock missing{
            };
            bool hasmissing=false;
            for(int i=0;i<NI;i++){
                missing[i]=max(0,req[i]-inv[i]);
                hasmissing|=missing[i]>0;
            }
            if(dhome==0){
                int pick=-1;
                for(int i:{
                    9,10,11,0,8
                })if(missing[i]>0&&shed[i]-(i==0?feedreserve:0)>0){
                    pick=i;
                    break;
                }
                if(pick>=0){
                    int q=min(missing[pick],shed[pick]-(pick==0?feedreserve:0));
                    shed[pick]-=q;
                    actions.push_back(JA{
                        "PICKUP",names[pick],q
                    });
                    continue;
                }
                if(hasmissing&&hour<3&&o.money>50){
                    actions.push_back(JA{
                        "PASS"
                    });
                    continue;
                }
            }
            if(route.empty()){
                actions.push_back(JA{
                    "PASS"
                });
                continue;
            }
            int p=route[0];
            auto ops=needs(o,p);
            int choice=pickop(ops,inv,seeds);
            if(choice<0){
                bool avail=false;
                for(int i=0;i<NI;i++)if(missing[i]&&shed[i]>0)avail=true;
                if(avail&&hour+dhome+2<24){
                    actions.push_back(moveact(pos,depot));
                    continue;
                }
                pair<int,int> best{
                    999,999
                };
                for(int j=1;j<int(route.size());j++)if(pickop(needs(o,route[j]),inv,seeds)>=0)best=min(best,pair{
                    dist(pos,route[j]),j
                });
                if(best.second<999){
                    int pp=route[best.second];
                    route.erase(route.begin()+best.second);
                    route.insert(route.begin(),pp);
                    p=route[0];
                    ops=needs(o,p);
                    choice=pickop(ops,inv,seeds);
                }
                else{
                    JA act;
                    if(goods&&hour+dhome<=22){
                        if(dhome)act=moveact(pos,depot);
                        else{
                            act=JA{
                                "DROP"
                            };
                            for(int i=0;i<NI;i++){
                                deposit[i]+=inv[i];
                                shed[i]+=inv[i];
                            }
                        }
                    }
                    else act=JA{
                        "PASS"
                    };
                    actions.push_back(act);
                    continue;
                }
            }
            if(choice<0){
                actions.push_back(JA{
                    "PASS"
                });
                continue;
            }
            auto op=ops[choice];
            if(op.name=="PLANT"&&hour+dist(pos,p)+2>24){
                actions.push_back(JA{
                    "PASS"
                });
                continue;
            }
            if(last&&op.name=="HARVEST"&&hour+dist(pos,p)+dist(p,access(p))+2>23){
                actions.push_back(JA{
                    "PASS"
                });
                continue;
            }
            claimed.insert(p);
            if(pos!=p){
                actions.push_back(moveact(pos,p));
                continue;
            }
            JA act{
                op.name
            };
            if(op.name=="PLANT"||op.name=="PLACE")act.push_back(J(names[op.res]));
            if(op.name=="PLANT")seeds[op.res]--;
            actions.push_back(act);
        }
        JA hands;
        for(int u=1;u<int(actions.size());u++)hands.push_back(actions[u]);
        return JO{
            {
                "farmer",actions[0]
            },{
                "hands",hands
            },{
                "market",marketorders(o,c,deposit)
            }
        };
    }
    JO act(const J&obs,const J&cfg){
        auto o=parseobs(obs);
        auto c=parsecfg(cfg);
        if(o.step==0||o.day<m.day)m=Memory{
        };
        if(o.day!=m.day)plan(o,c);
        return execute(o,c);
    }
    J debug()const{
        JA pp,pol,rs;
        for(auto [p,i]:m.projects)pp.push_back(JA{
            X(p),Y(p),names[i]
        });
        for(int p=0;p<NT;p++)if(m.haspolicy[p]){
            JA ev;
            for(auto e:m.policy[p].events)ev.push_back(JA{
                e.age,e.item,e.q
            });
            pol.push_back(JA{
                X(p),Y(p),m.policy[p].length,ev
            });
        }
        for(auto&rr:m.routes){
            JA r;
            for(int p:rr)r.push_back(jpos(p));
            rs.push_back(r);
        }
        return JO{
            {
                "day",m.day
            },{
                "new",pp
            },{
                "policy",pol
            },{
                "routes",rs
            },{
                "daily_records",m.daily_records
            },{
                "hire_target",m.hire_target
            },{
                "c2_service_calls",m.c2_service_calls
            },{
                "c2_refresh_changes",m.c2_refresh_changes
            }
        };
    }
};
struct Context{
    Agent agent;
    string result;
    Context(){
        agent.s.update(bj::parse("{\"animal_dp\":1,\"market_wait\":21,\"storage_cap\":60,\"route_value\":3,\"replant_hour\":14,\"cycle_gap\":0,\"rescue_hour\":20,\"rescue_water\":1}"));
    }
};
extern "C" {
    void* dp_create(){
        try{
            return new Context;
        }
        catch(...){
            return nullptr;
        }
    }
    void dp_destroy(void*ctx){
        delete static_cast<Context*>(ctx);
    }
    const char* dp_call(void*ctx,const char*input){
        auto*c=static_cast<Context*>(ctx);
        try{
            J r=bj::parse(input);
            c->agent.s.update(get(r,"settings"));
            auto a=c->agent.act(get(r,"observation"),get(r,"configuration"));
            c->result=bj::serialize(a);
        }
        catch(const exception&e){
            c->result=bj::serialize(JO{
                {
                    "error",string(e.what())
                }
            });
        }
        return c->result.c_str();
    }
    const char* dp_debug(void*ctx){
        auto*c=static_cast<Context*>(ctx);
        c->result=bj::serialize(c->agent.debug());
        return c->result.c_str();
    }
}
#ifndef DP_LIBRARY
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Context c;
    string line;
    while(getline(cin,line)){
        try{
            J r=bj::parse(line);
            c.agent.s.update(get(r,"settings"));
            auto a=c.agent.act(get(r,"observation"),get(r,"configuration"));
            cout<<bj::serialize(a)<<'\n'<<flush;
        }
        catch(const exception&e){
            cerr<<e.what()<<'\n';
            return 1;
        }
    }
}
#endif
