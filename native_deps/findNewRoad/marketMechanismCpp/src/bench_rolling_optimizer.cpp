#include "rolling_optimizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {
using namespace g001::rolling;

struct Case { std::string name; int ticks,candidates,scenarios;std::size_t beam,depth,repetitions; };

PersistentOption make_option(int index){PersistentOption o;o.kind=static_cast<g001::option::Kind>(index%7);o.product=(index/7+index)%9;o.quota=1+(index%6);o.window_steps=8+4*(index%5);o.reservation_price=20+10*(index%20);o.target_inventory=index%12;o.impact_limit=2+(index%8);return o;}

void run(const Case& c){CurrentState state;state.step=240;state.own_money=7000;state.opponent_money=7200;state.market_inventory.fill(10000);for(std::size_t p=0;p<state.own_stock.size();++p)state.own_stock[p]=3+(p%4);FixedForecast forecast;forecast.shed_capacity=100;forecast.liquidate_own_at_end=true;forecast.ticks.resize(c.ticks);for(int i=0;i<c.ticks;++i){auto&t=forecast.ticks[i];t.step=state.step+i;t.sale_window=i%4==0;t.decision_epoch=i==0||i==c.ticks/3||i==2*c.ticks/3;if(i%8==3)t.production[(i/8)%9]=2;if(i%12==5)t.town_drain[(i/12)%8]=1;if(i%24==7)t.requirements.push_back({RequirementKind::CriticalPurchase,300,-1,0});if(i%24==11)t.requirements.push_back({RequirementKind::Feed,0,0,1});}
    std::vector<PersistentOption> options;for(int i=0;i<c.candidates;++i)options.push_back(make_option(i));std::vector<Scenario> scenarios(c.scenarios);for(int s=0;s<c.scenarios;++s){scenarios[s].weight=1.0/c.scenarios;for(int p=0;p<9;++p)scenarios[s].belief_stock[p]=(s+p)%18;if(s%2==0)scenarios[s].dumps.push_back({state.step+c.ticks/2,s%9,2+s%12});if(s%3==0)scenarios[s].dumps.push_back({state.step+3*c.ticks/4,(s+3)%9,1+s%7});}
    Config cfg;cfg.beam_width=c.beam;cfg.max_decisions=c.depth;cfg.threads=1;cfg.cvar_fraction=.25;
    for(int i=0;i<8;++i)(void)optimize(state,forecast,scenarios,options,cfg);
    std::vector<double> us;us.reserve(c.repetitions);std::size_t evaluated=0;for(std::size_t i=0;i<c.repetitions;++i){auto start=std::chrono::steady_clock::now();auto result=optimize(state,forecast,scenarios,options,cfg);auto stop=std::chrono::steady_clock::now();us.push_back(std::chrono::duration<double,std::micro>(stop-start).count());evaluated=result.evaluated_sequences;}
    std::sort(us.begin(),us.end());auto q=[&](double f){return us[static_cast<std::size_t>(f*(us.size()-1))]/1000.0;};std::cout<<std::fixed<<std::setprecision(3)<<c.name<<" ticks="<<c.ticks<<" candidates="<<c.candidates<<" scenarios="<<c.scenarios<<" beam="<<c.beam<<" depth="<<c.depth<<" reps="<<c.repetitions<<" evaluated="<<evaluated<<" p50_ms="<<q(.50)<<" p95_ms="<<q(.95)<<" p99_ms="<<q(.99)<<" max_ms="<<us.back()/1000.0<<'\n';
}
}
int main(){run({"grid-small",24,7,9,64,3,1000});run({"grid-medium",48,14,9,64,3,300});run({"grid-large",72,28,15,64,3,100});run({"deploy-conservative",48,7,9,16,2,1000});}
