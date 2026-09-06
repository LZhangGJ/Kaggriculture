#include "f3_policy.cpp"
#include <functional>

namespace {
int checks=0;
void require(bool okay,const char*message){
    checks++;
    if(!okay)throw std::runtime_error(message);
}
double quantity(const Flow&flow,int item){
    double q=0;for(const auto&day:flow)q+=day[item];return q;
}
void equal(const Flow&a,const Flow&b,const char*message){
    require(ProjectForecastLedger::same(a,b),message);
}
void ledger_cases(){
    fastkag::Simulator env(fastkag::Config{},64000000);
    econrl::Input in{0,0,0,0,&env.farms()[0],&env.farms()[1],&env.privates()[0],&env.market(),&env.shops()};
    Obs o=observe(in);Config cfg;Policy policy("",17,0);
    auto&s=policy.ctx.agent.s;
    int streams_tested=0,repeat_regressions=0;
    for(bool repeat:{false,true})for(int day:{0,1,6,12,20,26,28,29}){
        o.day=day;o.step=24*day;s.crop_repeat=repeat;
        Economy eco(o,cfg,s);Flow px{};
        for(int d=0;d<30;d++)for(int i=0;i<9;i++)px[d][i]=BASE[i];
        std::vector<std::pair<int,Flow>> streams;
        for(int kind:{0,1,2,3,4,9,10,11}){
            Flow f{};
            if(kind>=9)f=policy.ctx.agent.animalplan(kind,day,day,nullptr,px).flow;
            else{
                auto cp=eco.newcrop(kind,px);if(!cp.valid)continue;f=cp.flow;
                Flow once=cropflow(kind,day,day,cp.variant);
                if(repeat&&quantity(f,kind)>quantity(once,kind)){
                    repeat_regressions++;
                    if(day==0)std::cout<<"REPRO "<<names[kind]<<" old_total="<<quantity(f,kind)
                        <<" old_bug_removed="<<quantity(once,kind)<<" fixed_remaining=0\n";
                }
            }
            streams.emplace_back(kind,f);streams_tested++;
        }
        for(const auto&[oldkind,oldflow]:streams){
            Flow background{};background[day][6]=7;background[day][0]=-3;
            Flow other{};other[29][7]=4;other[day][8]=-2;
            std::array<Flow,NT> entries{};entries[44]=oldflow;entries[43]=other;
            Flow initial=background;addflow(initial,oldflow);addflow(initial,other);
            ProjectForecastLedger original;
            original.capture(day,background,{{44,oldkind},{43,11}},entries,initial);
            equal(original.total(),initial,"capture");
            auto cancelled=original;auto remainder=cancelled.replace(day,44,oldkind,-1,Flow{});
            auto expected=background;addflow(expected,other);
            equal(remainder,expected,"cancel removed all cycles, inputs and outputs");
            equal(cancelled.replace(day,44,-1,-1,Flow{}),expected,"repeated cancel is idempotent");
            equal(original.total(),initial,"candidate copy did not mutate baseline");
            equal(cancelled.replace(day,44,-1,oldkind,oldflow),initial,"cancel restore roundtrip");
            for(const auto&[nextkind,newflow]:streams){
                auto branch=original;auto changed=branch.replace(day,44,oldkind,nextkind,newflow);
                auto wanted=expected;addflow(wanted,newflow);
                equal(changed,wanted,"replace full old with full new once");
                equal(branch.replace(day,44,nextkind,nextkind,newflow),wanted,"same replacement does not double book");
                equal(branch.replace(day,44,nextkind,oldkind,oldflow),initial,"A B A roundtrip");
                equal(branch.committed,background,"committed assets unchanged");
            }
            bool rejected=false;try{original.replace(day+1,44,oldkind,-1,Flow{});}catch(const std::exception&){rejected=true;}
            require(rejected,"cross-day edit rejected");equal(original.total(),initial,"failed edit unchanged");
            rejected=false;try{original.replace(day,44,-1,-1,Flow{});}catch(const std::exception&){rejected=true;}
            require(rejected,"wrong prior identity rejected");equal(original.total(),initial,"identity rejection atomic");
        }
    }
    require(repeat_regressions>0,"must exercise old repeat-cycle bug");
    std::cout<<"STREAMS "<<streams_tested<<" REPEAT_REGRESSIONS "<<repeat_regressions<<"\n";
}
void lifecycle_cases(){
    int games=0,edits=0,held_days=0;
    for(uint64_t seed:{64000000ULL,64000001ULL,64000007ULL,64000013ULL})for(int mode:{0,3}){
        fastkag::Simulator env(fastkag::Config{},seed);Policy p("",seed+71,mode);
        int previous=-1;
        while(!env.done()){
            econrl::Input in{env.step_count(),env.day(),env.hour(),0,&env.farms()[0],&env.farms()[1],&env.privates()[0],&env.market(),&env.shops()};
            bool new_day=in.day!=previous;previous=in.day;
            auto a=p.act(in);auto&m=p.ctx.agent.m;
            equal(m.forecast,m.forecast_ledger.total(),"live forecast reconstruction");
            if(new_day){
                require(m.forecast_ledger.day==in.day,"daily reset");
                require(m.projects.size()==m.forecast_ledger.entries.size(),"project ownership count");
                for(auto[pos,kind]:m.projects){
                    auto e=m.forecast_ledger.find(pos);require(e&&e->item==kind,"project moved with exact stream");
                }
                for(int item=9;item<12;item++)held_days+=in.priv->shed[item]>0;
                edits+=p.rl.records.back().changed;
            }
            std::array<fastkag::PlayerAction,2> actions;actions[0]=a;env.step(actions);
        }
        require(p.rl.records.size()==30,"full daily sequence");games++;
    }
    require(edits>0,"random policy must exercise replacements");
    std::cout<<"LIFECYCLE_GAMES "<<games<<" EDITS "<<edits<<" HELD_ANIMAL_DAYS "<<held_days<<"\n";
}
}
int main(){
    try{ledger_cases();lifecycle_cases();std::cout<<"PASS checks="<<checks<<"\n";}
    catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
