#include "../policy/search.hpp"
#include <iostream>
#include <cstdlib>
static int checks=0;
static void ck(bool ok,const char*what){++checks;if(!ok){std::cerr<<what<<"\n";std::exit(1);}}
int main(){
 for(int mode:{20,22,23,24,25}){
  fastkag::Simulator env(fastkag::Config{},12345);
  dp7::View v{696,29,0,env.farms()[0],env.farms()[1],env.privates()[0],env.market(),env.shops()};
  triad::Settings s;s.scenario=0;s.layout=mode;s.competition=2;s.max_animals=0;s.max_land=1;s.max_hands=1;
  triad::SearchController c(s);double own=v.own.money,enemy=v.opponent.money;auto stock=v.priv.shed;
  ck(c.regime_day==-1,"new game has no inherited regime");
  c.act(v);
  ck(c.regime_day==29,"one terminal continuation comparison is recorded");
  ck(c.regime_competition==2.,"ties preserve the declared incumbent");
  ck(c.base.competition==c.regime_competition,"selected regime is installed in actual controller");
  ck(c.nominal.competition==2,"nominal configuration is not overwritten");
  ck(v.own.money==own&&v.opponent.money==enemy&&v.priv.shed==stock,"forecast never mutates caller observation");
  ck(c.last_regime.find("public conditional terminal rollouts")!=std::string::npos,"forecast is labelled conditional");
  int margin=mode==23?1500:mode==25?0:500;
  ck(c.last_regime.find("\"safety_margin\":"+std::to_string(margin))!=std::string::npos,"decision audit records its actual threshold");
 }
 {
  fastkag::Simulator env(fastkag::Config{},987);
  auto own=env.farms()[0],other=env.farms()[1];other.money=100000;
  dp7::View v{0,0,0,own,other,env.privates()[0],env.market(),env.shops()};
  competitive::Flow flows{};flows[0][dp7::W]=-240;
  fastkag::PublicFlowScenario retained(v,flows,1.,nullptr,false),consumed(v,flows,1.,nullptr,true);
  fastkag::PlayerAction pass;
  for(int h=0;h<19;h++){retained.advance(pass);consumed.advance(pass);}
  ck(consumed.rival_cash()<retained.rival_cash(),"forecast input consumption prevents fictitious warehouse blockage");
  ck(v.opponent.money==100000,"synthetic procurement never changes public caller cash");
  ck(consumed.own_cash()==retained.own_cash(),"proxy-input cleanup never transfers money to own player");
 }
 ck(!triad::learned::available,"no learned model is present");
 std::cout<<"{\"checks\":"<<checks<<",\"failed\":0,\"scope\":\"terminal regime ties/installation/reset/observation immutability, synthetic proxy-input consumption, model unavailable\"}\n";
}
