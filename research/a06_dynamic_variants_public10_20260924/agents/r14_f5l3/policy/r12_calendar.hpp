#pragma once
#include "planner.hpp"
#include <array>
#include <vector>
#include <limits>
#include <cmath>
#include <sstream>

namespace triad::r12 {
// A finite stopping-calendar DAG. A terminal label is the VALUE OF THE ENTIRE
// counterfactual cash-flow path, not a local sale quote. There are no additive
// day rewards here: adding those again would double-count maintenance/returns.
// H[d] = max(eligible terminal labels at d, H[d+1]); terminal labels include
// all costs up to harvest and market effects on all other committed streams.
// KEEP is an explicit incumbent label and wins exact/numerical ties.
struct Label { int finish=-1; double value=-std::numeric_limits<double>::infinity(); };
struct Decision {int index=-1;std::array<int,31> best{};};
inline Decision solve(const std::vector<Label>& labels,int begin,int keep) {
 Decision out;out.best.fill(-1);
 if(begin<0||begin>=30||keep<0||keep>=int(labels.size()))return out;
 auto better=[&](int i,int j){
  if(i<0||!std::isfinite(labels[i].value))return j;
  if(j<0||!std::isfinite(labels[j].value))return i;
  if(labels[i].value>labels[j].value+1e-8)return i;
  if(labels[j].value>labels[i].value+1e-8)return j;
  if(i==keep||j==keep)return keep;
  if(labels[i].finish!=labels[j].finish)return labels[i].finish<labels[j].finish?i:j;
  return std::min(i,j);
 };
 for(int day=29;day>=begin;--day){
  out.best[day]=out.best[day+1];
  for(int i=0;i<int(labels.size());++i)if(labels[i].finish==day)
   out.best[day]=better(i,out.best[day]);
 }
 out.index=out.best[begin];return out;
}
struct Stats {
 long long calls=0,labels=0,changes=0,existing_changes=0;
 std::string last="null",last_change="null";
};
// Diagnostic only, mirrors the installed Planner::value() decomposition. The
// primary score always calls Planner::value itself; this cannot change actions.
struct CashParts {double discounted_own=0,discounted_rival=0,wages=0,work=0,liquidity=0,score=0;};
inline CashParts decompose(const competitive::Planner& m,const dp7::View&o,const competitive::Asset&a){
 CashParts r;std::array<double,9> inv{};for(int i=0;i<9;i++)inv[i]=o.market.inventory[i];double balance=o.own.money;
 for(int d=m.day;d<30;d++){
  double own=a.fixed[d],enemy=0;
  for(int i=0;i<9;i++){
   inv[i]-=m.dem[d][i]*.5;
#if R2_SALE_CLOCK_MODE >= 2
   double share=(m.cfg.r14_forecast&2)?m.rival_early_share[i]:.5;double early=m.cfg.supply*m.rival[d][i]*share,late=m.cfg.supply*m.rival[d][i]-early;
#else
   double early=m.cfg.supply*m.rival[d][i]*.5,late=early;
#endif
   enemy+=m.transact(i,inv[i],early);
   own+=m.transact(i,inv[i],a.f[d][i]);
   enemy+=m.transact(i,inv[i],late);inv[i]-=m.dem[d][i]*.5;
  }
  const double wage=m.wages(a.labor[d]),work=m.cfg.action_cost*a.labor[d];
  own-=wage+work;balance+=own;r.liquidity+=std::max(0.,-balance);
  const double discount=std::pow(1+m.cfg.discount*std::max(0.,1-o.own.money/20000.),d-m.day);
  r.discounted_own+=own/discount;r.discounted_rival+=enemy/discount;r.wages+=wage/discount;r.work+=work/discount;
 }
 r.score=r.discounted_own-m.cfg.competition*r.discounted_rival-m.cfg.risk*r.liquidity;
 return r;
}
inline std::string json(const CashParts&r){std::ostringstream o;o.precision(17);o<<"{\"discounted_own_conditional\":"<<r.discounted_own<<",\"discounted_rival_conditional\":"<<r.discounted_rival<<",\"wages_conditional\":"<<r.wages<<",\"action_cost_conditional\":"<<r.work<<",\"liquidity_deficits\":"<<r.liquidity<<",\"score\":"<<r.score<<"}";return o.str();}
}
