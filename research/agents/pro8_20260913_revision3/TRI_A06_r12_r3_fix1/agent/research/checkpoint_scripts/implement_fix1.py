from pathlib import Path
w=Path(__file__).resolve().parent
p=w/'candidate/policy/ongoing_supply_admission.inc'
s=p.read_text();s=s.replace('int ongoing_supply_previews=0,ongoing_supply_skips=0,ongoing_supply_preview_errors=0;', '''int ongoing_supply_previews=0,ongoing_supply_skips=0,ongoing_supply_preview_errors=0;
// Fix1: a reachable production bonus is not a terminal cash profit. Only when
// the real terminal is within a bounded horizon, compare both complete routes
// through liquidation. No residual stock, standing yield or fertilizer is
// credited at terminal. Earlier profitable r3 maintenance is left unchanged.
int ongoing_terminal_checks=0,ongoing_terminal_rejected=0,ongoing_terminal_errors=0;
std::array<double,2> ongoing_terminal_own_gain{},ongoing_terminal_margin_gain{};
std::string ongoing_terminal_last="null";
static bool terminal_supply_profitable(const std::array<double,2>&own,
                                       const std::array<double,2>&margin){
 for(int k=0;k<2;k++)if(!std::isfinite(own[k])||!std::isfinite(margin[k])||
                       own[k]<=1e-6||margin[k]<=1e-6)return false;
 return true;
}''')
s=s.replace('  if(!extra||!preserves)use_fallback();', r'''  if(!extra||!preserves){use_fallback();return;}
  // 48 is a computation bound, NOT a crop/seed/opponent rule. We require an
  // actual terminal endpoint so there is no arbitrary liquidation valuation.
  if(actual.step>=0 && 719-actual.step<=48){
   ongoing_terminal_checks++;
   auto terminal=[&](const Controller&policy,double rival_scale){
    auto roll=policy;roll.ongoing_supply_previewing=true;
    roll.r11_evaluating=true; // same bounded labor-search setting as r3 profile
    roll.previous_step=actual.step-1;roll.core.last_step=actual.step-1;
    roll.call_origin=CallOrigin::PublicPrediction;
    // PublicFlowScenario uses only visible rival plants/animals and the public
    // flow forecast already computed by this policy. Opponent private storage
    // is not accessed. Scale zero is a separate no-new-supply stress scenario.
    fastkag::PublicFlowScenario world(actual,model.rival,rival_scale);
    int ticks=0;
    while(!world.done() && ticks<48){auto v=world.view();world.advance(roll.act(v));++ticks;}
    if(!world.done())throw std::runtime_error("terminal supply horizon");
    return std::array<double,2>{world.own_cash(),world.rival_cash()};
   };
   std::ostringstream audit;audit.precision(17);
   audit<<"{\"step\":"<<actual.step<<",\"scenarios\":[";
   for(int k=0;k<2;k++){
    double scale=k?s.supply:0.;auto a=terminal(fallback,scale),b=terminal(*this,scale);
    ongoing_terminal_own_gain[k]=b[0]-a[0];
    ongoing_terminal_margin_gain[k]=b[0]-a[0]-s.competition*(b[1]-a[1]);
    if(k)audit<<",";
    audit<<"{\"public_supply_scale\":"<<scale<<",\"keep_own_cash\":"<<a[0]<<",\"keep_rival_cash\":"<<a[1]
         <<",\"supply_own_cash\":"<<b[0]<<",\"supply_rival_cash\":"<<b[1]
         <<",\"own_gain\":"<<ongoing_terminal_own_gain[k]<<",\"margin_gain\":"<<ongoing_terminal_margin_gain[k]<<"}";
   }
   const bool profitable=terminal_supply_profitable(ongoing_terminal_own_gain,ongoing_terminal_margin_gain);
   audit<<"],\"accepted\":"<<(profitable?"true":"false")<<"}";ongoing_terminal_last=audit.str();
   if(!profitable){ongoing_terminal_rejected++;use_fallback();return;}
  }''')
s=s.replace('  ongoing_supply_preview_errors++;use_fallback();','  ongoing_supply_preview_errors++;if(actual.step>=671)ongoing_terminal_errors++;use_fallback();')
p.write_text(s)
p=w/'candidate/policy/triad.hpp';s=p.read_text();needle='std::ostringstream o;o<<"{\\\"r3_supply_previews\\\":"';assert needle in s
s=s.replace(needle,'std::ostringstream o;o<<"{\\\"r3_fix1_terminal_checks\\\":"<<ongoing_terminal_checks<<",\\\"r3_fix1_terminal_rejected\\\":"<<ongoing_terminal_rejected<<",\\\"r3_fix1_terminal_errors\\\":"<<ongoing_terminal_errors<<",\\\"r3_fix1_terminal_last\\\":"<<ongoing_terminal_last<<",\\\"r3_supply_previews\\\":"')
p.write_text(s)
p=w/'candidate/main.py';s=p.read_text().replace('TRI_A06_r12_r3: approved ongoing maintenance supply; awaiting central win-rate acceptance.','TRI_A06_r12_r3_fix1: terminal net-cash admission for r3 maintenance supply.');p.write_text(s)
