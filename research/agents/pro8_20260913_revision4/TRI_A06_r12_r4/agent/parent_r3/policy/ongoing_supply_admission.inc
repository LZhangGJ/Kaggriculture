// R3: do not purchase inputs merely because the first reservation is short.
// Existing collection/recovery can already supply all approved service. Check
// that exact resource-and-route possibility in a copied, current-day scenario.
// No economic score, future cash credit, rival private inventory or random tape.
bool ongoing_supply_previewing=false;
int ongoing_supply_previews=0,ongoing_supply_skips=0,ongoing_supply_preview_errors=0;
void prepare_with_ongoing_supply(const View&planning,const View&actual,int released){
 core.prepare_orders(planning,actual,released);
#if TRI_A06_R12_R3_FERT_SUPPLY
 if(ongoing_supply_previewing || core.day<15 || core.day>=29)return;
 bool purchase=false;
 for(auto a:core.queue)purchase|=a.op==Op::BUY_PRODUCT&&int(a.item)==F&&a.quantity>0;
 if(!purchase)return;
 const auto js=core.jobs(planning,true);
 if(core.approved_ongoing_fertilizer_gap(actual,js)<=0)return;
 std::array<bool,100>wanted{};
 for(const auto&j:js){
  if(j.pos<0||j.pos>=100||j.needs[F]<=0)continue;
  const auto&t=actual.own.tiles[j.pos];
  if(!plant(t)||!ongoing(int(t.crop))||!core.fertilize_due(t)||
     core.crop_kind[j.pos]!=int(t.crop)||core.crop_birth[j.pos]!=t.planted_day||
     core.crop_fertilize[j.pos]<=0)continue;
  bool f=false,w=t.watered_today,replacement=false;
  for(auto a:j.actions){f|=a.op==Op::FERTILIZE;w|=a.op==Op::WATER;
   replacement|=a.op==Op::DIG||a.op==Op::PLANT||a.op==Op::PLACE;}
  wanted[j.pos]=f&&w&&!replacement;
 }
 ongoing_supply_previews++;
 auto fallback=*this;
 fallback.core.suppress_ongoing_fertilizer_supply=true;
 fallback.core.prepare_orders(planning,actual,released);
 auto use_fallback=[&](){
  fallback.core.suppress_ongoing_fertilizer_supply=false;
  core=std::move(fallback.core);ongoing_supply_skips++;
 };
 try{
  // A profile records only realized, valid own-unit maintenance within this
  // conditional current-day execution. It is not cash, a future crop balance,
  // or an assertion about the opponent's response.
  auto profile=[&](const Controller&policy){
   auto roll=policy;roll.ongoing_supply_previewing=true;
   roll.r11_evaluating=true; // Do not nest discretionary labor searches.
   roll.previous_step=actual.step-1;roll.core.last_step=actual.step-1;
   roll.call_origin=CallOrigin::PublicPrediction;
   fastkag::ObservedDayScenario world(actual);
   std::array<bool,100>covered{};
   while(!world.finished()){
    auto v=world.view();auto out=roll.act(v);
    if(v.hour==23){
     // Read after own actions and before the daily WATER reset. A bought unit
     // only counts if a feasible PICKUP/travel/FERTILIZE/WATER chain used it.
     auto post=world.project_units(out.units,-1);
     for(int p=0;p<100;p++)if(wanted[p]){
      const auto&a=actual.own.tiles[p];const auto&t=post.farms()[0].tiles[p];
      covered[p]=plant(t)&&t.crop==a.crop&&t.planted_day==a.planted_day&&
                 t.watered_today&&t.fertilized_until_day>=actual.day;
     }
     break;
    }
    world.advance(out);
   }
   return covered;
  };
  const auto baseline=profile(fallback);
  bool complete=true;for(int p=0;p<100;p++)if(wanted[p]&&!baseline[p])complete=false;
  if(complete){use_fallback();return;}
  // Resource shortfall alone is insufficient: supplying it must actually add
  // approved maintenance, without removing service the baseline could perform.
  const auto supplied=profile(*this);
  bool extra=false,preserves=true;
  for(int p=0;p<100;p++)if(wanted[p]){
   extra|=supplied[p]&&!baseline[p];preserves&=!baseline[p]||supplied[p];
  }
  if(!extra||!preserves)use_fallback();
 }catch(const std::exception&){
  // Fail closed to parent preparation. An uncertain projection must not spend
  // additional money. Normal resource recovery remains enabled in live play.
  ongoing_supply_preview_errors++;use_fallback();
 }
#endif
}
