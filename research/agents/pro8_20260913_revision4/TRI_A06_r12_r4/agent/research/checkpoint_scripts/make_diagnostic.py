from pathlib import Path
w=Path(__file__).resolve().parent
p=w/'diagnostic_r3/policy/ongoing_supply_admission.inc'; s=p.read_text()
s=s.replace('bool ongoing_supply_previewing=false;', '''struct SupplyDiagnosticProfile {
 std::array<bool,100> covered{}; double cash=0; Counts stock{}; std::array<int,9> yield{};int work=0;
 bool operator[](int p)const{return covered[p];}
};
std::string supply_audit_last="null";
bool ongoing_supply_previewing=false;''')
s=s.replace('std::array<bool,100>covered{};', 'SupplyDiagnosticProfile result;auto&covered=result.covered;')
s=s.replace('auto v=world.view();auto out=roll.act(v);','auto v=world.view();auto out=roll.act(v);for(auto a:out.units)result.work+=a.op!=Op::PASS;')
s=s.replace('     break;','     world.advance(out);break;')
s=s.replace('   return covered;', '''   result.cash=world.own().money;result.stock=world.inventory().shed;
   for(auto&b:world.inventory().inventories)add(result.stock,b);
   for(const auto&t:world.own().tiles){if(plant(t))result.yield[int(t.crop)]+=t.yield_units;else if(animal(t))result.yield[product[int(t.animal)-9]]+=t.yield_units;}
   return result;''')
s=s.replace('  bool extra=false,preserves=true;', r'''  if(!prediction()){
   std::ostringstream j;j.precision(17);j<<"{\"step\":"<<actual.step<<",\"actual_fert_price\":"<<actual.market.prices[F]<<",\"forecast_next\":[";
   for(int k=0;k<9;k++){if(k)j<<",";j<<prices[std::min(29,actual.day+1)][k];}j<<"],\"profiles\":[";
   auto write=[&](int q,const auto&r){j<<"{\"buy\":"<<q<<",\"cash\":"<<r.cash<<",\"work\":"<<r.work<<",\"covered\":[";bool c=false;for(int p=0;p<100;p++)if(r.covered[p]){if(c)j<<",";c=true;j<<p;}j<<"],\"stock\":[";for(int k=0;k<9;k++){if(k)j<<",";j<<r.stock[k];}j<<"],\"yield\":[";for(int k=0;k<9;k++){if(k)j<<",";j<<r.yield[k];}j<<"]}";};
   int original=0;for(auto a:core.queue)if(a.op==Op::BUY_PRODUCT&&int(a.item)==F)original+=a.quantity;
   write(0,baseline);
   for(int q=1;q<original&&q<32;q++){auto reduced=*this;int left=q;for(auto&a:reduced.core.queue)if(a.op==Op::BUY_PRODUCT&&int(a.item)==F){int n=std::min(left,a.quantity);a.quantity=n;left-=n;}j<<",";write(q,profile(reduced));}
   j<<",";write(original,supplied);j<<"]}";supply_audit_last=j.str();
  }
  bool extra=false,preserves=true;''')
p.write_text(s)
p=w/'diagnostic_r3/policy/bridge.cpp';p.write_text(p.read_text()+'\nextern "C" const char*td_supply_audit(void*p){return static_cast<Handle*>(p)->policy.live.supply_audit_last.c_str();}\n')
