from pathlib import Path
b=Path('/mnt/data/r3_work/candidate')
p=b/'policy/executor/policy.hpp';s=p.read_text();s=s.replace('namespace exchange {void apply(Controller&,const View&);}','namespace exchange {void apply(Controller&,const View&);}\nnamespace deliveryhandoff {bool apply(Controller&,const View&);}')
s=s.replace(' bool resume_compiled_tick=false;', ''' // Only the latest real observation may authorize the new fleet delivery.
 // Later MPC states must not pretend the observable cargo is already cash.
 int delivery_observation_step=-1;
 int delivery_handoff_checks=0,delivery_handoff_insertions=0,delivery_handoff_quantity=0;
 int delivery_handoff_last_step=-1,delivery_handoff_donor=-1,delivery_handoff_receiver=-1;
 bool resume_compiled_tick=false;''')
s=s.replace('dispatch_idle(o);dispatch_midroute(o);','dispatch_idle(o);int old_deliveries=midroute_delivery_insertions;dispatch_midroute(o);if(midroute_delivery_insertions==old_deliveries)deliveryhandoff::apply(*this,o);')
s=s.replace('#include "resource_exchange.hpp"','#include "resource_exchange.hpp"\n#include "delivery_handoff.hpp"')
p.write_text(s)
p=b/'policy/search.hpp';s=p.read_text().replace('  auto out=live.act(o);','  live.core.delivery_observation_step=o.step;\n  auto out=live.act(o);');p.write_text(s)
p=b/'main.py';s=p.read_text().replace('TRI_A08_r11_r2_fix1','TRI_A08_r11_r3').replace('tri_a08_r11_r2_fix1','tri_a08_r11_r3').replace('collection-work-consistent animal service valuation','resource-preserving fleet delivery handoff');p.write_text(s)
p=b/'build.py';s=p.read_text().replace('TRI_A08_r11_r2_fix1 builder (parent: exact TRI_A08_r11_r2)','TRI_A08_r11_r3 builder (parent: exact TRI_A08_r11_r2_fix1)')
s=s.replace("p=argparse.ArgumentParser();", "p=argparse.ArgumentParser();p.add_argument('--delivery-handoff',type=int,choices=[0,1],default=1);")
s=s.replace("ROOT/'policy/tri_a08_r11_r2_fix1.so'", "ROOT/'policy/tri_a08_r11_r3.so'")
s=s.replace("if out.is_relative_to(ROOT/'reference'):", "if out.is_relative_to(ROOT/'reference') or out.is_relative_to(ROOT/'parent'):")
s=s.replace("flags=[f'-DA08_ANIMAL_COLLECTION_LABOR", "flags=[f'-DA08_DELIVERY_HANDOFF={a.delivery_handoff}',f'-DA08_ANIMAL_COLLECTION_LABOR")
s=s.replace("'revision':'TRI_A08_r11_r2_fix1'", "'revision':'TRI_A08_r11_r3','delivery_handoff':a.delivery_handoff")
s=s.replace('afca1d69fde38a0474546b70ecee38911477548dfde9aadc685c58a2a0b88a2f','91828b8a9f6882030ba5247a8e40f15d788a9219fe64667502ebd6a5d211beed')
s=s.replace('TRI_A08_r11_r2_from_2f1557875d550000ae492c3814c84c78d5402e99b3a754308dc3907aa32b2178','TRI_A08_r11_r2_fix1_source_inputs_759b6a9992dfcb1d0dd9d7718561eb586fb04e57cafbc532e6ba7755584e58b2')
p.write_text(s)
