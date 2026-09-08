// Capability audit only: do NOT turn this constructed state into a policy rule.
#include "policy.hpp"
#include <iostream>
using namespace dp7;
View observation(const Simulator&e){return {e.step_count(),e.day(),e.hour(),e.farms()[0],e.farms()[1],e.privates()[0],e.market(),e.shops()};}
int main(){
 Simulator e(Config{},203);while(e.step_count()<4*24+23)e.step({});
 auto&f=const_cast<Farm&>(e.farms()[0]);f.farmer={4,4};f.hands={{4,4}};f.tiles[44]=Tile{};
 auto&p=const_cast<PrivateState&>(e.privates()[0]);p.seeds[W]=1;p.inventories.assign(2,{});p.inventory_order.assign(2,{});
 Controller c;c.day=4;c.phase=3;c.target={{44,W}};c.p.shared_task_atoms_v2=true;
 auto js=c.jobs(observation(e));auto allocation=c.pack(js,{44,44},1,false);
 Counts pickup{},seed_used{};bool original_water=bool(c.valid(observation(e),1,action(Op::WATER),44,pickup,seed_used));
 auto projected=e.project_unit_phase(0,{action(Op::PLANT,W),action(Op::PASS)});
 pickup={};seed_used={};bool after_plant=bool(c.valid(observation(projected),1,action(Op::WATER),44,pickup,seed_used));
 auto joint=e.project_unit_phase(0,{action(Op::PLANT,W),action(Op::WATER)});
 const auto&t=joint.farms()[0].tiles[44];
 if(js.size()!=1||js[0].actions.size()!=2||allocation.second!=1||original_water||!after_plant||!plant(t)||!t.watered_today||joint.privates()[0].seeds[W]!=0)return 2;
 e.step({PlayerAction{{action(Op::PLANT,W),action(Op::WATER)},{}},{}});
 if(e.day()!=5||!plant(e.farms()[0].tiles[44])||e.farms()[0].tiles[44].consecutive_unwatered!=0)return 3;
 std::cout<<"{\"status\":\"PASS_REPRODUCED_CAPABILITY_GAP\",\"jobs\":1,\"locked_chain_actions\":2,\"units\":2,\"remaining_ticks\":1,\"current_packer_unassigned\":1,\"initial_water_legal_mask\":false,\"prefix_projected_water_legal_mask\":true,\"official_kernel_joint_success\":true,\"next_day_alive\":true,\"not_a_profit_claim\":true}\n";
}
