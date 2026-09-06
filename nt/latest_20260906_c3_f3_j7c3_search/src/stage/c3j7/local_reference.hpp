// Unchanged TEAM-OWNED J7 configuration and macro candidate. Not a public opponent.
#pragma once
inline dp7::Params frozen_params(){dp7::Params p;
p.opponent_supply_weight=0.75;
p.competitive_sell_weight=0.5;
p.sell_horizon_days=1;
p.opening_animals={10,10,11,9,9};
p.opening_crops={4,0,0,6,0};
p.economic_land=true;
p.operating_reserve=100;
p.crop_harvest_age={4,3,8,10,10};
p.finite_fertilizer=false;
p.fix_finite_projection=true;
p.efficient_water=true;
p.efficient_care=true;
p.fix_logistics=true;
p.own_feed_demand_weight=0;
p.committed_feed_weight=0;
p.existing_replant_weight=0;
p.insertion_hire_estimate=false;
p.reconcile_seed_drift=false;
p.plan_zero_expiry=true;
p.cashflow_value_mode=0;
p.funded_bundle_mode=0;
p.preparation_work=false;
p.split_service_jobs=false;
p.regret_schedule=false;
p.regret_compile=false;
p.regret_hire_estimate=false;
p.terminal_deposit_schedule=true;
p.overflow_idle_dispatch=false;
p.capital_time_rank=false;
p.shared_task_atoms_v2=true;
p.stepwise_recoordination=true;
p.preparation_pipeline_v2=true;
p.schedule_value_compare=true;
p.net_feed_buffer=true;
p.preparation_spawn_guard=true;
p.midroute_delivery=true;
p.resource_aware_exchange=false;
p.intraday_admission=true;
p.intraday_procurement=true;
p.intraday_declared_value=false;
p.autonomous_start=false;
p.joint_investment_portfolio=false;
p.portfolio_crop_calendar=false;
p.recover_service_inputs=false;
p.procure_service_inputs=false;
p.finance_service_inputs=false;
p.incremental_pickup_repair=true;
p.shared_service_insertions=true;
p.intraday_future_workforce=false;
p.day_consequence_compare=false;
p.day_value_public_supply=false;
p.day_value_replan_next_day=false;
p.idle_task_handoff=false;
p.exact_schedule_cache=false;
p.incremental_regret_cost=false;
p.continuous_market_execution=false;
p.insert_missing_feed=false;
return p;}
struct FrozenRecipe{const char*family;int kind,amount;};

#pragma once
inline const int switch_opening=0;
inline const std::vector<std::array<FrozenRecipe,29>> switch_plans={
std::array<FrozenRecipe,29>{{
{"REPLACE_NEW",3,1},
{"REPLACE_NEW",0,4},
{"REBUILD_NEW",0,9},
{"REPLACE_NEW",3,4},
{"REBUILD_NEW",2,1},
{"REBUILD_NEW",3,4},
{"REBUILD_NEW",3,21},
{"REPLACE_NEW",10,1},
{"REBUILD_NEW",3,5},
{"REBUILD_NEW",1,7},
{"REBUILD_NEW",3,25},
{"REBUILD_NEW",3,24},
{"REBUILD_NEW",1,13},
{"REPLACE_NEW",0,1},
{"KEEP",-1,0},
{"KEEP",-1,0},
{"KEEP",-1,0},
{"KEEP",-1,0},
{"KEEP",-1,0},
{"REBUILD_NEW",2,2},
{"REBUILD_NEW",1,2},
{"REBUILD_NEW",1,1},
{"REBUILD_NEW",1,6},
{"REBUILD_NEW",0,4},
{"DEFER_NEW",-1,0},
{"REBUILD_NEW",0,1},
{"REBUILD_NEW",1,4},
{"KEEP",-1,0},
{"KEEP",-1,0},
}},
};
struct SwitchRule{int day;std::vector<int>left,right,feature,target;std::vector<double>threshold;};
inline const std::vector<SwitchRule>switch_rules={};
