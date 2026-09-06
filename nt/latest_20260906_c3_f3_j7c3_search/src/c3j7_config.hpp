#pragma once
inline competitive::Config frozen_config(){competitive::Config c;
c.new_limit=1;
c.hold=0;
c.expansion=2;
c.reserve=150;
c.competition=1.5;
c.service_dp=1;
c.service_cost=0;
c.execution_variant=4;
c.harvest_threshold=1;
c.investment_blend=1;
c.investment_scope=2;
c.investment_timing=2;
c.crop_dp=2;
c.crop_start=0;
c.crop_gain=0;
c.crop_max_changes=100;
c.calendar_samples=3;
c.investment_crop_consistent=1;
return c;}
