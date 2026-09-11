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
