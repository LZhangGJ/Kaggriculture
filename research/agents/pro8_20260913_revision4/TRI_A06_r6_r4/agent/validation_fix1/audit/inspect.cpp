#include "../bundle/policy/bridge.cpp"
extern "C" const char* td_live_needs(void* ptr){
 auto&h=*static_cast<Handle*>(ptr);auto&c=h.policy.live.core;std::ostringstream j;
 j<<"{\"day\":"<<c.day<<",\"phase\":"<<c.phase<<",\"planned_land\":"<<c.planned_land<<",\"daily_need\":[";
 for(int i=0;i<12;i++){if(i)j<<",";j<<c.daily_need[i];}j<<"],\"feed\":[";
 for(int i=0;i<100;i++){if(i)j<<",";j<<int(c.service_feed[i]);}j<<"],\"fertilize\":[";
 for(int i=0;i<100;i++){if(i)j<<",";j<<int(c.crop_fertilize[i]);}j<<"],\"target\":[";
 for(size_t i=0;i<c.target.size();i++){if(i)j<<",";j<<"["<<c.target[i].first<<","<<c.target[i].second<<"]";}j<<"]}";h.text=j.str();return h.text.c_str();
}
