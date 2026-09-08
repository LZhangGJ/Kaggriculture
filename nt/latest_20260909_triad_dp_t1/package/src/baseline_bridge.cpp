#include "../agent/hybrid.hpp"
#include <cstring>
#include <sstream>
struct Baseline{
 competitive::Config cfg;std::unique_ptr<competitive::Hybrid>j7;std::unique_ptr<competitive::Planner>automatic;
 std::string text;Baseline(const double*x,size_t n){if(n!=49)throw std::runtime_error("baseline config");std::memcpy(&cfg,x,sizeof(cfg));
 #ifdef BASE_AUTO
 automatic=std::make_unique<competitive::Planner>(cfg);
 #else
 j7=std::make_unique<competitive::Hybrid>(cfg);
 #endif
 }
};
extern "C" void*td_new(const double*x,size_t n){try{return new Baseline(x,n);}catch(...){return nullptr;}}
extern "C" void td_delete(void*p){delete static_cast<Baseline*>(p);}
extern "C" int td_act(void*p,const dp7::View*v,fastkag::PlayerAction*a){auto&b=*static_cast<Baseline*>(p);try{*a=b.automatic?b.automatic->act(*v):b.j7->act(*v);return 0;}catch(const std::exception&e){b.text=e.what();return -1;}}
extern "C" const char*td_debug(void*p){auto&b=*static_cast<Baseline*>(p);if(b.text.empty())b.text=b.automatic?"{\"reference_calls\":0,\"baseline\":\"C3_SEP05_AUTONOMOUS\"}":"{\"baseline\":\"C3_SEP05_J7\"}";return b.text.c_str();}
