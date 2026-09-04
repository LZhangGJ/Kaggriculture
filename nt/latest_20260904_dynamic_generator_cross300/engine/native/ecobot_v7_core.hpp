// Frozen public EcoBot v7 port, NOT the autonomous v7 candidate.
// Source SHA256 0dc02e03c94ef60c06b5093efc2e2fd0530aa6eea20df507a90b90d6651bd067.
#pragma once
#include "vendor/simulator.hpp"
#include <array>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace eco7 {
using fastkag::Action;using fastkag::Op;using fastkag::Item;
using Pos=std::pair<int,int>;using Stock=std::array<int,12>;using Seeds=std::array<int,5>;
constexpr int W=0,C=1,T=2,S=3,M=4,F=8,G=9,COW=10,SHEEP=11;
inline constexpr std::array<Pos,4> SHEDS{{{4,4},{5,4},{4,5},{5,5}}};
inline constexpr std::array<Pos,18> CLUSTER{{{4,4},{5,4},{4,5},{4,3},{3,4},{5,3},{6,4},{3,5},{4,6},{3,3},{4,2},{2,4},{6,3},{5,2},{7,4},{2,5},{3,6},{4,7}}};
inline constexpr std::array<Pos,12> BLOCKED{{{0,0},{1,0},{0,1},{9,0},{8,0},{9,1},{0,9},{1,9},{0,8},{9,9},{8,9},{9,8}}};
inline constexpr std::array<int,5> FIRST{{2,2,8,10,10}},MAXAGE{{4,3,11,16,10}};
inline constexpr std::array<int,5> BONUS{{2,2,8,10,6}};
inline constexpr std::array<int,4> TOM_DAYS{{8,9,10,11}},STR_DAYS{{10,12,14,16}};
inline bool ongoing(int crop){return crop==T||crop==S;}
inline bool premium(int crop){return crop==M||crop==S;}
inline int product(int animal){return animal==G?5:animal==COW?6:7;}
inline Action act(Op op,int item=-1,int q=1){return {op,Item(item),q};}
int dist(Pos a,Pos b);
int quad(Pos p);
struct Animal{Pos pos;int species;bool fed=false;int unfed=0;bool cared=false,fertilizer=false;int yield=0;};
struct Plant{Pos pos;int crop=0,age=0;bool watered=false,water_needed=false;int yield=0;bool fert_due=false;int fert_until=-1;bool expired=false;};
struct FarmState{std::vector<Animal>animals;std::vector<Plant>plants;std::vector<Pos>pastures,coops,weeds,empty;int unlocked_count=0;};
struct Census{Stock field{},shed{},carried{};int total(int sp)const{return field[sp]+shed[sp]+carried[sp];}int grazers()const{return total(COW)+total(SHEEP);}int all()const{return grazers()+total(G);}};
struct Hints{int grazers=0,geese=0;Seeds crops{};};
using Caps=std::array<int,12>;
inline Caps all_caps(){Caps c;c.fill(999);return c;}
struct Unit{Pos pos;Stock inv{};};
using Produce=std::optional<std::pair<int,int>>;
struct Task{int urgency;Pos pos;std::vector<Action>actions;std::vector<int>need;Produce produces;};
struct Stop{Pos pos;std::vector<Action>actions;Produce produces;};
struct Route{Pos start;std::vector<Stop>stops;int cost=0;Stock carried{};std::array<int,12>pickup;int upto=-1,from=0;Route(){pickup.fill(-1);}};
enum class Mode{PLAIN,CARRIED,BUMP,EXTEND,NEW};
struct Candidate{int cost,at,item;Mode mode;int pickup_idx=-1;std::optional<Pos>pickup_pos;};

FarmState parse(const fastkag::Farm&f,int day);
Census census(const FarmState&f,const Stock&shed,const std::vector<Unit>&units);
int kept_feedable(const Census&c,const Caps&caps);
std::pair<int,int>feed_thresholds(const Census&c,const Caps&caps,int day);
int fert_due_tomorrow(const FarmState&f,int day);
int maturing_tomorrow(const FarmState&f,int crop);
int coop_capacity(const FarmState&f);
std::set<Pos>reserved(const FarmState&f,int quads,const Hints&h,int day);
std::vector<Pos>needed_pastures(const FarmState&f,const Census&c);
std::vector<Pos>needed_coops(const FarmState&f,const Census&c,const std::vector<Pos>&exclude);
bool ready(const Plant&p,int day);
std::vector<Plant>ready_premium(const FarmState&f,int day);
std::vector<Task>catalog(const FarmState&f,const Stock&shed,const Seeds&seeds,const std::vector<Pos>&pastures,const std::vector<Pos>&coops,const std::vector<Unit>&units,int day,int hour,const Hints&h,int quads,const Caps&caps);
std::optional<Candidate>eval_route(const Route&r,const Task&t,const Stock&shed);
void commit(Route&r,const Task&t,const Candidate&c,Stock&shed);
std::vector<Route>solve(const std::vector<Task>&tasks,const std::vector<Plant>&premium,const std::vector<Unit>&units,const Stock&shed,const std::vector<int>&budgets,int day);
std::vector<Action>materialize(const Route&r);
int required_hands(const std::vector<Task>&tasks,const std::vector<Plant>&premium,const std::vector<Unit>&units,const Stock&shed,int max_hands,int day);
std::vector<std::vector<Action>>day_plan(const std::vector<Unit>&units,const FarmState&f,const Stock&shed,const Seeds&seeds,int day,int hour,const std::vector<Pos>&pastures,const std::vector<Pos>&coops,const Hints&h,int quads,const Caps&caps,int pending_hires);
}
