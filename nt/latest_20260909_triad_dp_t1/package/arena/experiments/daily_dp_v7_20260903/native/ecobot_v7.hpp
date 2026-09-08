#pragma once
#include "ecobot_v7_core.hpp"
namespace eco7 {
using Values=std::array<double,9>;
// Match fastkag::shop_name, NOT insertion order of Python SHOP_DEMANDS.
enum Shop:int{BAKERY=0,BRUNCH=1,FARMERS=2,ICE_CREAM=3,PET_CAFE=4,PIZZA=5,SMOOTHIE=6,YARN=7};
struct Input{int step,day,hour;const fastkag::Farm&farm;const fastkag::PrivateState&priv;const fastkag::Market&market;const std::vector<int8_t>&shops;};
struct EvalState{int prev_day=-1,cull_day=-1;std::array<int32_t,9>prev_inv{};Values observed{};Stock negative{};std::array<bool,12>downsized{};Caps retained=all_caps();};
struct Decision{std::vector<Action>orders;Hints hints;Caps caps=all_caps();std::vector<Pos>pastures,coops;};
struct Score{int kind,item;double cost,daily,npv;};
int market_price(int item,int inventory);
Decision evaluate(const Input&o,EvalState&state);
struct Controller{
  EvalState state;int plan_day=-1;std::vector<std::vector<Action>>queues;Decision decision;
  fastkag::PlayerAction act(const Input&o);
};
}
