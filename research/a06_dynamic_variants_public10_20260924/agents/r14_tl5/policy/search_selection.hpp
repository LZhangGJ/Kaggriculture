#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>
namespace triad {
// Ties keep the first candidate. Index zero is the current-state base plan,
// not yesterday's action tape. No fitted coefficients or opponent IDs.
struct MarginSelection {
 double minimum_gain=0,best=-std::numeric_limits<double>::infinity();
 double baseline=0,selected_raw=0;int winner=-1;
 explicit MarginSelection(double gain):minimum_gain(gain){
  if(!std::isfinite(gain)||gain<0)throw std::invalid_argument("invalid switch margin");
 }
 void observe(int index,double raw){
  if(!std::isfinite(raw))throw std::runtime_error("non-finite candidate value");
  if(index==0)baseline=raw;
  double decision=raw-(index==0?0:minimum_gain);
  if(decision>best+1e-6){best=decision;selected_raw=raw;winner=index;}
 }
 double gain()const{return selected_raw-baseline;}
};
}
