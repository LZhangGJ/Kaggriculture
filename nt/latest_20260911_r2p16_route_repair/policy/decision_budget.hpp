#pragma once
#include <ctime>
namespace triad::budget {
struct Exhausted {};
inline thread_local std::clock_t deadline=0;
inline thread_local bool interruptible=false;
struct Decision {
 std::clock_t saved=deadline;
 Decision(){deadline=std::clock()+std::clock_t(.65*CLOCKS_PER_SEC);}
 ~Decision(){deadline=saved;}
};
struct Optional {
 bool saved=interruptible;
 Optional(){interruptible=true;}
 ~Optional(){interruptible=saved;}
};
// Only speculative copies may throw. The live incumbent always remains usable.
inline void check(){if(interruptible&&deadline&&std::clock()>=deadline)throw Exhausted{};}
}
