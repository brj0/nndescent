#pragma once

#include <cstdint>

const int NONE = -1;
const int STATE_SIZE = 4;

typedef uint64_t RandomState[STATE_SIZE];

void seed_state(RandomState &s, uint64_t seed=NONE);
uint64_t rand_int(RandomState &s);

namespace RandNumGen
{
    extern RandomState rng_state;
}
