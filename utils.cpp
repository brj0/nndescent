#include <chrono>
#include "utils.h"


// Random number generator for seeding, code from
// https://prng.di.unimi.it/splitmix64.c
uint64_t splitmix64(uint64_t &x)
{
    uint64_t z = (x += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

// Initialize the seed states of the random number generator.
void seed_state(RandomState &s, uint64_t seed)
{
    if (seed == (uint64_t)NONE)
    {
        seed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
    }
    s[0] = splitmix64(seed);
    s[1] = splitmix64(seed);
    s[2] = splitmix64(seed);
    s[3] = splitmix64(seed);
}

// xoshiro256+ 1.0 is a fast random number generator. Code modified from
// https://prng.di.unimi.it/xoshiro256plus.c
uint64_t rand_int(RandomState &s)
{
    const uint64_t result = s[0] + s[3];
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = ((s[3] << 45) | (s[3] >> 19));
    return result;
}

namespace RandNumGen
{
    RandomState rng_state;
}
