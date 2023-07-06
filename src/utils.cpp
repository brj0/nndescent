/**
 * @file utils.cpp
 *
 * @brief Contains utility functions.
 */


#include "utils.h"


namespace nndescent
{


/*
 * @brief Random number generator used for seeding.
 *
 * @see https://prng.di.unimi.it/splitmix64.c
 */
uint64_t splitmix64(uint64_t &x)
{
    uint64_t z = (x += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}


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


float rand_float(RandomState &s)
{
    const uint64_t result = s[0] + s[3];
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = ((s[3] << 45) | (s[3] >> 19));
    return (float)result / ((float)UINT64_MAX);
}


void log(const std::string &text, bool verbose)
{
    if (!verbose)
    {
        return;
    }
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::cout << std::put_time(localtime(&time), "%F %T ") << text << "\n";
}


std::ostream& operator<<(std::ostream& out, const RandomState& state)
{
    out << "(";
    for (size_t i = 0; i < STATE_SIZE; ++i) {
        out << state[i] << ", ";
    }
    out << ")";
    return out;
}


} // namespace nndescent
