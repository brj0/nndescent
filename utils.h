#pragma once

#include <cstdint>
#include <iostream>

const int NONE = -1;
const int STATE_SIZE = 4;

typedef uint64_t RandomState[STATE_SIZE];

void seed_state(RandomState &s, uint64_t seed=NONE);
uint64_t rand_int(RandomState &s);

// Timer class for debugging and benchmarking.
class Timer
{
    private:
        std::chrono::time_point<std::chrono::system_clock> time;
    public:
        void start()
        {
            time = std::chrono::high_resolution_clock::now();
        }
        void stop(std::string text)
        {
            auto end = std::chrono::high_resolution_clock::now();
            std::cout << "Time passed: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       end - time
                   ).count()
                << " ms ("
                << text
                << ")\n";
            this->start();
        }
};

