#pragma once

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <iomanip>

const int NONE = -1;
const int STATE_SIZE = 4;

typedef uint64_t RandomState[STATE_SIZE];

void seed_state(RandomState &s, uint64_t seed=NONE);
uint64_t rand_int(RandomState &s);
float rand_float(RandomState &s);

// Timer class for debugging and benchmarking.
class Timer
{
    private:
        std::chrono::time_point<std::chrono::system_clock> time;
    public:
        Timer() { start(); }
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

// Global timer for debugging
// TODO del
extern Timer global_timer;

class ProgressBar
{
private:
    int end;
    int cur;
    int interval;
    bool verbose;
    std::string prefix = "";
public:
    ProgressBar(int limit, int interval, bool verbose, std::string prefix="")
        : end(limit)
        , cur(0)
        , interval(interval)
        , verbose(verbose)
        , prefix(prefix)
        {}

    void show()
    {
        #pragma omp critical
        {
            ++cur;
        }
        if (!verbose || ((cur != end) && (cur % interval != 0)))
        {
            return;
        }
        const int width = 50;
        std::cout.flush();

        std::cout << prefix << "[";

        int pos = width * cur / end;
        for (int i = 0; i < width; ++i)
        {
            if (i < pos)
            {
                std::cout << "=";
            }
            else if (i == pos)
            {
                std::cout << ">";
            }
            else
            {
                std::cout << " ";
            }
        }
        std::cout << "] " << int(cur * 100 / end) << " %\r";
        #pragma omp critical
        {
            std::cout.flush();
        }
        if (cur == end)
        {
            std::cout << "\n";
        }
    }
};

void log(std::string text, bool verbose=true);
