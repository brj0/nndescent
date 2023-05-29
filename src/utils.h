#pragma once

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <iomanip>

namespace nndescent {

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


class ProgressBar
{
private:
    int end;
    int cur;
    const int width;
    int interval;
    bool verbose;
    std::string prefix = "";
public:
    ProgressBar(int limit, bool verbose, std::string prefix="")
        : end(limit)
        , cur(0)
        , width(50)
        , interval(limit / width + 1)
        , verbose(verbose)
        , prefix(prefix)
        {}

    void show()
    {
        if (!verbose)
        {
            return;
        }
        #pragma omp critical
        {
            ++cur;
        }
        if ((cur != end) && (cur % interval != 0))
        {
            return;
        }
        std::stringstream prog_bar;
        prog_bar << prefix << "[";

        int pos = width * cur / end;
        for (int i = 0; i < width; ++i)
        {
            if (i < pos)
            {
                prog_bar << "=";
            }
            else if (i == pos)
            {
                prog_bar << ">";
            }
            else
            {
                prog_bar << " ";
            }
        }
        prog_bar << "] " << int(cur * 100 / end) << " %\r";
        #pragma omp critical
        {
            std::cout << prog_bar.str() << std::flush;
        }
        if (cur == end)
        {
            std::cout << "\n";
        }
    }
};

void log(std::string text, bool verbose=true);

} // namespace nndescent
