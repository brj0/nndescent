/**
 * @file utils.h
 *
 * @brief Contains utility functions.
 */


#pragma once

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>


namespace nndescent
{


// Constants
const int NONE = -1;
const int STATE_SIZE = 4;

// Types
typedef uint64_t RandomState[STATE_SIZE];


/*
 * @brief Prints random state to output stream.
 */
std::ostream& operator<<(std::ostream& out, const RandomState& state);


/*
 * @brief Initialize the state of the random number generator.
 *
 * This function initializes the state of the random number generator
 * represented by 's' from the specified seed value. If the seed value is set
 * to NONE, a random value will be used.
 *
 * @param s The random number generator seed state.
 * @param seed The seed value to initialize the random number generator. Use
 * 'NONE' for a random value.
 */
void seed_state(RandomState &s, uint64_t seed=NONE);


/*
 * @brief Generate a random 64-bit integer using the xoshiro256+ random number
 * generator.
 *
 * This function generates a random 64-bit integer using the highly efficient
 * xoshiro256+ random number generator.
 *
 * @param s The random number generator state.
 *
 * @return A random 64-bit integer.
 *
 * @see https://prng.di.unimi.it/xoshiro256plus.c
 */
uint64_t rand_int(RandomState &s);


/*
 * @brief Generate a random float in the range [0, 1) using the xoshiro256+
 * random number generator.
 *
 * This function generates a random float in the range [0, 1) using the highly
 * efficient xoshiro256+ random number generator based on the provided random
 * number generator state 's'.
 *
 * @param s The random number generator state.
 * @return A random float in the range [0, 1).
 */
float rand_float(RandomState &s);


/*
 * @brief Timer class for measuring elapsed time.
 *
 * The Timer class provides a simple interface for measuring elapsed time. It
 * can be used for debugging purposes or for benchmarking code performance.
 *
 * Usage:
 * Timer timer;
 * timer.start();                // Start the timer
 * // Code to be timed
 * timer.stop("Output text");    // Stop the timer and print to standart output
 */
class Timer
{
    private:

        std::chrono::time_point<std::chrono::system_clock> time;

    public:

        Timer() { start(); }

        /*
         * @brief Start the timer.
         *
         * This function starts the timer and sets the initial timestamp.
         */
        void start()
        {
            time = std::chrono::high_resolution_clock::now();
        }

        /*
         * @brief Stop the timer and log a message.
         *
         * This function stops the timer, updates the end timestamp, and
         * logs a message.
         *
         * @param text The message to be logged.
         */
        void stop(const std::string& text)
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


/*
 * @brief Class for displaying a terminal progress bar.
 */
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
    ProgressBar(int limit, bool verbose, const std::string &prefix="")
        : end(limit)
        , cur(0)
        , width(50)
        , interval(limit / width + 1)
        , verbose(verbose)
        , prefix(prefix)
    {
    }

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


/*
 * @brief Log the provided text if the verbose flag is set.
 *
 * This function logs the provided text if the verbose flag is set to true. It
 * can be used to output informational messages with time stamps during program
 * execution when verbosity is desired.
 *
 * @param text The text to be logged.
 * @param verbose A flag indicating whether the log message should be printed
 * (true) or not (false).
 */
void log(const std::string &text, bool verbose=true);


/*
 * @brief Counts the number of elements in a range that are not equal to a
 * given value.
 */
template<class Iter, class T>
size_t count_if_not_equal(Iter first, Iter last, T value)
{
    size_t cnt = 0;
    for (; first != last; ++first)
    {
        if (*first != value)
        {
            ++cnt;
        }
    }
    return cnt;
}


/*
 * @brief Counts the number of elements in a range that are not equal to a
 * given value (sparse version).
 */
template<class IterCol, class IterData, class T>
size_t sparse_count_if_not_equal(
    IterCol first,
    IterCol last,
    IterData data,
    T value
)
{
    size_t cnt = 0;

    // Pass through the index list
    while (first != last)
    {
        if (*data != value)
        {
            ++cnt;
        }

        ++first;
        ++data;
    }

    return cnt;
}


} // namespace nndescent
