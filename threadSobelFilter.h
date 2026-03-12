#pragma once

#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <vector>
#include <utility>

/*-----------------------------------------------------------------------*
* Tunable constants
*------------------------------------------------------------------------*/

// Number of worker threads — one per physical core.
// Leave core 3 for OS interrupts if you want cleaner PAPI numbers;
// set to 3 in that case and re-run compute_thread_ranges.
#define NUM_THREADS     4

// X-tile width for cache-blocking the sobel inner loop.
// 128 fits comfortably in A76 L1 (64KB) with 3 row buffers in flight.
// Tune up/down in powers of two and watch PAPI L1 miss numbers.
#define SOBEL_TILE_W    128

/*-----------------------------------------------------------------------*
* threadArgs
*
* Per-thread descriptor passed to fnForThread1.
* Pointers to rgb/sobel are shared (read-only rgb, write-only sobel stripe).
* gray_row{0,1,2} are private per-thread ring buffers — 64-byte aligned,
* allocated in main before threads are spawned.
*------------------------------------------------------------------------*/

struct threadArgs {
    // shared frame pointers (main thread owns lifetime)
    cv::Mat* rgb;
    cv::Mat* sobel;

    // row stripe assigned to this thread [y0, y1)
    int y0;
    int y1;

    // thread index (0..NUM_THREADS-1), also used as core id for affinity
    int threadindex;

    // private 64-byte-aligned grayscale ring buffers (one row each)
    unsigned char* gray_row0;
    unsigned char* gray_row1;
    unsigned char* gray_row2;

    // pad to 64 bytes to prevent false sharing between thread descriptors
    char _pad[64 - (
        sizeof(cv::Mat*)       // rgb
      + sizeof(cv::Mat*)       // sobel
      + sizeof(int)            // y0
      + sizeof(int)            // y1
      + sizeof(int)            // threadindex
      + sizeof(int)            // implicit struct padding
      + sizeof(unsigned char*) // gray_row0
      + sizeof(unsigned char*) // gray_row1
      + sizeof(unsigned char*) // gray_row2
    ) % 64];
};

/*-----------------------------------------------------------------------*
* Globals shared between main and worker threads
*------------------------------------------------------------------------*/

// One pthread handle per worker
extern pthread_t thread[NUM_THREADS];

// Per-thread argument structs — indexed same as thread[]
extern threadArgs targetthread[NUM_THREADS];

// barrier_start: main signals workers a frame is ready
// barrier_done:  workers signal main that sobel is complete
extern pthread_barrier_t barrier_start;
extern pthread_barrier_t barrier_done;

/*-----------------------------------------------------------------------*
* Function declarations
*------------------------------------------------------------------------*/

// Worker thread entry point (all threads use the same function)
void* fnForThread1(void* arg);

// Spawn all NUM_THREADS workers
void startThreads();

// Divide [1, rows-1) into num_threads contiguous stripes
void compute_thread_ranges(int rows, int num_threads,
                           std::vector<std::pair<int,int>>& ranges);
