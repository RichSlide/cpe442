#pragma once

#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <vector>
#include <utility>

#define NUM_THREADS  4
#define SOBEL_TILE_W 128

// alignas(64) replaces the previous manual _pad calculation which could
// produce a zero-length array on some compilers, causing the segfault.
struct alignas(64) threadArgs {
    cv::Mat*       rgb;
    cv::Mat*       sobel;
    int            y0;
    int            y1;
    int            threadindex;
    unsigned char* gray_row0;
    unsigned char* gray_row1;
    unsigned char* gray_row2;
};

extern pthread_t          thread[NUM_THREADS];
extern threadArgs         targetthread[NUM_THREADS];
extern pthread_barrier_t  barrier_start;
extern pthread_barrier_t  barrier_done;

void* fnForThread1(void* arg);
void  startThreads();
void  compute_thread_ranges(int rows, int num_threads,
                             std::vector<std::pair<int,int>>& ranges);
