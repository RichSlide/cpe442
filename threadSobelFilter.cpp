/***************************************************************************
* File: threadSobelFilter.cpp
*
* Description: Takes a video as input, performs grayscale + Sobel filter
* per frame using 4 pthreads pinned to Cortex-A76 cores on Raspberry Pi 5.
*
* Optimizations applied:
*   - NEON SIMD for grayscale and Sobel inner loops
*   - Fused grayscale+Sobel with 3-row ring buffers per thread
*   - X-tiling (cache blocking) in the Sobel pass
*   - 64-byte aligned row buffers (matches A76 cache line)
*   - Double-buffered decode: cap.read overlaps with worker compute
*   - Hardware-accelerated H.264/H.265 decode via CAP_PROP_HW_ACCELERATION
*   - Thread affinity: each worker pinned to its own core
*   - PAPI L1-miss and cycle counters scoped to compute only
*   - --no-display flag to eliminate imshow overhead during benchmarking
*
**************************************************************************/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <vector>
#include <utility>

#include <pthread.h>
#include <sched.h>
#include <arm_neon.h>
#include <papi.h>

#include "threadSobelFilter.h"

/*-----------------------------------------------------------------------*
* Globals (declared extern in header)
*------------------------------------------------------------------------*/

pthread_t         thread[NUM_THREADS];
threadArgs        targetthread[NUM_THREADS];
pthread_barrier_t barrier_start;
pthread_barrier_t barrier_done;

/*-----------------------------------------------------------------------*
* Internal atomics for performance counters
*------------------------------------------------------------------------*/

static std::atomic_bool       threads_should_stop(false);
static std::atomic<long long> core_misses[NUM_THREADS];
static std::atomic<long long> core_frames[NUM_THREADS];
static std::atomic<long long> core_cycles[NUM_THREADS];

/*-----------------------------------------------------------------------*
* Function: pin_thread_to_core
*------------------------------------------------------------------------*/

static inline void pin_thread_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (ret != 0)
        std::cerr << "pthread_setaffinity_np failed (core " << core_id << ")\n";
}

/*-----------------------------------------------------------------------*
* Function: gray_row_neon
*
* Converts one BGR row to grayscale using integer NEON arithmetic.
* Weights: R=54, G=183, B=19  (sum=256, equivalent to ITU-R BT.601)
*------------------------------------------------------------------------*/

static inline void gray_row_neon(const cv::Vec3b* srcRow, uchar* dstRow, int cols) {
    int c = 0;
    for (; c + 8 <= cols; c += 8) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&srcRow[c]);
        uint8x8x3_t bgr = vld3_u8(p);

        uint16x8_t b   = vmovl_u8(bgr.val[0]);
        uint16x8_t g   = vmovl_u8(bgr.val[1]);
        uint16x8_t red = vmovl_u8(bgr.val[2]);

        uint16x8_t sum = vmulq_n_u16(red, 54);
        sum = vmlaq_n_u16(sum, g,   183);
        sum = vmlaq_n_u16(sum, b,    19);
        sum = vaddq_u16(sum, vdupq_n_u16(128));
        sum = vshrq_n_u16(sum, 8);

        vst1_u8(&dstRow[c], vmovn_u16(sum));
    }
    for (; c < cols; ++c) {
        const cv::Vec3b& px = srcRow[c];
        dstRow[c] = (uchar)((54 * px[2] + 183 * px[1] + 19 * px[0] + 128) >> 8);
    }
}

/*-----------------------------------------------------------------------*
* Function: sobel_row_neon_tiled
*
* Computes Sobel magnitude for one tile of one row given three grayscale
* rows (top/mid/bot). Writes output to dstRow[x0..x1).
*------------------------------------------------------------------------*/

static inline void sobel_row_neon_tiled(
    const uchar* topRow,
    const uchar* midRow,
    const uchar* botRow,
    uchar*       dstRow,
    int cols, int x0, int x1)
{
    int start = (x0 < 1)         ? 1        : x0;
    int end   = (x1 > cols - 1)  ? cols - 1 : x1;
    int c = start;

    for (; c + 8 <= end; c += 8) {
        // pull next cache line ahead of time
        __builtin_prefetch(&topRow[c + 64], 0, 1);
        __builtin_prefetch(&midRow[c + 64], 0, 1);
        __builtin_prefetch(&botRow[c + 64], 0, 1);

        uint8x8_t tL8 = vld1_u8(&topRow[c-1]);
        uint8x8_t tC8 = vld1_u8(&topRow[c  ]);
        uint8x8_t tR8 = vld1_u8(&topRow[c+1]);
        uint8x8_t mL8 = vld1_u8(&midRow[c-1]);
        uint8x8_t mR8 = vld1_u8(&midRow[c+1]);
        uint8x8_t bL8 = vld1_u8(&botRow[c-1]);
        uint8x8_t bC8 = vld1_u8(&botRow[c  ]);
        uint8x8_t bR8 = vld1_u8(&botRow[c+1]);

        int16x8_t tL = vreinterpretq_s16_u16(vmovl_u8(tL8));
        int16x8_t tC = vreinterpretq_s16_u16(vmovl_u8(tC8));
        int16x8_t tR = vreinterpretq_s16_u16(vmovl_u8(tR8));
        int16x8_t mL = vreinterpretq_s16_u16(vmovl_u8(mL8));
        int16x8_t mR = vreinterpretq_s16_u16(vmovl_u8(mR8));
        int16x8_t bL = vreinterpretq_s16_u16(vmovl_u8(bL8));
        int16x8_t bC = vreinterpretq_s16_u16(vmovl_u8(bC8));
        int16x8_t bR = vreinterpretq_s16_u16(vmovl_u8(bR8));

        int16x8_t gx = vsubq_s16(
            vaddq_s16(vaddq_s16(tR, bR), vshlq_n_s16(mR, 1)),
            vaddq_s16(vaddq_s16(tL, bL), vshlq_n_s16(mL, 1)));

        int16x8_t gy = vsubq_s16(
            vaddq_s16(vaddq_s16(tL, tR), vshlq_n_s16(tC, 1)),
            vaddq_s16(vaddq_s16(bL, bR), vshlq_n_s16(bC, 1)));

        uint8x8_t out = vqmovn_u16(vaddq_u16(
            vreinterpretq_u16_s16(vabsq_s16(gx)),
            vreinterpretq_u16_s16(vabsq_s16(gy))));

        vst1_u8(&dstRow[c], out);
    }

    for (; c < end; ++c) {
        int tL = topRow[c-1], tC = topRow[c], tR = topRow[c+1];
        int mL = midRow[c-1],                 mR = midRow[c+1];
        int bL = botRow[c-1], bC = botRow[c], bR = botRow[c+1];
        int gx = (tR + 2*mR + bR) - (tL + 2*mL + bL);
        int gy = (tL + 2*tC + tR) - (bL + 2*bC + bR);
        int G  = abs(gx) + abs(gy);
        dstRow[c] = (uchar)(G > 255 ? 255 : G);
    }
}

/*-----------------------------------------------------------------------*
* Function: fused_sobel
*
* Fuses grayscale conversion and Sobel filter in a single row-by-row pass
* using a 3-row ring buffer per thread to minimise memory traffic.
*------------------------------------------------------------------------*/

static inline void fused_sobel(threadArgs* threadinfo) {
    cv::Mat* rgbMat   = threadinfo->rgb;
    cv::Mat* sobelMat = threadinfo->sobel;
    const int rows = rgbMat->rows;
    const int cols = rgbMat->cols;
    const int y0   = threadinfo->y0;
    const int y1   = threadinfo->y1;

    uchar* row0 = threadinfo->gray_row0;
    uchar* row1 = threadinfo->gray_row1;
    uchar* row2 = threadinfo->gray_row2;

    auto load_gray_row = [&](int r, uchar* out) {
        if (r < 0)     r = 0;
        if (r >= rows) r = rows - 1;
        gray_row_neon(rgbMat->ptr<cv::Vec3b>(r), out, cols);
    };

    load_gray_row(y0 - 1, row0);
    load_gray_row(y0,     row1);
    load_gray_row(y0 + 1, row2);

    for (int r = y0; r < y1; r++) {
        uchar* dstRow    = sobelMat->ptr<uchar>(r);
        dstRow[0]        = 0;
        dstRow[cols - 1] = 0;

        for (int x = 1; x < cols - 1; x += SOBEL_TILE_W) {
            int xEnd = x + SOBEL_TILE_W;
            if (xEnd > cols - 1) xEnd = cols - 1;
            sobel_row_neon_tiled(row0, row1, row2, dstRow, cols, x, xEnd);
        }

        uchar* tmp = row0; row0 = row1; row1 = row2; row2 = tmp;
        load_gray_row(r + 2, row2);
    }
}

/*-----------------------------------------------------------------------*
* Function: fnForThread1
*
* Worker thread entry point. Loops over frames gated by two barriers.
* PAPI counters are scoped tightly around fused_sobel only.
*------------------------------------------------------------------------*/

void* fnForThread1(void* arg) {
    threadArgs* threadinfo = (threadArgs*)arg;
    int tid = threadinfo->threadindex;

    pin_thread_to_core(tid);

    int ret = PAPI_register_thread();
    if (ret != PAPI_OK) {
        std::cerr << "PAPI_register_thread: " << PAPI_strerror(ret) << "\n"; std::exit(1);
    }

    int EventSet = PAPI_NULL;
    ret = PAPI_create_eventset(&EventSet);
    if (ret != PAPI_OK) { std::cerr << "PAPI_create_eventset: " << PAPI_strerror(ret) << "\n"; exit(1); }

    ret = PAPI_add_event(EventSet, PAPI_L1_DCM);
    if (ret != PAPI_OK) { std::cerr << "PAPI add L1_DCM: " << PAPI_strerror(ret) << "\n"; exit(1); }

    int cyc_code = PAPI_NULL;
    ret = PAPI_event_name_to_code((char*)"perf::PERF_COUNT_HW_CPU_CYCLES", &cyc_code);
    if (ret != PAPI_OK) { std::cerr << "PAPI cyc code: " << PAPI_strerror(ret) << "\n"; exit(1); }

    ret = PAPI_add_event(EventSet, cyc_code);
    if (ret != PAPI_OK) { std::cerr << "PAPI add cycles: " << PAPI_strerror(ret) << "\n"; exit(1); }

    while (!threads_should_stop.load(std::memory_order_acquire)) {
        pthread_barrier_wait(&barrier_start);
        if (threads_should_stop.load(std::memory_order_acquire)) break;

        ret = PAPI_start(EventSet);
        if (ret != PAPI_OK) { std::cerr << "PAPI_start: " << PAPI_strerror(ret) << "\n"; exit(1); }

        fused_sobel(threadinfo);

        long long values[2] = {0, 0};
        ret = PAPI_stop(EventSet, values);
        if (ret != PAPI_OK) { std::cerr << "PAPI_stop: " << PAPI_strerror(ret) << "\n"; exit(1); }

        core_misses[tid].fetch_add(values[0], std::memory_order_relaxed);
        core_cycles[tid].fetch_add(values[1], std::memory_order_relaxed);
        core_frames[tid].fetch_add(1,         std::memory_order_relaxed);

        pthread_barrier_wait(&barrier_done);
    }

    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    PAPI_unregister_thread();
    return nullptr;
}

/*-----------------------------------------------------------------------*
* Function: startThreads
*------------------------------------------------------------------------*/

void startThreads() {
    for (int i = 0; i < NUM_THREADS; i++) {
        int r = pthread_create(&thread[i], nullptr, fnForThread1, &targetthread[i]);
        if (r != 0) { std::cerr << "pthread_create failed for thread " << i << "\n"; exit(1); }
    }
}

/*-----------------------------------------------------------------------*
* Function: compute_thread_ranges
*------------------------------------------------------------------------*/

void compute_thread_ranges(int rows, int num_threads,
                           std::vector<std::pair<int,int>>& ranges) {
    int interior = std::max(0, rows - 2);
    int base = interior / num_threads;
    int rem  = interior % num_threads;
    ranges.resize(num_threads);
    int cur = 1;
    for (int i = 0; i < num_threads; ++i) {
        int add = base + (i < rem ? 1 : 0);
        int y0 = cur, y1 = cur + add;
        if (add == 0) { y0 = y1 = 1; }
        ranges[i] = {y0, y1};
        cur = y1;
    }
}

/*-----------------------------------------------------------------------*
* Function: main
*
* Usage:  ./sobel <video_file> [--no-display]
*
*   --no-display   skips imshow/waitKey for clean benchmarking
*------------------------------------------------------------------------*/

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video> [--no-display]\n";
        return 1;
    }

    bool display = true;
    for (int i = 2; i < argc; i++)
        if (std::string(argv[i]) == "--no-display") display = false;

    // request hardware-accelerated decode; falls back to SW automatically
    cv::VideoCapture cap(argv[1], cv::CAP_FFMPEG);
    cap.set(cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY);
    if (!cap.isOpened()) { std::cerr << "Cannot open: " << argv[1] << "\n"; return 1; }

    // double-buffer: frames[0] and frames[1] alternate as current / next
    cv::Mat frames[2];
    if (!cap.read(frames[0])) { std::cerr << "Cannot read first frame\n"; return 1; }
    int cur = 0;

    const int rows = frames[0].rows;
    const int cols = frames[0].cols;

    cv::Mat sobelMat(rows, cols, CV_8UC1, cv::Scalar(0));

    std::vector<std::pair<int,int>> ranges;
    compute_thread_ranges(rows, NUM_THREADS, ranges);

    // PAPI initialisation
    int ret = PAPI_library_init(PAPI_VER_CURRENT);
    if (ret != PAPI_VER_CURRENT) { std::cerr << "PAPI_library_init failed\n"; exit(1); }

    ret = PAPI_thread_init((unsigned long (*)(void))(pthread_self));
    if (ret != PAPI_OK) { std::cerr << "PAPI_thread_init: " << PAPI_strerror(ret) << "\n"; exit(1); }

    // 64-byte aligned, cache-line-padded row buffers per thread
    const size_t row_buf_size = ((size_t)cols + 63) & ~63ULL;
    for (int i = 0; i < NUM_THREADS; i++) {
        void *p0 = nullptr, *p1 = nullptr, *p2 = nullptr;
        if (posix_memalign(&p0, 64, row_buf_size) != 0 ||
            posix_memalign(&p1, 64, row_buf_size) != 0 ||
            posix_memalign(&p2, 64, row_buf_size) != 0) {
            std::cerr << "posix_memalign failed\n"; exit(1);
        }
        std::memset(p0, 0, row_buf_size);
        std::memset(p1, 0, row_buf_size);
        std::memset(p2, 0, row_buf_size);
        targetthread[i].gray_row0 = (uchar*)p0;
        targetthread[i].gray_row1 = (uchar*)p1;
        targetthread[i].gray_row2 = (uchar*)p2;
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        targetthread[i].rgb         = &frames[cur];
        targetthread[i].sobel       = &sobelMat;
        targetthread[i].y0          = ranges[i].first;
        targetthread[i].y1          = ranges[i].second;
        targetthread[i].threadindex = i;
    }

    // NUM_THREADS workers + 1 main thread at each barrier
    pthread_barrier_init(&barrier_start, nullptr, NUM_THREADS + 1);
    pthread_barrier_init(&barrier_done,  nullptr, NUM_THREADS + 1);

    for (int i = 0; i < NUM_THREADS; i++) {
        core_misses[i].store(0, std::memory_order_relaxed);
        core_frames[i].store(0, std::memory_order_relaxed);
        core_cycles[i].store(0, std::memory_order_relaxed);
    }

    startThreads();

    long long total_frames = 0;
    std::chrono::high_resolution_clock::duration compute_time{0};

    while (true) {
        std::memset(sobelMat.ptr<uchar>(0),        0, (size_t)cols);
        std::memset(sobelMat.ptr<uchar>(rows - 1), 0, (size_t)cols);

        for (int i = 0; i < NUM_THREADS; i++)
            targetthread[i].rgb = &frames[cur];

        auto t0 = std::chrono::high_resolution_clock::now();
        pthread_barrier_wait(&barrier_start);

        // decode next frame while workers process current — free overlap
        int next = 1 - cur;
        bool got_frame = cap.read(frames[next]);

        pthread_barrier_wait(&barrier_done);
        auto t1 = std::chrono::high_resolution_clock::now();
        compute_time += (t1 - t0);
        ++total_frames;

        if (display) {
            cv::imshow("sobel", sobelMat);
            if (cv::waitKey(1) == 27) break;
        }

        if (!got_frame) break;
        cur = next;
    }

    // shut down workers
    threads_should_stop.store(true, std::memory_order_release);
    pthread_barrier_wait(&barrier_start);
    for (int i = 0; i < NUM_THREADS; i++) pthread_join(thread[i], nullptr);

    pthread_barrier_destroy(&barrier_start);
    pthread_barrier_destroy(&barrier_done);

    // report
    double total_secs = std::chrono::duration<double>(compute_time).count();
    double fps = (total_secs > 0.0) ? (double)total_frames / total_secs : 0.0;

    std::cout << "\n=== Performance Results ===\n";
    std::cout << "Total frames:             " << total_frames << "\n";
    std::cout << "Compute time (s):         " << total_secs   << "\n";
    std::cout << "Compute FPS:              " << fps          << "\n\n";

    for (int i = 0; i < NUM_THREADS; i++) {
        long long f      = core_frames[i].load();
        long long misses = core_misses[i].load();
        long long cycles = core_cycles[i].load();
        std::cout << "Core " << i
                  << "  frames="           << f
                  << "  L1-misses/frame="  << (f ? (double)misses/f : 0.0)
                  << "  cycles/frame="     << (f ? (double)cycles/f : 0.0) << "\n";
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        free(targetthread[i].gray_row0);
        free(targetthread[i].gray_row1);
        free(targetthread[i].gray_row2);
    }

    cap.release();
    if (display) cv::destroyAllWindows();
    return 0;
}
