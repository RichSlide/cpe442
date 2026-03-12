/***************************************************************************
* File: threadSobelfilter.cpp
*
* Descripion: This file takes a video as an input, then preforms a grayscale
* and sobel algorithm for each frame using threading
*
* Revisions: 
*
**************************************************************************/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include "threadSobelFilter.h"
#include <arm_neon.h>
#include <atomic>
#include <cstdlib> 
#include <papi.h>
#include <chrono>
#include <cstring>
#include <sched.h>

// globals declared extern in header
pthread_t         thread[NUM_THREADS];
threadArgs        targetthread[NUM_THREADS];
pthread_barrier_t barrier_start;
pthread_barrier_t barrier_done;

static std::atomic_bool threads_should_stop(false);
static std::atomic<long long> core_misses[4];
static std::atomic<long long> core_frames[4];
static std::atomic<long long> core_cycles[4];

/*-----------------------------------------------------------------------*
* Function: pin_thread_to_core
*
* Description: pins the calling thread to a specific CPU core
*
* param core_id: core number to pin to
*
* return: void
*------------------------------------------------------------------------*/

static inline void pin_thread_to_core(int core_id){
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(core_id, &cpuset);

	int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
	if (ret != 0){
		// non-fatal: keep running, but print once
		std::cerr << "pthread_setaffinity_np failed (core " << core_id << ")\n";
	}
}

/*-----------------------------------------------------------------------*
* Function: gray_row
*
* Description: takes an RGB row and converts it to a grayscale row buffer
*
* param srcRow: pointer to first pixel in RGB row
* param dstRow: pointer to first pixel in grayscale row buffer
* param cols: number of columns in frame
*
* return: void
*------------------------------------------------------------------------*/

static inline void gray_row_neon(const cv::Vec3b* srcRow, uchar* dstRow, int cols){
	int c = 0;

	for (; c + 8 <= cols; c += 8){
		const uint8_t* p = reinterpret_cast<const uint8_t*>(&srcRow[c]);
		uint8x8x3_t bgr = vld3_u8(p);

		uint16x8_t b = vmovl_u8(bgr.val[0]);
		uint16x8_t g = vmovl_u8(bgr.val[1]);
		uint16x8_t red = vmovl_u8(bgr.val[2]);

		//int gray = (54 * red + 183 * green + 19 * blue + 128) >> 8;
		uint16x8_t sum = vmulq_n_u16(red, 54);
		sum = vmlaq_n_u16(sum, g, 183);
		sum = vmlaq_n_u16(sum, b, 19);
		sum = vaddq_u16(sum, vdupq_n_u16(128));
		sum = vshrq_n_u16(sum, 8);

		uint8x8_t gray8 = vmovn_u16(sum);
		vst1_u8(&dstRow[c], gray8);
	}

	//any left over pixels do scalar :(
	for(; c < cols; ++c){
		const cv::Vec3b& px = srcRow[c];
		int gray = (54 * px[2] + 183 * px[1] + 19 * px[0] + 128) >> 8;
		dstRow[c] = (uchar)gray;
	}
}

/*-----------------------------------------------------------------------*
* Function: sobel_row_tiled
*
* Description: This function takes 3 grayscale row buffers and computes
* the Sobel output for the middle row and writes to dstRow, but only for
* a given x-range (tile) to improve cache locality
*
* param topRow: grayscale row r-1
* param midRow: grayscale row r
* param botRow: grayscale row r+1
* param dstRow: output row in sobel Mat
* param cols: number of columns in frame
* param x0: starting x (inclusive) for tile
* param x1: ending x (exclusive) for tile
*
* return: void
*------------------------------------------------------------------------*/

static inline void sobel_row_neon_tiled(
	const uchar* topRow,
	const uchar* midRow,
	const uchar* botRow,
	uchar* dstRow,
	int cols,
	int x0,
	int x1
){
	// clamp tile to interior range [1, cols-1)
	int start = (x0 < 1) ? 1 : x0;
	int end   = (x1 > (cols - 1)) ? (cols - 1) : x1; // exclusive

	int c = start;

	// 16-wide NEON: process 16 pixels per iteration using 128-bit q-registers
	// This halves the loop overhead vs the previous 8-wide version.
	for (; c + 16 <= end; c += 16){
		// prefetch the next 64-byte cache line for all three rows
		__builtin_prefetch(&topRow[c + 64], 0, 1);
		__builtin_prefetch(&midRow[c + 64], 0, 1);
		__builtin_prefetch(&botRow[c + 64], 0, 1);

		uint8x16_t tL8 = vld1q_u8(&topRow[c-1]);
		uint8x16_t tC8 = vld1q_u8(&topRow[c]);
		uint8x16_t tR8 = vld1q_u8(&topRow[c+1]);

		uint8x16_t mL8 = vld1q_u8(&midRow[c-1]);
		uint8x16_t mR8 = vld1q_u8(&midRow[c+1]);

		uint8x16_t bL8 = vld1q_u8(&botRow[c-1]);
		uint8x16_t bC8 = vld1q_u8(&botRow[c]);
		uint8x16_t bR8 = vld1q_u8(&botRow[c+1]);

		// split each 16-wide vector into low/high 8-wide halves, widen to s16
		// low halves
		int16x8_t tL_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(tL8)));
		int16x8_t tC_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(tC8)));
		int16x8_t tR_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(tR8)));
		int16x8_t mL_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mL8)));
		int16x8_t mR_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mR8)));
		int16x8_t bL_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bL8)));
		int16x8_t bC_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bC8)));
		int16x8_t bR_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bR8)));

		// high halves
		int16x8_t tL_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(tL8)));
		int16x8_t tC_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(tC8)));
		int16x8_t tR_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(tR8)));
		int16x8_t mL_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mL8)));
		int16x8_t mR_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mR8)));
		int16x8_t bL_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bL8)));
		int16x8_t bC_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bC8)));
		int16x8_t bR_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bR8)));

		// gx low
		int16x8_t gx_lo = vsubq_s16(
			vaddq_s16(vaddq_s16(tR_lo, bR_lo), vshlq_n_s16(mR_lo, 1)),
			vaddq_s16(vaddq_s16(tL_lo, bL_lo), vshlq_n_s16(mL_lo, 1)));
		// gy low
		int16x8_t gy_lo = vsubq_s16(
			vaddq_s16(vaddq_s16(tL_lo, tR_lo), vshlq_n_s16(tC_lo, 1)),
			vaddq_s16(vaddq_s16(bL_lo, bR_lo), vshlq_n_s16(bC_lo, 1)));

		// gx high
		int16x8_t gx_hi = vsubq_s16(
			vaddq_s16(vaddq_s16(tR_hi, bR_hi), vshlq_n_s16(mR_hi, 1)),
			vaddq_s16(vaddq_s16(tL_hi, bL_hi), vshlq_n_s16(mL_hi, 1)));
		// gy high
		int16x8_t gy_hi = vsubq_s16(
			vaddq_s16(vaddq_s16(tL_hi, tR_hi), vshlq_n_s16(tC_hi, 1)),
			vaddq_s16(vaddq_s16(bL_hi, bR_hi), vshlq_n_s16(bC_hi, 1)));

		// |gx|+|gy| with saturation, recombine low+high into 16 bytes
		uint8x8_t out_lo = vqmovn_u16(vaddq_u16(
			vreinterpretq_u16_s16(vabsq_s16(gx_lo)),
			vreinterpretq_u16_s16(vabsq_s16(gy_lo))));
		uint8x8_t out_hi = vqmovn_u16(vaddq_u16(
			vreinterpretq_u16_s16(vabsq_s16(gx_hi)),
			vreinterpretq_u16_s16(vabsq_s16(gy_hi))));

		vst1q_u8(&dstRow[c], vcombine_u8(out_lo, out_hi));
	}

	// 8-wide cleanup for columns that don't fill a 16-wide group
	for (; c + 8 <= end; c += 8){
		uint8x8_t tL8 = vld1_u8(&topRow[c-1]);
		uint8x8_t tC8 = vld1_u8(&topRow[c]);
		uint8x8_t tR8 = vld1_u8(&topRow[c+1]);

		uint8x8_t mL8 = vld1_u8(&midRow[c-1]);
		uint8x8_t mR8 = vld1_u8(&midRow[c+1]);

		uint8x8_t bL8 = vld1_u8(&botRow[c-1]);
		uint8x8_t bC8 = vld1_u8(&botRow[c]);
		uint8x8_t bR8 = vld1_u8(&botRow[c+1]);

		// widen to signed 16-bit
		int16x8_t tL = vreinterpretq_s16_u16(vmovl_u8(tL8));
		int16x8_t tC = vreinterpretq_s16_u16(vmovl_u8(tC8));
		int16x8_t tR = vreinterpretq_s16_u16(vmovl_u8(tR8));

		int16x8_t mL = vreinterpretq_s16_u16(vmovl_u8(mL8));
		int16x8_t mR = vreinterpretq_s16_u16(vmovl_u8(mR8));

		int16x8_t bL = vreinterpretq_s16_u16(vmovl_u8(bL8));
		int16x8_t bC = vreinterpretq_s16_u16(vmovl_u8(bC8));
		int16x8_t bR = vreinterpretq_s16_u16(vmovl_u8(bR8));

		// gx = (tR + 2*mR + bR) - (tL + 2*mL + bL)
		int16x8_t rightSum = vaddq_s16(vaddq_s16(tR, bR), vshlq_n_s16(mR, 1));
		int16x8_t leftSum  = vaddq_s16(vaddq_s16(tL, bL), vshlq_n_s16(mL, 1));
		int16x8_t gx = vsubq_s16(rightSum, leftSum);

		// gy = (tL + 2*tC + tR) - (bL + 2*bC + bR)
		int16x8_t topSum = vaddq_s16(vaddq_s16(tL, tR), vshlq_n_s16(tC, 1));
		int16x8_t botSum = vaddq_s16(vaddq_s16(bL, bR), vshlq_n_s16(bC, 1));
		int16x8_t gy = vsubq_s16(topSum, botSum);

		// |gx| + |gy|
		uint16x8_t mag = vaddq_u16(
			vreinterpretq_u16_s16(vabsq_s16(gx)),
			vreinterpretq_u16_s16(vabsq_s16(gy))
		);

		// shrink to 8-bit with saturation
		uint8x8_t out = vqmovn_u16(mag);

		// write to dstRow
		vst1_u8(&dstRow[c], out);
	}

	// scalar remainder
	for (; c < end; ++c) {
		int tL = topRow[c-1],  tC = topRow[c],  tR = topRow[c+1];
		int mL = midRow[c-1],                mR = midRow[c+1];
		int bL = botRow[c-1],  bC = botRow[c],  bR = botRow[c+1];

		int gx = (tR + 2*mR + bR) - (tL + 2*mL + bL);
		int gy = (tL + 2*tC + tR) - (bL + 2*bC + bR);

		int G = abs(gx) + abs(gy);
		if (G > 255) G = 255;
		dstRow[c] = (uchar)G;
	}
}

/*-----------------------------------------------------------------------*
* Function: fused_sobel
*
* Description: This function fuses grayscale + sobel in one pass using
* 3 grayscale row buffers (ring buffer) per thread to reduce memory traffic
*
* param threadinfo: holds meta data of each thread
*
* return: void
*------------------------------------------------------------------------*/

static inline void fused_sobel(threadArgs* threadinfo){
	// grab frame metadata
	cv::Mat* rgbMat   = threadinfo->rgb;
	cv::Mat* sobelMat = threadinfo->sobel;

	const int rows = rgbMat->rows;
	const int cols = rgbMat->cols;

	int y0 = threadinfo->y0;
	int y1 = threadinfo->y1;

	// local ring buffers
	uchar* row0 = threadinfo->gray_row0;
	uchar* row1 = threadinfo->gray_row1;
	uchar* row2 = threadinfo->gray_row2;

	// helper to clamp row index and compute grayscale row into buffer
	auto load_gray_row = [&](int r, uchar* out){
		if (r < 0) r = 0;
		if (r >= rows) r = rows - 1;
		const cv::Vec3b* srcRow = rgbMat->ptr<cv::Vec3b>(r);
		gray_row_neon(srcRow, out, cols);
	};

	// preload gray(y0-1), gray(y0), gray(y0+1)
	load_gray_row(y0 - 1, row0);
	load_gray_row(y0,     row1);
	load_gray_row(y0 + 1, row2);

	// for each interior output row in this thread stripe
	for (int r = y0; r < y1; r++){
		uchar* dstRow = sobelMat->ptr<uchar>(r);

		// explicitly set left/right borders for this row
		dstRow[0] = 0;
		dstRow[cols - 1] = 0;

		for (int x = 1; x < cols - 1; x += SOBEL_TILE_W){
			int xEnd = x + SOBEL_TILE_W;
			if (xEnd > cols - 1) xEnd = cols - 1; // exclusive

			sobel_row_neon_tiled(row0, row1, row2, dstRow, cols, x, xEnd);
		}

		// rotate ring buffers and load next grayscale row
		uchar* tmp = row0;
		row0 = row1;
		row1 = row2;
		row2 = tmp;

		// next needed grayscale row is r+2
		load_gray_row(r + 2, row2);
	}
}

/*-----------------------------------------------------------------------*
* Function: fnForThread1
*
* Description: This function is used by all threads to call appropriate 
* functions and deal with barriers
*
* param arg: is a struct threadArgs passed in as a void * that needs to be 
* casted for each thread
*
* return: void
-------------------------------------------------------------------------*/

void *fnForThread1(void *arg){
	threadArgs* threadinfo = (threadArgs*)arg;
	int tid = threadinfo->threadindex;

	pin_thread_to_core(tid);

	int ret = PAPI_register_thread();
	if (ret != PAPI_OK) {
		std::cerr << "PAPI_register_thread failed: " << PAPI_strerror(ret) << "\n";
		std::exit(1);
	}

	int EventSet = PAPI_NULL;
	ret = PAPI_create_eventset(&EventSet);
	if (ret != PAPI_OK) {
		std::cerr << "Error creating event set: " << PAPI_strerror(ret) << "\n";
		exit(1);
	}

	// attempt to add L1 cache miss counter — soft fail if unavailable
	bool have_l1dcm = false;
	ret = PAPI_add_event(EventSet, PAPI_L1_DCM);
	if (ret == PAPI_OK) {
		have_l1dcm = true;
	} else {
		if (tid == 0)
			std::cerr << "PAPI L1_DCM unavailable (" << PAPI_strerror(ret)
			          << ") — miss counts will be 0\n";
	}

	// attempt to add cycle counter — soft fail if unavailable
	bool have_cycles = false;
	int cyc_code = PAPI_NULL;
	ret = PAPI_event_name_to_code((char*)"perf::PERF_COUNT_HW_CPU_CYCLES", &cyc_code);
	if (ret == PAPI_OK) {
		ret = PAPI_add_event(EventSet, cyc_code);
		if (ret == PAPI_OK) {
			have_cycles = true;
		} else {
			if (tid == 0)
				std::cerr << "PAPI cycles unavailable (" << PAPI_strerror(ret)
				          << ") — cycle counts will be 0\n";
		}
	}

	// only call PAPI_start/stop if at least one event was added
	const bool papi_active = have_l1dcm || have_cycles;

	// values[] index depends on which events were added and in what order
	const int idx_misses = 0;
	const int idx_cycles = have_l1dcm ? 1 : 0;

	while (!threads_should_stop.load(std::memory_order_acquire)) {
		pthread_barrier_wait(&barrier_start);
		if (threads_should_stop.load(std::memory_order_acquire))
			break;

		if (papi_active) {
			ret = PAPI_start(EventSet);
			if (ret != PAPI_OK){
				std::cerr << "PAPI_start failed: " << PAPI_strerror(ret) << "\n";
				exit(1);
			}
		}

		fused_sobel(threadinfo);

		if (papi_active) {
			long long values[2] = {0, 0};
			ret = PAPI_stop(EventSet, values);
			if (ret != PAPI_OK) {
				std::cerr << "PAPI_stop failed: " << PAPI_strerror(ret) << "\n";
				std::exit(1);
			}
			if (have_l1dcm)  core_misses[tid].fetch_add(values[idx_misses], std::memory_order_relaxed);
			if (have_cycles) core_cycles[tid].fetch_add(values[idx_cycles], std::memory_order_relaxed);
		}

		core_frames[tid].fetch_add(1, std::memory_order_relaxed);

		pthread_barrier_wait(&barrier_done);
	}

	PAPI_cleanup_eventset(EventSet);
	PAPI_destroy_eventset(&EventSet);

	ret = PAPI_unregister_thread();
	if (ret != PAPI_OK) {
		std::cerr << "PAPI_unregister_thread failed: " << PAPI_strerror(ret) << "\n";
		std::exit(1);
	}

	return NULL;
}

/*-----------------------------------------------------------------------*
* Function: startThreads
*
* Description: helper function to start threads and remove cluter in main
*
* return: void
-------------------------------------------------------------------------*/

void startThreads(){
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	int retVal1 = pthread_create(&thread[0], NULL, fnForThread1, (void *)&targetthread[0]);
	int retVal2 = pthread_create(&thread[1], NULL, fnForThread1, (void *)&targetthread[1]);
	int retVal3 = pthread_create(&thread[2], NULL, fnForThread1, (void *)&targetthread[2]);
	int retVal4 = pthread_create(&thread[3], NULL, fnForThread1, (void *)&targetthread[3]);

	(void)retVal1; (void)retVal2; (void)retVal3; (void)retVal4;
}

/*-----------------------------------------------------------------------*
* Function: compute_thread_ranges
*
* Description: This function takes a information about the frame and returns
* the rows each thread will need to compute
*
* param rows: number of total rows in frame
* param num_threads: number of threads we have
* param ranges: the vector where ranges of rows will be stored 
* for each thread
*
* return: void
-------------------------------------------------------------------------*/

void compute_thread_ranges(int rows, int num_threads, std::vector<std::pair<int,int>>&ranges){
	int interior = std::max(0, rows - 2);
	int base = interior / num_threads;
	int rem  = interior % num_threads;

	ranges.resize(num_threads);
	int cur = 1;
	for (int i = 0; i < num_threads; ++i) {
		int add = base + (i < rem ? 1 : 0);
		int y0 = cur; 
		int y1 = cur + add;
		if (add == 0) { y0 = y1 = 1; }
		ranges[i] = { y0, y1 };
		cur = y1; 
	}
}

/*-----------------------------------------------------------------------*
* Function: main
*
* Description: takes a video from the command line and passes it threw
* a grayscale function and sobel filter function then shows each frame
* to the user using threading
*
* param: argv[1] = Video
*
* return: int
-------------------------------------------------------------------------*/

int main (int argc, char *argv[]){
	if (argc < 2) return -1;

	cv::VideoCapture cap(argv[1]);
	if(!cap.isOpened()) return -1;

	cv::Mat frame; // kept for initial setup — workers use frames[] double-buffer
	if(!cap.read(frame)) return 0;

	cv::Mat sobelMat(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));

	std::vector<std::pair<int,int>> ranges;
	compute_thread_ranges(frame.rows, NUM_THREADS, ranges);

	int ret = PAPI_library_init(PAPI_VER_CURRENT);
	if (ret != PAPI_VER_CURRENT){
		std::cerr << "PAPI init failed\n";
		exit(1);
	}

	ret = PAPI_thread_init((unsigned long (*)(void))(pthread_self));
	if (ret != PAPI_OK) {
		std::cerr << "PAPI_thread_init failed: " << PAPI_strerror(ret) << "\n";
		exit(1);
	}

	for (int i = 0; i < NUM_THREADS; i++){
		void* p0 = nullptr;
		void* p1 = nullptr;
		void* p2 = nullptr;

		// round up to nearest 64-byte cache line so no row buffer tail
		// shares a cache line with the next thread's buffer
		size_t rowbuf_sz = ((size_t)frame.cols + 63) & ~63ULL;
		if (posix_memalign(&p0, 64, rowbuf_sz) != 0) { std::cerr << "posix_memalign failed\n"; exit(1); }
		if (posix_memalign(&p1, 64, rowbuf_sz) != 0) { std::cerr << "posix_memalign failed\n"; exit(1); }
		if (posix_memalign(&p2, 64, rowbuf_sz) != 0) { std::cerr << "posix_memalign failed\n"; exit(1); }

		std::memset(p0, 0, rowbuf_sz);
		std::memset(p1, 0, rowbuf_sz);
		std::memset(p2, 0, rowbuf_sz);

		targetthread[i].gray_row0 = (uchar*)p0;
		targetthread[i].gray_row1 = (uchar*)p1;
		targetthread[i].gray_row2 = (uchar*)p2;
	}

	for (int i = 0; i < 4; i++){
		targetthread[i].rgb = &frame;
		targetthread[i].sobel = &sobelMat;
		targetthread[i].y0 = ranges[i].first;
		targetthread[i].y1 = ranges[i].second;
		targetthread[i].threadindex = i;
	}

	pthread_barrier_init(&barrier_start, NULL, 5);
	pthread_barrier_init(&barrier_done, NULL, 5);

	for (int i = 0; i < NUM_THREADS; i++) {
		core_misses[i].store(0, std::memory_order_relaxed);
		core_frames[i].store(0, std::memory_order_relaxed);
		core_cycles[i].store(0, std::memory_order_relaxed);
	}

	startThreads();

	// double-buffer: frames[0]/frames[1] alternate so cap.read for frame N+1
	// overlaps with worker threads processing frame N — decode is now free
	cv::Mat frames[2];
	frames[0] = frame;
	if (!cap.read(frames[1])) frames[1] = frames[0]; // single-frame video fallback
	int cur = 0;

	auto start_time = std::chrono::high_resolution_clock::now();
	while (true){
		cv::Mat& curFrame = frames[cur];
		int next = 1 - cur;

		std::memset(sobelMat.ptr<uchar>(0), 0, (size_t)curFrame.cols);
		std::memset(sobelMat.ptr<uchar>(curFrame.rows - 1), 0, (size_t)curFrame.cols);

		// point all workers at the current frame
		for (int i = 0; i < NUM_THREADS; i++)
			targetthread[i].rgb = &curFrame;

		pthread_barrier_wait(&barrier_start);

		// decode next frame while workers process current — free overlap
		bool got_next = cap.read(frames[next]);

		pthread_barrier_wait(&barrier_done);

		cv::imshow("sobel", sobelMat);
		int key = cv::waitKey(1);
		if (key == 27 || key == 'q') break;

		if (!got_next) break;
		cur = next;
	}

	threads_should_stop.store(true, std::memory_order_release);
	pthread_barrier_wait(&barrier_start);

	for(int i = 0; i < NUM_THREADS; i++){
		pthread_join(thread[i], NULL);
	}

	pthread_barrier_destroy(&barrier_start);
	pthread_barrier_destroy(&barrier_done);

	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end_time - start_time;
	double total_seconds = elapsed.count();

	long long frames = core_frames[0].load();
	double fps = frames / total_seconds;
	std::cout << "FPS:" << fps << "\n";	

	for (int i = 0; i < NUM_THREADS; i++){
		long long misses = core_misses[i].load();
		long long cycles = core_cycles[i].load();
		long long f = core_frames[i].load();
		double missesPerFrame = (f > 0) ? (double)misses / (double)f : 0.0;
		double cyclesPerFrame = (f > 0) ? (double)cycles / (double)f : 0.0;
		std::cout << "Avg misses per frame for core " << i << ": " << missesPerFrame << "\n";
		std::cout << "Avg cycles per frame for core " << i << ": " << cyclesPerFrame << "\n";
		std::cout << "Misses: " << misses << "\n";
		std::cout << "Frames: " << f << "\n";
	}

	for (int i = 0; i < NUM_THREADS; i++){
		free(targetthread[i].gray_row0);
		free(targetthread[i].gray_row1);
		free(targetthread[i].gray_row2);
		targetthread[i].gray_row0 = nullptr;
		targetthread[i].gray_row1 = nullptr;
		targetthread[i].gray_row2 = nullptr;
	}

	cap.release();
	cv::destroyAllWindows();
	return 0;
}
