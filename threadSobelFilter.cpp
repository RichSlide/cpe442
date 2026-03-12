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


