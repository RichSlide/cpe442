#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdint>
#include <pthread.h>
#include <algorithm>
#include <cmath>
#include <arm_neon.h>

using namespace cv;
using namespace std;

constexpr int NUM_THREADS = 4;

struct SharedFrame {
    cv::Mat* frame;
    cv::Mat* gray;
    cv::Mat* sobel;

    int rows;
    int cols;
    int channels;

    pthread_barrier_t* barrier;
    bool* running;

    cv::VideoCapture* cap;
};

struct ThreadArgs {
    SharedFrame* shared;
    int threadId;
    int rowsPerThread;
};

// Forward declarations
void* worker(void* arg);
void toGrayscale(ThreadArgs* thread);
void toSobel(ThreadArgs* thread);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: ./sobel <video_path>\n";
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video\n";
        return -1;
    }

    cv::Mat frame;
    bool success = cap.read(frame);

    if (!success) {
        cerr << "Error: Could not read first frame\n";
        return -1;
    }

    int rows = frame.rows;
    int cols = frame.cols;
    int channels = frame.channels();
    int rowsPerThread = (rows + NUM_THREADS - 1) / NUM_THREADS; // round up

    pthread_t threads[NUM_THREADS];
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, NUM_THREADS); // only worker threads

    cv::Mat gray(rows, cols, CV_8UC1);
    cv::Mat sobel(rows, cols, CV_8UC1);
    bool running = true;

    SharedFrame shared = {
        &frame, &gray, &sobel,
        rows, cols, channels,
        &barrier, &running,
        &cap
    };

    ThreadArgs threadArgs[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        threadArgs[i] = { &shared, i, rowsPerThread };
        pthread_create(&threads[i], nullptr, worker, &threadArgs[i]);
    }

    while (running) {
        imshow("Original", frame);
        imshow("Grayscale", gray);
        imshow("Sobel", sobel);

        if (waitKey(30) == 27) { // ESC
            running = false;
            break;
        }

        success = cap.read(frame);
        if (!success) {
            running = false;
            break;
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], nullptr);
    }

    pthread_barrier_destroy(&barrier);
    cap.release();
    destroyAllWindows();

    return 0;
}

void* worker(void* arg) {
    ThreadArgs* thread = static_cast<ThreadArgs*>(arg);

    while (*(thread->shared->running)) {
        toGrayscale(thread);
        pthread_barrier_wait(thread->shared->barrier);

        toSobel(thread);
        pthread_barrier_wait(thread->shared->barrier);
    }

    return nullptr;
}

void toGrayscale(ThreadArgs* thread) {
    cv::Mat* frame = thread->shared->frame;
    cv::Mat* gray  = thread->shared->gray;

    uint8_t* srcData = frame->data;
    uint8_t* dstData = gray->data;
    size_t srcStride = frame->step;
    size_t dstStride = gray->step;

    int startRow = thread->threadId * thread->rowsPerThread;
    int endRow   = std::min(startRow + thread->rowsPerThread, thread->shared->rows);

    int cols = thread->shared->cols;
    int channels = thread->shared->channels; // should be 3 for BGR

    // Fixed-point weights (sum to 256)
    const uint8_t WB = 19;
    const uint8_t WG = 183;
    const uint8_t WR = 54;

    for (int y = startRow; y < endRow; y++) {
        uint8_t* srcRow = srcData + y * srcStride;
        uint8_t* dstRow = dstData + y * dstStride;

        int x = 0;

        // Process 8 pixels per loop: 8 * 3 = 24 bytes
        for (; x <= cols - 8; x += 8) {
            const uint8_t* p = srcRow + x * channels;

            // Load 8 interleaved BGR pixels
            // bgr.val[0]=B, val[1]=G, val[2]=R
            uint8x8x3_t bgr = vld3_u8(p);

            // Widen and multiply: (B*WB + G*WG + R*WR)
            uint8x8_t wb = vdup_n_u8(WB);
            uint8x8_t wg = vdup_n_u8(WG);
            uint8x8_t wr = vdup_n_u8(WR);
            
            uint16x8_t acc = vmull_u8(bgr.val[0], wb);
            acc = vmlal_u8(acc, bgr.val[1], wg);
            acc = vmlal_u8(acc, bgr.val[2], wr);

            // Add rounding offset then shift down by 8
            acc = vaddq_u16(acc, vdupq_n_u16(128));
            uint8x8_t out = vshrn_n_u16(acc, 8);

            vst1_u8(dstRow + x, out);
        }

        // Scalar tail
        for (; x < cols; x++) {
            uint8_t* srcPixel = srcRow + x * channels;
            // BGR order
            uint16_t b = srcPixel[0];
            uint16_t g = srcPixel[1];
            uint16_t r = srcPixel[2];

            uint16_t val = (WB*b + WG*g + WR*r + 128) >> 8;
            dstRow[x] = (uint8_t)val;
        }
    }
}


void toSobel(ThreadArgs* thread) {
    cv::Mat* gray  = thread->shared->gray;
    cv::Mat* sobel = thread->shared->sobel;

    int rows = thread->shared->rows;
    int cols = thread->shared->cols;

    uint8_t* src = gray->data;
    uint8_t* dst = sobel->data;
    size_t stride = gray->step;

    int startRow = thread->threadId * thread->rowsPerThread;
    int endRow   = std::min(startRow + thread->rowsPerThread, rows);

    // Clamp to interior rows for vector path
    int y0 = std::max(startRow, 1);
    int y1 = std::min(endRow, rows - 1);

    // Optional: clear borders for rows this thread owns
    // (so we don't leave old pixels there)
    for (int y = startRow; y < endRow; y++) {
        if (y == 0 || y == rows - 1) {
            std::memset(dst + y * stride, 0, cols);
        } else {
            dst[y * stride + 0] = 0;
            dst[y * stride + (cols - 1)] = 0;
        }
    }

    for (int y = y0; y < y1; y++) {
        const uint8_t* row0 = src + (y - 1) * stride;
        const uint8_t* row1 = src + (y    ) * stride;
        const uint8_t* row2 = src + (y + 1) * stride;

        uint8_t* outRow = dst + y * stride;

        int x = 1;

        // Need x+8 to be valid when we load starting at (x-1) for vext
        for (; x <= cols - 9; x += 8) {
            // Load 16 bytes starting at x-1 (covers x-1 .. x+14)
            uint8x16_t t0 = vld1q_u8(row0 + x - 1);
            uint8x16_t t1 = vld1q_u8(row1 + x - 1);
            uint8x16_t t2 = vld1q_u8(row2 + x - 1);

            // For 8 outputs (x..x+7), we need left/center/right (shift 0/1/2)
            uint8x8_t tl = vget_low_u8(t0);
            uint8x8_t tc = vget_low_u8(vextq_u8(t0, t0, 1));
            uint8x8_t tr = vget_low_u8(vextq_u8(t0, t0, 2));

            uint8x8_t ml = vget_low_u8(t1);
            uint8x8_t mc = vget_low_u8(vextq_u8(t1, t1, 1));
            uint8x8_t mr = vget_low_u8(vextq_u8(t1, t1, 2));

            uint8x8_t bl = vget_low_u8(t2);
            uint8x8_t bc = vget_low_u8(vextq_u8(t2, t2, 1));
            uint8x8_t br = vget_low_u8(vextq_u8(t2, t2, 2));

            // Gx = (tr - tl) + 2*(mr - ml) + (br - bl)
            // Widen to int16 (values 0..255) then subtract signed => result in [-255, 255]
            int16x8_t tl_s = vreinterpretq_s16_u16(vmovl_u8(tl));
            int16x8_t tr_s = vreinterpretq_s16_u16(vmovl_u8(tr));
            int16x8_t ml_s = vreinterpretq_s16_u16(vmovl_u8(ml));
            int16x8_t mr_s = vreinterpretq_s16_u16(vmovl_u8(mr));
            int16x8_t bl_s = vreinterpretq_s16_u16(vmovl_u8(bl));
            int16x8_t br_s = vreinterpretq_s16_u16(vmovl_u8(br));
            
            int16x8_t dx_top = vsubq_s16(tr_s, tl_s);
            int16x8_t dx_mid = vsubq_s16(mr_s, ml_s);
            int16x8_t dx_bot = vsubq_s16(br_s, bl_s);
            
        

            int16x8_t gx = vaddq_s16(vaddq_s16(dx_top, vshlq_n_s16(dx_mid, 1)), dx_bot);

            // Gy = (tl + 2*tc + tr) - (bl + 2*bc + br)
            uint16x8_t tl16 = vmovl_u8(tl);
            uint16x8_t tc16 = vmovl_u8(tc);
            uint16x8_t tr16 = vmovl_u8(tr);

            uint16x8_t bl16 = vmovl_u8(bl);
            uint16x8_t bc16 = vmovl_u8(bc);
            uint16x8_t br16 = vmovl_u8(br);

            int16x8_t tc_s = vreinterpretq_s16_u16(vmovl_u8(tc));
            int16x8_t bc_s = vreinterpretq_s16_u16(vmovl_u8(bc));
            
            int16x8_t top = vaddq_s16(vaddq_s16(tl_s, tr_s), vshlq_n_s16(tc_s, 1));
            int16x8_t bot = vaddq_s16(vaddq_s16(bl_s, br_s), vshlq_n_s16(bc_s, 1));
            
            int16x8_t gy = vsubq_s16(top, bot);

            // magnitude = abs(gx) + abs(gy), saturated to 255
            uint16x8_t ax = vreinterpretq_u16_s16(vabsq_s16(gx));
            uint16x8_t ay = vreinterpretq_u16_s16(vabsq_s16(gy));
            uint16x8_t mag = vqaddq_u16(ax, ay);
            uint8x8_t out = vqmovn_u16(mag);
            vst1_u8(outRow + x, out);;

       
        }

        // Scalar tail for remaining interior pixels until cols-2
        for (; x < cols - 1; x++) {
            int16_t gx = 0, gy = 0;

            uint8_t p00 = row0[x - 1], p01 = row0[x], p02 = row0[x + 1];
            uint8_t p10 = row1[x - 1],             p12 = row1[x + 1];
            uint8_t p20 = row2[x - 1], p21 = row2[x], p22 = row2[x + 1];

            gx = (int16_t)(p02 - p00) + (int16_t)2 * (p12 - p10) + (int16_t)(p22 - p20);
            gy = (int16_t)(p00 + 2*p01 + p02) - (int16_t)(p20 + 2*p21 + p22);

            int mag = std::abs(gx) + std::abs(gy);
            if (mag > 255) mag = 255;
            outRow[x] = (uint8_t)mag;
        }
    }
}
