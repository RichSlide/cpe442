#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdint>
#include <pthread.h>

using namespace cv;
using namespace std;

const int khang_num_threads = 4;

struct KhangSharedFrame {
    cv::Mat* khang_frame;
    cv::Mat* khang_gray;
    cv::Mat* khang_sobel;
    
    int khang_rows;
    int khang_cols;
    int khang_channels;
    
    pthread_barrier_t* khang_barrier;
    bool* khang_running;
    
    cv::VideoCapture* khang_cap;
};

struct KhangThreadArgs {
    KhangSharedFrame* khang_shared;
    int khang_thread_id;
    int khang_row_per_thread;
};

// Forward declarations
void* khang_worker(void* arg);
void khang_to442_grayscale(KhangThreadArgs* khang_thread);
void khang_to442_sobel(KhangThreadArgs* khang_thread);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: ./sobel <video_path>\n";
        return -1;
    }

    cv::VideoCapture khang_cap(argv[1]);
    if (!khang_cap.isOpened()) {
        std::cerr << "Error: Could not open video\n";
        return -1;
    }

    cv::Mat khang_frame;
    bool khang_success = khang_cap.read(khang_frame);
    
    if(!khang_success){
        cerr << "Error: Could not read first frame\n";
        return -1;
    }

    int khang_rows = khang_frame.rows;
    int khang_row_per_thread = (khang_rows + khang_num_threads - 1) / khang_num_threads; // round up
    int khang_cols = khang_frame.cols;
    int khang_channels = khang_frame.channels();

    pthread_t khang_threads[khang_num_threads];
    pthread_barrier_t khang_barrier;
    pthread_barrier_init(&khang_barrier, nullptr, khang_num_threads); // Only worker threads

    cv::Mat khang_gray(khang_rows, khang_cols, CV_8UC1);  // Single channel for grayscale
    cv::Mat khang_sobel(khang_rows, khang_cols, CV_8UC1);
    bool khang_running = true;

    KhangSharedFrame khang_shared = {&khang_frame, &khang_gray, &khang_sobel, khang_rows, khang_cols, khang_channels, &khang_barrier, &khang_running, &khang_cap};

    KhangThreadArgs khang_thread_args[khang_num_threads];
    
    // Create threads
    for (int khang_i = 0; khang_i < khang_num_threads; khang_i++) {
        khang_thread_args[khang_i] = {&khang_shared, khang_i, khang_row_per_thread};
        pthread_create(&khang_threads[khang_i], nullptr, khang_worker, &khang_thread_args[khang_i]);
    }

    // Main loop - just display and read frames
    while(khang_running) {
        // Display result
        imshow("Original", khang_frame);
        imshow("Grayscale", khang_gray);
        imshow("Sobel", khang_sobel);
        
        if(waitKey(30) == 27) { // ESC to quit
            khang_running = false;
            break;
        }
        
        // Read next frame
        khang_success = khang_cap.read(khang_frame);
        if(!khang_success){
            khang_running = false;
            break;
        }
    }

    // Wait for all threads to finish
    for (int khang_i = 0; khang_i < khang_num_threads; khang_i++) {
        pthread_join(khang_threads[khang_i], nullptr);
    }

    pthread_barrier_destroy(&khang_barrier);
    khang_cap.release();
    destroyAllWindows();
    
    return 0;
}

void* khang_worker(void* khang_arg) {
    KhangThreadArgs* khang_thread = (KhangThreadArgs*)khang_arg;
    
    while(*(khang_thread->khang_shared->khang_running)) {
        // Process grayscale
        khang_to442_grayscale(khang_thread);
        pthread_barrier_wait(khang_thread->khang_shared->khang_barrier);
        
        // Process sobel
        khang_to442_sobel(khang_thread);
        pthread_barrier_wait(khang_thread->khang_shared->khang_barrier);
    }
    
    return nullptr;
}

void khang_to442_grayscale(KhangThreadArgs* khang_thread) {
    cv::Mat* khang_frame = khang_thread->khang_shared->khang_frame;
    cv::Mat* khang_gray = khang_thread->khang_shared->khang_gray;
    
    uint8_t* khang_src_data = khang_frame->data;
    uint8_t* khang_dst_data = khang_gray->data;
    size_t khang_src_stride = khang_frame->step;
    size_t khang_dst_stride = khang_gray->step;
    
    int khang_start_row = khang_thread->khang_thread_id * khang_thread->khang_row_per_thread;
    int khang_end_row = min(khang_start_row + khang_thread->khang_row_per_thread, khang_thread->khang_shared->khang_rows);
    
    for (int khang_y = khang_start_row; khang_y < khang_end_row; khang_y++) {
        uint8_t* khang_src_row = khang_src_data + khang_y * khang_src_stride;
        uint8_t* khang_dst_row = khang_dst_data + khang_y * khang_dst_stride;
        
        for (int khang_x = 0; khang_x < khang_thread->khang_shared->khang_cols; khang_x++) {
            uint8_t* khang_src_pixel = khang_src_row + khang_x * khang_thread->khang_shared->khang_channels;
            
            uint8_t khang_gray_val = (uint8_t)(
                0.2126 * khang_src_pixel[2] +  // R
                0.7152 * khang_src_pixel[1] +  // G
                0.0722 * khang_src_pixel[0]    // B
            );
            
            khang_dst_row[khang_x] = khang_gray_val;
        }
    }
}

void khang_to442_sobel(KhangThreadArgs* khang_thread) {
    cv::Mat* khang_image = khang_thread->khang_shared->khang_gray;
    cv::Mat* khang_filtered_image = khang_thread->khang_shared->khang_sobel;
    
    int khang_rows = khang_thread->khang_shared->khang_rows;
    int khang_cols = khang_thread->khang_shared->khang_cols;
    
    uint8_t* khang_data = khang_image->data;
    size_t khang_stride = khang_image->step;
    
    int khang_G_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    int khang_G_y[3][3] = {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1}
    };
    
    int khang_start_row = khang_thread->khang_thread_id * khang_thread->khang_row_per_thread;
    int khang_end_row = min(khang_start_row + khang_thread->khang_row_per_thread, khang_rows);
    
    for (int khang_y = khang_start_row; khang_y < khang_end_row; khang_y++) {
        for (int khang_x = 0; khang_x < khang_cols; khang_x++) {
            int16_t khang_gx = 0;
            int16_t khang_gy = 0;
            
            for (int khang_ky = -1; khang_ky <= 1; khang_ky++) {
                for (int khang_kx = -1; khang_kx <= 1; khang_kx++) {
                    int khang_px = khang_x + khang_kx;
                    int khang_py = khang_y + khang_ky;
                    
                    // bounds check
                    if (khang_px >= 0 && khang_px < khang_cols && khang_py >= 0 && khang_py < khang_rows) {
                        uint8_t khang_pixel_val = khang_data[khang_py * khang_stride + khang_px];
                        
                        khang_gx += khang_pixel_val * khang_G_x[khang_ky + 1][khang_kx + 1];
                        khang_gy += khang_pixel_val * khang_G_y[khang_ky + 1][khang_kx + 1];
                    }
                }
            }
            
            // Gradient magnitude approximation
            int khang_magnitude = std::abs(khang_gx) + std::abs(khang_gy);
            
            // Clamp to 8-bit (0-255)
            if (khang_magnitude > 255) khang_magnitude = 255;
            
            khang_filtered_image->at<uint8_t>(khang_y, khang_x) = (uint8_t)khang_magnitude;
        }
    }
}
