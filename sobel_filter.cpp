#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdint>

using namespace cv;
using namespace std;

// Forward declarations
void to442_grayscale(cv::Mat& image);
cv::Mat to442_sobel(const cv::Mat& image);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: ./sobel <video_path>\n";
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video\n";
        return -1;
    }
    
	cv::Mat frame;

	while(true){
		bool success = cap.read(frame);

		if(!success){
			break;
		}

		to442_grayscale(frame);
		
		// Convert to single-channel for Sobel
		Mat gray;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		
		Mat edges = to442_sobel(gray);
		
		imshow("Sobel", edges);	

		if(waitKey(1) == 27) break; //ESC to quit
	}


	
    

    return 0;
}




void to442_grayscale(cv::Mat& image) {
    int rows = image.rows;
    int cols = image.cols;
    int channels = image.channels();

    uint8_t* data = image.data;
    size_t stride = image.step;

    for (int y = 0; y < rows; y++) {
        uint8_t* row = data + y * stride;

        for (int x = 0; x < cols; x++) {
            uint8_t* pixel = row + x * channels;

            uint8_t gray = (uint8_t)(
                0.2126 * pixel[2] +  // R
                0.7152 * pixel[1] +  // G
                0.0722 * pixel[0]    // B
            );

            pixel[0] = gray;
            pixel[1] = gray;
            pixel[2] = gray;
        }
    }
}





cv::Mat to442_sobel(const cv::Mat& image) {
    int rows = image.rows;
    int cols = image.cols;
    int channels = image.channels(); // should be 1 for grayscale

    uint8_t* data = image.data;
    size_t stride = image.step; // bytes per row

    int G_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int G_y[3][3] = {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1}
    };

    // Create output image (8-bit grayscale)
    cv::Mat filtered_image(rows, cols, CV_8UC1, cv::Scalar(0));

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {

            int16_t gx = 0;
            int16_t gy = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {

                    int px = x + kx;
                    int py = y + ky;

                    // bounds check
                    if (px >= 0 && px < cols && py >= 0 && py < rows) {
                        uint8_t* pixel =
                            data + py * stride + px * channels;

                        gx += (*pixel) * G_x[ky + 1][kx + 1];
                        gy += (*pixel) * G_y[ky + 1][kx + 1];
                    }
                }
            }

            // Gradient magnitude approximation
            int magnitude = std::abs(gx) + std::abs(gy);

            // Clamp to 8-bit
            if (magnitude > (1<<8)) magnitude = 1<<8;

            filtered_image.at<uint8_t>(y, x) = (uint8_t)magnitude;
        }
    }

    return filtered_image;
}

