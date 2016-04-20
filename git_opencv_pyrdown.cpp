/* OPENCV RELATED */
#include <cv.h>
#include <highgui.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <opencv2/gpu/gpu.hpp>  

#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <opencv2/opencv.hpp>

#include <iostream>

int main(int argc, char *argv[])
{
    cv::Mat m = cv::imread("br1.png", cv::IMREAD_GRAYSCALE);
    if (m.empty()){
        return -1;
    }

    double f   = 1000.0f / cv::getTickFrequency();
    double sum = 0.0;
    int iter = 100;

    cv::gpu::GpuMat d_m(m);
    cv::gpu::GpuMat d_m2(m);
    cv::gpu::GpuMat d_m3(m);
    d_m3.setTo(cv::Scalar(255));

    for (int i = 0; i <= iter; i++)
    {
        int64 start = cv::getTickCount();
        cv::gpu::multiply(d_m, d_m2, d_m3);
        int64 end = cv::getTickCount();

        if (iter > 0){
            sum += ((end - start) * f);
        }
    }

    std::cout << "time: " << (sum / iter) << " ms" << std::endl;

    return 0;
}