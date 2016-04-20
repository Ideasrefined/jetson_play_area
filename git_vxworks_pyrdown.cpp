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
#include <VX/vx.h>
#include <VX/vxu.h>

#include <iostream>

#define VX_SAFE_CALL(vxOp) \
    do \
    { \
        vx_status status = (vxOp); \
        if(status != VX_SUCCESS) \
        { \
            std::cout << " failure [status = " << status << "]" << " in file " << __FILE__ << " line " << __LINE__ << std::endl; \
        } \
    } while (0)

#define VX_ASSERT(cond) \
    do \
    { \
        bool status = (cond); \
        if (!status) \
        { \
            std::cout << " failure in file " << __FILE__ << " line " << __LINE__ << std::endl; \
        } \
    } while (0)

vx_image createImageFromMat(vx_context context, const cv::Mat & mat);
vx_status createMatFromImage(cv::Mat &mat, vx_image image);

//#define DUMP_RESULT

int main(int argc, char *argv[])
{
    cv::Mat m = cv::imread("br1.png", cv::IMREAD_GRAYSCALE);
    if (m.empty()){
        return -1;
    }

    vx_context context = vxCreateContext();

    vx_uint32 width  = m.cols;
    vx_uint32 height = m.rows;
    vx_image vx_m    = createImageFromMat(context, m);
    vx_image vx_m2   = vxCreateImage(context, width,  height,  VX_DF_IMAGE_U8);
    vx_image vx_m3   = vxCreateImage(context, width,  height,  VX_DF_IMAGE_U8);

    vx_image vx_l1   = vxCreateImage(context, width/2,  height/2,  VX_DF_IMAGE_U8);
    vx_image vx_l2   = vxCreateImage(context, width/4,  height/4,  VX_DF_IMAGE_U8);
    vx_image vx_l3   = vxCreateImage(context, width/8,  height/8,  VX_DF_IMAGE_U8);
    vx_image vx_l4   = vxCreateImage(context, width/16, height/16, VX_DF_IMAGE_U8);

    int iter   = 100;
    double f   = 1000.0f / cv::getTickFrequency();
    double sum = 0.0;

    for (int i = 0; i <= iter; i++)
    {
        int64 start = cv::getTickCount();
        VX_SAFE_CALL(vxuMultiply(context, vx_m, vx_m2,1,VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN ,vx_m3));
        //VX_SAFE_CALL(vxuHalfScaleGaussian(context, vx_m, vx_l1, 3));
        // VX_SAFE_CALL(vxuHalfScaleGaussian(context, vx_l1, vx_l2, 3));
        // VX_SAFE_CALL(vxuHalfScaleGaussian(context, vx_l2, vx_l3, 3));
        // VX_SAFE_CALL(vxuHalfScaleGaussian(context, vx_l3, vx_l4, 3));
        int64 end = cv::getTickCount();
        if (iter > 0){
            sum += ((end - start) * f);
        }
    }

    std::cout << "time: " << (sum / iter) << " ms" << std::endl;

#ifdef DUMP_RESULT
    cv::Mat l4(cv::Size(width/16, height/16), m.type());
    VX_SAFE_CALL(createMatFromImage(l4, vx_l4));
    cv::imwrite("l4.png", l4);
#endif

    VX_SAFE_CALL(vxReleaseImage(&vx_m));
    VX_SAFE_CALL(vxReleaseImage(&vx_l1));
    VX_SAFE_CALL(vxReleaseImage(&vx_l2));
    VX_SAFE_CALL(vxReleaseImage(&vx_l3));
    VX_SAFE_CALL(vxReleaseImage(&vx_l4));
    VX_SAFE_CALL(vxReleaseContext(&context));

    return 0;
}

vx_df_image convertMatTypeToImageFormat(int mat_type)
{
    switch (mat_type)
    {
    case CV_8UC1:
        return VX_DF_IMAGE_U8;
    case CV_16SC1:
        return VX_DF_IMAGE_S16;
    case CV_8UC3:
        return VX_DF_IMAGE_RGB;
    case CV_8UC4:
        return VX_DF_IMAGE_RGBX;
    }
    CV_Error(CV_StsUnsupportedFormat, "Unsupported format");
    return 0;
}

vx_image createImageFromMat(vx_context context, const cv::Mat & mat)
{
    vx_imagepatch_addressing_t patch = { (vx_uint32)mat.cols, (vx_uint32)mat.rows,
        (vx_int32)mat.elemSize(), (vx_int32)mat.step,
        VX_SCALE_UNITY, VX_SCALE_UNITY,
        1u, 1u };
    void *ptr = (void*)mat.ptr();
    vx_df_image format = convertMatTypeToImageFormat(mat.type());
    return vxCreateImageFromHandle(context, format, &patch, (void **)&ptr, VX_IMPORT_TYPE_HOST);
}

vx_status createMatFromImage(cv::Mat &mat, vx_image image)
{
    vx_status status   = VX_SUCCESS;
    vx_uint32 width    = 0;
    vx_uint32 height   = 0;
    vx_df_image format = VX_DF_IMAGE_VIRT;
    int cv_format      = 0;
    vx_size planes     = 0;

    VX_SAFE_CALL(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH,  &width,  sizeof(width)));
    VX_SAFE_CALL(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
    VX_SAFE_CALL(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
    VX_SAFE_CALL(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_PLANES, &planes, sizeof(planes)));

    switch (format)
    {
    case VX_DF_IMAGE_U8:
        cv_format = CV_8U;
        break;
    case VX_DF_IMAGE_S16:
        cv_format = CV_16S;
        break;
    case VX_DF_IMAGE_RGB:
        cv_format = CV_8UC3;
        break;
    default:
        return VX_ERROR_INVALID_FORMAT;
    }

    vx_rectangle_t rect{0, 0, width, height};
    vx_uint8 *src[4] = {NULL, NULL, NULL, NULL};
    vx_uint32 p;
    void *ptr = NULL;
    vx_imagepatch_addressing_t addr[4] = {0, 0, 0, 0};
    vx_uint32 y = 0u;

    for (p = 0u; (p < (int)planes); p++)
    {
        VX_SAFE_CALL(vxAccessImagePatch(image, &rect, p, &addr[p], (void **)&src[p], VX_READ_ONLY));
        size_t len = addr[p].stride_x * (addr[p].dim_x * addr[p].scale_x) / VX_SCALE_UNITY;
        for (y = 0; y < height; y += addr[p].step_y)
        {
            ptr = vxFormatImagePatchAddress2d(src[p], 0, y - rect.start_y, &addr[p]);
            memcpy(mat.data + y * mat.step, ptr, len);
        }
    }

    for (p = 0u; p < (int)planes; p++)
    {
        VX_SAFE_CALL(vxCommitImagePatch(image, &rect, p, &addr[p], src[p]));
    }

    return status;
}