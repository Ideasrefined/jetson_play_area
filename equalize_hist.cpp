#include <VX/vx.h>
#include <VX/vxu.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>


#include <stdio.h>
#include <queue>
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


using namespace std;
using namespace cv;

using namespace gpu;
using namespace cv::detail;

/* COMPILE STRING 
g++ visionWorks.cpp -o visionWorks -I/usr/local/include/ -L/usr/local/lib/ `pkg-config --cflags --libs visionworks`  `pkg-config --cflags --libs opencv` -Ofast
*/


vx_image createImageFromMat(vx_context& context, cv::Mat& mat);

vx_status createMatFromImage(vx_image& image, cv::Mat& mat);

vx_image createRGBImageFromRGBMat(vx_context& context, cv::Mat& mat);

vx_status createRGBMatFromRGBImage(vx_image& image, cv::Mat& mat);

/* Entry point. */
int main(int argc,char* argv[])
{

	Mat cv_src1 = imread("br1.png");
  Mat cv_dst1 = Mat(cv_src1.size(), cv_src1.type());
  int width = 1280;
  int height = 720;


  /* Image data. */
 	
	
	if (cv_src1.empty() )
    {
        std::cerr << "Can't load input images" << std::endl;
        return -1;
    }
 

  /* Create our context. */
  vx_context context = vxCreateContext();

  /* Image to process. RGB*/
  vx_image image = createRGBImageFromRGBMat(context, cv_src1);



  vx_image input_yuv = vxCreateImage(context, width, height , VX_DF_IMAGE_IYUV);
  vx_image output_yuv = vxCreateImage(context, width, height , VX_DF_IMAGE_IYUV);
  vx_image output_rgb = vxCreateImage(context, width, height , VX_DF_IMAGE_RGB);
  
  vx_image input_y = vxCreateImage(context, width, height , VX_DF_IMAGE_U8);
  vx_image input_u = vxCreateImage(context, width, height , VX_DF_IMAGE_U8);
  vx_image input_v = vxCreateImage(context, width, height , VX_DF_IMAGE_U8);
  vx_status  status;

  if(vxuColorConvert(context, image, input_yuv) != VX_SUCCESS)
  {
    cout <<"failed to convert Color "<<endl;
  }


  vx_df_image img_format;
  vxQueryImage(input_yuv, VX_IMAGE_ATTRIBUTE_FORMAT, &img_format, sizeof(img_format));
  cout <<"Fromat is :"<<img_format<<endl;
  
  vx_size img_planes;
  vxQueryImage(input_yuv, VX_IMAGE_ATTRIBUTE_PLANES, &img_planes, sizeof(img_planes));
  cout <<"Planes are : "<<img_planes<<endl;

  status = vxuChannelExtract(context, input_yuv,  VX_CHANNEL_Y, input_y);
  
  if( status != VX_SUCCESS)
  {

    cout <<"Failed to extract Y with status : "<<status <<endl;
  }

  status = vxuChannelExtract(context, input_yuv,  VX_CHANNEL_U, input_u);
  if( status != VX_SUCCESS)
  {
    cout <<"Failed to extract U with status : "<<status<<endl;
  }
  if(vxuChannelExtract(context, input_yuv,  VX_CHANNEL_V, input_v) != VX_SUCCESS)
  {
    cout <<"Failed to extract V"<<endl;
  }
  
  if(vxuChannelCombine(context, input_y,input_u, input_v, NULL, output_yuv) != VX_SUCCESS)
  {
    cout <<"failed to combine channels "<<endl;
  }

  if(vxuColorConvert(context, output_yuv,output_rgb) != VX_SUCCESS)
  {
    cout <<" failed to convert color to rgb "<<endl;
  }

  createRGBMatFromRGBImage(output_rgb, cv_dst1 );

  imwrite("rgb_image.jpg",cv_dst1);

  vxReleaseContext(&context);
}


vx_image createRGBImageFromRGBMat(vx_context& context, cv::Mat& mat)
{
    vx_imagepatch_addressing_t src_addr = {
        mat.cols, mat.rows, sizeof(vx_uint8)*3, mat.cols * sizeof(vx_uint8)*3, VX_SCALE_UNITY, VX_SCALE_UNITY, 1, 1 };
    void* src_ptr = mat.data;

    vx_image image = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &src_addr, &src_ptr, VX_IMPORT_TYPE_HOST);

    return image;
}

vx_image createImageFromMat(vx_context& context, cv::Mat& mat)
{
    vx_imagepatch_addressing_t src_addr = {
        mat.cols, mat.rows, sizeof(vx_uint8), mat.cols * sizeof(vx_uint8), VX_SCALE_UNITY, VX_SCALE_UNITY, 1, 1 };
    void* src_ptr = mat.data;

    vx_image image = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &src_addr, &src_ptr, VX_IMPORT_TYPE_HOST);

    return image;
}


vx_status createMatFromImage(vx_image& image, cv::Mat& mat)
{
    vx_status status = VX_SUCCESS;
    vx_uint8 *ptr = NULL;

    cout <<"Creating image "<<mat.cols << " " <<mat.rows <<endl;
    vx_rectangle_t rect;
    vxGetValidRegionImage(image, &rect);
    vx_imagepatch_addressing_t addr = {
        mat.cols, mat.rows, sizeof(vx_uint8), mat.cols * sizeof(vx_uint8), VX_SCALE_UNITY, VX_SCALE_UNITY, 1, 1 };

    status = vxAccessImagePatch(image, &rect, 0, &addr, (void **)&ptr, VX_READ_ONLY);
    mat.data = ptr;

    return status;
}

vx_status createRGBMatFromRGBImage(vx_image& image, cv::Mat& mat)
{
    vx_status status = VX_SUCCESS;
    vx_uint8 *ptr = NULL;

    cout <<"Creating RGB image "<<mat.cols << " " <<mat.rows <<endl;
    vx_rectangle_t rect;
    vxGetValidRegionImage(image, &rect);
    vx_imagepatch_addressing_t addr = {
        mat.cols, mat.rows, sizeof(vx_uint8)*3, mat.cols * sizeof(vx_uint8)*3, VX_SCALE_UNITY, VX_SCALE_UNITY, 1, 1 };

    status = vxAccessImagePatch(image, &rect, 0, &addr, (void **)&ptr, VX_READ_ONLY);
    mat.data = ptr;

    return status;
}
