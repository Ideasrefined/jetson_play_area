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

// Mat型データからvx_image型データを生成するヘルパー関数（自作）
vx_image createImageFromMat(vx_context& context, cv::Mat& mat);

// vx_image型データからMat型データを生成するヘルパー関数（自作）
vx_status createMatFromImage(vx_image& image, cv::Mat& mat);


/* Entry point. */
int main(int argc,char* argv[])
{

	Mat cv_src1 = imread("br1.png");
  int width = 1280;
  int height = 720;

  int half_width = width/2;
  int half_height = height/2;
	Mat dstMat(cv_src1.size(), CV_8UC3);
  Mat dstMat_c1(cv_src1.size(), CV_8UC1);
  Mat half_dstMat(Size(width/16,height/16),cv_src1.type());

  /* Image data. */
 	
	
	if (cv_src1.empty() )
    {
        std::cerr << "Can't load input images" << std::endl;
        return -1;
    }
 

  /* Create our context. */
  vx_context context = vxCreateContext();

  /* Image to process. */
  vx_image image = createImageFromMat(context, cv_src1);
   //NVXIO_CHECK_REFERENCE(image);

 

  /* Intermediate images. */
  vx_image dx = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
  vx_image dy = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
  vx_image mag = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
  vx_image half_image = vxCreateImage(context, half_width, half_height,  VX_DF_IMAGE_U8);
  vx_image half_image_2 = vxCreateImage(context, half_width/2, half_height/2,  VX_DF_IMAGE_U8);
  vx_image half_image_3 = vxCreateImage(context, half_width/4, half_height/4,  VX_DF_IMAGE_U8);
  vx_image half_image_4 = vxCreateImage(context, half_width/8, half_height/8,  VX_DF_IMAGE_U8);


  int64 e = getTickCount();
  int iter = 100;
  float sum = 0.0;
  
  /* SOBEL OPERATION */
#ifdef SOBEL
  //for(int i = 0 ; i < iter ; i ++)
  {
  		e = getTickCount();
	  /* Perform Sobel convolution. */
	  if (vxuSobel3x3(context,image,dx, dy)!=VX_SUCCESS)
	  {
	    printf("ERROR: failed to do sobel!\n");
	  }

	  /* Calculate magnitude from gradients. */
	  if (vxuMagnitude(context,dx,dy,mag)!=VX_SUCCESS)
	  {
	    printf("ERROR: failed to do magnitude!\n");
	  }

	   //Convert result back to U8 image. 
	  if (vxuConvertDepth(context,mag,image,VX_CONVERT_POLICY_WRAP,0)!=VX_SUCCESS)
	  {
	    printf("ERROR: failed to do color convert!\n");
	  }

	  sum += (getTickCount() - e) / getTickFrequency();
	}
#endif
  /* END SOBEL OPERATION */

  cout <<"Vx works sobel "<<sum / iter<<endl;
  

  e = getTickCount();
  iter = 100;
  for(int i = 0 ; i < iter; i ++)
  {
     e = getTickCount();
    /* RESIZEZ OPERATION */
    if(vxuHalfScaleGaussian(context,image,half_image,3) != VX_SUCCESS)
    {
      cout <<"ERROR :"<<"failed to perform scaling"<<endl;
    }

    if(vxuHalfScaleGaussian(context,half_image,half_image_2,3) != VX_SUCCESS)
    {
      cout <<"ERROR :"<<"failed to perform scaling"<<endl;
    }

    if(vxuHalfScaleGaussian(context,half_image_2,half_image_3,3) != VX_SUCCESS)
    {
      cout <<"ERROR :"<<"failed to perform scaling"<<endl;
    }

    if(vxuHalfScaleGaussian(context,half_image_3,half_image_4,3) != VX_SUCCESS)
    {
      cout <<"ERROR :"<<"failed to perform scaling"<<endl;
    }


    sum += (getTickCount() - e) / getTickFrequency();  
  }

  cout <<"Manual Gaussian " <<sum/iter<<endl;

  
  vx_size levels = 4;
  
  vx_pyramid pyr = vxCreatePyramid (context, levels, VX_SCALE_PYRAMID_HALF , width, height, VX_DF_IMAGE_U8 );
  for(int i = 0 ; i < iter; i ++)
  {
    e = getTickCount();
    if(vxuGaussianPyramid(context,image,pyr) != VX_SUCCESS)
    {
      cout <<"Failed to create pyramid"<<endl;
    }
    sum += (getTickCount() - e) / getTickFrequency();  
  }
  cout <<"GaussianPyramid " <<sum/iter<<endl;


  vx_image r_image = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
  vx_image g_image = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
  vx_image b_image = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
  vx_image t_image = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);

  vx_image r_mag = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
  vx_image g_mag = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
  vx_image b_mag = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);


  vx_image input_yuv = vxCreateImage(context,  width,height , VX_DF_IMAGE_IYUV);
  vx_image output_yuv = vxCreateImage(context,  width,height , VX_DF_IMAGE_IYUV);

  if(vxuColorConvert(context, image, input_yuv) != VX_SUCCESS)
  {
    cout <<"Failed to convert color "<<endl;
  }


  if(vxuChannelExtract(context, input_yuv,VX_CHANNEL_Y,r_image) != VX_SUCCESS)
  {
      cout <<"Failed to convert"<<endl;
  }

  for (int i = 0; i < 0; ++i)
  {
    e = getTickCount();
    if(vxuChannelExtract(context, image,VX_CHANNEL_R,r_image) != VX_SUCCESS)
    {
      cout <<"Failed to convert"<<endl;
    }
    //vxuChannelExtract(context, image,VX_CHANNEL_R,t_image);

    if(vxuChannelExtract(context, image,VX_CHANNEL_G,g_image) != VX_SUCCESS)
    {
     cout <<"Failed to convert"<<endl; 
    }

    if(vxuChannelExtract(context, image,VX_CHANNEL_B,b_image) != VX_SUCCESS)
    {
      cout <<"FAILED to convert"<<endl;
    }
/*
    if(vxuMultiply(context,t_image,r_image,1, VX_CONVERT_POLICY_WRAP,  VX_ROUND_POLICY_TO_NEAREST_EVEN,r_mag) != VX_SUCCESS)
    {
      cout <<"Failed to multiply"<<endl;
    }
    if(vxuMultiply(context,t_image,g_image,1, VX_CONVERT_POLICY_WRAP,  VX_ROUND_POLICY_TO_NEAREST_EVEN,g_mag) != VX_SUCCESS)
    {
      cout <<"Failed to multiply"<<endl;
    }
    if(vxuMultiply(context,t_image,b_image,1, VX_CONVERT_POLICY_WRAP,  VX_ROUND_POLICY_TO_NEAREST_EVEN,b_mag) != VX_SUCCESS)
    {
      cout <<"Failed to multiply"<<endl;
    }
*/
    // if (vxuConvertDepth(context,r_mag,t_image,VX_CONVERT_POLICY_WRAP,0)!=VX_SUCCESS)
    // {
    //   printf("ERROR: failed to do color convert!\n");
    // }

    sum += (getTickCount() - e) / getTickFrequency();  
  }
  cout <<"Multiply " <<sum/iter<<endl;
  
  createMatFromImage(image,dstMat);
  imwrite("RES_SRC.jpg",dstMat);

  createMatFromImage(r_image,dstMat_c1);
  imwrite("RES_r.jpg",dstMat_c1);

  // createMatFromImage(g_image,dstMat);
  // imwrite("RES_g.jpg",dstMat);

  // createMatFromImage(b_image,dstMat);
  // imwrite("RES_b.jpg",dstMat);
  /* Tidy up. */
  vxReleaseImage(&dx);
  vxReleaseImage(&dy);
  vxReleaseImage(&mag);
  vxReleaseContext(&context);
}


// Mat型データからvx_image型データを生成するヘルパー関数（自作）
vx_image createImageFromMat(vx_context& context, cv::Mat& mat)
{
    vx_imagepatch_addressing_t src_addr = {
        mat.cols, mat.rows, sizeof(vx_uint8), mat.cols * sizeof(vx_uint8), VX_SCALE_UNITY, VX_SCALE_UNITY, 1, 1 };
    void* src_ptr = mat.data;

    vx_image image = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &src_addr, &src_ptr, VX_IMPORT_TYPE_HOST);

    return image;
}

// vx_image型データからMat型データを生成するヘルパー関数（自作）
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
