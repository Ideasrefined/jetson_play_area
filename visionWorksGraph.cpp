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


// Mat型データからvx_image型データを生成するヘルパー関数（自作）
vx_image createImageFromMat(vx_context& context, cv::Mat& mat);

// vx_image型データからMat型データを生成するヘルパー関数（自作）
vx_status createMatFromImage(vx_image& image, cv::Mat& mat);

/*
g++ visionWorksGraph.cpp -o visioGraph -I/usr/local/include/ -L/usr/local/lib/ `pkg-config --cflags --libs visionworks`  `pkg-config --cflags --libs opencv` -Ofast

*/

/* Entry point. */
int main(int argc,char* argv[])
{

  Mat cv_src1 = imread("br1.png", IMREAD_GRAYSCALE);
  Mat dstMat(cv_src1.size(), cv_src1.type());
  /* Image data. */
  int width = 1280;
  int height = 720;
  Mat dstMat2(Size(width/16,height/16),cv_src1.type());
  
  
  if (cv_src1.empty() )
    {
        std::cerr << "Can't load input images" << std::endl;
        return -1;
    }
 

  /* Create our context. */
  vx_context context = vxCreateContext();

  /* Image to process. */
  int64 e = getTickCount();
  vx_image image = createImageFromMat(context, cv_src1);
  cout <<"TIME TO CONVERT MAT TO VxIMAGE "<<(getTickCount() - e) / getTickFrequency() <<endl;
   //NVXIO_CHECK_REFERENCE(image);

  /* Intermediate images. */
  vx_image dx = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
  vx_image dy = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
  vx_image mag = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
  vx_image mag2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);

  vx_graph graph = vxCreateGraph( context );
  vx_graph graph2 = vxCreateGraph( context );

  vx_image intermediate_a = vxCreateVirtualImage( graph, width, height, VX_DF_IMAGE_U8 );
  vx_image intermediate1 = vxCreateVirtualImage( graph, width/2, height/2, VX_DF_IMAGE_U8 );
  vx_image intermediate2 = vxCreateVirtualImage( graph, width/4, height/4, VX_DF_IMAGE_U8 );
  vx_image intermediate3 = vxCreateVirtualImage( graph, width/8, height/8, VX_DF_IMAGE_U8 );
  vx_image intermediate4 = vxCreateVirtualImage( graph, width/16, height/16, VX_DF_IMAGE_U8 );//vxCreateImage(context, width/16, height/16,  VX_DF_IMAGE_U8);// 



   /* RESIZEZ OPERATION */
    //vx_node bitnot = vxNotNode (graph, image, intermediate_a);
    vx_node node1 = vxHalfScaleGaussianNode(graph,image,intermediate1,3);
    // if(vxHalfScaleGaussianNode(graph,image,intermediate1,3)  == 0)
    // {
    //   cout <<"ERROR :"<<"failed to perform scaling"<<endl;
    // }

    vx_node node2 = vxHalfScaleGaussianNode(graph,intermediate1,intermediate2,3) ;
    // if(vxHalfScaleGaussianNode(graph,intermediate1,intermediate2,3)  == 0)
    // {
    //   cout <<"ERROR :"<<"failed to perform scaling"<<endl;
    // }

    vx_node node3 = vxHalfScaleGaussianNode(graph,intermediate2,intermediate3,3);
    
    // if(vxHalfScaleGaussianNode(graph,intermediate2,intermediate3,3)  == 0)
    // {
    //   cout <<"ERROR :"<<"failed to perform scaling"<<endl;
    // }

    vx_node node4 = vxHalfScaleGaussianNode(graph,intermediate3,intermediate4,3);
    
    /*if(vxHalfScaleGaussianNode(graph,intermediate3,intermediate4,3)  == 0)
    {
      cout <<"ERROR :"<<"failed to perform scaling"<<endl;
    }*/




    vx_pyramid pyr = vxCreateVirtualPyramid  (graph2, 4, VX_SCALE_PYRAMID_HALF , width, height, VX_DF_IMAGE_U8 );
    vx_node pyrNode= vxGaussianPyramidNode  (graph2,image,pyr);
  


/*

  if(vxSobel3x3Node(graph,image,intermediate1,intermediate2) == 0)
  {
    printf("FAILED TO Create 1 graph node");
  }

  if(vxMagnitudeNode(graph,intermediate1,intermediate2,intermediate3) == 0)
  {
      printf("ERROR: failed to do magnitude!\n");
  }

  if(vxConvertDepthNode(graph,intermediate3,mag2,VX_CONVERT_POLICY_WRAP,0) == 0)
  {
    printf("ERROR failed to do color convert");
  }
*/
  if(vxVerifyGraph( graph ) != VX_SUCCESS)
  {
    cout <<"failed to verify graph "<<endl;
  }
  if(vxVerifyGraph( graph2 ) != VX_SUCCESS)
  {
    cout <<"failed to verify graph "<<endl;
  }
  e = getTickCount();
  int iter = 100;
  float sum = 0.0;

  for(int i = 0 ; i < iter ; i ++)
  {
      
    e = getTickCount();
    vxProcessGraph( graph2 ); // run in a loop
    //vxScheduleGraph(graph);
    //vxScheduleGraph(graph2);

    //vxWaitGraph(graph);
    //vxWaitGraph(graph2);
  
     sum += (getTickCount() - e) / getTickFrequency();
      //
            // Report performance
            //
#ifdef PRINT_REPORT
            vx_perf_t perf;

            
            vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) ;
            std::cout << "Graph Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

            vxQueryNode(pyrNode, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) ;
            std::cout << "\t bitnot Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

            vxQueryNode(node1, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) ;
            std::cout << "\t Gaussian Blur 1 Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

            vxQueryNode(node2, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) ;
            std::cout << "\t Gaussian Blur 2 Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

            vxQueryNode(node3, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) ;
            std::cout << "\t 3 Comp Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

            vxQueryNode(node4, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) ;
            std::cout << "\t 4 Comp Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;
#endif
  }

  cout <<"Vx works Gaussian "<<sum / iter<<endl;

 


  createMatFromImage(intermediate4,dstMat2);

  imwrite("RES_G.jpg",dstMat2);
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

    vx_rectangle_t rect;
    vxGetValidRegionImage(image, &rect);
    vx_imagepatch_addressing_t addr = {
        mat.cols, mat.rows, sizeof(vx_uint8), mat.cols * sizeof(vx_uint8), VX_SCALE_UNITY, VX_SCALE_UNITY, 1, 1 };

    status = vxAccessImagePatch(image, &rect, 0, &addr, (void **)&ptr, VX_READ_ONLY);
    mat.data = ptr;

    return status;
}