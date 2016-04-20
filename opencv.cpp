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


int main()
{
	Mat m = imread("br1.png");
	Stream str;

	GpuMat d_m  = GpuMat (m);
	GpuMat d_m2;
	GpuMat l1,l2,l3,l4;
	int iter = 100;
	int64 e = getTickCount();
	float sum = 0;
	for(int i = 0 ; i < iter ; i ++)
	{
		e = getTickCount();
		gpu::cvtColor(d_m,d_m2,CV_BGR2GRAY);
		sum+= (getTickCount() - e) / getTickFrequency();
	}

	cout <<"Time taken by cvtColor \t\t\t"<<sum/iter<<" sec"<<endl;

	sum = 0;
	for(int i = 0 ; i < iter;  i++)
	{
		e = getTickCount();
		str.enqueueConvert(d_m,d_m2,CV_16U);
		str.waitForCompletion();
		sum+= (getTickCount() - e) / getTickFrequency(); 
	}

	cout <<"Time taken by ConverType \t\t\t"<<sum/iter<<" sec"<<endl;


	sum = 0;
	
	for(int i = 0 ; i < iter;  i++)
	{
		e = getTickCount();
		gpu::pyrDown(d_m,l1);
		gpu::pyrDown(l1,l2);
		gpu::pyrDown(l2,l3);
		gpu::pyrDown(l3,l4);
		sum+= (getTickCount() - e) / getTickFrequency(); 
	}

	cout <<"Time taken by Gussian Pyramid Level 4 \t\t\t"<<sum/iter<<" sec"<<endl;



	sum = 0;
	for(int i = 0 ; i < iter;  i++)
	{
		e = getTickCount();
		gpu::Sobel(d_m,l1,CV_16U,1,1);
		
		sum+= (getTickCount() - e) / getTickFrequency(); 
	}

	cout <<"Time taken by Sobel \t\t\t"<<sum/iter<<" sec"<<endl;


	sum = 0;
	for(int i = 0 ; i < iter;  i++)
	{
		e = getTickCount();
	
		//gpu::resize(d_m,d_m2,Size(640,360));
		gpu::pyrDown(d_m,d_m2);

	sum+= (getTickCount() - e) / getTickFrequency(); 
	}

	cout <<"Time taken by single pyrdown \t\t\t"<<sum/iter<<" sec"<<endl;



	/* CUDA MEM based gaussian pyramid */
	int width = 1280;
	int height = 720;
	CudaMem cudamem_src(d_m.size(), CV_8UC3, cv::gpu::CudaMem::ALLOC_ZEROCOPY);
	CudaMem cudamem_l1(Size(width/2, height/2), CV_8UC3, cv::gpu::CudaMem::ALLOC_ZEROCOPY);
	CudaMem cudamem_l2(Size(width/4, height/4), CV_8UC3, cv::gpu::CudaMem::ALLOC_ZEROCOPY);
	CudaMem cudamem_l3(Size(width/8, height/8), CV_8UC3, cv::gpu::CudaMem::ALLOC_ZEROCOPY);
	CudaMem cudamem_l4(Size(width/16, height/16), CV_8UC3, cv::gpu::CudaMem::ALLOC_ZEROCOPY);

	GpuMat gpu_src = cudamem_src.createGpuMatHeader();
	GpuMat gpu_l1 = cudamem_l1.createGpuMatHeader();
	GpuMat gpu_l2 = cudamem_l2.createGpuMatHeader();
	GpuMat gpu_l3 = cudamem_l3.createGpuMatHeader();
	GpuMat gpu_l4 = cudamem_l4.createGpuMatHeader();

    Mat dst = cudamem_src.createMatHeader();

    sum = 0;
	
	for(int i = 0 ; i < iter;  i++)
	{
		e = getTickCount();
		gpu::pyrDown(gpu_src,gpu_l1);
		gpu::pyrDown(gpu_l1,gpu_l2);
		gpu::pyrDown(gpu_l2,gpu_l3);
		gpu::pyrDown(gpu_l3,gpu_l4);
		sum+= (getTickCount() - e) / getTickFrequency(); 
	}

	cout <<"Time taken by CUDA mem Pyramid Level 4 \t\t\t"<<sum/iter<<" sec"<<endl;



	sum = 0;
	iter = 100;
	Mat ml1,ml2,ml3,ml4;
	for(int i = 0 ; i < iter;  i++)
	{
		e = getTickCount();
		pyrDown(m,ml1);
		pyrDown(ml1,ml2);
		pyrDown(ml2,ml3);
		pyrDown(ml3,ml4);
		sum+= (getTickCount() - e) / getTickFrequency(); 
	}

	cout <<"Time taken by CPU Pyramid Level 4 \t\t\t"<<sum/iter<<" sec"<<endl;

	/* CUDA mem ends */


	/* Multiplication Test */

	//d_m
	d_m2 = GpuMat(d_m);
	GpuMat mul1;

	iter = 100;
	sum = 0;
	for (int i = 0; i < iter; ++i)
	{
		e = getTickCount();
		/* code */
		gpu::multiply(d_m,d_m2, mul1,1,CV_16U);
		sum+= (getTickCount() - e) / getTickFrequency(); 

	}
	cout <<"Time taken by GPU multiply 8bx8b->16b\t"<<sum/iter<<" sec"<<endl;


	GpuMat d_m1_16;
	GpuMat d_m2_16;
	GpuMat mul1_16;

	Mat intermediate = Mat(m.size() , CV_16UC3);
	m.convertTo(intermediate,CV_16U);

	d_m1_16= GpuMat(intermediate);
	d_m2_16 = GpuMat(intermediate);

	iter = 100;
	sum = 0;
	for (int i = 0; i < iter; ++i)
	{
		e = getTickCount();
		/* code */
		gpu::multiply(d_m1_16,d_m2_16, mul1_16,1,CV_16U);
		sum+= (getTickCount() - e) / getTickFrequency(); 

	}
	cout <<"Time taken by GPU multiply 16bx16b->16b\t"<<sum/iter<<" sec"<<endl;


	Mat m_mul1;

	sum = 0;
	for(int i = 0 ; i < iter; i++)
	{
		e = getTickCount();
		/* code */
		multiply(m,m, m_mul1,1,CV_16U);
		sum+= (getTickCount() - e) / getTickFrequency(); 
	}
	
	cout <<"Time taken by CPU multiply \t"<<sum/iter<<" sec"<<endl;



	iter = 100;
	sum = 0;
	for (int i = 0; i < iter; ++i)
	{
		e = getTickCount();
		/* code */
		gpu::multiply(d_m,d_m2, mul1);
		sum+= (getTickCount() - e) / getTickFrequency(); 

	}
	cout <<"Time taken by GPU subtract 8bx8b->8b\t"<<sum/iter<<" sec"<<endl;

	iter = 100;
	sum = 0;
	for (int i = 0; i < iter; ++i)
	{
		e = getTickCount();
		/* code */
		gpu::subtract(d_m1_16,d_m2_16, mul1_16);
		sum+= (getTickCount() - e) / getTickFrequency(); 

	}
	cout <<"Time taken by GPU subtract 16bx16b->16b\t"<<sum/iter<<" sec"<<endl;


	sum = 0;
	for(int i = 0 ; i < iter; i++)
	{
		e = getTickCount();
		/* code */
		subtract(m,m, m_mul1);
		sum+= (getTickCount() - e) / getTickFrequency(); 
	}
	
	cout <<"Time taken by CPU subtract 8b \t"<<sum/iter<<" sec"<<endl;


	sum = 0;
	for(int i = 0 ; i < iter; i++)
	{
		e = getTickCount();
		/* code */
		subtract(intermediate,intermediate, m_mul1);
		sum+= (getTickCount() - e) / getTickFrequency(); 
	}
	
	cout <<"Time taken by CPU subtract 16b \t"<<sum/iter<<" sec"<<endl;
	

	CudaMem cm (1280, 720, CV_8UC3);
	Mat mat_hdr = cm;
	m.copyTo(mat_hdr);
	sum = 0;
	for(int i = 0 ; i < iter; i++)
	{
		e = getTickCount();
		multiply(mat_hdr,mat_hdr,m_mul1);
		sum+= (getTickCount() - e) / getTickFrequency(); 
	}

	cout <<"Time taken by cudamem multiply 8b \t"<<sum/iter<<" sec"<<endl;

	CudaMem cm2 (1280, 720, CV_16UC3);
	Mat mat_hdr2 = cm2;
	intermediate.copyTo(mat_hdr2);
	sum = 0;
	for(int i = 0 ; i < iter; i++)
	{
		e = getTickCount();
		multiply(mat_hdr2,mat_hdr2,m_mul1);
		sum+= (getTickCount() - e) / getTickFrequency(); 
	}

	cout <<"Time taken by cudamem multiply 16b \t"<<sum/iter<<" sec"<<endl;

	GpuMat gpgp1 = GpuMat(mat_hdr2);

	Mat res ; 
	mul1.download(res);
	imwrite("cv_res.jpg",res);
	return 0;
}