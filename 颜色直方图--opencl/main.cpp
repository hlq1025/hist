#include <stdlib.h>  

#include <string.h>  

#include <iostream>  

/* OpenCL includes */

#include <CL/cl.h>  

#include <opencv2/opencv.hpp>

#include <opencv2/highgui.hpp>

#include <opencv2/imgproc.hpp>

#include<fstream>


using namespace cv;

using namespace std;

static const int HIST_BINS = 256;
cl_int ConvertToString(const char *pFileName, std::string &Str)
{	size_t		uiSize = 0;
	size_t		uiFileSize = 0;
	char		*pStr = NULL;
	std::fstream fFile(pFileName, (std::fstream::in | std::fstream::binary));
	if (fFile.is_open())
{		fFile.seekg(0, std::fstream::end);
		uiSize = uiFileSize = (size_t)fFile.tellg();  // 获得文件大小
		fFile.seekg(0, std::fstream::beg);
		pStr = new char[uiSize + 1];
		if (NULL == pStr)
{
			fFile.close();
			return 0;
}







		fFile.read(pStr, uiFileSize);				// 读取uiFileSize字节



		fFile.close();



		pStr[uiSize] = '\0';



		Str = pStr;







		delete[] pStr;







		return 0;



	}







	cout << "Error: Failed to open cl file\n:" << pFileName << endl;







	return -1;



}

void check(cl_int status){

	if (status != CL_SUCCESS)

	{

		cout << "error:status is " << status << endl;

		return;

	}

}


int main()
{
	Mat img = imread("1.jpg");

	int imageRows = img.rows;

	int imageCols = img.cols;

	//CPU result,reconfirm the GPU result...

	int rgbhistogram[256 * 3];

	memset(rgbhistogram, 0, 256 * 3 * sizeof(int));

	for (int i = 0; i<imageRows; i++)

	{

		uchar *rowdatas = img.ptr<uchar>(i);

		for (int j = 0; j<imageCols * 3; j += 3)

		{

			int r = rowdatas[j + 2];

			int g = rowdatas[j + 1];

			int b = rowdatas[j];

			rgbhistogram[r * 3] += 1;

			rgbhistogram[g * 3 + 1] += 1;

			rgbhistogram[b * 3 + 2] += 1;

		}

	}

	const int imageElements = imageRows*imageCols;

	const size_t imageSize = imageElements*sizeof(unsigned char)* 3;

	const int histogramSize = HIST_BINS*sizeof(int);

	int *hOutputHistogramR = (int*)malloc(histogramSize);

	int *hOutputHistogramG = (int*)malloc(histogramSize);

	int *hOutputHistogramB = (int*)malloc(histogramSize);

	cl_int status;
	
	cl_uint nPlatform;
	
	clGetPlatformIDs(0, NULL, &nPlatform);
	
	cl_platform_id *listPlatform = (cl_platform_id*)malloc(nPlatform * sizeof(cl_platform_id));
	
	clGetPlatformIDs(nPlatform, listPlatform, NULL);

	cl_uint nDevice = 0;

	clGetDeviceIDs(listPlatform[0], CL_DEVICE_TYPE_GPU, 0, NULL, &nDevice);
	
	cl_device_id *device = (cl_device_id*)malloc(nDevice * sizeof(cl_device_id));

	clGetDeviceIDs(listPlatform[0], CL_DEVICE_TYPE_GPU, nDevice, device, NULL);

	cl_context context;

	context = clCreateContext(NULL, 1, device, NULL, NULL, &status);

	check(status);

	cl_command_queue cmdQueue = clCreateCommandQueue(context, device[0], 0, &status);

	check(status);

	cl_mem  bufInputImage = clCreateBuffer(context, CL_MEM_READ_ONLY, imageSize, NULL, &status);

	check(status);

	cl_mem bufOutputHistogramR = clCreateBuffer(context, CL_MEM_WRITE_ONLY, histogramSize, NULL, &status);

	cl_mem bufOutputHistogramG = clCreateBuffer(context, CL_MEM_WRITE_ONLY, histogramSize, NULL, &status);

	cl_mem bufOutputHistogramB = clCreateBuffer(context, CL_MEM_WRITE_ONLY, histogramSize, NULL, &status);

	check(status);

	status = clEnqueueWriteBuffer(cmdQueue, bufInputImage, CL_TRUE, 0, imageSize, img.data, 0, NULL, NULL);

	check(status);

	int zero = 0;

	status = clEnqueueFillBuffer(cmdQueue, bufOutputHistogramR, &zero, sizeof(int), 0, histogramSize, 0, NULL, NULL);

	status = clEnqueueFillBuffer(cmdQueue, bufOutputHistogramG, &zero, sizeof(int), 0, histogramSize, 0, NULL, NULL);

	status = clEnqueueFillBuffer(cmdQueue, bufOutputHistogramB, &zero, sizeof(int), 0, histogramSize, 0, NULL, NULL);

	//check(status);

	/* Create a program with source code */
	// 将cl文件中的代码转为字符串
	string strSource;
	
	const char *pSource;
	
	status = ConvertToString("kernel.cl", strSource);
	
	pSource = strSource.c_str();			// 获得strSource指针

	size_t uiArrSourceSize = strlen(pSource);	// 字符串大小

	// 创建程序对象

	cl_program program = clCreateProgramWithSource(context, 1, &pSource, &uiArrSourceSize, NULL);
	
	if (NULL == program)
	{
		cout << "Error: Can not create program" << endl;

		return 0;
	}

	// -----------------------------6. 编译程序--------------------------------

	// 编译程序

	status = clBuildProgram(program, 1, device, NULL, NULL, NULL);

	if (CL_SUCCESS != status)	// 编译错误
	{	cout << "Error: Can not build program" << endl;

		char szBuildLog[16384];

		clGetProgramBuildInfo(program, *device, CL_PROGRAM_BUILD_LOG, sizeof(szBuildLog), szBuildLog, NULL);

		cout << "Error in Kernel: " << endl << szBuildLog;

		clReleaseProgram(program);
		
		return 0;
	}

	cl_kernel kernel;

	kernel = clCreateKernel(program, "histogramforRGB", &status);

	check(status);

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufInputImage);

	status |= clSetKernelArg(kernel, 1, sizeof(int), &imageElements);

	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufOutputHistogramR);

	status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufOutputHistogramG);

	status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufOutputHistogramB);

	check(status);

	size_t globalWorkSize[1];

	globalWorkSize[0] = 1024;

	size_t localWorkSize[1];

	localWorkSize[0] = 64;

	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

	check(status);

	status = clEnqueueReadBuffer(cmdQueue, bufOutputHistogramR, CL_TRUE, 0, histogramSize, hOutputHistogramR, 0, NULL, NULL);

	status = clEnqueueReadBuffer(cmdQueue, bufOutputHistogramG, CL_TRUE, 0, histogramSize, hOutputHistogramG, 0, NULL, NULL);

	status = clEnqueueReadBuffer(cmdQueue, bufOutputHistogramB, CL_TRUE, 0, histogramSize, hOutputHistogramB, 0, NULL, NULL);

	check(status);

	for (int i = 0; i < HIST_BINS; i++) 
	{

		printf("histogram %d  OpenCL-result:R:%d G:%d B:%d---CPU-result:R:%d G:%d B:%d\n", i, hOutputHistogramR[i], hOutputHistogramG[i], hOutputHistogramB[i], rgbhistogram[i * 3], rgbhistogram[i * 3 + 1], rgbhistogram[i * 3 + 2]);

	}

	clReleaseKernel(kernel);

	clReleaseProgram(program);

	clReleaseCommandQueue(cmdQueue);

	clReleaseMemObject(bufInputImage);

	clReleaseMemObject(bufOutputHistogramR);

	clReleaseMemObject(bufOutputHistogramG);

	clReleaseMemObject(bufOutputHistogramB);

	clReleaseContext(context);

	free(hOutputHistogramR);

	free(hOutputHistogramG);

	free(hOutputHistogramB);

	//free(programSource);
	return 0;
}
