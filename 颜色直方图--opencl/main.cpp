// main.c
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <time.h>

#define SIZE_OF_BIN 256

char *sourceText = "kernel.cl";

int readText(const char* kernelPath, char **pcode)// 读取文本文件放入 pcode，返回字符串长度
{
	FILE *fp;
	int size;
	//printf("<readText> File: %s\n", kernelPath);
	fopen_s(&fp, kernelPath, "rb");
	if (!fp)
	{
		printf("<readText> Open file failed\n");
		getchar();
		exit(-1);
	}
	if (fseek(fp, 0, SEEK_END) != 0)
	{
		printf("<readText> Seek end of file failed\n");
		getchar();
		exit(-1);
	}
	if ((size = ftell(fp)) < 0)
	{
		printf("<readText> Get file position failed\n");
		getchar();
		exit(-1);
	}
	rewind(fp);
	if ((*pcode = (char *)malloc(size + 1)) == NULL)
	{
		printf("<readText> Allocate space failed\n");
		getchar();
		exit(-1);
	}
	fread(*pcode, 1, size, fp);
	(*pcode)[size] = '\0';
	fclose(fp);
	return size + 1;
}

int main(int argc, char * argv[])
{
	//_putenv("GPU_DUMP_DEVICE_KERNEL=3");// 在程序目录输出出 il 和 isa 形式的 kernel 文件，可以使用 isa 汇编调试

	cl_int status;
	cl_uint nPlatform;
	clGetPlatformIDs(0, NULL, &nPlatform);
	cl_platform_id *listPlatform = (cl_platform_id*)malloc(nPlatform * sizeof(cl_platform_id));
	clGetPlatformIDs(nPlatform, listPlatform, NULL);
	cl_uint nDevice = 0;
	clGetDeviceIDs(listPlatform[0], CL_DEVICE_TYPE_ALL, 0, NULL, &nDevice);
	cl_device_id *listDevice = (cl_device_id*)malloc(nDevice * sizeof(cl_device_id));
	clGetDeviceIDs(listPlatform[0], CL_DEVICE_TYPE_ALL, nDevice, listDevice, NULL);
	cl_context context = clCreateContext(NULL, nDevice, listDevice, NULL, NULL, &status);
	cl_command_queue queue = clCreateCommandQueue(context, listDevice[0], 0, &status);

	char *code;
	size_t length = readText(sourceText, &code);
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&code, &length, NULL);
	status = clBuildProgram(program, 1, listDevice, NULL, NULL, NULL);
	cl_kernel histogram = clCreateKernel(program, "histogramKernel", &status);
	cl_kernel reduce = clCreateKernel(program, "reduceKernel", &status);

	cl_uint nThreads = 1024;                                        // 64 * 1024
	cl_uint nThreadsPerGroup = 256;                                 // 原始值：KernelCompileWorkGroupSize[0]
	cl_uint nGroups = nThreads / nThreadsPerGroup;

	cl_uint inputByte = 2147483648;                                 // 原始值：DeviceMaxMemAllocSize == 2147483648 == 2^31
	cl_uint outputNBytes = nGroups * SIZE_OF_BIN * sizeof(cl_uint);
	cl_uint element = inputByte / sizeof(cl_uint);
	cl_uint element4 = inputByte / sizeof(cl_uint4);
	cl_uint element4PerThread = element4 / nThreads;

	unsigned int *input = (unsigned int*)malloc(inputByte);
	unsigned int *cpuhist = (unsigned int*)malloc(outputNBytes);
	unsigned int *gpuhist = (unsigned int*)malloc(outputNBytes);
	memset(input, 0, inputByte);
	memset(cpuhist, 0, outputNBytes);
	memset(gpuhist, 0, outputNBytes);

	int i;
	time_t ltime;
	cl_uint a, b;
	time(&ltime);
	for (i = 0, a = b = (cl_uint)ltime; i < element; i++)// b 的低 16 位乘 a 再加上 b 的高 16 位，赋给新的 b
		input[i] = (b = (a * (b & 65535)) + (b >> 16));

	cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputByte, input, &status);
	cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, outputNBytes, gpuhist, &status);

	clSetKernelArg(histogram, 0, sizeof(cl_mem), (void *)&inputBuffer);
	clSetKernelArg(histogram, 1, sizeof(cl_mem), (void *)&outputBuffer);
	clSetKernelArg(histogram, 2, sizeof(cl_uint), (void *)&element4PerThread);
	clSetKernelArg(reduce, 0, sizeof(cl_mem), (void *)&outputBuffer);
	clSetKernelArg(reduce, 1, sizeof(cl_uint), (void *)&nGroups);

	size_t globalSizeHist = nThreads, localSizeHist = nThreadsPerGroup;
	size_t globalSizeReduce = SIZE_OF_BIN, localSizeReduce = 64;

	clEnqueueNDRangeKernel(queue, histogram, 1, NULL, &globalSizeHist, &localSizeHist, 0, NULL, NULL);
	clEnqueueNDRangeKernel(queue, reduce, 1, NULL, &globalSizeReduce, &localSizeReduce, 0, NULL, NULL);
	clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, outputNBytes / nGroups, gpuhist, 0, NULL, NULL);

	for (i = 0, input; i < element; i++)  // 使用 CPU 计算，注意每 8 位看做一个元素，取值范围 0 ~ 255
	{
		cpuhist[(input[i] >> 24) & 0xff]++;
		cpuhist[(input[i] >> 16) & 0xff]++;
		cpuhist[(input[i] >> 8) & 0xff]++;
		cpuhist[(input[i] >> 0) & 0xff]++;
	}

	int countError;
	for (i = countError = 0; i < SIZE_OF_BIN; i++)// 检查结果
	{
		if (gpuhist[i] != cpuhist[i])
			countError++;
	}
	printf("\n<main> CPU, GPU %s.\n\n\n", countError ? "mismatched" : "matched");

	free(listPlatform);
	free(listDevice);
	free(input);
	free(cpuhist);
	free(gpuhist);
	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseKernel(histogram);
	clReleaseKernel(reduce);
	clReleaseMemObject(inputBuffer);
	clReleaseMemObject(outputBuffer);
	getchar();
	return 0;
}