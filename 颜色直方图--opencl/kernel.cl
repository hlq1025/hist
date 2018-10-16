#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#define HIST_BINS 256  

__kernel void histogramforRGB(__global unsigned char *data,int  numData, __global int *histogramR, __global int *histogramG, __global int *histogramB){  

  
__local int localHistorgramR[HIST_BINS];  

 __local int localHistorgramG[HIST_BINS];  

 __local int localHistorgramB[HIST_BINS];  

  int lid = get_local_id(0);  

 for (int i = lid; i < HIST_BINS; i += get_local_size(0)){  

    localHistorgramR[i] = 0;  

    localHistorgramG[i] = 0;  

    localHistorgramB[i] = 0;  

  }  

  barrier(CLK_LOCAL_MEM_FENCE);  

  

  int gid = get_global_id(0);  

  for (int i = gid; i < numData*3; i += get_global_size(0)){  

    if(i%3==0)  

    {  

        atomic_add(&(localHistorgramB[data[i]]), 1);  

        //continue; 网友问的不加continue时其实也一样  

    }  

    if(i%3==1)  

    {  

        atomic_add(&(localHistorgramG[data[i]]), 1);  

        //continue;  

    }  

    if(i%3==2)  

    {  

        atomic_add(&(localHistorgramR[data[i]]), 1);  

        //continue;  

    }  

  }  

  barrier(CLK_LOCAL_MEM_FENCE);  

  

  for (int i = lid; i < HIST_BINS; i += get_local_size(0)){  

    atomic_add(&(histogramR[i]), localHistorgramR[i]);  

    atomic_add(&(histogramG[i]), localHistorgramG[i]);  

    atomic_add(&(histogramB[i]), localHistorgramB[i]);  

  }  

}  
