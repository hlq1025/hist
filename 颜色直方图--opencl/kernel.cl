// kernel.cl
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#define N_BANK          16              // 缓存 bank 数
#define SIZE_OF_BIN     256             // 256 个桶
#define BIT_OF_ELEMENT  8               // 单元素为 8 位，取值范围 0 ~ 255
#define MIN(a,b) ((a) < (b)) ? (a) : (b)
#define MAX(a,b) ((a) > (b)) ? (a) : (b)

__kernel __attribute__((reqd_work_group_size(SIZE_OF_BIN, 1, 1)))                               // 工作组尺寸对齐到桶的个数的整数倍
void histogramKernel(__global uint4 *input, __global uint  *Histogram, uint  element4PerThread)
{
    __local uint subhists[N_BANK * SIZE_OF_BIN];
    const uint gid = get_global_id(0), lid = get_local_id(0), stride = get_global_size(0);
    uint i, idx, lmem_items = N_BANK * SIZE_OF_BIN, lmem_items_per_thread, lmem_max_threads, bin;
    uint4 temp, temp2;

    lmem_max_threads = MAX(1, get_local_size(0) / lmem_items);  // 计算局部内存中每个工作项对应的线程数，至少为 1，后面几行没看懂，不改                                                              
    lmem_max_threads = MAX(1, lmem_max_threads / lmem_items);   // but no more than we have items    
    lmem_max_threads = lmem_items / lmem_max_threads;           // calculate threads total  
    lmem_max_threads = MIN(get_local_size(0), lmem_max_threads);// but no more than LDS banks
    lmem_items_per_thread = lmem_items / lmem_max_threads;

    if (lid < lmem_max_threads)// 初始化桶
        for (i = 0, idx = lid; i < lmem_items_per_thread / 4; subhists[idx] = 0, i++, idx += lmem_max_threads);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (i = 0, idx = gid; i < element4PerThread; i++, idx += stride)// 原子计数部分
    {
        temp = input[idx];
        temp2 = (temp & (SIZE_OF_BIN - 1)) * N_BANK + lid % N_BANK;// 取 input[idx] 的低 8 位（4个 0 ~ 255 的数），对准缓存行位置

        (void)atom_inc(subhists + temp2.x);
        (void)atom_inc(subhists + temp2.y);
        (void)atom_inc(subhists + temp2.z);
        (void)atom_inc(subhists + temp2.w);

        temp = temp >> BIT_OF_ELEMENT;
        temp2 = (temp & (SIZE_OF_BIN - 1)) * N_BANK + lid % N_BANK;// 取 input[idx] 的次低 8 位

        (void)atom_inc(subhists + temp2.x);
        (void)atom_inc(subhists + temp2.y);
        (void)atom_inc(subhists + temp2.z);
        (void)atom_inc(subhists + temp2.w);

        temp = temp >> BIT_OF_ELEMENT;
        temp2 = (temp & (SIZE_OF_BIN - 1)) * N_BANK + lid % N_BANK;

        (void)atom_inc(subhists + temp2.x);
        (void)atom_inc(subhists + temp2.y);
        (void)atom_inc(subhists + temp2.z);
        (void)atom_inc(subhists + temp2.w);

        temp = temp >> BIT_OF_ELEMENT;
        temp2 = (temp & (SIZE_OF_BIN - 1)) * N_BANK + lid % N_BANK;

        (void)atom_inc(subhists + temp2.x);
        (void)atom_inc(subhists + temp2.y);
        (void)atom_inc(subhists + temp2.z);
        (void)atom_inc(subhists + temp2.w);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < SIZE_OF_BIN)// 规约部分
    {
        for (i = 0, bin = 0; i<N_BANK; bin += subhists[(lid * N_BANK) + i], i++);// 每个线程负责连续的 N_BANK 个数据，加到一起写入 Histogram            
        Histogram[(get_group_id(0) * SIZE_OF_BIN) + lid] = bin;
    }
}

__kernel void reduceKernel(__global uint *Histogram, uint nSubHists)
{
    const uint gid = get_global_id(0);
    int i;
    uint bin;
    for (i = 0, bin = 0; i < nSubHists; bin += Histogram[(i * SIZE_OF_BIN) + gid], i++);// 每个工作项负责间隔为 SIZE_OF_BIN 的项进行求和
    Histogram[gid] = bin;                                                               // 结果放入原直方图的头部
}