// kernel.cl
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#define N_BANK          16              // ���� bank ��
#define SIZE_OF_BIN     256             // 256 ��Ͱ
#define BIT_OF_ELEMENT  8               // ��Ԫ��Ϊ 8 λ��ȡֵ��Χ 0 ~ 255
#define MIN(a,b) ((a) < (b)) ? (a) : (b)
#define MAX(a,b) ((a) > (b)) ? (a) : (b)

__kernel __attribute__((reqd_work_group_size(SIZE_OF_BIN, 1, 1)))                               // ������ߴ���뵽Ͱ�ĸ�����������
void histogramKernel(__global uint4 *input, __global uint  *Histogram, uint  element4PerThread)
{
    __local uint subhists[N_BANK * SIZE_OF_BIN];
    const uint gid = get_global_id(0), lid = get_local_id(0), stride = get_global_size(0);
    uint i, idx, lmem_items = N_BANK * SIZE_OF_BIN, lmem_items_per_thread, lmem_max_threads, bin;
    uint4 temp, temp2;

    lmem_max_threads = MAX(1, get_local_size(0) / lmem_items);  // ����ֲ��ڴ���ÿ���������Ӧ���߳���������Ϊ 1�����漸��û����������                                                              
    lmem_max_threads = MAX(1, lmem_max_threads / lmem_items);   // but no more than we have items    
    lmem_max_threads = lmem_items / lmem_max_threads;           // calculate threads total  
    lmem_max_threads = MIN(get_local_size(0), lmem_max_threads);// but no more than LDS banks
    lmem_items_per_thread = lmem_items / lmem_max_threads;

    if (lid < lmem_max_threads)// ��ʼ��Ͱ
        for (i = 0, idx = lid; i < lmem_items_per_thread / 4; subhists[idx] = 0, i++, idx += lmem_max_threads);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (i = 0, idx = gid; i < element4PerThread; i++, idx += stride)// ԭ�Ӽ�������
    {
        temp = input[idx];
        temp2 = (temp & (SIZE_OF_BIN - 1)) * N_BANK + lid % N_BANK;// ȡ input[idx] �ĵ� 8 λ��4�� 0 ~ 255 ����������׼������λ��

        (void)atom_inc(subhists + temp2.x);
        (void)atom_inc(subhists + temp2.y);
        (void)atom_inc(subhists + temp2.z);
        (void)atom_inc(subhists + temp2.w);

        temp = temp >> BIT_OF_ELEMENT;
        temp2 = (temp & (SIZE_OF_BIN - 1)) * N_BANK + lid % N_BANK;// ȡ input[idx] �Ĵε� 8 λ

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

    if (lid < SIZE_OF_BIN)// ��Լ����
    {
        for (i = 0, bin = 0; i<N_BANK; bin += subhists[(lid * N_BANK) + i], i++);// ÿ���̸߳��������� N_BANK �����ݣ��ӵ�һ��д�� Histogram            
        Histogram[(get_group_id(0) * SIZE_OF_BIN) + lid] = bin;
    }
}

__kernel void reduceKernel(__global uint *Histogram, uint nSubHists)
{
    const uint gid = get_global_id(0);
    int i;
    uint bin;
    for (i = 0, bin = 0; i < nSubHists; bin += Histogram[(i * SIZE_OF_BIN) + gid], i++);// ÿ�����������Ϊ SIZE_OF_BIN ����������
    Histogram[gid] = bin;                                                               // �������ԭֱ��ͼ��ͷ��
}