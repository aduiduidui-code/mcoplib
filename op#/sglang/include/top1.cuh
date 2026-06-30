// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

template<class scalar_t>
__device__ __forceinline__ scalar_t get_weight(const int32_t& v) {
    const scalar_t* idx_and_weight = (const scalar_t*)&v;
    return idx_and_weight[0];
}

template<class scalar_t, int OFFSET=16, uint32_t MASK=0xffffffff>
__device__ __forceinline__ void warp_top1(scalar_t (&idx_and_weight)[2], scalar_t (&max_idx_and_weight)[2], int tid) {
    int32_t temp_val = *(int32_t*)idx_and_weight;
    for (int offset = OFFSET; offset > 0; offset /= 2) {
        int32_t other_thread_temp_val = __shfl_xor_sync(MASK, temp_val, offset);
        if (get_weight<scalar_t>(other_thread_temp_val) > get_weight<scalar_t>(temp_val)) {
            temp_val = other_thread_temp_val;
        } else if (get_weight<scalar_t>(other_thread_temp_val) == get_weight<scalar_t>(temp_val) && (other_thread_temp_val >> 16) < (temp_val >> 16)) {
            //If the remote val equals to current val, we are prefering a smaller index
            temp_val = other_thread_temp_val;
        }
    }
    *(int32_t*)max_idx_and_weight = temp_val;
}

template<class scalar_t, int NUM_GROUPS = 8, int TOPK_GROUP=2, int TOPK=8>
__device__ __forceinline__ void moe_topk_block(const scalar_t* w, const scalar_t* bias, int tid, float* topk_w, int32_t* topk_indices) {
    scalar_t idx_and_weight[2]; //Stores weight and index for further top1
    scalar_t idx_and_weight_2[2]; //used for top8 calculation

    float fw = __bfloat162float(w[tid]);
    float fw_sim = 1.0f / (1.0f + expf(-fw));
    fw = fw_sim + __bfloat162float(bias[tid]);
    idx_and_weight[0] = __float2bfloat16(fw);
    idx_and_weight[1] = __float2bfloat16(0.0f);
    *(int32_t*)idx_and_weight |= (tid << 16);
    *(int32_t*)idx_and_weight_2 = *(int32_t*)idx_and_weight;

    //top2 is broadcasted to all groups
    //~ 240 cycles
    //top2 records max value and the second max value of a group
    scalar_t top2[2][2];
    for (int i = 0; i < 2; i++) {
        warp_top1<scalar_t>(idx_and_weight, top2[i], tid);
        if (tid == *((int32_t*)top2[i]) >> 16) {
            //reset idx_and_weight if thread data is selected
            //so we can select the second max value
            idx_and_weight[0] = __float2bfloat16(0.0f);
        }
    }
    // if (tid % 32 == 0) {
    //     printf("Group %d top2 is [%d, %f], [%d, %f]\n", tid /32, *(int32_t*)top2[0] >> 16, __bfloat162float(top2[0][0]), *(int32_t*)top2[1] >> 16, __bfloat162float(top2[1][0]));
    // }

    //~ 70 cycles
    scalar_t group_weight[2];
    group_weight[0] = top2[0][0] + top2[1][0];
    group_weight[1] = __float2bfloat16(0.0f); //reset the high 16 bits
    *(int32_t*)group_weight |= (tid / 32) << 16;

    //Eight groups
    __shared__ scalar_t shared_group_weights[8][2];
    //group weight are distributed in different groups, we need a shared memory to broadcast to all threads
    //save group_weight to shared memory
    if (tid % 32 == 0) {
        int group_idx = tid / 32;
        *((int32_t*)(shared_group_weights) + group_idx) = *(int32_t*)group_weight;
    }
    __syncthreads();

    //get top 4
    // ~ 200 cycles
    scalar_t group_weight_for_sort[2] = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};
    // All top4 weights are stored locally
    scalar_t top4_group_weight[4][2];
    //Move all the group weights into one warp so that we can do top4
    if (tid < 8) {
        *(int32_t*)group_weight_for_sort = *((int32_t*)(shared_group_weights) + tid);
        // printf("group = %d, group_weight = %f\n", *(int32_t*)group_weight_for_sort >> 16, __bfloat162float(group_weight_for_sort[0]));
    }
    //Doing top4, be attention to offset and mask
    for (int i = 0; i < 4; i++) {
        warp_top1<scalar_t, 4, 0x000000ff>(group_weight_for_sort, top4_group_weight[i], tid);
        // int32_t temp_val = *(int32_t*)group_weight_for_sort;
        // for (int offset = 4; offset > 0; offset /= 2) {
        //     int32_t other_temp_val = __shfl_xor_sync(0x000000ff, temp_val, offset);
        //     if (tid < 8) {
        //         printf("Before exchange tid = %d, temp_val = [%d, %f] other_temp_val = [%d, %f]\n", tid, temp_val >> 16, __bfloat162float(get_weight<scalar_t>(temp_val)), other_temp_val >> 16, __bfloat162float(get_weight<scalar_t>(other_temp_val)));
        //     }
        //     if (get_weight<scalar_t>(other_temp_val) > get_weight<scalar_t>(temp_val)) {
        //         temp_val = other_temp_val;
        //     } else if (get_weight<scalar_t>(other_temp_val) == get_weight<scalar_t>(temp_val) && (other_temp_val >> 16) < (temp_val >> 16)) {
        //         temp_val = other_temp_val;
        //     }
        //     if (tid < 8) {
        //         printf("After exchange tid = %d, temp_val = [%d, %f] other_temp_val = [%d, %f]\n", tid, temp_val >> 16, __bfloat162float(get_weight<scalar_t>(temp_val)), other_temp_val >> 16, __bfloat162float(get_weight<scalar_t>(other_temp_val)));
        //     }
        // }
        // *(int32_t*)top4_group_weight[i] = temp_val;
        //if (tid < 4) {
            // printf("II tid = %d, find top %d group idx %d, w %f\n", tid, i, *(int32_t*)top4_group_weight[i] >> 16, __bfloat162float(top4_group_weight[i][0]));
        //}
        if (tid == *(int32_t*)top4_group_weight[i] >> 16) {
            // printf("tid = %d, find top %d group idx %d, w %f\n", tid, i, *(int32_t*)top4_group_weight[i] >> 16, __bfloat162float(top4_group_weight[i][0]));
            group_weight_for_sort[0] = __float2bfloat16(0.0f);
        }
    }
    //if (tid == 0) {
        // if (tid < 4)
        //     printf("shlfed disabled group = %d, group_weight = %f\n", *(int32_t*)top4_group_weight[tid] >> 16, __bfloat162float(top4_group_weight[tid][0]));
    //}
    // broadcast disable flags
    // ~ 60 cycles
    __shared__ bool shared_flags[8];
    if (tid < 8) {
        shared_flags[tid] = true;
    }

    if (tid < 4) {
        shared_flags[*(int32_t*)top4_group_weight[tid] >> 16] = false;
    }
    __syncthreads();

    //~ 300 cycles

    bool group_disabled = shared_flags[tid / 32];
    // if (tid % 32 == 0) {
    //     printf("group %d, disable flag = %d\n", tid / 32, group_disabled);
    // }

    //topk 8
    //The max value of each group is already stored in top2, so we only need to compare 4 max value of activated groups and we can get a global max value
    __shared__ scalar_t shared_max_experts[8][2];
    //Firstly, put all max values into shared_group_weights
    if (tid % 32 == 0) {
        if (group_disabled) top2[0][0] = __float2bfloat16(0.0f);
        *((int32_t*)(shared_group_weights) + tid / 32) = *(int32_t*)top2[0];
    }
    __syncthreads();
    if (tid < 8) {
        *(int32_t*)top2[0] = *((int32_t*)shared_group_weights + tid);
    }
    //Sort eight values and find the maximum one
    warp_top1<scalar_t, 4, 0x000000ff>(top2[0], top2[0], tid);
    if (tid == 0) {
        *(int32_t*)shared_max_experts[0] = *(int32_t*)top2[0];
    }
    __syncthreads();
    //Note that currently, the maximum values are stored only in warp 0, other group needs to visit shared memory to get current max value
    //FIXME: Maybe we can do this procedure in all warps so that there is no need to transfer data from shared memory
    if (tid == *(int32_t*)shared_max_experts[0] >> 16) {
        //disable max value
        idx_and_weight_2[0] = __float2bfloat16(0.0f);
    }
    // if (tid == 0) {
    //     printf("max 0 is [%d, %f]\n", *(int32_t*)top2[0] >> 16, __bfloat162float(top2[0][0]));
    // }

    //Next top 7
    //~ 1000 cycles * 7
    //Other top7 procedure process almost likely with previous global top1
    for (int i = 1; i < 8; i++) {
        //Get top one for each group
        warp_top1<scalar_t>(idx_and_weight_2, top2[0], tid);
        if (tid % 32 == 0) {
            if (group_disabled) top2[0][0] = __float2bfloat16(0.0f);
            *((int32_t*)(shared_group_weights) + tid / 32) = *(int32_t*)top2[0];
        }
        __syncthreads();
        if (tid < 8) {
            *(int32_t*)top2[0] = *((int32_t*)shared_group_weights + tid);
            // printf("Participates %dth max group %d, [%d, %f]\n", i, tid, *(int32_t*)top2[0] >> 16, __bfloat162float(top2[0][0]));
        }
        //Sort eight values and find the maximum one
        warp_top1<scalar_t, 4, 0x000000ff>(top2[0], top2[0], tid);
        if (tid == 0) {
            *(int32_t*)shared_max_experts[i] = *(int32_t*)top2[0];
        }
        __syncthreads();
        if (tid == *(int32_t*)shared_max_experts[i] >> 16) {
            //disable max value
            idx_and_weight_2[0] = __float2bfloat16(0.0f);
        }

        // if (tid == 0) {
        //     printf("max %d is in group %d [%d, %f]\n", i, (*(int32_t*)top2[0] >> 16) / 32, *(int32_t*)top2[0] >> 16, __bfloat162float(top2[0][0]));
        // }
    }

    __syncthreads();
    float ww, sum;

    //Now all values are stored into shared_max_experts
    if (tid < 8) {
        topk_indices[tid] = ((int32_t*)shared_max_experts)[tid] >> 16;
        //Calculate sum
        ww = __bfloat162float(w[topk_indices[tid]]);
        ww = 1.0f / (1.0f + expf(-ww));
        // ww = __bfloat162float(shared_max_experts[tid][0]);
        sum = ww;
        // printf("topk_indices[%d] = %d\n", tid, topk_indices[tid]);
    }

    // ~ 400 cycles
    for (int i = 4; i > 0; i /= 2) {
        sum += __shfl_xor_sync(0x000000ff, sum, i);
    }
    float norm_weight = ww / sum;
    // if (tid == 0) {
    //     printf("sum = %f, norm_weight = %f\n", sum, norm_weight);
    // }
    if (tid < 8) {
        topk_w[tid] = norm_weight;
    }
}