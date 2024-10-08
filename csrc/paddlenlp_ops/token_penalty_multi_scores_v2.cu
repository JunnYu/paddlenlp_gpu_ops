// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "helper.h"


template<typename T>
__global__ inline void min_length_logits_process_v2(T* logits,
                                                 const int64_t *cur_len,
                                                 const int64_t *min_len,
                                                 const int64_t *eos_token_id,
                                                 const int64_t bs,
                                                 const int64_t length,
                                                 const int64_t end_length) {
    int bi = threadIdx.x;
    if (bi >= bs) return;
    if (cur_len[bi] < 0) {
        return;
    }
    if (cur_len[bi] < min_len[bi]) {
        for (int i=0; i < end_length; i++) {
            logits[bi * length + eos_token_id[i]] = -1e10;
        }
    }
}

template<>
__global__ inline void min_length_logits_process_v2<half>(half* logits,
                                                       const int64_t *cur_len,
                                                       const int64_t *min_len,
                                                       const int64_t *eos_token_id,
                                                       const int64_t bs,
                                                       const int64_t length,
                                                       const int64_t end_length) {
    int bi = threadIdx.x;
    if (bi >= bs) return;
    if (cur_len[bi] < 0) {
        return;
    }
    if (cur_len[bi] < min_len[bi]) {
        for (int i=0; i < end_length; i++) {
            logits[bi * length + eos_token_id[i]] = -1e4;
        }
    }
}


__global__ void update_repeat_times_v2(const int64_t *pre_ids,
                                    const int64_t *cur_len,
                                    int *repeat_times,
                                    const int64_t bs,
                                    const int64_t length,
                                    const int64_t length_id) {
    int bi = blockIdx.x;
    if (cur_len[bi] < 0) {
        return;
    }
    int tid = threadIdx.x;
    const int64_t *pre_ids_now = pre_ids + bi * length_id;
    int *repeat_times_now = repeat_times + bi * length;
    for (int i = tid; i < length_id; i += blockDim.x) {
        int64_t id = pre_ids_now[i];
        if (id < 0) break;
        atomicAdd(&repeat_times_now[id], 1);
    }
}

template<typename T>
__global__ void update_value_by_repeat_times_v2(const int *repeat_times,
                                             const T *penalty_scores,
                                             const T *frequency_score,
                                             const T *presence_score,
                                             const float *temperatures,
                                             T *logits,
                                             const int64_t bs,
                                             const int64_t length) {
    int bi = blockIdx.x;
    int tid = threadIdx.x;
    T *logits_now = logits + bi * length;
    const int *repeat_times_now = repeat_times + bi * length;
    float alpha = static_cast<float>(penalty_scores[bi]);
    float beta = static_cast<float>(frequency_score[bi]);
    float gamma = static_cast<float>(presence_score[bi]);
    for (int i = tid; i < length; i += blockDim.x) {
        int times = repeat_times_now[i];
        float logit_now = static_cast<float>(logits_now[i]);
        if (times != 0) {
            logit_now = logit_now < 0 ? logit_now * alpha : logit_now / alpha;
            logit_now = logit_now - times * beta - gamma;
        }
        logits_now[i] = static_cast<T>(logit_now / temperatures[bi]);
    }
}

template<typename T>
__global__ void ban_bad_words(T *logits,
                              const int64_t *bad_words_list,
                              const int64_t bs,
                              const int64_t length,
                              const int64_t bad_words_length) {
    const int bi = blockIdx.x;
    int tid = threadIdx.x;
    T *logits_now = logits + bi * length;
    for (int i = tid; i < bad_words_length; i += blockDim.x) {
        const int64_t bad_words_token_id = bad_words_list[i];
        if (bad_words_token_id >= length || bad_words_token_id < 0) continue;
        logits_now[bad_words_token_id] = -1e10;
    }
}

template <paddle::DataType D>
void token_penalty_multi_scores_kernel_v2(const paddle::Tensor& pre_ids,
                                       const paddle::Tensor& logits,
                                       const paddle::Tensor& penalty_scores,
                                       const paddle::Tensor& frequency_score,
                                       const paddle::Tensor& presence_score,
                                       const paddle::Tensor& temperatures,
                                       const paddle::Tensor& bad_tokens,
                                       const paddle::Tensor& cur_len,
                                       const paddle::Tensor& min_len,
                                       const paddle::Tensor& eos_token_id) {

    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;
    auto cu_stream = logits.stream();
    std::vector<int64_t> shape = logits.shape();
    auto repeat_times = paddle::full(shape, 0, paddle::DataType::INT32, pre_ids.place());
    int64_t bs = shape[0];
    int64_t length = shape[1];
    int64_t length_id = pre_ids.shape()[1];
    int64_t length_bad_words = bad_tokens.shape()[0];

    int64_t end_length = eos_token_id.shape()[0];

    int block_size = (bs + 32 - 1) / 32 * 32;
    min_length_logits_process_v2<<<1, block_size, 0, cu_stream>>>(
        reinterpret_cast<DataType_*>(const_cast<data_t*>(logits.data<data_t>())),
        cur_len.data<int64_t>(),
        min_len.data<int64_t>(),
        eos_token_id.data<int64_t>(),
        bs, length, end_length);

    block_size = (length_id + 32 - 1) / 32 * 32;
    block_size = min(block_size, 512);
    update_repeat_times_v2<<<bs, block_size, 0, cu_stream>>>(
        pre_ids.data<int64_t>(), 
        cur_len.data<int64_t>(), 
        repeat_times.data<int>(), 
        bs, 
        length, 
        length_id);
    
    block_size = (length + 32 - 1) / 32 * 32;
    block_size = min(block_size, 512);
    update_value_by_repeat_times_v2<DataType_><<<bs, block_size, 0, cu_stream>>>(
        repeat_times.data<int>(),
        reinterpret_cast<DataType_*>(const_cast<data_t*>(penalty_scores.data<data_t>())),
        reinterpret_cast<DataType_*>(const_cast<data_t*>(frequency_score.data<data_t>())),
        reinterpret_cast<DataType_*>(const_cast<data_t*>(presence_score.data<data_t>())),
        temperatures.data<float>(),
        reinterpret_cast<DataType_*>(const_cast<data_t*>(logits.data<data_t>())),
        bs, 
        length);

    block_size = (length_bad_words + 32 - 1) / 32 * 32;
    block_size = min(block_size, 512);
    ban_bad_words<DataType_><<<bs, block_size, 0, cu_stream>>>(
        reinterpret_cast<DataType_*>(const_cast<data_t*>(logits.data<data_t>())),
        bad_tokens.data<int64_t>(),
        bs,
        length,
        length_bad_words
    );
}

void TokenPenaltyMultiScoresV2(const paddle::Tensor& pre_ids,
                             const paddle::Tensor& logits,
                             const paddle::Tensor& penalty_scores,
                             const paddle::Tensor& frequency_scores,
                             const paddle::Tensor& presence_scores,
                             const paddle::Tensor& temperatures,
                             const paddle::Tensor& bad_tokens,
                             const paddle::Tensor& cur_len,
                             const paddle::Tensor& min_len,
                             const paddle::Tensor& eos_token_id) {

    switch (logits.type()) {
        case paddle::DataType::BFLOAT16: {
            return token_penalty_multi_scores_kernel_v2<paddle::DataType::BFLOAT16>(
                pre_ids,
                logits,
                penalty_scores,
                frequency_scores,
                presence_scores,
                temperatures,
                bad_tokens,
                cur_len,
                min_len,
                eos_token_id
            );
        }
        case paddle::DataType::FLOAT16: {
            return token_penalty_multi_scores_kernel_v2<paddle::DataType::FLOAT16>(
                pre_ids,
                logits,
                penalty_scores,
                frequency_scores,
                presence_scores,
                temperatures,
                bad_tokens,
                cur_len,
                min_len,
                eos_token_id
            );
        }
        case paddle::DataType::FLOAT32: {
            return token_penalty_multi_scores_kernel_v2<paddle::DataType::FLOAT32>(
                pre_ids,
                logits,
                penalty_scores,
                frequency_scores,
                presence_scores,
                temperatures,
                bad_tokens,
                cur_len,
                min_len,
                eos_token_id
            );
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only float16, bfloat16 and float32 are supported. ");
            break;
        }
    }
}

PD_BUILD_OP(get_token_penalty_multi_scores_v2)
    .Inputs({"pre_ids", 
             "logits", 
             "penalty_scores", 
             "frequency_scores", 
             "presence_scores", 
             "temperatures", 
             "bad_tokens", 
             "cur_len", 
             "min_len", 
             "eos_token_id"})
    .Outputs({"logits_out"})
    .SetInplaceMap({{"logits", "logits_out"}})
    .SetKernelFn(PD_KERNEL(TokenPenaltyMultiScoresV2));