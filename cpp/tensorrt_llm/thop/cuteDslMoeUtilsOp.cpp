/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/cuteDslKernels/moeUtils.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <cuda_fp4.h>

namespace torch_ext
{
// Sort
using tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::RoutingMethodType;

std::vector<torch::Tensor> moe_topk_sort_impl(torch::optional<torch::Tensor> const& routing_logits,
    torch::optional<torch::Tensor> const& routing_bias, torch::optional<torch::Tensor> const& token_selected_experts,
    torch::optional<torch::Tensor> const& token_final_scales, int64_t const num_experts, int64_t const top_k,
    std::optional<int64_t> const n_group, std::optional<int64_t> const topk_group, int64_t const local_expert_offset,
    int64_t const local_num_experts, std::optional<double> const routed_scaling_factor, int64_t const tile_tokens_dim,
    int64_t const routing_method_type)
{
    int64_t const num_tokens
        = token_selected_experts.has_value() ? token_selected_experts->size(0) : routing_logits->size(0);
    int64_t const max_num_padded_tokens
        = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::getMaxPermutedPaddedCount(
            num_tokens, top_k, num_experts, tile_tokens_dim);
    int64_t const max_num_ctas = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::getMaxNumCtasInBatchDim(
        num_tokens, top_k, num_experts, tile_tokens_dim);
    int64_t const size_of_expert_count_histogram = std::max(num_experts * 2, int64_t(256 * 2));
    auto const routing_bias_dtype = routing_bias.has_value() ? routing_bias->scalar_type() : torch::kBFloat16;

    auto routing_logits_ptr = routing_logits.has_value() ? routing_logits->data_ptr() : nullptr;
    auto routing_bias_ptr = routing_bias.has_value() ? routing_bias->data_ptr() : nullptr;
    auto token_selected_experts_ptr
        = token_selected_experts.has_value() ? token_selected_experts->data_ptr<int32_t>() : nullptr;
    auto token_final_scales_ptr = token_final_scales.has_value() ? token_final_scales->data_ptr() : nullptr;

    torch::optional<torch::Tensor> new_token_final_scales;
    if (token_final_scales_ptr == nullptr)
    {
        new_token_final_scales
            = torch::empty({num_tokens, top_k}, torch::dtype(routing_bias_dtype).device(torch::kCUDA));
        token_final_scales_ptr = new_token_final_scales->data_ptr();
    }

    auto expert_indexes = torch::empty({num_tokens, top_k}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto expert_count_histogram
        = torch::empty({size_of_expert_count_histogram}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto total_num_padded_tokens = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto expanded_idx_to_permuted_idx
        = torch::empty({num_tokens, top_k}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto permuted_idx_to_expanded_idx
        = torch::empty({max_num_padded_tokens}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto num_tokens_per_expert = torch::empty({num_experts}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto tile_idx_to_batch_idx = torch::empty({max_num_ctas}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto tile_idx_to_mn_limit = torch::empty({max_num_ctas}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto num_non_exiting_tiles = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::Runner routing_runner(tile_tokens_dim);
    auto const& stream = at::cuda::getCurrentCUDAStream(
        routing_logits.has_value() ? routing_logits.value().get_device() : token_selected_experts.value().get_device());
    routing_runner.run(routing_logits_ptr, routing_bias_ptr, num_tokens, num_experts, top_k, n_group.value_or(0),
        topk_group.value_or(0), local_expert_offset, local_num_experts, routed_scaling_factor.value_or(1.0),
        expert_indexes.data_ptr<int>(), expert_count_histogram.data_ptr<int>(), total_num_padded_tokens.data_ptr<int>(),
        expanded_idx_to_permuted_idx.data_ptr<int>(), permuted_idx_to_expanded_idx.data_ptr<int>(),
        nullptr /*permuted_idx_to_token_idx.data_ptr<int>()*/, token_final_scales_ptr, token_selected_experts_ptr,
        num_tokens_per_expert.data_ptr<int>(), tile_idx_to_batch_idx.data_ptr<int>(),
        tile_idx_to_mn_limit.data_ptr<int>(), num_non_exiting_tiles.data_ptr<int>(),
        batchedGemm::trtllm::gen::Dtype::Void /* dtypeElt */, false /* use_routing_scales_on_input */,
        false /* use_deep_seek_fp8 */, static_cast<RoutingMethodType>(routing_method_type), stream);

    std::vector<torch::Tensor> results{tile_idx_to_batch_idx, tile_idx_to_mn_limit, expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx, total_num_padded_tokens, num_non_exiting_tiles};
    if (new_token_final_scales.has_value())
    {
        results.push_back(new_token_final_scales.value());
    }
    return results;
}

std::vector<torch::Tensor> moe_topk_sort(torch::Tensor const& routing_logits,
    torch::optional<torch::Tensor> const& routing_bias, int64_t const num_experts, int64_t const top_k,
    std::optional<int64_t> const n_group, std::optional<int64_t> const topk_group, int64_t const local_expert_offset,
    int64_t const local_num_experts, std::optional<double> const routed_scaling_factor, int64_t const tile_tokens_dim,
    int64_t const routing_method_type)
{
    TORCH_CHECK(routing_logits.dim() == 2, "routing_logits must be 2D.");
    TORCH_CHECK(routing_logits.size(1) == num_experts, "routing_logits.size(1) must be num_experts.");
    if (routing_bias.has_value())
    {
        TORCH_CHECK(routing_bias->dim() == 1, "routing_bias must be 1D.");
        TORCH_CHECK(routing_bias->size(0) == num_experts, "routing_bias.size(0) must be num_experts.");
    }
    return moe_topk_sort_impl(routing_logits, routing_bias, std::nullopt, std::nullopt, num_experts, top_k, n_group,
        topk_group, local_expert_offset, local_num_experts, routed_scaling_factor, tile_tokens_dim,
        routing_method_type);
}

std::vector<torch::Tensor> moe_sort(torch::Tensor const& token_selected_experts,
    torch::Tensor const& token_final_scales, int64_t const num_experts, int64_t const top_k,
    std::optional<int64_t> const n_group, std::optional<int64_t> const topk_group, int64_t const local_expert_offset,
    int64_t const local_num_experts, std::optional<double> const routed_scaling_factor, int64_t const tile_tokens_dim,
    int64_t const routing_method_type)
{
    TORCH_CHECK(token_selected_experts.dim() == 2, "token_selected_experts must be 2D.");
    int64_t const num_tokens = token_selected_experts.size(0);
    TORCH_CHECK(token_selected_experts.size(1) == top_k, "token_selected_experts.size(1) must be top_k.");
    TORCH_CHECK(token_final_scales.dim() == 2, "token_final_scales must be 2D.");
    TORCH_CHECK(token_final_scales.size(0) == num_tokens, "token_final_scales.size(0) must be num_tokens.");
    TORCH_CHECK(token_final_scales.size(1) == top_k, "token_final_scales.size(1) must be top_k.");
    return moe_topk_sort_impl(std::nullopt, std::nullopt, token_selected_experts, token_final_scales, num_experts,
        top_k, n_group, topk_group, local_expert_offset, local_num_experts, routed_scaling_factor, tile_tokens_dim,
        routing_method_type);
}

// Permute

std::tuple<torch::Tensor, torch::optional<torch::Tensor>> moe_permute(torch::Tensor const& input,
    torch::optional<torch::Tensor> const& input_sf, torch::Tensor const& permuted_idx_to_expanded_idx,
    torch::Tensor const& num_non_exiting_tiles, int64_t const tile_tokens_dim, int64_t const top_k)
{
    int64_t const hidden_size = input.scalar_type() == torch::kFloat4_e2m1fn_x2 ? input.size(1) * 2 : input.size(1);
    int64_t const num_permuted_tokens = permuted_idx_to_expanded_idx.size(0);
    TORCH_CHECK(
        num_permuted_tokens % tile_tokens_dim == 0, "num_permuted_tokens must be divisible by tile_tokens_dim.");

    auto permuted_output
        = torch::empty({num_permuted_tokens, input.size(1)}, torch::dtype(input.scalar_type()).device(torch::kCUDA));

    void* input_sf_ptr = nullptr;
    void* permuted_sf_ptr = nullptr;
    torch::optional<torch::Tensor> permuted_sf;
    if (input.scalar_type() == torch::kFloat4_e2m1fn_x2)
    {
        TORCH_CHECK(input_sf.has_value(), "input_sf is required for NVFP4.");
        input_sf_ptr = input_sf->data_ptr();
        int64_t constexpr kSFVecSize = 16;
        permuted_sf = torch::empty({num_permuted_tokens * hidden_size / kSFVecSize},
            torch::dtype(input_sf->scalar_type()).device(torch::kCUDA));
        permuted_sf_ptr = permuted_sf->data_ptr();
    }

    auto const& stream = at::cuda::getCurrentCUDAStream(input.get_device());

#define DISPATCH_MOE_PERMUTE(InputType, SFType)                                                                        \
    tensorrt_llm::kernels::cute_dsl::moePermute<InputType, SFType>(static_cast<InputType*>(input.data_ptr()),          \
        static_cast<InputType*>(permuted_output.data_ptr()), static_cast<SFType*>(input_sf_ptr),                       \
        static_cast<SFType*>(permuted_sf_ptr), permuted_idx_to_expanded_idx.data_ptr<int32_t>(),                       \
        num_non_exiting_tiles.data_ptr<int32_t>(), hidden_size, top_k, tile_tokens_dim, stream)

    if (input.scalar_type() == torch::kHalf)
    {
        DISPATCH_MOE_PERMUTE(half, uint8_t);
    }
    else if (input.scalar_type() == torch::kBFloat16)
    {
        DISPATCH_MOE_PERMUTE(__nv_bfloat16, uint8_t);
    }
    else if (input.scalar_type() == torch::kFloat8_e4m3fn)
    {
        DISPATCH_MOE_PERMUTE(__nv_fp8_e4m3, uint8_t);
    }
    else if (input.scalar_type() == torch::kFloat4_e2m1fn_x2)
    {
        DISPATCH_MOE_PERMUTE(__nv_fp4_e2m1, uint8_t);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported input dtype: ", input.scalar_type());
    }

#undef DISPATCH_MOE_PERMUTE

    return {permuted_output, permuted_sf};
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "moe_topk_sort(Tensor routing_logits, Tensor? routing_bias, int num_experts, int top_k, int? n_group, "
        "int? topk_group, int local_expert_offset, int local_num_experts, float? routed_scaling_factor, int "
        "tile_tokens_dim, int routing_method_type) -> Tensor[]");
    m.def(
        "moe_sort(Tensor token_final_scales, Tensor token_selected_experts, int num_experts, int top_k, int? n_group, "
        "int? topk_group, int local_expert_offset, int local_num_experts, float? routed_scaling_factor, int "
        "tile_tokens_dim, int routing_method_type) -> Tensor[]");
    m.def(
        "moe_permute(Tensor input, Tensor? input_sf, Tensor permuted_idx_to_expanded_idx, Tensor "
        "num_non_exiting_tiles, "
        "int tile_tokens_dim, int top_k) -> (Tensor, Tensor?)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_topk_sort", &torch_ext::moe_topk_sort);
    m.impl("moe_sort", &torch_ext::moe_sort);
    m.impl("moe_permute", &torch_ext::moe_permute);
}
