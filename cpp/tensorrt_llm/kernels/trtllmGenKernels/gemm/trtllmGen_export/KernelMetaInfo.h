/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "GemmOptions.h"

namespace tensorrt_llm
{
namespace kernels
{
// clang-format off

#define TLLM_GEN_COMMIT "23d32a5"
#define TLLM_GEN_EXPORT_VERSION "0.0"

static constexpr size_t tllmGenGemmListLen = 12;

#ifndef EXCLUDE_SM_100
extern unsigned char GemmKernel_Bfloat16_E2m1_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x64_cluster1x1x1_sm100a_cubin[];
extern unsigned char GemmKernel_Bfloat16_E4m3_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x32_cluster1x1x1_sm100a_cubin[];
extern unsigned char GemmKernel_Bfloat16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x1_transposeMmaOutput_sm100a_cubin[];
extern unsigned char GemmKernel_Bfloat16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x2_splitK2_transposeMmaOutput_sm100a_cubin[];
extern unsigned char GemmKernel_E4m3_E4m3_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x32_cluster1x1x1_sm100a_cubin[];
extern unsigned char GemmKernel_E4m3_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x1_transposeMmaOutput_sm100a_cubin[];
extern unsigned char GemmKernel_E4m3_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x2_splitK2_transposeMmaOutput_sm100a_cubin[];
extern unsigned char GemmKernel_Fp16_E2m1_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x64_cluster1x1x1_sm100a_cubin[];
extern unsigned char GemmKernel_Fp16_E4m3_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x32_cluster1x1x1_sm100a_cubin[];
extern unsigned char GemmKernel_Fp16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x1_transposeMmaOutput_sm100a_cubin[];
extern unsigned char GemmKernel_Fp16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x2_splitK2_transposeMmaOutput_sm100a_cubin[];
extern unsigned char GemmKernel_Fp32_E2m1_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x64_cluster1x1x1_sm100a_cubin[];
#endif // EXCLUDE_SM_100

#ifndef EXCLUDE_SM_100
extern unsigned int GemmKernel_Bfloat16_E2m1_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x64_cluster1x1x1_sm100a_cubin_len;
extern unsigned int GemmKernel_Bfloat16_E4m3_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x32_cluster1x1x1_sm100a_cubin_len;
extern unsigned int GemmKernel_Bfloat16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x1_transposeMmaOutput_sm100a_cubin_len;
extern unsigned int GemmKernel_Bfloat16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x2_splitK2_transposeMmaOutput_sm100a_cubin_len;
extern unsigned int GemmKernel_E4m3_E4m3_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x32_cluster1x1x1_sm100a_cubin_len;
extern unsigned int GemmKernel_E4m3_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x1_transposeMmaOutput_sm100a_cubin_len;
extern unsigned int GemmKernel_E4m3_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x2_splitK2_transposeMmaOutput_sm100a_cubin_len;
extern unsigned int GemmKernel_Fp16_E2m1_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x64_cluster1x1x1_sm100a_cubin_len;
extern unsigned int GemmKernel_Fp16_E4m3_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x32_cluster1x1x1_sm100a_cubin_len;
extern unsigned int GemmKernel_Fp16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x1_transposeMmaOutput_sm100a_cubin_len;
extern unsigned int GemmKernel_Fp16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x2_splitK2_transposeMmaOutput_sm100a_cubin_len;
extern unsigned int GemmKernel_Fp32_E2m1_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x64_cluster1x1x1_sm100a_cubin_len;
#endif // EXCLUDE_SM_100


static const gemm::GemmConfig tllmGenGemmList[] = {
#ifndef EXCLUDE_SM_100
{GemmKernel_Bfloat16_E2m1_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x64_cluster1x1x1_sm100a_cubin, GemmKernel_Bfloat16_E2m1_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x64_cluster1x1x1_sm100a_cubin_len, 150528, "gemmKernel_Bfloat16_E2m1_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x64_cluster1x1x1_sm100a", 320, { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056777)
, /* mDtypeElt */ trtllm::gen::Dtype(17826819)
, /* mDtypeC */ trtllm::gen::Dtype(1052672)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 128
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mM */ 256
, /* mMmaK */ 64
, /* mMmaM */ 128
, /* mMmaN */ 128
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 3
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mUseShuffledMatrixA */ 0
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 0
, /* mTileM */ 128
, /* mTileN */ 128
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, gemm::SmVersion::Sm100a },
{GemmKernel_Bfloat16_E4m3_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x32_cluster1x1x1_sm100a_cubin, GemmKernel_Bfloat16_E4m3_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x32_cluster1x1x1_sm100a_cubin_len, 168960, "gemmKernel_Bfloat16_E4m3_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x32_cluster1x1x1_sm100a", 224, { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056777)
, /* mDtypeElt */ trtllm::gen::Dtype(1050630)
, /* mDtypeC */ trtllm::gen::Dtype(1052672)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 128
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaM */ 128
, /* mMmaN */ 128
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 2
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mUseShuffledMatrixA */ 0
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 0
, /* mTileM */ 128
, /* mTileN */ 128
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, gemm::SmVersion::Sm100a },
{GemmKernel_Bfloat16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x1_transposeMmaOutput_sm100a_cubin, GemmKernel_Bfloat16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x1_transposeMmaOutput_sm100a_cubin_len, 217088, "gemmKernel_Bfloat16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x1_transposeMmaOutput_sm100a", 224, { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056777)
, /* mDtypeElt */ trtllm::gen::Dtype(1050630)
, /* mDtypeC */ trtllm::gen::Dtype(1052672)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 8
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 1
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 0
, /* mGridWaitForPrimaryB */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaM */ 128
, /* mMmaN */ 8
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 3
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mUseShuffledMatrixA */ 1
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 8
, /* mTileK */ 512
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, gemm::SmVersion::Sm100a },
{GemmKernel_Bfloat16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x2_splitK2_transposeMmaOutput_sm100a_cubin, GemmKernel_Bfloat16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x2_splitK2_transposeMmaOutput_sm100a_cubin_len, 215040, "gemmKernel_Bfloat16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x2_splitK2_transposeMmaOutput_sm100a", 224, { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 2
, /* mDtypeAcc */ trtllm::gen::Dtype(1056777)
, /* mDtypeElt */ trtllm::gen::Dtype(1050630)
, /* mDtypeC */ trtllm::gen::Dtype(1052672)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 8
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 1
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 0
, /* mGridWaitForPrimaryB */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaM */ 128
, /* mMmaN */ 8
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 2
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 3
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mUseShuffledMatrixA */ 1
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(2)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 8
, /* mTileK */ 512
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, gemm::SmVersion::Sm100a },
{GemmKernel_E4m3_E4m3_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x32_cluster1x1x1_sm100a_cubin, GemmKernel_E4m3_E4m3_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x32_cluster1x1x1_sm100a_cubin_len, 218112, "gemmKernel_E4m3_E4m3_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x32_cluster1x1x1_sm100a", 224, { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056777)
, /* mDtypeElt */ trtllm::gen::Dtype(1050630)
, /* mDtypeC */ trtllm::gen::Dtype(1050630)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 128
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaM */ 128
, /* mMmaN */ 128
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 3
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mUseShuffledMatrixA */ 0
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 0
, /* mTileM */ 128
, /* mTileN */ 128
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, gemm::SmVersion::Sm100a },
{GemmKernel_E4m3_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x1_transposeMmaOutput_sm100a_cubin, GemmKernel_E4m3_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x1_transposeMmaOutput_sm100a_cubin_len, 216064, "gemmKernel_E4m3_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x1_transposeMmaOutput_sm100a", 224, { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056777)
, /* mDtypeElt */ trtllm::gen::Dtype(1050630)
, /* mDtypeC */ trtllm::gen::Dtype(1050630)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 8
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 1
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 0
, /* mGridWaitForPrimaryB */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaM */ 128
, /* mMmaN */ 8
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 3
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mUseShuffledMatrixA */ 1
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 8
, /* mTileK */ 512
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, gemm::SmVersion::Sm100a },
{GemmKernel_E4m3_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x2_splitK2_transposeMmaOutput_sm100a_cubin, GemmKernel_E4m3_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x2_splitK2_transposeMmaOutput_sm100a_cubin_len, 215040, "gemmKernel_E4m3_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x2_splitK2_transposeMmaOutput_sm100a", 224, { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 2
, /* mDtypeAcc */ trtllm::gen::Dtype(1056777)
, /* mDtypeElt */ trtllm::gen::Dtype(1050630)
, /* mDtypeC */ trtllm::gen::Dtype(1050630)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 8
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 1
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 0
, /* mGridWaitForPrimaryB */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaM */ 128
, /* mMmaN */ 8
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 2
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 3
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mUseShuffledMatrixA */ 1
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(2)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 8
, /* mTileK */ 512
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, gemm::SmVersion::Sm100a },
{GemmKernel_Fp16_E2m1_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x64_cluster1x1x1_sm100a_cubin, GemmKernel_Fp16_E2m1_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x64_cluster1x1x1_sm100a_cubin_len, 150528, "gemmKernel_Fp16_E2m1_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x64_cluster1x1x1_sm100a", 320, { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056777)
, /* mDtypeElt */ trtllm::gen::Dtype(17826819)
, /* mDtypeC */ trtllm::gen::Dtype(1052680)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 128
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mM */ 256
, /* mMmaK */ 64
, /* mMmaM */ 128
, /* mMmaN */ 128
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 3
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mUseShuffledMatrixA */ 0
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 0
, /* mTileM */ 128
, /* mTileN */ 128
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, gemm::SmVersion::Sm100a },
{GemmKernel_Fp16_E4m3_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x32_cluster1x1x1_sm100a_cubin, GemmKernel_Fp16_E4m3_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x32_cluster1x1x1_sm100a_cubin_len, 168960, "gemmKernel_Fp16_E4m3_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x32_cluster1x1x1_sm100a", 224, { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056777)
, /* mDtypeElt */ trtllm::gen::Dtype(1050630)
, /* mDtypeC */ trtllm::gen::Dtype(1052680)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 128
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaM */ 128
, /* mMmaN */ 128
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 2
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mUseShuffledMatrixA */ 0
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 0
, /* mTileM */ 128
, /* mTileN */ 128
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, gemm::SmVersion::Sm100a },
{GemmKernel_Fp16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x1_transposeMmaOutput_sm100a_cubin, GemmKernel_Fp16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x1_transposeMmaOutput_sm100a_cubin_len, 217088, "gemmKernel_Fp16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x1_transposeMmaOutput_sm100a", 224, { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056777)
, /* mDtypeElt */ trtllm::gen::Dtype(1050630)
, /* mDtypeC */ trtllm::gen::Dtype(1052680)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 8
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 1
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 0
, /* mGridWaitForPrimaryB */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaM */ 128
, /* mMmaN */ 8
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 3
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mUseShuffledMatrixA */ 1
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 8
, /* mTileK */ 512
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, gemm::SmVersion::Sm100a },
{GemmKernel_Fp16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x2_splitK2_transposeMmaOutput_sm100a_cubin, GemmKernel_Fp16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x2_splitK2_transposeMmaOutput_sm100a_cubin_len, 215040, "gemmKernel_Fp16_E4m3_Fp32_tile128x8x512_epilogueTile128x8_mma128x8x32_cluster1x1x2_splitK2_transposeMmaOutput_sm100a", 224, { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 2
, /* mDtypeAcc */ trtllm::gen::Dtype(1056777)
, /* mDtypeElt */ trtllm::gen::Dtype(1050630)
, /* mDtypeC */ trtllm::gen::Dtype(1052680)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 8
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 1
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 0
, /* mGridWaitForPrimaryB */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaM */ 128
, /* mMmaN */ 8
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 2
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 3
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mUseShuffledMatrixA */ 1
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(2)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 8
, /* mTileK */ 512
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, gemm::SmVersion::Sm100a },
{GemmKernel_Fp32_E2m1_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x64_cluster1x1x1_sm100a_cubin, GemmKernel_Fp32_E2m1_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x64_cluster1x1x1_sm100a_cubin_len, 183296, "gemmKernel_Fp32_E2m1_Fp32_tile128x128x256_epilogueTile128x128_mma128x128x64_cluster1x1x1_sm100a", 320, { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056777)
, /* mDtypeElt */ trtllm::gen::Dtype(17826819)
, /* mDtypeC */ trtllm::gen::Dtype(1056777)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 128
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mM */ 256
, /* mMmaK */ 64
, /* mMmaM */ 128
, /* mMmaN */ 128
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 3
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mUseShuffledMatrixA */ 0
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 0
, /* mTileM */ 128
, /* mTileN */ 128
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, gemm::SmVersion::Sm100a },
#endif // EXCLUDE_SM_100
};
// clang-format on
} // namespace kernels
} // namespace tensorrt_llm
