/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file Use external cudnn utils function
 */
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <limits>
#include "cudnn_utils.h"
#include <assert.h>  

// cudnn v8
//#include <cudnn_frontend_find_plan.h>
//#include <cudnn_frontend_get_plan.h>

#define GET_DOUBLE(X) *((double*)X)
#define GET_FLOAT(X) *((float*)(X))

namespace tvm {
namespace contrib {

using namespace runtime;
/*
using common_convbias_descriptors = std::tuple<cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor>;
*/
enum {
    X_TENSOR,
    Y_TENSOR,
    W_TENSOR,
    Z_TENSOR,
    B_TENSOR,
    AFTERADD_TENSOR,
    AFTERBIAS_TENSOR,
    AFTERCONV_TENSOR,
};

void generateStrides(const int64_t* dimA, int64_t* strideA, int nbDims, cudnnTensorFormat_t filterFormat) {
    if (filterFormat == CUDNN_TENSOR_NCHW) {
        strideA[nbDims - 1] = 1;
        for (int64_t d = nbDims - 2; d >= 0; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
    } else {
        // Here we assume that the format is CUDNN_TENSOR_NHWC
        strideA[1]          = 1;
        strideA[nbDims - 1] = strideA[1] * dimA[1];
        for (int64_t d = nbDims - 2; d >= 2; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
        strideA[0] = strideA[2] * dimA[2];
    }
}

/*
common_convbias_descriptors
create_conv_bias_add_act_descriptors(
                                     int dim,
                                     int64_t* x_dim_padded,
                                     const int64_t* padA,
                                     const int64_t* convstrideA,
                                     const int64_t* dilationA,
                                     int64_t* w_dim_padded,
                                     int64_t* y_dim_padded,
                                     cudnnDataType_t data_type,
                                     cudnnTensorFormat_t format
                                     ) {
  assert(dim==2);
  int full_dims = dim+2;
  int64_t b_dim_padded[full_dims];

  b_dim_padded[0] = y_dim_padded[0];
  b_dim_padded[1] = y_dim_padded[1];
  b_dim_padded[2] = 1;
  b_dim_padded[3] = 1;

  int64_t x_stride_padded[full_dims] = {0};
  int64_t y_stride_padded[full_dims] = {0};
  int64_t w_stride_padded[full_dims] = {0};
  int64_t b_stride_padded[full_dims] = {0};

  generateStrides(w_dim_padded, w_stride_padded, full_dims, format);
  generateStrides(x_dim_padded, x_stride_padded, full_dims, format);
  generateStrides(y_dim_padded, y_stride_padded, full_dims, format);
  generateStrides(b_dim_padded, b_stride_padded, full_dims, format);


  return common_convbias_descriptors(cudnn_frontend::TensorBuilder()
                                      .setDim(full_dims, x_dim_padded)
                                      .setStrides(full_dims, x_stride_padded)
                                      .setId('x')
                                      .setAlignment(full_dims)
                                      .setDataType(data_type)
                                      .build(),
                              cudnn_frontend::TensorBuilder()
                                      .setDim(full_dims, y_dim_padded)
                                      .setStrides(full_dims, y_stride_padded)
                                      .setId('y')
                                      .setAlignment(full_dims)
                                      .setDataType(data_type)
                                      .build(),
                              cudnn_frontend::TensorBuilder()
                                      .setDim(full_dims, w_dim_padded)
                                      .setStrides(full_dims, w_stride_padded)
                                      .setId('w')
                                      .setAlignment(full_dims)
                                      .setDataType(data_type)
                                      .build(),
                              cudnn_frontend::TensorBuilder()
                                      .setDim(full_dims, y_dim_padded)
                                      .setStrides(full_dims, y_stride_padded)
                                      .setId('z')
                                      .setAlignment(full_dims)
                                      .setDataType(data_type)
                                      .build(),
                              cudnn_frontend::TensorBuilder()
                                      .setDim(full_dims, b_dim_padded)
                                      .setStrides(full_dims, b_stride_padded)
                                      .setId('b')
                                      .setAlignment(full_dims)
                                      .setDataType(data_type)
                                      .build(),
                              cudnn_frontend::TensorBuilder()
                                      .setDim(full_dims, y_dim_padded)
                                      .setStrides(full_dims, y_stride_padded)
                                      .setVirtual()
                                      .setId('A')  // after add
                                      .setAlignment(full_dims)
                                      .setDataType(data_type)
                                      .build(),
                              cudnn_frontend::TensorBuilder()
                                      .setDim(full_dims, y_dim_padded)
                                      .setStrides(full_dims, y_stride_padded)
                                      .setVirtual()
                                      .setId('B')  // after bias
                                      .setAlignment(full_dims)
                                      .setDataType(data_type)
                                      .build(),
                              cudnn_frontend::TensorBuilder()
                                      .setDim(full_dims, y_dim_padded)
                                      .setStrides(full_dims, y_stride_padded)
                                      .setId('C')  // after conv
                                      .setAlignment(full_dims)
                                      .setVirtual()
                                      .setDataType(data_type)
                                      .build());

}

bool
allowAll(cudnnBackendDescriptor_t engine_config) {
    return false;
}


// Method for engine config generator based on heuristics
auto heurgen_method = [](cudnn_frontend::OperationGraph &opGraph) -> cudnn_frontend::EngineConfigList {
    auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                          .setOperationGraph(opGraph)
                          .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                          .build();
    //std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;

    auto &engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
    cudnn_frontend::EngineConfigList filtered_configs;
    cudnn_frontend::filter(engine_configs, filtered_configs, allowAll);
    return filtered_configs;
};

*/
void MatmulActivationForward(int mode, int format, int algo, int mmdim, int groups,
    const int64_t pad[],const int64_t stride[], const int64_t dilation[],
    DLTensor* x, DLTensor* w, DLTensor* y,
    const std::string& dtype, const void* alphas[], int actvMode, int reluNanOpt, double actvCoeff) {

  /*
  
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  // Set Format
  entry_ptr->fused_matmul_entry.tensor_format = static_cast<cudnnTensorFormat_t>(format);
  // Set device
  entry_ptr->fused_matmul_entry.device = x->device;
  // Set Data Type
  entry_ptr->fused_matmul_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(dtype));
  //cudnnDataType_t data_type = CuDNNDataType::DLTypeToCuDNNType(x->dtype);
  // Dims includes N and C
  int full_dims = mmdim + 2;
  assert(mmdim==2);

  

  std::vector<int> dim(full_dims);
  std::vector<int> tensor_stride(full_dims);

  int64_t x_dim_padded[full_dims], w_dim_padded[full_dims], y_dim_padded[full_dims];
  if (mmdim == 2) {
    int ni, ci, hi, wi;
    if (entry_ptr->fused_matmul_entry.tensor_format == CUDNN_TENSOR_NHWC) {
      ni = 0;
      ci = 3;
      hi = 1;
      wi = 2;
    } else {
      ni = 0;
      ci = 1;
      hi = 2;
      wi = 3;
    }
    int order[4] = {ni,ci,hi,wi};

    for(int i=0;i<full_dims;i++){
        x_dim_padded[i] = static_cast<int>(x->shape[order[i]]);
        w_dim_padded[i] = static_cast<int>(w->shape[order[i]]);
        y_dim_padded[i] = static_cast<int>(y->shape[order[i]]);
    }

  }else{
      for(int i=0;i<full_dims;i++){
        x_dim_padded[i] = static_cast<int>(x->shape[i]);
        w_dim_padded[i] = static_cast<int>(w->shape[i]);
        y_dim_padded[i] = static_cast<int>(y->shape[i]);
      }

  }
  ///////////////////////////////////
  // Creates the necessary tensor descriptors
  int64_t stride[3];
  // the intension is to compute stride for a [1, M, K] matrix with K in the inner most dimension, and
  //
  // CUDNN_TENSOR_NCHW is a borrowed notation
  generateStrides(x_dim_padded, stride, 3, CUDNN_TENSOR_NCHW);
  auto aMatrixTensor = cudnn_frontend::TensorBuilder()
                           .setDim(3, x_dim_padded)
                           .setStrides(3, stride)
                           .setId('a')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();

  generateStrides(w_dim_padded, stride, 3, CUDNN_TENSOR_NCHW);
  auto bMatrixTensor = cudnn_frontend::TensorBuilder()
                           .setDim(3, w_dim_padded)
                           .setStrides(3, stride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

  generateStrides(y_dim_padded, stride, 3, CUDNN_TENSOR_NCHW);
  auto afterMatMulTensor = cudnn_frontend::TensorBuilder()
                               .setDim(3, y_dim_padded)
                               .setStrides(3, stride)
                               .setId('A')  // after matmul
                               .setAlignment(16)
                               .setVirtual()
                               .setDataType(dataType)
                               .build();

  auto outputTensor = cudnn_frontend::TensorBuilder()
                          .setDim(3, y_dim_padded)
                          .setStrides(3, stride)
                          .setId('c')  // output after gelu
                          .setAlignment(16)
                          .setDataType(dataType)
                          .build();

  std::cout << aMatrixTensor.describe() << std::endl;
  std::cout << bMatrixTensor.describe() << std::endl;
  std::cout << afterMatMulTensor.describe() << std::endl;
  std::cout << outputTensor.describe() << std::endl;
  
  // Define the activation descriptor
  auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                     .setMode(CUDNN_POINTWISE_RELU_FWD)
                     .setMathPrecision(CUDNN_DATA_FLOAT)
                     .build();
  std::cout << actDesc.describe() << std::endl;

  // Define the matmul desc
  auto matmulDesc = cudnn_frontend::MatMulDescBuilder().setMathPrecision(CUDNN_DATA_FLOAT).build();
  std::cout << matmulDesc.describe() << std::endl;

  // Create a matmul Node
  auto matmul_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                       .setaMatDesc(aMatrixTensor)
                       .setbMatDesc(bMatrixTensor)
                       .setcMatDesc(afterMatMulTensor)
                       .setmatmulDesc(matmulDesc)
                       .build();
  std::cout << matmul_op.describe() << std::endl;

  // Create an Activation Node.
  auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                    .setxDesc(afterMatMulTensor.getOutputTensor())
                    .setyDesc(outputTensor)
                    .setpwDesc(actDesc)
                    .build();
  std::cout << act_op.describe() << std::endl;


  ///////////////////////////////////
  // Create an Operation Graph. In this case it is convolution add bias activation
  std::array<cudnn_frontend::Operation const*, 2> ops = {&matmul_op, &act_op};

  //std::cerr << "Operation Graph Builder\n";
  auto opGraph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(entry_ptr->handle)
                      .setOperationGraph(ops.size(), ops.data())
                      .build();


  // How many engines support this operation graph ?
  auto total_engines = opGraph.getEngineCount();
  //std::cout << opGraph.describe() << " has " << total_engines << " engines." << std::endl;
  // We have to randomly pick one engine from [0, total_engines)
  // Selecting "0" by default
  auto engine = cudnn_frontend::EngineBuilder().setGlobalEngineIdx(0).setOperationGraph(opGraph).build();
  //std::cout << engine.describe() << std::endl;
  auto& knobs = engine.getSupportedKnobs();
  //for (auto it = std::begin(knobs); it != std::end(knobs); ++it) {
  //    std::cout << it->describe() << std::endl;
  //}
  if (knobs.begin() != knobs.end()) {
      //std::cout << "Updated knob choice" << std::endl;
      knobs.begin()->setChoice(knobs.begin()->getMinValue() + 1);
      //std::cout << knobs.begin()->describe() << std::endl;
  }

  // Create the requisite engine config
  auto engine_config = cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
  //std::cout << engine_config.describe() << std::endl;

  auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(entry_ptr->handle).setEngineConfig(engine_config).build();

  //std::cout << "Plan tag: " << plan.getTag() << std::endl;

  auto workspace_size = plan.getWorkspaceSize();
  entry_ptr->fused_matmul_entry.UpdateWorkspace(workspace_size);

  //std::cerr << "After Update workspace\n";
  void* data_ptrs[] = {x->data, y->data, w->data};
  int64_t uids[]    = {'x', 'y', 'w'};
  auto variantPack  = cudnn_frontend::VariantPackBuilder()
                      .setWorkspacePointer(entry_ptr->fused_matmul_entry.workspace)
                      .setDataPointers(3, data_ptrs)
                      .setUids(3, uids)
                      .build();
  //std::cout << "variantPack " << variantPack.describe() << std::endl;

  auto sample_predicate_function = [=](cudnn_frontend::ExecutionPlan const& plan) -> bool {
            return (size_t)plan.getWorkspaceSize() > workspace_size;
  };

  std::array<cudnn_frontend::GeneratorSource const, 1> sources = {heurgen_method};
  cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());

  auto options = generator.cudnnFindPlan<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(
            entry_ptr->handle, std::move(opGraph), variantPack, sample_predicate_function);

  //cudnnStatus_t status =
  CUDNN_CALL(cudnnBackendExecute(entry_ptr->handle, options.front().plan.get_raw_desc(), variantPack.get_raw_desc()));
  */

}


void ConvolutionActivationForward(int mode, int format, int algo, int convDim, int groups,
    const int64_t pad[],const int64_t stride[], const int64_t dilation[],
    DLTensor* x, DLTensor* w, DLTensor* y,
    const std::string& conv_dtype, const void* alphas[], int actvMode, int reluNanOpt, double actvCoeff) {

/*
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  // Set Mode
  entry_ptr->fused_conv_entry.mode = static_cast<cudnnConvolutionMode_t>(mode);
  // Set Format
  entry_ptr->fused_conv_entry.tensor_format = static_cast<cudnnTensorFormat_t>(format);
  // Set Algo
  entry_ptr->fused_conv_entry.fwd_algo = static_cast<cudnnConvolutionFwdAlgo_t>(algo);
  // Set device
  entry_ptr->fused_conv_entry.device = x->device;
  // Set Data Type
  entry_ptr->fused_conv_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(conv_dtype));
  //cudnnDataType_t data_type = CuDNNDataType::DLTypeToCuDNNType(x->dtype);
  // Dims includes N and C
  int full_dims = convDim + 2;
  assert(convDim==2);

  CUDNN_CALL(cudnnSetConvolutionGroupCount(entry_ptr->fused_conv_entry.conv_desc, groups));
  std::vector<int> dim(full_dims);
  std::vector<int> tensor_stride(full_dims);

  int64_t x_dim_padded[full_dims], w_dim_padded[full_dims], y_dim_padded[full_dims];

  if (convDim == 2) {
    int ni, ci, hi, wi;
    if (entry_ptr->fused_conv_entry.tensor_format == CUDNN_TENSOR_NHWC) {
      ni = 0;
      ci = 3;
      hi = 1;
      wi = 2;
    } else {
      ni = 0;
      ci = 1;
      hi = 2;
      wi = 3;
    }
    int order[4] = {ni,ci,hi,wi};

    for(int i=0;i<full_dims;i++){
        x_dim_padded[i] = static_cast<int>(x->shape[order[i]]);
        w_dim_padded[i] = static_cast<int>(w->shape[order[i]]);
        y_dim_padded[i] = static_cast<int>(y->shape[order[i]]);
    }

  }else{
      for(int i=0;i<full_dims;i++){
        x_dim_padded[i] = static_cast<int>(x->shape[i]);
        w_dim_padded[i] = static_cast<int>(w->shape[i]);
        y_dim_padded[i] = static_cast<int>(y->shape[i]);
      }

  }

  common_convbias_descriptors tensors = create_conv_bias_add_act_descriptors(
      convDim, x_dim_padded,
      pad, stride, dilation,
      w_dim_padded, y_dim_padded,
      entry_ptr->fused_conv_entry.data_type,
      entry_ptr->fused_conv_entry.tensor_format
      );
  */
  /*
  std::cout << "X:\t" << std::get<X_TENSOR>(tensors).describe() << std::endl;
  std::cout << "Y:\t " << std::get<Y_TENSOR>(tensors).describe() << std::endl;
  std::cout << "W:\t" << std::get<W_TENSOR>(tensors).describe() << std::endl;
  std::cout << "After conv:\t" << std::get<AFTERCONV_TENSOR>(tensors).describe() << std::endl;
  */

/*
  // Define the activation operation
  auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                     .setMode(CUDNN_POINTWISE_RELU_FWD)
                     .setMathPrecision(CUDNN_DATA_FLOAT)
                     .build();
  //std::cout << actDesc.describe() << std::endl;


  // Define the convolution problem
  auto convDesc = cudnn_frontend::ConvDescBuilder()
                      .setDataType(entry_ptr->fused_conv_entry.data_type)
                      .setMathMode(entry_ptr->fused_conv_entry.mode)
                      //.setMathMode(CUDNN_CONVOLUTION)
                      .setNDims(convDim)
                      .setStrides(convDim, stride)
                      .setPrePadding(convDim, pad)
                      .setPostPadding(convDim, pad)
                      .setDilation(convDim, dilation)
                      .build();
  //std::cout << convDesc.describe() << std::endl;

  // TODO: What is the best practice for supporting diverse type of alphas?
  // Create a convolution Node
  auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                      .setxDesc(std::get<X_TENSOR>(tensors))
                      .setwDesc(std::get<W_TENSOR>(tensors))
                      .setyDesc(std::get<AFTERCONV_TENSOR>(tensors))
                      .setcDesc(convDesc)
                      .setAlpha(GET_FLOAT(alphas[0]))
                      .setBeta(GET_FLOAT(alphas[1]))
                      .build();
  //std::cout << conv_op.describe() << std::endl;

  // Create an Activation Node.
  auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                      .setxDesc(conv_op.getOutputTensor())
                      .setyDesc(std::get<Y_TENSOR>(tensors))
                      .setpwDesc(actDesc)
                      .build();
  //std::cout << act_op.describe() << std::endl;



  // Create an Operation Graph. In this case it is convolution add bias activation
  std::array<cudnn_frontend::Operation const*, 2> ops = {&conv_op, &act_op};

  //std::cerr << "Operation Graph Builder\n";
  auto opGraph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(entry_ptr->handle)
                      .setOperationGraph(ops.size(), ops.data())
                      .build();
*/
  /*
  //std::cerr << "ConvForwardWorkspace\n";
  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      entry_ptr->handle, entry_ptr->conv_entry.input_desc, entry_ptr->conv_entry.filter_desc,
      entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.output_desc,
      entry_ptr->conv_entry.fwd_algo, &workspace_size));

  std::cerr << "Space: " << workspace_size << "\n";

  size_t limit = 0;
  cudaDeviceGetLimit(&limit, cudaLimitStackSize);
  printf("cudaLimitStackSize: %u\n", (unsigned)limit);
  cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
  printf("cudaLimitPrintfFifoSize: %u\n", (unsigned)limit);
  cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
  printf("cudaLimitMallocHeapSize: %u\n", (unsigned)limit);

  //size_t max_workspace_size = 10*1024;  // KB 
  //assert(workspace_size <= max_workspace_size);
  //workspace_size = std::min(max_workspace_size, workspace_size);
  
  // workspace error
  if(workspace_size>=10*1024 or workspace_size==0) {
    CUDNN_CALL(CUDNN_STATUS_INTERNAL_ERROR);
  }
  */

	/*
  // How many engines support this operation graph ?
  //auto total_engines = opGraph.getEngineCount();
  //std::cout << opGraph.describe() << " has " << total_engines << " engines." << std::endl;
  // We have to randomly pick one engine from [0, total_engines)
  // Selecting "0" by default
  auto engine = cudnn_frontend::EngineBuilder().setGlobalEngineIdx(0).setOperationGraph(opGraph).build();
  //std::cout << engine.describe() << std::endl;
  auto& knobs = engine.getSupportedKnobs();
  //for (auto it = std::begin(knobs); it != std::end(knobs); ++it) {
  //    std::cout << it->describe() << std::endl;
  //}
  if (knobs.begin() != knobs.end()) {
      //std::cout << "Updated knob choice" << std::endl;
      knobs.begin()->setChoice(knobs.begin()->getMinValue() + 1);
      //std::cout << knobs.begin()->describe() << std::endl;
  }

  // Create the requisite engine config
  auto engine_config = cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
  //std::cout << engine_config.describe() << std::endl;

  auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(entry_ptr->handle).setEngineConfig(engine_config).build();

  //std::cout << "Plan tag: " << plan.getTag() << std::endl;

  auto workspace_size = plan.getWorkspaceSize();
  //std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;


  //assert(workspace_size <= 10*1024);
  entry_ptr->fused_conv_entry.UpdateWorkspace(workspace_size);

  //std::cerr << "After Update workspace\n";
  void* data_ptrs[] = {x->data, y->data, w->data};
  int64_t uids[]    = {'x', 'y', 'w'};
  auto variantPack  = cudnn_frontend::VariantPackBuilder()
                      .setWorkspacePointer(entry_ptr->fused_conv_entry.workspace)
                      .setDataPointers(3, data_ptrs)
                      .setUids(3, uids)
                      .build();
  //std::cout << "variantPack " << variantPack.describe() << std::endl;

  auto sample_predicate_function = [=](cudnn_frontend::ExecutionPlan const& plan) -> bool {
            return (size_t)plan.getWorkspaceSize() > (size_t)workspace_size;
  };

  std::array<cudnn_frontend::GeneratorSource const, 1> sources = {heurgen_method};
  cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());

  auto options = generator.cudnnFindPlan<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(
            entry_ptr->handle, std::move(opGraph), variantPack, sample_predicate_function);
*/
  /*
  std::for_each(options.begin(), options.end(), [](struct cudnn_frontend::executionOption& opt) {
      std::cout << "Plan: " << opt.plan.getTag() << " finished in " << opt.time_ms << " ms,"
                << " workspace: " << opt.plan.getWorkspaceSize() << " bytes" << std::endl;
  });
  */
/*
  //cudnnStatus_t status =
  CUDNN_CALL(cudnnBackendExecute(entry_ptr->handle, options.front().plan.get_raw_desc(), variantPack.get_raw_desc()));
*/
}

void ConvolutionBiasActivationForward(int mode, int format, int algo, int convDim, int groups,
    const int pad[],const int stride[], const int dilation[],
    DLTensor* x, DLTensor* w, DLTensor* z, DLTensor* bias, DLTensor* y,
    const std::string& conv_dtype, const void* alphas[], int actvMode, int reluNanOpt, double actvCoeff) {

  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  // Set Mode
  entry_ptr->fused_conv_entry.mode = static_cast<cudnnConvolutionMode_t>(mode);

  //std::cerr << "ConvBiasActForward: " << mode << "\n";
  // Set Format
  entry_ptr->fused_conv_entry.tensor_format = static_cast<cudnnTensorFormat_t>(format);
  // Set Algo
  entry_ptr->fused_conv_entry.fwd_algo = static_cast<cudnnConvolutionFwdAlgo_t>(algo);
  // Set device
  entry_ptr->fused_conv_entry.device = x->device;
  // Set Data Type
  entry_ptr->fused_conv_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(conv_dtype));
  cudnnDataType_t data_type = entry_ptr->fused_conv_entry.data_type; //CuDNNDataType::DLTypeToCuDNNType(x->dtype);
  
  // Dims includes N and C
  int full_dims = convDim + 2;
  assert(convDim==2);

  // Note: For 2D tenor, using ND setters causes CUDNN_STATUS_NOT_SUPPORTED error
  // in following cudnnGetConvolutionForwardWorkspaceSize() when data type is fp16, int
  CUDNN_CALL(cudnnSetConvolutionGroupCount(entry_ptr->fused_conv_entry.conv_desc, groups));
  if (convDim == 2) {
    // Set Desc
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        entry_ptr->fused_conv_entry.conv_desc, pad[0], pad[1], stride[0], stride[1], dilation[0],
        dilation[1], entry_ptr->fused_conv_entry.mode, entry_ptr->fused_conv_entry.data_type));
    int ni, ci, hi, wi;
    if (entry_ptr->fused_conv_entry.tensor_format == CUDNN_TENSOR_NHWC) {
      ni = 0;
      ci = 3;
      hi = 1;
      wi = 2;
    } else {
      ni = 0;
      ci = 1;
      hi = 2;
      wi = 3;
    }

    // Set Filter
    CUDNN_CALL(cudnnSetFilter4dDescriptor(
        entry_ptr->fused_conv_entry.filter_desc, data_type, entry_ptr->fused_conv_entry.tensor_format,
        static_cast<int>(w->shape[ni]), static_cast<int>(w->shape[ci]),
        static_cast<int>(w->shape[hi]), static_cast<int>(w->shape[wi])));
    // Set Input
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        entry_ptr->fused_conv_entry.input_desc, entry_ptr->fused_conv_entry.tensor_format, data_type,
        static_cast<int>(x->shape[ni]), static_cast<int>(x->shape[ci]),
        static_cast<int>(x->shape[hi]), static_cast<int>(x->shape[wi])));
    // Set Bias
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        entry_ptr->fused_conv_entry.bias_desc, entry_ptr->fused_conv_entry.tensor_format, data_type,
        static_cast<int>(bias->shape[ni]), static_cast<int>(bias->shape[ci]),
        static_cast<int>(bias->shape[hi]), static_cast<int>(bias->shape[wi])));
    // Set Z
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        entry_ptr->fused_conv_entry.z_desc, entry_ptr->fused_conv_entry.tensor_format, data_type,
        static_cast<int>(z->shape[ni]), static_cast<int>(z->shape[ci]),
        static_cast<int>(z->shape[hi]), static_cast<int>(z->shape[wi])));
    // Set Output
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        entry_ptr->fused_conv_entry.output_desc, entry_ptr->fused_conv_entry.tensor_format, data_type,
        static_cast<int>(y->shape[ni]), static_cast<int>(y->shape[ci]),
        static_cast<int>(y->shape[hi]), static_cast<int>(y->shape[wi])));
  } else {
    std::vector<int> dim(full_dims);
    std::vector<int> tensor_stride(full_dims);
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(entry_ptr->fused_conv_entry.conv_desc, convDim, pad, stride,
                                               dilation, entry_ptr->fused_conv_entry.mode,
                                               entry_ptr->fused_conv_entry.data_type));

    // Set Filter
    for (int i = 0; i < full_dims; i++) {
      dim[i] = static_cast<int>(w->shape[i]);
    }
    CUDNN_CALL(cudnnSetFilterNdDescriptor(entry_ptr->fused_conv_entry.filter_desc, data_type,
                                          entry_ptr->fused_conv_entry.tensor_format, full_dims,
                                          dim.data()));
    // Set Input
    for (int i = 0; i < full_dims; i++) {
      dim[i] = static_cast<int>(x->shape[i]);
    }
    GetCudnnStride(full_dims, dim.data(), tensor_stride.data());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->fused_conv_entry.input_desc, data_type, full_dims,
                                          dim.data(), tensor_stride.data()));

    // Set Bias
    for (int i = 0; i < full_dims; i++) {
      dim[i] = static_cast<int>(bias->shape[i]);
    }
    GetCudnnStride(full_dims, dim.data(), tensor_stride.data());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->fused_conv_entry.bias_desc, data_type, full_dims,
                                          dim.data(), tensor_stride.data()));
 
    // Set Z
    for (int i = 0; i < full_dims; i++) {
      dim[i] = static_cast<int>(z->shape[i]);
    }
    GetCudnnStride(full_dims, dim.data(), tensor_stride.data());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->fused_conv_entry.z_desc, data_type, full_dims,
                                          dim.data(), tensor_stride.data()));
 
    // Set Output
    for (int i = 0; i < full_dims; i++) {
      dim[i] = static_cast<int>(y->shape[i]);
    }
    GetCudnnStride(full_dims, dim.data(), tensor_stride.data());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->fused_conv_entry.output_desc, data_type, full_dims,
                                          dim.data(), tensor_stride.data()));
  }

  if (cudnnGetVersion() > 7000) {
    CUDNN_CALL(cudnnSetConvolutionMathType(entry_ptr->fused_conv_entry.conv_desc, CUDNN_TENSOR_OP_MATH))
  }

  CUDNN_CALL(cudnnSetActivationDescriptor(entry_ptr->fused_conv_entry.activation_desc,
        static_cast<cudnnActivationMode_t>(actvMode),
        static_cast<cudnnNanPropagation_t>(reluNanOpt),
        actvCoeff));


  // Set workspace
  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      entry_ptr->handle, entry_ptr->fused_conv_entry.input_desc, entry_ptr->fused_conv_entry.filter_desc,
      entry_ptr->fused_conv_entry.conv_desc, entry_ptr->fused_conv_entry.output_desc,
      entry_ptr->fused_conv_entry.fwd_algo, &workspace_size));
  entry_ptr->fused_conv_entry.UpdateWorkspace(workspace_size);

  
  assert(entry_ptr->fused_conv_entry.input_desc);
  assert(entry_ptr->fused_conv_entry.filter_desc);
  assert(entry_ptr->fused_conv_entry.conv_desc);
  assert(entry_ptr->fused_conv_entry.z_desc);
  assert(entry_ptr->fused_conv_entry.bias_desc);
  assert(entry_ptr->fused_conv_entry.activation_desc);
  assert(entry_ptr->fused_conv_entry.output_desc);
  assert(x->data);
  assert(w->data);
  assert(z->data);
  assert(y->data);
  assert(entry_ptr->fused_conv_entry.workspace);
  std::cerr << "Workspace size: " << workspace_size << "\n";
  std::cerr << x->data << " // " << y->data << " // " << z->data << "\n";

  CUDNN_CALL(cudnnConvolutionBiasActivationForward(
      entry_ptr->handle, 
      CuDNNDataType::GetConst<1>(entry_ptr->fused_conv_entry.data_type),
      entry_ptr->fused_conv_entry.input_desc, 
      x->data, 
      entry_ptr->fused_conv_entry.filter_desc, 
      w->data,
      entry_ptr->fused_conv_entry.conv_desc,
      entry_ptr->fused_conv_entry.fwd_algo,
      entry_ptr->fused_conv_entry.workspace, 
      workspace_size,
      CuDNNDataType::GetConst<0>(entry_ptr->fused_conv_entry.data_type),
      entry_ptr->fused_conv_entry.z_desc, 
      z->data, 
      entry_ptr->fused_conv_entry.bias_desc, 
      bias->data,
      entry_ptr->fused_conv_entry.activation_desc, 
      entry_ptr->fused_conv_entry.output_desc,
      y->data));
}


// conv+bias+relu with fuse engine
void ConvolutionBiasActivationForwardWithFE(int mode, int format, int algo, int convDim, int groups,
    const int64_t pad[],const int64_t stride[], const int64_t dilation[],
    DLTensor* x, DLTensor* w, DLTensor* z, DLTensor* bias, DLTensor* y,
    const std::string& conv_dtype, const void* alphas[], int actvMode, int reluNanOpt, double actvCoeff) {

/*
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  // Set Mode
  entry_ptr->fused_conv_entry.mode = static_cast<cudnnConvolutionMode_t>(mode);

  //std::cerr << "ConvBiasActForward: " << mode << "\n";
  // Set Format
  entry_ptr->fused_conv_entry.tensor_format = static_cast<cudnnTensorFormat_t>(format);
  // Set Algo
  entry_ptr->fused_conv_entry.fwd_algo = static_cast<cudnnConvolutionFwdAlgo_t>(algo);
  // Set device
  entry_ptr->fused_conv_entry.device = x->device;
  // Set Data Type
  entry_ptr->fused_conv_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(conv_dtype));
  //cudnnDataType_t data_type = CuDNNDataType::DLTypeToCuDNNType(x->dtype);
  // Dims includes N and C
  int full_dims = convDim + 2;
  assert(convDim==2);

  CUDNN_CALL(cudnnSetConvolutionGroupCount(entry_ptr->fused_conv_entry.conv_desc, groups));
  std::vector<int> dim(full_dims);
  std::vector<int> tensor_stride(full_dims);

  int64_t x_dim_padded[full_dims], w_dim_padded[full_dims], y_dim_padded[full_dims];

  if (convDim == 2) {
    int ni, ci, hi, wi;
    if (entry_ptr->fused_conv_entry.tensor_format == CUDNN_TENSOR_NHWC) {
      ni = 0;
      ci = 3;
      hi = 1;
      wi = 2;
    } else {
      ni = 0;
      ci = 1;
      hi = 2;
      wi = 3;
    }
    int order[4] = {ni,ci,hi,wi};

    for(int i=0;i<full_dims;i++){
        x_dim_padded[i] = static_cast<int>(x->shape[order[i]]);
        w_dim_padded[i] = static_cast<int>(w->shape[order[i]]);
        y_dim_padded[i] = static_cast<int>(y->shape[order[i]]);
    }

  }else{
      for(int i=0;i<full_dims;i++){
        x_dim_padded[i] = static_cast<int>(x->shape[i]);
        w_dim_padded[i] = static_cast<int>(w->shape[i]);
        y_dim_padded[i] = static_cast<int>(y->shape[i]);
      }

  }

  common_convbias_descriptors tensors = create_conv_bias_add_act_descriptors(
      convDim, x_dim_padded,
      pad, stride, dilation,
      w_dim_padded, y_dim_padded,
      entry_ptr->fused_conv_entry.data_type,
      entry_ptr->fused_conv_entry.tensor_format
      );
  */
  /* 
  std::cout << "X:\t" << std::get<X_TENSOR>(tensors).describe() << std::endl;
  std::cout << "Y:\t " << std::get<Y_TENSOR>(tensors).describe() << std::endl;
  std::cout << "W:\t" << std::get<W_TENSOR>(tensors).describe() << std::endl;
  std::cout << "Z:\t" << std::get<Z_TENSOR>(tensors).describe() << std::endl;
  std::cout << "B:\t" << std::get<B_TENSOR>(tensors).describe() << std::endl;
  std::cout << "After add:\t" << std::get<AFTERADD_TENSOR>(tensors).describe() << std::endl;
  std::cout << "After bias:\t" << std::get<AFTERBIAS_TENSOR>(tensors).describe() << std::endl;
  std::cout << "After conv:\t" << std::get<AFTERCONV_TENSOR>(tensors).describe() << std::endl;
  */
/*
  // Define the add operation
  auto addDesc = cudnn_frontend::PointWiseDescBuilder()
                     .setMode(CUDNN_POINTWISE_ADD)
                     .setMathPrecision(CUDNN_DATA_FLOAT)
                     .build();
  //std::cout << addDesc.describe() << std::endl;

  // Define the bias operation
  auto addDesc2 = cudnn_frontend::PointWiseDescBuilder()
                      .setMode(CUDNN_POINTWISE_ADD)
                      .setMathPrecision(CUDNN_DATA_FLOAT)
                      .build();
  //std::cout << addDesc2.describe() << std::endl;

  // Define the activation operation
  auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                     .setMode(CUDNN_POINTWISE_RELU_FWD)
                     .setMathPrecision(CUDNN_DATA_FLOAT)
                     .build();
  //std::cout << actDesc.describe() << std::endl;


  // Define the convolution problem
  auto convDesc = cudnn_frontend::ConvDescBuilder()
                      .setDataType(entry_ptr->fused_conv_entry.data_type)
                      .setMathMode(entry_ptr->fused_conv_entry.mode)
                      //.setMathMode(CUDNN_CONVOLUTION)
                      .setNDims(convDim)
                      .setStrides(convDim, stride)
                      .setPrePadding(convDim, pad)
                      .setPostPadding(convDim, pad)
                      .setDilation(convDim, dilation)
                      .build();
  //std::cout << convDesc.describe() << std::endl;

  // TODO: What is the best practice for supporting diverse type of alphas?
  // Create a convolution Node
  auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                      .setxDesc(std::get<X_TENSOR>(tensors))
                      .setwDesc(std::get<W_TENSOR>(tensors))
                      .setyDesc(std::get<AFTERCONV_TENSOR>(tensors))
                      .setcDesc(convDesc)
                      .setAlpha(GET_FLOAT(alphas[0]))
                      .setBeta(GET_FLOAT(alphas[1]))
                      .build();
  //std::cout << conv_op.describe() << std::endl;

  // Create a Add Node with scaling parameters.
  auto add_op1 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                      .setxDesc(conv_op.getOutputTensor())
                      .setbDesc(std::get<Z_TENSOR>(tensors))
                      .setyDesc(std::get<AFTERADD_TENSOR>(tensors))
                      .setpwDesc(addDesc)
                      .setAlpha(GET_FLOAT(alphas[2]))
                      .setAlpha2(GET_FLOAT(alphas[3]))
                      .build();
  //std::cout << add_op1.describe() << std::endl;

  // Create a Bias Node.
  auto add_op2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                      .setxDesc(add_op1.getOutputTensor())
                      .setbDesc(std::get<B_TENSOR>(tensors))
                      .setyDesc(std::get<AFTERBIAS_TENSOR>(tensors))
                      .setpwDesc(addDesc2)
                      .build();
  //std::cout << add_op2.describe() << std::endl;

  // Create an Activation Node.
  auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                      .setxDesc(add_op2.getOutputTensor())
                      .setyDesc(std::get<Y_TENSOR>(tensors))
                      .setpwDesc(actDesc)
                      .build();
  //std::cout << act_op.describe() << std::endl;



  // Create an Operation Graph. In this case it is convolution add bias activation
  std::array<cudnn_frontend::Operation const*, 4> ops = {&conv_op, &add_op1, &add_op2, &act_op};

  //std::cerr << "Operation Graph Builder\n";
  auto opGraph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(entry_ptr->handle)
                      .setOperationGraph(ops.size(), ops.data())
                      .build();

*/
  /*
  //std::cerr << "ConvForwardWorkspace\n";
  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      entry_ptr->handle, entry_ptr->conv_entry.input_desc, entry_ptr->conv_entry.filter_desc,
      entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.output_desc,
      entry_ptr->conv_entry.fwd_algo, &workspace_size));

  std::cerr << "Space: " << workspace_size << "\n";

  size_t limit = 0;
  cudaDeviceGetLimit(&limit, cudaLimitStackSize);
  printf("cudaLimitStackSize: %u\n", (unsigned)limit);
  cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
  printf("cudaLimitPrintfFifoSize: %u\n", (unsigned)limit);
  cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
  printf("cudaLimitMallocHeapSize: %u\n", (unsigned)limit);

  //size_t max_workspace_size = 10*1024;  // KB 
  //assert(workspace_size <= max_workspace_size);
  //workspace_size = std::min(max_workspace_size, workspace_size);
  
  // workspace error
  if(workspace_size>=10*1024 or workspace_size==0) {
    CUDNN_CALL(CUDNN_STATUS_INTERNAL_ERROR);
  }
  */

	/*
  // How many engines support this operation graph ?
  //auto total_engines = opGraph.getEngineCount();
  //std::cout << opGraph.describe() << " has " << total_engines << " engines." << std::endl;
  // We have to randomly pick one engine from [0, total_engines)
  // Selecting "0" by default
  auto engine = cudnn_frontend::EngineBuilder().setGlobalEngineIdx(0).setOperationGraph(opGraph).build();
  //std::cout << engine.describe() << std::endl;
  auto& knobs = engine.getSupportedKnobs();
  //for (auto it = std::begin(knobs); it != std::end(knobs); ++it) {
  //    std::cout << it->describe() << std::endl;
  //}
  if (knobs.begin() != knobs.end()) {
      //std::cout << "Updated knob choice" << std::endl;
      knobs.begin()->setChoice(knobs.begin()->getMinValue() + 1);
      //std::cout << knobs.begin()->describe() << std::endl;
  }

  // Create the requisite engine config
  auto engine_config = cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
  //std::cout << engine_config.describe() << std::endl;

  auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(entry_ptr->handle).setEngineConfig(engine_config).build();

  //std::cout << "Plan tag: " << plan.getTag() << std::endl;

  auto workspace_size = plan.getWorkspaceSize();
  //std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;



  //assert(workspace_size <= 10*1024);
  entry_ptr->fused_conv_entry.UpdateWorkspace(workspace_size);

  //std::cerr << "After Update workspace\n";
  void* data_ptrs[] = {x->data, y->data, w->data, z->data, bias->data};
  int64_t uids[]    = {'x', 'y', 'w', 'z', 'b'};
  auto variantPack  = cudnn_frontend::VariantPackBuilder()
                      .setWorkspacePointer(entry_ptr->fused_conv_entry.workspace)
                      .setDataPointers(5, data_ptrs)
                      .setUids(5, uids)
                      .build();
  //std::cout << "variantPack " << variantPack.describe() << std::endl;

  auto sample_predicate_function = [=](cudnn_frontend::ExecutionPlan const& plan) -> bool {
            return (size_t)plan.getWorkspaceSize() > (size_t)workspace_size;
  };

  std::array<cudnn_frontend::GeneratorSource const, 1> sources = {heurgen_method};
  cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());

  auto options = generator.cudnnFindPlan<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(
            entry_ptr->handle, std::move(opGraph), variantPack, sample_predicate_function);

	    */
  /*
  std::for_each(options.begin(), options.end(), [](struct cudnn_frontend::executionOption& opt) {
      std::cout << "Plan: " << opt.plan.getTag() << " finished in " << opt.time_ms << " ms,"
                << " workspace: " << opt.plan.getWorkspaceSize() << " bytes" << std::endl;
  });
  */

	/*
  //cudnnStatus_t status =
  CUDNN_CALL(cudnnBackendExecute(entry_ptr->handle, options.front().plan.get_raw_desc(), variantPack.get_raw_desc()));
*/
}


void ConvolutionForward(int mode, int format, int algo, int dims, int groups, const int pad[],
                        const int stride[], const int dilation[], DLTensor* x, DLTensor* w,
                        DLTensor* y, const std::string& conv_dtype) {
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  // Set Mode
  entry_ptr->conv_entry.mode = static_cast<cudnnConvolutionMode_t>(mode);
  // Set Format
  entry_ptr->conv_entry.tensor_format = static_cast<cudnnTensorFormat_t>(format);
  // Set Algo
  entry_ptr->conv_entry.fwd_algo = static_cast<cudnnConvolutionFwdAlgo_t>(algo);
  // Set device
  entry_ptr->conv_entry.device = x->device;
  // Set Data Type
  entry_ptr->conv_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(conv_dtype));
  cudnnDataType_t data_type = CuDNNDataType::DLTypeToCuDNNType(x->dtype);
  // Dims includes N and C
  int full_dims = dims + 2;

  std::vector<int> dim(full_dims);
  std::vector<int> tensor_stride(full_dims);


  // Note: For 2D tenor, using ND setters causes CUDNN_STATUS_NOT_SUPPORTED error
  // in following cudnnGetConvolutionForwardWorkspaceSize() when data type is fp16, int
  CUDNN_CALL(cudnnSetConvolutionGroupCount(entry_ptr->conv_entry.conv_desc, groups));
  if (dims == 2) {
    // Set Desc
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        entry_ptr->conv_entry.conv_desc, pad[0], pad[1], stride[0], stride[1], dilation[0],
        dilation[1], entry_ptr->conv_entry.mode, entry_ptr->conv_entry.data_type));
    int ni, ci, hi, wi;
    if (entry_ptr->conv_entry.tensor_format == CUDNN_TENSOR_NHWC) {
      ni = 0;
      ci = 3;
      hi = 1;
      wi = 2;
    } else {
      ni = 0;
      ci = 1;
      hi = 2;
      wi = 3;
    }

    // Set Filter
    CUDNN_CALL(cudnnSetFilter4dDescriptor(
        entry_ptr->conv_entry.filter_desc, data_type, entry_ptr->conv_entry.tensor_format,
        static_cast<int>(w->shape[ni]), static_cast<int>(w->shape[ci]),
        static_cast<int>(w->shape[hi]), static_cast<int>(w->shape[wi])));
    // Set Input
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        entry_ptr->conv_entry.input_desc, entry_ptr->conv_entry.tensor_format, data_type,
        static_cast<int>(x->shape[ni]), static_cast<int>(x->shape[ci]),
        static_cast<int>(x->shape[hi]), static_cast<int>(x->shape[wi])));
    // Set Output
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        entry_ptr->conv_entry.output_desc, entry_ptr->conv_entry.tensor_format, data_type,
        static_cast<int>(y->shape[ni]), static_cast<int>(y->shape[ci]),
        static_cast<int>(y->shape[hi]), static_cast<int>(y->shape[wi])));
  } else {
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(entry_ptr->conv_entry.conv_desc, dims, pad, stride,
                                               dilation, entry_ptr->conv_entry.mode,
                                               entry_ptr->conv_entry.data_type));

    // Set Filter
    for (int i = 0; i < full_dims; i++) {
      dim[i] = static_cast<int>(w->shape[i]);
    }
    CUDNN_CALL(cudnnSetFilterNdDescriptor(entry_ptr->conv_entry.filter_desc, data_type,
                                          entry_ptr->conv_entry.tensor_format, full_dims,
                                          dim.data()));
    // Set Input
    for (int i = 0; i < full_dims; i++) {
      dim[i] = static_cast<int>(x->shape[i]);
    }
    GetCudnnStride(full_dims, dim.data(), tensor_stride.data());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->conv_entry.input_desc, data_type, full_dims,
                                          dim.data(), tensor_stride.data()));
    // Set Output
    for (int i = 0; i < full_dims; i++) {
      dim[i] = static_cast<int>(y->shape[i]);
    }
    GetCudnnStride(full_dims, dim.data(), tensor_stride.data());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->conv_entry.output_desc, data_type, full_dims,
                                          dim.data(), tensor_stride.data()));
  }

  if (cudnnGetVersion() > 7000) {
    CUDNN_CALL(cudnnSetConvolutionMathType(entry_ptr->conv_entry.conv_desc, CUDNN_TENSOR_OP_MATH))
  }

  // Set workspace
  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      entry_ptr->handle, entry_ptr->conv_entry.input_desc, entry_ptr->conv_entry.filter_desc,
      entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.output_desc,
      entry_ptr->conv_entry.fwd_algo, &workspace_size));
  entry_ptr->conv_entry.UpdateWorkspace(workspace_size);
  CUDNN_CALL(cudnnConvolutionForward(
      entry_ptr->handle, CuDNNDataType::GetConst<1>(entry_ptr->conv_entry.data_type),
      entry_ptr->conv_entry.input_desc, x->data, entry_ptr->conv_entry.filter_desc, w->data,
      entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.fwd_algo,
      entry_ptr->conv_entry.workspace, workspace_size,
      CuDNNDataType::GetConst<0>(entry_ptr->conv_entry.data_type),
      entry_ptr->conv_entry.output_desc, y->data));
}

void OutputShape(int format, int dims, int groups, const int pad[], const int stride[],
                 const int dilation[], const int x_dim[], const int w_dim[], void* out_shape,
                 const std::string& data_dtype, const std::string& conv_dtype) {

  /*
  std::cerr << "format: " << format << ", dims:" << dims << ", groups: " << groups << ", data_dtype: " << data_dtype << ", conv_dtype: " << conv_dtype << "\n";

  for(int i=0;i<dims;i++){
      std::cerr << pad[i] << ", " << stride[i] << ", " << dilation[i] << "\n";
    }
  */


  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();

  // Set Data Type
  entry_ptr->conv_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(conv_dtype));
  cudnnDataType_t data_type = CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(data_dtype));
  // Set Format
  entry_ptr->conv_entry.tensor_format = static_cast<cudnnTensorFormat_t>(format);
  // Dims includes N and C
  int full_dims = dims + 2;

  // conv desc
  CUDNN_CALL(cudnnSetConvolutionGroupCount(entry_ptr->conv_entry.conv_desc, groups));
  CUDNN_CALL(cudnnSetConvolutionNdDescriptor(entry_ptr->conv_entry.conv_desc, dims, pad, stride,
                                             dilation, CUDNN_CROSS_CORRELATION,
                                             entry_ptr->conv_entry.data_type));

  if (dims == 2 && entry_ptr->conv_entry.tensor_format == CUDNN_TENSOR_NHWC) {
    // Set Input
    CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->conv_entry.input_desc,
                                          entry_ptr->conv_entry.tensor_format, data_type, x_dim[0],
                                          x_dim[3], x_dim[1], x_dim[2]));

    // filter desc
    CUDNN_CALL(cudnnSetFilter4dDescriptor(entry_ptr->conv_entry.filter_desc, data_type,
                                          entry_ptr->conv_entry.tensor_format, w_dim[0], w_dim[3],
                                          w_dim[1], w_dim[2]));

    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.input_desc,
        entry_ptr->conv_entry.filter_desc, static_cast<int*>(out_shape),
        static_cast<int*>(out_shape) + 3, static_cast<int*>(out_shape) + 1,
        static_cast<int*>(out_shape) + 2));
  } else {
    //std::cerr << "Before creating tensor stride\n";
    // Set Input
    std::vector<int> tensor_stride(full_dims);
    GetCudnnStride(full_dims, x_dim, tensor_stride.data());

    //for(int i=0;i<full_dims;i++){
    //  std::cerr << x_dim[i] << ", " << w_dim[i] << "\n";
    //}

    CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->conv_entry.input_desc, data_type, full_dims,
                                          x_dim, tensor_stride.data()));
    // filter desc
    CUDNN_CALL(cudnnSetFilterNdDescriptor(entry_ptr->conv_entry.filter_desc, data_type,
                                          entry_ptr->conv_entry.tensor_format, full_dims, w_dim));

    
    //std::cerr << "Compute OutputDim\n";
    CUDNN_CALL(cudnnGetConvolutionNdForwardOutputDim(
        entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.input_desc,
        entry_ptr->conv_entry.filter_desc, full_dims, static_cast<int*>(out_shape)));
  }
}

void FindAlgo(int format, int dims, int groups, const int pad[], const int stride[],
              const int dilation[], const int x_dim[], const int w_dim[], const int y_dim[],
              const std::string& data_dtype, const std::string& conv_dtype, TVMRetValue* ret) {
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();

  // Set Data Type
  entry_ptr->conv_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(conv_dtype));
  cudnnDataType_t data_type = CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(data_dtype));
  // Set Format
  entry_ptr->conv_entry.tensor_format = static_cast<cudnnTensorFormat_t>(format);
  // Dims includes N and C
  int full_dims = dims + 2;

  // conv desc
  CUDNN_CALL(cudnnSetConvolutionGroupCount(entry_ptr->conv_entry.conv_desc, groups));
  CUDNN_CALL(cudnnSetConvolutionNdDescriptor(entry_ptr->conv_entry.conv_desc, dims, pad, stride,
                                             dilation, CUDNN_CROSS_CORRELATION,
                                             entry_ptr->conv_entry.data_type));

  std::vector<int> tensor_stride(full_dims);
  // input desc
  GetCudnnStride(full_dims, x_dim, tensor_stride.data());
  CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->conv_entry.input_desc, data_type, full_dims,
                                        x_dim, tensor_stride.data()));
  // filter desc
  CUDNN_CALL(cudnnSetFilterNdDescriptor(entry_ptr->conv_entry.filter_desc, data_type,
                                        entry_ptr->conv_entry.tensor_format, full_dims, w_dim));

  // output desc
  GetCudnnStride(full_dims, y_dim, tensor_stride.data());
  CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->conv_entry.output_desc, data_type, full_dims,
                                        y_dim, tensor_stride.data()));
  if (cudnnGetVersion() > 7000) {
    CUDNN_CALL(cudnnSetConvolutionMathType(entry_ptr->conv_entry.conv_desc, CUDNN_TENSOR_OP_MATH))
  }

  int returned_algo_count = 0;
  cudnnConvolutionFwdAlgoPerf_t perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
  CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(
      entry_ptr->handle, entry_ptr->conv_entry.input_desc, entry_ptr->conv_entry.filter_desc,
      entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.output_desc,
      CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned_algo_count, perf_results));

  const std::vector<std::string> fwd_algo_names{"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"};

  auto best_algo = perf_results[0].algo;
  /*
  LOG(INFO) << "\tCUDNN Found " << returned_algo_count << " fwd algorithms, choosing "
            << fwd_algo_names[best_algo];
  for (int i = 0; i < returned_algo_count; ++i) {
    LOG(INFO) << "\t\t" << i << ") " << fwd_algo_names[perf_results[i].algo]
              << " - time: " << perf_results[i].time << " ms"
              << ", Memory: " << perf_results[i].memory;
  }
  */
  ret[0] = best_algo;
}

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv2d+bias+activation.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) {

      //std::cerr << "### Fused ops\n";

      int mode = args[0];
      int format = args[1];
      int algo = args[2];

      int64_t pad_v[2], stride_v[2], dilation_v[2];
      for (int i = 0; i < 2; i++) {
        pad_v[i] = args[3 + i];
        stride_v[i] = args[5 + i];
        dilation_v[i] = args[7 + i];
      }

      std::string conv_dtype = args[9];

      DLTensor* x = args[10];
      DLTensor* w = args[11];
      DLTensor* z = args[12];
      DLTensor* bias = args[13];
      DLTensor* y = args[14];

      int groups = args[15];

      const void* alphas[4];
      alphas[0] = CuDNNDataType::GetConst(CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(conv_dtype)), (double)args[16]);
      alphas[1] = CuDNNDataType::GetConst(CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(conv_dtype)), (double)args[17]);
      alphas[2] = CuDNNDataType::GetConst(CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(conv_dtype)), (double)args[18]);
      alphas[3] = CuDNNDataType::GetConst(CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(conv_dtype)), (double)args[19]);

      int actvMode = args[20];
      int reluNanOpt = args[21];
      double actvCoeff = args[22];

      ConvolutionBiasActivationForwardWithFE(mode, format, algo, 2, groups, pad_v, stride_v, dilation_v,
          x, w, z, bias, y, conv_dtype, alphas, actvMode, reluNanOpt, actvCoeff);


    });

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv2d+activation.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      //std::cerr << "### Fused ops\n";

      int mode = args[0];
      int format = args[1];
      int algo = args[2];

      int64_t pad_v[2], stride_v[2], dilation_v[2];
      for (int i = 0; i < 2; i++) {
        pad_v[i] = args[3 + i];
        stride_v[i] = args[5 + i];
        dilation_v[i] = args[7 + i];
      }

      std::string conv_dtype = args[9];

      DLTensor* x = args[10];
      DLTensor* w = args[11];
      DLTensor* y = args[12];

      int groups = args[13];

      const void* alphas[4];
      alphas[0] = CuDNNDataType::GetConst(CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(conv_dtype)), (double)args[14]);
      alphas[1] = CuDNNDataType::GetConst(CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(conv_dtype)), (double)args[15]);
      alphas[2] = CuDNNDataType::GetConst(CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(conv_dtype)), (double)args[16]);
      alphas[3] = CuDNNDataType::GetConst(CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(conv_dtype)), (double)args[17]);

      int actvMode = args[18];
      int reluNanOpt = args[19];
      double actvCoeff = args[20];

      ConvolutionActivationForward(mode, format, algo, 2, groups, pad_v, stride_v, dilation_v,
          x, w, y, conv_dtype, alphas, actvMode, reluNanOpt, actvCoeff);
    });




TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv2d.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      int mode = args[0];
      int format = args[1];
      int algo = args[2];
      int pad_v[2], stride_v[2], dilation_v[2];
      for (int i = 0; i < 2; i++) {
        pad_v[i] = args[3 + i];
        stride_v[i] = args[5 + i];
        dilation_v[i] = args[7 + i];
      }
      DLTensor* x = args[9];
      DLTensor* w = args[10];
      DLTensor* y = args[11];
      std::string conv_dtype = args[12];
      int groups = args[13];

      ConvolutionForward(mode, format, algo, 2, groups, pad_v, stride_v, dilation_v, x, w, y,
                         conv_dtype);
    });

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv3d.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      int mode = args[0];
      int format = args[1];
      int algo = args[2];
      int pad_v[3], stride_v[3], dilation_v[3];
      for (int i = 0; i < 3; i++) {
        pad_v[i] = args[3 + i];
        stride_v[i] = args[6 + i];
        dilation_v[i] = args[9 + i];
      }
      DLTensor* x = args[12];
      DLTensor* w = args[13];
      DLTensor* y = args[14];
      std::string conv_dtype = args[15];
      int groups = args[16];

      ConvolutionForward(mode, format, algo, 3, groups, pad_v, stride_v, dilation_v, x, w, y,
                         conv_dtype);
    });

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv.output_shape")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      int format = args[0];
      int dims = args[1];
      int* pad = static_cast<int*>(static_cast<void*>(args[2]));
      int* stride = static_cast<int*>(static_cast<void*>(args[3]));
      int* dilation = static_cast<int*>(static_cast<void*>(args[4]));
      int* x_dim = static_cast<int*>(static_cast<void*>(args[5]));
      int* w_dim = static_cast<int*>(static_cast<void*>(args[6]));
      void* out_shape = args[7];
      std::string data_dtype = args[8];
      std::string conv_dtype = args[9];
      int groups = args[10];

      OutputShape(format, dims, groups, pad, stride, dilation, x_dim, w_dim, out_shape, data_dtype,
                  conv_dtype);
    });

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv.find_algo")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      int format = args[0];
      int dims = args[1];
      int* pad = static_cast<int*>(static_cast<void*>(args[2]));
      int* stride = static_cast<int*>(static_cast<void*>(args[3]));
      int* dilation = static_cast<int*>(static_cast<void*>(args[4]));
      int* x_dim = static_cast<int*>(static_cast<void*>(args[5]));
      int* w_dim = static_cast<int*>(static_cast<void*>(args[6]));
      int* y_dim = static_cast<int*>(static_cast<void*>(args[7]));
      std::string data_dtype = args[8];
      std::string conv_dtype = args[9];
      int groups = args[10];

      FindAlgo(format, dims, groups, pad, stride, dilation, x_dim, w_dim, y_dim, data_dtype,
               conv_dtype, ret);
    });

}  // namespace contrib
}  // namespace tvm
