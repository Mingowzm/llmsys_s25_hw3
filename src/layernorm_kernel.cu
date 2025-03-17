#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32

struct SumPair {
  float sum_x;
  float sum_x_squared;
};

__device__ SumPair blockReduceSumPair(SumPair val) {
  blockReduce<ReduceType::kSum, 1>(&val.sum_x);
  blockReduce<ReduceType::kSum, 1>(&val.sum_x_squared);
  return val;
}

/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {

  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for speedup

  // Step 1
  float sum_x = 0.0f, sum_x_squared = 0.0f;
  const float4 *inp_vec4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;
  for (uint i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float4 val = inp_vec4[i];
    float val_x = val.x;
    float val_y = val.y;
    float val_z = val.z;
    float val_w = val.w;
    float sum_elements = val_x + val_y + val_z + val_w;
    float sum_squares = val_x * val_x + val_y * val_y + val_z * val_z + val_w * val_w;
    sum_x += sum_elements;
    sum_x_squared += sum_squares;
  }

  // Step 2
  blockReduce<ReduceType::kSum, 1>(&sum_x);
  __syncthreads();
  blockReduce<ReduceType::kSum, 1>(&sum_x_squared);
  __syncthreads();

  __shared__ float shared_mean, shared_variance;
  if (threadIdx.x == 0) {
    float num_elements = hidden_size * 4;
    float mean_numerator = sum_x;
    shared_mean = mean_numerator / num_elements;
    float variance_numerator = sum_x_squared / num_elements;
    float mean_squared = shared_mean * shared_mean;
    shared_variance = variance_numerator - mean_squared + LN_EPSILON;
    vars[blockIdx.x] = shared_variance;
    if (means) means[blockIdx.x] = shared_mean;
  }
  __syncthreads();

  // Step 3
  float inv_std_dev = rsqrtf(shared_variance);
  float4 *ln_res_vec4 = reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  const float4 *scale_vec4 = reinterpret_cast<const float4 *>(scale);
  const float4 *bias_vec4 = reinterpret_cast<const float4 *>(bias);

  for (uint i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float4 val = inp_vec4[i];
    float4 scale_val = scale_vec4[i];
    float4 bias_val = bias_vec4[i];

    float norm_x = (val.x - shared_mean) * inv_std_dev;
    float norm_y = (val.y - shared_mean) * inv_std_dev;
    float norm_z = (val.z - shared_mean) * inv_std_dev;
    float norm_w = (val.w - shared_mean) * inv_std_dev;

    float res_x = scale_val.x * norm_x + bias_val.x;
    float res_y = scale_val.y * norm_y + bias_val.y;
    float res_z = scale_val.z * norm_z + bias_val.z;
    float res_w = scale_val.w * norm_w + bias_val.w;

    ln_res_vec4[i] = {res_x, res_y, res_z, res_w};
  }

  /// END ASSIGN3_2
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;


  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);

}
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backword kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *gamma,
                                        const T *betta, const T *vars,
                                        const T *means, int rows, int width) {

  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  //      -> More hints about `g.shfl_down`:
  //      -> https://developer.nvidia.com/blog/cooperative-groups/#:~:text=Using%20thread_block_tile%3A%3Ashfl_down()%20to%20simplify%20our%20warp%2Dlevel%20reduction%20does%20benefit%20our%20code%3A%20it%20simplifies%20it%20and%20eliminates%20the%20need%20for%20shared%20memory
  //      -> The highlighted line gives you a conceptual understanding of what the g.shfl_down is doing. Usually, the threads inside a block need to load everything to shared memory and work together to reduce the result (like what you have implemented in the hw1 for reduce function).
  //      -> Now g.shfl_down helps you do so without consuming any shared memory. g.shfl_down makes it more efficient.
  // 4. Assign the final result to the correct position in the global output

  __shared__ float betta_partial_sum[TILE_DIM][TILE_DIM];
  __shared__ float gamma_partial_sum[TILE_DIM][TILE_DIM];

  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> tile = cg::tiled_partition<TILE_DIM>(block);

  betta_partial_sum[threadIdx.y][threadIdx.x] = 0.0f;
  gamma_partial_sum[threadIdx.y][threadIdx.x] = 0.0f;

  int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (col_idx < width) {
    for (int row_idx = threadIdx.y; row_idx < rows; row_idx += blockDim.y) {
      int data_idx = row_idx * width + col_idx;
      float grad_out_val = out_grad[data_idx];
      float inp_val = inp[data_idx];
      float x_hat;

      if (means != nullptr) {
        float mean_val = means[row_idx];
        float variance_val = vars[row_idx];
        float rsqrt_var = rsqrtf(variance_val);
        x_hat = (inp_val - mean_val) * rsqrt_var;
      } else {
        float betta_val = betta[col_idx];
        float gamma_val = gamma[col_idx];
        x_hat = (inp[data_idx] - betta_val) / gamma_val;
      }

      float betta_contrib = grad_out_val;
      float gamma_contrib = grad_out_val * x_hat;
      betta_partial_sum[threadIdx.y][threadIdx.x] += betta_contrib;
      gamma_partial_sum[threadIdx.y][threadIdx.x] += gamma_contrib;
    }
  }

  __syncthreads();

  __shared__ float s_betta_reduction[TILE_DIM];
  __shared__ float s_gamma_reduction[TILE_DIM];

  if (threadIdx.y == 0) {
    s_betta_reduction[threadIdx.x] = 0.0f;
    s_gamma_reduction[threadIdx.x] = 0.0f;

    for (int y = 0; y < TILE_DIM; y++) {
      float partial_betta = betta_partial_sum[y][threadIdx.x];
      float partial_gamma = gamma_partial_sum[y][threadIdx.x];
      s_betta_reduction[threadIdx.x] += partial_betta;
      s_gamma_reduction[threadIdx.x] += partial_gamma;
    }
  }

  __syncthreads();

  if (threadIdx.y == 0 && col_idx < width) {
    betta_grad[col_idx] = s_betta_reduction[threadIdx.x];
    gamma_grad[col_idx] = s_gamma_reduction[threadIdx.x];
  }
  /// END ASSIGN3_2
}

/**
@brief: ker_ln_bw_dinp
Layer norm backword kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {

  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute dxhat=dy*w with reinterpret_cast by casting to float4 for speedup
  // 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  // 3. Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  // 4. Compute final gradient

  int row_idx = blockIdx.x;

  float mean_val;
  if (means != nullptr) {
      mean_val = means[row_idx];
  } else {
      mean_val = 0.0f;
  }

  float variance_val = vars[row_idx];
  float inv_std_dev = rsqrtf(variance_val);

  // Step 1
  float local_dxhat_sum = 0.0f;
  float local_dxhat_xhat_sum = 0.0f;
  const float4 *inp_vec = reinterpret_cast<const float4 *>(inp) + row_idx * hidden_dim;
  const float4 *gamma_vec = reinterpret_cast<const float4 *>(gamma);

  const float4 *betta_vec;
  if (betta != nullptr) {
      betta_vec = reinterpret_cast<const float4 *>(betta);
  } else {
      betta_vec = nullptr;
  }

  const float4 *out_grad_vec = reinterpret_cast<const float4 *>(out_grad) + row_idx * hidden_dim;
  float4 *inp_grad_vec = reinterpret_cast<float4 *>(inp_grad) + row_idx * hidden_dim;

  for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    float4 grad_out_val = out_grad_vec[i];
    float4 gamma_val = gamma_vec[i];
    float4 inp_val = inp_vec[i];

    float4 betta_val;
    if (betta_vec != nullptr) {
      betta_val = betta_vec[i];
    } else {
      betta_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float4 dxhat;
    dxhat.x = grad_out_val.x * gamma_val.x;
    dxhat.y = grad_out_val.y * gamma_val.y;
    dxhat.z = grad_out_val.z * gamma_val.z;
    dxhat.w = grad_out_val.w * gamma_val.w;
    local_dxhat_sum += dxhat.x + dxhat.y + dxhat.z + dxhat.w;

    float4 xhat;
    if (means != nullptr) {
      xhat.x = (inp_val.x - mean_val) * inv_std_dev;
      xhat.y = (inp_val.y - mean_val) * inv_std_dev;
      xhat.z = (inp_val.z - mean_val) * inv_std_dev;
      xhat.w = (inp_val.w - mean_val) * inv_std_dev;
    } else {
      xhat.x = (inp_val.x - betta_val.x) / gamma_val.x;
      xhat.y = (inp_val.y - betta_val.y) / gamma_val.y;
      xhat.z = (inp_val.z - betta_val.z) / gamma_val.z;
      xhat.w = (inp_val.w - betta_val.w) / gamma_val.w;
    }

    local_dxhat_xhat_sum += dxhat.x * xhat.x + dxhat.y * xhat.y + dxhat.z * xhat.z + dxhat.w * xhat.w;
  }

  // Step 2
  blockReduce<ReduceType::kSum, 1>(&local_dxhat_sum);
  blockReduce<ReduceType::kSum, 1>(&local_dxhat_xhat_sum);

  __shared__ float shared_dxhat_sum;
  __shared__ float shared_dxhat_xhat_sum;

  if (threadIdx.x == 0) {
    shared_dxhat_sum = local_dxhat_sum;
    shared_dxhat_xhat_sum = local_dxhat_xhat_sum;
  }
  __syncthreads();

  // Step 3
  int hidden_dims = hidden_dim * 4;
  const float scale_factor = 1.0f / hidden_dims;

  for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    float4 grad_out_val = out_grad_vec[i];
    float4 gamma_val = gamma_vec[i];
    float4 inp_val = inp_vec[i];

    float4 betta_val;
    if (betta_vec != nullptr) {
      betta_val = betta_vec[i];
    } else {
      betta_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float4 dxhat;
    dxhat.x = grad_out_val.x * gamma_val.x;
    dxhat.y = grad_out_val.y * gamma_val.y;
    dxhat.z = grad_out_val.z * gamma_val.z;
    dxhat.w = grad_out_val.w * gamma_val.w;

    float4 xhat;
    if (means != nullptr) {
      float norm_factor = inv_std_dev;
      xhat.x = (inp_val.x - mean_val) * norm_factor;
      xhat.y = (inp_val.y - mean_val) * norm_factor;
      xhat.z = (inp_val.z - mean_val) * norm_factor;
      xhat.w = (inp_val.w - mean_val) * norm_factor;
    } else {
      xhat.x = (inp_val.x - betta_val.x) / gamma_val.x;
      xhat.y = (inp_val.y - betta_val.y) / gamma_val.y;
      xhat.z = (inp_val.z - betta_val.z) / gamma_val.z;
      xhat.w = (inp_val.w - betta_val.w) / gamma_val.w;
    }

    float4 final_grad;
    float scale_adjustment = shared_dxhat_sum * scale_factor;
    final_grad.x = (dxhat.x - (scale_adjustment + xhat.x * shared_dxhat_xhat_sum * scale_factor)) * inv_std_dev;
    final_grad.y = (dxhat.y - (scale_adjustment + xhat.y * shared_dxhat_xhat_sum * scale_factor)) * inv_std_dev;
    final_grad.z = (dxhat.z - (scale_adjustment + xhat.z * shared_dxhat_xhat_sum * scale_factor)) * inv_std_dev;
    final_grad.w = (dxhat.w - (scale_adjustment + xhat.w * shared_dxhat_xhat_sum * scale_factor)) * inv_std_dev;

    inp_grad_vec[i] = final_grad;
  }
  /// END ASSIGN3_2
}

extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp, const float *gamma,
                         const float *betta, const float *vars,
                         const float *means, int batch_size, int hidden_dim,
                         cudaStream_t stream_1, cudaStream_t stream_2) {

  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_betta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
  // The result is then multiplied by TILE_DIM to ensure that the grid size is a multiple of TILE_DIM.
  dim3 grid_dim(((hidden_dim + TILE_DIM - 1) / TILE_DIM) * TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);

  // Compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}}
}}
