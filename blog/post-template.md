<!-- #!ftulabs-scripts
title: Scaling Transformer Inference with Custom CUDA Kernels
description: This is a post template description.
date: 2026-03-15
authors: Minh Tran, Alex Rivera
readtime: 12 min
lang: en
-->

![Just a placeholder image](/img/1.ftu-ai-2.jpg)

When we started optimizing inference for FTU-7B, our initial benchmarks showed that standard PyTorch attention was the primary bottleneck — accounting for roughly 60% of total inference time on A100 GPUs. This post describes how we wrote custom CUDA kernels to achieve a 3x speedup, and the lessons we learned along the way.

## The Problem: Memory-Bound Attention

Transformer attention is fundamentally a memory-bound operation. The standard implementation materializes the full N×N attention matrix, requiring O(N²) memory and bandwidth. For long sequences (4K+ tokens), this becomes the dominant cost — not because of compute, but because of how data moves between GPU memory hierarchies.

Our profiling revealed that the standard attention path spent most of its time waiting for memory transfers rather than doing useful computation. The arithmetic intensity was well below what the A100 can sustain:

```text
# Profiling results (A100 80GB, batch=1, seq_len=4096)
Standard Attention:     142ms  (memory bandwidth: 1.2 TB/s)
cuBLAS GEMM (Q×K):      38ms
Softmax:                 24ms
cuBLAS GEMM (A×V):      36ms
Memory allocation:       44ms  ← the hidden cost
```

That 44ms spent on memory allocation alone was a red flag. Every forward pass was allocating and deallocating temporary buffers for the attention matrix — a pattern that's optimized away in Flash Attention but present in many standard implementations.

## Our Approach: Fused Tiled Attention

Rather than computing attention in separate GEMM → softmax → GEMM stages, we fuse everything into a single kernel that operates on tiles of the input. This is conceptually similar to [Flash Attention](#), but with several modifications specific to our model architecture:

* Grouped-query attention (GQA) — FTU-7B uses 8 KV heads shared across 32 query heads, requiring careful shared memory management

* RoPE integration — We compute rotary position embeddings inside the kernel rather than as a separate pre-processing step

* Online softmax with FP32 accumulators — Maintaining numerical stability while keeping the data path in FP16/BF16

## Implementation Details

The core idea is to tile the computation so that each thread block processes a block of queries against all key-value pairs, accumulating the output in shared memory:

```cpp
// Simplified kernel structure (actual implementation is ~800 lines)
__global__ void fused_attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int seq_len,
    const int head_dim
) {
    // Each block handles BLOCK_M query positions
    const int q_start = blockIdx.x * BLOCK_M;

    // Shared memory for Q, K, V tiles
    __shared__ half smem_q[BLOCK_M][HEAD_DIM];
    __shared__ half smem_k[BLOCK_N][HEAD_DIM];
    __shared__ half smem_v[BLOCK_N][HEAD_DIM];

    // Online softmax state
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc[BLOCK_M][HEAD_DIM] = {0};

    // Iterate over K,V blocks
    for (int kv_start = 0; kv_start < seq_len; kv_start += BLOCK_N) {
        load_tile(K, smem_k, kv_start);
        load_tile(V, smem_v, kv_start);
        __syncthreads();

        // Compute QK^T, apply RoPE, scale, mask
        // Online softmax update
        // Accumulate O = softmax(QK^T) * V
    }

    store_output(O, acc, row_sum, q_start);
}
```

We compile the kernel ahead of time to avoid JIT overhead in production:

```bash
# Compile for A100 (SM 8.0) and H100 (SM 9.0)
nvcc -O3 -arch=sm_80 -gencode=arch=compute_90,code=sm_90 \
     --use_fast_math -lineinfo \
     -o fused_attn.so --shared fused_attention.cu

# Run the benchmark suite
python bench.py --kernel fused --seq-lengths 2048 4096 8192 16384
```

## Results

After several iterations of tuning tile sizes, shared memory layout, and warp scheduling, we achieved significant improvements across all sequence lengths:

> The key insight was not algorithmic — it was about understanding the GPU memory hierarchy deeply enough to keep the SMs fed with data at every cycle.

On our standard benchmark (A100 80GB, BF16, batch size 1):

* Seq 2048: 2.1x speedup (68ms → 32ms)

* Seq 4096: 2.8x speedup (142ms → 51ms)

* Seq 8192: 3.2x speedup (412ms → 129ms)

* Seq 16384: 3.4x speedup (1.6s → 470ms)

## Packaging for Deployment

To keep the inference environment reproducible across machines, we containerize everything with a multi-stage Docker build:

```dockerfile
# Stage 1: compile the CUDA kernel
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder
WORKDIR /build
COPY kernels/ ./kernels/
RUN nvcc -O3 -arch=sm_80 --use_fast_math \
        -o fused_attn.so --shared kernels/fused_attention.cu

# Stage 2: slim runtime image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/fused_attn.so /app/lib/
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY src/ ./src/

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t ftu-inference .
docker run --gpus all -p 8000:8000 ftu-inference
```

## Lessons Learned

Writing CUDA kernels is as much art as science. Here are the non-obvious lessons from this project:

1. Profile before you optimize. Our initial assumption was that the GEMM operations were the bottleneck. They weren't — memory allocation and data layout were.

2. Shared memory bank conflicts are real. A 2-line change to our memory layout (adding padding to avoid bank conflicts) gave us a 15% speedup.

3. Test numerics obsessively. FP16 accumulation introduced subtle errors that only manifested at long sequence lengths. FP32 accumulators are non-negotiable.

The full kernel implementation is available in our [FlashKernel repository](#). We welcome contributions and benchmarks on other GPU architectures.
