import jittor as jt
import os

header = """
#pragma once
#include "op.h"

namespace jittor {

struct FusedBiasActOp : Op {
    Var *output;
    Var *input, *bias, *refer;
    int act, grad;
    float alpha, scale;
    FusedBiasActOp(Var* input, Var* bias, Var* refer, int act, int grad, double alpha, double scale);

    const char* name() const override { return "fused_bias_act"; }
    DECLARE_jit_run;
};

} // jittor
"""

src = """
#include "var.h"
#include "fused_bias_act_op.h"

namespace jittor {
#ifndef JIT
FusedBiasActOp::FusedBiasActOp(Var* input, Var* bias, Var* refer, int act, int grad, double alpha, double scale) : input(input), bias(bias), refer(refer),act(act), grad(grad), alpha(alpha), scale(scale) {
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(input->shape, input->dtype());
}


void FusedBiasActOp::jit_prepare(JK& jk) {
    jk << _CS("[To:") << output->dtype();
    jk << _CS("][Ti:") << input->dtype();
    jk << _CS("][Tb:") << bias->dtype();
    jk << _CS("][Tr:") << refer->dtype() <<']';
}

#else // JIT

template <typename scalar_t>
__global__ void kernel(scalar_t* out, const scalar_t* p_x, const scalar_t* p_b, const scalar_t* p_ref, int act, int grad, scalar_t alpha, scalar_t scale, int loop_x, int size_x, int step_b, int size_b, int use_bias, int use_ref) {
    int xi = blockIdx.x * loop_x * blockDim.x + threadIdx.x;

    scalar_t zero = 0.0;

    for (int loop_idx = 0; loop_idx < loop_x && xi < size_x; loop_idx++, xi += blockDim.x) {
        scalar_t x = p_x[xi];

        if (use_bias) {
            x += p_b[(xi / step_b) % size_b];
        }

        scalar_t ref = use_ref ? p_ref[xi] : zero;

        scalar_t y;

        switch (act * 10 + grad) {
            default:
            case 10: y = x; break;
            case 11: y = x; break;
            case 12: y = 0.0; break;

            case 30: y = (x > 0.0) ? x : x * alpha; break;
            case 31: y = (ref > 0.0) ? x : x * alpha; break;
            case 32: y = 0.0; break;
        }

        out[xi] = y * scale;
    }
}

void FusedBiasActOp::jit_run() {
    int curDevice = -1;
    cudaGetDevice(&curDevice);

    auto x = input;
    auto b = bias;
    auto ref = refer;

    int use_bias = b->numel() ? 1 : 0;
    int use_ref = ref->numel() ? 1 : 0;

    int size_x = x->numel();
    int size_b = b->numel();
    int step_b = 1;

    for (int i = 1 + 1; i < x->shape.size(); i++) {
        step_b *= x->shape[i];
    }

    int loop_x = 4;
    int block_size = 4 * 32;
    int grid_size = (size_x - 1) / (loop_x * block_size) + 1;

    auto y = new Var(x->shape, x->dtype());

    kernel<<<grid_size, block_size, 0>>>(
        y->ptr<float32>(),
        x->ptr<float32>(),
        b->ptr<float32>(),
        ref->ptr<float32>(),
        act,
        grad,
        alpha,
        scale,
        loop_x,
        size_x,
        step_b,
        size_b,
        use_bias,
        use_ref
    );

    output = y;
}
#endif // JIT

} // jittor
"""
if jt.compiler.has_cuda:
    jt.flags.use_cuda = 1

fused_bias_act_op = jt.compile_custom_op(header, src, "fused_bias_act", warp=False)

input = jt.Var([[[[4.1289e-01, 8.6104e-01, 4.1847e-01, 4.8154e-01, 5.4599e-01],
          [1.9763e-01, 3.0711e-01, 9.0967e-01, 8.1241e-01, 2.2268e-01],
          [3.7746e-01, 3.3018e-01, 4.9468e-01, 7.2670e-01, 4.6834e-01],
          [7.2801e-01, 2.8106e-01, 8.5124e-01, 5.2168e-01, 4.2136e-01]],

         [[7.2284e-01, 8.8035e-01, 7.1085e-02, 9.3986e-01, 1.0058e-01],
          [1.4385e-01, 5.8178e-02, 7.2018e-01, 4.7816e-01, 4.0098e-02],
          [9.9472e-01, 7.8700e-01, 6.7646e-01, 2.3609e-01, 8.6025e-01],
          [1.5367e-01, 6.8150e-01, 4.0920e-01, 9.9816e-01, 6.1706e-01]],

         [[4.4694e-01, 8.1519e-01, 4.1261e-01, 7.7234e-01, 4.2843e-01],
          [1.8978e-01, 2.2789e-01, 5.4615e-01, 9.4389e-02, 3.1061e-01],
          [2.9517e-01, 3.5345e-01, 9.1983e-01, 3.8440e-01, 4.6680e-01],
          [6.0278e-01, 6.6551e-02, 4.7572e-01, 2.7459e-01, 1.0970e-01]]],


        [[[3.1318e-01, 1.2000e-04, 3.3605e-01, 6.0732e-01, 1.9992e-01],
          [6.4791e-01, 3.8084e-01, 2.7621e-01, 8.6844e-02, 9.5495e-01],
          [5.2105e-01, 2.7782e-01, 6.8674e-02, 7.9531e-01, 8.4269e-01],
          [1.2383e-01, 5.0543e-01, 4.3160e-01, 6.6576e-01, 6.0647e-01]],

         [[8.1410e-01, 5.7513e-01, 6.4728e-01, 7.8244e-01, 3.3586e-01],
          [3.9313e-01, 5.4360e-01, 7.0434e-01, 2.2576e-01, 1.0376e-01],
          [7.6051e-01, 9.1371e-01, 5.3020e-01, 1.4286e-01, 1.2928e-01],
          [1.3396e-01, 7.3906e-01, 3.0547e-01, 2.2090e-03, 3.7084e-02]],

         [[3.9402e-01, 8.3073e-01, 7.9441e-01, 6.4661e-01, 4.8639e-01],
          [6.7884e-01, 9.6930e-01, 2.1589e-01, 3.4418e-01, 9.2157e-01],
          [9.7674e-01, 7.9746e-01, 5.7921e-01, 5.2418e-02, 9.6114e-01],
          [8.7905e-01, 8.9969e-01, 1.8315e-01, 8.4478e-01, 7.6681e-01]]]],)

bias = jt.Var([0.6415, 0.8838, 0.5172],)
empty = jt.rand((0,), dtype=float)
negative_slope = 0.2
scale = 1.5
out = fused_bias_act_op(input, bias, empty, 3, 1, negative_slope, scale).fetch_sync()
print(out)