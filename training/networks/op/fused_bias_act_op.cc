#include "var.h"
#include "fused_bias_act_op.h"
#include <cuda_runtime.h>

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

    auto x = input;
    auto b = bias;
    auto ref = refer;

    int use_bias = b->numel() ? 1 : 0;
    int use_ref = ref->numel() ? 1 : 0;

    int size_x = x->numel();
    int size_b = b->numel();
    int size_ref = ref->numel();
    int step_b = 1;

    for (int i = 1 + 1; i < x->shape.size(); i++) {
        step_b *= x->shape[i];
    }

    int loop_x = 4;
    int block_size = 4 * 32;
    int grid_size = (size_x - 1) / (loop_x * block_size) + 1;

    auto* __restrict__ yp = output->ptr<float32>();
    auto* __restrict__ xp = x->ptr<float32>();
    auto* __restrict__ bp = b->ptr<float32>();
    auto* __restrict__ refp = ref->ptr<float32>();

    kernel<<<grid_size, block_size, 0>>>(
        yp,
        xp,
        bp,
        refp,
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
}
#endif // JIT

} // jittor