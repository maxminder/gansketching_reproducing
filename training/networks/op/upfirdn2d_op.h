#pragma once
#include "op.h"

namespace jittor {

struct Upfirdn2dOp : Op {
    Var *output;
    Var *input, *kernel;
    int up_x, up_y;
    int down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1;
    Upfirdn2dOp(Var* input, Var* kernel, int up_x, int up_y, int down_x, int down_y, int pad_x0, int pad_x1, int pad_y0, int pad_y1);

    const char* name() const override { return "upfirdn2d"; }
    DECLARE_jit_run;
};

} // jittor