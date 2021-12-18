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