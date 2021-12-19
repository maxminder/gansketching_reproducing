import os

# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.autograd import Function
#from torch.utils.cpp_extension import load
import jittor as jt

# module_path = os.path.dirname(__file__)
# fused = load(
#     "fused",
#     sources=[
#         os.path.join(module_path, "fused_bias_act.cpp"),
#         os.path.join(module_path, "fused_bias_act_kernel.cu"),
#     ],
# )


# class FusedLeakyReLUFunctionBackward(jt.Function):
#     @staticmethod
#     def execute(ctx, grad_output, out, bias, negative_slope, scale):
#         ctx.save_for_backward(out)
#         ctx.negative_slope = negative_slope
#         ctx.scale = scale

#         empty = grad_output.new_empty(0)

#         grad_input = fused.fused_bias_act(
#             grad_output, empty, out, 3, 1, negative_slope, scale
#         )

#         dim = [0]

#         if grad_input.ndim > 2:
#             dim += list(range(2, grad_input.ndim))

#         if bias:
#             grad_bias = grad_input.sum(dim).detach()

#         else:
#             grad_bias = empty

#         return grad_input, grad_bias

#     @staticmethod
#     def backward(ctx, gradgrad_input, gradgrad_bias):
#         out, = ctx.saved_tensors
#         gradgrad_out = fused.fused_bias_act(
#             gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
#         )

#         return gradgrad_out, None, None, None, None


# class FusedLeakyReLUFunction(jt.Function):
#     # @staticmethod
#     def execute(self, input, bias, negative_slope, scale):
#         # empty = input.new_empty(0)
#         empty = jt.randn((0))
#         self.bias = bias is not None

#         if bias is None:
#             bias = empty

#         out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
#         self.save_for_backward(out)
#         self.negative_slope = negative_slope
#         self.scale = scale

#         return out

#     # @staticmethod
#     def grad(self, grad_output):
#         out, = self.saved_tensors

#         grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
#             grad_output, out, self.bias, self.negative_slope, self.scale
#         )

#         if not self.bias:
#             grad_bias = None

#         return grad_input, grad_bias, None, None


class FusedLeakyReLU(jt.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        if bias:
            self.bias = jt.zeros(channel)
        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def execute(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        # rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return jt.nn.leaky_relu(input + bias.view((1, bias.shape[0])), scale=0.2) * scale
    else:
        return jt.nn.leaky_relu(input, scale=0.2) * scale
