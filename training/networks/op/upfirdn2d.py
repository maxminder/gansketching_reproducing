import os
import jittor as jt

jt.flags.use_cuda=jt.has_cuda

module_path = os.path.dirname(__file__)
src = os.path.join(module_path, "upfirdn2d_op.cc")
header = os.path.join(module_path, "upfirdn2d_op.h")
upfirdn2d_op = jt.compile_custom_ops((src, header))


class UpFirDn2dBackward(jt.Function):
    def execute(
        self, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):

        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d_op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        self._kernel = kernel

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        self.up_x = up_x
        self.up_y = up_y
        self.down_x = down_x
        self.down_y = down_y
        self.pad_x0 = pad_x0
        self.pad_x1 = pad_x1
        self.pad_y0 = pad_y0
        self.pad_y1 = pad_y1
        self.in_size = in_size
        self.out_size = out_size

        return grad_input

    def grad(self, gradgrad_input):
        kernel = self._kernel

        gradgrad_input = gradgrad_input.reshape(-1, self.in_size[2], self.in_size[3], 1)

        gradgrad_out = upfirdn2d_op.upfirdn2d(
            gradgrad_input,
            kernel,
            self.up_x,
            self.up_y,
            self.down_x,
            self.down_y,
            self.pad_x0,
            self.pad_x1,
            self.pad_y0,
            self.pad_y1,
        )
        gradgrad_out = gradgrad_out.view(
            self.in_size[0], self.in_size[1], self.out_size[0], self.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(jt.Function):
    def execute(self, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        self.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)

        self._kernel = kernel
        self._grad_kernel = jt.flip(kernel, [0, 1])

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        self.out_size = (out_h, out_w)

        self.up = (up_x, up_y)
        self.down = (down_x, down_y)
        self.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        self.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d_op.upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )

        out = out.view(-1, channel, out_h, out_w)

        return out

    def grad(self, grad_output):
        kernel = self._kernel
        grad_kernel = self._grad_kernel

        grad_input = UpFirDn2dBackward.apply(
            grad_output,
            kernel,
            grad_kernel,
            self.up,
            self.down,
            self.pad,
            self.g_pad,
            self.in_size,
            self.out_size,
        )

        return grad_input, None, None, None, None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if jt.has_cuda:  # cuda版本
        if jt.flags.use_cuda == 0:
            jt.flags.use_cuda = 1
        out = UpFirDn2d.apply(
            input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
        )
    else:  # cpu版本
        out = upfirdn2d_native(
            input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
        )
    return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = jt.nn.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = jt.nn.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = jt.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = jt.nn.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)
