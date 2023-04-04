# ------------------------------------------------------------------------------
# custom modules for partial sum accumulation
# ------------------------------------------------------------------------------

import math
import copy

import numpy as np
import numba

import torch
import torch.nn as nn


# --------------------------------------------------------------------
# Conv2d forward method with JIT compilation
# --------------------------------------------------------------------
@numba.njit
def np_forward_numba(
        in_arr,
        weights,
        padding=(0, 0),
        stride=(1, 1),
        #int_bias=0,
        psum_range=(-(1 << 31), (1 << 31) - 1),
        psum_num_ops=36,
        ):
    # extract input properties
    in_num, in_ch, in_height, in_width = in_arr.shape
    out_ch, w_ch, w_height, w_width = weights.shape
    kernel_size = (w_height, w_width)
    if in_ch != w_ch:
        msg = f'Wrong input channel; Expect {w_ch} but get {in_ch}'
        print('Error: ' + msg)
        return
        #raise ValueError(msg)

    # input operand size
    in_op_height = in_height + 2 * padding[0]
    in_op_width = in_width + 2 * padding[1]

    # initialize 4D output tensor (zeros)
    out_height = math.floor((in_op_height - (kernel_size[0] - 1) - 1) / stride[0]) + 1
    out_width = math.floor((in_op_width - (kernel_size[1] - 1) - 1) / stride[1]) + 1
    out_arr = np.empty((in_num, out_ch, out_height, out_width), dtype=np.int32)

    # input number iteration
    for num in range(in_num):
        # add zero padding to input tensor
        vertical_pad = np.zeros((in_ch, padding[0], in_width), dtype=np.int32)
        horizontal_pad = np.zeros((in_ch, in_op_height, padding[1]), dtype=np.int32)
        in_target = np.concatenate((
                horizontal_pad,
                np.concatenate((vertical_pad, in_arr[num], vertical_pad), axis=1),
                horizontal_pad,
            ), axis=2)

        # initialize 3D output tensor
        out_target = np.empty((out_ch, out_height, out_width), dtype=np.int32)

        in_target_4d = in_target.reshape(1, *in_target.shape)
        for i in range(0, out_height):
            for j in range(0, out_width):
                # weight indices
                w_i, w_j = i * stride[0], j * stride[1]

                # direct array per-entry product
                product_4d = weights * \
                             in_target_4d[:, :, w_i:w_i+w_height, w_j:w_j+w_width]

                # accumulation
                sum_arr = np.empty(out_ch)
                for k in range(out_ch):
                    vec = product_4d[k].flatten()
                    total_sum = 0
                    for ii in range(0, vec.size, psum_num_ops):
                        # add partial sum and clip range
                        total_sum = max(
                            min(
                                total_sum + sum(vec[ii:ii+psum_num_ops]),
                                psum_range[1]  # max value
                            ),
                            psum_range[0]  # min value
                        )
                    sum_arr[k] = total_sum

                # assign sum to 3D output array
                #out_target[:, i, j] = sum_arr + int_bias
                out_target[:, i, j] = sum_arr

        # assign result to 4D output array
        out_arr[num] = out_target

    return out_arr


# --------------------------------------------------------------------
# partial sum customized Conv2d
# --------------------------------------------------------------------
class PsumCustomConv2d(nn.Conv2d):
    def __init__(
            self, 
            in_channels, 
            out_channels,
            kernel_size, # tuple
            stride=(1, 1),
            padding=(0, 0),
            #int_bias=None,
            psum_bits=32,
            psum_num_ops=2,
            ):

        super().__init__(1, 1, 1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weights = np.ones((out_channels, in_channels, *kernel_size))
        #self.int_bias = int_bias if int_bias else np.zeros(out_channels)
        self.stride = stride
        self.padding = padding
        self.psum_range = (-(1 << (psum_bits - 1)), (1 << (psum_bits - 1)) - 1)
        self.psum_num_ops = psum_num_ops

    def forward(self, in_arr):
        return np_forward_numba(
            in_arr,
            self.weights,
            padding=self.padding,
            stride=self.stride,
            #int_bias=self.bias,
            psum_range=self.psum_range,
            psum_num_ops=self.psum_num_ops,
        )


# --------------------------------------------------------------------
# custom quantized Conv2d
# --------------------------------------------------------------------
class EmuQuantConv2d(nn.Conv2d):
    def __init__(self, ref_conv2d, psum_bits=32, psum_num_ops=36):
        super(EmuQuantConv2d, self).__init__(1, 1, 1, 1)  # not used

        self.weight_scale = 1.
        self.act_scale = 1.
        self.conv = PsumCustomConv2d(
            ref_conv2d.in_channels, 
            ref_conv2d.out_channels, 
            ref_conv2d.kernel_size, 
            stride=ref_conv2d.stride, 
            padding=ref_conv2d.padding,
            psum_bits=psum_bits,
            psum_num_ops=psum_num_ops
            )
        self.sep_bias = None  # separated bias
        
    def set_params(self, weight, bias, weight_scale, act_scale):
        self.conv.weights = np.array(weight).astype(np.int32)
        self.sep_bias = torch.tensor(bias).reshape(1, -1, 1, 1)
        self.weight_scale = weight_scale
        self.act_scale = act_scale

    def forward(self, x):
        x = torch.round(x / self.act_scale).numpy().astype(np.int32)
        x = self.conv(x)
        x = torch.add(
            torch.from_numpy(x) * self.act_scale * self.weight_scale, 
            self.sep_bias
            )
        return x


# --------------------------------------------------------------------
# custom AlexNet
# --------------------------------------------------------------------
class CustomAlexnet(nn.Module):
    def __init__(self, ref_model, psum_bits=32, psum_num_ops=36):
        super().__init__()
        self.custom_model = copy.deepcopy(ref_model)
        self.psum_bits = psum_bits
        self.psum_num_ops = psum_num_ops  # number of operands

    def forward(self, x):
        x = self.custom_model(x)
        return x

    def customize(self, params):
        for i, li in enumerate(params['layer_indices']):  # li: layer index
            new_layer = EmuQuantConv2d(
                self.custom_model.features[li],
                psum_bits=self.psum_bits,
                psum_num_ops=self.psum_num_ops,
                )
            new_layer.set_params(
                params['weights'][i],
                params['biases'][i],
                params['weight_scales'][i],
                params['act_scales'][i],
            )
            self.custom_model.features[li] = new_layer
