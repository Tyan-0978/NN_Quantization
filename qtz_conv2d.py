# ------------------------------------------------------------------------------
# conv2d with output accumulation bit width control
# ------------------------------------------------------------------------------

import itertools
import torch
import torch.nn as nn
from tqdm.auto import tqdm

class qtz_conv2d:
  def __init__(
    self,
    in_channels,
    out_channels,
    kernel_size, # tuple
    stride=1,
    padding=0,
    bias=0,
    acc_bit_width=32,
  ):
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.weights = torch.ones(out_channels, in_channels, *kernel_size)
    self.bias = bias
    self.stride = stride
    self.padding = padding
    self.acc_bit_width = acc_bit_width

  def qtz_forward(self, in_tensor):
    acc_max = (1 << (self.acc_bit_width - 1)) - 1
    acc_min = - (1 << (self.acc_bit_width - 1))

    in_num, in_channel, in_height, in_width = in_tensor.size()
    if in_channel != self.out_channels:
      print('Error: wrong input channel number')
      return

    out_height = in_height + 2 * self.padding - self.kernel_size[0] + 1
    out_width = in_width + 2 * self.padding - self.kernel_size[1] + 1
    out_tensor = torch.zeros(in_num, self.out_channels, out_height, out_width)

    # input iteration
    for num in range(in_num):
      # zero padding
      vertical_pad = torch.zeros(in_channel, self.padding, in_width)
      horizontal_pad = torch.zeros(in_channel, in_height + 2 * self.padding, self.padding)
      in_target = torch.cat((
        horizontal_pad,
        torch.cat((vertical_pad, in_tensor[num], vertical_pad), axis=1),
        horizontal_pad
      ), axis=2)
      out_target = torch.empty(self.out_channels, out_height, out_width)

      # weight iteration
      for ch in range(self.out_channels):
        out_acc = torch.empty(out_height, out_width)
        weight_tensor = self.weights[ch]
        # convolution (sliding window)
        for i, j in itertools.product(range(0, out_height, self.stride), range(0, out_width, self.stride)):
          # inner product
          _, h, w = weight_tensor.size()
          product_2d = weight_tensor * in_target[:, i:i+h, j:j+w]
          # accumulation
          out_acc_sum = 0
          for n in torch.flatten(product_2d):
            out_acc_sum += n.item()
            # output accumulation bitwidth control
            out_acc_sum = max(min(acc_max, out_acc_sum), acc_min)
          out_acc[i][j] = out_acc_sum

        out_target[ch] = out_acc + self.bias

      out_tensor[num] = out_target

    return out_tensor

if __name__ == '__main__':
  # test difference of qtz_conv2d and nn.Conv2d
  qconv = qtz_conv2d(
    256, 256, (3, 3),
    stride=1,
    padding=1,
    acc_bit_width=32
  )

  conv = nn.Conv2d(
    256, 256,
    kernel_size=(3, 3),
    stride=1,
    padding=(1, 1),
    dtype=torch.float32
  )

  # assign random weights
  rand_weight = torch.randint(0, 256, (256, 256, 3, 3))
  qconv.weights = rand_weight
  qconv.bias = 0
  conv.weight.data = rand_weight.type(torch.float32)
  conv.bias.data.fill_(0)

  # test random inputs
  num_tests = 100
  in_size = (1, 256, 3, 3)
  err_count = 0
  for i in tqdm(range(num_tests)):
    rand_in = torch.randint(-128, 128, in_size)
    q_out = qconv.qtz_forward(rand_in)
    conv_out = conv(rand_in.type(torch.float32))
    avg_err = torch.mean(torch.abs(q_out - conv_out))
    if avg_err != 0:
      print(f'Average error: {avg_err} on test {i}')
      err_count += 1

  print(f'Number of passed tests: {num_tests - err_count} / {num_tests}')
