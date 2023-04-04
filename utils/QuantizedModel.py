# ------------------------------------------------------------------------------
# Quantized AlexNet Module
# ------------------------------------------------------------------------------

import pickle

import torch
import torch.nn as nn


class QuantizedModel(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = model_fp32

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x

    def save_params(self, file_path):
        params = {
            'layer_indices': [],
            'weights': [],
            'biases': [],
            'weight_scales': [],
            'act_scales': [],
        }

        params['act_scales'].append(self.quant.scale.item())

        for i, layer in enumerate(self.model_fp32.features):
            if type(layer) == torch.ao.nn.quantized.modules.conv.Conv2d:
                params['layer_indices'].append(i)
                params['weights'].append(layer.weight().int_repr().tolist())
                params['biases'].append(layer.bias().tolist())
                params['weight_scales'].append(layer.weight().q_scale())
                params['act_scales'].append(layer.scale)

        params['act_scales'].pop()  # scale of last layer is not used

        with open(file_path, 'wb') as file:
            pickle.dump(params, file)
