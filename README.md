# Neural Network Quantization

## Notebooks

AlexNet quantization (notebook version): 
https://colab.research.google.com/drive/1ulFUy6sP2M20fw2MsIms9DtZD9DcE4Yl?usp=sharing

## Results

### Configuration 1

- Activations
    - Histogram observer
    - UINT8
    - Symmetric
    - Per tensor
- Weights
    - Histogram observer
    - INT8
    - Symmetric
    - Per tensor

Post-training quantization results on CIFAR10:

| Model                    | AlexNet | VGG16   |
| ------------------------ | ------- | ------- |
| Original model accuracy  | 83.24 % | 89.55 % |
| Quantized model accuracy | 82.33 % | 89.52 % |
| Accuracy loss            | 0.91 %  | 0.03 %  |

Post-training quantization results on CIFAR100:

| Model                    | AlexNet |         | VGG16   |         |
| ------------------------ | ------- | ------- | ------- | ------- |
| Top k                    | Top 1   | Top 5   | Top 1   | Top 5   |
| Original model accuracy  | 67.29 % | 91.43 % | 74.81 % | 94.55 % |
| Quantized model accuracy | 64.20 % | 89.45 % | 70.58 % | 93.07 % |
| Accuracy loss            | 3.09 %  | 1.98 %  | 4.23 %  | 1.48 %  |

Post-training quantization results on ImageNet:

| Model                    | AlexNet  |          | VGG16    |          |
| ------------------------ | -------- | -------- | -------- | -------- |
| Top k                    | Top 1    | Top 5    | Top 1    | Top 5    |
| Original model accuracy  | 56.556 % | 79.084 % | 71.586 % | 90.390 % |
| Quantized model accuracy | 54.206 % | 77.758 % | 70.138 % | 89.560 % |
| Accuracy loss            | 2.350 %  | 1.326 %  | 1.448 %  | 0.830 %  |

Calibration data: ImageNet training task 3

## References

- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
- [AlexNet implementation with PyTorch](https://github.com/Lornatang/AlexNet-PyTorch)
- [PyTorch static quantization example](https://leimao.github.io/blog/PyTorch-Static-Quantization/)
