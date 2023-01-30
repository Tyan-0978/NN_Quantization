# Neural Network Quantization

## Notebooks

AlexNet quantization (notebook version): 
https://colab.research.google.com/drive/1ulFUy6sP2M20fw2MsIms9DtZD9DcE4Yl?usp=sharing

## Results

Quantization results on CIFAR10:

| Model                 | AlexNet |   VGG16 |
| --------------------- | -------:| -------:|
| Original accuracy     | 83.24 % | 89.55 % |
| Quantization accuracy | 82.33 % | 89.52 % |
| Accuracy loss         |  0.91 % |  0.03 % |

Quantized results on CIFAR100:

| Model                    | AlexNet |   VGG16 |
| ------------------------ | -------:| -------:|
| Original top 1 accuracy  | 67.29 % | 74.81 % |
| Original top 5 accuracy  | 91.43 % | 94.55 % |
| Quantized top 1 accuracy | 64.20 % | 70.58 % |
| Quantized top 5 accuracy | 89.45 % | 93.07 % |
| Top 1 accuracy loss      |  3.09 % |  4.23 % |
| Top 5 accuracy loss      |  1.98 % |  1.48 % |

## References

- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
- [AlexNet implementation with PyTorch](https://github.com/Lornatang/AlexNet-PyTorch)
- [PyTorch static quantization example](https://leimao.github.io/blog/PyTorch-Static-Quantization/)
