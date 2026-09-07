"""AlexNet - ILSVRC 2012 winner (top-5 error 15.3%).

Krizhevsky, Sutskever, Hinton. "ImageNet Classification with Deep
Convolutional Neural Networks." NIPS 2012.

Paper-faithful single-tower version of the original 2-GPU model:
  - ReLU instead of tanh (much faster convergence, Sec 3.1)
  - Local Response Normalization after conv1/conv2 (Sec 3.3)
  - Overlapping max pooling: 3x3 window, stride 2 (Sec 3.4)
  - Dropout 0.5 on the two hidden FC layers (Sec 4.2)

Input is 227x227, not the 224x224 the paper's text says: 224 does not
divide evenly through conv1 (11x11, stride 4), while (227-11)/4+1 = 55,
the 55x55 map the paper reports. torchvision instead keeps 224 and pads
conv1 by 2 - same idea, different fudge.
"""

import torch
import torch.nn as nn


def _lrn() -> nn.Module:
    return nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            # conv1: 227 -> 55, pool -> 27
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            _lrn(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv2: 27 -> 27, pool -> 13
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            _lrn(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv3/4/5: 13 -> 13, no pooling in between
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Sec 5: weights ~ N(0, 0.01), biases 0 - except conv2/conv4/conv5
        and the FC layers, whose biases start at 1 so their ReLUs see
        positive inputs early in training."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        for i in (4, 10, 12):  # conv2, conv4, conv5
            nn.init.ones_(self.features[i].bias)
        for i in (1, 4, 6):  # all three FC layers
            nn.init.ones_(self.classifier[i].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


def alexnet(**kwargs) -> AlexNet:
    return AlexNet(**kwargs)


if __name__ == "__main__":
    m = alexnet()
    assert m(torch.randn(2, 3, 227, 227)).shape == (2, 1000)

    # feature map sizes must match the paper's Fig 2
    sizes, x = [], torch.randn(1, 3, 227, 227)
    for layer in m.features:
        x = layer(x)
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            sizes.append(tuple(x.shape[1:]))
    assert sizes[0] == (96, 55, 55), sizes[0]
    assert sizes[-1] == (256, 6, 6), sizes[-1]
    print("alexnet ok  params=%.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
    print("  maps:", sizes)
