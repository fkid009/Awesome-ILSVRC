"""GoogLeNet / Inception-v1 - ILSVRC 2014 winner (top-5 error 6.67%).

Szegedy et al. "Going Deeper with Convolutions." CVPR 2015.

Two ideas worth the study time:

1. Inception module (Fig 2b). Instead of picking one kernel size per layer,
   run 1x1, 3x3, 5x5 and a pooling branch in parallel and concatenate the
   results, letting the net choose its own mix of scales. Naive parallel
   branches explode in cost, so 1x1 "reduce" convs shrink the channel depth
   *before* the expensive 3x3/5x5 - that is the entire trick.

2. 12x fewer parameters than AlexNet (~6.8M vs ~60M) despite 22 weight
   layers, because global average pooling replaces the fat FC layers
   (borrowed from Network-in-Network). Only one 1024->1000 FC remains.

Two auxiliary classifiers hang off inception 4a/4d during training to push
gradient into the middle of the net (Sec 5); their losses are weighted 0.3
and they are discarded at inference. Later BN-Inception showed the real fix
was normalisation, not aux heads.
"""

import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    """conv -> ReLU. The 2014 paper had no BatchNorm (that is Inception-v2),
    but the standard reimplementation adds it; keep it switchable."""

    def __init__(self, in_ch: int, out_ch: int, batch_norm: bool = False, **conv_kw):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias=not batch_norm, **conv_kw)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3) if batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Inception(nn.Module):
    """Fig 2(b). Channel counts follow Table 1's columns:
    #1x1, #3x3reduce, #3x3, #5x5reduce, #5x5, poolproj."""

    def __init__(self, in_ch, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj,
                 batch_norm: bool = False):
        super().__init__()
        conv = lambda i, o, **kw: BasicConv2d(i, o, batch_norm, **kw)
        self.branch1 = conv(in_ch, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv(in_ch, ch3x3red, kernel_size=1),          # dimension reduction
            conv(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            conv(in_ch, ch5x5red, kernel_size=1),
            conv(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            # stride-1 pooling keeps the spatial size so branches can concat
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv(in_ch, pool_proj, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([b(x) for b in
                          (self.branch1, self.branch2, self.branch3, self.branch4)], 1)


class InceptionAux(nn.Module):
    """Sec 5 auxiliary head: 5x5/3 avgpool -> 1x1 conv 128 -> FC 1024 -> FC 1000,
    dropout 0.7."""

    def __init__(self, in_ch: int, num_classes: int, batch_norm: bool = False):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)  # 14x14 -> 4x4
        self.conv = BasicConv2d(in_ch, 128, batch_norm, kernel_size=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.pool(x))
        x = self.relu(self.fc1(torch.flatten(x, 1)))
        return self.fc2(self.dropout(x))


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int = 1000, aux_logits: bool = True,
                 batch_norm: bool = False, dropout: float = 0.4):
        super().__init__()
        self.aux_logits = aux_logits
        conv = lambda i, o, **kw: BasicConv2d(i, o, batch_norm, **kw)
        inception = lambda *a: Inception(*a, batch_norm=batch_norm)
        # ceil_mode matches the original Caffe pooling: 112 -> 56, not 55
        pool = lambda: nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # LRN as in AlexNet; a no-op once BatchNorm is on
        lrn = (lambda: nn.Identity()) if batch_norm else (lambda: nn.LocalResponseNorm(5))

        # stem: 224 -> 56x56x192 (Table 1, rows 1-4)
        self.stem = nn.Sequential(
            conv(3, 64, kernel_size=7, stride=2, padding=3),  # -> 112
            pool(),                                           # -> 56
            lrn(),
            conv(64, 64, kernel_size=1),                      # 3x3 reduce
            conv(64, 192, kernel_size=3, padding=1),
            lrn(),
            pool(),                                           # -> 28
        )
        self.inception3a = inception(192, 64, 96, 128, 16, 32, 32)    # -> 256
        self.inception3b = inception(256, 128, 128, 192, 32, 96, 64)  # -> 480
        self.pool3 = pool()                                           # -> 14
        self.inception4a = inception(480, 192, 96, 208, 16, 48, 64)   # -> 512
        self.inception4b = inception(512, 160, 112, 224, 24, 64, 64)  # -> 512
        self.inception4c = inception(512, 128, 128, 256, 24, 64, 64)  # -> 512
        self.inception4d = inception(512, 112, 144, 288, 32, 64, 64)  # -> 528
        self.inception4e = inception(528, 256, 160, 320, 32, 128, 128)  # -> 832
        self.pool4 = pool()                                           # -> 7
        self.inception5a = inception(832, 256, 160, 320, 32, 128, 128)  # -> 832
        self.inception5b = inception(832, 384, 192, 384, 48, 128, 128)  # -> 1024

        self.aux1 = InceptionAux(512, num_classes, batch_norm) if aux_logits else None
        self.aux2 = InceptionAux(528, num_classes, batch_norm) if aux_logits else None

        # global average pooling instead of an FC stack - the parameter saver
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor):
        """Returns logits, or (logits, aux2, aux1) while training with aux heads.
        Total loss = main + 0.3 * aux2 + 0.3 * aux1."""
        x = self.stem(x)
        x = self.inception3b(self.inception3a(x))
        x = self.inception4a(self.pool3(x))
        aux1 = self.aux1(x) if self.aux1 is not None and self.training else None
        x = self.inception4d(self.inception4c(self.inception4b(x)))
        aux2 = self.aux2(x) if self.aux2 is not None and self.training else None
        x = self.pool4(self.inception4e(x))
        x = self.inception5b(self.inception5a(x))
        x = self.dropout(torch.flatten(self.avgpool(x), 1))
        logits = self.fc(x)
        return (logits, aux2, aux1) if aux1 is not None else logits


def googlenet(**kw) -> GoogLeNet:
    return GoogLeNet(**kw)


if __name__ == "__main__":
    m = googlenet()

    m.train()
    logits, aux2, aux1 = m(torch.randn(2, 3, 224, 224))
    assert logits.shape == aux1.shape == aux2.shape == (2, 1000)

    m.eval()
    assert m(torch.randn(2, 3, 224, 224)).shape == (2, 1000)

    # Table 1 output depths at each stage
    x = m.stem(torch.randn(1, 3, 224, 224))
    assert x.shape[1:] == (192, 28, 28), x.shape
    x = m.inception3b(m.inception3a(x))
    assert x.shape[1:] == (480, 28, 28), x.shape
    x = m.inception5b(m.inception5a(m.pool4(m.inception4e(m.inception4d(
        m.inception4c(m.inception4b(m.inception4a(m.pool3(x)))))))))
    assert x.shape[1:] == (1024, 7, 7), x.shape

    # ~6.8M params in the paper's net (aux heads excluded)
    main = sum(p.numel() for n, p in m.named_parameters() if not n.startswith("aux"))
    assert 6e6 < main < 7.5e6, main
    print("googlenet ok  params=%.2fM (main only, aux excluded)" % (main / 1e6))
