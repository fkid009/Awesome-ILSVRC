"""ResNet - ILSVRC 2015 winner (top-5 error 3.57%, first below human ~5%).

He, Zhang, Ren, Sun. "Deep Residual Learning for Image Recognition." CVPR 2016.

The problem: past ~20 layers, plain deep nets get *worse* on training error
too, so it is degradation, not overfitting (Fig 1). The fix: have a block
learn a residual F(x) and add the input back, y = F(x) + x. If the optimal
map is close to identity, driving F to zero is far easier than fitting an
identity through a stack of convs. The shortcut is parameter-free.

Two block types (Fig 5):
  BasicBlock  3x3 -> 3x3                    (ResNet-18/34)
  Bottleneck  1x1 reduce -> 3x3 -> 1x1 restore, expansion 4  (ResNet-50/101/152)

When a block changes shape (stride 2, or channel count), the shortcut needs a
projection: option B in the paper, a 1x1 conv + BN, used everywhere here.
BN goes after every conv and before the addition; ReLU comes *after* the add.
"""

import torch
import torch.nn as nn


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv3x3(in_ch, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)  # add first, then ReLU


class Bottleneck(nn.Module):
    """1x1 down to `planes`, 3x3 at `planes`, 1x1 back up to 4*planes. Keeps
    the FLOPs of a BasicBlock while quadrupling the block's output width."""

    expansion = 4

    def __init__(self, in_ch: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None):
        super().__init__()
        out_ch = planes * self.expansion
        self.conv1 = conv1x1(in_ch, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        # stride on the 3x3 (torchvision/"ResNet v1.5"); the paper put it on the
        # first 1x1, which throws away 3/4 of the input pixels. v1.5 is ~0.5% better.
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, out_ch)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, block, layers: list[int], num_classes: int = 1000,
                 zero_init_residual: bool = True):
        super().__init__()
        self.in_ch = 64
        # stem: 224 -> 112 -> 56
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        # four stages; each doubles width and halves resolution (except the first)
        self.layer1 = self._make_layer(block, 64, layers[0])              # 56
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)   # 28
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)   # 14
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)   # 7
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._init_weights(zero_init_residual)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_ch != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_ch, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_ch, planes, stride, downsample)]
        self.in_ch = planes * block.expansion
        layers += [block(self.in_ch, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_residual:
            # zero the last BN gamma in each block so it starts as exact identity
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


# Table 1: blocks per stage
def resnet18(**kw) -> ResNet: return ResNet(BasicBlock, [2, 2, 2, 2], **kw)
def resnet34(**kw) -> ResNet: return ResNet(BasicBlock, [3, 4, 6, 3], **kw)
def resnet50(**kw) -> ResNet: return ResNet(Bottleneck, [3, 4, 6, 3], **kw)
def resnet101(**kw) -> ResNet: return ResNet(Bottleneck, [3, 4, 23, 3], **kw)
def resnet152(**kw) -> ResNet: return ResNet(Bottleneck, [3, 8, 36, 3], **kw)


if __name__ == "__main__":
    # (builder, weight-layer depth, params in millions) - Table 1 / torchvision
    specs = [(resnet18, 18, 11.7), (resnet34, 34, 21.8), (resnet50, 50, 25.6),
             (resnet101, 101, 44.5), (resnet152, 152, 60.2)]
    for build, depth, want_m in specs:
        m = build()
        assert m(torch.randn(2, 3, 224, 224)).shape == (2, 1000)
        # weight layers on the main path + fc (shortcut projections excluded)
        n = 1 + sum(1 for name, mod in m.named_modules()
                    if isinstance(mod, nn.Conv2d) and "downsample" not in name)
        assert n == depth, (depth, n)
        got_m = sum(p.numel() for p in m.parameters()) / 1e6
        assert abs(got_m - want_m) < 0.3, (depth, got_m, want_m)
        print("resnet%-3d ok  params=%.1fM" % (depth, got_m))

    # the residual branch starts at zero, so a fresh block is the identity
    m = resnet18().eval()
    blk = m.layer1[0]
    x = torch.randn(1, 64, 56, 56)
    assert torch.allclose(blk(x), torch.relu(x)), "residual branch should start at 0"
    print("identity-init ok")
