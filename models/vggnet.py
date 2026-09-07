"""VGGNet - ILSVRC 2014 runner-up in classification, winner in localisation.

Simonyan & Zisserman. "Very Deep Convolutional Networks for Large-Scale
Image Recognition." ICLR 2015.

The whole point of the paper is one idea: fix every conv to 3x3/stride 1
and just go deeper. Two stacked 3x3 have the receptive field of one 5x5,
three have one 7x7 - with fewer parameters (3*(3^2 C^2) = 27C^2 vs 49C^2)
and two extra non-linearities. Table 1 configs A..E are the cfgs below.

Spatial size is halved by 2x2/stride-2 max pool five times:
224 -> 112 -> 56 -> 28 -> 14 -> 7, so the classifier sees 512*7*7.
"""

import torch
import torch.nn as nn

# Table 1. Numbers = conv3-<out_channels>, "M" = maxpool, "L" = conv1-<out>
# (config C uses 1x1 convs where D uses 3x3 - the only non-3x3 conv in VGG).
cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "C": [64, 64, "M", 128, 128, "M", 256, 256, ("L", 256), "M",
          512, 512, ("L", 512), "M", 512, 512, ("L", 512), "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
          512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
          512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def make_layers(cfg: list, batch_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_ch = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        kernel, out_ch = (1, v[1]) if isinstance(v, tuple) else (3, v)
        layers.append(nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2))
        if batch_norm:  # not in the paper; VGG predates BN and used careful init
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        in_ch = out_ch
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, cfg: str = "D", num_classes: int = 1000,
                 batch_norm: bool = False, dropout: float = 0.5):
        super().__init__()
        self.features = make_layers(cfgs[cfg], batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Paper used N(0, 0.01) plus pre-training of config A to bootstrap the
        deeper nets; they later noted Glorot init removes that need. He init is
        the modern default for ReLU nets, so use it."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


def vgg11(**kw) -> VGG: return VGG("A", **kw)
def vgg13(**kw) -> VGG: return VGG("B", **kw)
def vgg16_c(**kw) -> VGG: return VGG("C", **kw)   # the 1x1-conv variant
def vgg16(**kw) -> VGG: return VGG("D", **kw)
def vgg19(**kw) -> VGG: return VGG("E", **kw)


if __name__ == "__main__":
    # Table 2 of the paper: weight-layer counts per config (conv + 3 FC)
    depths = {"A": 11, "B": 13, "C": 16, "D": 16, "E": 19}
    for name, cfg in cfgs.items():
        m = VGG(cfg=name)
        assert m(torch.randn(1, 3, 224, 224)).shape == (1, 1000)
        n_weight_layers = sum(isinstance(l, nn.Conv2d) for l in m.features) + 3
        assert n_weight_layers == depths[name], (name, n_weight_layers)
        assert m.features(torch.randn(1, 3, 224, 224)).shape[1:] == (512, 7, 7)
        print("vgg-%s ok  depth=%d  params=%.1fM"
              % (name, n_weight_layers,
                 sum(p.numel() for p in m.parameters()) / 1e6))
