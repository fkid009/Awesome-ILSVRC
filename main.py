from dataclasses import asdict

from config import CFG
from data import CIFAR10DataModule
from models.alexnet import AlexNet
from models.resnet import ResNet
from models.vggnet import VGG16
from trainer import run

# (모델, 이름, lr)
CONFIGS = [
    (AlexNet, "AlexNet", 0.01),   # BN 없음 -> lr 낮게
    (VGG16, "VGG16", 0.05),
    (ResNet, "ResNet20", 0.1),    # BN 있음 -> lr 크게
]


def main() -> None:
    cfg = CFG()
    print(asdict(cfg))

    dm = CIFAR10DataModule(cfg.data_root, cfg.val_frac, cfg.batch_size,
                           cfg.num_workers, cfg.seed)
    rows = [run(build, name, lr, cfg, dm) for build, name, lr in CONFIGS]

    print("\n%-10s %6s %9s %9s %9s"
          % ("model", "lr", "params_M", "val_acc", "test_acc"))
    for r in rows:
        print("%-10s %6g %9.3f %9.2f %9.2f"
              % (r["model"], r["lr"], r["params_M"], r["val_acc"], r["test_acc"]))


if __name__ == "__main__":
    main()
