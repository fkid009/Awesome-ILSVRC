"""ILSVRC 아키텍처 학습/평가 (PyTorch Lightning).

    uv run train.py --model resnet18 --epochs 30
    uv run train.py --model googlenet --fast-dev-run      # 1 batch 스모크 테스트
    uv run train.py --model resnet18 --test-only --ckpt runs/.../best.ckpt

데이터는 ImageNet이 아니라 CIFAR-10을 각 아키텍처가 요구하는 입력 크기
(224, AlexNet은 227)로 업스케일해서 쓴다. ImageNet은 150GB라 공부용으로
비현실적이고, 업스케일하면 아키텍처를 한 줄도 고치지 않고 그대로 돌려볼 수 있다.
top-1 / top-5 를 모두 로깅하는 이유는 논문들이 top-5로 경쟁했기 때문.
"""

from __future__ import annotations

import argparse
import os

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MulticlassAccuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10

from models.alexnet import alexnet
from models.googlenet import googlenet
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.vggnet import vgg11, vgg13, vgg16, vgg19

# name -> (builder, 아키텍처가 요구하는 입력 크기)
MODELS = {
    "alexnet": (alexnet, 227),
    "vgg11": (vgg11, 224),
    "vgg13": (vgg13, 224),
    "vgg16": (vgg16, 224),
    "vgg19": (vgg19, 224),
    "googlenet": (googlenet, 224),
    "resnet18": (resnet18, 224),
    "resnet34": (resnet34, 224),
    "resnet50": (resnet50, 224),
    "resnet101": (resnet101, 224),
    "resnet152": (resnet152, 224),
}


class Classifier(L.LightningModule):
    def __init__(self, model: str = "resnet18", num_classes: int = 10,
                 lr: float = 0.01, momentum: float = 0.9,
                 weight_decay: float = 5e-4, aux_weight: float = 0.3):
        super().__init__()
        self.save_hyperparameters()
        self.net = MODELS[model][0](num_classes=num_classes)
        acc = lambda k: MulticlassAccuracy(num_classes, top_k=k)
        # stage마다 별도 인스턴스가 필요하다 (torchmetrics는 상태를 누적한다)
        self.train_top1 = acc(1)
        self.val_top1, self.val_top5 = acc(1), acc(5)
        self.test_top1, self.test_top5 = acc(1), acc(5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, _) -> torch.Tensor:
        x, y = batch
        out = self.net(x)
        # GoogLeNet은 학습 중 (logits, aux2, aux1) 을 반환한다 (논문 가중치 0.3)
        logits, *aux = out if isinstance(out, tuple) else (out,)
        loss = F.cross_entropy(logits, y) + self.hparams.aux_weight * sum(
            F.cross_entropy(a, y) for a in aux
        )
        self.train_top1(logits, y)
        self.log_dict({"train/loss": loss, "train/top1": self.train_top1},
                      prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def _eval_step(self, batch, stage: str) -> None:
        x, y = batch
        logits = self.net(x)
        top1, top5 = getattr(self, f"{stage}_top1"), getattr(self, f"{stage}_top5")
        top1(logits, y)
        top5(logits, y)
        self.log_dict({f"{stage}/loss": F.cross_entropy(logits, y),
                       f"{stage}/top1": top1, f"{stage}/top5": top5}, prog_bar=True)

    def validation_step(self, batch, _) -> None:
        self._eval_step(batch, "val")

    def test_step(self, batch, _) -> None:
        self._eval_step(batch, "test")

    def configure_optimizers(self):
        """논문들은 plateau마다 lr을 10배씩 줄였다. cosine decay가 스케줄 튜닝
        없이 비슷하거나 더 나은 결과를 주므로 그걸 쓴다."""
        opt = torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
                              momentum=self.hparams.momentum,
                              weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs)
        return [opt], [sched]


class CIFAR10Data(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", image_size: int = 224,
                 batch_size: int = 64, num_workers: int | None = None,
                 val_size: int = 5000):
        super().__init__()
        self.save_hyperparameters()
        if num_workers is None:
            self.hparams.num_workers = min(8, (os.cpu_count() or 2) // 2)

        norm = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2470, 0.2435, 0.2616))
        # crop/flip은 32x32에서 먼저 (싸다), 그 다음 업스케일
        self.train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            norm,
        ])
        self.eval_tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            norm,
        ])

    def prepare_data(self) -> None:
        for train in (True, False):  # rank 0에서 한 번만 다운로드된다
            CIFAR10(self.hparams.data_dir, train=train, download=True)

    def setup(self, stage: str | None = None) -> None:
        d, n = self.hparams.data_dir, self.hparams.val_size
        # train/val은 같은 50k에서 갈라지지만 transform이 다르다. 데이터셋을 두 번
        # 만들고 같은 seed로 쪼개면 인덱스가 정확히 겹치지 않게 나뉜다.
        lengths = [50_000 - n, n]
        seed = torch.Generator().manual_seed(42)
        self.train_set = random_split(
            CIFAR10(d, train=True, transform=self.train_tf), lengths, seed)[0]
        seed = torch.Generator().manual_seed(42)
        self.val_set = random_split(
            CIFAR10(d, train=True, transform=self.eval_tf), lengths, seed)[1]
        self.test_set = CIFAR10(d, train=False, transform=self.eval_tf)

    def _loader(self, dataset, shuffle: bool = False) -> DataLoader:
        w = self.hparams.num_workers
        return DataLoader(dataset, batch_size=self.hparams.batch_size,
                          shuffle=shuffle, num_workers=w, pin_memory=True,
                          persistent_workers=w > 0, drop_last=shuffle)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_set)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="resnet18", choices=list(MODELS))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--precision", default="32-true",
                   help="16-mixed / bf16-mixed 로 학습 속도를 크게 올릴 수 있다")
    p.add_argument("--ckpt", help="resume 용, 또는 --test-only 로 평가할 체크포인트")
    p.add_argument("--test-only", action="store_true", help="학습 없이 test set 평가")
    p.add_argument("--fast-dev-run", action="store_true",
                   help="train/val/test 1 batch만 - 전체 경로 스모크 테스트")
    args = p.parse_args()
    if args.test_only and not args.ckpt:
        p.error("--test-only 는 --ckpt 가 필요하다")

    L.seed_everything(42, workers=True)
    data = CIFAR10Data(args.data_dir, MODELS[args.model][1],
                       args.batch_size, args.num_workers)
    model = Classifier(args.model, lr=args.lr)
    trainer = L.Trainer(
        max_epochs=args.epochs,
        precision=args.precision,
        fast_dev_run=args.fast_dev_run,
        default_root_dir=f"runs/{args.model}",
        callbacks=[
            ModelCheckpoint(monitor="val/top1", mode="max", save_last=True,
                            filename="best-{epoch:02d}"),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )
    if args.test_only:
        trainer.test(model, data, ckpt_path=args.ckpt)
        return
    trainer.fit(model, data, ckpt_path=args.ckpt)
    # fast-dev-run은 체크포인트를 저장하지 않으므로 현재 weight로 평가한다
    trainer.test(model, data, ckpt_path=None if args.fast_dev_run else "best")


if __name__ == "__main__":
    main()
