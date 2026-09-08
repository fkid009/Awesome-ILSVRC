import lightning as L
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10

# CIFAR-10 전체 학습셋의 채널별 mean/std
MEAN, STD = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]


def build_transform(mean=MEAN, std=STD):
    train_tf = T.Compose([
        # 4px 패딩 후 32로 다시 crop. reflect는 0 패딩보다 경계 인공물이 적다
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        # 좌우 반전만. vertical flip은 안 쓴다 - 뒤집힌 고양이/자동차는 없다
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),               # PIL -> tensor + [0,1] 스케일링
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, eval_tf


class TransformDataset(Dataset):
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), label


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_root: str = "./data", val_frac: float = 0.1,
                 batch_size: int = 64, num_workers: int = 4, seed: int = 42):
        super().__init__()
        self.save_hyperparameters()
        self.train_tf, self.eval_tf = build_transform()

    def prepare_data(self) -> None:
        for train in (True, False):  # rank 0에서 한 번만 다운로드된다
            CIFAR10(self.hparams.data_root, train=train, download=True)

    def setup(self, stage: str | None = None) -> None:
        root = self.hparams.data_root
        full = CIFAR10(root, train=True, transform=None)
        n_val = int(self.hparams.val_frac * len(full))
        g = torch.Generator().manual_seed(self.hparams.seed)
        train, val = random_split(full, [len(full) - n_val, n_val], generator=g)

        self.ds_train = TransformDataset(train, self.train_tf)
        self.ds_val = TransformDataset(val, self.eval_tf)
        self.ds_test = TransformDataset(
            CIFAR10(root, train=False, transform=None), self.eval_tf)

    def _loader(self, dataset, shuffle: bool = False) -> DataLoader:
        w = self.hparams.num_workers
        return DataLoader(dataset, batch_size=self.hparams.batch_size,
                          shuffle=shuffle, num_workers=w, pin_memory=True,
                          persistent_workers=w > 0)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.ds_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.ds_val)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.ds_test)
