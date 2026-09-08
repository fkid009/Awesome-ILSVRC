import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger


class LitClassifier(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 0.1, momentum: float = 0.9,
                 weight_decay: float = 5e-4, max_epochs: int = 10):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log_dict({f"{stage}_loss": loss, f"{stage}_acc": acc},
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        """논문들은 plateau마다 lr을 10배씩 줄였다. cosine decay가 스케줄 튜닝
        없이 비슷하거나 더 나은 결과를 주므로 그걸 쓴다."""
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
                                    momentum=self.hparams.momentum,
                                    weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def run(build, name: str, lr: float, cfg, dm) -> dict:
    """한 모델을 학습하고 best 체크포인트로 test까지 - 결과 한 줄을 돌려준다.

    cfg는 config.CFG (epochs / momentum / weight_decay / seed / amp 를 읽는다)."""
    L.seed_everything(cfg.seed, workers=True)

    model = build()
    lit = LitClassifier(model, lr=lr, momentum=cfg.momentum,
                        weight_decay=cfg.weight_decay, max_epochs=cfg.epochs)
    ckpt = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if cfg.amp else "32-true",
        default_root_dir=f"runs/{name}",
        logger=CSVLogger("runs", name=name),
        callbacks=[ckpt, LearningRateMonitor(logging_interval="epoch")],
    )
    trainer.fit(lit, datamodule=dm)
    test = trainer.test(lit, datamodule=dm, ckpt_path="best", verbose=False)[0]

    return {
        "model": name,
        "lr": lr,
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "val_acc": ckpt.best_model_score.item() * 100,
        "test_acc": test["test_acc"] * 100,
    }
