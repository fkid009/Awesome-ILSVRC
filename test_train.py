"""학습/평가 경로 self-check. 데이터 다운로드 없이 랜덤 텐서로 돌린다.

    uv run python test_train.py

GoogLeNet의 aux head 분기 (학습 시 tuple 반환 -> loss 3개 합산) 가 여기서
검증되는 유일한 비자명 로직이다. 실제 데이터까지 포함한 확인은
`uv run train.py --model X --fast-dev-run`.
"""

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

from train import MODELS, Classifier

if __name__ == "__main__":
    for name in ("googlenet", "resnet18", "alexnet"):  # aux 있음/없음/입력 227
        size = MODELS[name][1]
        loader = DataLoader(TensorDataset(torch.randn(4, 3, size, size),
                                          torch.randint(0, 10, (4,))), batch_size=2)
        model = Classifier(name)
        trainer = L.Trainer(fast_dev_run=True, accelerator="cpu", logger=False,
                            enable_checkpointing=False, enable_progress_bar=False,
                            enable_model_summary=False)
        trainer.fit(model, loader, loader)
        trainer.test(model, loader, verbose=False)
        m = trainer.callback_metrics
        assert 0.0 < m["test/loss"] < 100.0, m["test/loss"]
        assert m["test/top5"] >= m["test/top1"], (m["test/top5"], m["test/top1"])
        print("%-10s train+val+test ok  loss=%.3f" % (name, m["test/loss"]))
    print("train.py ok")
