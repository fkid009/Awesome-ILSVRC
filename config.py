from dataclasses import dataclass


@dataclass
class CFG:
    # data
    data_root: str = "./data"
    val_frac: float = 0.1           # train에서 떼어낼 validation 비율

    # training
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 10
    momentum: float = 0.9
    weight_decay: float = 5e-4

    # utils
    seed: int = 42
    amp: bool = True                # mixed precision
