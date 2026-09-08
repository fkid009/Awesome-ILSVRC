# Awesome-ILSVRC

ILSVRC 주요 아키텍처 구현 레포지토리

## 시작

```bash
uv sync                            # virtual env 
uv run main.py                     # training and evaluation
```

## 모델

| 모델 | 대회 | 논문 top-5 err | 가중치 층 | contributions |
|---|---|---|---|---|
| [AlexNet](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | ILSVRC'12 우승 | 15.3% | 8 | ReLU, dropout, GPU 학습으로 CNN 시대 개막 |
| [VGG16](https://arxiv.org/abs/1409.1556) | ILSVRC'14 2위 | 7.3% | 16 | 3x3 conv만 쌓아서 깊이 확보 |
| [ResNet20](https://arxiv.org/abs/1512.03385) | ILSVRC'15 우승 | 3.57% | 20 | residual connection으로 degradation 해결 |

## 성능

| model | best_val_acc | test_acc |
|---|---|---|
| AlexNet | 81.08 | 80.68 |
| VGG16 | 83.44 | 83.44 |
| ResNet20 | 85.62 | 84.96 |
