# Awesome-ILSVRC

ILSVRC 우승 아키텍처들을 논문 그대로 PyTorch로 구현하고, Lightning으로 학습/평가까지.

## 시작

```bash
uv sync                                    # .venv 생성 (torch, torchvision, lightning)
uv run python models/resnet.py             # 아키텍처 self-check (논문 수치 검증)
uv run python test_train.py                # 학습/평가 경로 self-check (데이터 불필요)
uv run train.py --model resnet18           # 학습 -> 자동으로 best ckpt로 test
uv run train.py --model googlenet --fast-dev-run   # 실제 데이터로 1 batch 스모크 테스트
```

## 모델

| 모델 | 대회 | top-5 err | 가중치 층 | 파라미터 | 핵심 아이디어 |
|---|---|---|---|---|---|
| [AlexNet](models/alexnet.py) | ILSVRC'12 우승 | 15.3% | 8 | 62.4M | ReLU, dropout, GPU 학습으로 CNN 시대 개막 |
| [VGGNet](models/vggnet.py) | ILSVRC'14 2위 | 7.3% | 11~19 | 133~144M | 3x3 conv만 쌓아서 깊이 확보 |
| [GoogLeNet](models/googlenet.py) | ILSVRC'14 우승 | 6.67% | 22 | 7.0M | Inception 병렬 분기 + 1x1 채널 축소, GAP로 FC 제거 |
| [ResNet](models/resnet.py) | ILSVRC'15 우승 | 3.57% | 18~152 | 11.7~60.2M | residual shortcut으로 degradation 해결 |

파라미터 수는 self-check 실행 결과 (구현체 실측값). 사용 가능한 이름:
`alexnet`, `vgg11/13/16/19`, `googlenet`, `resnet18/34/50/101/152`.

각 파일을 직접 실행하면 논문의 feature map 크기, 가중치 층 수, 파라미터 수를
assert로 검증한다 (ResNet은 residual 분기가 0에서 시작하는 identity init까지).

## 공부 포인트

**AlexNet → VGG**: "커널 크기를 어떻게 고를까"에 대한 답이 "3x3만 쓰고 깊게".
3x3 두 개 = 5x5 하나의 receptive field인데 파라미터는 27C² vs 49C²로 더 적고
비선형성은 하나 더 붙는다.

**VGG → GoogLeNet**: VGG는 138M 파라미터의 90%가 FC 세 층에 있다. GoogLeNet은
global average pooling으로 그걸 없애고 22층인데도 7M. 대신 병렬 분기 비용을
1x1 conv로 채널을 먼저 줄여서 감당한다.

**GoogLeNet → ResNet**: 층을 더 쌓으면 *학습* 오차까지 나빠지는 degradation이
문제였고 (overfitting이 아니다), y = F(x) + x 로 identity를 공짜로 만들어
152층까지 갔다. auxiliary classifier 같은 gradient 주입 트릭이 필요 없어졌다.

## 학습 (`train.py`)

- **데이터**: CIFAR-10을 각 아키텍처가 요구하는 입력 크기로 업스케일 (224,
  AlexNet은 227). ImageNet은 150GB라 공부용으로 비현실적이고, 업스케일하면
  아키텍처를 한 줄도 고치지 않고 돌려볼 수 있다. 첫 실행 시 `data/`에 자동 다운로드.
- **split**: train 45k / val 5k (같은 seed로 쪼개 겹치지 않게) + test 10k.
- **메트릭**: top-1과 top-5 둘 다 로깅 — 논문들이 top-5로 경쟁했으므로.
- **optimizer**: SGD(momentum 0.9, wd 5e-4) + cosine decay. 논문들은 plateau마다
  lr을 1/10로 줄였는데, cosine이 스케줄 튜닝 없이 비슷하거나 더 낫다.
- **GoogLeNet aux head**: 학습 시 `(logits, aux2, aux1)`을 반환하고 loss에
  0.3 가중치로 더한다 (논문 Sec 5). eval 모드에서는 logits만 나온다.

```bash
uv run train.py --model vgg16 --epochs 30 --batch-size 64 --lr 0.01
uv run train.py --model resnet50 --precision bf16-mixed        # 학습 속도 개선
uv run train.py --model resnet18 --test-only --ckpt runs/resnet18/.../best-12.ckpt
```

체크포인트와 로그는 `runs/<model>/`에 쌓인다. `--help`로 전체 옵션 확인.

## 구현 노트

- **AlexNet 입력은 227x227.** 논문 본문은 224라고 하지만 11x11/stride4를 통과하면
  55x55가 안 나온다. torchvision은 224 유지 + conv1 padding=2로 우회.
- **GoogLeNet의 LRN/BatchNorm.** 2014년 원논문에는 BN이 없다 (BN은 Inception-v2).
  `batch_norm=True`로 켜면 LRN이 Identity로 빠진다.
- **ResNet bottleneck stride 위치.** 논문은 첫 1x1에 stride 2를 걸지만 그러면
  입력 픽셀 3/4를 버린다. 여기서는 3x3에 거는 v1.5 방식 (~0.5% 더 정확).
- **shortcut은 option B** (1x1 conv projection) 고정. 논문의 A(zero-pad)/C(전부
  projection)는 생략 — B가 표준.
