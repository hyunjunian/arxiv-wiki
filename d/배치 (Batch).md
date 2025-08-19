# 배치란?  

- **배치**는 머신러닝·딥러닝에서 **한 번의 파라미터 업데이트에 사용되는 데이터 샘플들의 집합**을 의미합니다.  
- `전체 데이터 → 배치 → 파라미터 업데이트 → 다음 배치` 순으로 학습이 진행되며, 배치의 크기(=배치 사이즈)는 모델 성능·학습 효율에 큰 영향을 미칩니다.  

# 배치 학습과 온라인(또는 인크리멘털) 학습  

| 구분 | 배치 학습 (Batch Learning) | 온라인 학습 (Online / Incremental Learning) |
|------|---------------------------|---------------------------------------------|
| 데이터 사용 방식 | 전체 데이터를 여러 배치로 나누어 순차적으로 사용 | 한 샘플(또는 매우 작은 배치)씩 즉시 파라미터 업데이트 |
| 메모리 요구량 | 전체 데이터가 아니라 현재 배치만 메모리에 적재 | 메모리 사용량이 최소 |
| 수렴 특성 | 안정적인 그래디언트 추정 → 빠른 수렴 | 그래디언트 변동이 커서 학습이 불안정할 수 있음 |
| 적용 분야 | 대규모 이미지·텍스트 데이터, GPU 활용이 가능한 경우 | 스트리밍 데이터, 실시간 시스템 등 |
| 대표 알고리즘 | 미니배치 SGD, Adam, RMSProp 등 | 온라인 SGD, Perceptron, PA 알고리즘 등 |

# 배치 크기의 역할  

- **그래디언트 추정 정확도**: 배치가 클수록 전체 데이터에 가까운 그래디언트를 제공하지만, 계산 비용이 증가합니다.  
- **학습 안정성**: 작은 배치는 노이즈가 많아 발산 위험이 있지만, 일반화 성능을 높이는 경우도 있습니다.  
- **GPU/TPU 활용 효율**: 하드웨어는 병렬 연산에 최적화돼 있기 때문에, 적절히 큰 배치를 사용하면 **연산 효율**과 **메모리 사용**을 최적화할 수 있습니다.  

## 배치 크기와 메모리 사용량 예시 (표준 ResNet‑50, FP16)

| 배치 크기 (batch size) | GPU 메모리 사용량 (GB) | 1 epoch 시간 (분) | 최종 검증 정확도 (%) |
|------------------------|-----------------------|-------------------|----------------------|
| 16                     | 7.2                   | 45                | 76.3 |
| 32                     | 13.4                  | 32                | 76.5 |
| 64                     | 24.8                  | 24                | 76.2 |
| 128                    | 46.0                  | 18                | 75.9 |

> **Tip**: 메모리 한계에 걸리면 *Gradient Accumulation*(그래디언트 누적) 기법을 사용해 효과적으로 큰 배치를 시뮬레이션할 수 있습니다.  

# 배치 정규화 (Batch Normalization)  

- **핵심 아이디어**: 미니배치 내부에서 **채널별 평균·분산**을 정규화해 내부 공변량 이동(Internal Covariate Shift)을 완화합니다.  
- **효과**: 학습이 안정화되고, 높은 학습률을 사용할 수 있어 수렴 속도가 빨라집니다.  
- **제한점**: 배치 크기가 너무 작으면(예: ≤ 8) 통계가 불안정해 성능이 저하될 수 있습니다. 이런 경우 *Group Normalization*, *Layer Normalization* 등을 고려합니다.  

# 배치 구성과 데이터 파이프라인  

```mermaid
flowchart TD
    A[데이터 저장소] --> B[데이터 로더 (DataLoader)]
    B --> C[배치 샘플링 & 전처리]
    C --> D[GPU/TPU 메모리 로드]
    D --> E[모델 포워드]
    E --> F[손실 계산]
    F --> G[역전파]
    G --> H[파라미터 업데이트]
    H --> B
```

## 주요 포인트  

- **Shuffle**: epoch마다 데이터를 섞어 모델이 순서에 의존하지 않게 함.  
- **Prefetch / Parallel Workers**: `torch.utils.data.DataLoader(num_workers=N, pin_memory=True)`와 같은 옵션으로 CPU‑GPU 사이 I/O 병목을 최소화.  
- **데이터 증강(Augmentation)**: 배치 단위로 적용하면 GPU 메모리와 연산 효율이 개선됨.  

# 배치와 하드웨어 최적화  

| 최적화 요소 | 설명 | 구현 팁 |
|------------|------|--------|
| **Mixed Precision (FP16)** | 연산 속도와 메모리 절감 | `torch.cuda.amp.autocast()`와 `torch.cuda.amp.GradScaler` 사용 |
| **Tensor Cores 활용** | NVIDIA Volta/Turing/Ampere 이후 텐서 코어에 맞는 연산 | 배치 크기가 8의 배수일 때 최적 |
| **Distributed Data Parallel (DDP)** | 여러 GPU·노드에 배치 분산 | `torch.nn.parallel.DistributedDataParallel` 사용, 배치 크기를 `world_size` 로 나눔 |
| **Gradient Checkpointing** | 중간 활성값을 저장하지 않아 메모리 절감 | `torch.utils.checkpoint.checkpoint` 활용 |

# 배치 스케줄링과 학습률  

- **Warm‑up**: 초기 몇 epoch 동안 학습률을 점진적으로 늘려 큰 배치에 대한 급격한 파라미터 변동을 방지합니다.  
- **Learning Rate Scaling**: 배치 크기를 `k` 배 늘릴 경우 학습률을 `k` 배 (또는 `√k` 배) 스케일링하는 **Linear Scaling Rule**가 자주 사용됩니다.  

```python
 PyTorch 예시: Linear Scaling + Warmup
base_lr = 0.1
batch_size = 256
reference_batch = 32
scaled_lr = base_lr * batch_size / reference_batch   # Linear scaling

optimizer = torch.optim.SGD(model.parameters(), lr=scaled_lr, momentum=0.9)

def get_lr(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return scaled_lr * (epoch + 1) / warmup_epochs
    return scaled_lr

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                              lr_lambda=lambda epoch: get_lr(epoch) / scaled_lr)
```

# 배치와 분산 학습  

- **Data Parallelism**: 각 GPU가 서로 다른 배치 조각을 처리하고, **All‑Reduce**를 통해 그래디언트를 평균화합니다.  
- **Model Parallelism**: 하나의 배치를 모델의 서로 다른 파트에 분산시키지만, 배치 크기가 충분히 커야 효과적입니다.  
- **Hybrid Parallelism**: Data Parallelism + Model Parallelism 조합으로, 대규모 모델(예: GPT‑4) 학습에 사용됩니다.  

# 배치 최적화 팁 (베스트 프랙티스)  

- **배치 크기 선택**  
  - GPU 메모리 한계에 도달하기 직전까지 크게 설정 → 연산 효율 극대화.  
  - 배치가 너무 크면 일반화 성능이 저하될 수 있으니, **validation loss**를 모니터링하면서 조정.  
- **배치 정규화 사용 여부**  
  - 배치 크기 ≥ 32 → BatchNorm 권장.  
  - 배치 크기 < 32 → GroupNorm(4~8 groups) 혹은 LayerNorm 사용.  
- **Gradient Accumulation**  
  - 실제 메모리 한계보다 큰 배치 효과가 필요할 때 `accum_steps = desired_batch // actual_batch`.  
- **Learning Rate Warm‑up + Scaled LR**  
  - 큰 배치를 사용할 경우 반드시 warm‑up을 적용.  
- **Deterministic Training** (재현성)  
  - `torch.backends.cudnn.deterministic = True`와 같은 설정에 배치 샘플링 시드 고정.  

# 구현 예시 (PyTorch)  

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms

 1️⃣ 데이터 로더
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.425],
                         std=[0.229, 0.224, 0.225])
])

train_set = datasets.ImageFolder(root='data/train', transform=transform)
train_loader = DataLoader(
    train_set,
    batch_size=64,                # ← 배치 크기
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    sampler=RandomSampler(train_set, replacement=False)
)

 2️⃣ 모델 정의 (BatchNorm 포함)
model = torchvision.models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 100)  # 예시: 100 클래스

 3️⃣ Mixed Precision + Gradient Accumulation 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler()
accum_steps = 4      # 64 * 4 = 256 실제 배치 효과

 4️⃣ 학습 루프
model.train()
for epoch in range(30):
    optimizer.zero_grad()
    for i, (imgs, targets) in enumerate(train_loader):
        imgs, targets = imgs.to(device), targets.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = nn.CrossEntropyLoss()(outputs, targets) / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    print(f'Epoch {epoch+1} completed')
```

# 장점과 단점  

## 장점  

- **연산 효율**: 대규모 행렬 연산을 한 번에 수행해 GPU/TPU 활용도가 높아짐.  
- **그래디언트 노이즈 감소**: 배치 평균을 사용하므로 파라미터 업데이트가 안정적.  
- **배치 정규화와 함께 사용할 경우** 학습 속도와 일반화 능력 모두 향상.  

## 단점  

- **메모리 요구량**: 배치가 클수록 GPU 메모리 사용량 급증.  
- **일부 정규화 기법에 의존**: 배치가 작으면 BatchNorm이 불안정 → 대체 정규화 필요.  
- **데이터 셔플링 비용**: 큰 배치를 샘플링하려면 효율적인 셔플링 로직이 필요함.  

# 베스트 프랙티스 체크리스트  

- [ ] GPU 메모리 한계에 맞는 **배치 크기**를 결정했나요?  
- [ ] 배치 크기에 맞는 **정규화 기법**(BatchNorm / GroupNorm 등)을 선택했나요?  
- [ ] **Learning Rate Scaling**과 **Warm‑up**을 적용했나요?  
- [ ] **Gradient Accumulation**이 필요할 경우 `accum_steps`를 적절히 설정했나요?  
- [ ] 데이터 로딩 파이프라인에서 **멀티스레드**/`prefetch`를 사용했나요?  
- [ ] **Mixed Precision**을 활용해 메모리와 연산 효율을 최적화했나요?  
- [ ] 분산 학습을 할 경우 **All‑Reduce** 구현과 **배치 분할**이 올바른가요?  

# 관련 개념  

- **미니배치(Mini‑batch)** – 전체 데이터를 여러 작은 배치로 나눈 형태, 실제 딥러닝에서 가장 흔히 사용.  
- **스텝별 학습률(Learning Rate Scheduler)** – cosine annealing, step decay 등.  
- **데이터 파이프라인(Data Pipeline)** – `tf.data`, `torch.utils.data` 등 데이터 로딩·전처리 프레임워크.  
- **배치 정규화 외 정규화 기법** – LayerNorm, InstanceNorm, GroupNorm.  
- **배치 기반 강화학습** – 여러 에피소드를 동시에 시뮬레이션해 배치를 구성하는 RL 기법.  

# 참고 문헌  

1. **"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"**, Ioffe & Szegedy, 2015.  
   - https://arxiv.org/abs/1502.03167  
2. **"Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"**, Goyal *et al.*, 2017.  
   - https://arxiv.org/abs/1706.02677  
3. **"Training Deep Neural Networks with 8-bit Floating Point Numbers"**, Micikevicius *et al.*, 2018 (Mixed Precision).  
   - https://arxiv.org/abs/1710.03740  
4. **PyTorch Documentation – Data Loading and Parallelism**.  
   - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html  
5. **"On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"**, Keskar *et al.*, 2016.  
   - https://arxiv.org/abs/1609.04836  

---  

*이 문서는 GitHub‑Flavored Markdown(GFM) 형식으로 작성되었으며, 위키 페이지, README, 혹은 개발 문서에 그대로 복사·붙여넣기하여 사용할 수 있습니다.*