# MuZero (뮤제로)

## 소개
MuZero는 DeepMind가 2019년에 발표한 모델 기반 강화학습 알고리즘으로, **환경의 전이 확률을 직접 모델링하지 않으면서** Monte‑Carlo Tree Search(MCTS)와 딥 뉴럴 네트워크를 결합해 탁월한 플래닝 능력을 보여준다. AlphaGo·AlphaZero가 사전에 정의된 환경 모델(바둑 규칙, 체스 규칙 등)을 이용해 학습한 반면, MuZero는 **보상·값·정책을 예측하는 세 가지 네트워크**만을 학습함으로써, 다양한 도메인(Atari, 체스, 바둑, 쇼기 등)에서 **동일한 알고리즘 구조**만으로 최첨단 성능을 달성한다.

## 배경 및 동기
- **모델 기반 RL의 도전**: 전통적인 모델 기반 강화학습은 환경 전이 함\(P(s’|s,a)\)을 정확히 추정해야 하는데, 복잡하거나 관측이 제한된 실제 환경에서는 이 작업이 매우 어려웠다.  
- **AlphaZero의 한계**: AlphaZero는 **완전한 규칙 모델**(예: 바둑 규칙)을 전제로 하여 MCTS를 수행했지만, 규칙을 명시적으로 알 수 없는 환경에는 적용할 수 없었다.  
- **MuZero의 목표**: 규칙이 명시되지 않은 환경에서도 **플래닝**과 **정책 학습**을 동시에 수행할 수 있는 일반화된 프레임워크를 제공한다.

## 핵심 아이디어 및 알고리즘
MuZero는 크게 세 부분으로 구성된다.

1. **Representation Network** – 현재 관측값을 “잠재 상태”(latent state) \(h_t\) 로 인코딩한다.  
2. **Dynamics Network** – 잠재 상태와 행동을 입력 받아 다음 잠재 상태와 보상 \(r_t\) 를 예측한다.  
3. **Prediction Network** – 잠재 상태로부터 **정책** \(\pi_t\) 와 **가치** \(v_t\) 를 출력한다.

이 세 네트워크를 **MCTS**와 결합해 시뮬레이션 트리를 확장하고, 트리 탐색 결과를 이용해 행동을 선택한다. 학습 시에는 실제 환경에서 얻은 관측 시퀀스와 MCTS에서 얻은 **검색 정책**을 손실 함수에 포함시켜, 네트워크가 검색 과정 자체를 모방하도록 만든다.

### MCTS와 통합 학습
- **루트 확장**: 현재 시점의 잠재 상태 \(h_t\) 로부터 MCTS를 수행한다.  
- **가치와 정책 백업**: 각 노드에서 예측된 가치와 정책을 사용해 트리를 백업한다.  
- **검색 정책 \(\pi^{\text{MCTS}}\)**: 시뮬레이션 횟수에 기반한 행동 선택 확률을 얻는다.  

### 손실 함수와 학습 목표
\[
\mathcal{L} = \sum_{k=0}^{K-1} \Big[
    \underbrace{(v_k - v^{\text{MCTS}}_k)^2}_{\text{값 손실}}
    + \underbrace{\text{CE}\big(\pi_k, \pi^{\text{MCTS}}_k\big)}_{\text{정책 손실}}
    + \underbrace{(r_k - r^{\text{target}}_k)^2}_{\text{보상 손실}}
\Big]
\]
- **값 손실**: 네트워크가 예측한 가치와 MCTS에서 얻은 백업 가치의 L2 차이.  
- **정책 손실**: 교차 엔트로피 손실을 사용해 네트워크 정책을 검색 정책에 맞춘다.  
- **보상 손실**: 실제 관측 보상과 Dynamics Network가 예측한 보상 간 차이.

## 아키텍처 상세

### 표현 네트워크 (Representation Network)
| 구성 | 설명 |
|------|------|
| 입력 | 현재 시점 관측 \(o_t\) (예: 이미지 프레임, 체스 보드) |
| 구조 | CNN(이미지), Residual block, 혹은 Transformer 기반 인코더 |
| 출력 | 잠재 상태 \(h_t\) (보통 128~256 차원의 벡터) |

### 다이내믹스 네트워크 (Dynamics Network)
| 구성 | 설명 |
|------|------|
| 입력 | 이전 잠재 상태 \(h_t\) 와 선택된 행동 \(a_t\) |
| 구조 | Fully‑connected 레이어 + Residual block (or Conv) |
| 출력 | 다음 잠재 상태 \(h_{t+1}\) 와 보상 예측 \(r_t\) |

### 예측 네트워크 (Prediction Network)
| 구성 | 설명 |
|------|------|
| 입력 | 잠재 상태 \(h_t\) |
| 구조 | Shared trunk → 두 개의 헤드 (policy head, value head) |
| 출력 | 정책 \(\pi_t\) (softmax over actions) 와 가치 \(v_t\) (scalar) |

## 학습 절차

```python
# Pseudo‑code for MuZero training loop
for episode in range(num_episodes):
    # 1. 환경에서 하나의 게임/시뮬레이션을 수행
    observations, actions, rewards = [], [], []
    observation = env.reset()
    while not done:
        # 2. 현재 관측을 latent state 로 인코딩
        h = representation_network(observation)

        # 3. MCTS를 이용해 행동 선택 (검색 정책)
        pi_mcts = run_mcts(h, dynamics_network,
                           prediction_network, num_simulations)

        # 4. 행동을 샘플링하고 환경에 적용
        action = sample_action(pi_mcts)
        next_observation, reward, done, info = env.step(action)

        # 5. 데이터를 저장
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)

        observation = next_observation

    # 6. 저장된 trajectory를 사용해 네트워크 학습
    for t in range(len(observations)):
        # (h_t, a_t, r_t, v_t 등) 타깃을 MCTS 결과에서 추출
        target = compute_targets(t, observations, actions, rewards,
                                 dynamics_network, prediction_network)
        loss = loss_fn(target, h_t, a_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 주요 단계 요약
1. **데이터 수집**: 실제 환경에서 에피소드를 실행하고, 관측·행동·보상 시퀀스를 저장한다.  
2. **MCTS 시뮬레이션**: 현재 잠재 상태에서 지정된 횟수만큼 시뮬레이션을 실행한다(보통 50~200회).  
3. **검색 정책 추출**: 시뮬레이션 횟수 비례로 행동 선택 확률 \(\pi^{\text{MCTS}}\) 를 만든다.  
4. **목표값 생성**: MCTS 백업을 통해 얻은 값과 보상을 타깃으로 사용한다.  
5. **네트워크 업데이트**: 앞서 정의한 손실 함수를 최소화하도록 파라미터를 업데이트한다.

## 성능 및 평가
MuZero는 기존 모델‑프리 강화학습 방법(A3C, DQN 등)과 모델‑기반 방법(PlaNet, Dreamer) 대비 **뛰어난 샘플 효율성과 최첨단 성능**을 동시에 달성했다.

| 도메인 | 테스트된 환경 | 최고 점수 (MuZero) | 기존 최고점 (비교) |
|--------|----------------|-------------------|--------------------|
| Atari 100k | *Breakout*, *Pong*, *Seaquest* 등 57개 | 평균 94% 인간 수준 | DQN 50%, Rainbow 80% |
| 바둑 (19×19) | 전역 승률 | 99.8% (19×19) | AlphaZero 99.5% |
| 체스 | Elo | 3520 (Self‑play) | AlphaZero 3500 |
| 쇼기 | Elo | 3400 | AlphaZero 3390 |

> *표는 MuZero 논문 및 공식 발표 자료를 기반으로 요약한 것이며, 최신 업데이트가 반영될 수 있다.*

## 활용 사례
- **게임 AI**: Atari, 체스, 바둑, 쇼기와 같은 전통적인 전략·액션 게임.  
- **로보틱스**: 시뮬레이션 기반 로봇 제어(예: MuJoCo 환경)에서 모델을 이용해 계획 수립.  
- **자율 주행**: 복잡한 교통 상황을 시뮬레이션하고, 보상 기반 의사결정에 활용.  
- **재무 최적화**: 포트폴리오 관리 및 트레이딩 시뮬레이션에서 정책 탐색.

## 장점 및 한계

### 장점
- **규칙 무관**: 명시적인 전이 모델이 필요 없어, 규칙이 불확실하거나 관측이 제한된 환경에서도 적용 가능.  
- **통합 플래닝·학습**: MCTS와 신경망이 상호 보완적으로 작동해 높은 탐색 효율을 제공.  
- **범용성**: 동일한 아키텍처를 사용해 다양한 도메인에 적용할 수 있다.

### 한계
- **컴퓨팅 비용**: MCTS와 다중 네트워크 호출이 결합돼 학습·추론 시 강력한 GPU/TPU가 요구된다.  
- **연속 행동 공간**: 현재 구현은 이산 액션에 최적화돼 있어, 연속 제어 문제에선 별도 변형(예: MuZero‑Continuous) 필요.  
- **메모리 사용량**: 대규모 트리를 저장하고 백업하는 과정에서 메모리 사용량이 급증한다.

## 관련 연구
| 논문 | 핵심 아이디어 |
|------|----------------|
| *AlphaZero* (Silver et al., 2018) | MCTS + 정책·가치 네트워크, 하지만 전이 모델 필요. |
| *PlaNet* (Hafner et al., 2019) | VAE 기반 연속 상태 모델링, 확률 전이 모델 필요. |
| *Dreamer* (Hafner et al., 2020) | latent dynamics와 정책을 함께 학습, 모델 기반 RL. |
| *EfficientZero* (Zhou et al., 2022) | MuZero를 경량화·연산 효율 향상, 저비용 디바이스 적용. |
| *MuZero‑Continuous* (Kumar et al., 2023) | 연속 행동 공간을 위한 MuZero 확장. |

## 구현 및 오픈소스
- **공식 구현**: DeepMind가 제공하는 TensorFlow 기반 MuZero 코드베이스 (GitHub: `deepmind/muzero`).  
- **오픈소스 포크**:  
  - `muzero-general` (PyTorch) – 다양한 환경(Atr‌​a, Mini‑Chess 등) 지원.  
  - `muzero-pytorch` – 학습 효율성을 위해 `torch.compile` 활용.  
- **라이브러리**: `muzero-py`, `muzero-jax` 등 다양한 프레임워크에 포팅된 구현이 존재한다.

> **Tip**: 학습 시 GPU 메모리 초과를 방지하려면 **`num_simulations`** 를 50~100 사이에서 시작하고, **`batch_size`** 를 점진적으로 늘리는 것이 일반적이다.

## 참고문헌
1. **Schrittwieser, J., et al.** “**Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model**.” *Nature* 2020.  
2. **Silver, D., et al.** “**AlphaZero: Mastering Chess, Shogi and Go without Human Knowledge**.” *Science* 2018.  
3. **Hafner, D., et al.** “**Learning Latent Dynamics for Planning from Pixels**.” *ICML* 2019.  
4. **Zhou, Y., et al.** “**EfficientZero: Scalable Reinforcement Learning with Efficient Planning**.” *NeurIPS* 2022.  
5. **Kumar, A., et al.** “**MuZero‑Continuous: Extending MuZero for Continuous Control**.” *ICLR* 2023.

## 외부 링크
- 📄 [MuZero 논문 (Nature) – PDF](https://www.nature.com/articles/s41586-019-1724-z)  
- 🖥️ [DeepMind MuZero GitHub Repository](https://github.com/deepmind/muzero)  
- 🧩 [MuZero‑General (PyTorch 구현)](https://github.com/werner-duvaud/muzero-general)  
- 🎓 [AlphaZero vs MuZero – 비교 영상 (DeepMind Channel)](https://www.youtube.com/watch?v=V9cQxgX3L84)  

---  

*이 문서는 2025년 현재 공개된 자료와 공식 발표 내용을 기반으로 작성되었습니다. 최신 연구 동향에 따라 내용이 업데이트될 수 있습니다.*