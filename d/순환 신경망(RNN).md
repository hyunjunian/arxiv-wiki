# 개요

순환 신경망(Recurrent Neural Network, RNN)은 **시계열·시퀀스 데이터**를 처리하도록 설계된 인공 신경망 구조이다. 입력이 순차적으로 들어오고, 이전 단계의 출력을 현재 단계의 입력으로 다시 사용함으로써 **시간적인 의존성**을 모델링한다. 자연어, 음성, 주가 등 시간에 따라 변하는 데이터를 다루는 많은 AI 응용 분야에서 핵심적인 역할을 한다.

# 구조와 동작 원리

## 기본 아이디어

- **입력 $x_t$**, **은닉 상태 $h_t$**, **출력 $y_t$**가 각각 시간 단계 $t$마다 존재한다.
- 은닉 상태는 현재 입력과 이전 은닉 상태를 결합해 계산된다.
- 동일한 가중치 $W$와 $U$가 모든 시간 단계에 공유되어 파라미터 수를 크게 줄인다.

## 수식 표현

\[
\begin{aligned}
h_t &= \sigma\!\bigl(W_{ih}x_t + W_{hh}h_{t-1} + b_h\bigr) \\
y_t &= \phi\!\bigl(W_{ho}h_t + b_o\bigr)
\end{aligned}
\]

- $\sigma$ : 비선형 활성화 함수 (예: tanh, ReLU)  
- $\phi$ : 출력 층 활성화 (예: softmax)  

## 시간 전개(Unfolding)

RNN을 **시간 축으로 펼치면** $T$개의 동일한 신경망이 직렬로 연결된 형태가 된다. 이를 **시간 전개 그래프**라고 하며, 역전파 과정은 이 그래프 전체에 걸쳐 수행된다.

# 학습 방법

## 시간에 대한 역전파 (BPTT)

1. **순전파**를 통해 전체 시퀀스에 대한 은닉 상태와 출력값을 계산한다.  
2. **역전파**를 시간 역순으로 진행하면서 각 가중치에 대한 손실 기울기를 누적한다.  
3. 누적된 기울기를 이용해 **경사 하강법**(SGD, Adam 등)으로 파라미터를 업데이트한다.

## 기울기 클리핑

- 오래된 단계로 갈수록 기울기가 급격히 **소실**하거나 **폭발**할 수 있다.  
- **기울기 클리핑**(gradient clipping)을 적용해 일정 범위 이상으로 커지는 기울기를 제한한다.

# 주요 문제점

## 장기 의존성 학습 어려움

- 시퀀스가 길어질수록 초기 입력의 영향을 마지막 단계까지 전달하기가 힘들다.  
- 이는 **기울기 소실**(vanishing gradient) 현상으로, 전통적인 RNN이 오래된 정보를 기억하기 어려운 근본 원인이다.

## 폭발하는 기울기

- 가중치 행렬의 특성값(eigenvalue)이 1보다 크게 되면 역전파 시 기울기가 급격히 커져 **폭발하는 기울기**(exploding gradient) 문제가 발생한다.

# RNN의 변형

## LSTM (Long Short-Term Memory)

- **셀(state)**, **입력 게이트**, **망각 게이트**, **출력 게이트**로 구성되어 장기 의존성을 효과적으로 유지한다.
- 게이트 구조가 기울기 흐름을 조절해 소실·폭발 문제를 크게 완화한다.

## GRU (Gated Recurrent Unit)

- **리셋 게이트**와 **업데이트 게이트**만으로 LSTM보다 간결하지만 비슷한 성능을 보인다.
- 파라미터 수가 적어 학습이 빠른 장점이 있다.

## 양방향 RNN (Bidirectional RNN)

- 입력 시퀀스를 **정방향**과 **역방향**으로 동시에 처리해 양쪽 문맥을 모두 활용한다.
- 특히 **음성 인식**·**자연어 처리**에서 성능 향상이 두드러진다.

## 다층/깊은 RNN (Stacked RNN)

- 은닉층을 여러 개 쌓아 계층적 특징을 학습한다.
- 각 층이 서로 다른 시간 스케일을 포착하도록 설계할 수 있다.

## 기타 변형

- **Variational RNN**: 베이즈적 접근으로 불확실성을 모델링.  
- **Attention‑augmented RNN**: 특정 시점에 집중하도록 가중치를 동적으로 조정.  
- **IndRNN (Independent RNN)**: 각 뉴런이 독립적인 가중치를 갖게 하여 장기 의존성 문제를 완화.

# 응용 분야

- **언어 모델링**: 다음 단어 예측, 자동완성, 텍스트 생성.  
- **기계 번역**: Encoder‑Decoder 구조에서 순차적 인코딩/디코딩.  
- **음성 인식**: 음성 신호를 문자 혹은 음소 시퀀스로 변환.  
- **시계열 예측**: 주가, 날씨, 센서 데이터 등 연속적인 수치 예측.  
- **비디오 분석**: 프레임 간 동적 패턴 모델링.  
- **생성 모델**: 음악·그림·시놉시스 등 순차적 생성 작업.  

# 최신 연구 동향

- **Attention 메커니즘**과 **Transformer**가 RNN을 대체하거나 보완하는 흐름이 강해졌다.  
- **RNN‑CNN 하이브리드**: 공간적 특징은 CNN, 시간적 흐름은 RNN으로 처리해 영상·시계열 복합 데이터를 효과적으로 다룬다.  
- **경량화 및 저전력 RNN**: 모바일·IoT 디바이스에서 실시간 추론을 위한 양자화, 프루닝, 하드웨어 가속 기술이 활발히 연구되고 있다.  
- **메모리 효율적 구조**: 역전파 시 메모리 사용을 최소화하는 **Truncated BPTT**, **Reversible RNN** 등 새로운 학습 전략이 제안되고 있다.  

# 참고문헌

1. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735‑1780.  
2. Cho, K., et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*. EMNLP.  
3. Graves, A. (2013). *Speech Recognition with Deep Recurrent Neural Networks*. ICASSP.  
4. Bengio, Y., Simard, P., & Frasconi, P. (1994). *Learning Long-Term Dependencies with Gradient Descent is Difficult*. IEEE Transactions on Neural Networks.  
5. Vaswani, A., et al. (2017). *Attention Is All You Need*. NIPS. (Transformer와의 비교를 위한 참고)  

*이 위키 문서는 2025년 현재의 연구 흐름을 반영하여 작성되었습니다.*