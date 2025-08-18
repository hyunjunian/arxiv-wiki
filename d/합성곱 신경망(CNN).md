# 합성곱 신경망 (Convolutional Neural Network, CNN)

## 개요
합성곱 신경망(Convolutional Neural Network, CNN)은 **다층 퍼셉트론(MLP)** 에서 영감을 받아 이미지와 같은 격자(grid) 구조 데이터를 효율적으로 처리하도록 설계된 딥러닝 모델이다. CNN은 **지역 연결(local connectivity)**, **공유 가중치(weight sharing)**, **계층적 특징 학습(hierarchical feature learning)** 을 핵심 원리로 삼아, 입력 이미지에서 저수준 에지부터 고수준 객체까지 자동으로 특징을 추출한다.  

CNN은 1990년대 초반 **LeNet** 시리즈로 시작해 2012년 **AlexNet** 의 성공을 계기로 컴퓨터 비전 분야의 표준 모델로 자리 잡았으며, 현재는 의료 영상, 자율 주행, 자연어 처리 등 다양한 분야에 활용되고 있다.

## 역사적 배경
| 연도 | 주요 연구·시스템 | 핵심 기여 |
|------|------------------|-----------|
| 1980년대 | **Fukushima**의 Neocognitron | 계층적 특징 추출 구조 제안 |
| 1990년대 초 | **LeCun** 등, *LeNet-5* | 최초 실용적인 CNN 구현 (손글씨 인식) |
| 2006년 | 딥러닝 재부흥 (Deep Belief Networks) | 사전 학습(pre‑training) 아이디어 |
| 2012년 | **Krizhevsky** 등, *AlexNet* | GPU 기반 대규모 학습과 ReLU 활성화 도입으로 ImageNet 1위 |
| 2014년 | **Simonyan & Zisserman**, *VGG* | 깊고 균일한 구조 (3×3 컨볼루션) |
| 2014년 | **Szegedy** 등, *GoogLeNet* (Inception) | 멀티스케일 처리와 파라미터 효율 |
| 2015년 | **He** 등, *ResNet* | 잔차 연결(ResNet)으로 1000층 이상 학습 가능 |
| 2016년 이후 | **DenseNet**, **MobileNet**, **EfficientNet** 등 | 연결 구조 다양화 및 경량화 모델 개발 |

## 주요 구조와 구성 요소

### 합성곱 층 (Convolutional Layer)
- **필터(커널, kernel)** : 작은 학습 가능한 매트릭스로, 입력 데이터와 **컨볼루션 연산**을 수행한다.
- **스트라이드(stride)** : 필터 이동 간격; 일반적으로 1 또는 2.
- **패딩(padding)** : 입력 가장자리를 0 등으로 채워 출력 크기를 조절한다. *same* 패딩은 출력 크기를 입력과 동일하게 만든다.
- **특징 맵(feature map)** : 각각의 필터가 만든 출력 이미지.

> **공유 가중치** 덕분에 동일 필터가 전체 이미지에 적용되어 파라미터 수가 크게 감소한다.

### 활성화 함수 (Activation Function)
- **ReLU (Rectified Linear Unit)** : `f(x) = max(0, x)` 로 비선형성을 부여하고 학습을 가속화한다.
- 최근에는 **Leaky ReLU**, **ELU**, **Swish** 등이 사용된다.

### 풀링 층 (Pooling Layer)
- **맥스 풀링(max pooling)** : 각 영역에서 최댓값을 선택, 위치 불변성 향상.
- **평균 풀링(average pooling)** : 평균값 사용, 특징 부드럽게 유지.
- 보통 2×2 윈도우와 stride 2 로 차원 축소(다운샘플링) 효과를 얻는다.

### 정규화 층 (Normalization Layer)
- **배치 정규화(Batch Normalization)** : 각 미니배치에서 평균·분산을 정규화하여 학습 안정성 및 속도 향상.
- **레이어 정규화(Layer Normalization)**, **인스턴스 정규화(Instance Normalization)** 등도 활용된다.

### 완전 연결 층 (Fully Connected Layer)
- 마지막에 **Flatten** 혹은 **Global Average Pooling** 을 거쳐 1차원 벡터로 전환 후, 전통적인 다층 퍼셉트론 구조로 클래스 점수를 출력한다.

### 기타 구성 요소
- **드롭아웃(Dropout)** : 무작위로 뉴런을 비활성화해 과적합 방지.
- **잔차 연결(Residual Connection)** : 입력을 뒤쪽에 더해 학습이 깊어도 기울기 소실 방지 (ResNet).
- **인셉션 모듈(Inception Module)** : 서로 다른 크기의 필터를 병렬로 적용해 멀티스케일 특징 학습.

## 학습 및 최적화

### 손실 함수 (Loss Function)
- **분류** : 교차 엔트로피 손실 (Cross‑Entropy Loss)
- **회귀** : 평균 제곱 오차 (MSE)
- **다중 라벨/다중 클래스** : 시그모이드 교차 엔트로피, 소프트맥스 크로스 엔트로피 등

### 역전파와 최적화 알고리즘
| 알고리즘 | 특징 |
|--------|------|
| **SGD** (Stochastic Gradient Descent) | 기본적인 경사 하강법, 학습률 스케줄링 필요 |
| **Momentum** | 관성을 추가해 진동 감소 |
| **AdaGrad** | 개별 파라미터 학습률 조정 |
| **RMSProp** | 이동 평균을 이용해 학습률 자동 조정 |
| **Adam** | Momentum + RMSProp 결합, 현재 가장 널리 사용 |
| **AdamW** | 가중치 감쇠(Weight Decay)를 별도로 적용 |

### 정규화 및 일반화 기법
- **데이터 증강(Data Augmentation)** : 회전, 좌우 반전, 색상 변환 등으로 학습 데이터 다양성 확보.
- **라벨 스무딩(Label Smoothing)** : 과도한 확신을 억제해 일반화 향상.
- **Early Stopping** : 검증 손실이 상승하면 학습 중단.

## 응용 분야

### 컴퓨터 비전
- **이미지 분류** (예: ImageNet, CIFAR) – CNN 기반 모델이 최고 정확도 기록.
- **객체 탐지** – Faster R-CNN, YOLO, SSD 등.
- **시맨틱·인스턴스 분할** – FCN, U‑Net, DeepLab 등.
- **영상 초해상도** – SRGAN, ESRGAN 등.
- **동작 인식·비디오 분석** – 3D CNN, Two‑Stream Networks.

### 의료·생명 과학
- **병변 탐지** (X‑ray, CT, MRI) – 암, 폐렴 등 자동 진단.
- **세포·조직 이미지 분석** – 형태학적 특성 정량화.
- **약물 스크리닝** – 구조‑활성 관계 예측.

### 자연어 처리(NLP)
- **텍스트 분류** – 문자 수준 CNN이 짧은 문장에 효과적.
- **문장 임베딩** – Kim’s CNN, TextCNN 등.
- **시퀀스 라벨링** – CNN‑CRF 결합 모델.

### 음성·오디오
- **스펙트로그램 분석** – 음성 인식, 음악 장르 분류, 환경 소리 탐지.

### 로봇·자동차
- **자율 주행** – 카메라 기반 물체 인식·거리 추정.
- **로봇 조작** – 비전‑기반 피드백 제어.

## 구현 프레임워크와 라이브러리
- **TensorFlow** (tf.keras) – 대규모 분산 학습 지원.
- **PyTorch** – 동적 그래프와 직관적인 디버깅.
- **Keras** – 고수준 API, 빠른 프로토타이핑.
- **Caffe / Caffe2** – 이미지 처리에 최적화된 C++ 기반.
- **MXNet**, **Chainer**, **JAX** 등 – 연구·산업 현장 다양하게 활용.

### 간단한 Keras 예시 (Python)

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_simple_cnn(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_simple_cnn()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
```

## 최신 연구 동향 및 과제

### 경량화와 효율성
- **MobileNet**, **ShuffleNet**, **EfficientNet** 등은 **Depthwise Separable Convolution**·**Neural Architecture Search(NAS)** 를 활용해 모바일·임베디드 환경에 최적화된 모델을 제공한다.
- **Quantization**, **Pruning**, **Knowledge Distillation** 으로 파라미터와 연산량을 크게 감소시킨다.

### 해석 가능성 & 신뢰성
- **Grad‑CAM**, **Integrated Gradients** 등 시각화 기법으로 특징 맵을 해석.
- **Adversarial Robustness** : 적대적 공격에 대한 방어(방어적 훈련, 라벨 스무딩 등) 연구가 활발.

### 자동 설계와 메타러닝
- **Neural Architecture Search (NAS)** : 강화학습·베이지안 최적화·진화 알고리즘을 이용해 최적 구조 자동 발견.
- **Few‑shot & Zero‑shot Learning** : 적은 데이터 혹은 라벨이 없는 상황에서도 일반화 가능한 CNN 설계.

### 멀티모달 통합
- 이미지와 텍스트·오디오를 동시에 처리하는 **Multimodal CNN** 구조가 비디오 캡셔닝, VQA(Visual Question Answering) 등에 적용되고 있다.

### 지속 학습(Continual Learning)
- **Catastrophic Forgetting** 방지를 위한 **Replay**, **Regularization‑based** 기법들이 CNN 기반 시계열·비전 모델에 적용된다.

## 참고 문헌
- Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, “Gradient‑Based Learning Applied to Document Recognition,” *Proceedings of the IEEE*, 1998.
- A. Krizhevsky, I. Sutskever, G. Hinton, “ImageNet Classification with Deep Convolutional Neural Networks,” *Advances in Neural Information Processing Systems*, 2012.
- K. Simonyan, A. Zisserman, “Very Deep Convolutional Networks for Large‑Scale Image Recognition,” *International Conference on Learning Representations*, 2015.
- K. He, X. Zhang, S. Ren, J. Sun, “Deep Residual Learning for Image Recognition,” *Proceedings of the IEEE CVPR*, 2016.
- M. Tan, Q. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” *International Conference on Machine Learning*, 2019.
- J. Howard, S. Singh, “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,” *arXiv preprint arXiv:1704.04861*, 2017.
- 최신 리뷰: L. Du, J. Wang, “A Survey on Convolutional Neural Networks for Vision Applications,” *IEEE Transactions on Neural Networks and Learning Systems*, 2023.

*본 문서는 2025년 8월 현재까지 공개된 연구와 자료를 토대로 작성되었습니다.*