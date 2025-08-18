# Keras (케라스)

## 개요
Keras는 파이썬으로 작성된 고수준 딥러닝 API이며, 직관적인 인터페이스를 통해 신경망 모델을 빠르게 설계·학습·평가할 수 있도록 도와준다. 원래는 **프랑수아 숄레(François Chollet)** 가 2015년에 발표했으며, 현재는 **TensorFlow** 프로젝트에 공식적으로 포함돼 `tf.keras` 라는 형태로 사용된다. Keras는 **사용자 친화성**, **모듈식 설계**, **확장성**, **빠른 프로토타이핑**을 핵심 설계 목표로 삼고 있다.

| 항목 | 내용 |
|------|------|
| 최초 발표 | 2015년 3월 |
| 주요 개발자 | François Chollet, Google Brain 팀 등 |
| 지원 백엔드 | TensorFlow(주), Theano(구), CNTK(구) |
| 라이선스 | Apache License 2.0 |
| 공식 사이트 | https://keras.io/ |
| GitHub 저장소 | https://github.com/keras-team/keras |

## 역사
- **2015** – François Chollet가 Theano 위에 독립적인 고수준 API로 Keras를 공개. 직관적인 레이어와 모델 정의 방식을 강조.
- **2017** – TensorFlow 2.0에 맞춰 Keras가 공식 고수준 API로 채택, `tf.keras` 모듈이 추가되면서 TensorFlow와의 통합이 강화.
- **2020** – TensorFlow 2.x 시리즈에서 Keras가 기본 모델링 도구로 자리 잡으며, Eager Execution 및 `tf.data`와 같은 최신 기능과 원활히 연동.
- **2023~2024** – Keras 3 프로젝트가 진행 중이며, JAX와 PyTorch 백엔드 지원, 새로운 모델 서빙 파이프라인 등이 논의되고 있다.

## 설계 목표
- **사용자 친화적**: 최소한의 코드로 복잡한 모델을 정의할 수 있게 함.  
- **모듈식**: 레이어, 손실 함수, 옵티마이저 등을 독립적인 구성 요소로 제공해 재사용이 용이.  
- **확장성**: 커스텀 레이어·모델·콜백 등을 쉽게 구현 가능.  
- **다양한 백엔드 지원**: TensorFlow 외에도 과거 Theano·CNTK를 지원했으며, 최신 버전에서는 JAX·PyTorch와의 호환성을 목표로 함.

## 핵심 구성 요소

### 레이어 (Layers)
Keras는 `keras.layers` 모듈에 수백 개의 사전 구현 레이어를 제공한다. 예시는 다음과 같다.

- **Dense**: 완전 연결 레이어
- **Conv2D**, **Conv3D**: 2D·3D 합성곱 레이어
- **LSTM**, **GRU**, **SimpleRNN**: 순환 신경망 레이어
- **Embedding**: 단어 임베딩 레이어
- **BatchNormalization**, **Dropout**: 정규화·정규화 레이어

### 모델 정의 방식
1. **Sequential API**  
   레이어를 순차적으로 쌓아 간단한 구조를 구성할 때 사용.  
   ```python
   model = keras.Sequential([
       keras.layers.Dense(64, activation='relu', input_shape=(784,)),
       keras.layers.Dense(10, activation='softmax')
   ])
   ```

2. **Functional API**  
   복잡한 그래프(다중 입력·다중 출력, 스킵 연결 등)를 구현할 때 사용.  
   ```python
   inputs = keras.Input(shape=(784,))
   x = keras.layers.Dense(64, activation='relu')(inputs)
   x = keras.layers.Dropout(0.5)(x)
   outputs = keras.layers.Dense(10, activation='softmax')(x)
   model = keras.Model(inputs=inputs, outputs=outputs)
   ```

3. **Subclassing API**  
   `keras.Model`을 직접 상속해 `call` 메서드를 구현함으로써 자유로운 연산 흐름을 정의한다.  
   ```python
   class MyModel(keras.Model):
       def __init__(self):
           super().__init__()
           self.dense1 = keras.layers.Dense(64, activation='relu')
           self.dense2 = keras.layers.Dense(10, activation='softmax')

       def call(self, inputs):
           x = self.dense1(inputs)
           return self.dense2(x)

   model = MyModel()
   ```

### 옵티마이저 (Optimizers)
- **SGD**, **Adam**, **RMSprop**, **Adagrad** 등 다양한 최적화 알고리즘을 제공한다.  
- 하이퍼파라미터(learning rate, momentum 등)를 쉽게 조절 가능.

### 손실 함수 & 메트릭 (Losses & Metrics)
- **categorical_crossentropy**, **binary_crossentropy**, **mean_squared_error** 등 기본 손실 함수를 제공.
- **accuracy**, **precision**, **recall**, **AUC** 등 평가 메트릭을 `model.compile()` 시 지정한다.

### 콜백 (Callbacks)
학습 과정 중에 자동으로 호출되는 훅(Hook)들로, 다양한 부가 기능을 구현한다.

- **EarlyStopping**: 검증 손실이 개선되지 않을 때 학습 중단
- **ModelCheckpoint**: 모델 가중치를 주기적으로 저장
- **TensorBoard**: 시각화 로그 출력
- **LearningRateScheduler**: 학습률 스케줄링

### 데이터 전처리 & 파이프라인
`keras.preprocessing` 및 `tf.data`와 연동해 이미지·텍스트·시계열 데이터를 효율적으로 로드·전처리한다.

- **ImageDataGenerator**: 실시간 이미지 증강(Augmentation)
- **Tokenizer**, **pad_sequences**: 텍스트 토큰화와 시퀀스 패딩
- **tf.data.Dataset**: 대규모 데이터셋을 배치, 셔플, 프리패치하는 파이프라인 구축

## 모델 학습 및 평가 흐름
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=20,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

- `compile` 단계에서 **옵티마이저**, **손실 함수**, **메트릭**을 지정한다.
- `fit` 은 **배치**, **에폭**, **검증 분할**, **콜백** 등을 받아 모델을 학습한다.
- `evaluate` 로 테스트 데이터를 사용해 최종 성능을 측정한다.
- `predict` 로 새로운 샘플에 대한 추론을 수행한다.

## TensorFlow와의 통합
Keras는 현재 **TensorFlow 2.x**의 기본 고수준 API다. 주요 통합 포인트는 다음과 같다.

| 특징 | 설명 |
|------|------|
| **Eager Execution** | 즉시 실행 모드가 기본이며, 디버깅과 연구에 유리 |
| **tf.keras.Model** | TensorFlow의 `Model` 클래스를 그대로 사용해 Graph 모드와 Eager 모드 간 전환이 자유 |
| **Distribution Strategy** | `tf.distribute.MirroredStrategy` 등을 통해 다중 GPU·TPU 분산 학습 지원 |
| **SavedModel** | `model.save()` 로 TensorFlow SavedModel 포맷에 저장 가능, 서빙·배포가 쉬움 |
| **TensorBoard** | `tf.keras.callbacks.TensorBoard` 로 학습 로그를 손쉽게 시각화 |

## 커뮤니티와 생태계
- **GitHub**: 2024년 현재 ★ 70k+ 스타와 활발한 PR/Issue 트래픽  
- **Keras 공식 문서**: 사용자 튜토리얼, API 레퍼런스, 베스트 프랙티스 가이드 제공  
- **Stack Overflow**·**Reddit**·**TensorFlow Forum**: 활발한 Q&A 및 팁 공유  
- **교육 자료**: Coursera, Udacity, Fast.ai 등에서 Keras 기반 강좌가 널리 사용  
- **확장 패키지**: `keras-contrib`, `kerastuner`(하이퍼파라미터 탐색), `tf.keras.experimental` 등

## 주요 응용 분야
- **컴퓨터 비전**: 이미지 분류, 객체 탐지, 세그멘테이션 등  
- **자연어 처리**: 텍스트 분류, 시퀀스‑투‑시퀀스 번역, BERT 파인튜닝  
- **시계열·예측**: LSTM 기반 시계열 예측, 강화학습 환경  
- **생물정보학**: 단백질 구조 예측, 유전체 데이터 분석  
- **자동차·로보틱스**: 센서 데이터 융합, 제어 정책 학습

## Keras와 다른 딥러닝 프레임워크 비교

| 항목 | Keras (tf.keras) | PyTorch | MXNet | JAX |
|------|------------------|---------|-------|-----|
| **개발 난이도** | 매우 쉬움, 직관적인 API | 비교적 쉬움, Pythonic | 중간 | 고급 (함수형 프로그래밍) |
| **동적/정적 그래프** | 기본 Eager, `tf.function` 으로 정적 그래프 지원 | 완전 동적 (define-by-run) | 정적 + 동적 혼합 | 완전 JIT 기반 |
| **분산 학습** | `tf.distribute` 로 손쉽게 다중 GPU·TPU 지원 | `torch.distributed` 로 유연하지만 설정 복잡 | `mxnet` 자체 분산 기능 | `jax.pmap` 등으로 고성능 |
| **생태계** | TensorFlow와 깊은 연동, TFLite·TensorRT·TF.js | TorchVision·TorchAudio·TorchServe 등 | Gluon API, AWS와 연동 | DeepMind/Google 연구 중심 |
| **커스터마이징** | 커스텀 레이어·콜백·손실 함수 제공 | 자유로운 autograd, 커스텀 연산 구현 용이 | Gluon 인터페이스 | JIT 컴파일러 활용 고성능 구현 가능 |

## 예제 코드: MNIST 손글씨 숫자 분류 (CNN)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. 데이터 로드 & 전처리
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[..., None] / 255.0   # (N, 28, 28, 1) 형태로 변환
x_test  = x_test[..., None] / 255.0

# 2. 모델 정의 (Functional API)
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# 3. 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. 학습
model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,
    validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(patience=2)]
)

# 5. 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# 6. 추론 (예시)
import numpy as np
sample = np.expand_dims(x_test[0], axis=0)   # (1, 28, 28, 1)
pred = model.predict(sample)
print(f"Predicted digit: {np.argmax(pred)}")
```

> **핵심 포인트**  
> - `tf.keras` 를 사용하면 TensorFlow와 완전한 호환성을 유지하면서 Keras의 간결함을 누릴 수 있다.  
> - `EarlyStopping` 과 `ModelCheckpoint` 로 과적합을 방지하고 최적 모델을 자동 저장한다.  
> - 모델을 `SavedModel` 포맷(`model.save('my_mnist_model')`) 으로 저장하면 TensorFlow Serving 혹은 TensorFlow.js 로 손쉽게 배포 가능하다.

## 미래 전망
- **Keras 3** 프로젝트: JAX·PyTorch 백엔드 지원, `tf.keras` 와의 완전 호환성 유지 목표.  
- **모델 서빙·ONNX** 연동 강화: 다양한 플랫폼(TF Lite, TensorRT, Edge TPU)으로 모델을 원활히 이식.  
- **AutoML** 기능 부각: `KerasTuner`, `AutoKeras` 와 같은 하이퍼파라미터 자동 탐색 도구가 점차 통합될 전망.  
- **대규모 학습**: `tf.distribute` 와 `tf.data` 의 지속적인 최적화로 수천 개 GPU/TPU 클러스터에서도 효율적인 학습 가능.  
- **설명 가능 AI**: `tf-explain`, `keras-vis` 와 같은 시각화 툴이 기본 API와 더 깊게 통합될 가능성이 있다.

## 참고문헌
1. Chollet, F. (2015). *Keras*. https://github.com/fchollet/keras  
2. TensorFlow Team. (2024). *TensorFlow & Keras Documentation*. https://www.tensorflow.org/guide/keras  
3. Abadi, M., et al. (2016). *TensorFlow: Large‑Scale Machine Learning on Heterogeneous Distributed Systems*. https://arxiv.org/abs/1605.08695  
4. Keras Team. (2023). *Keras 3 Design Draft*. https://github.com/keras-team/keras/pull/??? (draft)  
5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. (Chapter 6: Convolutional Networks)  

---  

*이 문서는 Keras에 대한 일반적인 이해를 돕기 위해 작성된 위키 스타일 문서이며, 최신 내용은 공식 문서와 GitHub 저장소를 참고하시기 바랍니다.*