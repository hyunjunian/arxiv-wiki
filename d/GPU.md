# GPU (Graphics Processing Unit)

GPU(그래픽 처리 장치)는 대량의 병렬 연산을 수행하도록 설계된 프로세서이다. 초기에는 실시간 3D 렌더링을 가속화하기 위해 그래픽스 전용 하드웨어로 등장했지만, 범용 연산(GPGPU) 지원과 고성능 수치 연산을 위한 설계가 추가되면서 인공지능(AI) 분야에서도 핵심 가속기로 자리 잡았다. 특히 딥러닝 모델의 학습·추론에 필요한 대규모 행렬 연산을 효율적으로 처리함으로써 AI 연구·산업 전반에 큰 영향을 미치고 있다.

## 개요

- **정의**: 그래픽스 파이프라인을 처리하도록 설계된 병렬 연산 장치이며, 수천 개의 코어가 동시에 작업을 수행한다.  
- **핵심 특징**  
  - **대규모 병렬성**: 수천 개의 작은 연산 유닛(SM, Streaming Multiprocessor)으로 구성.  
  - **고대역폭 메모리**: GDDR6, HBM2·HBM3 등 고속 메모리 인터페이스 사용.  
  - **다양한 연산 정밀도**: FP32, FP16(half), BF16, Tensor Core 전용 INT8·INT4 등 AI에 최적화된 정밀도 지원.  
- **AI에서의 역할**: 딥러닝 모델 학습 시 대규모 행렬·벡터 연산을 가속하고, 추론 단계에서는 실시간 응답성을 확보한다.

## 역사와 발전

1. **초기 GPU (1990‑2000)**  
   - 1999년 NVIDIA의 GeForce 256이 ‘GPU’라는 용어를 처음 사용, 정점·픽셀 파이프라인을 통합.  
2. **GPGPU 시대 (2006‑2015)**  
   - NVIDIA CUDA(Compute Unified Device Architecture)와 AMD Stream SDK 발표, 일반적인 수치 연산을 GPU에서 수행 가능해졌다.  
   - 2012년 AlexNet이 ImageNet 대회에서 GPU를 이용한 학습으로 대폭 성능을 끌어올리며 딥러닝 붐을 촉발.  
3. **AI 전용 GPU (2016‑현재)**  
   - NVIDIA Volta(대형 텐서 코어) → Ampere(TF32, FP16 Tensor Core) → Hopper(H100) 등 AI 연산에 특화된 구조 도입.  
   - AMD는 CDNA(Radeon Instinct) 시리즈, Intel은 Xe-HPG·Xe-HPC 라인업을 통해 AI 가속 기능을 강화.

## GPU 아키텍처

| 구성 요소 | 역할 | AI와의 연관성 |
|-----------|------|----------------|
| **SM (Streaming Multiprocessor)** | 여러 CUDA 코어, 텐서 코어, 레지스터 파일 포함 | 수천 개 코어가 동시에 행렬 연산 수행 |
| **Tensor Core** | 행렬‑행렬 곱셈·누적(MMA) 전용 하드웨어 | FP16·BF16·TF32·INT8·INT4 연산 가속 |
| **메모리 서브시스템** | GDDR6, HBM2·HBM3 등 고대역폭 메모리 | 대규모 모델 파라미터와 입력 데이터를 빠르게 전송 |
| **L2 캐시·시계열 캐시** | 데이터 재사용 최소화 | 연산 중 데이터 로컬리티 향상 |
| **인터커넥트** | PCIe, NVLink, NVSwitch, Infinity Fabric | 멀티 GPU 간 데이터 전송 효율 향상 |

GPU는 이러한 구성 요소들을 계층적으로 배치하여 연산 병목을 최소화하고, 특히 텐서 코어는 딥러닝에서 핵심적인 행렬 연산을 10~100배 가속한다.

## 인공지능에서 GPU의 역할

- **학습(Training)**  
  - *대규모 데이터*와 *깊은 네트워크*를 병렬로 처리하여 학습 시간을 크게 단축.  
  - 혼합 정밀도 학습(Mixed Precision Training): FP16·BF16을 활용해 메모리 사용량과 연산량을 절감하면서도 정확도 유지.  
  - *분산 학습*: 여러 GPU를 네트워크로 결합(NVLink, Ethernet 등)해 모델 파라미터를 동기화하고 스케일을 확대.  

- **추론(Inference)**  
  - 실시간 서비스(음성인식, 이미지/비디오 분석, 챗봇 등)에서 낮은 레이턴시와 높은 처리량 확보.  
  - INT8·INT4 양자화(Quantization)와 텐서 코어 기반 가속으로 전력 효율을 극대화.  

- **특수 작업**  
  - *멀티모달 모델* (예: GPT‑4, CLIP)에서는 대규모 텍스트·이미지·음성 데이터를 동시에 처리.  
  - *강화학습(RL)*, *시뮬레이션 기반 최적화* 등 복합 연산에서도 GPU의 높은 병렬성이 유리.

## 주요 GPU 제품군

### NVIDIA

| 제품군 | 주요 모델 | 메모리 | FP32 TFLOPS | FP16 TFLOPS | 출시 연도 |
|--------|----------|--------|------------|------------|-----------|
| RTX 30 시리즈 | RTX 3090 | 24 GB GDDR6X | 35.6 | 71.2 (Tensor) | 2020 |
| A100 (데이터센터) | A100 40GB | 40 GB HBM2 | 19.5 | 312 (Tensor) | 2020 |
| H100 (Hopper) | H100 80GB | 80 GB HBM3 | 60 | 1000+ (Tensor) | 2022 |
| Jetson (엣지) | Jetson Orin | 32 GB LPDDR5 | 37.5 | - | 2022 |

### AMD

| 제품군 | 주요 모델 | 메모리 | FP32 TFLOPS | FP16 TFLOPS | 출시 연도 |
|--------|----------|--------|------------|------------|-----------|
| Instinct MI 시리즈 | MI250 X | 128 GB HBM2e | 47.9 | 383 (Matrix) | 2021 |
| Radeon PRO | W6800 | 32 GB GDDR6 | 21.3 | - | 2020 |
| Radeon RX | RX 7900 XT | 20 GB GDDR6 | 61.4 | - | 2022 |

### Intel

| 제품군 | 주요 모델 | 메모리 | FP32 TFLOPS | FP16 TFLOPS | 출시 연도 |
|--------|----------|--------|------------|------------|-----------|
| Xe‑HPG (Arc) | ARC A770 | 16 GB GDDR6 | 20.1 | - | 2022 |
| Xe‑HPC (Ponte Vecchio) | Ponte Vecchio | 408 GB HBM2e | 30+ | 120+ (Matrix) | 2023 |

## 소프트웨어 생태계

- **프로그래밍 모델**  
  - *CUDA*: NVIDIA GPU 전용 고성능 프레임워크, C/C++·Python 바인딩 제공.  
  - *ROCm*: AMD GPU용 오픈소스 스택, HIP(하이브리드 인터페이스) 제공.  
  - *OneAPI*: Intel GPU·CPU·FPGA를 위한 통합 프로그래밍 모델.  

- **딥러닝 라이브러리**  
  - *cuDNN*: CUDA 기반 딥러닝 연산 가속 라이브러리, Convolution·RNN·Pooling 지원.  
  - *cuBLAS*: 고성능 BLAS 연산, 행렬·벡터 연산 가속.  
  - *TensorRT*: 추론 최적화·양자화·배치 크기 조정 도구.  
  - *MIOpen*: AMD GPU용 딥러닝 연산 라이브러리.  

- **프레임워크 연동**  
  - TensorFlow, PyTorch, MXNet, JAX 등 주요 딥러닝 프레임워크는 CUDA 및 ROCm을 기본 백엔드로 지원한다.  
  - *NCCL*(NVIDIA Collective Communications Library)·*ROCm‑MPI* 등 멀티 GPU 통신 라이브러리로 분산 학습 효율을 높인다.  

- **툴링 및 디버깅**  
  - Nsight Systems/Compute, NVIDIA Visual Profiler, AMD Radeon™ GPU Profiler 등 성능 분석 도구 제공.  

## 성능 지표와 벤치마크

AI 워크로드에서 흔히 사용하는 주요 성능 지표는 다음과 같다.

| 지표 | 설명 |
|------|------|
| **TFLOPS (Tera FLOPS)** | 초당 10¹²개의 부동소수점 연산 수행량. FP32·FP16·Tensor Core 별로 구분. |
| **Memory Bandwidth (GB/s)** | GPU 메모리와 연산 유닛 간 데이터 전송 속도. |
| **Power Efficiency (TFLOPS/W)** | 와트당 연산 성능, 데이터센터 운영 비용에 직결. |
| **Throughput (samples/sec)** | 모델 추론 시 초당 처리 가능한 입력 샘플 수, 레이턴시와 함께 중요한 지표. |
| **Training Time (hours)** | 대규모 데이터셋(예: ImageNet) 전체 학습에 소요되는 시간. |

### 대표 벤치마크 (ResNet‑50, BERT, GPT‑3)

| GPU | ResNet‑50 (FP32) | BERT (FP16) | GPT‑3 (다중 GPU) | Power (W) |
|-----|-------------------|--------------|------------------|-----------|
| RTX 3090 | 145 samples/s | 320 samples/s | 8 GPU에서 1.2 TFLOPS | 350 |
| A100 40GB | 210 samples/s | 530 samples/s | 8 GPU에서 2.7 TFLOPS | 400 |
| H100 80GB | 460 samples/s | 1,200 samples/s | 8 GPU에서 8.5 TFLOPS | 700 |
| MI250 X | 190 samples/s | 420 samples/s | 8 GPU에서 2.4 TFLOPS | 300 |
| Ponte Vecchio | 340 samples/s | 950 samples/s | 8 GPU에서 5.6 TFLOPS | 500 |

※ 벤치마크는 최적화된 CUDA/ROCm 드라이버와 최신 프레임워크 버전 기준이며, 실제 성능은 워크로드와 설정에 따라 변동될 수 있다.

## 멀티 GPU 및 클러스터링

- **NVLink/NVSwitch**: GPU‑GPU 간 300 GB/s 이상의 고대역폭 인터커넥트, 데이터 복제·분산에 효율적.  
- **PCIe Gen4/Gen5**: 범용 인터페이스, 다수 GPU를 단일 서버에 장착 가능.  
- **InfiniBand**: 클러스터 수준의 고속 네트워킹, GPU Direct RDMA를 통해 직접 메모리 전송 지원.  
- **소프트웨어 스택**: *NCCL*, *Horovod*, *DeepSpeed*, *Megatron‑LM* 등 분산 학습 프레임워크가 GPU 간 통신을 자동으로 최적화한다.  

### 스케일링 전략

| 전략 | 특징 |
|------|------|
| **데이터 병렬(Data Parallelism)** | 동일 모델 복제, 입력 배치를 나눠서 처리. 파라미터 동기화 비용이 주요 병목. |
| **모델 병렬(Model Parallelism)** | 모델을 층·파라미터 단위로 분할, 큰 모델(예: 175 B 파라미터 GPT‑3) 학습에 필수. |
| **파이프라인 병렬(Pipeline Parallelism)** | 연산 흐름을 단계별로 분할, 각 GPU가 파이프라인 단계 담당. |
| **혼합 병렬(Hybrid Parallelism)** | 데이터·모델·파이프라인 병렬을 조합해 최적 성능 달성. |

## 에너지 효율 및 비용

- **전력 관리**: 최신 GPU는 동적 전압·주파수 스케일링(DVFS)과 전력 제한(Power Capping) 기능을 제공, 워크로드 특성에 맞춰 에너지 효율을 최적화한다.  
- **총소유비용(TCO)**: GPU 가격(수천 달러)·전력 소비·냉각 비용·소프트웨어 라이선스(예: CUDA 개발 툴킷) 등을 종합적으로 고려해야 한다.  
- **클라우드 활용**: AWS EC2 G5·P4, Azure ND·NC 시리즈, Google Cloud A2 등 GPU 인스턴스를 온디맨드 방식으로 사용하면 초기 투자 비용을 크게 낮출 수 있다.  

## 미래 전망

1. **아키텍처 혁신**  
   - *Hopper·H100* 이후 NVIDIA는 **Hopper‑2**(예정)와 **Ada**(멀티모달 전용) 라인업을 개발하며, 더 높은 텐서 연산 밀도와 레이턴시 감소를 목표로 한다.  
   - AMD는 **CDNA 3**(MI300X)와 **Infinity Fabric 3**를 통해 멀티 GPU 간 초고속 상호 연결을 강화한다.  

2. **칩렛·모듈형 설계**  
   - CPU와 GPU를 하나의 패키지에 집적하거나, GPU 코어와 메모리·I/O를 별도 칩렛으로 분리해 확장성을 높이는 방향이 활발히 연구되고 있다.  

3. **AI 전용 가속기와의 경쟁**  
   - Google TPU, Graphcore IPU, Cerebras Wafer‑Scale Engine 등 전용 AI 가속기가 등장했지만, GPU는 범용성·소프트웨어 에코시스템·가격 경쟁력 면에서 여전히 우위를 점하고 있다.  

4. **양자화·프루닝·스파스 연산**  
   - 4비트·2비트 정밀도와 스파스 행렬 연산을 네이티브 지원하는 텐서 코어가 도입되면서, 모델당 메모리·연산 요구량이 크게 감소할 전망이다.  

5. **에너지·환경 규제 대응**  
   - 데이터센터 전력 소비 감소와 탄소 배출량 저감을 위해, GPU 설계 단계에서 **전력 효율(PUE)**와 **친환경 소재** 사용이 강조될 것이다.  

## 참고 문헌

1. NVIDIA Corporation, *CUDA C Programming Guide*, 2024.  
2. AMD, *ROCm Platform Documentation*, 2023.  
3. J. Dean et al., “Large Scale Distributed Deep Networks,” *Advances in Neural Information Processing Systems*, 2012.  
4. K. He et al., “Deep Residual Learning for Image Recognition,” *CVPR*, 2016.  
5. T. Mikolov et al., “Efficient Estimation of Word Representations in Vector Space,” *arXiv preprint arXiv:1301.3781*, 2013.  
6. NVIDIA, *NVIDIA DGX A100 System Architecture*, Whitepaper, 2022.  
7. H. Chen et al., “A Survey on Deep Learning Accelerators: GPUs, TPUs and Beyond,” *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, 2023.  
8. Intel, *OneAPI Programming Guide*, 2024.  
9. M. Rastegari et al., “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” *CVPR*, 2018.  
10. S. Wang et al., “The Rise of AI-Optimized GPUs: Architectural Trends and Future Directions,” *Journal of Parallel and Distributed Computing*, 2024.

---  

*본 문서는 2025년 현재 공개된 자료와 주요 벤치마크를 기반으로 작성되었으며, 최신 제품 및 기술 발표에 따라 내용이 업데이트될 수 있다.*