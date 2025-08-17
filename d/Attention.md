# Attention (주의)

> **Attention**(주의)은 **인지·신경과학**, **인공지능**, **경제·사회학** 등 여러 분야에서 핵심적으로 다루어지는 개념이다. 인간·동물의 **정보 처리·선택·집중** 메커니즘을 가리키는 **심리·신경학적 의미**와, 머신러닝 모델에서 **가중치를 동적으로 할당**하는 **알고리즘적 메커니즘**을 모두 포함한다.  

---

## 목차
1. [어원 및 정의](#어원-및-정의)  
2. [심리학·신경과학에서의 Attention](#심리학·신경과학에서의-attention)  
   - 2.1. [주의의 종류](#주의의-종류)  
   - 2.2. [주요 이론·모델](#주요-이론·모델)  
   - 2.3. [신경 기초](#신경-기초)  
   - 2.4. [측정·연구 방법](#측정·연구-방법)  
   - 2.5. [관련 장애](#관련-장애)  
3. [인공지능에서의 Attention 메커니즘](#인공지능에서의-attention-메커니즘)  
   - 3.1. [기본 개념](#기본-개념)  
   - 3.2. [주요 종류](#주요-종류-1)  
   - 3.3. [대표 모델·아키텍처](#대표-모델·아키텍처)  
   - 3.4. [응용 분야](#응용-분야)  
4. [Attention Economy (주의 경제)](#attention-economy-주의-경제)  
5. [사회·문화적 파급효과](#사회·문화적-파급효과)  
6. [관련 용어·연관 주제](#관련-용어·연관-주제)  
7. [참고 문헌](#참고-문헌)  

---  

## 어원 및 정의 <a id="어원-및-정의"></a>

| 항목 | 내용 |
|---|---|
| **영어 어원** | 라틴어 *attendere* “to give heed to” → 중세 영어 *attenden* “to watch, observe” |
| **한국어** | ‘주의’, ‘집중’, ‘관심’ 등으로 번역되며 상황에 따라 “집중력”, “관심도” 등으로 사용 |
| **일반적 정의** | 외부·내부 자극 중 **특정 정보에 선택적으로 자원을 할당**하여 처리하는 정신·신경 과정[^1] |
| **전산학 정의** | 입력 시퀀스의 각 요소에 **가중치(Attention weight)** 를 부여해 **동적으로 정보를 강조**하는 연산 메커니즘[^2] |

---

## 심리학·신경과학에서의 Attention <a id="심리학·신경과학에서의-attention"></a>

### 주의의 종류 <a id="주의의-종류"></a>

| 종류 | 설명 | 주요 연구·실험 |
|---|---|---|
| **선택적 주의(Selective Attention)** | 다수의 자극 중 **특정 자극**을 선택해 처리하고 나머지는 억제 | Stroop test, Dichotic listening |
| **지속적 주의(Sustained Attention)** | **오랜 시간** 동안 동일한 자극에 집중 유지 | Continuous Performance Task (CPT) |
| **분산 주의(Divided Attention)** | 두 개 이상의 과제 **동시 수행** 시 자원 분배 | Dual‑task paradigm |
| **실행적 주의(Executive Attention)** | 목표‑지향적 **조절·전환** 기능, 전전두‑두정 피질 연관 | Stroop, Flanker task |
| **공간주의적 주의(Spatial Attention)** | 시야 내 **특정 위치**에 자원을 집중 | Posner cueing paradigm |
| **특성 기반 주의(Feature‑Based Attention)** | 색·운동·형태 등 **특정 특성**에 대한 선택성 | Feature‑integration theory |

### 주요 이론·모델 <a id="주요-이론·모델"></a>

| 모델 | 핵심 아이디어 | 주요 논문 |
|---|---|---|
| **필터 이론 (Broadbent, 1958)** | 감각 입력은 **초기 단계에서 물리적 특징**에 따라 필터링 | Broadbent, *Perception and Communication* |
| **감쇠 이론 (Treisman, 1964)** | 필터가 **완전 차단**이 아니라 **감쇠(attenuation)** 하여 약한 신호도 부분적으로 통과 | Treisman, *Verbal Cues, Language, and Meaning* |
| **후기 선택 이론 (Deutsch & Deutsch, 1963)** | 모든 정보가 **의미 수준**까지 처리된 뒤 선택 | Deutsch & Deutsch, *Attention: A Psychobiological Perspective* |
| **특성 통합 이론 (Treisman & Gelade, 1980)** | **전전두피질**이 “feature maps”를 결합해 **전체 객체**를 인식 | Treisman & Gelade, *A Feature-Integration Theory of Attention* |
| **Posner의 주의 네트워크 (1980)** | **전두엽(전두전전두피질)** → **두정피질** → **시각피질** 경로를 통한 **위치 이동** | Posner, *Orienting of Attention* |
| **다중자원 이론 (Kahneman, 1973)** | 인지 자원은 **용량이 제한**된 **다중 채널**에 분배 | Kahneman, *Attention and Effort* |

### 신경 기초 <a id="신경-기초"></a>

* **전두–두정 네트워크 (Frontoparietal Control Network)** – 실행적·전환적 주의 담당  
* **배측 주의 네트워크 (Dorsal Attention Network)** – 목표‑지향적, 공간적 선택  
* **복측 주의 네트워크 (Ventral Attention Network)** – 외부 자극에 대한 **놀람·전이** 반응(감각 ‘갱신’)  
* **시상(Thalamus)·시상하핵(Pulvinar)** – 전반적인 **감각 필터링** 역할  
* **시각 피질(V1–V4)·전두 전전두 피질(FEF)** – 시각·동작 연계 주의 조절  

> **뇌 영상 연구**(fMRI, PET)와 **전기생리학**(EEG/MEG)에서 P1, N1, N2pc, P300 등 ERP 성분이 **주의 배분**을 반영한다는 것이 입증되었다[^3][^4].

### 측정·연구 방법 <a id="측정·연구-방법"></a>

| 방법 | 주요 지표 | 장점·제한 |
|---|---|---|
| **행동 과제** (Stroop, Flanker, Posner cueing) | 반응 시간(RT), 오류율 | 간단·저비용, 인지 과정 추론에 한계 |
| **안구 추적(Eye‑tracking)** | fixation duration, saccade latency | 시각 주의 직접 관찰, 비시각 주의 측정 어려움 |
| **EEG/ERP** | P1, N1, N2pc, P300 등 | 높은 시간 해상도, 공간 해상도 제한 |
| **MEG** | 자기장 변동 기반 주의 파동 | 높은 시간·공간 해상도, 비용 높음 |
| **fMRI** | BOLD signal in frontoparietal·시상 하부 | 높은 공간 해상도, 시간 해상도 낮음 |
| **뇌 자극** (TMS, tDCS) | 특정 영역 억제·활성화 효과 | 인과관계 검증에 유용 |

### 관련 장애 <a id="관련-장애"></a>

* **주의결핍·과잉행동장애(ADHD)** – 지속·선택적 주의 결함, 전두‑두정 네트워크 기능 저하  
* **시공간 무시증(Unilateral neglect)** – 복측 주의 네트워크 손상으로 일방적 자극 무시  
* **주의력 장애(하위 유형)** – 알츠하이머·파킨슨병 등 신경퇴행성 질환에서 주의 저하가 조기 증상으로 나타남  

---

## 인공지능에서의 Attention 메커니즘 <a id="인공지능에서의-attention-메커니즘"></a>

### 기본 개념 <a id="기본-개념"></a>

전통적인 **RNN·CNN** 구조는 **고정된** 가중치와 **순차적** 정보 흐름에 의존한다.  
**Attention**은 **각 입력 요소**에 **동적으로 가중치**를 부여해 **중요도**를 학습하고, 이를 **가중합**해 **출력**을 생성한다.  

> **수식** (scaled dot‑product attention)  
> \[
> \text{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
> \]  

* \(Q\) – query 행렬  
* \(K\) – key 행렬  
* \(V\) – value 행렬  
* \(d_k\) – key 차원의 스케일링 팩터  

### 주요 종류 <a id="주요-종류-1"></a>

| 종류 | 특징 | 대표 논문/모델 |
|---|---|---|
| **Soft Attention** | 연속적 가중치 → 미분 가능, **엔드‑투‑엔드 학습** 가능 | Bahdanau et al., 2015 (Seq2Seq) |
| **Hard Attention** | 이산적인 선택(샘플링) → 비미분, **REINFORCE** 등 강화학습 사용 | Mnih et al., 2014 (RAM) |
| **Self‑Attention** | 동일 시퀀스 내 **자기‑쿼리/키/값** 사용, **전역 의존성** 포착 | Transformer (Vaswani et al., 2017) |
| **Multi‑Head Attention** | 여러 개의 독립 sub‑space에서 **동시** attention 수행 → 표현 다양성 ↑ | Transformer |
| **Cross‑Attention** | 서로 다른 시퀀스(예: Encoder‑Decoder) 사이에서 적용 | Encoder‑Decoder 구조 전체 |
| **Local / Windowed Attention** | 제한된 윈도우 내만 연산 → 계산량 감소 | Longformer, Swin Transformer |
| **Sparse / Adaptive Attention** | **희소 행렬** 기반, 입력 길이에 비례해 O(N)·O(N log N) 효율 | Reformer, Routing Transformer |

### 대표 모델·아키텍처 <a id="대표-모델·아키텍처"></a>

| 모델 | 주요 특징 | 적용 분야 |
|---|---|---|
| **Neural Machine Translation with Attention** (Bahdanau et al., 2015) | **Encoder‑Decoder**에 **soft attention** 삽입 → 번역 품질 크게 향상 | 기계 번역 |
| **Transformer** (Vaswani et al., 2017) | **Self‑attention만** 사용, **전역 병렬화** 가능, **멀티‑헤드** | NLP 전반 (BERT, GPT 등) |
| **BERT** (Devlin et al., 2018) | 양방향 **masked self‑attention** → 사전학습 → 다양한 다운스트림 태스크 | 문장 이해, 질의응답 |
| **GPT 시리즈** (OpenAI) | **단방향 causal attention** → autoregressive 생성 | 텍스트 생성, 대화형 AI |
| **Vision Transformer (ViT)** (Dosovitskiy et al., 2020) | 이미지 **패치**를 토큰화 후 **self‑attention** 적용 | 이미지 분류, 비전 작업 |
| **Swin Transformer** (Liu et al., 2021) | **계층적 윈도우 attention** → 고해상도 이미지 처리 | 객체 검출, 세그멘테이션 |
| **Longformer** (Beltagy et al., 2020) | **슬라이딩 윈도우 + 전역 토큰** → 긴 시퀀스 효율 처리 | 장문 문서 요약, 유전체 서열 분석 |

### 응용 분야 <a id="응용-분야"></a>

* **자연어 처리** – 기계 번역, 요약, 질의응답, 텍스트 생성, 감정 분석  
* **컴퓨터 비전** – 이미지 캡셔닝, 객체 검출, 영상 분할, 이미지 생성 (DALL·E)  
* **음성·오디오** – 음성 인식, 스피커 다이어리제이션, 음악 생성  
* **멀