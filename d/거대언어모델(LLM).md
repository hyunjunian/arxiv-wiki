# 거대언어모델 (LLM)

## 정의  
**거대언어모델(Large Language Model, LLM)** 은 수십억~수조 개의 파라미터를 갖는 **딥러닝 기반** 언어 모델을 의미한다. 주로 **Transformer** 아키텍처를 기반으로 하며, 방대한 텍스트 코퍼스를 **자기 지도 학습(self‑supervised learning)** 방식으로 사전 학습(pre‑training)한다. 사전 학습이 끝난 뒤에는 **few‑shot**, **zero‑shot**, **instruction‑following** 등 다양한 추론 방식을 통해 자연어 이해·생성, 코딩, 번역 등 다목적 작업을 수행한다.[^1]

> LLM은 “언어를 이해하고 생성하는 범용 인공지능”에 가까운 형태로, 모델 규모와 데이터 양이 급격히 증가하면서 **인간 수준의 언어 능력**에 근접하는 성능을 보여주고 있다.

---

## 역사  
- **1940‑1950년대**: 초기 통계적 언어 모델(N‑gram) 등장.  
- **1990‑2000년대**: 신경망 기반 언어 모델이 제안되지만 규모는 작았다.  
- **2017**: **Transformer** 논문 발표, 셀프 어텐션(self‑attention) 메커니즘으로 효율적인 시퀀스 처리 가능.[^2]  
- **2018**: **BERT**(Bidirectional Encoder Representations from Transformers) 공개, 양방향 사전 학습으로 다양한 다운스트림 작업에서 혁신적 성능 달성.[^3]  
- **2020**: **GPT‑3** 공개, 175 B 파라미터로 **few‑shot** 학습 능력 입증.[^4]  
- **2022‑2023**: **ChatGPT**, **GPT‑4**, **PaLM**, **LLaMA**, **Mistral**, **Gemma** 등 초대형 모델이 연속적으로 발표되며 **멀티모달**·**지시 기반** 학습이 대두됨.[^5][^6][^7]  
- **2024** 이후: 효율적인 파인튜닝(LoRA, QLoRA), **RLHF**(Reinforcement Learning from Human Feedback) 기반 정렬(at alignment) 연구가 활발히 진행 중이다.[^8]

---

## 주요 모델  

### 대표 모델 및 특징  

| 모델 | 파라미터 수 | 출시 연도 | 주요 특징 | 공개 여부 |
|------|------------|----------|-----------|-----------|
| **BERT‑Base** | 110 M | 2018 | 양방향 사전 학습, 마스크드 LM | ✅ |
| **GPT‑3** | 175 B | 2020 | Few‑shot 학습, 완전 자동 회귀형 | ❌ (API) |
| **PaLM** | 540 B | 2022 | 대규모 체인‑오브‑생각(Chain‑of‑Thought) 지원 | ❌ (API) |
| **LLaMA 2** | 7 B ~ 70 B | 2023 | 오픈 소스, 높은 효율성 | ✅ |
| **Mistral‑7B** | 7 B | 2023 | 무압축(Non‑compressed) 모델, 높은 성능/파라미터 비율 | ✅ |
| **Gemma** | 2 B ~ 7 B | 2024 | 구글 DeepMind에서 제공, 저전력 기기 최적화 | ✅ |
| **GPT‑4** | >1 T (추정) | 2023 | 멀티모달 입력, 높은 정밀도, RLHF 기반 정렬 | ❌ (API) |
| **Claude 3** | ~100 B (추정) | 2023 | 안전성 강화, 대화형 최적화 | ❌ (API) |

[^1]: Brown, T. B., et al. (2020). *Language Models are Few-Shot Learners*. arXiv:2005.14165.  
[^2]: Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.  
[^3]: Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.  
[^4]: Brown, T. B., et al. (2020). *Language Models are Few-Shot Learners*.  
[^5]: OpenAI. (2023). *Introducing GPT‑4*. https://openai.com/gpt-4.  
[^6]: Google AI. (2022). *PaLM: Scaling Language Modeling with Pathways*. https://ai.googleblog.com/2022/04/palm-scaling-language-models.html.  
[^7]: Meta AI. (2023). *LLaMA: Open and Efficient Foundation Language Models*. https://ai.facebook.com/blog/large-language-model-llama/.  
[^8]: Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback*. arXiv:2203.02155.

---

## 작동 원리  

### Transformer 아키텍처  
1. **입력 토크나이징**  
   - BPE(Byte‑Pair Encoding)·WordPiece 등 서브워드 토크나이저를 사용해 텍스트를 토큰 시퀀스로 변환한다.  
2. **임베딩(Embedding)**  
   - 토큰을 고정 차원의 벡터(`embedding`)로 매핑하고, 위치 정보를 나타내는 **포지셔널 인코딩**을 더한다.  
3. **셀프 어텐션(Self‑Attention)**  
   - 각 토큰이 **쿼리(Q)**, **키(K)**, **밸류(V)** 로 변환되어 전체 시퀀스와의 유사도를 계산한다.  
   - 어텐션 가중치는 `softmax(QKᵀ / √d_k)` 로 구해지고, 이를 밸류에 곱해 토큰 간 종속성을 파악한다.  
4. **멀티‑헤드 어텐션(Multi‑Head Attention)**  
   - 여러 독립적인 어텐션 헤드를 병렬로 적용해 다양한 관계를 동시에 학습한다.  
5. **피드‑포워드 네트워크(Feed‑Forward Network, FFN)**  
   - 각 토큰을 독립적으로 변환하는 두 개의 선형 레이어와 비선형 활성화(ReLU/GELU)로 구성된다.  
6. **잔차 연결 및 레이어 정규화**  
   - 어텐션·FFN 출력에 입력을 더하고, 레이어 정규화로 안정성을 높인다.  

![Transformer Architecture](https://github.com/user/repo/raw/main/transformer.png){: .center }

> **시각화**: 위 그림은 기본 Transformer 블록을 나타내며, LLM은 이러한 블록을 수십~수백 개 쌓아 모델 규모를 확장한다.

### 자기 지도 학습(Self‑Supervised Learning)  
- **마스크드 언어 모델링(Masked LM)**: 입력 토큰 중 일부를 `[MASK]` 로 대체하고 원래 토큰을 예측한다. (예: BERT)  
- **연속 언어 모델링(Causal LM)**: 왼쪽 컨텍스트만 사용해 다음 토큰을 예측한다. (예: GPT 시리즈)  
- **다중 태스크 프리트레인**: 문장 순서 예측, NSP(Next Sentence Prediction) 등 부가 작업을 병합해 일반화 능력을 향상시킨다.

---

## 학습 방법  

| 학습 단계 | 핵심 내용 | 주요 논문·도구 |
|----------|----------|----------------|
| **사전 학습(Pre‑training)** | 대규모 비지도 텍스트 코퍼스(수백 GB~수 PB)로 셀프 어텐션 기반 모델 파라미터 학습 | GPT‑3, PaLM, LLaMA |
| **파인튜닝(Fine‑tuning)** | 특정 도메인·작업(질의응답, 감성 분석 등)에 맞춰 전체 혹은 일부 파라미터 업데이트 | BERT, T5 |
| **지시 기반 학습(Instruction Tuning)** | “사용자 지시를 따르는” 형태의 데이터셋으로 모델을 정렬 → 보다 자연스러운 대화형 응답 가능 | InstructGPT, FLAN‑T5 |
| **강화 학습(RLHF)** | 인간 피드백을 보상 신호로 활용해 모델의 행동을 최적화 (예: 유해 콘텐츠 억제) | *Training language models to follow instructions with human feedback* |
| **효율적 파인튜닝** | LoRA, QLoRA, Adapter 등 파라미터 효율성을 높이는 경량화 기법 | *LoRA: Low‑Rank Adaptation of Large Language Models* |

---

## 응용 분야  

- **자연어 생성(NLG)**  
  - 기사·보고서 자동 작성, 창작 스토리텔링  
- **기계 번역**  
  - 다국어 번역 서비스(구글 번역, DeepL)  
- **텍스트 요약**  
  - 긴 문서·논문 요약, 뉴스 헤드라인 생성  
- **검색 및 질의응답**  
  - 검색 엔진 강화, 기업 내 지식베이스 QA  
- **코드 생성·보조**  
  - GitHub Copilot, 코딩 인터뷰 문제 해결  
- **의료·법률**  
  - 임상 기록 자동 요약, 법률 문서 검토 지원  
- **교육**  
  - 맞춤형 튜터링, 학습 자료 생성  
- **멀티모달**  
  - 이미지·텍스트 결합(예: GPT‑4 Vision, Flamingo)  

> **특징**: LLM은 **zero‑shot**·**few‑shot** 능력 덕분에 기존에 별도의 데이터셋이 필요했던 작업을 최소한의 프롬프트만으로 수행할 수 있다.

---

## 윤리와 사회적 영향  

- **편향(Bias) 및 차별**  
  - 학습 데이터에 내재된 사회적 편향이 모델 출력에 반영될 위험이 있다.  
- **허위 정보 및 악용**  
  - 사실과 다른 텍스트를 생성하여 가짜 뉴스·피싱 등에 악용될 가능성이 존재한다.  
- **프라이버시**  
  - 모델이 훈련 데이터에 포함된 개인 정보를 기억·재생산할 위험.  
- **에너지·탄소 발자국**  
  - 초대형 모델 학습에 수천 kWh 전력이 소요되며, 환경 영향이 점점 중요해지고 있다.[^9]  
- **법적·규제적 이슈**  
  - 저작권, 책임 소재, AI 안전 규제 등 다양한 법적 논쟁이 진행 중이다.  

> **대응 방안**: 모델 정렬(alignment) 연구, 투명성 보고서 공개, 지속 가능한 훈련 방법(효율적 파라미터 공유, 프루닝) 등이 제시되고 있다.

---

## 한계와 도전 과제  

- **환각(Hallucination)**: 실제 존재하지 않는 사실을 자신감 있게 생성한다.  
- **해석 가능성(Interpretability)**: 내부 메커니즘이 블랙박스로 남아 있어 의사결정 과정을 설명하기 어렵다.  
- **데이터 품질**: 웹 크롤링 데이터에 포함된 노이즈·중복이 모델 성능에 부정적 영향을 미친다.  
- **지속 가능한 스케일링**: 파라미터 수를 무한정 늘리면 비용·환경 부담이 급증한다.  
- **다중 언어·문화 포괄성**: 저리소스 언어와 문화에 대한 지원이 여전히 부족하다.  
- **보안**: 역공학·프롬프트 주입 공격 등에 대비한 방어 메커니즘이 필요하다.  

---

## 향후 전망  

- **멀티모달 LLM**: 텍스트·이미지·오디오·비디오를 동시에 다루는 **통합형 모델**이 대세가 될 전망이다.  
- **스파스(분산) 학습**: **Mixture‑of‑Experts(MoE)**, **경량 파라미터 공유** 등으로 효율성을 극대화하고, 동일한 파라미터 수 대비 성능을 크게 상승시킬 수 있다.  
- **신뢰성 및 정렬 강화**: RLHF·DAIR(Direct AI‑Feedback) 등 인간 피드백을 통한 지속적 정렬 방법론이 표준화될 것이다.  
- **오픈 소스 생태계**: LLaMA‑2, Mistral, Gemma와 같은 공개 모델이 **커뮤니티 중심** 개발을 촉진하며, 산학·연구 협업이 확대될 전망이다.  
- **규제와 표준**: 국제 기구(UNESCO, ISO 등)와 각국 정부가 **AI 윤리·안전 가이드라인**을 제정하고, 인증 체계가 도입될 가능성이 높다.  
- **응용 특화 모델**: 의료·법률·금융 등 특정 도메인에 최적화된 **전문 LLM**이 등장해 고도화된 의사결정 지원을 제공할 것이다.  

---

## 참고 자료  

[^9]: Strubell, E., Ganesh, A., & McCallum, A. (2019). *Energy and Policy Considerations for Deep Learning in NLP*. ACL. https://arxiv.org/abs/1906.02243  

1. Brown, T. B., et al. (2020). *Language Models are Few-Shot Learners*. arXiv:2005.14165.  
2. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.  
3. Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.  
4. OpenAI. (2023). *Introducing GPT‑4*. https://openai.com/gpt-4.  
5. Google AI Blog. (2022). *PaLM: Scaling Language Modeling with Pathways*. https://ai.googleblog.com/2022/04/palm-scaling-language-models.html.  
6. Meta AI. (2023). *LLaMA: Open and Efficient Foundation Language Models*. https://ai.facebook.com/blog/large-language-model-llama/.  
7. Touvron, H., et al. (2023). *LLaMA‑2: Open Foundation Language Models*. arXiv:2307.09288.  
8. Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback*. arXiv:2203.02155.  
9. Strubell, E., Ganesh, A., & McCallum, A. (2019). *Energy and Policy Considerations for Deep Learning in NLP*. ACL. https://arxiv.org/abs/1906.02243.  
10. Liu, Y., et al. (2023). *LoRA: Low‑Rank Adaptation of Large Language Models*. arXiv:2106.09685.  

---  

*본 문서는 2025년 8월 기준 최신 연구와 공개 자료를 기반으로 작성되었습니다.*