---
layout: default
title: "Day 12 CLIP — 00 · 검증 팩트시트 — CLIP (Learning Transferable Visual Models From Natural Language Supervision)"
paper_id: clip
date: 2026-10-23
---

# 00 · 검증 팩트시트 — CLIP (Learning Transferable Visual Models From Natural Language Supervision)

> **검증 방법**: arXiv v1 **LaTeX 원본 전체**(`src/clip_paper.tex` — arxiv.org/e-print/2103.00020 압축 해제)를 통째로 대조. 수치가 LaTeX 소스에 그대로 있으므로 PDF 중간 추정 없이 원문 라인 그대로 인용한다.
> **표·그림 인용 규약**: 컴파일 번호는 라벨 순서 역산 시 오류 위험이 있어 **LaTeX 라벨**로 인용한다. 확정 예: `zeroshot_table`은 본문 첫 `\begin{table}` = **논문 Table 1**. 나머지는 라벨(`table:robustness`, `table:action`, `sec:linear-probe` 등) + 캡션 문구로 식별.
> **레포**: `openai/CLIP` — README·모델카드 수치는 "README 수치"로 별도 라벨, 논문 수치와 혼용 금지. 원칙: *"every number must come from the paper's own table/abstract or the official README, with conditions attached"*.

### 1.1 신원

- arXiv: [2103.00020](https://arxiv.org/abs/2103.00020)
- 저자: Alec Radford†, Jong Wook Kim†(공동 1저자), Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever — **OpenAI** (`clip_paper.tex` icmlauthorlist)
- Venue: **ICML 2021** (PMLR v139, `proceedings.mlr.press/v139/radford21a` — 카메라레디 PDF를 직접 받아 확인)
- 공식 레포: [github.com/openai/CLIP](https://github.com/openai/CLIP) · 라이선스 **MIT** Copyright (c) 2021 OpenAI (repo LICENSE 원문 확인)
- 가중치: `clip.load(name)`이 OpenAI 서버에서 공식 다운로드

### 1.2 CLIP 핵심 수치 — 모두 원문 위치 표기

**(a) 데이터 — WIT (§2.1)**

| 항목 | 수치 |
|---|---|
| 총량 | **400 million (image, text) pairs** — 인터넷 공개 출처 |
| 쿼리 | **500,000 queries**: 영문 위키백과에서 ≥100회 등장 단어 + 고PMI bi-gram + 검색량↑ 위키 문서명 + 미포함 WordNet synset(각주) |
| 쿼리당 | **최대 20,000 pairs/query**로 근사 클래스 균형 |
| 단어 규모 | WebText(GPT-2 학습 데이터)와 "similar total word count" |
| 참고(기존 데이터) | MS-COCO·Visual Genome 각 ~100k 사진; YFCC100M 100M → 영문 자연어 제목/설명 필터 후 **~15M**(≈6배 축소, ImageNet 규모) |

> 본문: *"we constructed a new dataset of 400 million (image, text) pairs ... similar total word count as the WebText dataset used to train GPT-2 ... We refer to this dataset as WIT for WebImageText."*

**(b) 학습 목표·과정 (§2.2)**

- 배치 N개 (image, text) 페어 → **N×N 가능 페어 중 실제 페어를 예측**; 실제 쌍의 코사인 유사도 최대화, 나머지 N²−N 쌍 최소화.
- 손실 = 유사도 점수에 대한 **대칭 교차엔트로피(symmetric cross entropy)**; 그림 `pseudocode`에 numpy 유사 코드.
- 계승: multi-class N-pair loss (Sohn 2016) → InfoNCE (van den Oord 2018) → ConVIRT (Zhang 2020).
- **비선형 projection 제거** — 인코더 표현 → 멀티모달 임베딩 공간으로 **선형 projection만** ("did not notice a difference in training efficiency").
- **temperature τ를 학습으로 최적화**(log-parameterized); 초기값 Wu 2018의 **0.07 환산치**; 학습 불안정 방지 위해 **logit 스케일 ×100 클립** — §4 문단 + 부록 하이퍼파라미터 표 `Maximum temperature 100.0`.
- 증강 = **"random square crop from resized images"만**; 문자 개정을 제거한 텍스트 파이프라인.

**(c) 아키텍처 (§2.4)**

| 요소 | 제원 |
|---|---|
| 텍스트 인코더 | **63M 파라미터, 12층, 폭 512, 8 heads** Transformer |
| 토큰화 | 소문자 BPE. **§2.4 본문 = "49,152 vocab", 부록 하이퍼파라 표 = "Vocabulary size 49408"** — 불일치(§1.4-1) |
| 최대 시퀀스 | **76** 캡 + `[SOS]`/`[EOS]` 브래킷 → README 기본 `clip.tokenize(..., context_length=77)` |
| 텍스트 피처 | **[EOS] 위치 최상위 활성화** → layer norm → linear projection. 마스크드 self-attention(LM 초기화 여지) |
| 이미지 RN 계열 | RN50 베이스, **ResNet-D 개선 + antialiased rect-2 blur pooling + attention pooling**(query를 이미지 전체 GAP로 하는 1층 QKV attention) |
| 이미지 ViT 계열 | ViT + patch/position 임베딩 합친 뒤 **layer norm 1층 추가** + 다른 초기화 |
| 스케일링 | ResNet: 폭·깊이·해상도 **균등 배분**(EfficientNet식 단순화); 텍스트: **폭만** 스케일 |

**(d) 모델 변형·연산 (§4 세팅 문단)**

| 항목 | 값 |
|---|---|
| 모델 세트 | **8개 모델** — "training a series of eight models spanning almost 2 orders of magnitude of compute"(§1) |
| 변형 | RN50 · RN101 · **RN50x4 · RN50x16 · RN50x64**(RN50의 약 4×/16×/64× 연산) · ViT-B/32 · ViT-B/16 · ViT-L/14 + **ViT-L/14@336px(FixRes식 1 에폭 추가 사전 학습)** |
| 기본 "CLIP" | **ViT-L/14@336px** — *"Unless otherwise specified, all results reported ... as 'CLIP' use this model"* |
| 에폭 | **32 epochs**(전 모델) |
| 배치 | **32,768**(§4 본문 + 부록 하이퍼파라 표 `Batch size 32768`) |
| 학습 비용 | **RN50x64 = 592 V100 GPU × 18일** / **최대 ViT = 256 V100 × 12일** |
| 처리 총량 | **12.8B 이미지 × 32 에폭** — §7: "1장/초면 **405년**" |
| 나머지 | decoupled weight decay Adam + cosine LR, mixed-precision, gradient checkpointing, half-precision Adam 통계, GPU별 부분 유사도 연산 |

**(e) Zero-shot 전이 (§3.1.1)**

| 항목 | 값 | 출처 |
|---|---|---|
| 절차 | 이미지·텍스트(클래스명) 임베딩 → **코사인 유사도 × τ → softmax**; "multinomial logistic regression classifier with L2-normalized inputs, L2-normalized weights, no bias, temperature scaling" 관점; **텍스트 인코더 = 하이퍼네트워크**(텍스트로 선형분류기 가중치 생성) 해석; 분류기 **캐시** 후 재사용 | §3.1.1 본문 |
| Visual N-Grams 비교 | 열 = **aYahoo / ImageNet / SUN**: Visual N-Grams 72.4 / 11.5 / 23.0 → **CLIP 98.4 / 76.2 / 58.5** | `zeroshot_table` = 논문 **Table 1** |
| ImageNet zero-shot | **11.5% → 76.2%**(Visual N-Grams 대비), top-5 **95%** — "matches the performance of the original ResNet-50"(1.28M 라벨 미사용), top-5은 Inception-V4 평형 | Table 1 + 본문 |
| aYahoo / SUN | aYahoo 오류 **95% 감소**, SUN "more than doubles" | §3.1.1 본문 |
| 27 데이터셋 vs "RN50 피처 + fully supervised 정규화 로지스틱 회귀" | **16/27 승**(ImageNet 포함) | Fig. `zeroshot_vs_supervised` |
| 개별 데이터셋 | **STL10 99.3%**(zero-shot, "appears to be a new state of the art") · Stanford Cars·Food101 +20%p↑ · Flowers102·FGVCAircraft −10%p↓ · Kinetics700 +14.5%(full frame) · UCF101 +7.7%(RN50 대비) | §3.1.2 본문 |
| 약점 데이터셋 | EuroSAT · RESISC45(위성) · PatchCamelyon(림프절 종양) · CLEVRCounts(개수) · GTSRB(교통 표지) · KITTI Distance — "quite weak on several specialized, complex, or abstract tasks" | §3.1.2 본문 |

**(f) 비디오 행동 — 부록 표 (`table:action`)**

| 세팅 | UCF101 Top-1 | K700 AVG | RareAct mWAP | RareAct mWSAP |
|---|---|---|---|---|
| Finetune 베스트 | R(2+1)D-BERT **98.7** | NS ENet-L2 **84.8** | – | – |
| Linear | **CLIP 92.0** | **CLIP 73.0**(NS ENet-L2 68.2) | – | – |
| **Zero-shot** | **CLIP 80.3** | **CLIP 69.6** | **CLIP 40.7** | **CLIP 44.8**(S3D 30.5/34.8) |

- 본문 컨텍스트: 비디오 → **단일 프레임 서브샘플** 평가(과소추정 명시); K700 zero-shot은 **전 프레임 평균**, "fully supervised I3D(545,000 라벨 비디오)와 1% 이내"; RareAct **+10 포인트** 갱신.

**(g) 프롬프트 엔지니어링·앙상블 (§3.1.1)**

- 문제 2가지: **다의성**(crane(기계/새, ImageNet에 둘 다 포함), boxer(견종/선수, Oxford-IIIT Pet)) + 학습 분포 격차(텍스트는 대개 문장).
- 기본 템플릿 **"A photo of a {label}."** 단독 = **ImageNet +1.3%**.
- 카테고리 지시형: Pets "a type of pet" / Food101 "a type of food" / FGVC Aircraft "a type of aircraft" / OCR = 따옴표 / 위성 = "a satellite photo of a {label}."
- **앙상블은 probability가 아니라 임베딩 공간에서** — "cache a single set of averaged text embeddings so that the compute cost of the ensemble is the same as using a single classifier when amortized" — **80개 컨텍스트 프롬프트 앙상블 = ImageNet +3.5%**(단일 기본 템플릿 대비) → 프롬프트 엔지니어링+앙상블 합쳐 **약 +5%**.
- 36 데이터셋 평균: 컨텍스트 없는 라벨명 기준선 대비 **~5 points** — "similar to the gain from using 4 times more compute, but is 'free'" — Fig. `prompt_engineering`.

**(h) Zero-shot vs few-shot (§3.1.2)**

- **같은 피처 공간의 4-shot 로지스틱 회귀와 평균 성능 동등**; 공개 최고 **16-shot**(BiT-M ResNet-152x2, ImageNet-21K 학습)을 "roughly matches" — Fig. `zeroshot_vs_fewshot`(클래스당 ≥16예제인 20개 데이터셋):
  - CLIP zero-shot이 few-shot 정규화 분류기보다 나은 데이터셋 **12개**, few-shot이 나은 **15개**.
- 직관(0-shot<1-shot)과 다른 이유 해석: zero-shot 분류기는 자연어로 개념을 **"communicated"**; 1-shot은 음수/양수 가설 모호성 문제.
- 후속 연구 링크: zero-shot 가중치를 few-shot 학습의 L2 prior로 → "just" zero-shot 남는 관측.

**(i) 데이터 효율 해석 (Fig. `zeroshot_data_efficiency`)**

- 같은 CLIP 피처 공간에서 zero-shot 성능을 맞는 데 필요한 **라벨 예제/클래스 수**(1·2·4·8·16-shot + fully supervised 로그보간): **1-shot 미달 2개 ~ 최대 추정 184개/클래스**(CIFAR-10 CIFAR-100? — 본문 캡션 문구 기준) — "median around **5.4 examples** per class".

**(j) 표현 학습 — 선형 분류 (§3.2)**

- 방식 선택 이유: fine-tuning은 사전학습 실패를 마스킹하기 쉽고, 선형 분류는 실패를 드러냄 + zero-shot 분류기와 동형 비교 가능 + **66 모델 × 27 데이터셋 = 1,782 evaluations** 공정 통제.
- Suite: Kornblith 12 + 확장 15 = **27** — MNIST · FER2013 · STL-10 · EuroSAT · NWPU-RESISC45 · GTSRB · KITTI · PatchCamelyon(2클래스) · UCF101 · Kinetics700 · CLEVRCounts(2,500샘플) · Hateful Memes · ImageNet-1k(COCO 파일? ❓; **불명** 부분은 부록 `sec:linear-probe` §A 직접 확인) + **신규 제안데이터 Country211**(211국가 × 200/100)·**Rendered SST2**(SST2 문장을 픽셀 448×448 렌더링, OCR·표현 분석).
- 결과: **Best CLIP ViT-L/14@336px** — "12데이터셋 기존 최고 대비 **+2.6%**" · 확장 27 스위트 **+5%** · ViT가 CLIP ResNet보다 **약 3× 연산 효율** · Fig. `linear-probe-clip-vs-enet`: **21/27 데이터셋에서 CLIP 피처 > Noisy Student EfficientNet-L2 피처** — 개선 대상 = OCR · 지리(Country211, SUN397) · 비디오(K700, UCF101) · 표지(GTSRB +14.7%?) — **(x) 표시 부분은 원문 라벨 표를 직접 열어 확인 요구**.
- 경계 명시: NS EfficientNet-L2가 이기는 것은 ImageNet/CIFAR10-100/PatchCamelyon/CLEVRCounts 등 — "remains strongest ... on the datasets that it was explicitly trained on"(§3.2 문맥).

**(k) Robustness — 자연 분포 이동 (§3.3)**

- 이론 입력: Taori 2020 — IN 정확도가 OOD 정확도를 예측(log-log) → **effective robustness**(예측선 위) vs relative robustness 구분.
- **Zero-shot CLIP은 effective robustness gap을 최대 75% 축소** — Fig. `robust_main_fig` 캡션; 최고 = ViT-L/14@336px vs ResNet-101 대비.
- **ImageNet 지도 적응(+9.2%)은 robustness 무개선** — V2 증가, ImageNet-R **−4.7** · ObjectNet **−3.8** · Sketch **−2.8** · ImageNet-A **−1.9** (Fig. `robustness_interventions` 캡션 + 본문); Vid/YTBB 개선 미미.
- Per-dataset 커스텀 zero-shot 분류기 교체: 평균 effective robustness **+5%**, ObjectNet **+2.3%** (개선 일부 집중).
- Few-shot(16-shot) 선형 분류기: effective robustness 개선이 zero-shot보다 약함 — "high effective robustness seems to result from **minimizing the amount of distribution specific training data**."
- 부록 표 `table:robustness`: **Zero-Shot CLIP = 76.2 · 70.1 · 77.2 · 88.9 · 72.3 · 60.2**(IN / IN-V2 / IN-A / IN-R / ObjectNet / IN-Sketch); IN-Vid PM0/PM10 = **95.3 / 89.2**, YTBB PM0/PM10 = **95.2 / 88.5** — 대응 NS EfficientNet-L2 = 88.3 · 80.2 · 84.9 · 74.7 · 68.5 · 47.6 · 88.0/82.1 · 67.7/63.5; FixResNeXt101-32x48d V2 = 86.4 · 78.0 · 68.4 · 80.0 · 57.8 · 59.1 · 85.8/72.2 · 68.9/57.7.
- 부록 서술: ImageNet-Vid/DOTAT? 등 "5 of the 7 datasets에서 SOTA 갱신" — Vid·YTBB 등은 zero-shot을 key 컨셉으로 취급.

**(l) 인간 비교 (§5)**

- 인간 zero→1-shot 성능은 대개 더 나옴("I don't know" 미기재 참여자 확정 — 성능 상승 성격 차이 명시); CLIP의 zero→few-shot 성능에서 **하락 존재**.
- 어려운 이미지 분포에 대한 정성 분석 Fig. `clip_human_difficulty_fig`.
- 표 캡션: "Most of the gain in performance when going from the human zero shot case to the human one shot case is on images that participants were highly uncertain on."

**(m) 데이터 중복 (§6)**

- 근사 near-duplicate 임베더로 검출 시 **35 데이터셋 중 5개**만이 99.5% Clopper-Pearson CI로 의미 있음 — **그중 2개는 중복 데이터에서 오히려 하락**. 전체 정확도 추정 증가 최대 **+0.6% Birdsnap**.
- 검출기 학습: **fixed τ=0.07 InfoNCE**(증강 파이프라인 재사용) — §6 본문.

**(n) 공정성·사회 편향 (Broader Impacts)**

- **FairFace** 분석: 레이스·젠더·나이 분류 시각 + White급 명확한 편향 — 상세 수치는 원문 표를 직접 열 것(이 시트는 성능 수치 중심으로 절제).
- 감시 사용 가능성 논의: OCR·행동 인식·지리 등 — §8 텍스트 통찰.

**(o) 규모 경향 (Fig. `zeroshot_scaling`)**

- "39 evals on 36 different datasets, average zero-shot error is well modeled by a log-log linear trend across a **44× range of compute spanning 5 different CLIP models**".
- §7: "zero-shot CLIP이 전반적 SOTA에 도달하는 데 **약 1000× 연산** 추정" — 무한 스케일 규약이 아님(명시 경계).

**(p) §7 Limitations 추가 핵심**

- OOD 대조 예: **Rendered SST2은 잘되는데 MNIST는 88%** — *"An embarrassingly simple baseline of logistic regression on raw pixels outperforms zero-shot CLIP"*.
- 출력 유연성 한계: *"limited to choosing from only those concepts in a given zero-shot classifier"*(캡션 생성 아님).
- 개발 중 full validation set 반복 조회(진짜 zero-shot 아님) + 27-스위트가 *"undeniably co-adapted with the development and capabilities of CLIP"*.
- *"CLIP models learn many social biases ... training data was unfiltered/uncurated"*(표 + §8 연결).

### 1.3 이 데모의 설계 근거 (03_실습 활용)

1. **배치 N×N 대칭 교차엔트로피** — `pseudocode` 재현: 실제 페어 대각 우위 구조.
2. **temperature τ와 ×100 클립** — softmax 로짓 스케일을 변동시키며 학습 중 τ 최적화 방향 재현.
3. **프롬프트 템플릿 = 라벨 → 문장** — "A photo of a {label}." vs naive {label}.의 매핑 차이를 재현 가능한 구조로.
4. **임베딩 공간 앙상블** — 평균 텍스트 임베딩 1세트로 동일 점수: 추론 비용 동일성("amortized") 재현.
5. **규모 → zero-shot 오류의 log-log 직선 경향** — 모델 스케일 변동에 따른 추세성만 확인(값 자체보다 기울기 방향).

### 1.4 관측: 원문 내부 불일치 3건

1. **BPE 어휘 크기**: 본문 §2.4 "49,152 vocab" vs 부록 하이퍼파라 표 "Vocabulary size 49408" — 코드가 49,408이므로 **본문 오기 가능성 높음**, 시트는 **49,408(부록 표·코드 일치)**로 표기.
2. `zeroshot_table` 해석: CLIP 행 **98.4/76.2/58.5** = aYahoo 98.4 · ImageNet 76.2 · SUN 58.5 — Visual N-Grams 행 72.4/11.5/23.0 대비. 본문 "on aYahoo, 95% reduction in errors", "on SUN, more than doubles"와 정합.
3. `linear-probe-table.tex`·`zero-shot-table.tex`는 LaTeX 소스에 **미컴파일 잔여 파일**(`\input` 미사용) — 부록성 표 해석 불가 — 반드시 논문 PDF(부록 A)에서 직접 열어 확인.

### 1.5 README 릴리즈 수치 vs 논문 수치

- README 첫 문장: "matches the performance of the original ResNet50 on ImageNet 'zero-shot' without using any of the original 1.28M labeled examples" — 논문(`abstract`·본문) 동일 주장.
- README 확률 예시 `[[0.9927937 0.00421068 0.00299572]]` — **ViT-B/32 데모 샘플**("a diagram"/"a dog"/"a cat") — 벤치마크 수치 아님 → 인용 금지.
- `clip.tokenize(..., context_length=77)` — 본문 "max sequence length was capped at 76" + [SOS]/[EOS].

### 1.6 금지 목록 ("아는 것 같은데 검증 안 된 것")

1. "Table 13"식 **컴파일 표 번호 인용** — Table 1 외 미확정; 라벨 사용 권장.
2. 개별 모델별 ImageNet zero-shot 상세(ViT-B/32 = 66.7% 등) — **논문 본문·표에 없음**; 사용时要 레포 모델카드를 "README 수치" 라벨로 확인.
3. GFLOPs · FPS · ETA — **논문에 절대 수치 없음**(V100 GPU × 일수만); 다른 연구·모델카드 출처는 별도 라벨.
4. "63M 파라미터"를 전체 파라미터로 착각 — **텍스트 인코더만** 63M; 전체는 본문에 없음.
5. WIT "401M" 등 라운드 상승 — 본문 공식 **"400 million"**.
6. Zero-shot 성능 76.2%는 **ViT-L/14@336px** 전용 — 다른 모델의 ImageNet zero-shot 수치는 본문에 없음.
7. STL10 99.3% — 본문 문구 그대로 사용 가능(단, zero-shot 성격 + "appears to be a new state of the art" 명시).
8. FairFace·Race/Gender/Age/공정성 표의 세부 수치 — 필요 시 원문 PDF(§8)에서 직접 열어 인용.
9. Video 표의 R(2+1)D-BERT · HT100M S3D 등 Finetune 베스트 성능 값은 `table:action` 원문 그대로 가능 — 단, 표 전체 컨텍스트 없이 부분 인용 금지.
10. 논문-figure 속 수치를 눈으로 계수(=그림에서 값 읽기) — prefix 금지: 캡션과 본문만 근거로.

### 1.7 인용 링크

- abs [2103.00020](https://arxiv.org/abs/2103.00020) — **ar5iv 접속 실패(타임아웃, 2026-10-23)** → LaTeX 원본(e-print)으로 대체검증; PMLR 카메라레디 PDF 병행 확인
- PMLR: [proceedings.mlr.press/v139/radford21a](https://proceedings.mlr.press/v139/radford21a.html)
- 레포: [openai/CLIP](https://github.com/openai/CLIP) — **MIT**, `pip install git+https://github.com/openai/CLIP.git`
- 블로그: [openai.com/blog/clip/](https://openai.com/blog/clip/)
