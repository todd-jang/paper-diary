# Day 13 · LLaVA — Visual Instruction Tuning (2026-10-26 세션)

> **논문**: LLaVA = arXiv [2304.08485](https://arxiv.org/abs/2304.08485) (v1 원문 접근 완료) · **LLaVA-1.5 = 별편 보고서** [2310.03744](https://arxiv.org/abs/2310.03744) "Improved Baselines with Visual Instruction Tuning" — *abs comment: CVPR 2024 (highlight)* ✅ abs 확인. 핵심 인물: Haotian Liu, Chunyuan Li, Qingyang Wu (UW–Madison·MSR·Columbia), Yong Jae Lee — 구조 = **CLIP ViT-L/14(동결) + 단일 선형 프로젝션 W(최초) + Vicuna**.

## 🧪 심화 산출물 (`day13_llava/` — 제작 진행 중)

<!-- 키트 제작 예정: 00_검증팩트시트 → 01_코드워크스루 → 02_수식유도 → 03_실습 → 04_비교표. 완성 시 이 절에 링크 · 데모 PASS 조건 기입. -->

## 🎯 오늘의 관통 질문

- **아키텍처**: 비전 인코더 + 프로젝터 + LLM의 결합이 Q-former/gated cross-attention 없이 *선형만으로* 성립하는가?
- **학습**: "동결→정렬"과 "unlock→instruction tuning"의 2단계는 각각 무엇을 사고 있다는 결론이 있는가?
- **인스트럭션 데이터**: "본 라벨링을 project 후반"을 넘어, 데이터의 *문체·캐시*가 무엇인가에 달렸나?
- **관통**: **전통 캡셔닝/검출 데이터는 "정답"이고, LLaVA는 "문맥"을 쓴다** — 왜 달라져야만 instruction tuning이 지연 수 있는 것들이 나는가?

## 📖 읽기 범위

- `ar5iv.labs.arxiv.org/html/2304.08485` — 초록 · §3(GPT-4 시각 인스트럭션 데이터) · §4.1 아키텍처 · §4.2 2단계 학습 · §5 실험(Table 4·5·경우 Tables 1·2·3·6) — *본 세션에서 원문 직접 조회로 수치 채확*.
- LLaVA-1.5: [2310.03744](https://arxiv.org/abs/2310.03744) **abs 기준만**(본문 세부는 미확보) — 원문 세부 대조는 요청 사항으로 명시.

## 🧠 Socratic Quest (3문항 — 티쳐 출제, 2026-10-26)

1. **Q1** — 단일 선형 프로젝션의 단순함·한계: *왜 작동했고 왜 1.5에서 2층 MLP로*.
2. **Q2** — 2단계 학습 전역 분업: *Stage 1 freeze-ProJ only의 이론, 처음부터 end-to-end라면*.
3. **Q3** — GPT-4 시각 인스트럭션 데이터 합성 엔진: *기존 캡셔닝/검출 데이터와의 결정적 차별*.

## ✍️ 나의 답변 (학생이 작성 — ① 주장 ② 근거(표·절·라벨 + 조건) ③ 경계)

**Q1 — 단일 선형이 작동할 수 있었던 이유와 1.5로의 발전**

주장: 프로젝션 W의 역할은 *새 지식 학습이 아니라 좌표계 번역*이다. CLIP ViT-L/14의 그리드 피처(§4.1 Eq. 1 — `H_v = W·Z_v`; "grid features before/after the last Transformer layer" 양쪽을 실험)는 대조 사전학습으로 caption 어휘 공간과 정렬돼 있으므로(D12 CLIP에서 확인한 구조), Vicuna word embedding 공간과의 위상이 아핀 대응 범주에 가깝고 → 복잡한 크로스어텐션 없이 선형만으로 붙을 수 있다. 가벼움의 실익: 초래 파라미터가 작아 **데이터 중심 반복 실험**이 가능(§4.1 "simple projection scheme is lightweight … allows us to iterate data centric experiments quickly") · Flamingo의 gated cross-attention·BLIP-2의 Q-former 같은 정교한 구조는 **후속 과제**로 명시(§4.1). 그리고 LLaVA-1.5(2310.03744 abs — CVPR 2024 highlight)의 자체 요약이 이를 뒷칠한다: **"LLaVA의 fully-connected(MLP) connector 가 놀라울 만큼 강력하고 데이터 효율적이었다"** — Q-former 후속 구조의 정밀 시스템이 아니라 **CLIP-ViT-L-336px + MLP + 학술 과제 대상 VQA 데이터 + 간단한 response-formatting 프롬프트**만으로 **11 벤치마크 SOTA**, 공개 데이터 ~1.2M · ~8-A100 1일 full training(abs 인용 — 조건 명시). "MLP가 이득인 정확한 분배"는 비선형 용량 + 해상도 상향의 합 동효과로 본다 — 단독 소거는 1.5 본문 참조 필요. 경계: v1 입력 해상도·그리드 수치는 v1 본문 확보에 없음 — **원문 대조 요청 사항**.

**Q2 — Freeze 전역의 역할 분담, 그리고 처음부터 End-to-End라면**

주장: Stage 1은 학습이라기보다 **정렬(adapter) 축소** — v1 논문이 스스로 "고정 LLM을 위한 호환 시각 토크나이저 학습"으로 정의(§4.2). 지식 역할 분담: CLIP = *지각 모듈*(모든 단계 동결 — 대조 사전학습 정렬 유지), W = *좌표계*(Stage 1 유일 trainables θ={W}), Vicuna = *지식·응답 능력*(Stage 2에서만 풀림 θ={W,φ}). 근거: Stage 1 = CC3M 필터 **595K** 단일-턴 단순확장·동결 both·**lr 2e-3/1 epoch/bs 128**(§5 학습 세부) vs Stage 2 = LLaVA-Instruct-**158K**·**3 epoch/lr 2e-5/bs 32**·CLIP 동결(§4.2·§5) — *데이터 방향(다량-단순 → 소량-고퀄)과 학습률(100× 감소)이 역할 전환을 그대로 드러냄*. 능력 검증 [Table 4]: No instruction tuning **21.5 → full 튜닝 85.1(+63.6)** — 능력은 레이블 데이터에 의해 떨어지는 것이 아니라 인스트럭션 데이터로 조정짐. 만약 0-step부터 전체 unlock라면: ① **신호-모델 비율 붕괴** — 단일-턴 캡션 확장 데이터는 거대한 Vicuna 전체를 조준한 *표적 잡을 리 주가*의 희박한 신호 → ② 초기 W 오차가 유입되고, LLM이 *자기 공간을 시각 노이즈에 "치료"하려는* 전현 압력 — 사전학습 표현 붕괴(catastrophic forgetting 위험) ③ 계산 비용 수십 배. 근거로 ①·③ 가치 — **시트 밖 자문 라벨**(v1 본문에 직접 소거 부재, 습득 스타일 label 표기).

**Q3 — GPT-4 시각 인스트럭션 합성: 기존 데이터와의 결정적 차별**

주장: 이 엔진의 본질 = **"박스+캡션을 정답이 아니라 문맥(context)으로"** 텍스트 전용 GPT-4에 주입해, *동일 이미지로 다층적 질문-응답 형식*(SFT-ready)을 합성하는 것 — **이미지 자체는 GPT-4에게 주입받지 않음(Table 1 note: "visual image is not used to prompt GPT")**. 전통 캡셔닝/검출과의 차별 3방향: ① **표현층위** — 전통 = (이미지→정형 라벨)*한 벌*; LLaVA = **3종 응답 형식** — 대화 **58K · 상세 설명 23K · 복잡 추론 77K**(합계 158K, §3 — *흔히 27K로 오기되나 본문은 77K*), 즉 표어가 *문체/추론/대화형 다양성*을 동시 학습 ② **공동 확장 효과** — [Table 4] 대화 전용 학습 73.8(−11.3) → 5% detail+10% complex 80.5 → detail+complex만 81.9 — **추론 능력 향상이 대화 능력을 보완**(본문 명시) ③ **인적 비용의 이동** — 사람은 image당 소수의 **seed in-context 예시**만(본문 "the only human annotations"); 실데이터 소비처는 `∼80K unique images`(§5.1) — COCO 이미지 + *기존 캡션/박스 메타(이미 존재 비용)*를 재활용. 품질 통제: GPT-4가 이미지를 보지 못하는 v1 — ① "확실한 답만 있는 질문만 고려"(§3 conversation 설계) ② GPT-4 > ChatGPT 소거(공간 추론 품질 일관 도움) — 한계: **Table 6 냉장고 사례 "yes" 환각 → "bag of patches" 한계 명시**(§5.1 Limitations). 성적 조건 명시: [Table 5] In-the-Wild — 전체 **67.3** vs BLIP-2 38.1(+29%) · OpenFlamingo 19.1(+48%) · 복잡 추론 81.7(*기준 = 지상 캡션·박스를 본 텍스트 전용 GPT-4 대비 상대점수*) · [초록] **ScienceQA 92.53 = GPT-4 앙상블 조건**(단독 수치는 본문 후반 미확보 — 요청 사항). 연결: 박스를 텍스트 전용 LLM에 그대로 인자 제공하는 것 = D10 GLIP의 *box-as-language* 선례 관찰.

> **정직 선언 3건 (티쳐 승인 — "모두 완벽하게 타당")**: ① LLaVA-1.5 수치는 abs 기준만(세부 장표 미확보) ② Q2의 end-to-end 소거는 v1 본문 직접 부재 → 시트 밖 자문 라벨 ③ Day13 루트 노트 신규 생성 필요.

## 🔍 자기점검 (답변 직후 스스로)

- **Q1**: (i) Eq. 1 라벨 인용했는가 ✓ (ii) 1.5를 "별편 보고서"로 라벨 분리했는가 ✓ (iii) v1 해상도 수치 미확보를 경계로 표기했는가 ✓
- **Q2**: (i) Stage 하이퍼(595K→158K·lr 100× 감소)를 근거로 역할 전환을 조인했는가 ✓ (ii) CLIP 상시 동결의 이유(지각 모듈)를 분리했는가 ✓ (iii) end-to-end 반례에 시트 밖 라벨을 붙였는가 ✓
- **Q3**: (i) 3종 비율 58/23/**77K**를 본문 채확했는가 ✓ (ii) Table 4 자료 띠덩 종류 소거로 공동 확장을 증명했는가 ✓ (iii) ScienceQA 92.53의 앙상블 조건을 명시했는가 ✓ (iv) "bag of patches" 한계를 짚었는가 ✓

## 🔬 Feynman 설명

<!-- 대기 — 티쳐 출제 시 작성 -->

## ❓ 의문점 / 헷갈린 것

- v1 입력 해상도(224 vs 336)·그리드 크기·시각 토큰 수 — v1 본문 아키텍처 절의 세부 확인 필요.
- LLaVA-1.5의 576토큰/MLP 소거 절차·SQA 13B 수치 — 2310.03744 본문 확보 시 조인.
- ScienceQA 단독 성적(앙상블 92.53 대비) — v1 본문 후반부(TRUNC54) 미확보.

## ✅ 채점 & 교정 (선생님이 채움)

**2026-10-26 세션 · 티쳐 채점 — 퀘스트 평균 95.6%"Day 13 LLaVA 세션 퀘스트를 탁월하게 완수"**

- **Q1 (프로젝터와 아키텍처 확장)**: CLIP 그리드 피처가 이미 어느 정도 언어 공간과 유사한 고차원 거리 구조를 갖는다는 해석 정확 · LLaVA-1.5가 Q-Former 대신 **336px + MLP 강화만으로 SOTA**에 접근한 포인트 연계 선정.
- **Q2 (학습 파이프라인의 분업)**: Stage 1 freeze 논리를 **"지각 모듈의 표현력 파괴 방지"**와 연결한 논리 타당 · 처음부터 전체 unlock 시 **Vicuna 사전학습 가중치 붕괴(catastrophic forgetting)** 지적 정확.
- **Q3 (데이터 엔진과 환각의 양면성)**: 텍스트 전용 GPT-4의 3종 타입(대화/상세/복잡 추론) 합성 전략 우수성 서술 ✓ · **"Bag of patches" 환각 한계(Table 6)**를 논문 한계로 정직 수용한 서술을 높이 평가(티쳐 원문: *논문의 한계를 정직하게 수용한 훌륭한 서술*).
- **정직 선언 3건 승인**: ① 1.5 세부 수치 확장 필요 ② E2E 소거 실험 논리적 한계(시트 밖 자문 명시) ③ `Day13_LLaVA.md` 신규 생성 — "모두 완벽하게 타당 · 본 답변 내용은 그대로 Day 13 루트 노트의 초안 및 나의 답변 섹션으로 등재 가능" (등재 완료).
