# 🗣️Dialogue Summarization
## **[HighFive]** 어려워도 힘들어도 소통하며 문제해결하자!

| ![김종화](https://avatars.githubusercontent.com/u/221108223?v=4) | ![박준영](https://avatars.githubusercontent.com/u/221072284?v=4) | ![권효주](https://avatars.githubusercontent.com/u/13392737?v=4) | ![권문진](https://avatars.githubusercontent.com/u/89570502?v=4) | ![최보경](https://avatars.githubusercontent.com/u/110219144?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김종화](https://github.com/JHKIM-ItG)             |            [박준영](https://github.com/juny79)             |            [권효주](https://github.com/hopeplanting)             |            [권문진](https://github.com/moongs95)             |            [최보경](https://github.com/bekky1016)             |
|                            팀장, 모델설계 및 테스트                             |                            모델 설계 및 테스트                             |                            모델 설계 및 테스트                             |                            EDA 및 전처리                             |                            EDA 및 전처리                             |
## 0. Overview
### Environment
- Python 3.10  
- PyTorch / Transformers / ROUGE / pandas  
- QLoRA(4bit) + BitsAndBytes  
- 실험 관리: WandB  
- Git 버전관리  
- seed 고정으로 재현성 확보  

### Requirements
- 10.7B 모델을 GPU 메모리 내에서 학습하기 위해 **4bit 양자화**
- Encoder/Decoder max length 조정 필수
- Special tokens 추가 (`#Person#`, `#PhoneNumber#`, `#Address#`, …)

## 1. Competiton Info

### Overview

- 일상 멀티턴 대화를 2–3문장으로 요약하는 **Dialogue Summarization** 과제  
- 학교·직장·여가·상담·쇼핑 등 다양한 카테고리  
- 기간: **2025.11.27 ~ 2025.12.10**

### Timeline
- **Start:** 2025-11-27  
- **Final Submission:** 2025-12-10 

## 2. Components

### Directory

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data descrption

### Dataset overview

- Train: **12,457**, Dev: **499**, Test: **250**
- 결측치 없음 → 추가 정제 불필요  
- 대화 길이: 평균 **406 chars**, 최대 **2165 chars**  
- 요약 길이: 평균 **20–30 tokens**  
- 턴 수: 평균 **9.47턴**, 최대 **59턴** → 일부 롱컨텍스트 존재  

### EDA

### ✔ 카테고리 분포
- 가장 높은 카테고리 비율도 **1.04%**  
→ 라벨은 모델 성능에 **거의 영향 없음**

### ✔ 특수문자·노이즈 제거
- `<br>`, HTML escape, 괄호 내 지시문 등 제거  
→ **KoBART 기준 +0.5 ROUGE 상승**

### ✔ 대화 filler 제거
- "아, 어, 음, 근데, 그냥…"  
- OKT/TF-IDF 실패 → 규칙 기반 사전 구축  
→ **+0.93 ROUGE 추가 상승**

### Data Processing

- 특수문자 정규화 & 불필요 토큰 삭제  
- 구어체 filler 제거  
- Special Tokens 확장 (`#Person#`, `#PhoneNumber#` 등)  

## 4. Modeling

### Model descrition
## 1) KoBART Baseline 개선
- encoder_max_len 증가  
- decoder_max_len 증가  
- label smoothing=0.1  
- num_beams=3–4  
- 전처리 최적화로 **성능 대폭 향상**


---

## 2) SOLAR-10.7B-Instruct (QLoRA)
- 4bit 양자화 + LoRA r=8/16  
- max_seq_length=1024  
- SFTTrainer 활용  
- 24GB VRAM에서도 학습 가능  
- Base FT 점수 **44.54 → inference tuning으로 51.69**  
- 추가 재학습 모델 **51.77점**

---

## 3) Inference Optimization
- 프롬프트 4종 실험  
- Beam search / repetition penalty / length penalty 그리드서치  
- 후처리로 미완성 문장 제거  

### Modeling Process
- 전처리 → tokenizer 확장 → KoBART baseline → SOLAR FT → inference 튜닝 → ensemble(KoBART + SOLAR)

- WandB로 실험 관리

## 5. Result

### Leader Board

**최종 제출 점수: 51.4457**

KoBART baseline 대비 큰 폭 개선

### Presentation

- 발표자료(PDF): *첨부 예정*

## etc

### Meeting Log

- Notion 자동 로그  
- Slack 기록 공유  
- 매일 Zoom 회의 진행  

### Reference

- Dialogue Summarization 논문  
- Upstage SOLAR HuggingFace Docs  
- Solar Prompting Handbook  
