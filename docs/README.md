# AI

## 🔎 Overview
AI 기능을 활용한 동화 서비스 "꼬꼬북"을 도움될 목적으로, 아이의 상상을 기반으로 동화를 구현하는 AI 통합 클라우드 서비스를 계획.

---------------------------------------

## 🤖 Key AI Features

### 1. 동화 줄거리 생성 서비스 (GPT API)
- **Model:** `GPT-4o-mini` + RAG
- **Use:** 아이가 입력한 사이트에 맞게 Prompt 검색 + RAG 결과를 기반으로 희망적인 녹마 구조 및 주요 기술 형식 생성

### 2. 동화 삽화 생성 (DALL·E API)
- **Model:** `DALL·E 3`
- **Technique:** Prompt Engineering + Style Consistency
- **Function:** 생성된 이야기 내용에 맞게 일관된 색상과 형상의 이미지 생성
- **Use:** 아이들이 상상한 장면을 시각화하여 동화의 삽화로 활용

### 3. 모음 발생 (Swivid TTS) – 사용자 모음 생성
- **Model:** `SWivid TTS` (open source) & `OpenAI API`
- **Technique:** 초기에는 OpenAI TTS를 사용하여 동화의 내레이션 및 대사 음성을 생성하며, 향후 SWivid TTS를 통해 사용자의 음성을 학습하여 주인공의 목소리로 활용
- **Function:**  OpenAI TTS를 활용한 기본 음성 합성 및 SWivid TTS Fine-tuning을 통한 사용자 음성 학습
- **Use:** 주인공의 대사는 사용자 목소리로, 조연 및 내레이션은 OpenAI TTS로 생성하여 보다 개인화된 동화를 제공.

---

## 📊 Architecture Overview

- **Frontend:** React Native (Mobile)
- **Backend:** FastAPI (AI Model relay, Server)
- **Infra:** AWS EC2, Google Colab (Fine-tuning), AWS S3 (Image, Voice 저장)
- **DB/Memory:** MySQL and VectorDB (Chroma -> Pinecone)

---

## 📆 Development Environment
- 포털: Google Colab, Cursor (AI IDE)
- 경로 구성: 
  - `GPT API`: CLI / RESTful
  - `DALL-E`: Image API endpoint
  - `SWivid`: 입력 시 TTS Training 및 출력
  - `OpenAI TTS` : 텍스트 입력 시 음성 출력
- 단계적 구조: Prompt → Story → Image → Voice → Final Export

---

## 🔍 To-Do
- [ ] GPT-4o-mini 매칭 목적으로 Fine-tuning 시작
- [ ] DALL-E Prompt Style 정체 유지를 위한 Prompt Chain 구조 고정
- [ ] SWivid TTS 학습 파이프라인 (Jupyter 기반) 구성 및 테스트
- [ ] 전체 Flow 통합 테스트 (사용자 입력 → 동화 생성 → 삽화 → 음성 출력)

---

> 본 파일은 꼬꼬북 프로젝트의 AI 설계를 Cursor 기반 개발 환경에서 빠르게 확인하고 이어서 작업할 수 있도록 정리된 Markdown입니다. 
> 최신화 및 테스트 결과는 `.ipynb` / `API 연동 로그`로 별도 관리 예정입니다.
> SWivid TTS는 MIT 라이선스를 사용하고 있으므로 이걸 사용할 때는 저작권 공지와 License 내용을 모든 복사본에 포함해야 하며, 사용 과정에서 발생하는 모든 책임은 사용자에게 있음.