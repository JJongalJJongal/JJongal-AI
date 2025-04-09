# AI

## 🔎 Overview
AI 기능을 활용한 동화 서비스 "꼬꼬북"을 도움될 목적으로, 아이의 상상을 기반으로 동화를 구현하는 AI 통합 클라우드 서비스를 제공.

---------------------------------------

## 🤖 Key AI Features

### Chat-bot A (아이와의 상호작용)
- **역할** : 아이와의 대화를 통해 동화의 대략적인 줄거리 생성
- **모델** : gpt-4o-mini API
- **입력** : 아이의 음성
- **출력** : 아이의 음성에 대한 적절한 응답, 동화의 대략적인 줄거리
- **특징** : 실시간 음성 수집 및 노이즈 필터링 (아이의 음성 실시간 수집하고, 배경 소음을 최소화하기 위해  RNNoise 노이즈 캔슬링 기술 적용)

### Chat-bot B (스토리 구체화 및 멀티미디어 생성)
- **역할** : Chat-bot A가 생성한 줄거리를 기반으로 상세한 스토리를 작성하고, 이에 맞는 삽화와 음성 생성.
- **모델** : gpt-4o API, ElevenLabs API
- **입력** : 동화의 대략적인 줄거리
- **출력** : 삽화 이미지 (JPEG), 음성 파일(MP3)
- **특징** : 음성 클로닝 통합 (주인공의 음성을 자연스럽게 표현하기 위해 ElevenLabs의 음성 클로닝 기능을 활용하여 주인공의 음성을 AI Chat-bot 과의 대화 음성에서 학습하여 대사 생성)

---

## 📊 인프라 및 배포

- **API 형태** : FastAPI
- **Container** : Docker, Docker-compose
- **Infra:** AWS EC2를 활용한 서버 배포
- **Storage** : AWS S3를 이용한 이미지 및 음성 파일 저장


---

## 데이터베이스 및 메모리 관리
- **관계형 데이터베이스(RDB)** : MySQL
- **벡터 데이터베이스 (VectorDB)** : Chroma or Pinecone or Faiss

---

## 노이즈 필터링
- **라이브러리** : RNNoise
- **특징** : 오픈 소스 기반의 실시간 노이즈 억제 라이브러리, 한국어 데이터 최적화 모델 적용

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

> 본 파일은 꼬꼬북 프로젝트의 AI 설계를 Cursor 기반 개발 환경에서 빠르게 확인하고 이어서 작업할 수 있도록 정리된 Markdown입니다. 
> 최신화 및 테스트 결과는 `.ipynb` / `API 연동 로그`로 별도 관리 예정입니다.
> SWivid TTS는 MIT 라이선스를 사용하고 있으므로 이걸 사용할 때는 저작권 공지와 License 내용을 모든 복사본에 포함해야 하며, 사용 과정에서 발생하는 모든 책임은      사용자에게 있음.