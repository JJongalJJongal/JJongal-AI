# 🧠 쫑알쫑알 - AI 설계 개요

**쫑알쫑알**은 아이들이 상상한 이야기를 실시간으로 AI와 함께 동화책으로 만들어주는 EduTech 기반 프로젝트입니다.  
이 디렉터리는 해당 서비스의 **AI 처리 파트**로, 음성 수집부터 스토리 구성, 이미지 생성, 음성 합성까지 전 과정을 포함합니다.

---

## 🧩 AI 시스템 구성

### 🤖 Chat-bot A (이야기 시작 챗봇 - "쫑이")
- **역할**: 아이의 음성을 실시간 수집하고, 줄거리를 요약, 그리고 아이의 이야기를 만들기 위해 대화를 통해 이야기 유도
- **모델**: `gpt-4o-mini`
- **대화 프롬프트 특징**:
  - 연령별(4-7세, 8-9세) 맞춤형 언어 사용
  - 아이의 관심사 기반 자연스러운 대화 유도
  - 이야기 단계별(캐릭터, 배경, 문제, 해결) 수집 전략
  - 격려와 자연스러운 후속 질문으로 상상력 확장
  - 교육적 가치를 담은 내용으로 유도
- **기술 요소**:
  - 실시간 음성 스트리밍 수신 (WebSocket 기반)
  - `RNNoise`로 노이즈 제거
  - `Whisper`를 이용한 음성 → 텍스트 변환
  - GPT-4o-mini로 대화 요약 및 줄거리 생성
  - `ElevenLabs` API를 통해 음성 클로닝 요청
  - **LangChain 기반 RAG 시스템** 통합
  - **엔드포인트**: `/ws/audio`
  - **프로토콜**: WebSocket (ws/wss)
  - **인증**: Query 파라미터로 토큰 전달 (example: `?token=valid_token`)
  - **사용자 정보**: Query 파라미터로 전달 (example: `?child_name=민준&age=5&interests=공룡,우주,동물`)
  - **오디오 전송**: chunk 단위 바이너리(16kHz, mono, wav/opus 등)
  - **chunk 기준**: 2초 또는 128KB마다 서버가 처리
  - **응답**: 항상 JSON 패킷
    - `type`: "ai_response"
    - `text`: AI 텍스트 응답
    - `audio`: base64 인코딩된 mp3(음성)
    - `status`: "ok", "partial", "error"
    - `user_text`: 인식된 사용자 텍스트 (STT 결과)
    - `error_message`, `error_code`: (에러 발생 시)
  - **에러**: type이 "error"인 패킷으로 안내
  - **보안**: 운영 환경에서는 HTTPS/WSS, 인증 필수
  - **모니터링**: 서버 로그(logging) 기반 에러 추적
  - **대화 저장**: 연결 종료 시 자동 저장 (output/conversations 폴더)

### 🔍 LangChain과 Chroma DB 기반 RAG (Retrieval-Augmented Generation) 시스템
- **역할**: 동화 창작 과정에서 기존 동화 지식을 활용하여 창의적이고 일관된 스토리를 생성하도록 지원
- **기술 요소**:
  - `LangChain`을 활용한 벡터 저장소 관리 및 검색
  - `Chroma DB`를 통한 효율적인 벡터 저장 및 유사도 검색
  - `OpenAI Embeddings`로 문서 벡터화
  - 요약 및 상세 컨텐츠 각각에 대한 독립적인 벡터 저장소 지원
  - 텍스트 분할(Chunking)을 통한 효율적인 컨텐츠 관리
- **데이터 처리**:
  - 텍스트 데이터: 동화 내용, 요약, 교훈, 태그 분석
  - 연령별 컨텐츠 필터링 지원
  - 자동 메타데이터 관리 (제목, 태그, 스토리ID 등)
- **통합 기능**:
  - 이야기 주제 풍부화
  - 연령대별 맞춤형 이야기 참고 자료 제공
  - 대화 키워드 기반 유사 이야기 검색
  - 퍼시스턴스 지원으로 재시작 시에도 데이터 유지

### 🐢 Chat-bot B (스토리 완성 챗봇 - "아리")
- **역할**: 부기가 만든 줄거리와 음성 클론을 바탕으로 전체 동화 구성
- **모델**: `GPT-4o, DALL·E 3, ElevenLabs API`
- **기술 요소**:
  - `GPT-4o`로 상세 스토리 및 대사 생성
  - `DALL·E 3`로 삽화 생성 (프롬프트 엔지니어링 기반)
  - `ElevenLabs`로 감정/톤 반영된 음성 합성 (주인공 역할 등장인물만 부기가 요청을 보낸 음성 클로닝을 받아 아이의 음성 클로닝)
  - `GPT-4o`로 나머지 조연 등장인물들의 대화, 그리고 내레이션의 대화를 TTS 실행

---

## 🔍 AI 내부 구조 흐름

```
[아이 음성 입력]
       ↓
[부기: 음성 인식 + 단계별 이야기 수집 + 음성 클로닝 요청]
       ↓
[LangChain 기반 RAG 시스템: 동화 지식 기반 이야기 강화]
       ↓
[꼬기: 줄거리 기반 상세 스토리/삽화/음성 생성]
       ↓
[앱으로 동화책 전달 및 사용자 피드백 수집]
```

---

## 📚 LangChain 기반 RAG 시스템 사용법

RAG 시스템은 LangChain과 Chroma DB를 활용하여 동화 창작 과정에서 다양한 지식을 활용하도록 지원합니다.

### 초기화 방법
```bash
# 1. 필요 패키지 설치
pip install -r requirements.txt

# 2. RAG 시스템 초기화 및 샘플 데이터 추가
python -c "from chatbot.models.rag_system import RAGSystem; rag = RAGSystem(); rag.import_sample_stories()"
```

### 디렉토리 구조
```
data/
 └── vector_db/
     ├── summary/    # 요약 텍스트 벡터 저장소
     └── detailed/   # 상세 내용 벡터 저장소
     └── main/       # 전체 내용 벡터 저장소
     
```

### 주요 기능
- **이야기 저장 및 검색**: 스토리 ID, 제목, 태그 등의 메타데이터와 함께 요약 및 상세 내용 저장
- **유사 이야기 검색**: 요약 또는 상세 내용 기반으로 유사한 이야기 검색 지원
- **이야기 주제 강화**: 아이의 관심사와 대화 내용을 바탕으로 이야기 주제를 풍부하게 만듦
- **연령별 필터링**: 태그 기반으로 아이의 연령에 맞는 이야기를 자동으로 필터링
- **청크 기반 처리**: 긴 텍스트를 자동으로 나누어 관리하고 검색 시 관련 청크 반환
- **퍼시스턴스**: 디스크에 벡터 데이터 저장으로 서버 재시작 시에도 데이터 유지

### 코드 사용 예시
```python
# RAG 시스템 초기화
from chatbot.models.rag_system import RAGSystem
rag = RAGSystem()

# 새 스토리 추가
story_id = rag.add_story(
    title="용감한 토끼의 모험",
    tags="용기,평화,대화,화해,4-7세",
    summary="토끼가 무서운 여우와 대화를 통해 숲속 평화를 이루는 이야기",
    content="옛날 옛적, 작은 숲 속에 토미라는 작은 토끼가 살고 있었어요..."
)

# 스토리 검색
results = rag.query("용기와 화해에 대한 이야기", use_summary=True)
print(results["answer"])

# 연령대별 유사 이야기 검색
similar_stories = rag.get_similar_stories("우정", age_group=5, n_results=3)

# 이야기 주제 풍부화
enhanced_theme = rag.enrich_story_theme("우주 모험", age_group=7)
```

---

## 🔄 WebSocket API 사용법

### WebSocket 연결
```javascript
// ✅ 올바른 연결 방법
const ws = new WebSocket("ws://13.124.141.8:8000/ws/audio?" + new URLSearchParams({
  child_name: "민준",
  age: 5,
  interests: "공룡,우주,동물",
  token: "development_token"  // 개발용 또는 JWT 토큰
}));

// 메시지 수신 처리
ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  
  // AI 응답 처리
  if (response.type === "ai_response") {
    console.log("AI 응답:", response.text);
    console.log("사용자 음성 인식:", response.user_text);
    
    // Base64 오디오 재생
    if (response.audio) {
      const audio = new Audio("data:audio/mp3;base64," + response.audio);
      audio.play();
    }
  }
  
  // 에러 처리
  else if (response.type === "error") {
    console.error("에러:", response.error_message);
    console.error("에러 코드:", response.error_code);
  }
  
  // 음성 인식 중간 결과
  else if (response.type === "transcription") {
    console.log("음성 인식:", response.text, "신뢰도:", response.confidence);
  }
};

// ⚠️ 중요: 바이너리 오디오 데이터 전송
// JSON이 아닌 순수 바이너리로 전송해야 함

// React Native 예시
const sendAudioFile = async (audioFilePath) => {
  if (ws.readyState !== WebSocket.OPEN) {
    console.error("WebSocket 연결이 열려있지 않습니다");
    return;
  }
  
  try {
    // 파일을 base64로 읽고 바이너리로 변환
    const base64Audio = await RNFS.readFile(audioFilePath, 'base64');
    const audioBuffer = Buffer.from(base64Audio, 'base64');
    
    // 바이너리 데이터 직접 전송
    ws.send(audioBuffer);
  } catch (error) {
    console.error("오디오 전송 실패:", error);
  }
};

// Web 브라우저 예시 (실시간 녹음)
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    const mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm'
    });
    
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
        // Blob 데이터를 바이너리로 직접 전송
        ws.send(event.data);
      }
    };
    
    // 1초마다 청크 전송 (서버 처리 기준에 맞춤)
    mediaRecorder.start(1000);
  });
```

### 응답 형식
```json
{
  "type": "ai_response",
  "text": "안녕 민준아! 공룡과 우주에 대한 멋진 이야기를 만들어볼까?",
  "audio": "UklGRnoGAABXQVZFZm10IBAAAA...",
  "status": "ok",
  "user_text": "안녕 반가워, 이야기 만들어줘",
  "confidence": 0.95,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### 에러 응답 형식
```json
{
  "type": "error",
  "error_message": "음성 인식에 실패했습니다",
  "error_code": "WHISPER_ERROR",
  "status": "error",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### 🧪 개발 및 테스트

#### 연결 테스트
```javascript
// 1. 기본 연결 테스트
const testWs = new WebSocket('ws://13.124.141.8:8000/ws/test?token=development_token');

testWs.onopen = () => {
  console.log('✅ WebSocket 연결 성공');
  testWs.send(JSON.stringify({ type: 'test', message: 'Hello' }));
};

// 2. 바이너리 전송 테스트
const binaryWs = new WebSocket('ws://13.124.141.8:8000/ws/binary-test?token=development_token');

binaryWs.onopen = () => {
  console.log('✅ 바이너리 테스트 연결 성공');
  const testData = new Uint8Array([1, 2, 3, 4, 5]);
  binaryWs.send(testData);
};
```

### ⚠️ 주의사항
1. **바이너리 전송 필수**: `/ws/audio`는 JSON이 아닌 바이너리 데이터만 받음
2. **청크 기준**: 서버는 1초 또는 64KB마다 오디오 처리
3. **토큰 인증**: 개발용 `development_token` 또는 실제 JWT 토큰 필요
4. **연결 상태 확인**: 전송 전에 반드시 `ws.readyState === WebSocket.OPEN` 확인
5. **에러 처리**: 모든 `type: "error"` 응답에 대한 적절한 처리 로직 필요 

## ��️ **실시간 음성 클로닝 기능 (완전 구현!)**

부기와 대화하면서 자동으로 아이의 목소리를 학습하고 클론하여 동화에 적용하는 혁신적인 기능입니다.

### ✨ **완전한 작동 플로우**

1. **음성 수집 단계** (자동)
   - 아이가 부기와 대화할 때마다 음성 샘플을 자동 수집
   - 각 샘플은 `output/temp/voice_samples/{child_name}/` 폴더에 저장
   - 2개씩 수집될 때마다 진행상황 WebSocket 메시지 전송

2. **클론 생성 단계** (5개 샘플 수집 후 자동 실행)
   - ElevenLabs Instant Voice Cloning API로 새 음성 ID 생성
   - ChatBotB 인스턴스에 클론된 음성 자동 매핑
   - 성공/실패 WebSocket 알림 전송

3. **동화 적용 단계** (즉시)
   - 생성된 클론 음성을 동화의 주인공 목소리로 자동 설정
   - WebSocket 스트리밍 음성 생성에서 클론 음성 사용
   - Enhanced Mode의 캐릭터별 음성 생성에 통합

### 📡 **WebSocket 메시지 포맷**

```javascript
// 진행상황 알림 (2개씩 수집될 때마다)
{
  "type": "voice_clone_progress",
  "sample_count": 4,
  "ready_for_cloning": false,
  "has_cloned_voice": false,
  "message": "목소리 수집 중... (4/5)",
  "timestamp": "2024-12-19T10:30:45.123Z"
}

// 클론 생성 시작 알림
{
  "type": "voice_clone_starting",
  "message": "목소리 복제를 시작합니다...",
  "timestamp": "2024-12-19T10:31:00.456Z"
}

// 클론 생성 성공 알림
{
  "type": "voice_clone_success",
  "voice_id": "c38kUX8pkfYO2kHyqfFy",
  "message": "민준님의 목소리가 성공적으로 복제되었어요! 이제 동화에서 주인공 목소리로 사용됩니다.",
  "timestamp": "2024-12-19T10:32:15.789Z"
}

// 동화 생성시 클론 음성 적용 알림
{
  "type": "voice_clone_applied",
  "message": "민준님의 복제된 목소리를 동화에 적용했어요!",
  "voice_id": "c38kUX8pkfYO2kHyqfFy",
  "timestamp": "2024-12-19T10:35:20.123Z"
}
```

### 🔧 **핵심 구현 컴포넌트**

#### 1. VoiceCloningProcessor
```python
# 음성 샘플 수집
processor = VoiceCloningProcessor()
sample_saved = await processor.collect_user_audio_sample(
    user_id="민준",
    audio_data=audio_bytes
)

# 클론 생성 (5개 샘플 수집 후)
voice_id, error = await processor.create_instant_voice_clone(
    user_id="민준",
    voice_name="민준_voice_clone"
)

# 상태 확인
sample_count = processor.get_sample_count("민준")
is_ready = processor.is_ready_for_cloning("민준")
cloned_voice_id = processor.get_user_voice_id("민준")
```

#### 2. ChatBotB 연동
```python
# 클론된 음성을 ChatBotB에 설정
chatbot_b.set_cloned_voice_info(
    child_voice_id=voice_id,
    main_character_name="민준"
)

# 동화 생성시 클론 음성 자동 사용
result = await chatbot_b.generate_detailed_story(
    use_websocket_voice=True  # 클론 음성 포함 스트리밍
)
```

#### 3. 실시간 WebSocket 통합
```python
# audio_handler.py에서 자동 처리
if voice_cloning_processor.is_ready_for_cloning(child_name):
    voice_id, error = await voice_cloning_processor.create_instant_voice_clone(...)
    if voice_id:
        chatbot_b.set_cloned_voice_info(voice_id, child_name)
```

### 🛠 **기술 특징**

- **실시간 처리**: 대화 중 자동으로 음성 수집 및 클론 생성
- **ElevenLabs IVC 연동**: Instant Voice Cloning API 완전 지원
- **WebSocket 스트리밍**: 생성된 클론 음성으로 실시간 TTS
- **자동 매핑**: ChatBotB의 캐릭터별 음성 시스템에 즉시 통합
- **오류 처리**: SSL, API 실패, 타임아웃 등 모든 예외 상황 대응
- **진행 추적**: 단계별 WebSocket 알림으로 사용자 피드백 제공

### 📁 **파일 구조**

```
chatbot/models/voice_ws/processors/
├── voice_cloning_processor.py    # 클로닝 로직 (완전 구현)
└── audio_processor.py           # 기존 TTS 로직

chatbot/models/voice_ws/handlers/
├── audio_handler.py             # 실시간 음성 수집 (클로닝 통합)
└── story_handler.py            # 동화 생성 (클론 음성 적용)

chatbot/models/chat_bot_b/
├── chat_bot_b.py               # set_cloned_voice_info 메서드
└── generators/voice_generator.py # 캐릭터별 음성 매핑
```

### 🚀 **사용 예시**

1. **아이가 부기와 대화 시작**
   ```
   WebSocket: /ws/audio?child_name=민준&age=7
   ```

2. **자동 음성 수집 (백그라운드)**
   ```
   [2개 수집] → "목소리 수집 중... (2/5)"
   [4개 수집] → "목소리 수집 중... (4/5)"
   [5개 수집] → "목소리 복제를 시작합니다..."
   ```

3. **클론 생성 완료**
   ```
   "민준님의 목소리가 성공적으로 복제되었어요!"
   voice_id: "c38kUX8pkfYO2kHyqfFy"
   ```

4. **동화 생성시 자동 적용**
   ```
   WebSocket: /ws/story_generation
   → "민준님의 복제된 목소리를 동화에 적용했어요!"
   → 주인공 대사가 민준이의 목소리로 생성됨
   ```

이제 실시간 음성 클로닝 기능이 **완전히 구현되어 작동**합니다! 🎉
