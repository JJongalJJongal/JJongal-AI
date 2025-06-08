# 🧠 꼬꼬북 - AI 설계 개요

**꼬꼬북**은 아이들이 상상한 이야기를 실시간으로 AI와 함께 동화책으로 만들어주는 EduTech 기반 프로젝트입니다.  
이 디렉터리는 해당 서비스의 **AI 처리 파트**로, 음성 수집부터 스토리 구성, 이미지 생성, 음성 합성까지 전 과정을 포함합니다.

---

## 🧩 AI 시스템 구성

### 🤖 Chat-bot A (이야기 시작 챗봇 - "부기")
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

### 🐢 Chat-bot B (스토리 완성 챗봇 - "꼬기")
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