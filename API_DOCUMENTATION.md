# 🌟 CCB_AI API 문서

## 📋 목차
- [개요](#개요)
- [인증](#인증)
- [기본 정보](#기본-정보)
- [REST API 엔드포인트](#rest-api-엔드포인트)
- [WebSocket API 엔드포인트](#websocket-api-엔드포인트)
- [데이터 모델](#데이터-모델)
- [에러 코드](#에러-코드)
- [SDK 예제](#sdk-예제)

---

## 🎯 개요

꼬꼬북(CCB_AI)은 아동을 위한 AI 동화 생성 시스템입니다.
- **부기 (ChatBot A)**: 음성 대화 수집 및 처리
- **꼬기 (ChatBot B)**: 동화 생성 및 멀티미디어 제작

### 서비스 구조
```
Frontend ↔ Nginx ↔ FastAPI Backend
                      ├── WebSocket API (실시간 음성/대화)
                      └── REST API (이야기 관리/조회)
```

---

## 🔐 인증

### 토큰 인증
API 요청에는 JWT 토큰 또는 개발용 토큰이 필요합니다.

#### 토큰 획득
```http
POST /api/v1/auth/token
Content-Type: application/json

{}
```

**응답**
```json
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600,
  "token_type": "Bearer"
}
```

#### REST API 인증
```http
Authorization: Bearer <your-jwt-token>
```

#### WebSocket 인증
```javascript
// 개발 환경
const ws = new WebSocket('ws://localhost:8000/ws/audio?token=development_token');

// 프로덕션 환경
const ws = new WebSocket('ws://your-domain.com/ws/audio?token=your-jwt-token');
```

#### 토큰 종류
- **개발용**: `development_token` (개발 및 테스트 환경)
- **프로덕션**: JWT 토큰 (실제 서비스 환경)

---

## 📡 기본 정보

### 베이스 URL
- **개발**: `http://localhost:8000`
- **프로덕션**: `https://your-domain.com`

### Rate Limiting
- **일반 API**: 60 requests/minute
- **이야기 생성**: 5 requests/minute
- **WebSocket**: 연결당 1시간 제한

---

## 🌐 REST API 엔드포인트

### 🩺 헬스체크

#### `GET /health`
서버 상태 확인

**응답**
```json
{
  "status": "online",
  "whisper_model": "loaded"
}
```

**상태 코드**
- `200`: 정상
- `503`: Whisper 모델 미초기화

---

### 👥 연결 관리

#### `GET /connections`
활성 WebSocket 연결 정보 조회

**응답**
```json
{
  "connections": [
    {
      "client_id": "아이이름_1234567890",
      "connected_at": "2024-01-01T12:00:00Z",
      "child_name": "아이이름",
      "age": 7
    }
  ],
  "count": 1
}
```

---

### 🎭 이야기 관리 API

#### `POST /api/v1/stories`
새 이야기 생성 요청

**요청 본문**
```json
{
  "child_profile": {
    "name": "민지",
    "age": 7,
    "interests": ["공주", "마법", "동물"],
    "language_level": "basic",
    "special_needs": []
  },
  "conversation_data": {
    "messages": [
      {
        "content": "공주님이 나오는 이야기 만들어줘",
        "timestamp": "2024-01-01T12:00:00Z"
      }
    ]
  },
  "story_preferences": {
    "theme": "fantasy",
    "length": "medium"
  },
  "enable_multimedia": true
}
```

**응답**
```json
{
  "success": true,
  "story_id": "uuid-1234-5678-9012",
  "message": "이야기 생성이 시작되었습니다",
  "data": {
    "child_name": "민지",
    "estimated_completion_time": "3-5분"
  }
}
```

#### `GET /api/v1/stories/{story_id}`
이야기 조회

**응답**
```json
{
  "success": true,
  "message": "이야기 조회 성공",
  "data": {
    "story_id": "uuid-1234-5678-9012",
    "title": "마법의 공주와 친구들",
    "chapters": [
      {
        "chapter_number": 1,
        "title": "공주의 만남",
        "content": "옛날 옛적에...",
        "image_url": "/images/chapter1.jpg",
        "audio_url": "/audio/chapter1.mp3"
      }
    ],
    "characters": [
      {
        "name": "소피아 공주",
        "description": "착한 마음을 가진 공주",
        "image_url": "/images/sofia.jpg"
      }
    ],
    "status": "completed",
    "created_at": "2024-01-01T12:00:00Z"
  }
}
```

#### `GET /api/v1/stories/{story_id}/status`
이야기 생성 상태 조회

**응답**
```json
{
  "story_id": "uuid-1234-5678-9012",
  "current_stage": "voice_generation",
  "workflow_state": "in_progress",
  "progress_percentage": 75.5,
  "error_count": 0,
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:05:00Z",
  "errors": []
}
```

#### `GET /api/v1/stories`
이야기 목록 조회

**쿼리 파라미터**
- `active_only`: boolean (기본값: false)

**응답**
```json
{
  "success": true,
  "message": "이야기 목록 조회 성공",
  "data": {
    "stories": [
      {
        "story_id": "uuid-1234-5678-9012",
        "title": "마법의 공주와 친구들",
        "child_name": "민지",
        "status": "completed",
        "created_at": "2024-01-01T12:00:00Z"
      }
    ],
    "count": 1,
    "active_only": false
  }
}
```

#### `POST /api/v1/stories/{story_id}/cancel`
이야기 생성 취소

**응답**
```json
{
  "success": true,
  "message": "이야기 생성이 취소되었습니다",
  "data": {
    "story_id": "uuid-1234-5678-9012"
  }
}
```

#### `POST /api/v1/stories/{story_id}/resume`
중단된 이야기 재개

**응답**
```json
{
  "success": true,
  "message": "이야기 재개가 시작되었습니다",
  "data": {
    "story_id": "uuid-1234-5678-9012"
  }
}
```

#### `GET /api/v1/statistics`
통계 정보 조회

**응답**
```json
{
  "success": true,
  "message": "통계 조회 성공",
  "data": {
    "total_stories": 150,
    "completed_stories": 142,
    "active_stories": 3,
    "average_completion_time": "4.2분",
    "popular_themes": ["fantasy", "animals", "adventure"]
  }
}
```

---

## 🔌 WebSocket API 엔드포인트

### 🎤 음성 대화 WebSocket

#### `WebSocket /ws/audio`
실시간 음성 대화 처리 (메인 오디오 엔드포인트)

**연결 URL**
```javascript
// 개발 환경
const ws = new WebSocket('ws://localhost:8000/ws/audio?' + new URLSearchParams({
  child_name: '민지',
  age: 7,
  interests: '공주,마법,동물',
  token: 'development_token'  // 개발용 토큰 또는 JWT
}));

// AWS 프로덕션 환경
const ws = new WebSocket('ws://13.124.141.8:8000/ws/audio?' + new URLSearchParams({
  child_name: '민지',
  age: 7,
  interests: '공주,마법,동물',
  token: 'your_jwt_token'
}));
```

**연결 파라미터**
- `child_name`: 아이 이름 (string, 필수)
- `age`: 아이 나이 (integer, 필수)
- `interests`: 아이 관심사 (string, 쉼표로 구분, 선택사항)
- `token`: 인증 토큰 (string, 필수)
  - 개발: `development_token`
  - 프로덕션: JWT 토큰

**⚠️ 중요: 오디오 전송 방식**
서버는 **순수 바이너리 데이터**만 받습니다. JSON 형태가 아닙니다!

**오디오 데이터 전송**
```javascript
// ❌ 잘못된 방식 (JSON)
ws.send(JSON.stringify({
  type: "audio_chunk",
  data: "base64-audio-data"
}));

// ✅ 올바른 방식 (바이너리)
// React Native 예시
const audioFile = await RNFS.readFile(audioFilePath, 'base64');
const audioBuffer = Buffer.from(audioFile, 'base64');
ws.send(audioBuffer);

// Web 브라우저 예시
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
        // 바이너리 Blob 직접 전송
        ws.send(event.data);
      }
    };
    // 1초마다 청크 전송 (서버는 1초 또는 64KB마다 처리)
    mediaRecorder.start(1000);
  });
```

**처리 기준**
- **시간**: 1초마다 누적된 오디오 처리
- **크기**: 64KB 이상 누적되면 즉시 처리
- **음성 인식**: Whisper 모델 사용
- **응답 생성**: GPT-4o-mini 사용
- **음성 합성**: ElevenLabs API 사용

**서버 응답 (JSON 형태)**

##### 1. AI 응답 (정상)
```json
{
  "type": "ai_response",
  "text": "안녕 민지야! 어떤 공주님 이야기를 듣고 싶어?",
  "audio": "UklGRnoGAABXQVZFZm10IBAAAA...",  // base64 인코딩된 MP3
  "status": "ok",
  "user_text": "공주님 이야기 해줘",  // STT 결과
  "confidence": 0.95,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

##### 2. 음성 인식 결과 (중간 단계)
```json
{
  "type": "transcription",
  "text": "공주님 이야기 해줘",
  "confidence": 0.95,
  "status": "partial",  // 또는 "final"
  "timestamp": "2024-01-01T12:00:00Z"
}
```

##### 3. 처리 상태 알림
```json
{
  "type": "processing",
  "message": "음성을 분석하고 있어요...",
  "stage": "speech_recognition",  // "ai_generation", "voice_synthesis"
  "timestamp": "2024-01-01T12:00:00Z"
}
```

##### 4. 에러 응답
```json
{
  "type": "error",
  "error_message": "음성 인식에 실패했습니다",
  "error_code": "WHISPER_ERROR",  // "STT_FAILED", "AI_ERROR", "TTS_ERROR"
  "status": "error",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

##### 5. 연결 종료 알림
```json
{
  "type": "conversation_end",
  "message": "대화가 저장되었습니다",
  "conversation_file": "/app/output/conversations/민지_20240101_120000.json",
  "total_exchanges": 5,
  "duration_minutes": 3.2
}
```

### 🧪 테스트용 WebSocket 엔드포인트

#### `WebSocket /ws/test`
기본 연결 및 JSON 메시지 테스트용

**연결**
```javascript
const testWs = new WebSocket('ws://localhost:8000/ws/test?token=development_token');

testWs.onopen = () => {
  console.log('테스트 연결 성공');
  
  // 테스트 메시지 전송
  testWs.send(JSON.stringify({
    type: 'test',
    message: 'Hello WebSocket'
  }));
};

testWs.onmessage = (event) => {
  const response = JSON.parse(event.data);
  console.log('서버 응답:', response);
  // { type: "echo", original_message: {...}, timestamp: "..." }
};
```

#### `WebSocket /ws/binary-test`
바이너리 데이터 전송 테스트용

**연결**
```javascript
const binaryWs = new WebSocket('ws://localhost:8000/ws/binary-test?token=development_token');

binaryWs.onopen = () => {
  console.log('바이너리 테스트 연결 성공');
  
  // 테스트 바이너리 데이터 전송
  const testData = new Uint8Array([1, 2, 3, 4, 5]);
  binaryWs.send(testData);
};

binaryWs.onmessage = (event) => {
  const response = JSON.parse(event.data);
  console.log('바이너리 수신 확인:', response);
  // { type: "binary_received", chunk_number: 1, chunk_size: 5, ... }
};
```

### 📚 이야기 생성 WebSocket

#### `WebSocket /ws/story_generation`
실시간 이야기 생성 상태 알림

**연결 파라미터**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/story_generation?' + new URLSearchParams({
  child_name: '민지',
  age: 7,
  interests: '공주,마법,동물',
  token: 'development_token'
}));
```

**서버 응답**

##### 1. 생성 상태 업데이트
```json
{
  "type": "generation_status",
  "story_id": "story_uuid_12345",
  "stage": "text_generation",
  "progress": 45.5,
  "message": "이야기 텍스트를 생성하고 있습니다...",
  "estimated_remaining": "2분"
}
```

##### 2. 챕터 완료 알림
```json
{
  "type": "chapter_completed",
  "story_id": "story_uuid_12345",
  "chapter_number": 1,
  "title": "공주의 만남",
  "preview": "옛날 옛적에 마법의 성에..."
}
```

##### 3. 이야기 완료
```json
{
  "type": "story_completed",
  "story_id": "story_uuid_12345",
  "title": "마법의 공주와 친구들",
  "total_chapters": 3,
  "download_url": "/api/v1/stories/story_uuid_12345"
}
```

### 🔧 WebSocket 연결 관리

**연결 상태 확인**
```javascript
ws.onopen = () => console.log('✅ 연결 성공');
ws.onclose = () => console.log('❌ 연결 종료');
ws.onerror = (error) => console.error('🚨 연결 에러:', error);

// 연결 상태 체크
if (ws.readyState === WebSocket.OPEN) {
  // 데이터 전송 가능
} else if (ws.readyState === WebSocket.CONNECTING) {
  // 연결 중...
}
```

**자동 재연결 (권장)**
```javascript
class AutoReconnectWebSocket {
  constructor(url) {
    this.url = url;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.connect();
  }
  
  connect() {
    this.ws = new WebSocket(this.url);
    
    this.ws.onopen = () => {
      console.log('WebSocket 연결됨');
      this.reconnectAttempts = 0;
    };
    
    this.ws.onclose = () => {
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        console.log(`재연결 시도 ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
        setTimeout(() => this.connect(), 2000 * this.reconnectAttempts);
      }
    };
  }
}
```

**⚠️ 주의사항**
1. **토큰 인증**: 개발 환경에서는 `development_token`, 프로덕션에서는 JWT 필수
2. **바이너리 전송**: `/ws/audio`는 반드시 바이너리 데이터로 전송
3. **청크 크기**: 너무 작은 청크(<1초)는 피하고, 너무 큰 청크(>5초)도 피할 것
4. **에러 처리**: 항상 `type: "error"` 응답에 대한 처리 로직 포함
5. **연결 제한**: 하나의 클라이언트당 하나의 WebSocket 연결만 유지

---

## 📊 데이터 모델

### ChildProfile
```typescript
interface ChildProfile {
  name: string;                    // 아이 이름
  age: number;                     // 나이 (3-12세)
  interests: string[];             // 관심사 목록
  language_level: "basic" | "intermediate" | "advanced";
  special_needs: string[];         // 특별한 요구사항
}
```

### StoryData
```typescript
interface StoryData {
  story_id: string;
  title: string;
  chapters: Chapter[];
  characters: Character[];
  status: "pending" | "in_progress" | "completed" | "failed";
  created_at: string;
  updated_at: string;
}

interface Chapter {
  chapter_number: number;
  title: string;
  content: string;
  image_url?: string;
  audio_url?: string;
}

interface Character {
  name: string;
  description: string;
  image_url?: string;
  voice_id?: string;
}
```

---

## ⚠️ 에러 코드

### HTTP 상태 코드
- `200`: 성공
- `400`: 잘못된 요청
- `401`: 인증 실패
- `403`: 권한 없음
- `404`: 리소스 없음
- `429`: Rate limit 초과
- `500`: 서버 오류

### 커스텀 에러 코드
```json
{
  "success": false,
  "message": "에러 메시지",
  "error_code": "ERROR_CODE",
  "details": "상세 정보"
}
```

| 에러 코드 | 설명 |
|-----------|------|
| `ORCHESTRATOR_NOT_INITIALIZED` | 오케스트레이터 미초기화 |
| `STORY_CREATION_FAILED` | 이야기 생성 실패 |
| `STORY_NOT_FOUND` | 이야기를 찾을 수 없음 |
| `STT_FAILED` | 음성 인식 실패 |
| `TTS_FAILED` | 음성 합성 실패 |
| `INVALID_AUDIO_FORMAT` | 잘못된 오디오 형식 |

---

## 💻 SDK 예제

### JavaScript/TypeScript

#### REST API 클라이언트
```typescript
class CCBApiClient {
  private baseUrl: string;
  private token: string;

  constructor(baseUrl: string, token: string) {
    this.baseUrl = baseUrl;
    this.token = token;
  }

  async createStory(request: StoryCreationRequest): Promise<StoryResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/stories`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request)
    });
    
    return response.json();
  }

  async getStory(storyId: string): Promise<StoryData> {
    const response = await fetch(`${this.baseUrl}/api/v1/stories/${storyId}`, {
      headers: {
        'Authorization': `Bearer ${this.token}`
      }
    });
    
    return response.json();
  }

  async getStoryStatus(storyId: string): Promise<StoryStatus> {
    const response = await fetch(`${this.baseUrl}/api/v1/stories/${storyId}/status`, {
      headers: {
        'Authorization': `Bearer ${this.token}`
      }
    });
    
    return response.json();
  }
}
```

#### WebSocket 클라이언트
```typescript
class CCBWebSocketClient {
  private ws: WebSocket | null = null;

  connectAudio(params: AudioWebSocketParams): Promise<void> {
    return new Promise((resolve, reject) => {
      const url = `ws://localhost:8000/ws/audio?${new URLSearchParams(params)}`;
      this.ws = new WebSocket(url);

      this.ws.onopen = () => resolve();
      this.ws.onerror = (error) => reject(error);
      
      this.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        this.handleMessage(message);
      };
    });
  }

  sendAudioChunk(audioData: ArrayBuffer, chunkIndex: number, isFinal: boolean): void {
    if (!this.ws) return;
    
    const base64Data = btoa(String.fromCharCode(...new Uint8Array(audioData)));
    
    this.ws.send(JSON.stringify({
      type: 'audio_chunk',
      data: base64Data,
      chunk_index: chunkIndex,
      is_final: isFinal
    }));
  }

  private handleMessage(message: any): void {
    switch (message.type) {
      case 'transcription':
        console.log('음성 인식:', message.text);
        break;
      case 'ai_response':
        console.log('AI 응답:', message.text);
        this.playAudio(message.audio_url);
        break;
      case 'error':
        console.error('에러:', message.message);
        break;
    }
  }

  private playAudio(audioUrl: string): void {
    const audio = new Audio(audioUrl);
    audio.play();
  }
}
```

### Python

#### REST API 클라이언트
```python
import asyncio
import aiohttp
from typing import Dict, Any

class CCBApiClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.token = token
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }

    async def create_story(self, request: Dict[str, Any]) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'{self.base_url}/api/v1/stories',
                headers=self.headers,
                json=request
            ) as response:
                return await response.json()

    async def get_story(self, story_id: str) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f'{self.base_url}/api/v1/stories/{story_id}',
                headers=self.headers
            ) as response:
                return await response.json()
```

#### WebSocket 클라이언트
```python
import asyncio
import websockets
import json
import base64

class CCBWebSocketClient:
    def __init__(self):
        self.ws = None

    async def connect_audio(self, params: dict):
        query_string = '&'.join([f'{k}={v}' for k, v in params.items()])
        uri = f'ws://localhost:8000/ws/audio?{query_string}'
        
        self.ws = await websockets.connect(uri)
        
        # 메시지 수신 루프 시작
        asyncio.create_task(self.listen_messages())

    async def send_audio_chunk(self, audio_data: bytes, chunk_index: int, is_final: bool):
        if not self.ws:
            return
            
        base64_data = base64.b64encode(audio_data).decode('utf-8')
        
        message = {
            'type': 'audio_chunk',
            'data': base64_data,
            'chunk_index': chunk_index,
            'is_final': is_final
        }
        
        await self.ws.send(json.dumps(message))

    async def listen_messages(self):
        async for message in self.ws:
            data = json.loads(message)
            await self.handle_message(data)

    async def handle_message(self, message: dict):
        msg_type = message.get('type')
        
        if msg_type == 'transcription':
            print(f"음성 인식: {message['text']}")
        elif msg_type == 'ai_response':
            print(f"AI 응답: {message['text']}")
        elif msg_type == 'error':
            print(f"에러: {message['message']}")
```

---

## 🔧 개발자 도구

### Swagger UI
API 문서 및 테스트: `http://localhost:8000/docs`

### 헬스체크 스크립트
```bash
#!/bin/bash
# health_check.sh

echo "=== CCB_AI 서비스 상태 확인 ==="

# 메인 서비스 헬스체크
echo "1. 메인 서비스 상태:"
curl -s http://localhost:8000/health | jq

# 통합 API 헬스체크  
echo "2. 통합 API 상태:"
curl -s http://localhost:8000/api/v1/health | jq

# 활성 연결 확인
echo "3. 활성 연결:"
curl -s http://localhost:8000/connections | jq

# 컨테이너 상태
echo "4. 컨테이너 상태:"
docker-compose ps
```

### 로그 모니터링
```bash
# 실시간 로그 확인
docker-compose logs -f ccb-ai

# 특정 시간대 로그
docker-compose logs --since="1h" ccb-ai

# 에러만 필터링
docker-compose logs ccb-ai 2>&1 | grep -i error
```

---

## 📞 지원

- **GitHub Issues**: [프로젝트 이슈 페이지]
- **이메일**: support@ccb-ai.com
- **문서**: [상세 문서 링크]

---

**버전**: 1.0.0  
**마지막 업데이트**: 2024-01-01 