# 쫑알쫑알 API 명세서 v2.0 (Simple is Better)

## 1. 소개 (Introduction)

### 1.1. 설계 철학: "Simple is Better"
- **최소 필수 기능**: 정말 필요한 API만 제공 (4개 엔드포인트)
- **직관적 구조**: 복잡한 중첩 없는 플랫한 데이터 구조  
- **설정 최소화**: 복잡한 설정 단계 제거
- **명확한 역할**: 각 API의 단일 책임 원칙

### 1.2. 서비스 개요
'쫑알쫑알'은 아이와의 실시간 음성 대화를 통해 개인화된 동화를 AI로 생성하는 서비스입니다.

**핵심 플로우:**
```
[아이] ↔ [쫑이(대화)] → [아리(동화생성)] → [완성된 멀티미디어 동화]
```

### 1.3. API 개요 (총 7개)
1. **WebSocket 통합 대화** - 실시간 대화 + 음성 클로닝 + 동화 생성
2. **JWT 인증** - 간단한 토큰 발급
3. **동화 생성** - 직접 동화 생성 요청
4. **동화 조회** - 완성된 동화 확인
5. **가족 음성 업로드** - 엄마/아빠 목소리 클로닝
6. **가족 음성 조회** - 업로드된 가족 음성 목록
7. **가족 음성 삭제** - 불필요한 음성 제거 

---

## 2. 핵심 데이터 모델 (Core Data Models)

### 2.1. `StoryRequest` - 동화 생성 요청
```json
{
  "child_name": "민준",
  "age": 7,
  "interests": ["공주", "마법"],
  "conversation_summary": "공주와 마법에 대해 이야기했습니다",
  "story_elements": {
    "main_character": "용감한 공주",
    "setting": "마법의 성",
    "theme": "용기"
  },
  "voice_config": {
    "child_voice_id": "voice_child_123",        // 아이 음성
    "mom_voice_id": "voice_mom_789",            // 엄마 음성
    "dad_voice_id": "voice_dad_012",            // 아빠 음성
    "narrator_voice": "mother"                     // "default" | "mother" | "dad" | 
  }
}
```

**음성 역할 매핑 규칙:**
- **아이 음성**: 주인공 캐릭터 음성으로 사용
- **엄마 음성**: 내레이터, 엄마 캐릭터, 여성 조연 음성으로 사용
- **아빠 음성**: 내레이터, 아빠 캐릭터, 남성 조연, 악역 음성으로 사용  
- **narrator_voice 옵션**: "mother"(엄마), "dad"(아빠), "default"(기본 AI)
- **자동 할당**: narrator_voice 미지정 시 우선순위: 엄마 → 아빠 → 아이 → 기본

### 2.2. `Chapter` - 동화 챕터
```json
{
  "chapter_number": 1,
  "title": "마법의 성",
  "content": "옛날 옛적, 마법의 성에 용감한 공주가 살고 있었습니다...",
  "image_url": "/files/story_456_ch1.jpg", 
  "audio_url": "/files/story_456_ch1.mp3"  
}
```

### 2.3. `Story` - 완성된 동화
```json
{
  "story_id": "story_456",
  "title": "용감한 공주의 모험",
  "status": "completed",  // "generating" | "completed" | "failed"
  "chapters": [/* Chapter 배열 */],
  "created_at": "2024-07-30T10:00:00Z",
  "generation_time": 45.2  // 초 단위, 선택사항
}
```

---

## 3. 에러 처리 (Error Handling)

### 3.1. 표준 에러 응답
```json
{
  "success": false,
  "error": {
    "code": "STORY_GENERATION_FAILED",
    "message": "동화 생성 중 오류가 발생했습니다."
  }
}
```

### 3.2. 주요 에러 코드
| 코드 | HTTP 상태 | 설명 |
|------|-----------|------|
| `INVALID_TOKEN` | 401 | JWT 토큰 무효 |
| `VALIDATION_ERROR` | 400 | 요청 데이터 오류 |
| `STORY_NOT_FOUND` | 404 | 동화를 찾을 수 없음 |
| `STORY_GENERATION_FAILED` | 500 | 동화 생성 실패 |
| `VOICE_CLONE_FAILED` | 500 | 음성 클로닝 실패 |

---

## 4. API 엔드포인트 (총 4개)

### 4.1. 통합 대화 WebSocket
```
WebSocket: /wss/v1/audio?token={jwt_token}&child_name={name}&age={age}
```

**목적**: 아이와의 실시간 대화 + 음성 클로닝 + 동화 생성

**메시지 플로우:**
```json
// 1. 대화 시작
{ "type": "start_conversation" }

// 2. 사용자 메시지 (Google STT 결과)
{ "type": "user_message", "text": "공주님 이야기 해줘" }

// 3. 음성 클로닝용 바이너리 데이터 전송

// 4. 서버 응답들
{ "type": "jjong_response", "text": "어떤 공주님을 좋아해?", "audio_url": "..." }
{ "type": "voice_clone_ready", "voice_id": "child_voice_123" }
{ "type": "story_completed", "story_id": "story_456" }
```

### 4.2. JWT 인증
```http
POST /api/v1/auth/token
```
**Request:**
```json
{ "user_id": "parent_123" }
```
**Response:**
```json
{ "access_token": "jwt_token", "expires_in": 3600 }
```

### 4.3. 동화 생성 (직접 호출)
```http
POST /api/v1/stories
```
**Request:** `StoryRequest` 객체

**Response:** `Story` 객체
```json
{
  "story_id": "story_456",
  "title": "용감한 공주의 모험",
  "status": "completed",
  "chapters": [
    {
      "chapter_number": 1,
      "title": "마법의 성",
      "content": "옛날 옛적, 마법의 성에...",
      "image_url": "/files/story_456_ch1.jpg",
      "audio_url": "/files/story_456_ch1.mp3"
    }
  ],
  "created_at": "2024-07-30T10:00:00Z",
  "generation_time": 45.2
}
```

### 4.4. 동화 조회
```http
GET /api/v1/stories/{story_id}
```
**Response:** 위와 동일한 `Story` 구조

---

## 5. 구현 가이드 (Implementation Guide)

### 5.1. 단순화된 ChatBot B 사용법
```python
chatbot_b = ChatBotB(
    target_age=7,
    story_outline=outline,
    voice_config={
        "child_voice_id": "voice_child_123",
        "parent_voice_id": "voice_parent_456",
        "narrator_voice": "parent"
    }
)
result = await chatbot_b.generate_story()
```

### 5.2. 쫑이-아리 협업 플로우
```python
# WebSocket에서 대화 완료 시
async def handle_conversation_end(websocket, conversation_data):
    # 1. 동화 생성 시작 알림
    await websocket.send_json({
        "type": "story_generation_started",
        "message": "Creating your fairy tale..."
    })
    
    # 2. 쫑이-아리 협업
    story_id = await collaborator.create_story(
        conversation_data["child_name"],
        conversation_data["age"],
        conversation_data
    )
    
    # 3. 완성 알림
    await websocket.send_json({
        "type": "story_completed",
        "story_id": story_id,
        "api_url": f"/api/v1/stories/{story_id}"
    })
```

### 5.3. API 호출 예시
```javascript
// 1. JWT 토큰 발급
const authResponse = await fetch('/api/v1/auth/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: 'parent_123' })
});
const { access_token } = await authResponse.json();

// 2. 동화 생성
const storyResponse = await fetch('/api/v1/stories', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${access_token}`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        child_name: "민준",
        age: 7,
        interests: ["공주", "마법"],
        conversation_summary: "공주와 마법에 대해 이야기했습니다",
        story_elements: {
            main_character: "용감한 공주",
            setting: "마법의 성",
            theme: "용기"
        },
        voice_config: {
          child_voice_id: "voice_child_123",
          parent_voice_id: "voice_parent_456",
          narrator_voice: "parent"
        }
    })
});
const story = await storyResponse.json();

// 3. 동화 조회
const getStoryResponse = await fetch(`/api/v1/stories/${story.story_id}`, {
    headers: { 'Authorization': `Bearer ${access_token}`}
});
const completedStory = await getStoryResponse.json();
```

---

## 6. 개발자 도구 (Developer Tools)

- **API 자동 문서 (Swagger UI)**: `http://localhost:8000/docs`
- **대체 API 문서 (ReDoc)**: `http://localhost:8000/redoc`

---
