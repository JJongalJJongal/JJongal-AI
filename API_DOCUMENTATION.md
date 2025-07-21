# 쫑알쫑알 API 명세서 v1.0

> 이 문서는 FastAPI와 Python의 공식 문서, 그리고 [Apidog의 API 기술 아티클](https://apidog.com/kr/blog/api-skills-ai-developer-needs-kr/)을 참고하여, 현대적이고 안정적인 AI 기반 애플리케이션의 API 설계를 목표로 작성되었습니다.

## 1. 소개 (Introduction)

### 1.1. 문서의 목적
본 문서는 '쫑알쫑알' 프로젝트의 백엔드 API에 대한 공식 명세입니다. 클라이언트(프론트엔드, 앱) 개발자와 백엔드 개발자 간의 명확한 약속이며, 프로젝트의 일관성, 확장성, 유지보수성을 확보하기 위한 기술적 청사진입니다.

### 1.2. 이 문서 읽는 법
- **핵심 설계 철학**을 먼저 이해하면 "왜" 이렇게 설계되었는지 파악하는 데 도움이 됩니다.
- **전역 규칙**은 모든 API에 공통으로 적용되므로 반드시 숙지해야 합니다.
- 각 **엔드포인트** 설명은 요청(Request)과 응답(Response)의 상세 구조와 예시, 그리고 발생 가능한 에러 케이스를 포함합니다.

### 1.3. '쫑알쫑알' 서비스 개요
'쫑알쫑알'은 아이와의 실시간 음성 대화를 통해 개인화된 동화를 AI로 생성하고, 관련 멀티미디어(이미지, 음성)를 함께 제공하는 인터랙티브 스토리텔링 플랫폼입니다.

### 1.4. 시스템 아키텍처 (System Architecture)
'쫑알쫑알' 서비스는 Nginx를 리버스 프록시로 사용하는 현대적인 웹 아키텍처를 따릅니다. 클라이언트의 모든 요청은 Nginx를 통해 백엔드 FastAPI 애플리케이션으로 라우팅됩니다.

**요청 흐름 (Request Flow):**
```
Client (Web/App)
       │
       ├─ HTTPS (REST API) ─> Nginx (Port 443/80) ─> FastAPI (Port 8000)
       │
       └─ WSS (WebSocket) ─> Nginx (Port 443/80) ─> FastAPI (Port 8000)
```

**Nginx의 주요 역할:**
- **리버스 프록시 및 로드 밸런싱**: 외부 요청을 안전하게 내부 서비스로 전달합니다.
- **SSL/TLS 종료**: HTTPS/WSS 암호화를 처리합니다.
- **속도 제한 및 보안**: 과도한 요청을 제어하고 기본적인 보안 헤더를 추가합니다.
- **정적/미디어 파일 서빙**: 이미지, 오디오 파일 등 생성된 콘텐츠를 직접 제공하여 백엔드의 부하를 줄입니다.

---

## 2. 핵심 설계 철학 (Core Design Philosophy)

- **타입 힌트를 통한 명확성 및 안정성**: Python 타입 힌트를 100% 활용합니다. FastAPI는 이를 통해 **데이터 유효성 검사, 직렬화, API 자동 문서 생성**을 수행하여 코드의 신뢰성을 극대화합니다.
- **Pydantic을 활용한 데이터 모델링**: 모든 외부 데이터(Request/Response)는 Pydantic 모델로 정의하여, 복잡한 JSON 객체를 파이썬 클래스처럼 명확하게 다루고 정교한 유효성 검사를 강제합니다.
- **비동기 우선(Async First)을 통한 고성능**: 데이터베이스 접근, 외부 AI API 호출 등 모든 I/O 작업은 `async def`를 사용하여 비동기적으로 처리함으로써 높은 동시성 환경에서도 뛰어난 성능을 보장합니다.
- **의존성 주입(Dependency Injection)을 통한 재사용성 및 테스트 용이성**: 데이터베이스 세션, 인증 로직 등 공통 기능은 FastAPI의 `Depends` 시스템으로 주입하여 코드 중복을 없애고 단위 테스트를 용이하게 합니다.

---

## 3. 전역 규칙 (Global Conventions)

### 3.1. URL 구조 (URL Structure)
- **Base URL (Production)**: `https://domain.com (or AWS IP)`
- **Base URL (Development)**: `http://localhost:8000`
- **API Versioning**: 모든 REST API 경로는 `/api/v1` 접두사를, WebSocket 경로는 `/wss/v1` 접두사를 포함합니다.

### 3.2. HTTP 메서드 (HTTP Methods)
- **`GET`**: 리소스 조회.
- **`POST`**: 새 리소스 생성.
- **`PATCH`**: 기존 리소스 부분 수정.
- **`DELETE`**: 리소스 삭제.

### 3.3. JSON 규약 (JSON Conventions)
- 모든 요청/응답의 본문은 `JSON` 형식입니다.
- **필드 명명 규칙**: `snake_case`를 사용합니다. (예: `"created_at"`)
- **날짜 및 시간 형식**: **UTC** 기준, **ISO 8601** 형식의 문자열을 사용합니다. (예: `2024-07-30T10:30:00.123Z`)
- **빈 값 처리**: 빈 문자열(`""`) 대신 명시적인 `null`을 사용합니다.

### 3.4. 인증 (Authentication)
- 모든 API 엔드포인트는 **JWT(JSON Web Token)**를 통한 Bearer 인증을 요구합니다.
- **REST API**: `Authorization: Bearer <your_jwt_token>` 헤더를 사용합니다.
- **WebSocket API**: `wss://.../ws/v1/audio?token=<your_jwt_token>` 쿼리 파라미터를 사용합니다.

### 3.5. 페이지네이션 (Pagination)
- 목록 조회(`GET`) 엔드포인트는 커서 기반 페이지네이션을 지원합니다.
- **요청 파라미터**: `?page_size=20&start_cursor=...`
- **응답 구조**:
  ```json
  {
    "object": "list",
    "results": [ ... ],
    "has_more": true,
    "next_cursor": "base64_encoded_string"
  }
  ```

### 3.6. 속도 및 크기 제한 (Rate & Size Limiting)
- **속도 제한**: 시스템 안정성을 위해 분당 요청 수를 제한하며, 초과 시 `429 Too Many Requests` 에러를 반환합니다.
- **크기 제한**: 요청 본문의 전체 크기는 1MB, 텍스트 필드는 10,000자로 제한됩니다.

---

## 4. 응답 및 에러 처리 (Responses & Errors)

### 4.1. 표준 응답 구조
- **성공 (`2xx`)**:
  ```json
  {
    "success": true,
    "message": "요청이 성공적으로 처리되었습니다.",
    "data": { ... }
  }
  ```
- **실패 (`4xx`, `5xx`)**:
  ```json
  {
    "success": false,
    "error": {
      "code": "UNIQUE_ERROR_CODE",
      "message": "사용자가 이해할 수 있는 에러 메시지."
    }
  }
  ```

### 4.2. 공통 에러 코드
| 코드(Code)              | HTTP 상태 | 의미                                   |
| ----------------------- | --------- | -------------------------------------- |
| `INVALID_TOKEN`         | 401       | 토큰이 유효하지 않거나 만료됨          |
| `AUTHENTICATION_FAILED` | 401       | 자격 증명실패                       |
| `INSUFFICIENT_PERMISSION` | 403       | 해당 리소스/작업에 대한 권한이 없음     |
| `VALIDATION_ERROR`      | 400       | 요청 데이터 형식이 잘못되었거나 필수값 누락 |
| `RESOURCE_NOT_FOUND`    | 404       | 요청한 리소스를 찾을 수 없음           |
| `RATE_LIMIT_EXCEEDED`   | 429       | 요청 한도를 초과함                     |
| `SERVER_ERROR`          | 500       | 예측하지 못한 서버 내부 오류 발생      |

---

## 5. 핵심 객체 모델 (Core Object Models)

### 5.1. `File` 객체
- `object` (string): `"file"`
- `type` (string): `"internal"`
- `url` (string, URL): 파일 접근용 임시 서명 URL.
- `expiry_time` (string, ISO 8601): URL 만료 시각.

### 5.2. `Chapter` 객체
- `id` (string, UUID)
- `object` (string): `"chapter"`
- `chapter_number` (integer)
- `title` (string)
- `content` (string): 챕터 텍스트 내용.
- `image` (File object, nullable)
- `audio` (File object, nullable)

### 5.3. `Story` 객체
- `id` (string, UUID)
- `object` (string): `"story"`
- `owner_id` (string, UUID): 소유자 `User`의 ID.
- `title` (string)
- `status` (string, Enum): `"pending" | "in_progress" | "completed" | "failed"`
- `chapters` (array of `Chapter` objects)
- `created_at` (string, ISO 8601)
- `updated_at` (string, ISO 8601)

### 5.4. `User` 객체
- `id` (string, UUID)
- `object` (string): `"user"`
- `name` (string)
- `age` (integer)
- `created_at` (string, ISO 8601)

---

## 6. REST API 엔드포인트

### 6.1. 인증 (Authentication)

#### `POST /api/v1/auth/token`
**요약**: 자격 증명을 받아 Access Token과 Refresh Token을 발급합니다.
- **요청 본문**: `{ "user_id": "string", "password": "..." }` (향후 확장)
- **성공 응답 (`200 OK`)**:
  ```json
  {
    "data": {
      "token_type": "Bearer",
      "access_token": "...",
      "expires_in": 3600,
      "refresh_token": "..."
    }
  }
  ```
- **에러 응답**: `401 AUTHENTICATION_FAILED`, `400 VALIDATION_ERROR`

#### `POST /api/v1/auth/refresh`
**요약**: 유효한 Refresh Token으로 새로운 Access Token을 발급받습니다.
- **요청 본문**: `{ "refresh_token": "..." }`
- **성공 응답 (`200 OK`)**: 위 `token` 발급과 동일한 구조의 응답.
- **에러 응답**: `401 INVALID_TOKEN`

### 6.2. 사용자 (Users)

#### `GET /api/v1/users/me`
**요약**: 현재 인증된 사용자의 정보를 조회합니다.
- **성공 응답 (`200 OK`)**: `User` 객체.
- **에러 응답**: `401 INVALID_TOKEN`

### 6.3. 이야기 (Stories)

#### `POST /api/v1/stories`
**요약**: 대화 내용과 사용자 선호를 바탕으로 새로운 이야기 생성을 비동기적으로 요청합니다.
- **요청 본문**:
  ```json
  {
    "user_preferences": {
      "theme": "모험",
      "characters": ["용감한 토끼", "지혜로운 거북이"],
      "length": "medium"
    },
    "conversation_id": "대화_세션_UUID"
  }
  ```
- **성공 응답 (`202 Accepted`)**: 생성 중인 `Story` 객체의 초기 상태를 반환합니다.
- **에러 응답**: `400 VALIDATION_ERROR`

#### `GET /api/v1/stories`
**요약**: 사용자의 모든 이야기 목록을 필터링, 정렬, 페이지네이션하여 조회합니다.
- **쿼리 파라미터**:
    - `status` (string, optional): `completed`, `in_progress` 등 상태로 필터링.
    - `sort_by` (string, optional, default: `created_at`): `title`, `updated_at` 등 정렬 기준.
    - `order` (string, optional, default: `desc`): `asc` 또는 `desc` 정렬 순서.
    - `page_size`, `start_cursor`: 페이지네이션.
- **성공 응답 (`200 OK`)**: 페이지네이션 구조를 따르는 `Story` 객체 목록.

#### `GET /api/v1/stories/{story_id}`
**요약**: 특정 이야기의 상세 정보를 조회합니다.
- **경로 파라미터**: `story_id` (string, UUID)
- **성공 응답 (`200 OK`)**: `Story` 객체.
- **에러 응답**: `404 RESOURCE_NOT_FOUND`, `403 INSUFFICIENT_PERMISSION`

#### `PATCH /api/v1/stories/{story_id}`
**요약**: 특정 이야기의 정보를 수정합니다 (예: 제목 변경).
- **요청 본문**: `{ "title": "새로운 멋진 제목" }`
- **성공 응답 (`200 OK`)**: 수정된 `Story` 객체.
- **에러 응답**: `404 RESOURCE_NOT_FOUND`, `400 VALIDATION_ERROR`

#### `DELETE /api/v1/stories/{story_id}`
**요약**: 특정 이야기를 삭제합니다.
- **성공 응답 (`204 No Content`)**: 성공 시 본문 내용 없음.
- **에러 응답**: `404 RESOURCE_NOT_FOUND`

---

## 7. WebSocket API 엔드포인트

### 7.1. 실시간 음성 대화 (`/wss/v1/audio`)
- **목적**: 프론트엔드에서 Google STT로 변환된 텍스트를 받아 챗봇 대화를 처리하고, Voice Clone용 오디오 샘플을 수집합니다.
- **핵심 원칙**: 프론트엔드는 Google STT를 사용하여 음성을 텍스트로 변환 후 전송하며, Voice Clone용 오디오는 별도 바이너리로 전송합니다.
- **프로토콜 흐름**:
  1. **연결 수립**: `wss://.../wss/v1/audio?token=<jwt>&child_name=민준&age=7`
  2. **대화 시작 (Client→Server, JSON)**: 대화 세션 초기화
     ```json
     { "type": "start_conversation", "payload": { "child_name": "민준", "age": 7, "interests": ["공주", "마법"] } }
     ```
     3. **텍스트 메시지 전송 (Client→Server, JSON)**: Google STT 결과를 JSON으로 전송
     ```json
     { "type": "user_message", "text": "공주님이 나오는 이야기 해줘" }
     ```
  4. **Voice Clone 오디오 전송 (Client→Server, Binary)**: 음성 복제용 오디오 파일을 바이너리로 전송
  5. **서버 응답 (Server→Client, JSON)**: 서버는 다양한 이벤트를 JSON 메시지로 전송합니다.
     - **AI 응답**: `{ "type": "ai_response", "text": "...", "audio_url": "https://...", "user_text": "..." }`
     - **Voice Clone 진행상황**: `{ "type": "voice_clone_progress", "sample_count": 3, "ready_for_cloning": false }`
     - **Voice Clone 성공**: `{ "type": "voice_clone_success", "voice_id": "...", "message": "목소리 복제 완료!" }`
     - **에러 발생**: `{ "type": "error", "error_message": "...", "error_code": "..." }`
  6. **대화 종료 (Client→Server, JSON)**: `{ "type": "end_conversation" }`

---

## 8. 개발자 도구 (Developer Tools)
- **API 자동 문서 (Swagger UI)**: `http://localhost:8000/docs`
- **대체 API 문서 (ReDoc)**: `http://localhost:8000/redoc`
> FastAPI에 의해 자동으로 생성되며, 코드 변경 시 항상 최신 상태를 유지합니다.