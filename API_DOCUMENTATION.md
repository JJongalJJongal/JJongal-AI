# 쫑알쫑알 API 명세서 v1.0

## 1. 소개 (Introduction)

### 1.1. 문서의 목적
이 문서는 '쫑알쫑알' 프로젝트의 백엔드 API에 대한 공식 명세입니다. 클라이언트 개발자와 백엔드 개발자 간의 명확한 소통을 보장하고, 프로젝트의 일관성, 확장성, 유지보수성을 확보하는 것을 목표로 합니다.

이 문서는 FastAPI와 Python의 모범 사례에 기반하여, 단순히 엔드포인트를 나열하는 것을 넘어 **"왜(Why)"** 이러한 설계 원칙을 채택했는지 설명합니다.

### 1.2. '쫑알쫑알' 서비스 개요
'쫑알쫑알'은 아이와의 실시간 음성 대화를 통해 개인화된 동화를 AI로 생성하고, 관련 멀티미디어(이미지, 음성)를 함께 제공하는 인터랙티브 스토리텔링 플랫폼입니다.

---

## 2. 핵심 설계 철학 (Core Design Philosophy)

우리는 Python과 FastAPI의 강력한 기능을 최대한 활용하여 현대적이고 안정적인 API를 구축합니다.

- **타입 힌트를 통한 명확성 및 안정성**: 모든 함수 시그니처와 데이터 모델에 Python 타입 힌트를 적용합니다. FastAPI는 이를 사용하여 요청 데이터의 유효성을 자동으로 검사하고, 응답 데이터를 직렬화하며, 인터랙티브 API 문서를 생성합니다. 이는 개발 과정에서의 실수를 줄이고 코드의 신뢰성을 높입니다.

- **Pydantic을 활용한 데이터 모델링**: 모든 외부 데이터(Request/Response Body)는 Pydantic 모델로 정의합니다. 이는 복잡한 JSON 객체를 파이썬 클래스처럼 명확하게 다루게 해주며, 정교한 유효성 검사 규칙(예: `gt=0`, `max_length=50`, 이메일 형식 검사)을 코드 수준에서 강제할 수 있습니다.

- **비동기 우선(Async First)을 통한 고성능**: 데이터베이스 접근, 외부 API 호출 등 모든 I/O 바운드 작업은 `async def`를 사용하여 비동기적으로 처리하는 것을 원칙으로 합니다. 이는 높은 동시성 환경에서도 뛰어난 성능을 보장합니다.

- **의존성 주입(Dependency Injection)을 통한 재사용성 및 테스트 용이성**: 데이터베이스 세션, 인증 로직 등 공통적으로 사용되는 기능은 FastAPI의 `Depends` 시스템을 통해 의존성으로 주입합니다. 이는 코드 중복을 최소화하고, 각 API 엔드포인트의 단위 테스트를 용이하게 만듭니다.

- **라우터 분리(APIRouter)를 통한 체계적인 관리**: API 엔드포인트는 리소스(`stories`, `users` 등)를 기준으로 파일을 분리하여 `APIRouter`를 통해 관리합니다. 이는 프로젝트의 규모가 커져도 코드베이스를 체계적이고 유지보수하기 쉽게 만듭니다.

---

## 3. 전역 규칙 (Global Conventions)

### 3.1. URL 구조 (URL Structure)
- **Base URL (Production)**: `https://api.jjong알jjong알.com`
- **Base URL (Development)**: `http://localhost:8000`
- **API Versioning**: 모든 REST API 경로는 `/api/v1` 접두사를 포함합니다. (예: `/api/v1/stories`)
- **WebSocket Versioning**: 모든 WebSocket 경로는 `/ws/v1` 접두사를 포함합니다. (예: `/ws/v1/audio`)

### 3.2. 명명 규칙 (Naming Conventions)
- **경로 (Path)**: 복수형 명사를 사용하며, 단어는 하이픈(`-`)으로 구분합니다. (예: `/story-chapters`)
- **JSON 필드**: `snake_case`를 사용합니다. (예: `"created_at"`)

### 3.3. 날짜 및 시간 형식 (Date & Time Format)
- 모든 날짜와 시간 정보는 **UTC**를 기준으로 하며, **ISO 8601** 형식의 문자열로 표현합니다.
  - 예시: `2024-07-29T10:30:00.123Z`

### 3.4. 인증 (Authentication)
- 모든 API 엔드포인트(헬스 체크 제외)는 **JWT(JSON Web Token)**를 통한 Bearer 인증을 요구합니다.
- **요청 헤더**: `Authorization: Bearer <your_jwt_token>`
- **WebSocket 연결**: 쿼리 파라미터로 토큰을 전달합니다. `wss://.../ws/v1/audio?token=<your_jwt_token>`

### 3.5. 페이지네이션 (Pagination)
- 목록을 반환하는 모든 `GET` 엔드포인트는 커서 기반 페이지네이션을 지원합니다.
- **요청 쿼리 파라미터**:
    - `page_size` (integer, optional, default: 20): 한 페이지에 포함할 항목의 수.
    - `start_cursor` (string, optional): 조회를 시작할 위치를 나타내는 커서.
- **응답 본문 구조**:
  ```json
  {
    "object": "list",
    "results": [ ... ],
    "has_more": true,
    "next_cursor": "base64_encoded_cursor_string"
  }
  ```

### 3.6. 속도 및 크기 제한 (Rate & Size Limiting)
- **속도 제한**: 사용자별, IP별로 분당 요청 수를 제한합니다. (예: 60 requests/minute). 초과 시 `429 Too Many Requests` 상태 코드와 함께 에러 응답을 반환합니다.
- **크기 제한**: 요청 본문의 전체 크기는 1MB로 제한됩니다. 텍스트 필드는 10,000자로 제한됩니다.

---

## 4. 응답 및 에러 처리 (Responses & Errors)

### 4.1. 표준 응답 구조 (REST API)
- **성공 (`2xx` 상태 코드)**:
  ```json
  {
    "success": true,
    "message": "요청이 성공적으로 처리되었습니다.",
    "data": { ... } // or [ ... ] or null
  }
  ```
- **실패 (`4xx`, `5xx` 상태 코드)**:
  ```json
  {
    "success": false,
    "error": {
      "code": "UNIQUE_ERROR_CODE",
      "message": "사용자가 이해할 수 있는 에러 메시지입니다.",
      "details": { ... } // (선택) 개발자를 위한 추가 정보
    }
  }
  ```

### 4.2. 표준 메시지 구조 (WebSocket API)
- 모든 메시지는 `event`와 `payload`를 포함하는 JSON 객체입니다.
- **클라이언트 -> 서버**: `{ "event": "event_name", "payload": { ... } }`
- **서버 -> 클라이언트**: `{ "event": "event_name", "payload": { ... } }`

### 4.3. HTTP 상태 코드
- `200 OK`: 요청 성공.
- `201 Created`: 리소스 생성 성공.
- `204 No Content`: 요청은 성공했으나 반환할 콘텐츠가 없음 (예: `DELETE`).
- `400 Bad Request`: 요청 유효성 검사 실패. (에러 응답에 `details` 포함)
- `401 Unauthorized`: 인증 실패 (유효하지 않은 토큰).
- `403 Forbidden`: 인가 실패 (권한 없음).
- `404 Not Found`: 요청한 리소스를 찾을 수 없음.
- `429 Too Many Requests`: 속도 제한 초과.
- `500 Internal Server Error`: 서버 내부 오류.

---

## 5. 핵심 객체 모델 (Core Object Models)

_(이 섹션은 Pydantic 모델과 직접적으로 매핑됩니다)_

### 5.1. `User` 객체
사용자 정보를 나타냅니다.
- `id` (string, UUID): 사용자 고유 ID.
- `object` (string): 항상 `"user"`.
- `name` (string): 아이 이름.
- `age` (integer): 아이 나이.
- `created_at` (string, ISO 8601): 계정 생성 시각.

### 5.2. `Story` 객체
하나의 동화 정보를 나타냅니다.
- `id` (string, UUID): 이야기 고유 ID.
- `object` (string): 항상 `"story"`.
- `owner_id` (string, UUID): 이 이야기를 소유한 사용자 ID.
- `title` (string): 이야기 제목.
- `status` (string, Enum): 생성 상태. (`pending`, `in_progress`, `completed`, `failed`)
- `chapters` (array of `Chapter` objects): 챕터 목록.
- `created_at` (string, ISO 8601): 생성 요청 시각.
- `updated_at` (string, ISO 8601): 마지막 업데이트 시각.

---

## 6. REST API 엔드포인트

### 6.1. 인증 (Authentication)
- **`POST /api/v1/auth/token`**: 토큰 발급
- **`POST /api/v1/auth/refresh`**: 토큰 갱신

### 6.2. 이야기 (Stories)
- **`POST /api/v1/stories`**: 새 이야기 생성
- **`GET /api/v1/stories`**: 내 이야기 목록 조회 (페이지네이션 적용)
- **`GET /api/v1/stories/{story_id}`**: 특정 이야기 상세 조회
- **`DELETE /api/v1/stories/{story_id}`**: 이야기 삭제

---

## 7. WebSocket API 엔드포인트

### 7.1. 실시간 음성 대화 (`/ws/v1/audio`)
아이와 챗봇 간의 실시간 음성 대화를 처리합니다.
- **연결**: `wss://.../ws/v1/audio?token=<jwt>`
- **주요 이벤트**:
    - `C->S: audio_stream`: Base64 인코딩된 오디오 청크 전송
    - `S->C: transcription_update`: 중간/최종 음성 인식 결과
    - `S->C: bot_response`: 챗봇의 텍스트 및 오디오 응답
    - `S->C: error`: 처리 중 발생한 에러

---

## 8. 개발자 도구 (Developer Tools)
- **API 자동 문서 (Swagger UI)**: `http://localhost:8000/docs`
- **대체 API 문서 (ReDoc)**: `http://localhost:8000/redoc`

이 두 문서는 코드(Pydantic 모델, 경로 데코레이터 등)가 변경되면 자동으로 최신화됩니다. 