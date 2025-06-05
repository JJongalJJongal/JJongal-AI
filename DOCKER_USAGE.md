# 🐳 Docker 사용 가이드 - 꼬꼬북 AI 시스템

## 📋 목차
- [Docker 빌드 및 실행](#docker-빌드-및-실행)
- [Swagger API 문서 접근](#swagger-api-문서-접근)
- [개발 환경 설정](#개발-환경-설정)
- [프로덕션 배포](#프로덕션-배포)
- [문제 해결](#문제-해결)

---

## 🚀 Docker 빌드 및 실행

### 1. 기본 빌드 및 실행

```bash
# 1. Docker 이미지 빌드
docker build -t ccb-ai:latest .

# 2. 환경 변수 파일 준비
cp .env.example .env
# .env 파일을 열어서 필요한 API 키들을 설정하세요

# 3. Docker 컨테이너 실행
docker run -d \
  --name ccb-ai-app \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs \
  ccb-ai:latest
```

### 2. Docker Compose로 전체 스택 실행 (권장)

```bash
# 전체 서비스 시작 (백그라운드)
docker-compose up -d

# 로그 확인
docker-compose logs -f ccb-ai

# 서비스 중지
docker-compose down

# 볼륨까지 완전 제거
docker-compose down -v
```

### 3. 개발 모드 실행

```bash
# 개발 환경용 docker-compose 실행 (자동 리로드 포함)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# 또는 직접 개발 모드로 실행
docker run -it --rm \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd):/app \
  -v $(pwd)/output:/app/output \
  ccb-ai:latest \
  uvicorn chatbot.integration_api:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📚 Swagger API 문서 접근

### 1. Swagger UI 접근 방법

Docker 컨테이너가 실행된 후, 웹 브라우저에서 다음 URL로 접근하세요:

```
# Swagger UI (추천)
http://localhost:8000/docs

# ReDoc (대안)
http://localhost:8000/redoc

# OpenAPI JSON 스키마
http://localhost:8000/openapi.json
```

### 2. API 테스트 방법

#### 방법 1: Swagger UI에서 직접 테스트
1. `http://localhost:8000/docs` 접속
2. 원하는 API 엔드포인트 클릭
3. "Try it out" 버튼 클릭
4. 필요한 파라미터 입력
5. "Execute" 버튼으로 테스트 실행

#### 방법 2: curl을 사용한 테스트
```bash
# 헬스체크
curl -X GET "http://localhost:8000/health"

# JWT 토큰 획득 (테스트용)
curl -X GET "http://localhost:8000/api/test-token"

# 이야기 생성 (토큰 필요)
curl -X POST "http://localhost:8000/api/v1/stories" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "child_profile": {
      "name": "테스트",
      "age": 7,
      "interests": ["공주", "마법"]
    },
    "conversation_data": {
      "messages": [
        {
          "content": "공주님 이야기 만들어줘",
          "timestamp": "2024-01-01T12:00:00Z"
        }
      ]
    }
  }'
```

#### 방법 3: Python requests 사용
```python
import requests

# 기본 URL
BASE_URL = "http://localhost:8000"

# 헬스체크
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# JWT 토큰 획득
token_response = requests.get(f"{BASE_URL}/api/test-token")
token = token_response.json()["access_token"]

# API 호출 시 헤더에 토큰 포함
headers = {"Authorization": f"Bearer {token}"}

# 이야기 생성
story_data = {
    "child_profile": {
        "name": "민지",
        "age": 7,
        "interests": ["공주", "마법", "동물"]
    },
    "conversation_data": {
        "messages": [
            {
                "content": "공주님이 나오는 이야기 만들어줘",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        ]
    }
}

response = requests.post(
    f"{BASE_URL}/api/v1/stories",
    json=story_data,
    headers=headers
)
print(response.json())
```

---

## 🛠 개발 환경 설정

### 1. 환경 변수 설정

`.env` 파일에 다음 변수들을 설정하세요:

```bash
# API 키 (필수)
OPENAI_API_KEY=your_openai_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# 인증 토큰
WS_AUTH_TOKEN=secure_default_token

# 서비스 설정
INTEGRATED_MODE=true
LOG_LEVEL=INFO

# Redis 설정 (옵션)
REDIS_PASSWORD=your_redis_password

# 성능 최적화
TORCH_NUM_THREADS=4
OMP_NUM_THREADS=4
TOKENIZERS_PARALLELISM=false
```

### 2. 볼륨 마운트 설명

```bash
# 출력 파일 저장
-v $(pwd)/output:/app/output

# 로그 파일 저장
-v $(pwd)/logs:/app/logs

# 벡터 DB 데이터 (영구 저장 필요 시)
-v $(pwd)/chatbot/data:/app/chatbot/data

# 개발 시 코드 동기화
-v $(pwd):/app
```

---

## 🚀 프로덕션 배포

### 1. 프로덕션 빌드

```bash
# 프로덕션용 이미지 빌드
docker build -t ccb-ai:production .

# 멀티 플랫폼 빌드 (ARM64, AMD64)
docker buildx build --platform linux/amd64,linux/arm64 -t ccb-ai:latest .
```

### 2. 프로덕션 환경 변수

```bash
# 프로덕션 환경 설정
export APP_ENV=production
export LOG_LEVEL=INFO
export WORKERS=2

# 보안 설정
export SECURE_MODE=true
export ALLOWED_ORIGINS="https://yourdomain.com"
```

### 3. 리소스 제한

```bash
# 메모리 및 CPU 제한
docker run -d \
  --name ccb-ai-prod \
  --memory=8g \
  --cpus=4 \
  -p 8000:8000 \
  ccb-ai:production
```

---

## 🔧 문제 해결

### 1. 일반적인 문제들

#### 포트 충돌
```bash
# 포트 사용 중인 프로세스 확인
lsof -i :8000

# 다른 포트로 실행
docker run -p 8001:8000 ccb-ai:latest
```

#### 메모리 부족
```bash
# Docker 메모리 제한 확인
docker stats

# 메모리 제한 늘리기
docker run --memory=12g ccb-ai:latest
```

#### API 키 오류
```bash
# 환경 변수 확인
docker exec ccb-ai-app env | grep API_KEY

# .env 파일 확인
cat .env
```

### 2. 로그 확인

```bash
# 컨테이너 로그 확인
docker logs ccb-ai-app

# 실시간 로그 모니터링
docker logs -f ccb-ai-app

# 특정 라인 수만 확인
docker logs --tail 50 ccb-ai-app
```

### 3. 컨테이너 진단

```bash
# 컨테이너 상태 확인
docker ps

# 컨테이너 내부 접속
docker exec -it ccb-ai-app bash

# 헬스체크 확인
docker inspect ccb-ai-app | grep Health

# 리소스 사용량 확인
docker stats ccb-ai-app
```

### 4. 네트워크 문제

```bash
# 네트워크 연결 테스트
docker exec ccb-ai-app curl -f http://localhost:8000/health

# 포트 매핑 확인
docker port ccb-ai-app
```

---

## 📈 성능 최적화

### 1. 이미지 크기 최적화

```bash
# 이미지 크기 확인
docker images ccb-ai

# 불필요한 이미지 정리
docker image prune

# 전체 시스템 정리
docker system prune -a
```

### 2. 캐시 활용

```bash
# 빌드 캐시 재사용
docker build --cache-from ccb-ai:latest -t ccb-ai:latest .

# 다단계 빌드 캐시 최적화
docker build --target builder -t ccb-ai:builder .
docker build --cache-from ccb-ai:builder -t ccb-ai:latest .
```

---

## 📞 지원

문제가 발생하거나 추가 도움이 필요한 경우:
1. 이 문서의 문제 해결 섹션을 참조하세요
2. GitHub Issues에 문제를 보고하세요
3. 로그 파일과 환경 정보를 포함해주세요

**Happy Coding! 🎉** 