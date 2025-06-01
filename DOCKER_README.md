# 꼬꼬북 동화 생성 프로젝트 Docker 가이드

이 문서는 꼬꼬북 동화 생성 프로젝트의 Docker 컨테이너화 및 배포 가이드입니다.

## 📋 목차

- [사전 요구사항](#사전-요구사항)
- [빠른 시작](#빠른-시작)
- [환경별 실행](#환경별-실행)
- [환경 변수 설정](#환경-변수-설정)
- [볼륨 및 데이터 저장](#볼륨-및-데이터-저장)
- [문제 해결](#문제-해결)

## 🔧 사전 요구사항

다음 소프트웨어가 설치되어 있어야 합니다:

- **Docker** (v20.10 이상)
- **Docker Compose** (v2.0 이상)
- **Make** (선택사항, 편의를 위해)

### Docker 설치 확인

```bash
docker --version
docker-compose --version
```

## 🚀 빠른 시작

### 1. 환경 변수 설정

`.env` 파일을 생성하고 필요한 API 키를 설정하세요:

```bash
# .env 파일 생성
cp .env.example .env

# 필수 환경 변수 설정
export OPENAI_API_KEY="your-openai-api-key"
export ELEVENLABS_API_KEY="your-elevenlabs-api-key"
```

### 2. Docker 이미지 빌드 및 실행

#### Makefile 사용 (권장)

```bash
# 도움말 확인
make help

# 빌드 및 개발 환경 실행
make build
make dev

# 또는 한 번에
make build && make dev
```

#### Docker Compose 직접 사용

```bash
# 프로덕션 환경
docker-compose up -d

# 개발 환경
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### 3. 서비스 확인

```bash
# 헬스체크
curl http://localhost:8000/health

# 또는 Makefile 사용
make health
```

## 🏗️ 환경별 실행

### 개발 환경

개발 환경에서는 소스 코드가 실시간으로 반영됩니다:

```bash
# 개발 환경 실행
make dev

# 또는
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

**개발 환경 특징:**
- 소스 코드 실시간 반영 (hot reload)
- 디버깅 로그 활성화
- 개발용 포트 노출

### 프로덕션 환경

```bash
# 프로덕션 배포
make deploy

# 또는
docker-compose up -d
```

**프로덕션 환경 특징:**
- 최적화된 이미지
- Nginx 리버스 프록시
- Redis 캐싱
- 헬스체크 활성화

## 🔐 환경 변수 설정

### 필수 환경 변수

```env
# API 키들
OPENAI_API_KEY=your-openai-api-key-here
ELEVENLABS_API_KEY=your-elevenlabs-api-key-here

# 애플리케이션 설정
APP_ENV=production
PYTHONPATH=/app

# 데이터베이스
CHROMA_DB_PATH=/app/data/vector_db

# 옵션: 로깅 레벨
LOG_LEVEL=INFO
```

### 환경 변수 파일 구조

```
.env                    # 메인 환경 설정 (Git에 포함되지 않음)
.env.example           # 환경 변수 템플릿
.env.development       # 개발 환경 설정
.env.production        # 프로덕션 환경 설정
```

## 💾 볼륨 및 데이터 저장

### 데이터 볼륨

```yaml
volumes:
  - ./output:/app/output              # 생성된 동화 파일들
  - ./chatbot/data:/app/chatbot/data  # 벡터 DB 및 프롬프트
  - ./logs:/app/logs                  # 로그 파일들
```

### 출력 파일 구조

```
output/
├── audio/           # 생성된 음성 파일
├── images/          # 생성된 이미지 파일
├── stories/         # 완성된 동화 JSON
├── conversations/   # 대화 로그
├── metadata/        # 메타데이터
└── temp/           # 임시 파일
```

## 🔍 모니터링 및 로그

### 로그 확인

```bash
# 실시간 로그 확인
make logs

# 또는
docker-compose logs -f ccb-ai
```

### 컨테이너 상태 확인

```bash
# 컨테이너 정보
make info

# 헬스체크
make health

# 컨테이너 내부 접속
make shell
```

## 🧪 테스트 실행

```bash
# 컨테이너에서 테스트 실행
make test

# 또는 직접 실행
docker run --rm -v $(pwd):/app ccb-ai:latest python -m pytest chatbot/tests/ -v
```

## 🔧 유용한 명령어

### 개발 도구

```bash
# Jupyter Notebook 실행 (개발용)
make dev-tools

# 접속: http://localhost:8888
```

### 데이터베이스 관리

```bash
# ChromaDB 데이터 초기화
docker exec -it ccb-ai-app python -c "
from chatbot.data.vector_db.core import VectorDB
db = VectorDB()
db.reset_collection()
"
```

### 서비스 관리

```bash
# 서비스 정지
make stop

# 완전 정리 (볼륨 포함)
make clean

# 이미지 재빌드
make build
```

## ⚠️ 문제 해결

### 일반적인 문제들

#### 1. 포트 충돌
```bash
# 포트 사용 중인 프로세스 확인
lsof -i :8000

# Docker Compose 포트 변경
sed -i 's/8000:8000/8001:8000/' docker-compose.yml
```

#### 2. 메모리 부족
```bash
# Docker 메모리 설정 확인
docker system df
docker system prune -f

# 컴포즈 파일에 메모리 제한 추가
deploy:
  resources:
    limits:
      memory: 4G
```

#### 3. 볼륨 권한 문제
```bash
# 권한 수정
sudo chown -R $USER:$USER ./output ./logs

# 또는 컨테이너에서 실행
docker exec -it ccb-ai-app chown -R app:app /app/output
```

#### 4. API 키 오류
```bash
# 환경 변수 확인
docker exec -it ccb-ai-app env | grep API_KEY

# .env 파일 확인
cat .env
```

### 로그 분석

```bash
# 에러 로그만 필터링
docker-compose logs ccb-ai | grep ERROR

# 특정 시간대 로그
docker-compose logs --since="2024-01-01T00:00:00" ccb-ai
```

## 📚 추가 자료

- [Docker 공식 문서](https://docs.docker.com/)
- [Docker Compose 문서](https://docs.docker.com/compose/)
- [FastAPI Docker 가이드](https://fastapi.tiangolo.com/deployment/docker/)

## 🤝 기여하기

Docker 설정 개선 사항이나 문제점을 발견하신 경우 이슈를 등록해 주세요.

---

**문의사항이 있으시면 개발팀에 연락해 주세요.** 