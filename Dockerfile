# 꼬꼬북(CCB_AI) - 아동 AI 동화 생성 시스템
# 멀티스테이지 빌드를 사용하여 이미지 크기 최적화

# =============================================================================
# Stage 1: Build Stage - 빌드 도구 및 컴파일 의존성 설치
# =============================================================================
FROM python:3.11-slim as builder

# 빌드 인수
ARG DEBIAN_FRONTEND=noninteractive

# 빌드 도구 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치를 위한 가상환경 생성
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# requirements.txt 복사 및 Python 패키지 설치
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# Stage 2: Runtime Stage - 실행 환경 구성
# =============================================================================
FROM python:3.11-slim as runtime

# 빌드 인수
ARG DEBIAN_FRONTEND=noninteractive

# 메타데이터
LABEL maintainer="CCB_AI Team"
LABEL description="꼬꼬북 - 아동 AI 동화 생성 시스템"
LABEL version="1.0.0"

# 시스템 의존성 설치 (런타임에 필요한 것들만)
RUN apt-get update && apt-get install -y \
    # FFmpeg (음성 처리용)
    ffmpeg \
    # 네트워킹 도구
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 비루트 사용자 생성 (보안 강화)
RUN useradd --create-home --shell /bin/bash ccb_user

# 가상환경을 빌드 스테이지에서 복사
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 작업 디렉토리 설정
WORKDIR /app

# 애플리케이션 소유자 설정
RUN chown -R ccb_user:ccb_user /app

# 필요한 디렉토리 생성
RUN mkdir -p /app/output \
             /app/logs \
             /app/chatbot/data/vector_db \
             /app/chatbot/data/processed \
             /app/chatbot/data/raw \
             /app/shared \
    && chown -R ccb_user:ccb_user /app

# 애플리케이션 코드 복사
COPY --chown=ccb_user:ccb_user . /app/

# Python 경로 설정
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 애플리케이션 설정
ENV APP_ENV=production
ENV INTEGRATED_MODE=true
ENV CHROMA_DB_PATH=/app/chatbot/data/vector_db
ENV LOG_LEVEL=INFO

# 성능 최적화 설정
ENV TORCH_NUM_THREADS=4
ENV OMP_NUM_THREADS=4
ENV TOKENIZERS_PARALLELISM=false

# 포트 노출
EXPOSE 8000

# 사용자 전환
USER ccb_user

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 시작 스크립트 생성
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "🚀 꼬꼬북 AI 시스템 시작 중..."\n\
echo "📁 작업 디렉토리: $(pwd)"\n\
echo "🐍 Python 버전: $(python --version)"\n\
echo "📦 설치된 패키지 확인..."\n\
\n\
# 필요한 디렉토리 확인 및 생성\n\
mkdir -p /app/output /app/logs /app/chatbot/data\n\
\n\
# FastAPI 서버 시작 (통합 API) - 올바른 경로로 수정\n\
echo "🌐 FastAPI 서버 시작 중... (포트: 8000)"\n\
exec uvicorn chatbot.workflow.integration_api:app \\\n\
    --host 0.0.0.0 \\\n\
    --port 8000 \\\n\
    --workers 1 \\\n\
    --log-level info \\\n\
    --access-log \\\n\
    --use-colors\n\
' > /app/start.sh && chmod +x /app/start.sh

# 기본 실행 명령
CMD ["/app/start.sh"]

# 개발 모드용 오버라이드 (docker-compose.dev.yml에서 사용)
# CMD ["uvicorn", "chatbot.workflow.integration_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 