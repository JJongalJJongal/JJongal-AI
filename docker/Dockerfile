# PeepPeep (쫑알쫑알) - 아동 AI 동화 생성 시스템
# 멀티스테이지 빌드를 사용하여 이미지 크기 최적화

# =============================================================================
# Stage 1: Build Stage - 빌드 도구 및 컴파일 의존성 설치
# =============================================================================
FROM python:3.11-slim AS builder

# 빌드 인수
ARG DEBIAN_FRONTEND=noninteractive

# 빌드 도구 설치 (최소한만)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    pkg-config \
    libsndfile1-dev \
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
FROM python:3.11-slim AS runtime

# 빌드 인수
ARG DEBIAN_FRONTEND=noninteractive

# 메타데이터
LABEL maintainer="PeepPeep Team"
LABEL description="PeepPeep (쫑알쫑알) - 아동 AI 동화 생성 시스템"
LABEL version="1.0.0"

# 런타임 의존성만 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 비루트 사용자 생성 (보안 강화)
RUN useradd --create-home --shell /bin/bash peeppeep_user

# 가상환경을 빌드 스테이지에서 복사
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 작업 디렉토리 설정
WORKDIR /app

# 필수 파일만 선택적으로 복사
COPY --chown=peeppeep_user:peeppeep_user src/ /app/src/
COPY --chown=peeppeep_user:peeppeep_user shared/ /app/shared/
COPY --chown=peeppeep_user:peeppeep_user config/ /app/config/
COPY --chown=peeppeep_user:peeppeep_user app.py main.py /app/
COPY --chown=peeppeep_user:peeppeep_user data/chroma_db/ /app/data/chroma_db/

# 필요한 디렉토리 생성
RUN mkdir -p /app/output /app/logs \
    && chown -R peeppeep_user:peeppeep_user /app

# Python 경로 설정
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 애플리케이션 설정
ENV APP_ENV=production
ENV INTEGRATED_MODE=true
ENV CHROMA_DB_PATH=/app/src/data/vector_db
ENV LOG_LEVEL=INFO

# 성능 최적화 설정
ENV TORCH_NUM_THREADS=4 
ENV OMP_NUM_THREADS=4
ENV TOKENIZERS_PARALLELISM=false

# 오디오 라이브러리 설정
ENV LIBROSA_CACHE_DIR=/tmp/librosa_cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# 포트 노출
EXPOSE 8000

# 사용자 전환
USER peeppeep_user

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# 시작 스크립트 생성 (정리된 버전)
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "PeepPeep (쫑알쫑알) AI 시스템 시작 중..."\n\
echo "Python 버전: $(python --version)"\n\
echo "설치된 패키지 확인..."\n\
\n\
# 디렉토리는 Docker 빌드 시 이미 생성되었으므로, 여기서는 확인 메시지만 출력합니다.\n\
echo "필수 디렉토리 확인 완료."\n\
\n\
# FastAPI 서버 시작\n\
echo "FastAPI 서버 시작 중... (포트: 8000)"\n\
exec uvicorn app:app \\\n\
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
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 