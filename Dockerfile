# 꼬꼬북 동화 생성 프로젝트 Dockerfile
# Python 3.12 공식 이미지를 베이스로 사용
FROM python:3.12-slim

# 메타데이터 설정
LABEL maintainer="CCB AI Team"
LABEL description="꼬꼬북 - AI 동화 생성 시스템"
LABEL version="1.0.0"

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV APP_ENV=production

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    # 기본 개발 도구
    build-essential \
    curl \
    wget \
    git \
    # 오디오 처리를 위한 패키지
    ffmpeg \
    libsndfile1 \
    libasound2-dev \
    portaudio19-dev \
    # 이미지 처리를 위한 패키지
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # 시스템 정리
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 파일 복사
COPY requirements.txt .

# pip 업그레이드 및 Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드 복사
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p \
    /app/output/audio \
    /app/output/images \
    /app/output/stories \
    /app/output/conversations \
    /app/output/metadata \
    /app/output/temp \
    /app/chatbot/data/vector_db \
    /app/logs

# 권한 설정
RUN chmod +x /app && \
    chmod -R 755 /app/output && \
    chmod -R 755 /app/chatbot && \
    chmod -R 755 /app/shared

# 포트 노출
EXPOSE 8000

# 환경 변수 파일 설정 (선택적)
# .env 파일이 있다면 복사하되, 없으면 무시
COPY .env* ./

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 애플리케이션 실행 명령어
# 개발 환경에서는 --reload 옵션 추가 가능
CMD ["uvicorn", "chatbot.models.voice_ws.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]



