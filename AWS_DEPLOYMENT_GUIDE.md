# 🌩️ CCB_AI AWS EC2 배포 가이드

## 📋 목차
- [사전 요구사항](#사전-요구사항)
- [EC2 인스턴스 설정](#ec2-인스턴스-설정)
- [환경 설정](#환경-설정)
- [배포 절차](#배포-절차)
- [모니터링](#모니터링)
- [문제 해결](#문제-해결)

## 🔧 사전 요구사항

### 1. EC2 인스턴스 스펙 권장사항
- **타입**: `t3.large` 이상 (8GB RAM, 2 vCPU)
- **스토리지**: 50GB 이상 EBS
- **OS**: Ubuntu 22.04 LTS
- **보안 그룹**: 포트 80, 443, 8000 허용

### 2. 필수 API 키
- **OpenAI API 키** (GPT-4o 사용 권한)
- **ElevenLabs API 키** (음성 생성용)

## 🖥️ EC2 인스턴스 설정

### 1. 인스턴스 생성
```bash
# AWS CLI로 인스턴스 생성 (선택사항)
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --count 1 \
    --instance-type t3.large \
    --key-name your-key-name \
    --security-group-ids sg-xxxxxxxx \
    --subnet-id subnet-xxxxxxxx
```

### 2. 보안 그룹 설정
```bash
# HTTP (80)
aws ec2 authorize-security-group-ingress \
    --group-id sg-xxxxxxxx \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

# HTTPS (443)
aws ec2 authorize-security-group-ingress \
    --group-id sg-xxxxxxxx \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0

# FastAPI (8000) - 개발/테스트용
aws ec2 authorize-security-group-ingress \
    --group-id sg-xxxxxxxx \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0
```

## 🔐 환경 설정

### 1. EC2 인스턴스 접속
```bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

### 2. 시스템 업데이트 및 Docker 설치
```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
sudo systemctl enable docker
sudo systemctl start docker

# Docker Compose 설치
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 3. 프로젝트 클론
```bash
# 프로젝트 클론
git clone <your-repository-url> ccb-ai
cd ccb-ai

# 필요한 디렉토리 생성
mkdir -p logs output/audio output/images output/stories
```

### 4. 환경변수 설정
```bash
# .env 파일 생성
cp .env.example .env  # 또는 아래 내용 직접 작성

# .env 파일 편집
nano .env
```

**.env 파일 내용**:
```env
# API Keys
OPENAI_API_KEY=sk-your-openai-key-here
ELEVENLABS_API_KEY=your-elevenlabs-key-here

# Application
APP_ENV=production
PYTHONPATH=/app

# Security
WS_AUTH_TOKEN=your-secure-websocket-token-here
JWT_SECRET_KEY=your-jwt-secret-key-here

# Database
CHROMA_DB_PATH=/app/chatbot/data/vector_db
REDIS_URL=redis://redis:6379/0

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/server.log
```

## 🚀 배포 절차

### 1. Docker 이미지 빌드
```bash
# 이미지 빌드
docker-compose build

# 빌드 확인
docker images | grep ccb-ai
```

### 2. 서비스 시작
```bash
# 백그라운드에서 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

### 3. 서비스 상태 확인
```bash
# 컨테이너 상태 확인
docker-compose ps

# 헬스체크
curl http://localhost:8000/health
curl http://localhost/health  # Nginx 통해서
```

### 4. 방화벽 설정 (Ubuntu UFW)
```bash
# UFW 활성화
sudo ufw enable

# 필요한 포트 허용
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw allow 8000  # FastAPI (임시)
```

## 📊 모니터링

### 1. 로그 모니터링
```bash
# 실시간 로그
docker-compose logs -f ccb-ai

# 특정 서비스 로그
docker-compose logs -f nginx
docker-compose logs -f redis
```

### 2. 리소스 모니터링
```bash
# 컨테이너 리소스 사용량
docker stats

# 시스템 리소스
htop
df -h
free -h
```

### 3. 애플리케이션 상태 확인
```bash
# 헬스체크 자동화
while true; do
  curl -s http://localhost:8000/health | jq
  sleep 30
done
```

## 🐛 문제 해결

### 1. 메모리 부족 에러
```bash
# 문제: OOMKilled 오류
# 해결: 스왑 파일 생성
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 2. API 키 관련 오류
```bash
# 환경변수 확인
docker-compose exec ccb-ai env | grep API_KEY

# 컨테이너 재시작
docker-compose restart ccb-ai
```

### 3. 포트 충돌
```bash
# 포트 사용 확인
sudo netstat -tulpn | grep :8000

# 프로세스 종료
sudo kill -9 <PID>
```

### 4. Docker 이미지 문제
```bash
# 이미지 및 컨테이너 정리
docker-compose down
docker system prune -a
docker volume prune

# 다시 빌드
docker-compose build --no-cache
docker-compose up -d
```

### 5. Nginx 설정 문제
```bash
# Nginx 설정 테스트
docker-compose exec nginx nginx -t

# Nginx 재로드
docker-compose exec nginx nginx -s reload
```

## 🔄 자동 배포 (GitHub Actions)

`.github/workflows/deploy.yml`:
```yaml
name: Deploy to EC2

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Deploy to EC2
      uses: appleboy/ssh-action@v0.1.5
      with:
        host: ${{ secrets.EC2_HOST }}
        username: ubuntu
        key: ${{ secrets.EC2_SSH_KEY }}
        script: |
          cd ccb-ai
          git pull origin main
          docker-compose down
          docker-compose build
          docker-compose up -d
```

## 📋 성능 최적화

### 1. Docker 이미지 최적화
```dockerfile
# multi-stage build 사용
FROM python:3.12-slim as builder
# 빌드 단계

FROM python:3.12-slim
# 실행 단계
```

### 2. Redis 캐싱 활용
```python
# 설정에서 Redis 캐싱 활성화
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600
```

### 3. 로그 로테이션
```bash
# logrotate 설정
sudo nano /etc/logrotate.d/ccb-ai
```

## 🚨 보안 체크리스트

- [ ] .env 파일에 실제 API 키 설정
- [ ] CORS 설정을 특정 도메인으로 제한
- [ ] JWT 시크릿 키 변경
- [ ] 방화벽 규칙 적용
- [ ] SSL 인증서 설정 (Let's Encrypt)
- [ ] 정기적인 보안 업데이트 