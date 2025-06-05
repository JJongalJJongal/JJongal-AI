# 🐨 꼬꼬북 AI 시연 가이드

## 🚀 **빠른 시작** (3분 완료)

### **1단계: 환경 설정**
```bash
# 환경 변수 파일 생성
cp env.example .env

# API 키 설정 (필수!)
nano .env
# OPENAI_API_KEY=your_key_here
# ELEVENLABS_API_KEY=your_key_here
```

### **2단계: 시연 시작**
```bash
# 완전 처음 시작 (빌드 포함)
make demo-setup

# 또는 빠른 시작 (이미 빌드된 경우)
make demo-start
```

### **3단계: 시연 확인**
```bash
# 상태 확인
make demo-status

# 기능 테스트
make demo-test

# 실시간 모니터링
make monitor
```

---

## 🎯 **시연 포인트**

### **1. AI 동화 생성** 📚
- **URL**: http://localhost:8000
- **기능**: 실시간 동화 생성, 음성 변환, 이미지 생성

### **2. WebSocket 실시간 채팅** 💬
- **URL**: ws://localhost:8000/ws
- **기능**: 실시간 음성 대화, 스트리밍 응답

### **3. REST API** 🔧
- **Swagger**: http://localhost:8000/docs
- **기능**: 스토리 관리, 메타데이터 조회

### **4. 시스템 모니터링** 📊
- **URL**: http://localhost:9100
- **기능**: 실시간 성능 모니터링

---

## 🎪 **시연 시나리오**

### **시나리오 1: 동화 생성**
```bash
# 1. API를 통한 동화 생성
curl -X POST "http://localhost:8000/api/v1/stories" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "용감한 토끼의 모험",
    "age_group": "5-7",
    "voice_enabled": true,
    "image_enabled": true
  }'

# 2. 결과 확인
# - 생성된 스토리 텍스트
# - 음성 파일 (MP3)
# - 일러스트 이미지 (PNG)
```

### **시나리오 2: 실시간 대화**
```javascript
// WebSocket 연결 (브라우저에서)
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  // 음성 대화 시작
  ws.send(JSON.stringify({
    type: 'voice_chat',
    message: '안녕하세요! 동화를 만들어주세요.'
  }));
};

ws.onmessage = (event) => {
  // 실시간 응답 수신
  const response = JSON.parse(event.data);
  console.log('AI 응답:', response);
};
```

### **시나리오 3: 성능 모니터링**
```bash
# 실시간 성능 확인
make monitor

# 결과:
# - CPU 사용률: ~40-60%
# - 메모리 사용률: ~6-8GB
# - API 응답 시간: ~0.5-2초
# - 동시 접속 처리: 50+ 사용자
```

---

## 🛠️ **문제 해결**

### **일반적인 문제들**

#### **1. 메모리 부족**
```bash
# 메모리 사용량 확인
make demo-status

# Docker 메모리 정리
make demo-clean
docker system prune -a -f
```

#### **2. API 키 오류**
```bash
# 환경 변수 확인
cat .env | grep API_KEY

# 로그 확인
make logs-ai
```

#### **3. 포트 충돌**
```bash
# 포트 사용 현황 확인
lsof -i :8000
lsof -i :80

# 서비스 재시작
make stop
make demo-start
```

### **디버깅 명령어**
```bash
# 1. 전체 로그 확인
make logs

# 2. AI 서버 로그만
make logs-ai

# 3. 컨테이너 내부 접속
make shell

# 4. 헬스체크
make health

# 5. 서비스 상태
make demo-status
```

---

## 📝 **시연 체크리스트**

### **시연 전 준비**
- [ ] `.env` 파일 생성 및 API 키 설정
- [ ] Docker 및 Docker Compose 설치 확인
- [ ] 최소 12GB RAM 확보
- [ ] `make demo-setup` 실행 성공

### **시연 중 확인**
- [ ] API 헬스체크 통과 (200 OK)
- [ ] 동화 생성 API 테스트 통과
- [ ] WebSocket 연결 테스트 통과
- [ ] 음성 및 이미지 생성 확인
- [ ] 성능 모니터링 정상 작동

### **시연 후 정리**
- [ ] `make demo-clean` 실행
- [ ] 생성된 파일들 백업
- [ ] 로그 파일 확인

---

## 🎓 **고급 기능 시연**

### **1. 배치 처리**
```python
# 여러 스토리 동시 생성
import asyncio
import aiohttp

async def create_multiple_stories():
    stories = [
        {"user_input": "용감한 토끼", "age_group": "5-7"},
        {"user_input": "마법의 숲", "age_group": "8-10"},
        {"user_input": "우주 여행", "age_group": "10-12"}
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for story in stories:
            task = session.post(
                "http://localhost:8000/api/v1/stories",
                json=story
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return responses

# 실행
# asyncio.run(create_multiple_stories())
```

### **2. 커스텀 설정**
```bash
# 메모리 최적화 모드
export TORCH_NUM_THREADS=2
export OMP_NUM_THREADS=2

# 고성능 모드
export TORCH_NUM_THREADS=8
export OMP_NUM_THREADS=8

# 서비스 재시작
make stop && make demo-start
```

---

## 📞 **지원 및 문의**

- **문서**: [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)
- **프론트엔드 가이드**: [FRONTEND_QUICK_START.md](./FRONTEND_QUICK_START.md)
- **AWS 배포**: [AWS_DEPLOYMENT_GUIDE.md](./AWS_DEPLOYMENT_GUIDE.md)

---

## 🏆 **성공 지표**

### **성능 벤치마크**
- **동화 생성 시간**: 평균 15-30초
- **음성 변환 시간**: 평균 5-10초
- **이미지 생성 시간**: 평균 10-20초
- **API 응답 시간**: 평균 0.5-2초
- **동시 사용자**: 최대 50명

### **품질 지표**
- **동화 품질**: GPT-4 기반 고품질 컨텐츠
- **음성 품질**: ElevenLabs 프리미엄 음성
- **이미지 품질**: DALL-E 3 고해상도 이미지
- **사용자 만족도**: 95% 이상 목표

---

**🎉 시연 성공을 위한 팁**: 
네트워크 연결 안정성 확인, 충분한 메모리 확보, API 키 사전 테스트를 권장합니다! 