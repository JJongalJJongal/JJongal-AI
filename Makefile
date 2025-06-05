# 꼬꼬북 동화 생성 프로젝트 Makefile

# 기본 변수 설정
DOCKER_IMAGE = ccb-ai
DOCKER_TAG = latest
CONTAINER_NAME = ccb-ai-app

# =============================================================================
# 기본 Docker 명령어
# =============================================================================

# Docker 빌드
.PHONY: build
build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

# Docker 실행 (프로덕션)
.PHONY: run
run:
	docker-compose up -d

# Docker 실행 (개발 환경)
.PHONY: dev
dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Docker 정지
.PHONY: stop
stop:
	docker-compose down

# Docker 완전 정리 (볼륨 포함)
.PHONY: clean
clean:
	docker-compose down -v
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) || true

# =============================================================================
# 시연 전용 명령어
# =============================================================================

# 시연 준비 (환경 변수 체크 + 빌드 + 실행)
.PHONY: demo-setup
demo-setup:
	@echo "🚀 꼬꼬북 AI 시연 준비 중..."
	@if [ ! -f .env ]; then \
		echo ".env 파일이 없습니다. env.example을 참고하여 생성하세요."; \
		exit 1; \
	fi
	@echo "환경 변수 파일 확인됨"
	@echo "Docker 이미지 빌드 중..."
	$(MAKE) build
	@echo "서비스 시작 중..."
	$(MAKE) run
	@echo "서비스 로딩 대기 중 (90초)..."
	@sleep 90
	@echo "헬스체크 실행 중..."
	$(MAKE) health
	@echo "시연 준비 완료!"

# 시연용 빠른 시작
.PHONY: demo-start
demo-start:
	@echo "꼬꼬북 AI 빠른 시작..."
	docker-compose up -d
	@echo "30초 대기 중..."
	@sleep 30
	@echo "시연 시작 가능!"
	@echo "Frontend: http://localhost:80"
	@echo "API: http://localhost:8000"
	@echo "모니터링: http://localhost:9100"

# 시연 상태 확인
.PHONY: demo-status
demo-status:
	@echo "꼬꼬북 AI 서비스 상태:"
	@echo ""
	@echo "컨테이너 상태:"
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(ccb-|NAMES)"
	@echo ""
	@echo "헬스체크:"
	@curl -s http://localhost:8000/api/v1/health | jq '.' || echo "API 서버 연결 실패"
	@echo ""
	@echo "메모리 사용량:"
	@docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}" | grep -E "(ccb-|NAME)"

# 시연용 테스트 실행
.PHONY: demo-test
demo-test:
	@echo "꼬꼬북 AI 기능 테스트 중..."
	@echo ""
	@echo "1️⃣ API 헬스체크:"
	@curl -s http://localhost:8000/api/v1/health || echo "실패"
	@echo ""
	@echo "2️⃣ WebSocket 연결 테스트:"
	@curl -s "http://localhost:8000/ws/test" || echo "WebSocket 테스트 실패"
	@echo ""
	@echo "3️⃣ 스토리 생성 API 테스트:"
	@curl -s -X POST "http://localhost:8000/api/v1/stories" \
		-H "Content-Type: application/json" \
		-d '{"user_input": "용감한 토끼", "age_group": "5-7"}' || echo "스토리 API 테스트 실패"

# 시연 정리
.PHONY: demo-clean
demo-clean:
	@echo "시연 환경 정리 중..."
	$(MAKE) stop
	@echo "임시 파일 정리 중..."
	@docker system prune -f
	@echo "정리 완료"

# =============================================================================
# 디버깅 및 모니터링
# =============================================================================

# 로그 확인
.PHONY: logs
logs:
	docker-compose logs -f ccb-ai

# 실시간 로그 (AI 서버만)
.PHONY: logs-ai
logs-ai:
	docker logs -f ccb-ai-app

# 컨테이너 내부 접속
.PHONY: shell
shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

# 테스트 실행
.PHONY: test
test:
	docker run --rm -v $(PWD):/app $(DOCKER_IMAGE):$(DOCKER_TAG) python -m pytest chatbot/tests/ -v

# 헬스체크
.PHONY: health
health:
	curl -f http://localhost:8000/api/v1/health

# 성능 모니터링
.PHONY: monitor
monitor:
	@echo "실시간 성능 모니터링:"
	@echo "Ctrl+C로 종료"
	@while true; do \
		clear; \
		echo "=== 꼬꼬북 AI 성능 모니터링 ==="; \
		echo "시간: $$(date)"; \
		echo ""; \
		docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}" | grep -E "(ccb-|NAME)"; \
		echo ""; \
		echo "API 응답 시간:"; \
		curl -w "응답시간: %{time_total}s\n" -s -o /dev/null http://localhost:8000/api/v1/health || echo "❌ API 연결 실패"; \
		sleep 5; \
	done

# =============================================================================
# 프로덕션 배포
# =============================================================================

# 프로덕션 배포
.PHONY: deploy
deploy: build
	docker-compose -f docker-compose.yml up -d

# 개발 도구 실행 (Jupyter 포함)
.PHONY: dev-tools
dev-tools:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml --profile dev-tools up -d

# 도커 이미지 정보 확인
.PHONY: info
info:
	docker images | grep $(DOCKER_IMAGE)
	docker ps | grep $(CONTAINER_NAME) || echo "Container not running"

# 도움말
.PHONY: help
help:
	@echo "꼬꼬북 동화 생성 프로젝트 Makefile 명령어:"
	@echo ""
	@echo "기본 명령어:"
	@echo "  build       - Docker 이미지 빌드"
	@echo "  run         - 프로덕션 환경으로 실행"
	@echo "  dev         - 개발 환경으로 실행"
	@echo "  stop        - 서비스 정지"
	@echo "  clean       - 컨테이너 및 이미지 정리"
	@echo ""
	@echo "시연 전용 명령어:"
	@echo "  demo-setup  - 시연 완전 준비 (환경체크+빌드+실행)"
	@echo "  demo-start  - 시연 빠른 시작"
	@echo "  demo-status - 시연 상태 확인"
	@echo "  demo-test   - 시연용 기능 테스트"
	@echo "  demo-clean  - 시연 환경 정리"
	@echo ""
	@echo "디버깅 명령어:"
	@echo "  logs        - 로그 확인"
	@echo "  logs-ai     - AI 서버 로그만"
	@echo "  shell       - 컨테이너 내부 접속"
	@echo "  test        - 테스트 실행"
	@echo "  health      - 헬스체크"
	@echo "  monitor     - 실시간 성능 모니터링"
	@echo ""
	@echo "배포 명령어:"
	@echo "  deploy      - 프로덕션 배포"
	@echo "  dev-tools   - 개발 도구 실행"
	@echo "  info        - 이미지 및 컨테이너 정보"
	@echo ""
	@echo "시연 사용 예시:"
	@echo "  make demo-setup     # 완전 처음 시작"
	@echo "  make demo-start     # 빠른 시작"
	@echo "  make demo-status    # 상태 확인"
	@echo "  make demo-test      # 기능 테스트"
	@echo "  make monitor        # 성능 모니터링"

# 기본 타겟
.DEFAULT_GOAL := help 