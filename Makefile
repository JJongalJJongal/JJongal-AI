# 꼬꼬북 동화 생성 프로젝트 Makefile

# 기본 변수 설정
DOCKER_IMAGE = ccb-ai
DOCKER_TAG = latest
CONTAINER_NAME = ccb-ai-app

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

# 로그 확인
.PHONY: logs
logs:
	docker-compose logs -f ccb-ai

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
	curl -f http://localhost:8000/health

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
	@echo "  build       - Docker 이미지 빌드"
	@echo "  run         - 프로덕션 환경으로 실행"
	@echo "  dev         - 개발 환경으로 실행"
	@echo "  stop        - 서비스 정지"
	@echo "  clean       - 컨테이너 및 이미지 정리"
	@echo "  logs        - 로그 확인"
	@echo "  shell       - 컨테이너 내부 접속"
	@echo "  test        - 테스트 실행"
	@echo "  health      - 헬스체크"
	@echo "  deploy      - 프로덕션 배포"
	@echo "  dev-tools   - 개발 도구 실행"
	@echo "  info        - 이미지 및 컨테이너 정보"
	@echo "  help        - 도움말"
	@echo ""
	@echo "사용 예시:"
	@echo "  make build && make dev  # 빌드 후 개발 환경 실행"
	@echo "  make deploy            # 프로덕션 배포"
	@echo "  make logs              # 실시간 로그 확인"

# 기본 타겟
.DEFAULT_GOAL := help 