# CCB_AI A↔B Integration Workflow System

부기(ChatBot A)와 꼬기(ChatBot B) 간의 완전한 통합 워크플로우 시스템입니다.

## 🎯 개요

이 시스템은 이야기 수집부터 최종 멀티미디어 동화 생성까지의 전체 파이프라인을 자동화합니다:

1. **부기(ChatBot A)**: 아이와의 대화를 통해 이야기 요소 수집
2. **데이터 검증**: 수집된 요소들의 유효성 검증
3. **꼬기(ChatBot B)**: 완전한 동화 이야기 생성
4. **멀티미디어 생성**: 이미지와 오디오 자산 생성
5. **최종 완성**: 통합된 동화 패키지 완성

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ChatBot A     │    │   Workflow      │    │   ChatBot B     │
│   (부기)        │───▶│  Orchestrator   │───▶│   (꼬기)        │
│                 │    │                 │    │                 │
│ Story Elements  │    │ Pipeline Mgmt   │    │ Story Generation│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Integration API │    │ State Manager   │    │ Multimedia      │
│                 │    │                 │    │ Coordinator     │
│ RESTful APIs    │    │ Persistence     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📦 주요 구성 요소

### 1. WorkflowOrchestrator
- **역할**: 전체 파이프라인 관리 및 조정
- **기능**: 
  - 단계별 이야기 생성 프로세스 관리
  - 챗봇 간 데이터 전달
  - 오류 처리 및 복구
  - 이벤트 기반 알림

### 2. StoryDataSchema
- **역할**: 표준화된 이야기 데이터 형식
- **구성요소**:
  - `ChildProfile`: 아이 프로필 정보
  - `StoryElement`: 개별 이야기 요소
  - `ConversationSummary`: 대화 요약
  - `GeneratedStory`: 생성된 이야기
  - `MultimediaAssets`: 멀티미디어 자산

### 3. StateManager
- **역할**: 워크플로우 상태 관리 및 지속성
- **기능**:
  - 이야기 생성 진행 상태 추적
  - 중단/재개 기능
  - 통계 및 분석
  - 자동 정리

### 4. IntegrationAPI
- **역할**: RESTful API 제공
- **엔드포인트**:
  - `POST /api/v1/stories`: 새 이야기 생성
  - `GET /api/v1/stories/{id}`: 이야기 조회
  - `GET /api/v1/stories/{id}/status`: 상태 조회
  - `POST /api/v1/stories/{id}/cancel`: 생성 취소

### 5. PipelineManager
- **역할**: 단계별 파이프라인 실행 관리
- **기능**:
  - 단계별 실행 및 검증
  - 재시도 로직
  - 병렬 처리 지원
  - 실행 통계

### 6. MultimediaCoordinator
- **역할**: 멀티미디어 자산 생성
- **기능**:
  - 이미지 생성 (DALL-E 연동)
  - 오디오 생성 (TTS 연동)
  - 자산 관리 및 최적화

## 🚀 사용 방법

### 기본 사용법

```python
from chatbot.workflow import WorkflowOrchestrator, ChildProfile, AgeGroup

# 1. 오케스트레이터 초기화
orchestrator = WorkflowOrchestrator(
    output_dir="output",
    enable_multimedia=True
)

# 2. 아이 프로필 생성
child_profile = ChildProfile(
    name="지민이",
    age=6,
    age_group=AgeGroup.YOUNG_CHILDREN,
    interests=["동물", "모험", "친구"]
)

# 3. 대화 데이터 준비
conversation_data = {
    "messages": [...],
    "topics": ["토끼", "숲", "모험"],
    "summary": "토끼의 모험 이야기 요청"
}

# 4. 이야기 생성
story_schema = await orchestrator.create_story(
    child_profile=child_profile,
    conversation_data=conversation_data
)
```

### API 사용법

```python
from chatbot.workflow import IntegrationAPI

# API 초기화
api = IntegrationAPI(orchestrator)

# 직접 호출
result = await api.create_story_direct(
    child_profile=child_profile,
    story_preferences={"genre": "adventure"}
)

# FastAPI 서버 실행 (선택사항)
if api.is_api_available():
    app = api.get_app()
    # uvicorn으로 실행
```

## 📊 데이터 흐름

### 1. 이야기 요소 수집 단계
```
ChatBot A → StoryElement[] → StoryDataSchema
```

### 2. 데이터 검증 단계
```
StoryDataSchema → Validation → Validated Data
```

### 3. 이야기 생성 단계
```
Validated Data → ChatBot B → GeneratedStory
```

### 4. 멀티미디어 생성 단계
```
GeneratedStory → MultimediaCoordinator → MultimediaAssets
```

### 5. 최종 완성 단계
```
Complete StoryDataSchema → File Output → Finished Story
```

## 🔧 설정 옵션

### WorkflowOrchestrator 설정
```python
orchestrator = WorkflowOrchestrator(
    output_dir="output",           # 출력 디렉토리
    enable_multimedia=True,        # 멀티미디어 생성 활성화
    enable_voice=False,           # 음성 처리 활성화
    config={
        "max_retries": 3,
        "timeout_seconds": 300
    }
)
```

### PipelineConfig 설정
```python
from chatbot.workflow import PipelineConfig

config = PipelineConfig(
    enable_multimedia=True,        # 멀티미디어 단계 활성화
    enable_validation=True,        # 검증 단계 활성화
    max_retries=3,                # 최대 재시도 횟수
    timeout_seconds=300,          # 단계별 타임아웃
    parallel_execution=False      # 병렬 실행 여부
)
```

## 📁 출력 구조

```
output/
├── stories/
│   └── {story_id}/
│       ├── story_data.json      # 완전한 스토리 데이터
│       ├── story.txt            # 텍스트 이야기
│       └── metadata.json        # 메타데이터
├── images/
│   └── {story_id}/
│       ├── scene_0.png          # 장면별 이미지
│       ├── scene_1.png
│       └── ...
├── audio/
│   └── {story_id}/
│       ├── full_story.mp3       # 전체 이야기 오디오
│       ├── chapter_0.mp3        # 챕터별 오디오
│       └── ...
├── workflow_states/
│   └── {story_id}.json          # 워크플로우 상태
└── metadata/
    ├── {story_id}_metadata.json # 이야기 메타데이터
    └── {story_id}_multimedia.json # 멀티미디어 메타데이터
```

## 🔍 모니터링 및 디버깅

### 상태 조회
```python
# 이야기 상태 확인
status = await orchestrator.get_story_status(story_id)
print(f"현재 단계: {status['current_stage']}")
print(f"진행률: {status['progress_percentage']}%")

# 워크플로우 통계
stats = await state_manager.get_workflow_statistics()
print(f"성공률: {stats['success_rate']}%")
```

### 로깅
```python
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('chatbot.workflow')

# 상세 로그 확인
logger.setLevel(logging.DEBUG)

# 각 모듈은 자체 로거를 사용합니다:
# - MultimediaCoordinator: 라이브러리 가용성을 로깅
# - 기타 모듈: 표준 로깅 패턴 적용
```

## 🚨 오류 처리

### 일반적인 오류 유형
1. **수집 오류**: 이야기 요소 부족
2. **검증 오류**: 필수 요소 누락
3. **생성 오류**: ChatBot B 응답 실패
4. **멀티미디어 오류**: 외부 API 연동 실패

### 오류 복구
```python
# 중단된 이야기 재개
story_schema = await orchestrator.resume_story(story_id)

# 특정 단계부터 재시작
from chatbot.workflow import PipelineStage
results = await pipeline_manager.execute_pipeline(
    story_schema,
    start_stage=PipelineStage.GENERATION
)
```

## 🔌 확장성

### 새로운 단계 추가
```python
from chatbot.workflow import PipelineStage

# 커스텀 단계 핸들러
async def custom_stage_handler(story_schema):
    # 커스텀 로직 구현
    return {"result": "success"}

# 핸들러 등록
pipeline_manager.register_stage_handler(
    PipelineStage.CUSTOM,
    custom_stage_handler
)
```

### 이벤트 핸들러
```python
# 단계 완료 이벤트 핸들러
def on_stage_complete(story_schema, result):
    print(f"단계 완료: {result.stage.value}")

# 이벤트 핸들러 등록
orchestrator.add_event_handler("stage_changed", on_stage_complete)
```

## 📋 요구사항

### 필수 의존성
- Python 3.8+
- asyncio
- dataclasses
- typing

### 선택적 의존성
- FastAPI (API 기능)
- OpenAI (이미지 생성)
- ElevenLabs (음성 생성)
- Pydantic (데이터 검증)

## 🧪 테스트

### 예제 실행
```bash
# 예제 스크립트 실행
python -m chatbot.workflow.example_usage

# 특정 예제만 실행
python -c "
from chatbot.workflow.example_usage import example_complete_workflow
import asyncio
asyncio.run(example_complete_workflow())
"
```

### 단위 테스트
```bash
# 테스트 실행 (pytest 필요)
pytest chatbot/workflow/tests/
```

## 🔄 버전 히스토리

### v1.0.0 (현재)
- ✅ 기본 A↔B 통합 워크플로우
- ✅ 표준화된 데이터 스키마
- ✅ 상태 관리 및 지속성
- ✅ RESTful API 지원
- ✅ 멀티미디어 생성 기능
- ✅ 파이프라인 관리 시스템

### 향후 계획
- 🔄 실시간 음성 처리 통합
- 🔄 고급 멀티미디어 편집
- 🔄 클라우드 배포 지원
- 🔄 성능 최적화
- 🔄 다국어 지원

## 🤝 기여 방법

1. 이슈 리포트: 버그나 개선사항 제안
2. 기능 요청: 새로운 기능 아이디어 제안
3. 코드 기여: Pull Request 제출
4. 문서 개선: 문서 업데이트 및 예제 추가

## 📞 지원

- 문서: 이 README 파일
- 예제: `example_usage.py` 참조
- 이슈: GitHub Issues 활용
- 토론: GitHub Discussions 활용

---

**CCB_AI A↔B Integration Workflow System** - 완전한 동화 생성 파이프라인 🎨📚🎵 