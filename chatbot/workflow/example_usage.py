"""
CCB_AI A↔B Integration Workflow Example

이 파일은 부기(ChatBot A)와 꼬기(ChatBot B) 간의 완전한 통합 워크플로우를
사용하는 방법을 보여주는 예제.
"""

import asyncio
import logging
from typing import Dict, Any

from .orchestrator import WorkflowOrchestrator
from .story_schema import ChildProfile, AgeGroup, StoryElement, ElementType
from .integration_api import IntegrationManager
from .state_manager import StateManager
from .pipeline_manager import PipelineManager, PipelineConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def example_complete_workflow():
    """완전한 워크플로우 예제"""
    print("=== CCB_AI A↔B 통합 워크플로우 예제 ===\n")
    
    # 1. 워크플로우 오케스트레이터 초기화
    print("1. 워크플로우 오케스트레이터 초기화...")
    orchestrator = WorkflowOrchestrator(
        output_dir="output",
        enable_multimedia=True,
        enable_voice=False
    )
    
    # 2. 아이 프로필 생성
    print("2. 아이 프로필 생성...")
    child_profile = ChildProfile(
        name="지민이",
        age=6,
        age_group=AgeGroup.YOUNG_CHILDREN,
        interests=["동물", "모험", "친구"],
        language_level="basic",
        special_needs=[]
    )
    print(f"   - 이름: {child_profile.name}")
    print(f"   - 나이: {child_profile.age}세")
    print(f"   - 관심사: {', '.join(child_profile.interests)}")
    
    # 3. 대화 데이터 시뮬레이션 (부기에서 수집된 데이터)
    print("\n3. 대화 데이터 준비...")
    conversation_data = {
        "messages": [
            {"role": "user", "content": "토끼가 나오는 이야기를 듣고 싶어요"},
            {"role": "assistant", "content": "어떤 토끼 이야기를 좋아하나요?"},
            {"role": "user", "content": "숲에서 친구들과 모험하는 토끼요"},
            {"role": "assistant", "content": "멋진 아이디어네요! 토끼가 어떤 친구들을 만날까요?"},
            {"role": "user", "content": "다람쥐랑 새랑 곰도 만났으면 좋겠어요"}
        ],
        "duration": 15.5,
        "topics": ["토끼", "숲", "모험", "친구", "동물"],
        "tone": "excited",
        "engagement": 0.9,
        "summary": "아이가 토끼가 숲에서 다양한 동물 친구들과 모험하는 이야기를 원함"
    }
    print(f"   - 대화 메시지: {len(conversation_data['messages'])}개")
    print(f"   - 주요 주제: {', '.join(conversation_data['topics'])}")
    
    # 4. 이야기 선호도 설정
    print("\n4. 이야기 선호도 설정...")
    story_preferences = {
        "genre": "adventure",
        "mood": "happy",
        "lesson": "friendship",
        "length": "medium",
        "include_dialogue": True
    }
    print(f"   - 장르: {story_preferences['genre']}")
    print(f"   - 분위기: {story_preferences['mood']}")
    print(f"   - 교훈: {story_preferences['lesson']}")
    
    # 5. 완전한 이야기 생성 워크플로우 실행
    print("\n5. 이야기 생성 워크플로우 실행...")
    try:
        story_schema = await orchestrator.create_story(
            child_profile=child_profile,
            conversation_data=conversation_data,
            story_preferences=story_preferences
        )
        
        print(f"   ✅ 이야기 생성 완료!")
        print(f"   - 이야기 ID: {story_schema.metadata.story_id}")
        print(f"   - 현재 단계: {story_schema.current_stage.value}")
        print(f"   - 완성도: {story_schema.get_completion_percentage():.1f}%")
        
        # 6. 생성된 이야기 내용 확인
        if story_schema.generated_story:
            print(f"\n6. 생성된 이야기 내용:")
            print(f"   - 제목: {story_schema.metadata.title or '토끼의 모험'}")
            print(f"   - 단어 수: {story_schema.generated_story.word_count}")
            print(f"   - 내용 미리보기:")
            content_preview = story_schema.generated_story.content[:200] + "..."
            print(f"     {content_preview}")
        
        # 7. 멀티미디어 자산 확인
        if story_schema.multimedia_assets:
            print(f"\n7. 멀티미디어 자산:")
            print(f"   - 이미지: {len(story_schema.multimedia_assets.images)}개")
            print(f"   - 오디오: {len(story_schema.multimedia_assets.audio_files)}개")
        
        return story_schema
        
    except Exception as e:
        print(f"   ❌ 이야기 생성 실패: {e}")
        return None

async def example_api_usage():
    """API 사용 예제"""
    print("\n=== API 사용 예제 ===\n")
    
    # 1. 통합 API 초기화
    print("1. 통합 API 초기화...")
    orchestrator = WorkflowOrchestrator(output_dir="output")
    integration_api = IntegrationManager(orchestrator)
    
    # 2. 아이 프로필 생성
    child_profile = ChildProfile(
        name="수연이",
        age=5,
        age_group=AgeGroup.YOUNG_CHILDREN,
        interests=["공주", "마법", "꽃"],
        language_level="basic"
    )
    
    # 3. API를 통한 직접 이야기 생성
    print("2. API를 통한 이야기 생성...")
    result = await integration_api.create_story_direct(
        child_profile=child_profile,
        story_preferences={"genre": "fantasy", "mood": "magical"}
    )
    
    if result["success"]:
        story_id = result["story_id"]
        print(f"   ✅ API 이야기 생성 성공!")
        print(f"   - 이야기 ID: {story_id}")
        
        # 4. 상태 조회
        print("\n3. 이야기 상태 조회...")
        status_result = await integration_api.get_status_direct(story_id)
        if status_result["success"]:
            status = status_result["status"]
            print(f"   - 현재 단계: {status['current_stage']}")
            print(f"   - 진행률: {status['progress_percentage']:.1f}%")
        
    else:
        print(f"   ❌ API 이야기 생성 실패: {result['error']}")

async def example_state_management():
    """상태 관리 예제"""
    print("\n=== 상태 관리 예제 ===\n")
    
    # 1. 상태 관리자 초기화
    print("1. 상태 관리자 초기화...")
    state_manager = StateManager("output")
    
    # 2. 모든 이야기 목록 조회
    print("2. 저장된 이야기 목록 조회...")
    all_stories = await state_manager.list_all_stories()
    print(f"   - 총 이야기 수: {len(all_stories)}")
    
    for story in all_stories[:3]:  # 최대 3개만 표시
        print(f"   - {story['story_id'][:8]}... ({story['current_stage']})")
    
    # 3. 워크플로우 통계
    print("\n3. 워크플로우 통계:")
    stats = await state_manager.get_workflow_statistics()
    print(f"   - 총 이야기: {stats.get('total_stories', 0)}")
    print(f"   - 활성 이야기: {stats.get('active_stories', 0)}")
    print(f"   - 성공률: {stats.get('success_rate', 0):.1f}%")

async def example_pipeline_management():
    """파이프라인 관리 예제"""
    print("\n=== 파이프라인 관리 예제 ===\n")
    
    # 1. 파이프라인 설정
    print("1. 파이프라인 설정...")
    config = PipelineConfig(
        enable_multimedia=True,
        enable_validation=True,
        max_retries=2,
        timeout_seconds=60
    )
    
    pipeline_manager = PipelineManager(config)
    
    # 2. 실행 통계 조회
    print("2. 파이프라인 실행 통계:")
    stats = pipeline_manager.get_execution_statistics()
    print(f"   - 총 실행: {stats['total_executions']}")
    print(f"   - 성공: {stats['successful_executions']}")
    print(f"   - 실패: {stats['failed_executions']}")

def example_story_schema():
    """스토리 스키마 사용 예제"""
    print("\n=== 스토리 스키마 예제 ===\n")
    
    # 1. 스토리 스키마 생성
    print("1. 스토리 스키마 생성...")
    from .story_schema import StoryDataSchema, StoryElement
    
    story_schema = StoryDataSchema()
    
    # 2. 이야기 요소 추가
    print("2. 이야기 요소 추가...")
    elements = [
        StoryElement(ElementType.CHARACTER, "용감한 토끼 토토", ["토끼", "용감한", "주인공"]),
        StoryElement(ElementType.SETTING, "신비한 마법의 숲", ["숲", "마법", "신비한"]),
        StoryElement(ElementType.PROBLEM, "길을 잃어버린 친구들을 찾아야 함", ["길잃음", "친구", "찾기"])
    ]
    
    for element in elements:
        story_schema.add_story_element(element)
        print(f"   - {element.element_type.value}: {element.content}")
    
    # 3. 생성 준비 상태 확인
    print(f"\n3. 이야기 생성 준비: {'✅' if story_schema.is_ready_for_generation() else '❌'}")
    
    # 4. JSON 변환
    print("4. JSON 형태로 변환...")
    json_data = story_schema.to_json()
    print(f"   - JSON 크기: {len(json_data)} 문자")

async def main():
    """메인 예제 실행"""
    print("CCB_AI A↔B 통합 워크플로우 시스템 예제\n")
    print("=" * 50)
    
    try:
        # 1. 완전한 워크플로우 예제
        story_result = await example_complete_workflow()
        
        # 2. API 사용 예제
        await example_api_usage()
        
        # 3. 상태 관리 예제
        await example_state_management()
        
        # 4. 파이프라인 관리 예제
        await example_pipeline_management()
        
        # 5. 스토리 스키마 예제
        example_story_schema()
        
        print("\n" + "=" * 50)
        print("모든 예제 실행 완료! 🎉")
        
        if story_result:
            print(f"\n생성된 이야기 ID: {story_result.metadata.story_id}")
            print("output/ 디렉토리에서 결과를 확인할 수 있습니다.")
        
    except Exception as e:
        print(f"\n❌ 예제 실행 중 오류 발생: {e}")
        logger.exception("예제 실행 오류")

if __name__ == "__main__":
    # 예제 실행
    asyncio.run(main()) 