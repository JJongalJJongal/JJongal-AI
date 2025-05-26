"""
CCB_AI Workflow Integration System

이 모듈은 부기(ChatBot A)와 꼬기(ChatBot B) 간의 통합 워크플로우를 관리합니다.
완전한 동화 생성 파이프라인을 제공하여 이야기 수집부터 멀티미디어 생성까지 
전체 과정을 자동화합니다.

주요 구성 요소:
- WorkflowOrchestrator: 전체 파이프라인 관리 
- StoryDataSchema: 표준화된 이야기 데이터 형식
- IntegrationAPI: 시스템 간 통신 API 
- PipelineManager: 단계별 이야기 생성 관리
- StateManager: 진행 상태 추적 및 관리
- MultimediaCoordinator: 멀티미디어 생성 조정

워크플로우 단계:
1. 이야기 수집 (부기 - ChatBot A)
2. 데이터 검증 및 변환
3. 이야기 생성 (꼬기 - ChatBot B)  
4. 멀티미디어 생성 (이미지, 오디오)
5. 최종 동화 완성 및 배포
"""

from .orchestrator import WorkflowOrchestrator
from .story_schema import StoryDataSchema, StoryElement, StoryMetadata
from .integration_api import IntegrationAPI, APIEndpoints
from .pipeline_manager import PipelineManager, PipelineStage
from .state_manager import StateManager, WorkflowState
from .multimedia_coordinator import MultimediaCoordinator

__all__ = [
    # Core orchestration
    "WorkflowOrchestrator",
    
    # Data schemas
    "StoryDataSchema",
    "StoryElement", 
    "StoryMetadata",
    
    # API integration
    "IntegrationAPI",
    "APIEndpoints",
    
    # Pipeline management
    "PipelineManager",
    "PipelineStage",
    
    # State management
    "StateManager",
    "WorkflowState",
    
    # Multimedia coordination
    "MultimediaCoordinator"
]

# 버전 정보
__version__ = "1.0.0"  # A↔B 통합 워크플로우 시스템 