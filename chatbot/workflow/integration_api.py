"""
CCB_AI Integration Manager

부기(ChatBot A)와 꼬기(ChatBot B) 간의 통신 및 외부 시스템과의
통합을 위한 핵심 기능을 제공.

주의: REST API 엔드포인트는 app.py에서 제공됩니다. 
이 모듈은 WorkflowOrchestrator와의 통합 로직만 담당합니다.
"""

import os
# HuggingFace Tokenizers 병렬 처리 경고 해결
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from .story_schema import ChildProfile, AgeGroup

class IntegrationManager:
    """
    CCB_AI 통합 관리자
    
    워크플로우 오케스트레이터와 외부 시스템 간의 통신 로직을 담당합니다.
    REST API는 app.py에서 제공되며, 이 클래스는 순수 비즈니스 로직만 처리합니다.
    """
    
    def __init__(self, orchestrator: Optional[Any] = None):
        """
        통합 관리자 초기화
        
        Args:
            orchestrator: 워크플로우 오케스트레이터 인스턴스
        """
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)
        self.story_states: Dict[str, Dict[str, Any]] = {}  # 스토리 상태 캐시
    
    def set_orchestrator(self, orchestrator: Any):
        """오케스트레이터 설정"""
        self.orchestrator = orchestrator
    
    async def create_story_with_id(
        self,
        child_profile: ChildProfile,
        conversation_data: Optional[Dict[str, Any]],
        story_preferences: Optional[Dict[str, Any]]
    ) -> str:
        """Story ID를 먼저 생성하고 반환"""
        if not self.orchestrator:
            raise RuntimeError("오케스트레이터가 초기화되지 않았습니다")
        
        # 사용자 친화적인 story_id 생성
        friendly_story_id = f"story_{child_profile.name.replace(' ', '_')}_{int(datetime.now().timestamp())}"
        
        # 상태 초기화
        self.story_states[friendly_story_id] = {
            "story_id": friendly_story_id,
            "uuid_story_id": None,  # 실제 UUID가 여기에 저장됨
            "status": "initializing",
            "created_at": datetime.now().isoformat(),
            "child_profile": child_profile.model_dump() if hasattr(child_profile, 'model_dump') else child_profile.__dict__
        }
        
        # 백그라운드에서 이야기 생성 시작
        asyncio.create_task(self._create_story_background(
            friendly_story_id, child_profile, conversation_data, story_preferences
        ))
            
        return friendly_story_id
    
    async def _create_story_background(
        self,
        story_id: str,
        child_profile: ChildProfile,
        conversation_data: Optional[Dict[str, Any]],
        story_preferences: Optional[Dict[str, Any]]
    ):
        """Background에서 이야기 생성"""
        self.logger.info(f"Background 이야기 생성 시작 (ID: {story_id})")
        
        try:
            # 상태 업데이트
            if story_id in self.story_states:
                self.story_states[story_id]["status"] = "in_progress"
                self.logger.info(f"상태 업데이트 완료: {story_id} -> in_progress")
            else:
                self.logger.warning(f"Story state not found for {story_id}")
            
            self.logger.info(f"Orchestrator 상태: {self.orchestrator is not None}")
            
            if self.orchestrator:
                self.logger.info(f"이야기 생성 호출 시작 (ID: {story_id})")
                # 실제 이야기 생성
                story_schema = await self.orchestrator.create_story(
                    child_profile=child_profile,
                    conversation_data=conversation_data,
                    story_preferences=story_preferences
                )
                
                self.logger.info(f"이야기 생성 완료 (ID: {story_id})")
                
                # 실제 UUID 저장
                actual_uuid = story_schema.metadata.story_id if story_schema else None
                self.logger.info(f"실제 UUID: {actual_uuid}")
                
                # 성공 상태 업데이트
                self.story_states[story_id].update({
                    "status": "completed",
                    "uuid_story_id": actual_uuid,  # 실제 UUID 저장
                    "completed_at": datetime.now().isoformat(),
                    "story_data": story_schema.to_dict() if story_schema else None
                })
                
                self.logger.info(f"Background 이야기 생성 성공 (ID: {story_id}, UUID: {actual_uuid})")
            else:
                self.logger.error(f"Orchestrator가 초기화되지 않음 (ID: {story_id})")
                self.story_states[story_id].update({
                    "status": "failed",
                    "error": "Orchestrator not initialized",
                    "failed_at": datetime.now().isoformat()
                })
                
        except Exception as e:
            self.logger.error(f"Background 이야기 생성 실패 (ID: {story_id}): {e}", exc_info=True)
            # 실패 상태 업데이트
            if story_id in self.story_states:
                self.story_states[story_id].update({
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.now().isoformat()
                })
            else:
                self.logger.error(f"스토리 상태 업데이트 실패 - story_id {story_id}")
    
    def determine_age_group(self, age: int) -> AgeGroup:
        """나이에 따른 연령대 결정"""
        if age <= 7:
            return AgeGroup.YOUNG_CHILDREN
        else:
            return AgeGroup.ELEMENTARY

    async def get_story_status(self, story_id: str) -> Optional[Dict[str, Any]]:
        """이야기 상태 조회"""
        # 먼저 로컬 상태에서 확인
        if story_id in self.story_states:
            local_state = self.story_states[story_id]
            
            # 실제 UUID가 있으면 orchestrator에서도 조회
            if local_state.get("uuid_story_id") and self.orchestrator:
                try:
                    orchestrator_status = await self.orchestrator.get_story_status(local_state["uuid_story_id"])
                    if orchestrator_status:
                        # orchestrator 상태와 로컬 상태 병합
                        local_state.update({
                            "current_stage": orchestrator_status.get("current_stage"),
                            "completion_percentage": orchestrator_status.get("completion_percentage"),
                            "errors": orchestrator_status.get("errors", [])
                        })
                except Exception as e:
                    self.logger.warning(f"Orchestrator 상태 조회 실패: {e}")
            
            return local_state
        
        # 로컬 상태에 없으면 orchestrator에서 직접 조회 (혹시 UUID일 수도 있음)
        if self.orchestrator:
            return await self.orchestrator.get_story_status(story_id)
        
        return None
    
    async def cancel_story(self, story_id: str) -> bool:
        """이야기 생성 취소"""
        if story_id in self.story_states:
            self.story_states[story_id]["status"] = "cancelled"
        
        if self.orchestrator:
            await self.orchestrator.cancel_story(story_id)
            return True
        
        return False
    
    async def resume_story(self, story_id: str) -> bool:
        """이야기 재개"""
        if self.orchestrator:
            await self.orchestrator.resume_story(story_id)
            return True
        
        return False

# 오케스트레이터 초기화 함수 (app.py에서 사용)
def init_orchestrator_for_integration():
    """통합을 위한 오케스트레이터 초기화"""
    try:
        from .orchestrator import WorkflowOrchestrator
        orchestrator = WorkflowOrchestrator(
            output_dir="/app/output",
            enable_multimedia=True,
            enable_voice=False
            )
        
        # 챗봇 초기화
        orchestrator.initialize_chatbots()
                
        logging.info("WorkflowOrchestrator 초기화 완료")
        return orchestrator
        
    except Exception as e:
        logging.error(f"WorkflowOrchestrator 초기화 실패: {e}")
        return None

# 통합 관리자 인스턴스
integration_manager = IntegrationManager()

# 로그 레벨 설정
logging.getLogger().setLevel(logging.INFO)