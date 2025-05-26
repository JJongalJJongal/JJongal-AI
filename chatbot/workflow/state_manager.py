"""
CCB_AI State Manager

워크플로우 상태 관리 및 지속성을 담당하는 모듈입니다.
이야기 생성 과정의 진행 상태를 추적하고 저장/복원 기능을 제공합니다.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from .story_schema import StoryDataSchema, StoryStage

class WorkflowState(Enum):
    """워크플로우 상태"""
    IDLE = "idle"                    # 대기 상태
    COLLECTING = "collecting"        # 이야기 요소 수집 중
    VALIDATING = "validating"        # 데이터 검증 중
    GENERATING = "generating"        # 이야기 생성 중
    MULTIMEDIA = "multimedia"        # 멀티미디어 생성 중
    FINALIZING = "finalizing"        # 최종 완성 중
    COMPLETED = "completed"          # 완료
    ERROR = "error"                  # 오류 상태
    CANCELLED = "cancelled"          # 취소됨

@dataclass
class StateSnapshot:
    """상태 스냅샷"""
    story_id: str
    workflow_state: WorkflowState
    story_stage: StoryStage
    timestamp: datetime
    progress_percentage: float
    error_count: int
    last_activity: str
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "story_id": self.story_id,
            "workflow_state": self.workflow_state.value,
            "story_stage": self.story_stage.value,
            "timestamp": self.timestamp.isoformat(),
            "progress_percentage": self.progress_percentage,
            "error_count": self.error_count,
            "last_activity": self.last_activity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateSnapshot':
        """딕셔너리에서 생성"""
        return cls(
            story_id=data["story_id"],
            workflow_state=WorkflowState(data["workflow_state"]),
            story_stage=StoryStage(data["story_stage"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            progress_percentage=data["progress_percentage"],
            error_count=data["error_count"],
            last_activity=data["last_activity"]
        )

class StateManager:
    """
    워크플로우 상태 관리자
    
    이야기 생성 과정의 상태를 추적하고 지속성을 제공합니다.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        상태 관리자 초기화
        
        Args:
            output_dir: 출력 디렉토리 경로
        """
        self.output_dir = output_dir
        self.state_dir = os.path.join(output_dir, "workflow_states")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        
        # 디렉토리 생성
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 활성 상태 추적
        self.active_states: Dict[str, StateSnapshot] = {}
        
        # 상태 변경 콜백
        self.state_change_callbacks: List[callable] = []
        
        self.logger.info("상태 관리자 초기화 완료")
    
    def add_state_change_callback(self, callback: callable):
        """상태 변경 콜백 추가"""
        self.state_change_callbacks.append(callback)
    
    def _notify_state_change(self, story_id: str, old_state: Optional[StateSnapshot], new_state: StateSnapshot):
        """상태 변경 알림"""
        for callback in self.state_change_callbacks:
            try:
                callback(story_id, old_state, new_state)
            except Exception as e:
                self.logger.error(f"상태 변경 콜백 실행 실패: {e}")
    
    async def save_story_state(self, story_schema: StoryDataSchema) -> bool:
        """
        이야기 상태 저장
        
        Args:
            story_schema: 저장할 이야기 스키마
            
        Returns:
            저장 성공 여부
        """
        try:
            story_id = story_schema.metadata.story_id
            
            # 상태 파일 경로
            state_file = os.path.join(self.state_dir, f"{story_id}.json")
            
            # 이야기 스키마 저장
            story_schema.save_to_file(state_file)
            
            # 상태 스냅샷 생성
            workflow_state = self._map_story_stage_to_workflow_state(story_schema.current_stage)
            snapshot = StateSnapshot(
                story_id=story_id,
                workflow_state=workflow_state,
                story_stage=story_schema.current_stage,
                timestamp=datetime.now(),
                progress_percentage=story_schema.get_completion_percentage(),
                error_count=len(story_schema.errors),
                last_activity=story_schema.stage_history[-1]["notes"] if story_schema.stage_history else "초기화"
            )
            
            # 이전 상태와 비교
            old_snapshot = self.active_states.get(story_id)
            self.active_states[story_id] = snapshot
            
            # 메타데이터 저장
            await self._save_metadata(snapshot)
            
            # 상태 변경 알림
            self._notify_state_change(story_id, old_snapshot, snapshot)
            
            self.logger.info(f"이야기 상태 저장 완료: {story_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"이야기 상태 저장 실패: {e}")
            return False
    
    async def load_story_state(self, story_id: str) -> Optional[StoryDataSchema]:
        """
        이야기 상태 로드
        
        Args:
            story_id: 로드할 이야기 ID
            
        Returns:
            로드된 이야기 스키마 또는 None
        """
        try:
            state_file = os.path.join(self.state_dir, f"{story_id}.json")
            
            if not os.path.exists(state_file):
                self.logger.warning(f"상태 파일을 찾을 수 없음: {story_id}")
                return None
            
            # 이야기 스키마 로드
            story_schema = StoryDataSchema.load_from_file(state_file)
            
            # 상태 스냅샷 복원
            workflow_state = self._map_story_stage_to_workflow_state(story_schema.current_stage)
            snapshot = StateSnapshot(
                story_id=story_id,
                workflow_state=workflow_state,
                story_stage=story_schema.current_stage,
                timestamp=story_schema.metadata.updated_at,
                progress_percentage=story_schema.get_completion_percentage(),
                error_count=len(story_schema.errors),
                last_activity=story_schema.stage_history[-1]["notes"] if story_schema.stage_history else "로드됨"
            )
            
            self.active_states[story_id] = snapshot
            
            self.logger.info(f"이야기 상태 로드 완료: {story_id}")
            return story_schema
            
        except Exception as e:
            self.logger.error(f"이야기 상태 로드 실패: {e}")
            return None
    
    async def get_story_status(self, story_id: str) -> Optional[Dict[str, Any]]:
        """
        이야기 상태 정보 조회
        
        Args:
            story_id: 조회할 이야기 ID
            
        Returns:
            상태 정보 딕셔너리 또는 None
        """
        # 활성 상태에서 먼저 확인
        if story_id in self.active_states:
            snapshot = self.active_states[story_id]
            return snapshot.to_dict()
        
        # 저장된 상태에서 확인
        try:
            story_schema = await self.load_story_state(story_id)
            if story_schema:
                return {
                    "story_id": story_id,
                    "workflow_state": self._map_story_stage_to_workflow_state(story_schema.current_stage).value,
                    "story_stage": story_schema.current_stage.value,
                    "progress_percentage": story_schema.get_completion_percentage(),
                    "error_count": len(story_schema.errors),
                    "created_at": story_schema.metadata.created_at.isoformat(),
                    "updated_at": story_schema.metadata.updated_at.isoformat()
                }
        except Exception as e:
            self.logger.error(f"상태 정보 조회 실패: {e}")
        
        return None
    
    async def list_active_stories(self) -> List[Dict[str, Any]]:
        """활성 이야기 목록 반환"""
        return [snapshot.to_dict() for snapshot in self.active_states.values()]
    
    async def list_all_stories(self) -> List[Dict[str, Any]]:
        """모든 이야기 목록 반환"""
        stories = []
        
        try:
            # 상태 디렉토리의 모든 파일 확인
            for filename in os.listdir(self.state_dir):
                if filename.endswith('.json'):
                    story_id = filename[:-5]  # .json 제거
                    status = await self.get_story_status(story_id)
                    if status:
                        stories.append(status)
        except Exception as e:
            self.logger.error(f"이야기 목록 조회 실패: {e}")
        
        return stories
    
    async def delete_story_state(self, story_id: str) -> bool:
        """
        이야기 상태 삭제
        
        Args:
            story_id: 삭제할 이야기 ID
            
        Returns:
            삭제 성공 여부
        """
        try:
            # 상태 파일 삭제
            state_file = os.path.join(self.state_dir, f"{story_id}.json")
            if os.path.exists(state_file):
                os.remove(state_file)
            
            # 메타데이터 삭제
            metadata_file = os.path.join(self.metadata_dir, f"{story_id}_metadata.json")
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            
            # 활성 상태에서 제거
            if story_id in self.active_states:
                del self.active_states[story_id]
            
            self.logger.info(f"이야기 상태 삭제 완료: {story_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"이야기 상태 삭제 실패: {e}")
            return False
    
    async def cleanup_old_states(self, days_old: int = 30) -> int:
        """
        오래된 상태 정리
        
        Args:
            days_old: 삭제할 상태의 최소 나이 (일)
            
        Returns:
            삭제된 상태 수
        """
        deleted_count = 0
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        
        try:
            for filename in os.listdir(self.state_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.state_dir, filename)
                    file_mtime = os.path.getmtime(file_path)
                    
                    if file_mtime < cutoff_date:
                        story_id = filename[:-5]
                        if await self.delete_story_state(story_id):
                            deleted_count += 1
            
            self.logger.info(f"오래된 상태 정리 완료: {deleted_count}개 삭제")
            
        except Exception as e:
            self.logger.error(f"상태 정리 실패: {e}")
        
        return deleted_count
    
    async def get_workflow_statistics(self) -> Dict[str, Any]:
        """워크플로우 통계 정보 반환"""
        try:
            all_stories = await self.list_all_stories()
            
            # 상태별 통계
            state_counts = {}
            stage_counts = {}
            total_errors = 0
            
            for story in all_stories:
                # 워크플로우 상태별 카운트
                workflow_state = story.get("workflow_state", "unknown")
                state_counts[workflow_state] = state_counts.get(workflow_state, 0) + 1
                
                # 스토리 단계별 카운트
                story_stage = story.get("story_stage", "unknown")
                stage_counts[story_stage] = stage_counts.get(story_stage, 0) + 1
                
                # 오류 카운트
                total_errors += story.get("error_count", 0)
            
            return {
                "total_stories": len(all_stories),
                "active_stories": len(self.active_states),
                "state_distribution": state_counts,
                "stage_distribution": stage_counts,
                "total_errors": total_errors,
                "success_rate": (state_counts.get("completed", 0) / len(all_stories) * 100) if all_stories else 0
            }
            
        except Exception as e:
            self.logger.error(f"통계 정보 조회 실패: {e}")
            return {}
    
    def _map_story_stage_to_workflow_state(self, story_stage: StoryStage) -> WorkflowState:
        """스토리 단계를 워크플로우 상태로 매핑"""
        mapping = {
            StoryStage.COLLECTION: WorkflowState.COLLECTING,
            StoryStage.VALIDATION: WorkflowState.VALIDATING,
            StoryStage.GENERATION: WorkflowState.GENERATING,
            StoryStage.MULTIMEDIA: WorkflowState.MULTIMEDIA,
            StoryStage.COMPLETION: WorkflowState.COMPLETED,
            StoryStage.ERROR: WorkflowState.ERROR
        }
        
        return mapping.get(story_stage, WorkflowState.IDLE)
    
    async def _save_metadata(self, snapshot: StateSnapshot):
        """메타데이터 저장"""
        try:
            metadata_file = os.path.join(self.metadata_dir, f"{snapshot.story_id}_metadata.json")
            
            metadata = {
                "story_id": snapshot.story_id,
                "workflow_state": snapshot.workflow_state.value,
                "story_stage": snapshot.story_stage.value,
                "last_updated": snapshot.timestamp.isoformat(),
                "progress_percentage": snapshot.progress_percentage,
                "error_count": snapshot.error_count,
                "last_activity": snapshot.last_activity
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"메타데이터 저장 실패: {e}")
    
    async def export_state_summary(self, output_file: str) -> bool:
        """상태 요약 정보 내보내기"""
        try:
            summary = {
                "export_timestamp": datetime.now().isoformat(),
                "statistics": await self.get_workflow_statistics(),
                "active_stories": await self.list_active_stories(),
                "all_stories": await self.list_all_stories()
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"상태 요약 내보내기 완료: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"상태 요약 내보내기 실패: {e}")
            return False 