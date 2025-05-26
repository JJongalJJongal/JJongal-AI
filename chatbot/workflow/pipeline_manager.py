"""
CCB_AI Pipeline Manager

이야기 생성 파이프라인의 단계별 관리를 담당하는 모듈입니다.
각 단계의 실행, 검증, 전환을 관리합니다.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from .story_schema import StoryDataSchema, StoryStage

class PipelineStage(Enum):
    """파이프라인 단계"""
    INITIALIZATION = "initialization"    # 초기화
    COLLECTION = "collection"           # 이야기 요소 수집
    VALIDATION = "validation"           # 데이터 검증
    GENERATION = "generation"           # 이야기 생성
    MULTIMEDIA = "multimedia"           # 멀티미디어 생성
    FINALIZATION = "finalization"       # 최종 완성
    CLEANUP = "cleanup"                 # 정리

@dataclass
class StageResult:
    """단계 실행 결과"""
    stage: PipelineStage
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    enable_multimedia: bool = True
    enable_validation: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300
    parallel_execution: bool = False
    stage_callbacks: Dict[PipelineStage, List[Callable]] = field(default_factory=dict)

class PipelineManager:
    """
    파이프라인 관리자
    
    이야기 생성 과정의 각 단계를 관리하고 실행합니다.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        파이프라인 관리자 초기화
        
        Args:
            config: 파이프라인 설정
        """
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger(__name__)
        
        # 단계별 핸들러
        self.stage_handlers: Dict[PipelineStage, Callable] = {}
        
        # 실행 통계
        self.execution_stats: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "stage_stats": {stage.value: {"count": 0, "success": 0, "avg_time": 0.0} for stage in PipelineStage}
        }
        
        # 활성 파이프라인
        self.active_pipelines: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("파이프라인 관리자 초기화 완료")
    
    def register_stage_handler(self, stage: PipelineStage, handler: Callable):
        """단계별 핸들러 등록"""
        self.stage_handlers[stage] = handler
        self.logger.info(f"단계 핸들러 등록: {stage.value}")
    
    def add_stage_callback(self, stage: PipelineStage, callback: Callable):
        """단계별 콜백 추가"""
        if stage not in self.config.stage_callbacks:
            self.config.stage_callbacks[stage] = []
        self.config.stage_callbacks[stage].append(callback)
    
    async def execute_pipeline(
        self,
        story_schema: StoryDataSchema,
        start_stage: Optional[PipelineStage] = None,
        end_stage: Optional[PipelineStage] = None
    ) -> List[StageResult]:
        """
        파이프라인 실행
        
        Args:
            story_schema: 이야기 스키마
            start_stage: 시작 단계 (None이면 처음부터)
            end_stage: 종료 단계 (None이면 끝까지)
            
        Returns:
            단계별 실행 결과 목록
        """
        story_id = story_schema.metadata.story_id
        self.active_pipelines[story_id] = {
            "start_time": datetime.now(),
            "current_stage": None,
            "results": []
        }
        
        try:
            self.logger.info(f"파이프라인 실행 시작: {story_id}")
            
            # 실행할 단계들 결정
            stages = self._get_execution_stages(start_stage, end_stage)
            results = []
            
            for stage in stages:
                self.active_pipelines[story_id]["current_stage"] = stage
                
                # 단계 실행
                result = await self._execute_stage(stage, story_schema)
                results.append(result)
                self.active_pipelines[story_id]["results"].append(result)
                
                # 실행 실패 시 처리
                if not result.success:
                    if self._should_retry(stage, result):
                        # 재시도
                        retry_result = await self._retry_stage(stage, story_schema)
                        results.append(retry_result)
                        if not retry_result.success:
                            break
                    else:
                        break
                
                # 콜백 실행
                await self._execute_stage_callbacks(stage, story_schema, result)
            
            # 통계 업데이트
            self._update_execution_stats(results)
            
            self.logger.info(f"파이프라인 실행 완료: {story_id}")
            return results
            
        except Exception as e:
            self.logger.error(f"파이프라인 실행 실패: {e}")
            raise
        
        finally:
            # 정리
            if story_id in self.active_pipelines:
                del self.active_pipelines[story_id]
    
    async def _execute_stage(self, stage: PipelineStage, story_schema: StoryDataSchema) -> StageResult:
        """개별 단계 실행"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"단계 실행 시작: {stage.value}")
            
            # 단계별 핸들러 실행
            if stage in self.stage_handlers:
                handler = self.stage_handlers[stage]
                
                # 타임아웃 설정
                result_data = await asyncio.wait_for(
                    handler(story_schema),
                    timeout=self.config.timeout_seconds
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return StageResult(
                    stage=stage,
                    success=True,
                    message=f"{stage.value} 단계 실행 완료",
                    data=result_data,
                    execution_time=execution_time
                )
            else:
                # 기본 처리
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return StageResult(
                    stage=stage,
                    success=True,
                    message=f"{stage.value} 단계 기본 처리 완료",
                    execution_time=execution_time
                )
                
        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start_time).total_seconds()
            return StageResult(
                stage=stage,
                success=False,
                message=f"{stage.value} 단계 타임아웃",
                execution_time=execution_time,
                errors=[f"타임아웃 ({self.config.timeout_seconds}초)"]
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return StageResult(
                stage=stage,
                success=False,
                message=f"{stage.value} 단계 실행 실패",
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    async def _retry_stage(self, stage: PipelineStage, story_schema: StoryDataSchema) -> StageResult:
        """단계 재시도"""
        self.logger.info(f"단계 재시도: {stage.value}")
        
        for attempt in range(self.config.max_retries):
            result = await self._execute_stage(stage, story_schema)
            if result.success:
                result.message += f" (재시도 {attempt + 1}회 성공)"
                return result
            
            # 재시도 간격
            await asyncio.sleep(1 * (attempt + 1))
        
        return StageResult(
            stage=stage,
            success=False,
            message=f"{stage.value} 단계 재시도 실패",
            errors=[f"최대 재시도 횟수 ({self.config.max_retries}) 초과"]
        )
    
    def _should_retry(self, stage: PipelineStage, result: StageResult) -> bool:
        """재시도 여부 결정"""
        # 타임아웃이나 일시적 오류의 경우 재시도
        if "타임아웃" in result.message or "일시적" in result.message:
            return True
        
        # 특정 단계는 재시도하지 않음
        if stage in [PipelineStage.FINALIZATION, PipelineStage.CLEANUP]:
            return False
        
        return True
    
    async def _execute_stage_callbacks(
        self,
        stage: PipelineStage,
        story_schema: StoryDataSchema,
        result: StageResult
    ):
        """단계별 콜백 실행"""
        callbacks = self.config.stage_callbacks.get(stage, [])
        
        for callback in callbacks:
            try:
                await callback(story_schema, result)
            except Exception as e:
                self.logger.error(f"콜백 실행 실패 ({stage.value}): {e}")
    
    def _get_execution_stages(
        self,
        start_stage: Optional[PipelineStage],
        end_stage: Optional[PipelineStage]
    ) -> List[PipelineStage]:
        """실행할 단계 목록 생성"""
        all_stages = list(PipelineStage)
        
        # 시작 인덱스
        start_idx = 0
        if start_stage:
            start_idx = all_stages.index(start_stage)
        
        # 종료 인덱스
        end_idx = len(all_stages)
        if end_stage:
            end_idx = all_stages.index(end_stage) + 1
        
        stages = all_stages[start_idx:end_idx]
        
        # 설정에 따른 단계 필터링
        if not self.config.enable_multimedia:
            stages = [s for s in stages if s != PipelineStage.MULTIMEDIA]
        
        if not self.config.enable_validation:
            stages = [s for s in stages if s != PipelineStage.VALIDATION]
        
        return stages
    
    def _update_execution_stats(self, results: List[StageResult]):
        """실행 통계 업데이트"""
        self.execution_stats["total_executions"] += 1
        
        # 전체 성공 여부
        all_success = all(result.success for result in results)
        if all_success:
            self.execution_stats["successful_executions"] += 1
        else:
            self.execution_stats["failed_executions"] += 1
        
        # 단계별 통계
        for result in results:
            stage_key = result.stage.value
            stage_stats = self.execution_stats["stage_stats"][stage_key]
            
            stage_stats["count"] += 1
            if result.success:
                stage_stats["success"] += 1
            
            # 평균 실행 시간 업데이트
            current_avg = stage_stats["avg_time"]
            new_avg = (current_avg * (stage_stats["count"] - 1) + result.execution_time) / stage_stats["count"]
            stage_stats["avg_time"] = new_avg
    
    def get_pipeline_status(self, story_id: str) -> Optional[Dict[str, Any]]:
        """파이프라인 상태 조회"""
        if story_id not in self.active_pipelines:
            return None
        
        pipeline_info = self.active_pipelines[story_id]
        
        return {
            "story_id": story_id,
            "start_time": pipeline_info["start_time"].isoformat(),
            "current_stage": pipeline_info["current_stage"].value if pipeline_info["current_stage"] else None,
            "completed_stages": len(pipeline_info["results"]),
            "results": [
                {
                    "stage": result.stage.value,
                    "success": result.success,
                    "message": result.message,
                    "execution_time": result.execution_time
                }
                for result in pipeline_info["results"]
            ]
        }
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """실행 통계 조회"""
        return self.execution_stats.copy()
    
    def get_active_pipelines(self) -> List[str]:
        """활성 파이프라인 목록"""
        return list(self.active_pipelines.keys())
    
    async def cancel_pipeline(self, story_id: str) -> bool:
        """파이프라인 취소"""
        if story_id in self.active_pipelines:
            del self.active_pipelines[story_id]
            self.logger.info(f"파이프라인 취소: {story_id}")
            return True
        return False
    
    def validate_stage_transition(self, from_stage: StoryStage, to_stage: StoryStage) -> bool:
        """단계 전환 유효성 검증"""
        # 단계 전환 규칙 정의
        valid_transitions = {
            StoryStage.COLLECTION: [StoryStage.VALIDATION, StoryStage.ERROR],
            StoryStage.VALIDATION: [StoryStage.GENERATION, StoryStage.COLLECTION, StoryStage.ERROR],
            StoryStage.GENERATION: [StoryStage.MULTIMEDIA, StoryStage.COMPLETION, StoryStage.ERROR],
            StoryStage.MULTIMEDIA: [StoryStage.COMPLETION, StoryStage.ERROR],
            StoryStage.COMPLETION: [],
            StoryStage.ERROR: [StoryStage.COLLECTION, StoryStage.VALIDATION, StoryStage.GENERATION]
        }
        
        return to_stage in valid_transitions.get(from_stage, [])
    
    async def resume_pipeline(
        self,
        story_schema: StoryDataSchema,
        from_stage: Optional[PipelineStage] = None
    ) -> List[StageResult]:
        """파이프라인 재개"""
        # 현재 단계에서 재개
        if not from_stage:
            # 스토리 단계를 파이프라인 단계로 매핑
            stage_mapping = {
                StoryStage.COLLECTION: PipelineStage.COLLECTION,
                StoryStage.VALIDATION: PipelineStage.VALIDATION,
                StoryStage.GENERATION: PipelineStage.GENERATION,
                StoryStage.MULTIMEDIA: PipelineStage.MULTIMEDIA,
                StoryStage.COMPLETION: PipelineStage.FINALIZATION
            }
            from_stage = stage_mapping.get(story_schema.current_stage, PipelineStage.COLLECTION)
        
        return await self.execute_pipeline(story_schema, start_stage=from_stage)
    
    def reset_statistics(self):
        """통계 초기화"""
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "stage_stats": {stage.value: {"count": 0, "success": 0, "avg_time": 0.0} for stage in PipelineStage}
        }
        self.logger.info("실행 통계 초기화 완료") 