"""
JJongal-AI Story Generation Workflow Orchestrator

Orchestrates the complete story generation workflow from conversation to final output.
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

class WorkflowStage(Enum):
    """Workflow stages"""
    INITIALIZED = "initialized"
    CONVERSATION = "conversation"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    MULTIMEDIA = "multimedia"
    PACKAGING = "packaging"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class WorkflowResult:
    """Workflow execution result"""
    success: bool
    stage: WorkflowStage
    data: Dict[str, Any]
    error: Optional[str] = None
    duration_seconds: float = 0.0

class StoryGenerationOrchestrator:
    """Main workflow orchestrator for story generation"""
    
    def __init__(self):
        """Initialize orchestrator"""
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.completed_workflows: Dict[str, WorkflowResult] = {}
        self.max_concurrent_workflows = 5
        
        # Service references (will be injected)
        self.chatbot_a = None
        self.chatbot_b = None
        self.collaborator = None
        
        logger.info("Story Generation Orchestrator initialized")
    
    async def start_workflow(
        self,
        child_name: str,
        child_age: int,
        child_interests: List[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Start new story generation workflow
        
        Args:
            child_name: Child's name
            child_age: Child's age
            child_interests: Child's interests
            progress_callback: Progress callback function
            
        Returns:
            str: Workflow ID
        """
        if len(self.active_workflows) >= self.max_concurrent_workflows:
            raise RuntimeError("Maximum concurrent workflows reached")
        
        workflow_id = f"workflow_{uuid.uuid4().hex[:12]}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        workflow_context = {
            "workflow_id": workflow_id,
            "child_name": child_name,
            "child_age": child_age,
            "child_interests": child_interests or [],
            "stage": WorkflowStage.INITIALIZED,
            "start_time": datetime.now(),
            "progress_callback": progress_callback,
            "session_data": {},
            "collected_elements": {},
            "generated_content": {},
            "error_count": 0
        }
        
        self.active_workflows[workflow_id] = workflow_context
        
        logger.info(f"Started workflow {workflow_id} for {child_name} (age {child_age})")
        
        # Start workflow execution asynchronously
        asyncio.create_task(self._execute_workflow(workflow_id))
        
        return workflow_id
    
    async def _execute_workflow(self, workflow_id: str):
        """Execute complete workflow"""
        workflow = self.active_workflows[workflow_id]
        start_time = time.time()
        
        try:
            # Simulate workflow stages
            await self._update_workflow_stage(workflow_id, WorkflowStage.CONVERSATION)
            await asyncio.sleep(0.1)  # Simulate conversation stage
            
            await self._update_workflow_stage(workflow_id, WorkflowStage.GENERATION) 
            await asyncio.sleep(0.1)  # Simulate generation stage
            
            await self._update_workflow_stage(workflow_id, WorkflowStage.PACKAGING)
            await asyncio.sleep(0.1)  # Simulate packaging stage
            
            # Complete workflow
            duration = time.time() - start_time
            await self._complete_workflow(workflow_id, success=True, duration=duration)
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            duration = time.time() - start_time
            await self._complete_workflow(workflow_id, success=False, error=str(e), duration=duration)
    
    async def _update_workflow_stage(self, workflow_id: str, stage: WorkflowStage):
        """Update workflow stage"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["stage"] = stage
            
            await self._send_progress_update(workflow_id, "stage_changed", {
                "new_stage": stage.value,
                "message": f"Stage updated: {stage.value}"
            })
    
    async def _send_progress_update(self, workflow_id: str, update_type: str, data: Dict[str, Any]):
        """Send progress update via callback"""
        workflow = self.active_workflows.get(workflow_id)
        
        if workflow and workflow.get("progress_callback"):
            try:
                progress_data = {
                    "workflow_id": workflow_id,
                    "stage": workflow["stage"].value,
                    "update_type": update_type,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
                
                callback = workflow["progress_callback"]
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress_data)
                else:
                    callback(progress_data)
                    
            except Exception as e:
                logger.error(f"Failed to send progress update: {e}")
    
    async def _complete_workflow(self, workflow_id: str, success: bool, duration: float, error: str = None):
        """Complete workflow and store result"""
        workflow = self.active_workflows.get(workflow_id)
        
        if workflow:
            stage = WorkflowStage.COMPLETED if success else WorkflowStage.FAILED
            
            result = WorkflowResult(
                success=success,
                stage=stage,
                data=workflow.get("final_package", {}),
                error=error,
                duration_seconds=duration
            )
            
            self.completed_workflows[workflow_id] = result
            del self.active_workflows[workflow_id]
            
            logger.info(f"Workflow {workflow_id} completed: {'success' if success else 'failed'}")
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            return {
                "workflow_id": workflow_id,
                "status": "active",
                "stage": workflow["stage"].value,
                "child_name": workflow["child_name"],
                "start_time": workflow["start_time"].isoformat(),
                "progress": self._calculate_progress(workflow["stage"])
            }
        elif workflow_id in self.completed_workflows:
            result = self.completed_workflows[workflow_id]
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "success": result.success,
                "stage": result.stage.value,
                "duration": result.duration_seconds,
                "error": result.error
            }
        
        return None
    
    def _calculate_progress(self, stage: WorkflowStage) -> float:
        """Calculate progress percentage"""
        stage_progress = {
            WorkflowStage.INITIALIZED: 0.1,
            WorkflowStage.CONVERSATION: 0.3,
            WorkflowStage.ANALYSIS: 0.5,
            WorkflowStage.GENERATION: 0.7,
            WorkflowStage.MULTIMEDIA: 0.9,
            WorkflowStage.PACKAGING: 0.95,
            WorkflowStage.COMPLETED: 1.0,
            WorkflowStage.FAILED: 0.0
        }
        return stage_progress.get(stage, 0.0)
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel active workflow"""
        if workflow_id in self.active_workflows:
            await self._complete_workflow(
                workflow_id,
                success=False,
                duration=0,
                error="Workflow cancelled by user"
            )
            return True
        return False

# Global orchestrator instance
_orchestrator = None

def get_orchestrator() -> StoryGenerationOrchestrator:
    """Get global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = StoryGenerationOrchestrator()
    return _orchestrator