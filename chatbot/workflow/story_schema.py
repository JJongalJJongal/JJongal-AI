"""
CCB_AI Story Data Schema

표준화된 이야기 데이터 형식을 정의하여 부기(ChatBot A)와 꼬기(ChatBot B) 간의
원활한 데이터 교환을 지원.

이 스키마는 이야기 수집부터 최종 생성까지 전체 파이프라인에서 사용되는
데이터 구조를 표준화.
"""

import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional

# 상위 디렉토리의 shared 모듈 임포트를 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from shared.utils.age_group_utils import AgeGroup, AgeGroupManager
except ImportError:
    # 폴백: 로컬 정의
    class AgeGroup(Enum):
        """연령대 분류"""
        YOUNG_CHILDREN = "young_children"      # 4-7세
        ELEMENTARY = "elementary"              # 8-9세

class StoryStage(Enum):
    """이야기 생성 단계"""
    COLLECTION = "collection"      # 부기에서 이야기 요소 수집
    VALIDATION = "validation"      # 수집된 데이터 검증
    GENERATION = "generation"      # 꼬기에서 이야기 생성
    MULTIMEDIA = "multimedia"      # 멀티미디어 생성
    COMPLETION = "completion"      # 최종 완성
    ERROR = "error"               # 오류 상태

class ElementType(Enum):
    """이야기 요소 타입"""
    CHARACTER = "character"        # 등장인물
    SETTING = "setting"           # 배경
    PROBLEM = "problem"           # 문제/갈등
    RESOLUTION = "resolution"     # 해결

@dataclass
class StoryElement:
    """개별 이야기 요소"""
    element_type: ElementType
    content: str
    keywords: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    source_conversation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "element_type": self.element_type.value,
            "content": self.content,
            "keywords": self.keywords,
            "confidence_score": self.confidence_score,
            "source_conversation": self.source_conversation,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoryElement':
        """딕셔너리에서 생성"""
        return cls(
            element_type=ElementType(data["element_type"]),
            content=data["content"],
            keywords=data.get("keywords", []),
            confidence_score=data.get("confidence_score", 0.0),
            source_conversation=data.get("source_conversation"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now()
        )

@dataclass
class ChildProfile:
    """아이 프로필 정보"""
    name: str
    age: int
    age_group: AgeGroup
    interests: List[str] = field(default_factory=list)
    language_level: str = "basic"
    special_needs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "name": self.name,
            "age": self.age,
            "age_group": self.age_group.value,
            "interests": self.interests,
            "language_level": self.language_level,
            "special_needs": self.special_needs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChildProfile':
        """딕셔너리에서 생성"""
        return cls(
            name=data["name"],
            age=data["age"],
            age_group=AgeGroup(data["age_group"]),
            interests=data.get("interests", []),
            language_level=data.get("language_level", "basic"),
            special_needs=data.get("special_needs", [])
        )

@dataclass
class ConversationSummary:
    """대화 요약 정보"""
    total_messages: int
    conversation_duration: float  # 분 단위
    key_topics: List[str] = field(default_factory=list)
    emotional_tone: str = "neutral"
    engagement_level: float = 0.0
    summary_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "total_messages": self.total_messages,
            "conversation_duration": self.conversation_duration,
            "key_topics": self.key_topics,
            "emotional_tone": self.emotional_tone,
            "engagement_level": self.engagement_level,
            "summary_text": self.summary_text
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSummary':
        """딕셔너리에서 생성"""
        return cls(
            total_messages=data["total_messages"],
            conversation_duration=data["conversation_duration"],
            key_topics=data.get("key_topics", []),
            emotional_tone=data.get("emotional_tone", "neutral"),
            engagement_level=data.get("engagement_level", 0.0),
            summary_text=data.get("summary_text", "")
        )

@dataclass
class StoryMetadata:
    """이야기 메타데이터"""
    story_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = None
    moral_lesson: Optional[str] = None
    educational_value: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "story_id": self.story_id,
            "title": self.title,
            "moral_lesson": self.moral_lesson,
            "educational_value": self.educational_value,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoryMetadata':
        """딕셔너리에서 생성"""
     
        return cls(
            story_id=data.get("story_id", str(uuid.uuid4())),
            title=data.get("title"),
            moral_lesson=data.get("moral_lesson"),
            educational_value=data.get("educational_value", []),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now()
        )

@dataclass
class GeneratedStory:
    """생성된 이야기 정보"""
    content: str
    chapters: List[Dict[str, str]] = field(default_factory=list)
    word_count: int = 0
    generated_at: datetime = field(default_factory=datetime.now)
    generation_model: str = "gpt-4o-mini"
    quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "content": self.content,
            "chapters": self.chapters,
            "word_count": self.word_count,
            "generated_at": self.generated_at.isoformat(),
            "generation_model": self.generation_model,
            "quality_score": self.quality_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeneratedStory':
        """딕셔너리에서 생성"""
        return cls(
            content=data["content"],
            chapters=data.get("chapters", []),
            word_count=data.get("word_count", 0),
            generated_at=datetime.fromisoformat(data["generated_at"]) if data.get("generated_at") else datetime.now(),
            generation_model=data.get("generation_model", "gpt-4o-mini"),
            quality_score=data.get("quality_score", 0.0)
        )

@dataclass
class MultimediaAssets:
    """멀티미디어 자산 정보"""
    images: List[Dict[str, str]] = field(default_factory=list)  # {"url": "", "description": "", "scene": ""}
    audio_files: List[Dict[str, str]] = field(default_factory=list)  # {"url": "", "type": "", "duration": ""}
    video_files: List[Dict[str, str]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "images": self.images,
            "audio_files": self.audio_files,
            "video_files": self.video_files,
            "generated_at": self.generated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultimediaAssets':
        """딕셔너리에서 생성"""
        return cls(
            images=data.get("images", []),
            audio_files=data.get("audio_files", []),
            video_files=data.get("video_files", []),
            generated_at=datetime.fromisoformat(data["generated_at"]) if data.get("generated_at") else datetime.now()
        )

@dataclass
class StoryDataSchema:
    """완전한 이야기 데이터 스키마"""
    # 메타데이터
    metadata: StoryMetadata = field(default_factory=StoryMetadata)
    
    # 아이 정보
    child_profile: Optional[ChildProfile] = None
    
    # 수집된 이야기 요소들
    story_elements: Dict[ElementType, List[StoryElement]] = field(default_factory=dict)
    
    # 대화 요약
    conversation_summary: Optional[ConversationSummary] = None
    
    # 생성된 이야기
    generated_story: Optional[GeneratedStory] = None
    
    # 멀티미디어 자산
    multimedia_assets: Optional[MultimediaAssets] = None
    
    # 워크플로우 상태
    current_stage: StoryStage = StoryStage.COLLECTION
    stage_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # 오류 정보
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """초기화 후 처리"""
        # 이야기 요소 딕셔너리 초기화
        if not self.story_elements:
            self.story_elements = {element_type: [] for element_type in ElementType}
    
    def add_story_element(self, element: StoryElement):
        """이야기 요소 추가"""
        if element.element_type not in self.story_elements:
            self.story_elements[element.element_type] = []
        self.story_elements[element.element_type].append(element)
        self.metadata.updated_at = datetime.now()
    
    def get_elements_by_type(self, element_type: ElementType) -> List[StoryElement]:
        """타입별 이야기 요소 반환"""
        return self.story_elements.get(element_type, [])
    
    def get_all_elements(self) -> List[StoryElement]:
        """모든 이야기 요소 반환"""
        all_elements = []
        for elements in self.story_elements.values():
            all_elements.extend(elements)
        return all_elements
    
    def update_stage(self, new_stage: StoryStage, notes: str = ""):
        """단계 업데이트"""
        # 이전 단계 기록
        self.stage_history.append({
            "from_stage": self.current_stage.value,
            "to_stage": new_stage.value,
            "timestamp": datetime.now().isoformat(),
            "notes": notes
        })
        
        self.current_stage = new_stage
        self.metadata.updated_at = datetime.now()
    
    def add_error(self, error_type: str, message: str, details: Dict[str, Any] = None):
        """오류 추가"""
        self.errors.append({
            "error_type": error_type,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
            "stage": self.current_stage.value
        })
        
        if self.current_stage != StoryStage.ERROR:
            self.update_stage(StoryStage.ERROR, f"Error: {error_type}")
    
    def is_ready_for_generation(self) -> bool:
        """이야기 생성 준비 상태 확인"""
        # 각 요소 타입별로 최소 1개씩 있는지 확인
        required_types = [ElementType.CHARACTER, ElementType.SETTING, ElementType.PROBLEM]
        
        for element_type in required_types:
            if not self.story_elements.get(element_type):
                return False
        
        return True
    
    def get_completion_percentage(self) -> float:
        """완성도 퍼센티지 계산"""
        total_stages = len(StoryStage) - 1  # ERROR 제외
        completed_stages = len(self.stage_history)
        
        return min(100.0, (completed_stages / total_stages) * 100)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "metadata": self.metadata.to_dict(),
            "child_profile": self.child_profile.to_dict() if self.child_profile else None,
            "story_elements": {
                element_type.value: [element.to_dict() for element in elements]
                for element_type, elements in self.story_elements.items()
            },
            "conversation_summary": self.conversation_summary.to_dict() if self.conversation_summary else None,
            "generated_story": self.generated_story.to_dict() if self.generated_story else None,
            "multimedia_assets": self.multimedia_assets.to_dict() if self.multimedia_assets else None,
            "current_stage": self.current_stage.value,
            "stage_history": self.stage_history,
            "errors": self.errors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoryDataSchema':
        """딕셔너리에서 생성"""
        instance = cls()
        
        # 메타데이터
        if "metadata" in data:
            instance.metadata = StoryMetadata.from_dict(data["metadata"])
        
        # 아이 프로필
        if data.get("child_profile"):
            instance.child_profile = ChildProfile.from_dict(data["child_profile"])
        
        # 이야기 요소들
        if "story_elements" in data:
            instance.story_elements = {}
            for element_type_str, elements_data in data["story_elements"].items():
                element_type = ElementType(element_type_str)
                instance.story_elements[element_type] = [
                    StoryElement.from_dict(element_data) for element_data in elements_data
                ]
        
        # 대화 요약
        if data.get("conversation_summary"):
            instance.conversation_summary = ConversationSummary.from_dict(data["conversation_summary"])
        
        # 생성된 이야기
        if data.get("generated_story"):
            instance.generated_story = GeneratedStory.from_dict(data["generated_story"])
        
        # 멀티미디어 자산
        if data.get("multimedia_assets"):
            instance.multimedia_assets = MultimediaAssets.from_dict(data["multimedia_assets"])
        
        # 워크플로우 상태
        instance.current_stage = StoryStage(data.get("current_stage", "collection"))
        instance.stage_history = data.get("stage_history", [])
        instance.errors = data.get("errors", [])
        
        return instance
    
    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StoryDataSchema':
        """JSON 문자열에서 생성"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def save_to_file(self, file_path: str):
        """파일로 저장"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'StoryDataSchema':
        """파일에서 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read()) 