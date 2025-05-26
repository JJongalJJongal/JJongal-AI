"""
CCB_AI Age Group Utilities

중앙화된 연령대 관리 유틸리티
모든 연령대 관련 로직을 통합하여 일관성을 보장합니다.
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

class AgeGroup(Enum):
    """연령대 분류 (통합된 2그룹 시스템)"""
    YOUNG_CHILDREN = "young_children"      # 4-7세
    ELEMENTARY = "elementary"              # 8-9세

@dataclass
class AgeGroupConfig:
    """연령대별 설정"""
    min_age: int
    max_age: int
    vocabulary_level: str
    sentence_complexity: str
    attention_span_minutes: int
    preferred_story_length: str
    educational_focus: List[str]

class AgeGroupManager:
    """연령대 관리자"""
    
    # 연령대별 설정
    AGE_GROUP_CONFIGS = {
        AgeGroup.YOUNG_CHILDREN: AgeGroupConfig(
            min_age=4,
            max_age=7,
            vocabulary_level="basic",
            sentence_complexity="simple_to_moderate",
            attention_span_minutes=10,
            preferred_story_length="short_to_medium",
            educational_focus=["basic_emotions", "friendship", "sharing", "simple_problem_solving"]
        ),
        AgeGroup.ELEMENTARY: AgeGroupConfig(
            min_age=8,
            max_age=9,
            vocabulary_level="intermediate",
            sentence_complexity="moderate_to_complex",
            attention_span_minutes=15,
            preferred_story_length="medium_to_long",
            educational_focus=["complex_emotions", "teamwork", "responsibility", "critical_thinking"]
        )
    }
    
    @classmethod
    def determine_age_group(cls, age: int) -> AgeGroup:
        """나이에 따른 연령대 결정"""
        if age <= 7:
            return AgeGroup.YOUNG_CHILDREN
        else:
            return AgeGroup.ELEMENTARY
    
    @classmethod
    def get_age_group_config(cls, age_group: AgeGroup) -> AgeGroupConfig:
        """연령대별 설정 반환"""
        return cls.AGE_GROUP_CONFIGS[age_group]
    
    @classmethod
    def get_age_group_by_age(cls, age: int) -> AgeGroupConfig:
        """나이로 연령대 설정 반환"""
        age_group = cls.determine_age_group(age)
        return cls.get_age_group_config(age_group)
    
    @classmethod
    def get_vocabulary_level(cls, age: int) -> str:
        """연령대별 어휘 수준 반환"""
        config = cls.get_age_group_by_age(age)
        return config.vocabulary_level
    
    @classmethod
    def get_sentence_complexity(cls, age: int) -> str:
        """연령대별 문장 복잡도 반환"""
        config = cls.get_age_group_by_age(age)
        return config.sentence_complexity
    
    @classmethod
    def get_attention_span(cls, age: int) -> int:
        """연령대별 집중 시간 반환 (분)"""
        config = cls.get_age_group_by_age(age)
        return config.attention_span_minutes
    
    @classmethod
    def get_story_length_preference(cls, age: int) -> str:
        """연령대별 선호 이야기 길이 반환"""
        config = cls.get_age_group_by_age(age)
        return config.preferred_story_length
    
    @classmethod
    def get_educational_focus(cls, age: int) -> List[str]:
        """연령대별 교육 초점 반환"""
        config = cls.get_age_group_by_age(age)
        return config.educational_focus
    
    @classmethod
    def is_age_appropriate_content(cls, age: int, content_complexity: str) -> bool:
        """연령대에 적합한 콘텐츠인지 확인"""
        config = cls.get_age_group_by_age(age)
        
        complexity_levels = {
            "very_simple": 1,
            "simple": 2,
            "simple_to_moderate": 3,
            "moderate": 4,
            "moderate_to_complex": 5,
            "complex": 6
        }
        
        content_level = complexity_levels.get(content_complexity, 3)
        max_level = complexity_levels.get(config.sentence_complexity, 3)
        
        return content_level <= max_level
    
    @classmethod
    def get_language_settings(cls, age: int) -> Dict[str, Any]:
        """연령대별 언어 설정 반환"""
        config = cls.get_age_group_by_age(age)
        age_group = cls.determine_age_group(age)
        
        if age_group == AgeGroup.YOUNG_CHILDREN:
            return {
                "sentence_length": "short",
                "vocabulary": "basic",
                "use_repetition": True,
                "use_sound_words": True,
                "max_concepts_per_scene": 2,
                "preferred_tense": "present",
                "use_questions": True,
                "emotional_complexity": "basic"
            }
        else:  # ELEMENTARY
            return {
                "sentence_length": "medium",
                "vocabulary": "intermediate",
                "use_repetition": False,
                "use_sound_words": False,
                "max_concepts_per_scene": 4,
                "preferred_tense": "mixed",
                "use_questions": True,
                "emotional_complexity": "moderate"
            }
    
    @classmethod
    def get_story_structure_guidelines(cls, age: int) -> Dict[str, Any]:
        """연령대별 이야기 구조 가이드라인"""
        config = cls.get_age_group_by_age(age)
        age_group = cls.determine_age_group(age)
        
        if age_group == AgeGroup.YOUNG_CHILDREN:
            return {
                "recommended_chapters": 5,
                "max_chapters": 7,
                "chapter_length_words": 100,
                "conflict_intensity": "low",
                "resolution_clarity": "very_clear",
                "character_development": "simple",
                "themes": ["friendship", "kindness", "sharing", "basic_emotions"]
            }
        else:  # ELEMENTARY
            return {
                "recommended_chapters": 8,
                "max_chapters": 12,
                "chapter_length_words": 200,
                "conflict_intensity": "moderate",
                "resolution_clarity": "clear",
                "character_development": "moderate",
                "themes": ["teamwork", "responsibility", "problem_solving", "empathy", "growth"]
            }
    
    @classmethod
    def validate_age_appropriateness(cls, age: int, content: Dict[str, Any]) -> Dict[str, Any]:
        """콘텐츠의 연령 적합성 검증"""
        config = cls.get_age_group_by_age(age)
        issues = []
        suggestions = []
        
        # 어휘 수준 검증
        if content.get("vocabulary_level") and content["vocabulary_level"] not in ["basic", config.vocabulary_level]:
            if config.vocabulary_level == "basic" and content["vocabulary_level"] in ["intermediate", "advanced"]:
                issues.append("어휘 수준이 너무 높습니다")
                suggestions.append("더 간단한 단어를 사용하세요")
        
        # 문장 복잡도 검증
        if content.get("sentence_complexity"):
            if not cls.is_age_appropriate_content(age, content["sentence_complexity"]):
                issues.append("문장 구조가 너무 복잡합니다")
                suggestions.append("더 짧고 간단한 문장을 사용하세요")
        
        # 이야기 길이 검증
        if content.get("word_count"):
            guidelines = cls.get_story_structure_guidelines(age)
            max_words = guidelines["max_chapters"] * guidelines["chapter_length_words"]
            if content["word_count"] > max_words:
                issues.append("이야기가 너무 깁니다")
                suggestions.append(f"최대 {max_words}단어로 줄이세요")
        
        return {
            "is_appropriate": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions,
            "recommended_config": config
        }
    
    @classmethod
    def get_multimedia_guidelines(cls, age: int) -> Dict[str, Any]:
        """연령대별 멀티미디어 가이드라인"""
        age_group = cls.determine_age_group(age)
        
        if age_group == AgeGroup.YOUNG_CHILDREN:
            return {
                "image_style": "very_simple_kawaii",
                "color_palette": "bright_primary_colors",
                "character_design": "ultra_cute_simple",
                "scene_complexity": "minimal",
                "audio_pace": "slow",
                "voice_tone": "very_gentle",
                "background_music": "soft_lullaby"
            }
        else:  # ELEMENTARY
            return {
                "image_style": "detailed_kawaii",
                "color_palette": "varied_harmonious",
                "character_design": "expressive_detailed",
                "scene_complexity": "moderate",
                "audio_pace": "normal",
                "voice_tone": "engaging",
                "background_music": "adventure_themed"
            }

# 하위 호환성을 위한 별칭
def determine_age_group(age: int) -> AgeGroup:
    """하위 호환성을 위한 함수"""
    return AgeGroupManager.determine_age_group(age)

def get_language_settings(age: int) -> Dict[str, Any]:
    """하위 호환성을 위한 함수"""
    return AgeGroupManager.get_language_settings(age)

def validate_age_appropriateness(age: int, content: Dict[str, Any]) -> Dict[str, Any]:
    """하위 호환성을 위한 함수"""
    return AgeGroupManager.validate_age_appropriateness(age, content) 