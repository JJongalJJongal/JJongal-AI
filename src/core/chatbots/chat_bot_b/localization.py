"""
Enhanced Ari Multi-Language Support System

Advanced localization framework with:
- Cultural context adaptation
- Age-appropriate language variations
- Dynamic prompt localization
- Voice and image localization
- Educational content adaptation
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

class Language(Enum):
    """Supported languages"""
    KOREAN = "ko"
    ENGLISH = "en"
    JAPANESE = "ja"
    CHINESE = "zh"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"

@dataclass
class CulturalContext:
    """Cultural context for localization"""
    language: Language
    traditional_values: List[str]
    storytelling_style: str
    educational_focus: List[str]
    character_archetypes: List[str]
    moral_themes: List[str]
    age_appropriate_content: Dict[str, Any]

@dataclass
class LocalizationConfig:
    """Configuration for localization"""
    language: Language
    cultural_context: CulturalContext
    voice_settings: Dict[str, str]
    prompt_templates: Dict[str, str]
    educational_standards: Dict[str, Any]

class EnhancedLocalizationManager:
    """
    Enhanced Localization Manager for multi-language story generation
    
    Features:
    - Cultural context adaptation
    - Age-specific language variations
    - Dynamic prompt localization
    - Educational content adaptation
    """
    
    def __init__(self):
        self.cultural_contexts = self._initialize_cultural_contexts()
        self.localization_configs = self._initialize_localization_configs()
        self.current_language = Language.KOREAN
        
    def _initialize_cultural_contexts(self) -> Dict[Language, CulturalContext]:
        """Initialize cultural contexts for supported languages"""
        contexts = {}
        
        # Korean Cultural Context
        contexts[Language.KOREAN] = CulturalContext(
            language=Language.KOREAN,
            traditional_values=["효도", "배려", "정직", "용기", "우정"],
            storytelling_style="따뜻하고 교훈적인 전래동화 스타일",
            educational_focus=["도덕성", "사회성", "창의성", "문제해결"],
            character_archetypes=["현명한 할머니", "용감한 아이", "착한 동물친구", "마법사"],
            moral_themes=["선악구분", "노력의 중요성", "타인배려", "자연사랑"],
            age_appropriate_content={
                "4-7": {
                    "vocabulary": "단순하고 친근한 어휘",
                    "sentence_structure": "짧고 반복적인 문장",
                    "themes": ["가족사랑", "친구관계", "기본예의"]
                },
                "8-9": {
                    "vocabulary": "조금 더 풍부한 어휘",
                    "sentence_structure": "복합문과 접속어 사용",
                    "themes": ["도덕적 판단", "사회규칙", "개인책임"]
                }
            }
        )
        
        # English Cultural Context
        contexts[Language.ENGLISH] = CulturalContext(
            language=Language.ENGLISH,
            traditional_values=["kindness", "honesty", "courage", "friendship", "perseverance"],
            storytelling_style="Classic fairy tale with modern sensibility",
            educational_focus=["critical thinking", "empathy", "creativity", "independence"],
            character_archetypes=["wise mentor", "brave hero", "loyal companion", "magical helper"],
            moral_themes=["good vs evil", "hard work pays off", "helping others", "believing in yourself"],
            age_appropriate_content={
                "4-7": {
                    "vocabulary": "Simple, familiar words",
                    "sentence_structure": "Short, rhythmic sentences",
                    "themes": ["family love", "making friends", "basic manners"]
                },
                "8-9": {
                    "vocabulary": "More descriptive language",
                    "sentence_structure": "Complex sentences with conjunctions",
                    "themes": ["moral choices", "responsibility", "problem-solving"]
                }
            }
        )
        
        # Japanese Cultural Context
        contexts[Language.JAPANESE] = CulturalContext(
            language=Language.JAPANESE,
            traditional_values=["思いやり", "礼儀", "努力", "協調", "自然愛"],
            storytelling_style="日本の昔話風で季節感のある語り",
            educational_focus=["思いやり", "協調性", "自然理解", "文化継承"],
            character_archetypes=["優しいおばあさん", "勇敢な子供", "動物の友達", "妖精"],
            moral_themes=["善悪の区別", "努力の大切さ", "他者への思いやり", "自然との共生"],
            age_appropriate_content={
                "4-7": {
                    "vocabulary": "やさしい日本語",
                    "sentence_structure": "短くてリズミカルな文",
                    "themes": ["家族の愛", "友達作り", "基本的なマナー"]
                },
                "8-9": {
                    "vocabulary": "少し豊かな表現",
                    "sentence_structure": "複文と接続詞の使用",
                    "themes": ["道徳的判断", "社会のルール", "個人の責任"]
                }
            }
        )
        
        return contexts
    
    def _initialize_localization_configs(self) -> Dict[Language, LocalizationConfig]:
        """Initialize localization configurations"""
        configs = {}
        
        for language, context in self.cultural_contexts.items():
            configs[language] = LocalizationConfig(
                language=language,
                cultural_context=context,
                voice_settings=self._get_voice_settings(language),
                prompt_templates=self._get_prompt_templates(language),
                educational_standards=self._get_educational_standards(language)
            )
            
        return configs
    
    def _get_voice_settings(self, language: Language) -> Dict[str, str]:
        """Get voice settings for language"""
        voice_mappings = {
            Language.KOREAN: {
                "narrator": "xi3rF0t7dg7uN2M0WUhr",  # Yuna
                "child_character": "pNInz6obpgDQGcFmaJgB",  # Adam
                "adult_character": "21m00Tcm4TlvDq8ikWAM",  # Rachel
                "model": "eleven_multilingual_v2"
            },
            Language.ENGLISH: {
                "narrator": "21m00Tcm4TlvDq8ikWAM",  # Rachel
                "child_character": "pNInz6obpgDQGcFmaJgB",  # Adam  
                "adult_character": "29vD33N1CtxCmqQRPOHJ",  # Drew
                "model": "eleven_multilingual_v2"
            },
            Language.JAPANESE: {
                "narrator": "xi3rF0t7dg7uN2M0WUhr",  # Yuna (multilingual)
                "child_character": "pNInz6obpgDQGcFmaJgB",  # Adam
                "adult_character": "21m00Tcm4TlvDq8ikWAM",  # Rachel
                "model": "eleven_multilingual_v2"
            }
        }
        
        return voice_mappings.get(language, voice_mappings[Language.KOREAN])
    
    def _get_prompt_templates(self, language: Language) -> Dict[str, str]:
        """Get prompt templates for language"""
        templates = {
            Language.KOREAN: {
                "planning": """당신은 한국 동화 제작 전문 기획자입니다.
한국 문화와 전통적 가치를 반영한 동화 계획을 수립해주세요.""",
                
                "generation": """당신은 한국 동화 작가 전문가입니다.
한국의 전통적 이야기 구조와 교육적 가치를 담은 동화를 작성해주세요.""",
                
                "evaluation": """당신은 한국 동화 품질 평가 전문가입니다.
한국 아동의 발달 단계와 교육과정에 맞는 평가를 해주세요.""",
                
                "enhancement": """당신은 한국 동화 편집 전문가입니다.
한국 문화적 맥락에서 동화를 최종 개선해주세요."""
            },
            
            Language.ENGLISH: {
                "planning": """You are a professional English fairy tale planning specialist.
Please create a plan that reflects Western cultural values and storytelling traditions.""",
                
                "generation": """You are an expert English fairy tale writer.
Please write engaging stories that incorporate classic Western narrative structures and educational values.""",
                
                "evaluation": """You are an English fairy tale quality assessment expert.
Please evaluate based on English-speaking children's developmental stages and educational standards.""",
                
                "enhancement": """You are an English fairy tale editing specialist.
Please provide final improvements within Western cultural contexts."""
            },
            
            Language.JAPANESE: {
                "planning": """あなたは日本の童話制作専門の企画者です。
日本の文化と伝統的価値観を反映した童話の計画を立ててください。""",
                
                "generation": """あなたは日本の童話作家の専門家です。
日本の伝統的な物語構造と教育的価値を込めた童話を書いてください。""",
                
                "evaluation": """あなたは日本の童話品質評価専門家です。
日本の子どもの発達段階と教育課程に合った評価をしてください。""",
                
                "enhancement": """あなたは日本の童話編集専門家です。
日本の文化的文脈で童話を最終的に改善してください。"""
            }
        }
        
        return templates.get(language, templates[Language.KOREAN])
    
    def _get_educational_standards(self, language: Language) -> Dict[str, Any]:
        """Get educational standards for language"""
        standards = {
            Language.KOREAN: {
                "age_4_7": {
                    "vocabulary_level": "누리과정 언어영역 수준",
                    "moral_development": "기본예절과 타인배려",
                    "cognitive_skills": "관찰력, 표현력, 상상력",
                    "social_skills": "가족사랑, 친구관계, 공동체의식"
                },
                "age_8_9": {
                    "vocabulary_level": "초등학교 저학년 수준",
                    "moral_development": "도덕적 판단력, 책임감",
                    "cognitive_skills": "논리적 사고, 문제해결, 창의성",
                    "social_skills": "협동, 배려, 리더십"
                }
            },
            
            Language.ENGLISH: {
                "age_4_7": {
                    "vocabulary_level": "Pre-K to Grade 1 level",
                    "moral_development": "Basic kindness and sharing",
                    "cognitive_skills": "Observation, expression, imagination",
                    "social_skills": "Family bonds, friendship, community"
                },
                "age_8_9": {
                    "vocabulary_level": "Grade 2-3 level",
                    "moral_development": "Moral reasoning, responsibility",
                    "cognitive_skills": "Critical thinking, problem-solving, creativity",
                    "social_skills": "Cooperation, empathy, leadership"
                }
            }
        }
        
        return standards.get(language, standards[Language.KOREAN])
    
    def set_language(self, language: Language):
        """Set current language"""
        self.current_language = language
        logger.info(f"Language set to: {language.value}")
    
    def get_cultural_context(self, language: Optional[Language] = None) -> CulturalContext:
        """Get cultural context for language"""
        lang = language or self.current_language
        return self.cultural_contexts.get(lang, self.cultural_contexts[Language.KOREAN])
    
    def get_localization_config(self, language: Optional[Language] = None) -> LocalizationConfig:
        """Get localization configuration for language"""
        lang = language or self.current_language
        return self.localization_configs.get(lang, self.localization_configs[Language.KOREAN])
    
    def localize_prompt(self, prompt_type: str, language: Optional[Language] = None) -> str:
        """Get localized prompt template"""
        lang = language or self.current_language
        config = self.get_localization_config(lang)
        return config.prompt_templates.get(prompt_type, "")
    
    def adapt_content_for_age(self, content: Dict[str, Any], target_age: int, 
                            language: Optional[Language] = None) -> Dict[str, Any]:
        """Adapt content for specific age and language"""
        lang = language or self.current_language
        context = self.get_cultural_context(lang)
        
        age_group = "4-7" if 4 <= target_age <= 7 else "8-9"
        age_content = context.age_appropriate_content.get(age_group, {})
        
        adapted_content = content.copy()
        adapted_content.update({
            "language": lang.value,
            "age_group": age_group,
            "cultural_context": {
                "values": context.traditional_values,
                "storytelling_style": context.storytelling_style,
                "vocabulary_level": age_content.get("vocabulary", ""),
                "sentence_structure": age_content.get("sentence_structure", ""),
                "themes": age_content.get("themes", [])
            }
        })
        
        return adapted_content
    
    def get_voice_config_for_character(self, character_type: str, 
                                     language: Optional[Language] = None) -> Dict[str, str]:
        """Get voice configuration for character type"""
        lang = language or self.current_language
        config = self.get_localization_config(lang)
        
        voice_settings = config.voice_settings
        character_mapping = {
            "narrator": voice_settings.get("narrator"),
            "child": voice_settings.get("child_character"),
            "adult": voice_settings.get("adult_character"),
            "default": voice_settings.get("narrator")
        }
        
        return {
            "voice_id": character_mapping.get(character_type, character_mapping["default"]),
            "model": voice_settings.get("model", "eleven_multilingual_v2"),
            "language_code": lang.value
        }
    
    def get_educational_guidelines(self, target_age: int, 
                                 language: Optional[Language] = None) -> Dict[str, Any]:
        """Get educational guidelines for age and language"""
        lang = language or self.current_language
        config = self.get_localization_config(lang)
        
        age_group = "age_4_7" if 4 <= target_age <= 7 else "age_8_9"
        return config.educational_standards.get(age_group, {})
    
    def create_localized_story_outline(self, base_outline: Dict[str, Any], 
                                     target_age: int,
                                     language: Optional[Language] = None) -> Dict[str, Any]:
        """Create localized story outline"""
        lang = language or self.current_language
        context = self.get_cultural_context(lang)
        educational_guidelines = self.get_educational_guidelines(target_age, lang)
        
        localized_outline = {
            **base_outline,
            "language": lang.value,
            "cultural_context": {
                "traditional_values": context.traditional_values,
                "storytelling_style": context.storytelling_style,
                "character_archetypes": context.character_archetypes,
                "moral_themes": context.moral_themes
            },
            "educational_guidelines": educational_guidelines,
            "localization_metadata": {
                "age_group": "4-7" if 4 <= target_age <= 7 else "8-9",
                "target_age": target_age,
                "adapted_for_culture": lang.value
            }
        }
        
        logger.info(f"Created localized outline for {lang.value}, age {target_age}")
        return localized_outline
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        return [lang.value for lang in Language]
    
    def is_language_supported(self, language_code: str) -> bool:
        """Check if language is supported"""
        return language_code in self.get_supported_languages()

# Global localization manager instance
localization_manager = EnhancedLocalizationManager()