"""
Story Templates for JJongal-AI

Provides template-based story generation with age-appropriate content,
cultural elements, and customizable story structures for Korean children.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
import random

class StoryGenre(Enum):
    """Story genre types"""
    ADVENTURE = "adventure"
    FRIENDSHIP = "friendship"
    FAMILY = "family"
    ANIMAL = "animal"
    FANTASY = "fantasy"
    EDUCATIONAL = "educational"
    SEASONAL = "seasonal"
    TRADITIONAL = "traditional"

class AgeGroup(Enum):
    """Age group classifications"""
    YOUNG = "4-6"  # Young children (4-6 years)
    OLD = "7-9"    # Elementary school (7-9 years)

@dataclass
class StoryTemplate:
    """Story template structure"""
    title_template: str
    opening_templates: List[str]
    development_templates: List[str]
    climax_templates: List[str]
    resolution_templates: List[str]
    ending_templates: List[str]
    moral_lessons: List[str]
    genre: StoryGenre
    age_group: AgeGroup
    cultural_elements: List[str] = field(default_factory=list)
    vocabulary_level: str = "simple"
    sentence_complexity: str = "basic"

class StoryTemplateManager:
    """
    Manages story templates and generates story structures
    
    Features:
    - Age-appropriate templates
    - Korean cultural elements
    - Genre-specific structures  
    - Customizable story components
    """
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, List[StoryTemplate]]:
        """Initialize all story templates"""
        templates = {
            "YOUNG_ADVENTURE": self._create_young_adventure_templates(),
            "YOUNG_FRIENDSHIP": self._create_young_friendship_templates(),
            "YOUNG_ANIMAL": self._create_young_animal_templates(),
            "OLD_ADVENTURE": self._create_old_adventure_templates(),
            "OLD_FRIENDSHIP": self._create_old_friendship_templates(),
            "OLD_FANTASY": self._create_old_fantasy_templates(),
            "EDUCATIONAL": self._create_educational_templates(),
            "TRADITIONAL": self._create_traditional_templates()
        }
        return templates
    
    def _create_young_adventure_templates(self) -> List[StoryTemplate]:
        """Create adventure templates for young children (4-6)"""
        return [
            StoryTemplate(
                title_template="{character}의 {place} 모험",
                opening_templates=[
                    "옛날 옛적에 {character}라는 {character_trait} {character_type}이 살고 있었어요.",
                    "{character}는 {place}에서 살고 있었어요. 어느 날 {character}는 모험을 떠나고 싶었어요.",
                    "작은 마을에 {character}라는 {character_trait} 아이가 살고 있었답니다."
                ],
                development_templates=[
                    "{character}는 {place}에 갔어요. 거기서 {problem}이 일어났어요.",
                    "길을 걷다가 {character}는 {helper}를 만났어요. {helper}가 말했어요: '{advice}'",
                    "{character}는 용기를 내어 {action}을 했어요."
                ],
                climax_templates=[
                    "드디어 {character}는 {challenge}를 해결해야 했어요.",
                    "가장 어려운 순간이 왔어요. {character}는 어떻게 해야 할까요?",
                    "{character}는 {skill}을 사용해서 문제를 풀어보기로 했어요."
                ],
                resolution_templates=[
                    "{character}는 {solution}으로 문제를 해결했어요.",
                    "친구들의 도움으로 {character}는 성공했어요.",
                    "{character}의 {positive_trait} 덕분에 모든 것이 잘되었어요."
                ],
                ending_templates=[
                    "그래서 {character}는 행복하게 집으로 돌아왔어요.",
                    "모든 친구들이 {character}를 축하해주었어요.",
                    "{character}는 새로운 친구들과 함께 즐겁게 놀았답니다."
                ],
                moral_lessons=[
                    "용기를 내면 어려운 일도 해결할 수 있어요",
                    "친구들과 함께하면 더 재미있어요",
                    "도움이 필요한 친구를 도와주는 것은 좋은 일이에요"
                ],
                genre=StoryGenre.ADVENTURE,
                age_group=AgeGroup.YOUNG,
                cultural_elements=["한국 전통 마을", "할머니의 지혜", "계절 변화", "자연과 함께하는 삶"],
                vocabulary_level="simple",
                sentence_complexity="basic"
            )
        ]
    
    def _create_young_friendship_templates(self) -> List[StoryTemplate]:
        """Create friendship templates for young children"""
        return [
            StoryTemplate(
                title_template="{character}와 {friend}의 우정",
                opening_templates=[
                    "{character}는 새로 이사온 동네에서 친구가 없어서 외로웠어요.",
                    "어린이집에 새로운 친구 {friend}가 왔어요.",
                    "{character}는 항상 혼자 놀고 있었어요."
                ],
                development_templates=[
                    "어느 날 {character}는 {friend}와 함께 {activity}를 했어요.",
                    "{friend}가 도움이 필요해 보였어요. {character}는 어떻게 해야 할까요?",
                    "처음에는 서로 다른 점이 많아서 어려웠어요."
                ],
                climax_templates=[
                    "{character}와 {friend} 사이에 {conflict}가 생겼어요.",
                    "큰 문제가 생겼을 때 둘은 어떻게 할까요?",
                    "{character}는 {friend}를 도와줄 방법을 생각했어요."
                ],
                resolution_templates=[
                    "서로 이야기를 나누니 오해가 풀렸어요.",
                    "함께 노력해서 문제를 해결했어요.",
                    "{character}와 {friend}는 서로를 더 잘 이해하게 되었어요."
                ],
                ending_templates=[
                    "이제 둘은 가장 좋은 친구가 되었어요.",
                    "매일 함께 놀며 즐거운 시간을 보냈어요.",
                    "진정한 우정이 무엇인지 알게 되었답니다."
                ],
                moral_lessons=[
                    "친구와 나누는 것은 즐거워요",
                    "서로 다른 점을 인정하고 존중해요",
                    "진짜 친구는 어려울 때 도와줘요"
                ],
                genre=StoryGenre.FRIENDSHIP,
                age_group=AgeGroup.YOUNG,
                cultural_elements=["어린이집", "동네 놀이터", "가족의 따뜻함"],
                vocabulary_level="simple",
                sentence_complexity="basic"
            )
        ]
    
    def _create_young_animal_templates(self) -> List[StoryTemplate]:
        """Create animal story templates for young children"""
        return [
            StoryTemplate(
                title_template="작은 {animal}의 큰 모험",
                opening_templates=[
                    "숲 속에 작은 {animal}가 엄마와 함께 살고 있었어요.",
                    "{animal}는 항상 엄마 곁에서만 놀았어요.",
                    "어느 봄날 {animal}는 처음으로 혼자 놀고 싶었어요."
                ],
                development_templates=[
                    "{animal}는 {place}에 가서 {discovery}를 발견했어요.",
                    "길을 잃은 {animal}는 {helper_animal}를 만났어요.",
                    "{animal}는 새로운 친구들과 {activity}를 배웠어요."
                ],
                climax_templates=[
                    "갑자기 {danger}가 나타났어요!",
                    "{animal}는 집으로 가는 길을 잊어버렸어요.",
                    "친구가 {trouble}에 빠졌을 때 {animal}는 어떻게 할까요?"
                ],
                resolution_templates=[
                    "{animal}는 {skill}을 사용해서 위험을 피했어요.",
                    "모든 동물 친구들이 함께 도와주었어요.",
                    "엄마가 가르쳐준 {wisdom}을 기억해냈어요."
                ],
                ending_templates=[
                    "{animal}는 무사히 엄마에게 돌아갔어요.",
                    "이제 {animal}는 더 용감한 동물이 되었어요.",
                    "숲의 모든 동물들이 {animal}와 친구가 되었답니다."
                ],
                moral_lessons=[
                    "엄마 말씀을 잘 들어야 해요",
                    "작아도 용감할 수 있어요",
                    "자연을 사랑하고 보호해요"
                ],
                genre=StoryGenre.ANIMAL,
                age_group=AgeGroup.YOUNG,
                cultural_elements=["한국의 숲", "계절의 변화", "동물과의 조화"],
                vocabulary_level="simple",
                sentence_complexity="basic"
            )
        ]
    
    def _create_old_adventure_templates(self) -> List[StoryTemplate]:
        """Create adventure templates for older children (7-9)"""
        return [
            StoryTemplate(
                title_template="{character}의 {magical_item} 찾기",
                opening_templates=[
                    "{character}는 {hometown}에서 평범한 생활을 하고 있었습니다. 하지만 어느 날 이상한 일이 벌어지기 시작했어요.",
                    "할아버지가 들려주신 {legend} 이야기를 듣고 {character}는 모험을 떠나기로 결심했습니다.",
                    "{character}는 우연히 오래된 지도를 발견했습니다. 그 지도에는 {treasure}의 위치가 표시되어 있었어요."
                ],
                development_templates=[
                    "첫 번째 시련은 {first_challenge}였습니다. {character}는 {skill}을 사용해서 이를 극복했어요.",
                    "여행 중 {character}는 {companion}을 만났습니다. 그들은 함께 {task}를 수행하기로 했어요.",
                    "{character}는 {mentor}로부터 중요한 {lesson}을 배웠습니다."
                ],
                climax_templates=[
                    "마침내 {character}는 {final_location}에 도착했습니다. 하지만 {final_boss}가 길을 막고 있었어요.",
                    "가장 어려운 퍼즐을 풀어야만 {goal}에 도달할 수 있었습니다.",
                    "{character}는 지금까지 배운 모든 것을 사용해야 했습니다."
                ],
                resolution_templates=[
                    "{character}는 {solution}을 통해 문제를 해결했습니다.",
                    "친구들과의 팀워크로 불가능해 보이던 일을 해냈어요.",
                    "{character}의 {virtue}이 마침내 빛을 발했습니다."
                ],
                ending_templates=[
                    "{character}는 {reward}를 얻었지만, 진짜 보물은 여행에서 얻은 {real_treasure}였습니다.",
                    "고향으로 돌아온 {character}는 모든 사람들과 자신의 경험을 나누었습니다.",
                    "이 모험을 통해 {character}는 {growth}하게 되었습니다."
                ],
                moral_lessons=[
                    "포기하지 않는 의지력이 중요해요",
                    "친구들과의 협력으로 더 큰 일을 할 수 있어요",
                    "진정한 보물은 경험과 성장이에요"
                ],
                genre=StoryGenre.ADVENTURE,
                age_group=AgeGroup.OLD,
                cultural_elements=["한국의 전설", "전통 무예", "옛날 궁궐", "산과 강"],
                vocabulary_level="intermediate",
                sentence_complexity="complex"
            )
        ]
    
    def _create_old_friendship_templates(self) -> List[StoryTemplate]:
        """Create friendship templates for older children"""
        return [
            StoryTemplate(
                title_template="서로 다른 {character1}와 {character2}",
                opening_templates=[
                    "{character1}는 {hobby1}을 좋아하고, {character2}는 {hobby2}를 좋아했습니다. 둘은 전혀 다른 성격이었어요.",
                    "새 학기가 시작되고 {character1}의 옆자리에 {character2}가 앉게 되었습니다.",
                    "{character1}와 {character2}는 같은 동아리에 들어갔지만 서로 맞지 않는다고 생각했어요."
                ],
                development_templates=[
                    "우연한 기회에 둘은 함께 {project}를 하게 되었습니다.",
                    "{character1}가 {problem}으로 어려움을 겪을 때 {character2}가 도움을 주었어요.",
                    "서로의 장점을 발견하게 된 계기는 {event}였습니다."
                ],
                climax_templates=[
                    "둘 사이에 큰 {misunderstanding}이 생겨서 서로 화를 내게 되었어요.",
                    "{competition}에서 서로 경쟁하게 되었지만 중요한 일이 생겼습니다.",
                    "친구를 도와야 하는 상황에서 {character1}는 선택을 해야 했어요."
                ],
                resolution_templates=[
                    "진심으로 이야기를 나눈 후 서로를 이해하게 되었습니다.",
                    "{character1}는 {sacrifice}를 통해 진정한 우정을 보여주었어요.",
                    "둘은 서로의 다른 점이 오히려 장점이라는 것을 깨달았습니다."
                ],
                ending_templates=[
                    "이제 둘은 서로를 존중하는 진정한 친구가 되었습니다.",
                    "{character1}와 {character2}는 각자의 꿈을 응원하며 함께 성장했어요.",
                    "다름을 인정하는 것이 진정한 우정의 시작임을 배웠습니다."
                ],
                moral_lessons=[
                    "서로 다른 것을 인정하고 존중해요",
                    "진정한 친구는 어려울 때 함께해요",
                    "편견을 버리고 열린 마음으로 사람을 만나요"
                ],
                genre=StoryGenre.FRIENDSHIP,
                age_group=AgeGroup.OLD,
                cultural_elements=["학교 생활", "동아리 활동", "한국의 전통 놀이"],
                vocabulary_level="intermediate",
                sentence_complexity="complex"
            )
        ]
    
    def _create_old_fantasy_templates(self) -> List[StoryTemplate]:
        """Create fantasy templates for older children"""
        return [
            StoryTemplate(
                title_template="{character}와 {magical_creature}의 비밀",
                opening_templates=[
                    "{character}는 평범한 {grade}학년 학생이었지만 어느 날 {magical_event}를 목격했습니다.",
                    "할머니 댁 다락방에서 {character}는 신비한 {magical_item}을 발견했어요.",
                    "달이 특별히 밝은 밤, {character}는 {magical_place}로 들어가게 되었습니다."
                ],
                development_templates=[
                    "{magical_creature}는 {character}에게 {magical_world}의 비밀을 알려주었습니다.",
                    "{character}는 특별한 {power}를 가지고 있다는 것을 알게 되었어요.",
                    "{magical_world}에 {crisis}가 닥쳐서 {character}의 도움이 필요했습니다."
                ],
                climax_templates=[
                    "{evil_force}가 {magical_world}를 위험에 빠뜨렸고, 오직 {character}만이 막을 수 있었어요.",
                    "{character}는 {difficult_choice}를 해야 했습니다. 평범한 생활을 포기할 것인가요?",
                    "마지막 시험은 {character}의 {inner_strength}를 시험하는 것이었습니다."
                ],
                resolution_templates=[
                    "{character}는 용기와 지혜로 {magical_world}를 구했습니다.",
                    "진정한 마법은 {character}의 마음 속에 있었어요.",
                    "{friendship}의 힘으로 불가능해 보이던 일을 해냈습니다."
                ],
                ending_templates=[
                    "{character}는 일상으로 돌아갔지만 특별한 비밀을 간직하게 되었어요.",
                    "이제 {character}는 두 세계를 연결하는 다리 역할을 하게 되었습니다.",
                    "마법의 세계는 항상 {character}를 기다리고 있을 거예요."
                ],
                moral_lessons=[
                    "평범해 보여도 누구나 특별한 면이 있어요",
                    "책임감을 가지고 자신의 역할을 다해요",
                    "상상력과 창의력은 소중한 보물이에요"
                ],
                genre=StoryGenre.FANTASY,
                age_group=AgeGroup.OLD,
                cultural_elements=["한국의 신화", "도깨비와 요정", "전통 무속", "자연 정령"],
                vocabulary_level="intermediate",
                sentence_complexity="complex"
            )
        ]
    
    def _create_educational_templates(self) -> List[StoryTemplate]:
        """Create educational story templates"""
        return [
            StoryTemplate(
                title_template="{character}가 배운 {lesson}",
                opening_templates=[
                    "{character}는 {subject}에 대해 잘 몰랐어요.",
                    "학교에서 {teacher}선생님이 {topic}에 대해 설명해주셨어요.",
                    "{character}는 {situation}에서 {problem}을 만났습니다."
                ],
                development_templates=[
                    "{character}는 {method}를 통해 {knowledge}를 배우기 시작했어요.",
                    "친구들과 함께 {experiment}를 해보았습니다.",
                    "{expert}가 {interesting_fact}를 알려주었어요."
                ],
                climax_templates=[
                    "{character}는 배운 지식을 실제로 사용해야 하는 상황이 왔어요.",
                    "{quiz}에서 {character}의 실력이 시험받았습니다.",
                    "중요한 순간에 {character}는 배운 것을 잊어버렸어요."
                ],
                resolution_templates=[
                    "차근차근 생각해보니 답을 찾을 수 있었어요.",
                    "친구들과 함께 협력해서 문제를 해결했습니다.",
                    "{character}는 {skill}을 통해 성공했어요."
                ],
                ending_templates=[
                    "이제 {character}는 {subject}를 좋아하게 되었어요.",
                    "배운 것을 다른 친구들에게도 알려주었습니다.",
                    "공부하는 즐거움을 알게 되었답니다."
                ],
                moral_lessons=[
                    "꾸준히 노력하면 무엇이든 배울 수 있어요",
                    "호기심을 가지고 질문하는 것이 중요해요",
                    "배운 것을 나누면 더 기뻐요"
                ],
                genre=StoryGenre.EDUCATIONAL,
                age_group=AgeGroup.YOUNG,
                cultural_elements=["한국의 교육", "전통 놀이를 통한 학습", "가족의 지혜"],
                vocabulary_level="educational",
                sentence_complexity="instructive"
            )
        ]
    
    def _create_traditional_templates(self) -> List[StoryTemplate]:
        """Create traditional Korean story templates"""
        return [
            StoryTemplate(
                title_template="옛날이야기: {traditional_character}",
                opening_templates=[
                    "아주 옛날 {dynasty}시대에 {character}라는 사람이 살았습니다.",
                    "옛날 어느 마을에 {character_description} {character}가 있었어요.",
                    "전설에 따르면 {mountain}에 {mystical_being}이 살고 있었다고 해요."
                ],
                development_templates=[
                    "{character}는 {virtue}로 유명했지만 {challenge}에 부딪혔습니다.",
                    "어느 날 {supernatural_event}가 일어나서 마을이 {crisis}에 빠졌어요.",
                    "{wise_elder}가 {character}에게 {ancient_wisdom}을 가르쳐주었습니다."
                ],
                climax_templates=[
                    "{character}는 {evil_spirit}와 맞서 싸워야 했습니다.",
                    "하늘의 시험을 받는 순간이 왔어요.",
                    "{sacrifice}를 해야만 사람들을 구할 수 있었습니다."
                ],
                resolution_templates=[
                    "{character}의 {virtue}이 {miracle}을 일으켰습니다.",
                    "하늘이 {character}의 마음을 알아주었어요.",
                    "{traditional_value}의 힘으로 모든 것이 해결되었습니다."
                ],
                ending_templates=[
                    "그래서 사람들은 지금도 {character}를 기억하고 있어요.",
                    "이 이야기는 {moral}의 소중함을 전해줍니다.",
                    "{place}에는 아직도 {character}의 흔적이 남아있다고 해요."
                ],
                moral_lessons=[
                    "효도와 존경하는 마음이 중요해요",
                    "정직하고 성실한 마음을 가져요",
                    "자연과 조상을 공경해요"
                ],
                genre=StoryGenre.TRADITIONAL,
                age_group=AgeGroup.OLD,
                cultural_elements=["한국의 전통", "효와 충", "자연 숭배", "조상 공경"],
                vocabulary_level="traditional",
                sentence_complexity="classical"
            )
        ]
    
    def get_template(
        self, 
        genre: StoryGenre, 
        age_group: AgeGroup, 
        specific_template: Optional[str] = None
    ) -> Optional[StoryTemplate]:
        """
        Get story template by genre and age group
        
        Args:
            genre: Story genre
            age_group: Target age group
            specific_template: Specific template name (optional)
            
        Returns:
            Optional[StoryTemplate]: Selected template
        """
        template_key = f"{age_group.value.replace('-', '_')}_{genre.value}".upper()
        
        if template_key in self.templates:
            templates = self.templates[template_key]
            return random.choice(templates) if templates else None
        
        # Fallback to generic templates
        for key, templates in self.templates.items():
            if genre.value in key.lower() or age_group.value.replace('-', '_') in key:
                return random.choice(templates) if templates else None
        
        return None
    
    def generate_story_structure(
        self, 
        template: StoryTemplate, 
        story_elements: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate story structure using template and story elements
        
        Args:
            template: Story template to use
            story_elements: Collected story elements
            
        Returns:
            Dict[str, str]: Generated story structure
        """
        structure = {}
        
        # Select random templates for each section
        structure["title"] = self._fill_template(
            random.choice([template.title_template]),
            story_elements
        )
        
        structure["opening"] = self._fill_template(
            random.choice(template.opening_templates),
            story_elements
        )
        
        structure["development"] = self._fill_template(
            random.choice(template.development_templates),
            story_elements
        )
        
        structure["climax"] = self._fill_template(
            random.choice(template.climax_templates),
            story_elements
        )
        
        structure["resolution"] = self._fill_template(
            random.choice(template.resolution_templates),
            story_elements
        )
        
        structure["ending"] = self._fill_template(
            random.choice(template.ending_templates),
            story_elements
        )
        
        structure["moral"] = random.choice(template.moral_lessons)
        structure["cultural_elements"] = template.cultural_elements
        
        return structure
    
    def _fill_template(self, template: str, elements: Dict[str, Any]) -> str:
        """
        Fill template with story elements
        
        Args:
            template: Template string with placeholders
            elements: Story elements to fill
            
        Returns:
            str: Filled template
        """
        filled = template
        
        # Default fallbacks for common placeholders
        defaults = {
            "character": "주인공",
            "place": "마법의 숲",
            "friend": "새로운 친구",
            "problem": "어려운 문제",
            "solution": "지혜로운 방법",
            "skill": "특별한 능력",
            "positive_trait": "용기와 친절함"
        }
        
        # Replace placeholders with story elements or defaults
        for key, value in elements.items():
            placeholder = "{" + key + "}"
            if placeholder in filled:
                if isinstance(value, list) and value:
                    filled = filled.replace(placeholder, random.choice(value))
                elif isinstance(value, str):
                    filled = filled.replace(placeholder, value)
        
        # Fill remaining placeholders with defaults
        for key, default in defaults.items():
            placeholder = "{" + key + "}"
            if placeholder in filled:
                filled = filled.replace(placeholder, default)
        
        return filled
    
    def get_available_genres(self, age_group: AgeGroup) -> List[StoryGenre]:
        """Get available genres for age group"""
        if age_group == AgeGroup.YOUNG:
            return [StoryGenre.ADVENTURE, StoryGenre.FRIENDSHIP, StoryGenre.ANIMAL, 
                   StoryGenre.EDUCATIONAL]
        else:
            return [StoryGenre.ADVENTURE, StoryGenre.FRIENDSHIP, StoryGenre.FANTASY,
                   StoryGenre.TRADITIONAL, StoryGenre.EDUCATIONAL]

# Global template manager instance
_template_manager = None

def get_template_manager() -> StoryTemplateManager:
    """Get global template manager instance"""
    global _template_manager
    if _template_manager is None:
        _template_manager = StoryTemplateManager()
    return _template_manager