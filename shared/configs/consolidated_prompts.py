"""
CCB_AI Consolidated Prompt Configuration

모든 프롬프트를 중앙화하여 중복을 제거하고 일관성을 보장합니다.
"""

from typing import Dict, List, Any
from shared.utils.age_group_utils import AgeGroup

class ConsolidatedPrompts:
    """통합된 프롬프트 관리자"""
    
    # 시스템 메시지 템플릿
    SYSTEM_MESSAGES = {
        "chatbot_a": {
            "base": "너는 부기라는 이름을 가진 친근한 AI 친구야. 아이들과 대화하며 재미있는 동화 만들기를 도와주는 역할을 해.",
            "conversation": "당신은 {age_group}세 아이 {child_name}와 대화하는 친근한 AI 친구 {chatbot_name}입니다.",
            "story_collection": """당신은 {age_group}세 아이 {child_name}와 대화하는 친근한 AI 친구 {chatbot_name}입니다.

주요 역할:
1. 아이와 자연스럽고 재미있는 대화를 나누기
2. 아이의 상상력을 자극하는 질문하기
3. 동화 만들기에 필요한 요소들을 수집하기
4. 아이의 관심사와 선호도 파악하기

대화 방식:
- 아이의 연령({age_group}세)에 맞는 쉬운 말을 사용하고, 재미있고 상상력을 자극하는 방식으로 접근하세요.
- 아이가 흥미를 잃지 않도록 짧고 명확한 질문을 하세요.
- 아이의 답변에 대해 긍정적으로 반응하고 격려해주세요.
- 동화에 필요한 등장인물, 배경, 문제, 해결방법 등을 자연스럽게 수집하세요."""
        },
        "chatbot_b": {
            "base": "너는 꼬기라는 이름을 가진 챗봇이야. 너는 부기(chatbot_a)와 대화를 하면서 일본 애니메이션 스타일의 카와이 동화를 만들어줄거야.",
            "story_generation": """당신은 아이들을 위한 동화를 작성하는 전문 동화 작가입니다.
다음 요소를 활용하여 {target_age}세 어린이를 위한 자세한 챕터별 동화를 작성해주세요.
작품은 5개의 챕터로 구성되어야 합니다."""
        }
    }
    
    # 연령대별 언어 설정
    AGE_APPROPRIATE_LANGUAGE = {
        AgeGroup.YOUNG_CHILDREN: {
            "vocabulary": "basic",
            "sentence_length": "short",
            "use_repetition": True,
            "use_sound_words": True,
            "emotional_complexity": "basic",
            "concepts_per_scene": 2
        },
        AgeGroup.ELEMENTARY: {
            "vocabulary": "intermediate",
            "sentence_length": "medium",
            "use_repetition": False,
            "use_sound_words": False,
            "emotional_complexity": "moderate",
            "concepts_per_scene": 4
        }
    }
    
    # 이야기 수집 프롬프트
    STORY_COLLECTION_PROMPTS = {
        "character": {
            AgeGroup.YOUNG_CHILDREN: [
                "{name}아/야, 어떤 친구가 나오는 이야기를 만들어볼까?",
                "재미있는 동물 친구나 사람 친구 중에 누가 나왔으면 좋겠어?",
                "{name}이/가 좋아하는 캐릭터가 있어? 그런 친구가 나오면 어떨까?"
            ],
            AgeGroup.ELEMENTARY: [
                "{name}아/야, 우리 이야기의 주인공은 어떤 친구였으면 좋겠어?",
                "어떤 성격을 가진 캐릭터가 나오면 재미있을까?",
                "주인공에게는 어떤 특별한 능력이나 특징이 있으면 좋을까?"
            ]
        },
        "setting": {
            AgeGroup.YOUNG_CHILDREN: [
                "그 친구들이 어디에서 살고 있을까?",
                "어떤 재미있는 곳에서 모험을 하면 좋을까?",
                "{name}이/가 가보고 싶은 신기한 곳이 있어?"
            ],
            AgeGroup.ELEMENTARY: [
                "이 이야기는 어떤 배경에서 일어나면 좋을까?",
                "주인공이 살고 있는 세계는 어떤 특별한 점이 있을까?",
                "어떤 장소에서 가장 흥미진진한 모험이 펼쳐질 수 있을까?"
            ]
        },
        "problem": {
            AgeGroup.YOUNG_CHILDREN: [
                "그런 곳에서 어떤 문제가 생길 수 있을까?",
                "우리 친구들에게 어떤 어려움이 찾아올까?",
                "무엇 때문에 주인공이 고민하게 될까?"
            ],
            AgeGroup.ELEMENTARY: [
                "이야기에서 주인공이 해결해야 할 문제는 무엇일까?",
                "어떤 갈등이나 위기가 생기면 이야기가 더 흥미로워질까?",
                "주인공이 가장 어려워할 만한 상황은 어떤 것일까?"
            ]
        },
        "resolution": {
            AgeGroup.YOUNG_CHILDREN: [
                "그 문제를 어떻게 해결하면 좋을까?",
                "우리 친구들이 어떻게 도와줄 수 있을까?",
                "어떻게 하면 모든 친구들이 행복해질 수 있을까?"
            ],
            AgeGroup.ELEMENTARY: [
                "주인공은 이 문제를 어떻게 해결할 수 있을까?",
                "어떤 방법으로 갈등을 해결하면 가장 의미 있을까?",
                "이야기의 결말에서 어떤 교훈을 얻을 수 있을까?"
            ]
        }
    }
    
    # 격려 문구
    ENCOURAGEMENT_PHRASES = {
        AgeGroup.YOUNG_CHILDREN: [
            "와! 정말 좋은 생각이야!",
            "{name}아/야, 너무 재미있어!",
            "멋진 아이디어네!",
            "상상력이 정말 대단해!",
            "더 이야기해줘!"
        ],
        AgeGroup.ELEMENTARY: [
            "정말 창의적인 생각이네!",
            "{name}의 상상력이 놀라워!",
            "흥미진진한 이야기가 되겠어!",
            "훌륭한 아이디어야!",
            "계속 들려줘, 너무 재미있어!"
        ]
    }
    
    # 후속 질문
    FOLLOW_UP_QUESTIONS = {
        AgeGroup.YOUNG_CHILDREN: [
            "그래서 어떻게 됐을까?",
            "그 다음에는 뭐가 일어났어?",
            "더 자세히 말해줄래?",
            "그때 기분이 어땠을까?"
        ],
        AgeGroup.ELEMENTARY: [
            "그 상황에서 어떤 일이 벌어졌을까?",
            "그때 등장인물들은 어떤 생각을 했을까?",
            "더 구체적으로 설명해줄 수 있어?",
            "그 결정이 이야기에 어떤 영향을 미쳤을까?"
        ]
    }
    
    # 이미지 생성 프롬프트
    IMAGE_GENERATION_PROMPTS = {
        AgeGroup.YOUNG_CHILDREN: """Cozy children's storybook illustration: {scene_description}

캐릭터: {characters}  
배경: {setting}

Hand-sketched with colored pencils and soft watercolor, like a beloved worn picture book found in grandmother's attic. Gentle pencil lines still visible, slightly uneven coloring that shows the loving human touch. Warm cream paper texture, soft rounded edges, delightfully imperfect. 

Colors are muted pastels like faded photographs - dusty pinks, sage greens, butter yellows, powder blues. The kind of gentle tones that feel safe and sleepy.

Simple composition that doesn't overwhelm little eyes, with the main character clearly the hero of their own small adventure. Cozy details like scattered flower petals, wooden toys, or soft fabric textures.""",
        
        AgeGroup.ELEMENTARY: """Beautiful detailed storybook illustration: {scene_description}

캐릭터: {characters}
배경: {setting}

Hand-drawn like a treasured children's book from a cozy library corner. Colored pencil sketches with delicate watercolor washes, showing the artist's gentle touch in every stroke. Visible paper grain and subtle artistic imperfections that make it feel real and loved.

Rich but muted earth tones - warm ochre, dusty rose, faded sage, cream whites, soft browns. The colors of autumn leaves and old quilts, nostalgic and timeless.

Detailed enough to reward close looking, with hidden treasures tucked into corners - tiny flowers, sleeping cats, forgotten books. The kind of illustration that has been lovingly pored over by generations of children, each discovering something new."""
    }
    
    # 내레이션 생성 프롬프트
    NARRATION_PROMPTS = {
        AgeGroup.YOUNG_CHILDREN: """다음 동화 장면을 4-7세 아이들을 위한 내레이션으로 작성해주세요.

장면: {scene_description}

다음 특징을 포함해주세요:
- 매우 짧고 단순한 문장 사용
- 이해하기 쉬운 기본 어휘만 사용
- 리듬감 있고 반복되는 표현 활용
- 의성어와 의태어 풍부하게 사용
- 질문 형식으로 아이의 참여 유도
- 천천히 읽히도록 문장 구성
- 밝고 따뜻한 톤 유지""",
        
        AgeGroup.ELEMENTARY: """다음 동화 장면을 8-9세 아이들을 위한 내레이션으로 작성해주세요.

장면: {scene_description}

다음 특징을 포함해주세요:
- 다양한 문장 구조와 길이 사용
- 풍부한 어휘와 표현 도입
- 감정과 분위기를 생생하게 묘사
- 창의적인 비유와 묘사적 언어 사용
- 이야기의 주제와 메시지를 미묘하게 강화
- 캐릭터의 내면과 생각을 더 깊이 탐색
- 상황에 따라 적절한 톤 변화"""
    }
    
    @classmethod
    def get_system_message(cls, chatbot_type: str, message_type: str = "base", **kwargs) -> str:
        """시스템 메시지 반환"""
        template = cls.SYSTEM_MESSAGES.get(chatbot_type, {}).get(message_type, "")
        return template.format(**kwargs) if kwargs else template
    
    @classmethod
    def get_story_collection_prompt(cls, element_type: str, age_group: AgeGroup) -> List[str]:
        """이야기 수집 프롬프트 반환"""
        return cls.STORY_COLLECTION_PROMPTS.get(element_type, {}).get(age_group, [])
    
    @classmethod
    def get_encouragement(cls, age_group: AgeGroup) -> List[str]:
        """격려 문구 반환"""
        return cls.ENCOURAGEMENT_PHRASES.get(age_group, [])
    
    @classmethod
    def get_follow_up_questions(cls, age_group: AgeGroup) -> List[str]:
        """후속 질문 반환"""
        return cls.FOLLOW_UP_QUESTIONS.get(age_group, [])
    
    @classmethod
    def get_image_prompt(cls, age_group: AgeGroup, **kwargs) -> str:
        """이미지 생성 프롬프트 반환"""
        template = cls.IMAGE_GENERATION_PROMPTS.get(age_group, "")
        return template.format(**kwargs) if kwargs else template
    
    @classmethod
    def get_narration_prompt(cls, age_group: AgeGroup, **kwargs) -> str:
        """내레이션 프롬프트 반환"""
        template = cls.NARRATION_PROMPTS.get(age_group, "")
        return template.format(**kwargs) if kwargs else template
    
    @classmethod
    def get_age_appropriate_language(cls, age_group: AgeGroup) -> Dict[str, Any]:
        """연령대별 언어 설정 반환"""
        return cls.AGE_APPROPRIATE_LANGUAGE.get(age_group, {}) 