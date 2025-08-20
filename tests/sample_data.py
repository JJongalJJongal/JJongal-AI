import uuid
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock

# API v2.0 호환 샘플 데이터
SAMPLE_STORY_REQUEST = {
    "child_name": "민준",
    "age": 6,
    "interests": ["동물", "모험"],
    "story_elements": {
        "main_character": "토끼 민준",
        "setting": "숲속 마을",
        "theme": "용기와 친구",
    },
    "voice_config": {
        "child_voice_id": "test_child_voice_123",
        "narrator_voice": "default",
        "parent_voice_id": "test_parent_voice_456",
    },
}

SAMPLE_STORY_DATA = {
    "title": "토끼 민준의 숲속 모험",
    "chapters": [
        {
            "chapter_number": 1,
            "title": "새로운 친구",
            "content": "토끼 민준이 숲에서 곰 친구를 만났어요. '안녕!' 민준이 인사했어요.",
            "image_prompt": "A cute rabbit meeting a friendly bear in a peaceful forest",
        },
        {
            "chapter_number": 2,
            "title": "함께하는 모험",
            "content": "민준과 곰은 함께 숲을 탐험했어요. 예쁜 꽃들과 나비들을 발견했어요.",
            "image_prompt": "A rabbit and bear exploring a beautiful forest with flowers and butterflies",
        },
    ],
}

SAMPLE_VOICE_CONFIG = {
    "child_voice_id": "test_child_voice_123",
    "narrator_voice": "default_narrator",
    "parent_voice_id": "test_parent_voice_456",
}

SAMPLE_IMAGE_GENERATION_INPUT = {
    "story_data": SAMPLE_STORY_DATA,
    "story_id": "test_story_123",
    "target_age": 6,
    "interests": ["동물", "모험"],
}

SAMPLE_VOICE_GENERATION_INPUT = {
    "story_data": SAMPLE_STORY_DATA,
    "voice_config": SAMPLE_VOICE_CONFIG,
    "target_age": 6,
    "story_id": "test_story_123",
}

# Mock 응답 데이터
MOCK_DALL_E_RESPONSE = "https://fake-dalle-image-url.com/image123.png"

MOCK_ELEVENLABS_AUDIO = b"fake_audio_data_bytes_for_testing"

MOCK_IMAGE_GENERATION_RESULT = {
    "images": [
        {
            "chapter_number": 1,
            "url": "https://fake-image-url.com/chapter1.png",
            "image_path": "/fake/path/chapter_1_test123.png",
            "image_prompt": "A cute rabbit meeting a friendly bear in a peaceful forest",
            "status": "success",
        },
        {
            "chapter_number": 2,
            "url": "https://fake-image-url.com/chapter2.png",
            "image_path": "/fake/path/chapter_2_test123.png",
            "image_prompt": "A rabbit and bear exploring a beautiful forest",
            "status": "success",
        },
    ],
    "metadata": {
        "total_images": 2,
        "successful_images": 2,
        "generation_time": 25.5,
        "model_used": "dall-e-3",
    },
}

MOCK_VOICE_GENERATION_RESULT = {
    "audio_files": [
        {
            "chapter_number": 1,
            "audio_path": "/fake/path/chapter_1_test123.mp3",
            "duration": 15.2,
            "voice_type": "narrator",
            "voice_id": "test_child_voice_123",
            "status": "success",
        },
        {
            "chapter_number": 2,
            "audio_path": "/fake/path/chapter_2_test123.mp3",
            "duration": 18.7,
            "voice_type": "narrator",
            "voice_id": "test_child_voice_123",
            "status": "success",
        },
    ],
    "metadata": {
        "total_duration": 33.9,
        "successful_audio": 2,
        "model_used": "eleven_multilingual_sts_v2",
        "generation_time": 8.3,
    },
}

# 쫑이(ChatBot A) 테스트 데이터
SAMPLE_CONVERSATION_DATA = {
    "conversation_summary": "아이가 토끼와 곰의 우정 이야기를 원함",
    "extracted_keywords": ["토끼", "곰", "숲", "친구", "모험"],
    "conversation_analysis": {
        "child_interests": ["동물", "자연"],
        "emotional_tone": "호기심 가득",
        "story_preferences": "동물 친구들",
    },
    "child_profile": {
        "name": "민준",
        "age": 6,
        "personality": "활발하고 상상력이 풍부함",
    },
}

SAMPLE_VOICE_SAMPLES = [
    {"sample_id": 1, "audio_data": b"sample1_audio_data", "duration": 3.2},
    {"sample_id": 2, "audio_data": b"sample2_audio_data", "duration": 2.8},
    {"sample_id": 3, "audio_data": b"sample3_audio_data", "duration": 4.1},
    {"sample_id": 4, "audio_data": b"sample4_audio_data", "duration": 3.5},
    {"sample_id": 5, "audio_data": b"sample5_audio_data", "duration": 3.9},
]

# 연령별 테스트 케이스
AGE_SPECIFIC_TEST_CASES = {
    "age_4_7": {
        "target_age": 5,
        "expected_voice_settings": {
            "stability": 0.7,
            "similarity_boost": 0.8,
            "style": 0.3,
            "use_speaker_boost": True,
        },
        "expected_image_style": "gentle watercolor",
        "content_complexity": "simple",
    },
    "age_8_9": {
        "target_age": 8,
        "expected_voice_settings": {
            "stability": 0.6,
            "similarity_boost": 0.9,
            "style": 0.5,
            "use_speaker_boost": True,
        },
        "expected_image_style": "detailed illustration",
        "content_complexity": "enriched",
    },
}

# 에러 시나리오 테스트 데이터
ERROR_SCENARIOS = {
    "empty_chapters": {
        "story_data": {"title": "빈 이야기", "chapters": []},
        "expected_error": "No chapters provided",
    },
    "missing_api_key": {"api_key": None, "expected_error": "API key not configured"},
    "invalid_voice_config": {
        "voice_config": {},
        "expected_fallback": "default_narrator_voice_id",
    },
}


def create_mock_openai_client():
    """OpenAI Client Mock 생성"""
    mock_client = MagicMock()
    mock_client.api_key = "test_openai_api_key"
    return mock_client


def create_mock_elevenlabs_client():
    """ElevenLabs Client Mock 생성"""
    mock_client = MagicMock()
    mock_convert = AsyncMock(return_value=MOCK_ELEVENLABS_AUDIO)
    mock_client.text_to_speech.convert = mock_convert
    return mock_client


def create_mock_dalle_tool():
    """DALL-E Tool Mock 생성"""
    mock_tool = MagicMock()
    mock_tool.run.return_value = MOCK_DALL_E_RESPONSE
    return mock_tool


def get_sample_story_request(age: int = 6, child_name: str = "민준") -> Dict[str, Any]:
    """API v2.0 StoryRequest 샘플 생성"""
    request = SAMPLE_STORY_REQUEST.copy()
    request["age"] = age
    request["child_name"] = child_name
    return request


def get_sample_story_data(chapter_count: int = 2) -> Dict[str, Any]:
    """Story 데이터 샘플 생성"""
    story = SAMPLE_STORY_DATA.copy()
    story["chapters"] = story["chapters"][:chapter_count]
    return story


def get_progress_callback_mock():
    """진행상황 콜백 Mock 생성"""
    return AsyncMock()


# 테스트 유틸리티 함수
def validate_api_v2_story_format(story_result: Dict[str, Any]) -> bool:
    """API v2.0 Story 형식 검증"""
    required_fields = ["story_id", "title", "status", "chapters", "created_at"]
    return all(field in story_result for field in required_fields)


def validate_image_generation_result(result: Dict[str, Any]) -> bool:
    """이미지 생성 결과 형식 검증"""
    if "images" not in result or "metadata" not in result:
        return False

    for image in result["images"]:
        required_fields = ["chapter_number", "status"]
        if not all(field in image for field in required_fields):
            return False

    return True


def validate_voice_generation_result(result: Dict[str, Any]) -> bool:
    """음성 생성 결과 형식 검증"""
    if "audio_files" not in result or "metadata" not in result:
        return False

    for audio in result["audio_files"]:
        required_fields = ["chapter_number", "status"]
        if not all(field in audio for field in required_fields):
            return False

    return True
