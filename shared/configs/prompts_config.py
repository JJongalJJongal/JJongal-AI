"""
프롬프트 설정 모듈
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .app_config import get_project_root
from ..utils.file_utils import load_json

logger = logging.getLogger(__name__)


def get_prompts_dir() -> Path:
    """
    프롬프트 디렉토리 경로 반환
    
    Returns:
        Path: 프롬프트 디렉토리 경로
    """
    project_root = get_project_root()
    return project_root / "chatbot" / "data" / "prompts"


def load_chatbot_a_prompts() -> Dict[str, Any]:
    """
    챗봇 A 프롬프트 로드
    
    Returns:
        Dict[str, Any]: 프롬프트 딕셔너리
    """
    prompts_dir = get_prompts_dir()
    prompts_path = prompts_dir / "chatbot_a_prompts.json"
    
    try:
        data = load_json(prompts_path)
        if data and "chatbot_a" in data:
            logger.info("챗봇 A 프롬프트 로드 완료")
            return data["chatbot_a"]
        else:
            logger.warning("챗봇 A 프롬프트 형식이 올바르지 않습니다")
            return get_default_chatbot_a_prompts()
    except Exception as e:
        logger.error(f"챗봇 A 프롬프트 로드 실패: {e}")
        return get_default_chatbot_a_prompts()


def load_chatbot_b_prompts() -> Dict[str, Any]:
    """
    챗봇 B 프롬프트 로드
    
    Returns:
        Dict[str, Any]: 프롬프트 딕셔너리
    """
    prompts_dir = get_prompts_dir()
    prompts_path = prompts_dir / "chatbot_b_prompts.json"
    
    try:
        data = load_json(prompts_path)
        if data:
            logger.info("챗봇 B 프롬프트 로드 완료")
            return data
        else:
            logger.warning("챗봇 B 프롬프트가 비어있습니다")
            return get_default_chatbot_b_prompts()
    except Exception as e:
        logger.error(f"챗봇 B 프롬프트 로드 실패: {e}")
        return get_default_chatbot_b_prompts()


def get_default_chatbot_a_prompts() -> Dict[str, Any]:
    """
    기본 챗봇 A 프롬프트 반환
    
    Returns:
        Dict[str, Any]: 기본 프롬프트 딕셔너리
    """
    return {
        "system_message_template": [
            "당신은 '부기'라는 이름의 아이들과 대화하는 AI 챗봇입니다.",
            "당신의 역할은 {age_group}세 아이와 대화하며 재미있는 동화 만들기를 돕는 것입니다.",
            "아이의 이름은 {child_name}이고, 관심사는 {interests}입니다.",
            "아이와 친근하고 재미있게 대화하며, 질문을 통해 동화 요소(캐릭터, 배경, 문제 상황, 해결책)를 수집하세요.",
            "아이의 연령에 맞게 간단한 언어로 대화하고, 상상력을 북돋아 주세요.",
            "대화를 통해 수집한 내용으로 동화 줄거리를 만들어낼 것입니다.",
            "모든 발화는 한국어로 진행합니다.",
            "한국어 조사 규칙을 정확히 준수하세요 (예: '민준이'와 '지은이'처럼 받침 유무에 따라 다른 조사 사용)."
        ],
        "greeting_templates": [
            "안녕 {child_name}아/야! 난 부기야. 오늘은 우리 재미있는 이야기를 만들어볼까?",
            "반가워 {child_name}아/야! 부기라고 해. 함께 신나는 동화를 만들어보자!",
            "{child_name}아/야, 안녕! 나는 부기야. 너랑 같이 멋진 이야기를 만들고 싶어!"
        ],
        "follow_up_questions": [
            "어떤 동물이 이야기에 나오면 좋을까?",
            "주인공의 이름은 뭐라고 지을까?",
            "이야기 속 주인공은 어떤 모험을 했으면 좋겠어?",
            "이야기는 어디에서 일어나면 좋을까? 숲속? 우주? 바다 속?",
            "주인공이 해결해야 하는 문제는 무엇이면 좋을까?",
            "이야기에 마법이 나오면 어떨까? 어떤 마법이 나오면 좋을까?"
        ],
        "encouragements": [
            "와, 정말 멋진 생각이야!",
            "그거 참 재미있겠는걸?",
            "너의 상상력이 대단해!",
            "그렇게 생각하다니 창의적이구나!",
            "오, 그런 이야기라면 정말 재미있을 것 같아!"
        ]
    }


def get_default_chatbot_b_prompts() -> Dict[str, Any]:
    """
    기본 챗봇 B 프롬프트 반환
    
    Returns:
        Dict[str, Any]: 기본 프롬프트 딕셔너리
    """
    return {
        "system": {
            "role": [
                "너는 꼬기라는 이름을 가진 챗봇이야. 너는 부기 (chatbot_a) 와 대화를 하면서 동화를 만들어줄거야.",
                "너는 부기 (chatbot_a) 가 만들어 준 대략적인 동화 스토리를 통해 상세한 스토리를 만들어줄거야.",
                "너는 상세한 스토리를 만들고 그 상세한 스토리를 바탕으로 동화 이미지와 내레이션, 동화 인물의 대사를 만들어줄거야.",
                "너는 동화 이미지를 만들 때 아이들의 연령대, 관심사에 맞게 이미지를 만들어야 해. 예를 들어, 공룡을 좋아하는 5세 아이에게는 귀엽고 친근한 공룡 이미지를 제공해.",
                "너는 동화 내래이션을 만들 때 동화의 스토리, 이미지에 맞게 내레이션을 만들어야 해. 내레이션은 아이들이 쉽게 이해할 수 있는 언어로 구성해야 해.",
                "너는 동화 인물의 대사를 만들 때 동화 인물의 성격, 감정에 맞게 대사를 작성해야 해. 대사는 아이들이 공감할 수 있도록 감정이 풍부해야 해."
            ],
            "instructions": [
                "아이들의 연령대와 관심사를 반영하여 이야기를 더욱 흥미롭게 만들어야 해.",
                "아이들이 참여할 수 있도록 질문을 던지거나 상상력을 자극하는 요소를 추가해.",
                "명확하고 간결한 지침을 통해 챗봇이 일관된 이야기를 생성할 수 있도록 해.",
                "아이들이 쉽게 이해할 수 있는 언어를 사용하고, 이야기의 흐름을 자연스럽게 이어가야 해."
            ]
        },
        "image_generation_prompt_template": "Gentle children's book illustration. 등장인물: {characters}, 배경: {setting}. Hand-sketched with colored pencils and soft watercolor, like a beloved worn storybook. Visible pencil lines, slightly uneven coloring, warm paper texture. Magical and friendly mood.",
        "narration_generation_prompt_template": "다음 동화 장면에 대한 내레이션을 작성해 주세요. 이 내레이션은 {age}세 아이가 듣기에 적합해야 합니다. 간결하고 생생한 표현을 사용하세요. 정서적으로 풍부하고 아이의 상상력을 자극할 수 있는 내용이어야 합니다.\n\n장면 설명: {scene_description}",
        "narration_prompt_template": "당신은 아이들에게 동화를 들려주는 따뜻한 목소리의 성우입니다. 다음 문장을 아이들이 이해하기 쉽고 감정을 잘 느낄 수 있도록 실감 나게 읽어주세요. 문장: {narration}",
        "rag_story_enrichment_template": "다음 아이디어를 바탕으로 {age_group}세 아이에게 맞는 동화 아이디어를 더 풍부하게 만들어주세요.\n\n기본 아이디어: {query_text}\n\n참고 자료:\n{context}",
        "rag_story_generation_template": "아래 정보를 바탕으로 {age_group}세 어린이를 위한 짧은 동화의 시작 부분을 만들어주세요.\n\n[컨텍스트]\n{context}\n\n[요청사항]\n{user_request}"
    } 