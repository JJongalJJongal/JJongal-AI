"""
동화 줄거리를 바탕으로 일러스트와 내레이션을 생성하는 AI 챗봇 모듈
"""

# 자체 개발 모듈 임포트
from .story_generation_chatbot import StoryGenerationChatBot

# StoryGenerationChatBot 클래스를 그대로 export
__all__ = ['StoryGenerationChatBot'] 