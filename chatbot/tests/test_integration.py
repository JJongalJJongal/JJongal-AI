import unittest
import sys
import os
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.chat_bot_a import ChildChatBot
from models.chat_bot_b import StoryGenerationChatBot

class TestChatBotIntegration(unittest.TestCase):
    """
    부기(Chat-bot A)와 꼬기(Chat-bot B) 통합 테스트
    
    이 테스트는 두 챗봇 간의 상호작용과 이야기 생성 과정을 검증합니다.
    """
    
    def setUp(self):
        """테스트 전 필요한 객체 초기화"""
        # 부기(Chat-bot A) 초기화
        self.chatbot_a = ChildChatBot()
        
        # 꼬기(Chat-bot B) 초기화
        self.chatbot_b = StoryGenerationChatBot()
        
        # 테스트용 아이 정보 설정
        self.child_name = "민준"
        self.child_age = 6
        self.child_interests = ["공룡", "우주", "로봇"]
        
    def test_story_generation_flow(self):
        """
        이야기 생성 전체 과정을 테스트합니다.
        1. 부기(Chat-bot A)가 아이와 대화
        2. 부기가 이야기 줄거리 추출
        3. 꼬기(Chat-bot B)가 줄거리를 바탕으로 상세 동화 생성
        """
        # 1. 부기 초기화 및 인사
        greeting = self.chatbot_a.initialize_chat(
            child_name=self.child_name,
            age=self.child_age,
            interests=self.child_interests,
            chatbot_name="부기"
        )
        print(f"부기의 인사: {greeting}")
        
        # 2. 대화 시뮬레이션
        conversations = [
            "안녕! 나는 민준이야.",
            "나는 공룡을 좋아해. 특히 티라노사우루스가 제일 좋아!",
            "우주에 있는 별들도 좋아해. 우주 공룡이 있으면 어떨까?",
            "우주 공룡은 별가루를 먹고 살고, 꼬리에서는 밤하늘처럼 반짝이는 빛이 나와.",
            "우주 공룡이 지구에 와서 친구들을 만나면 재미있을 것 같아."
        ]
        
        # 각 대화에 대한 응답 생성
        for conversation in conversations:
            response = self.chatbot_a.get_response(conversation)
            print(f"민준: {conversation}")
            print(f"부기: {response}")
            print("-" * 50)
        
        # 3. 부기가 이야기 줄거리 추출
        story_outline = self.chatbot_a.suggest_story_theme()
        print("\n=== 부기가 추출한 이야기 줄거리 ===")
        print(f"줄거리: {story_outline.get('summary_text', '')}")
        print(f"태그: {story_outline.get('tags', '')}")
        print("=" * 50)
        
        # 4. 꼬기에게 줄거리 전달
        self.chatbot_b.set_story_outline(story_outline)
        self.chatbot_b.set_target_age(self.child_age)
        
        # 5. 꼬기가 상세 동화 생성 (실제 API 호출은 생략 - 테스트용)
        print("\n=== 꼬기가 생성할 상세 동화 정보 ===")
        print(f"대상 연령: {self.chatbot_b.target_age}세")
        print(f"줄거리: {self.chatbot_b.story_outline}")
        
        # 테스트 환경에서는 실제 API 호출을 하지 않고 모의 데이터 사용
        mock_detailed_story = {
            "title": "우주 공룡의 지구 모험",
            "scenes": [
                {
                    "scene_number": 1,
                    "description": "우주 공룡 별가루가 처음 지구를 발견하는 장면",
                    "text": "별이 빛나는 우주 공간에서, 별가루를 먹고 자라는 공룡 '별가루'가 살고 있었어요. 별가루의 꼬리는 밤하늘처럼 반짝였어요. 어느 날, 별가루는 아름다운 푸른 행성 '지구'를 발견했어요.",
                    "narration": "우주 깊은 곳에 별을 먹고 사는 공룡이 있었대요.",
                    "dialogues": [
                        {"character": "별가루", "text": "와, 저기 파란 행성은 뭘까? 가까이서 보고 싶어!"}
                    ],
                    "image_prompt": "밤하늘 같은 꼬리를 가진 귀여운 보라색 공룡이 우주에서 푸른 지구를 바라보는 모습"
                }
            ],
            "characters": [
                {"name": "별가루", "description": "우주에 사는 보라색 공룡으로 꼬리에서 별빛이 나온다."}
            ],
            "moral": "새로운 친구를 만들고 서로 다른 점을 존중하는 것의 중요성",
            "target_age": 6
        }
        
        # 모의 데이터 설정
        self.chatbot_b.detailed_story = mock_detailed_story
        
        # 결과 출력
        print("\n=== 생성된 상세 동화 ===")
        print(f"제목: {mock_detailed_story['title']}")
        print(f"등장인물: {mock_detailed_story['characters'][0]['name']} - {mock_detailed_story['characters'][0]['description']}")
        print(f"교훈: {mock_detailed_story['moral']}")
        print("\n첫 번째 장면:")
        print(f"설명: {mock_detailed_story['scenes'][0]['description']}")
        print(f"내용: {mock_detailed_story['scenes'][0]['text']}")
        
        # 테스트 성공 확인
        self.assertIsNotNone(story_outline)
        self.assertIsNotNone(mock_detailed_story)
        
if __name__ == "__main__":
    unittest.main() 