#!/usr/bin/env python3
"""
부기(chat_bot_a)와 꼬기(chat_bot_b) 간의 동화 생성 연동 테스트

이 스크립트는 다음 과정을 테스트합니다:
1. 부기(chat_bot_a)를 사용하여 동화 줄거리 생성
2. 부기에서 생성된 줄거리를 꼬기(chat_bot_b)로 전달
3. 꼬기가 줄거리를 바탕으로 상세 스토리 생성
4. 꼬기가 상세 스토리를 바탕으로 이미지와 내레이션 생성
5. 생성된 동화의 미리보기 및 저장
"""

import os
import sys
import json
from pathlib import Path

# 프로젝트 루트 경로를 파이썬 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

from CCB_AI.chatbot.models.chat_bot_a import StoryCollectionChatBot
from CCB_AI.chatbot.models.chat_bot_b import StoryGenerationChatBot

def test_chatbot_integration():
    """
    부기와 꼬기의 연동 테스트 함수
    """
    print("=== 부기(chat_bot_a)와 꼬기(chat_bot_b) 연동 테스트 시작 ===")
    
    # 테스트 출력 디렉토리 생성
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 부기(chat_bot_a) 초기화 및 동화 줄거리 생성
    print("\n1. 부기(chat_bot_a) 초기화 및 동화 줄거리 생성")
    chatbot_a = StoryCollectionChatBot()
    
    # 아이 정보 설정
    child_name = "민준"
    age = 6
    interests = ["공룡", "우주", "로봇"]
    
    print(f"아이 정보: 이름={child_name}, 나이={age}세, 관심사={', '.join(interests)}")
    
    # 부기 초기화
    greeting = chatbot_a.initialize_chat(child_name, age, interests)
    print(f"부기 인사: {greeting}")
    
    # 테스트용 대화 시뮬레이션
    test_conversation = [
        "안녕! 나는 공룡이랑 우주 이야기를 좋아해!",
        "우주에 사는 공룡 이야기는 어때?",
        "용감한 우주 공룡이 다른 행성에서 온 친구들을 만나는 이야기가 좋을 것 같아!",
        "그 공룡은 파란색이고 날개도 있었으면 좋겠어!"
    ]
    
    # 대화 시뮬레이션
    for user_input in test_conversation:
        print(f"\n아이: {user_input}")
        response = chatbot_a.get_response(user_input)
        print(f"부기: {response}")
    
    # 2. 동화 줄거리 생성
    print("\n2. 동화 줄거리 생성")
    story_outline = chatbot_a.suggest_story_theme()
    
    print("생성된 동화 줄거리:")
    print(f"주제: {story_outline['theme']}")
    print(f"등장인물: {', '.join(story_outline['characters'])}")
    print(f"배경: {story_outline['setting']}")
    print(f"줄거리: {story_outline['plot_summary']}")
    print(f"교육적 가치: {story_outline['educational_value']}")
    
    # 3. 꼬기(chat_bot_b) 초기화 및 줄거리 전달
    print("\n3. 꼬기(chat_bot_b) 초기화 및 줄거리 전달")
    chatbot_b = StoryGenerationChatBot(output_dir=output_dir)
    chatbot_b.set_story_outline(story_outline)
    chatbot_b.set_target_age(age)
    
    # 4. 꼬기가 상세 스토리 생성
    print("\n4. 꼬기가 상세 스토리 생성")
    try:
        detailed_story = chatbot_b.generate_detailed_story()
        print(f"생성된 상세 스토리 제목: {detailed_story['title']}")
        print(f"장면 수: {len(detailed_story['scenes'])}")
        
        # 첫 번째 장면만 출력 (예시)
        first_scene = detailed_story['scenes'][0]
        print("\n첫 번째 장면 미리보기:")
        print(f"장면 제목: {first_scene['title']}")
        print(f"장면 설명: {first_scene['description'][:150]}...")
        if 'narration' in first_scene:
            print(f"내레이션: {first_scene['narration'][:150]}...")
        if 'dialogues' in first_scene and len(first_scene['dialogues']) > 0:
            dialogue = first_scene['dialogues'][0]
            print(f"대화 예시: {dialogue['character']}: {dialogue['text']}")
    except Exception as e:
        print(f"상세 스토리 생성 중 오류 발생: {str(e)}")
        return
    
    # 5. 이미지 생성 (옵션)
    print("\n5. 이미지 생성 (선택적 실행)")
    try_generate_image = input("이미지 생성을 시도하시겠습니까? (y/n): ").lower() == 'y'
    
    if try_generate_image:
        try:
            print("장면별 이미지 생성 중...")
            generated_images = chatbot_b.generate_illustrations()
            print(f"생성된 이미지 수: {len(generated_images)}")
            for i, img_path in enumerate(generated_images):
                print(f"이미지 {i+1}: {img_path}")
        except Exception as e:
            print(f"이미지 생성 중 오류 발생: {str(e)}")
    
    # 6. 내레이션 생성 (옵션)
    print("\n6. 내레이션 생성 (선택적 실행)")
    try_generate_voice = input("내레이션 생성을 시도하시겠습니까? (y/n): ").lower() == 'y'
    
    if try_generate_voice:
        try:
            print("장면별 내레이션 생성 중...")
            voice_result = chatbot_b.generate_voice()
            print("내레이션 생성 완료")
            
            # 내레이션 결과 요약
            if 'narration' in voice_result:
                print(f"내레이션 파일 수: {len(voice_result['narration'])}")
            if 'characters' in voice_result:
                print(f"캐릭터 음성 수: {len(voice_result['characters'])}")
        except Exception as e:
            print(f"내레이션 생성 중 오류 발생: {str(e)}")
    
    # 7. 미리보기 및 저장
    print("\n7. 동화 미리보기 및 저장")
    preview = chatbot_b.get_story_preview()
    print("동화 미리보기:")
    print(f"제목: {preview['title']}")
    print(f"요약: {preview['summary'][:200]}...")
    print(f"이미지 수: {preview['image_count']}")
    print(f"예상 재생 시간: {preview['duration']}")
    
    # 스토리 데이터 저장
    story_data_path = os.path.join(output_dir, "story_data.json")
    chatbot_b.save_story_data(story_data_path)
    print(f"스토리 데이터 저장 완료: {story_data_path}")
    
    print("\n=== 부기(chat_bot_a)와 꼬기(chat_bot_b) 연동 테스트 완료 ===")

if __name__ == "__main__":
    test_chatbot_integration() 