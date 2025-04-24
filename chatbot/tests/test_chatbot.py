import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.chat_bot_a import StoryCollectionChatBot
import argparse

def test_text_chat():
    """텍스트 모드로 챗봇 테스트"""
    chatbot = StoryCollectionChatBot()
    
    # 챗봇 초기화
    greeting = chatbot.initialize_chat(
        child_name="철수",
        age=5,
        interests=["공룡", "우주", "친구들"],
        chatbot_name="부기"
    )
    print(f"\n{greeting}\n")
    
    # 대화 횟수 제한
    max_conversations = 5
    conversation_count = 0
    
    while True:
        user_input = input("사용자: ")
        if user_input.lower() in ["종료", "끝내기", "그만"]:
            print("\n대화를 종료합니다.")
            break
            
        response = chatbot.get_response(user_input)
        print(f"\n챗봇: {response}\n")
        
        conversation_count += 1
        if conversation_count >= max_conversations:
            print("\n충분한 이야기가 수집되었습니다. 이야기를 정리해볼게요!")
            break
    
    # 수집된 대략적인 줄거리 및 태그 출력
    print("\n=== 수집된 정보 ===")
    try:
        story_info = chatbot.collect_story_outline()
        if story_info:
            print(f"줄거리 (Summary Text): {story_info.get('summary_text', '알 수 없음')}")
            print(f"태그 (Tags): {story_info.get('tags', '알 수 없음')}")
        else:
            print("이야기 정보(줄거리, 태그)를 생성할 수 없습니다.")
    except Exception as e:
        print(f"이야기 정보 수집 중 오류 발생: {str(e)}")
        print("기본 정보를 출력합니다.")
        print("줄거리: 주인공이 친구와 함께 모험을 하며 용기와 우정의 가치를 배우는 이야기입니다.")
        # 오류 발생 시 기본 태그 출력 (예시)
        interests_str = ",".join(chatbot.interests) if chatbot.interests else "모험,우정"
        print(f"태그: {interests_str}")
    
    # TODO: VectorDB 통합
    # - 대화 내용을 VectorDB에 저장
    # - Chroma/Pinecone/Faiss 중 선택하여 구현
    # - 임베딩 생성 및 저장 로직 추가

def main():
    test_text_chat()

if __name__ == "__main__":
    main() 