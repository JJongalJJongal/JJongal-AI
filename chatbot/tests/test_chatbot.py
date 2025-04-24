from chatbot.models.chat_bot import ChildChatBot
import argparse

def test_text_chat():
    """텍스트 모드로 챗봇 테스트"""
    chatbot = ChildChatBot()
    
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
        story_info = chatbot.suggest_story_theme()
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
    
    # 대화 내용 저장
    chatbot.save_conversation("data/chat/conversation.json")

def test_voice_chat():
    """음성 기반 대화 테스트"""
    # 챗봇 초기화
    chatbot = ChildChatBot()
    chatbot.initialize_chat(
        age_group=5,
        child_name="철수",
        chatbot_name="유미",
        interests=["공룡", "우주", "친구들"]
    )
    
    print("\n=== 음성 대화 테스트 시작 ===")
    print("음성 대화를 시작합니다. 마이크가 활성화되었습니다.")
    print("종료하려면 '종료' 또는 '끝내기'라고 말씀해주세요.")
    
    # 음성 대화 시작
    chatbot.start_voice_chat()
    
    # 대화 내용 저장
    chatbot.save_conversation("data/chat/voice_conversation.json")

def main():
    parser = argparse.ArgumentParser(description='챗봇 테스트')
    parser.add_argument('--mode', choices=['text', 'voice'], default='text',
                      help='대화 모드 선택 (text 또는 voice)')
    args = parser.parse_args()
    
    if args.mode == 'text':
        test_text_chat()
    else:
        test_voice_chat()

if __name__ == "__main__":
    main() 