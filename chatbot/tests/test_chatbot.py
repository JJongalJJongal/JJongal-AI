from chat_bot import ChildChatBot
import argparse

def test_text_chat():
    """텍스트 모드로 챗봇 테스트"""
    chatbot = ChildChatBot()
    
    # 챗봇 초기화
    greeting = chatbot.initialize_chat(
        child_name="철수",
        age=5,
        interests=["공룡", "우주", "친구들"],
        chatbot_name="유미"
    )
    print(f"\n{greeting}\n")
    
    while True:
        user_input = input("사용자: ")
        if user_input.lower() in ["종료", "끝내기", "그만"]:
            print("\n대화를 종료합니다.")
            break
            
        response = chatbot.get_response(user_input)
        print(f"\n챗봇: {response}\n")
    
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