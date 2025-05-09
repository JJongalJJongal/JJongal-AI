import os
import json
import argparse
from models.chat_bot_a import StoryCollectionChatBot
from models.chat_bot_b import StoryGenerationChatBot
from dotenv import load_dotenv
from pathlib import Path

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='꼬꼬북 AI 동화 생성 서비스')
    parser.add_argument('--mode', choices=['text', 'voice'], default='text',
                      help='대화 모드 선택 (text 또는 voice)')
    parser.add_argument('--child-name', type=str, default='아이',
                      help='아이의 이름')
    parser.add_argument('--age', type=int, default=5,
                      help='아이의 나이 (4-9세)')
    parser.add_argument('--interests', type=str, nargs='+', default=[],
                      help='아이의 관심사 목록')
    parser.add_argument('--output-dir', type=str, default='output',
                      help='생성된 파일을 저장할 디렉토리')
    args = parser.parse_args()
    
    # 환경 변수 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=dotenv_path)
    
    # 환경 변수 확인
    if not os.getenv('OPENAI_API_KEY'):
        print(f"Warning: OPENAI_API_KEY environment variable not found. Looking for .env file at: {dotenv_path}")
    
    # 출력 디렉토리 설정
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)
    
    # 챗봇 A 초기화 (이야기 수집)
    chatbot_a = StoryCollectionChatBot()
    
    # 챗봇 B 초기화 (이야기 생성)
    chatbot_b = StoryGenerationChatBot(output_dir=output_dir)
    
    try:
        # 1. 챗봇 A와 대화하여 이야기 수집
        print("\n=== 이야기 수집 시작 ===")
        
        # 챗봇 A 초기화
        greeting = chatbot_a.initialize_chat(
            child_name=args.child_name,
            age=args.age,
            interests=args.interests
        )
        print(f"\n{greeting}\n")
        
        # 대화 모드에 따라 다른 방식으로 대화 진행
        if args.mode == 'text':
            # 텍스트 모드로 대화
            while True:
                user_input = input("사용자: ")
                if user_input.lower() in ["종료", "끝내기", "그만"]:
                    print("\n대화를 종료합니다.")
                    break
                    
                response = chatbot_a.get_response(user_input)
                print(f"\n챗봇: {response}\n")
        else:
            # 음성 모드로 대화
            chatbot_a.start_voice_chat()
        
        # 대화 내용 저장
        chatbot_a.save_conversation(os.path.join(output_dir, "conversation.json"))
        
        # 2. 챗봇 A로부터 이야기 줄거리 수집
        story_outline = chatbot_a.collect_story_outline()
        
        # 수집된 이야기 저장
        with open(os.path.join(output_dir, "story_outline.json"), "w", encoding="utf-8") as f:
            json.dump(story_outline, f, ensure_ascii=False, indent=2)
        
        # 3. 챗봇 B에 이야기 전달
        print("\n=== 이야기 생성 시작 ===")
        chatbot_b.set_story_outline(story_outline)
        
        # 4. 이미지와 내레이션 생성
        images, narration = chatbot_b.generate_story()
        
        # 5. 생성된 이야기 미리보기 출력
        preview = chatbot_b.get_story_preview()
        print("\n=== 생성된 이야기 미리보기 ===")
        print(f"제목: {preview['title']}")
        print(f"요약: {preview['summary']}")
        print(f"이미지 수: {preview['image_count']}")
        print(f"예상 재생 시간: {preview['duration']}")
        
        # 6. 최종 데이터 저장
        chatbot_b.save_story_data(os.path.join(output_dir, "final_story.json"))
        
        print("\n=== 이야기 생성 완료 ===")
        print(f"생성된 파일은 {output_dir} 디렉토리에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
