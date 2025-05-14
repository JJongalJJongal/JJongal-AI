import sys
import os
import asyncio
import websockets
import json
import base64
import argparse
import time
import io
import wave
import signal
from pathlib import Path
from threading import Thread

# 상위 디렉토리 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # chatbot 폴더
sys.path.append(parent_dir)

# 모듈 임포트
from models.chat_bot_a import StoryCollectionChatBot
from models.chat_bot_b import StoryGenerationChatBot
from models.voice_ws_server import app
import uvicorn

# 테스트용 오디오 파일 경로
SAMPLE_AUDIO_PATH = os.path.join(current_dir, "test_audio.wav")
# 응답 저장 디렉토리
RESPONSES_DIR = os.path.join(current_dir, "responses")

# 서버 실행을 위한 전역 변수 추가
server_process = None

def test_chatbot_basic():
    """챗봇 기본 기능 테스트"""
    print("\n=== 챗봇 기본 기능 테스트 ===")
    
    # 챗봇 인스턴스 생성
    chatbot = StoryCollectionChatBot()
    
    # 챗봇 초기화
    child_name = "테스트"
    age = 6
    interests = ["공룡", "우주", "로봇"]
    
    greeting = chatbot.initialize_chat(
        child_name=child_name,
        age=age,
        interests=interests,
        chatbot_name="부기"
    )
    
    print(f"인사말: {greeting}")
    print(f"아이 정보: 이름={child_name}, 나이={age}, 관심사={', '.join(interests)}")
    
    # 테스트 대화
    test_inputs = [
        "안녕! 나는 공룡을 좋아해",
        "티라노사우루스가 제일 멋있어",
        "내 이야기에는 용감한 아이가 나올 거야"
    ]
    
    for user_input in test_inputs:
        print(f"\n사용자: {user_input}")
        response = chatbot.get_response(user_input)
        print(f"챗봇: {response}")
    
    # 이야기 요약 테스트
    story = chatbot.suggest_story_theme()
    print("\n=== 수집된 이야기 테마 ===")
    print(f"주제: {story.get('theme', '')}")
    print(f"줄거리: {story.get('plot_summary', '')}")
    
    # 토큰 사용량 확인
    token_info = chatbot.get_token_usage()
    print("\n=== 토큰 사용량 ===")
    print(f"사용된 토큰: {token_info['token_usage']['total']}")
    print(f"남은 토큰: {token_info['remaining_tokens']}")

async def test_websocket_connection():
    """웹소켓 음성 기능 테스트"""
    print("\n=== 웹소켓 음성 기능 테스트 ===")
    uri = "ws://localhost:8000/ws/audio?token=valid_token&child_name=민준&age=5&interests=공룡,우주"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("웹소켓 연결 성공!")
            
            # 인사말 수신 (서버에서 자동으로 보내는 메시지)
            greeting_response = await websocket.recv()
            greeting_data = json.loads(greeting_response)
            print(f"인사말: {greeting_data.get('text', '')}")
            
            # 샘플 오디오 존재 확인
            if os.path.exists(SAMPLE_AUDIO_PATH):
                print(f"오디오 파일 크기: {os.path.getsize(SAMPLE_AUDIO_PATH)} 바이트")
                
                # 샘플 오디오 파일 전송
                with open(SAMPLE_AUDIO_PATH, "rb") as audio_file:
                    audio_data = audio_file.read()
                
                print(f"전송할 오디오 데이터 크기: {len(audio_data)} 바이트")
                
                # 오디오 데이터 전송
                await websocket.send(audio_data)
                print("샘플 오디오 전송 완료")
                
                # 응답 대기 (타임아웃 설정)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    response_data = json.loads(response)
                
                    # 응답 분석
                    print("\n서버 응답:")
                    print(f"응답 유형: {response_data.get('type', '')}")
                    print(f"AI 응답 텍스트: {response_data.get('text', '')}")
                    print(f"사용자 음성 인식: {response_data.get('user_text', '')}")
                    print(f"상태: {response_data.get('status', '')}")
                    
                    if "error_message" in response_data:
                        print(f"오류: {response_data.get('error_message')}")
                        if "error_code" in response_data:
                            print(f"오류 코드: {response_data.get('error_code')}")
                        
                    # 오디오 응답 저장
                    if "audio" in response_data and response_data["audio"]:
                        try:
                            # base64 디코딩
                            audio_data = base64.b64decode(response_data["audio"])
                            
                            # 응답 오디오 저장
                            os.makedirs(RESPONSES_DIR, exist_ok=True)
                            response_audio_path = os.path.join(RESPONSES_DIR, "ai_response.mp3")
                            with open(response_audio_path, "wb") as audio_file:
                                audio_file.write(audio_data)
                            print(f"\n응답 오디오 저장 완료: {response_audio_path}")
                            print(f"오디오 파일 크기: {len(audio_data)} 바이트")
                        except Exception as audio_error:
                            print(f"오디오 저장 중 오류 발생: {audio_error}")
                
                except asyncio.TimeoutError:
                    print("서버 응답 타임아웃: 30초 동안 응답이 없습니다.")
            else:
                print(f"샘플 오디오 파일이 없습니다: {SAMPLE_AUDIO_PATH}")
                print("테스트 오디오 파일을 생성하거나 경로를 수정해주세요.")
    
    except Exception as e:
        print(f"웹소켓 연결 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

async def test_integration_flow():
    """통합 플로우 테스트: 부기 -> 꼬기 -> 이미지/오디오 생성"""
    print("\n=== 챗봇 통합 플로우 테스트 ===")
    
    # 1. 부기 챗봇을 통한 이야기 수집
    print("\n1. 부기 챗봇을 통한 이야기 수집")
    bugi = StoryCollectionChatBot()
    bugi.initialize_chat(
        child_name="민준",
        age=6,
        interests=["공룡", "우주", "로봇"],
        chatbot_name="부기"
    )
    
    # 가상의 사용자 입력 설정
    test_inputs = [
        "우주에서 모험하는 이야기를 만들고 싶어",
        "주인공은 용감한 우주 탐험가야",
        "외계인 친구도 나오면 좋겠어",
        "위험한 소행성 지대를 통과하는 모험이 있으면 좋겠어"
    ]
    
    for user_input in test_inputs:
        print(f"사용자: {user_input}")
        response = bugi.get_response(user_input)
        print(f"부기: {response}")
    
    # 이야기 주제 제안
    story_data = bugi.suggest_story_theme()
    print("\n수집된 이야기 주제:")
    print(f"제목: {story_data.get('theme', '제목 없음')}")
    print(f"줄거리: {story_data.get('plot_summary', '줄거리 없음')}")
    
    # 2. 꼬기 챗봇을 통한 이야기 생성
    print("\n2. 꼬기 챗봇을 통한 이야기 생성")
    try:
        kogi = StoryGenerationChatBot()
        story_generation_result = kogi.generate_story(story_data)
        
        print("\n생성된 이야기 미리보기:")
        preview = story_generation_result.get('story_text', '이야기 생성 실패')
        print(preview[:200] + "..." if len(preview) > 200 else preview)
        
        # 3. 이미지/오디오 생성 (목업)
        print("\n3. 멀티미디어 자산 생성 (모의 테스트)")
        print("- 이미지 생성 요청 완료")
        print("- 오디오 내레이션 생성 요청 완료")
        
    except Exception as e:
        print(f"통합 플로우 테스트 중 오류 발생: {e}")

class LiveAudioTestClient:
    """실시간 오디오 테스트 클라이언트"""
    
    def __init__(self, server_url="ws://localhost:8000"):
        self.server_url = server_url
        self.token = "valid_token"
        self.child_name = "테스트"
        self.age = 5
        self.interests = ["공룡", "우주"]
        self.is_streaming = False
        self.websocket = None
        
        # 응답 저장 디렉토리
        os.makedirs(RESPONSES_DIR, exist_ok=True)
        
        # 시그널 핸들러
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Ctrl+C 처리"""
        print("\n\n프로그램 종료 중...")
        self.is_streaming = False
    
    async def connect(self):
        """WebSocket 연결"""
        query_params = f"token={self.token}&child_name={self.child_name}&age={self.age}"
        if self.interests:
            interests_str = ",".join(self.interests)
            query_params += f"&interests={interests_str}"
            
        uri = f"{self.server_url}/ws/audio?{query_params}"
        try:
            print(f"서버 {self.server_url}에 연결 중...")
            self.websocket = await websockets.connect(uri)
            print(f"서버에 연결됨")
            
            # 인사말 메시지 수신
            greeting = await self.websocket.recv()
            greeting_data = json.loads(greeting)
            print(f"\n챗봇: {greeting_data.get('text', '')}")
            
            return True
        except Exception as e:
            print(f"연결 실패: {e}")
            return False
    
    async def send_sample_audio(self, audio_path):
        """샘플 오디오 파일 전송"""
        if not os.path.exists(audio_path):
            print(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
            return False
        
        try:
            # 오디오 파일 읽기
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            print(f"오디오 전송 중... ({len(audio_data)/1024:.1f} KB)")
            await self.websocket.send(audio_data)
            
            # 응답 대기
            response = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
            response_data = json.loads(response)
            
            # 응답 처리
            print("\n응답 수신:")
            user_text = response_data.get('user_text', '')
            bot_text = response_data.get('text', '')
            
            if user_text:
                print(f"인식된 텍스트: 「{user_text}」")
            print(f"챗봇: {bot_text}")
            
            # 오디오 응답 저장
            if response_data.get('audio'):
                audio_b64 = response_data.get('audio')
                audio_data = base64.b64decode(audio_b64)
                
                # 파일 저장
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"response_{timestamp}.mp3"
                filepath = os.path.join(RESPONSES_DIR, filename)
                
                with open(filepath, "wb") as f:
                    f.write(audio_data)
                
                print(f"오디오 응답 저장됨: {filepath}")
            
            return True
        except Exception as e:
            print(f"오디오 전송 중 오류: {e}")
            return False
    
    async def disconnect(self):
        """WebSocket 연결 종료"""
        if self.websocket:
            await self.websocket.close()
            print("서버 연결 종료됨")
    
    async def run_live_test(self, audio_path):
        """라이브 테스트 실행"""
        if await self.connect():
            await self.send_sample_audio(audio_path)
            await self.disconnect()

def run_server():
    """음성 서버 실행"""
    # Uvicorn 서버 설정
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    
    # 서버 시작
    global server_process
    server_process = server
    
    # 서버 실행
    try:
        server.run()
    except KeyboardInterrupt:
        print("서버 종료 신호 감지")
    finally:
        print("서버 종료 처리 완료")

def stop_server():
    """명시적으로 서버를 종료하는 함수"""
    global server_process
    if server_process:
        print("서버 종료 중...")
        # 서버에 종료 요청
        server_process.should_exit = True
        time.sleep(1)
        print("서버 종료 요청 완료")

def create_test_audio():
    """테스트용 빈 오디오 파일 생성"""
    # WAV 파일 헤더 (44바이트)
    wav_header = bytes.fromhex(
        "52494646" +  # "RIFF"
        "24000000" +  # Chunk size (36 + data size)
        "57415645" +  # "WAVE"
        "666d7420" +  # "fmt "
        "10000000" +  # Subchunk1 size (16 bytes)
        "0100" +      # Audio format (1 = PCM)
        "0100" +      # Num channels (1 = mono)
        "44AC0000" +  # Sample rate (44100 Hz)
        "88580100" +  # Byte rate (44100 * 2 = 88200)
        "0200" +      # Block align (2 bytes)
        "1000" +      # Bits per sample (16 bits)
        "64617461" +  # "data"
        "00000000"    # Subchunk2 size (0 bytes for empty file)
    )
    
    # 1초 정도의 무음 데이터 (44100 * 2 바이트)
    silence_data = bytes([0, 0] * 44100)
    
    # 최종 WAV 파일 데이터
    wav_data = bytearray(wav_header)
    
    # data 청크 크기 업데이트 (silence_data 크기)
    data_size = len(silence_data)
    wav_data[40:44] = data_size.to_bytes(4, byteorder='little')
    
    # 전체 파일 크기 업데이트 (36 + data_size)
    file_size = 36 + data_size
    wav_data[4:8] = file_size.to_bytes(4, byteorder='little')
    
    # 무음 데이터 추가
    wav_data.extend(silence_data)
    
    # 파일 저장
    with open(SAMPLE_AUDIO_PATH, "wb") as f:
        f.write(wav_data)
    
    print(f"테스트 오디오 파일 생성 완료: {SAMPLE_AUDIO_PATH}")

def test_combined():
    """통합 테스트 (텍스트 챗봇 + 음성)"""
    print("\n=== 통합 테스트 ===")
    # 먼저 텍스트 기반 챗봇 테스트
    test_chatbot_basic()
    
    # 오디오 파일이 없으면 생성
    if not os.path.exists(SAMPLE_AUDIO_PATH):
        create_test_audio()
    
    # 서버 실행 및 WebSocket 테스트
    server_thread = Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    print("\n서버 시작 중... (5초 대기)")
    time.sleep(5)  # 서버가 시작될 때까지 대기
    
    # WebSocket 테스트 실행
    asyncio.run(test_websocket_connection())

async def run_live_audio_test():
    """라이브 오디오 테스트 실행"""
    client = LiveAudioTestClient()
    await client.run_live_test(SAMPLE_AUDIO_PATH)

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="꼬꼬북 챗봇 통합 테스트")
    parser.add_argument("--create-audio", action="store_true", help="테스트용 오디오 파일 생성")
    parser.add_argument("--run-server", action="store_true", help="음성 서버 실행")
    parser.add_argument("--test-basic", action="store_true", help="기본 챗봇 테스트")
    parser.add_argument("--test-voice", action="store_true", help="음성 인식/합성 테스트")
    parser.add_argument("--test-all", action="store_true", help="모든 테스트 실행")
    parser.add_argument("--test-live", action="store_true", help="라이브 오디오 테스트")
    parser.add_argument("--test-integration", action="store_true", help="통합 플로우 테스트 (부기->꼬기->멀티미디어)")
    parser.add_argument("--save-response", action="store_true", help="응답 오디오 저장")
    args = parser.parse_args()
    
    # 오디오 파일 생성이 필요한 경우
    if args.create_audio:
        create_test_audio()
    
    server_thread = None
    server_started = False
    
    try:
        # 서버 실행이 필요한 경우
        if args.run_server or args.test_voice or args.test_live or args.test_all:
            # 오디오 파일이 없으면 생성
            if not os.path.exists(SAMPLE_AUDIO_PATH):
                create_test_audio()
            
            # 서버 스레드 시작
            server_thread = Thread(target=run_server)
            server_thread.daemon = True
            server_thread.start()
            print("서버 시작 중... (5초 대기)")
            time.sleep(5)  # 서버가 시작될 때까지 대기
            server_started = True
        
        # 기본 챗봇 테스트
        if args.test_basic:
            test_chatbot_basic()
        
        # 음성 인식/합성 테스트
        if args.test_voice:
            asyncio.run(test_websocket_connection())
        
        # 라이브 오디오 테스트
        if args.test_live:
            asyncio.run(run_live_audio_test())
        
        # 통합 플로우 테스트
        if args.test_integration:
            asyncio.run(test_integration_flow())
        
        # 모든 테스트 실행
        if args.test_all:
            test_combined()
            asyncio.run(test_integration_flow())
        
        # 아무 인자도 지정되지 않은 경우 기본 테스트 실행
        if not any([args.create_audio, args.run_server, args.test_basic, args.test_voice, 
                  args.test_all, args.test_live, args.test_integration]):
            test_chatbot_basic()
        
        # 테스트 완료 후 잠시 대기 (서버 스레드가 실행 중인 경우)
        if server_started:
            print("\n모든 테스트 완료. 종료하려면 Ctrl+C를 누르세요...")
            try:
                # 종료 이벤트가 설정될 때까지 대기
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
    
    finally:
        # 프로그램 종료 시 서버 종료
        if server_started:
            print("서버를 종료합니다...")
            stop_server()
            # 서버 종료 대기
            time.sleep(2)
            print("프로그램 종료 완료")

if __name__ == "__main__":
    main() 