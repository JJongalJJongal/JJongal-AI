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
import unittest
# 상위 디렉토리 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__)) # chatbot/tests
parent_dir = os.path.dirname(current_dir)  # chatbot
project_root = os.path.dirname(parent_dir) # CCB_AI (프로젝트 루트)
sys.path.append(project_root) # 프로젝트 루트를 sys.path에 추가

# 모듈 임포트
from chatbot.models.chat_bot_a import StoryCollectionChatBot
from chatbot.models.chat_bot_b import StoryGenerationChatBot
from chatbot.models.voice_ws.voice_ws_server import app
from shared.utils.file_utils import ensure_directory
import uvicorn

# 테스트용 오디오 파일 경로
SAMPLE_AUDIO_PATH = os.path.join(current_dir, "test_audio.wav")
# 응답 저장 디렉토리
RESPONSES_DIR = os.path.join(current_dir, "responses")

class TestBugiFunctionality(unittest.TestCase):
    def test_basic_interaction(self):
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
        
        self.assertIsNotNone(greeting, "인사말이 생성되지 않았습니다.")
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
            self.assertIsNotNone(response, f"'{user_input}'에 대한 응답이 없습니다.")
            print(f"챗봇: {response}")
        
        # 이야기 요약 테스트
        story = chatbot.suggest_story_theme()
        self.assertIsNotNone(story, "이야기 테마가 생성되지 않았습니다.")
        print("\n=== 수집된 이야기 테마 ===")
        print(f"주제: {story.get('theme', '')}")
        print(f"줄거리: {story.get('plot_summary', '')}")
        self.assertIsNotNone(story.get('plot_summary'), "줄거리가 생성되지 않았습니다.")
        
        # 토큰 사용량 확인 (선택 사항)
        token_info = chatbot.get_token_usage()
        self.assertIsNotNone(token_info, "토큰 정보가 반환되지 않았습니다.")
        print(f"사용된 토큰: {token_info.get('total', 0)}")
        if 'total' in token_info and 'token_limit' in token_info:
            print(f"남은 토큰 (추정): {token_info['token_limit'] - token_info['total']}")
        elif 'total' in token_info and hasattr(chatbot, 'conversation') and hasattr(chatbot.conversation, 'token_limit'):
            print(f"남은 토큰: {chatbot.conversation.token_limit - token_info['total']}")
        else:
            print("남은 토큰 정보를 계산할 수 없습니다.")

class TestWebSocketFunctionality(unittest.IsolatedAsyncioTestCase):
    server_thread = None
    server_process = None # 클래스 변수로 이동

    @classmethod
    def setUpClass(cls):
        print("\nSetting up WebSocket test environment...")
        # 오디오 파일이 없으면 생성
        if not os.path.exists(SAMPLE_AUDIO_PATH):
            create_test_audio()
        
        # Uvicorn 서버 설정
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        # cls.server_process는 uvicorn.Server 인스턴스를 직접 참조하도록 변경될 수 있습니다.
        # 여기서는 Thread를 사용하여 run_server 함수를 실행합니다.
        
        # run_server 함수를 Thread로 실행
        cls.server_thread = Thread(target=TestWebSocketFunctionality._run_server_in_thread, daemon=True)
        cls.server_thread.start()
        print("WebSocket Server starting... (waiting 5 seconds)")
        time.sleep(5) # 서버 시작 대기

    @classmethod
    def tearDownClass(cls):
        print("\nTearing down WebSocket test environment...")
        if hasattr(cls, 'server_process') and cls.server_process and hasattr(cls.server_process, 'should_exit'):
            print("Stopping WebSocket Server...")
            cls.server_process.should_exit = True # uvicorn 서버 종료 플래그
            if cls.server_thread:
                 cls.server_thread.join(timeout=5) # 스레드 종료 대기
        elif cls.server_thread and cls.server_thread.is_alive(): # Fallback if server_process was not set as uvicorn.Server
            print("Attempting to stop server thread by other means or wait for daemon thread to exit.")
            # 데몬 스레드는 메인 스레드 종료시 자동 종료되지만, 명시적 종료 시도
            # 실제 uvicorn.Server 인스턴스를 직접 제어하는 것이 더 안정적입니다.
        print("WebSocket Server stopped.")

    @staticmethod
    def _run_server_in_thread():
        # 이 메서드는 uvicorn 서버를 실행합니다.
        # setUpClass에서 Thread의 타겟으로 사용됩니다.
        # 전역 server_process 대신 TestWebSocketFunctionality.server_process를 사용하도록 수정.
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning") # 로그 레벨 조정 가능
        server = uvicorn.Server(config)
        TestWebSocketFunctionality.server_process = server # 클래스 변수에 uvicorn.Server 인스턴스 저장
        try:
            server.run()
        except KeyboardInterrupt: # 보통 테스트 환경에서는 발생하지 않음
            print("Server interrupted in thread.")
        finally:
            print("Server run method finished in thread.")


    async def test_audio_transmission(self):
        """웹소켓 음성 기능 테스트"""
        print("\n=== 웹소켓 음성 기능 테스트 ===")
        user_id = "test_user_ws"
        story_id = "test_story_ws"
        # uri = f"ws://localhost:8000/ws/audio?token=valid_token&user_id={user_id}&story_id={story_id}" # 기존 URI
        base_uri = f"ws://localhost:8000/ws/audio?user_id={user_id}&story_id={story_id}"
        
        # 테스트용 토큰 (auth.py의 validate_token이 "test_token"을 허용)
        test_auth_token = "test_token"
        headers = {
            "Authorization": f"Bearer {test_auth_token}"
        }

        print(f"Connecting to WebSocket: {base_uri} with token in header")
        try:
            # async with websockets.connect(uri) as websocket: # 기존 연결
            async with websockets.connect(base_uri, extra_headers=headers) as websocket:
                print("WebSocket connection established.")
                
                greeting_response = await websocket.recv()
                greeting_data = json.loads(greeting_response)
                self.assertIn("text", greeting_data, "인사말에 텍스트가 없습니다.")
                print(f"인사말: {greeting_data.get('text', '')}")
                
                self.assertTrue(os.path.exists(SAMPLE_AUDIO_PATH), f"샘플 오디오 파일 없음: {SAMPLE_AUDIO_PATH}")
                if os.path.exists(SAMPLE_AUDIO_PATH):
                    print(f"오디오 파일 크기: {os.path.getsize(SAMPLE_AUDIO_PATH)} 바이트")
                    
                    with open(SAMPLE_AUDIO_PATH, "rb") as audio_file:
                        audio_data = audio_file.read()
                    
                    print(f"전송할 오디오 데이터 크기: {len(audio_data)} 바이트")
                    await websocket.send(audio_data)
                    print("샘플 오디오 전송 완료")
                    
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        response_data = json.loads(response)
                    
                        print("\n서버 응답:")
                        self.assertIn("type", response_data, "응답에 유형 정보가 없습니다.")
                        print(f"응답 유형: {response_data.get('type', '')}")
                        print(f"AI 응답 텍스트: {response_data.get('text', '')}")
                        print(f"사용자 음성 인식: {response_data.get('user_text', '')}")
                        print(f"상태: {response_data.get('status', '')}")
                        
                        if "error_message" in response_data:
                            print(f"오류: {response_data.get('error_message')}")
                        
                        if "audio" in response_data and response_data["audio"]:
                            try:
                                audio_decoded_data = base64.b64decode(response_data["audio"])
                                ensure_directory(RESPONSES_DIR)
                                response_audio_path = os.path.join(RESPONSES_DIR, "ai_ws_response.mp3")
                                with open(response_audio_path, "wb") as audio_file_out:
                                    audio_file_out.write(audio_decoded_data)
                                print(f"\n응답 오디오 저장 완료: {response_audio_path}")
                                self.assertTrue(os.path.exists(response_audio_path), "응답 오디오 파일이 저장되지 않았습니다.")
                            except Exception as audio_error:
                                self.fail(f"오디오 저장 중 오류 발생: {audio_error}")
                    
                    except asyncio.TimeoutError:
                        self.fail("서버 응답 타임아웃: 30초 동안 응답이 없습니다.")
        
        except Exception as e:
            self.fail(f"웹소켓 연결 중 오류 발생: {e}")

class TestChatBotIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_story_generation_with_images(self):
        """부기 -> 꼬기 통합 테스트 (상세 스토리 및 이미지 생성 검증)"""
        print("\n=== 챗봇 통합 플로우 (이미지 생성 포함) 테스트 ===")
        
        # 1. 부기 챗봇을 통한 이야기 수집
        print("\n1. 부기 챗봇을 통한 이야기 수집")
        bugi = StoryCollectionChatBot()
        bugi.initialize_chat(
            child_name="민준",
            age=6,
            interests=["공룡", "우주", "로봇"],
            chatbot_name="부기"
        )
        
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
        
        story_data = bugi.suggest_story_theme()
        print("\n수집된 이야기 주제:")
        # story_data.get('theme')이 실제 story_data의 키와 일치하는지 확인 필요.
        # 이전 코드에서는 story_data.get('title')로 되어있던 부분을 theme으로 가정하고 수정.
        # 실제 Bugi의 suggest_story_theme() 반환값을 확인하여 정확한 키를 사용해야 함.
        print(f"주제: {story_data.get('theme', story_data.get('title', '제목 없음'))}") 
        print(f"줄거리: {story_data.get('plot_summary', '줄거리 없음')}")
        self.assertIsNotNone(story_data.get('plot_summary'), "줄거리가 수집되지 않았습니다.")

        # 2. 꼬기 챗봇을 통한 이야기 생성 및 이미지 생성 테스트
        print("\n\n2. 꼬기 챗봇을 통한 이야기 생성 및 이미지 생성 테스트")
        try:
            # RESPONSES_DIR이 정의되어 있다고 가정합니다. test_chatbot.py 상단에 정의되어 있을 것입니다.
            kogi_output_dir = os.path.join(RESPONSES_DIR, "kogi_integration_test_output")
            ensure_directory(kogi_output_dir) # 테스트 출력 디렉토리 생성 보장

            kogi = StoryGenerationChatBot(output_dir=kogi_output_dir)
            kogi.set_story_outline(story_data) 
            kogi.set_target_age(story_data.get('target_age', 6))
            
            main_char_name = story_data.get("characters", ["테스트주인공"])[0] if story_data.get("characters") else "테스트주인공"
            kogi.set_cloned_voice_info(child_voice_id="test_child_voice_id_placeholder", main_character_name=main_char_name)
            
            # kogi.generate_story()는 동기 함수이므로 await asyncio.to_thread를 사용합니다. # 이 주석은 이제 올바르지 않습니다.
            # generated_illustrations, main_narration_path = await asyncio.to_thread(kogi.generate_story)
            generated_illustrations, main_narration_path = await kogi.generate_story() # 직접 await 호출

            print("\n=== 생성된 상세 스토리 정보 ===")
            self.assertIsNotNone(kogi.detailed_story, "상세 스토리가 생성되지 않았습니다.")
            print(f"상세 스토리 제목: {kogi.detailed_story.get('title', '제목 없음')}")
            self.assertTrue(len(kogi.detailed_story.get('chapters', [])) > 0, "상세 스토리에 챕터가 없습니다.")
            print(f"챕터 수: {len(kogi.detailed_story.get('chapters', []))}")

            print("\n=== 생성된 이미지 정보 ===")
            self.assertTrue(generated_illustrations, "이미지가 생성되지 않았습니다.")
            print(f"생성된 이미지 개수: {len(generated_illustrations)}")
            for img_path_str in generated_illustrations:
                img_path = Path(img_path_str)
                print(f"이미지 경로: {img_path}")
                self.assertTrue(img_path.exists(), f"이미지 파일이 존재하지 않습니다: {img_path}")
                self.assertTrue(img_path.is_file(), f"이미지 경로가 파일이 아닙니다: {img_path}")

            if main_narration_path:
                print(f"\n메인 내레이션 오디오 경로: {main_narration_path}")
            else:
                print("\n메인 내레이션 오디오가 생성되지 않았습니다.") # 이 테스트에서는 오디오 생성 실패를 오류로 간주하지 않음
            
            print("\n통합 플로우 (상세 스토리 및 이미지 생성) 테스트 성공!")

        except Exception as e:
            print(f"통합 플로우 테스트 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"테스트 실패: {e}")

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
    print("\n=== 통합 테스트 (test_combined) ===")
    
    # 1. 기본 챗봇 기능 테스트 실행 (TestBugiFunctionality)
    print("TestBugiFunctionality 실행 중...")
    loader = unittest.TestLoader()
    suite_bugi = loader.loadTestsFromTestCase(TestBugiFunctionality)
    runner_bugi = unittest.TextTestRunner()
    runner_bugi.run(suite_bugi)

    # 2. 웹소켓 기능 테스트 실행 (TestWebSocketFunctionality)
    print("\nTestWebSocketFunctionality 실행 중...")
    suite_ws = loader.loadTestsFromTestCase(TestWebSocketFunctionality)
    runner_ws = unittest.TextTestRunner()
    runner_ws.run(suite_ws)
    
    print("\n통합 테스트 (test_combined) 완료.")


async def run_live_audio_test():
    """라이브 오디오 테스트 실행 (서버 별도 실행 필요)"""
    # 이 테스트는 서버가 이미 실행 중이라고 가정합니다.
    # TestWebSocketFunctionality와 서버를 공유하지 않도록 주의.
    print("\n=== 라이브 오디오 테스트 (run_live_audio_test) ===")
    print("주의: 이 테스트는 웹소켓 서버가 이미 실행 중이어야 합니다 (예: python main.py --run-server).")
    if not os.path.exists(SAMPLE_AUDIO_PATH):
        create_test_audio()
    
    client = LiveAudioTestClient() # LiveAudioTestClient는 자체적으로 서버에 연결합니다.
    await client.run_live_test(SAMPLE_AUDIO_PATH)

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="꼬꼬북 챗봇 통합 테스트")
    parser.add_argument("--create-audio", action="store_true", help="테스트용 오디오 파일 생성")
    parser.add_argument("--test-basic", action="store_true", help="기본 챗봇 테스트 (TestBugiFunctionality)")
    parser.add_argument("--test-voice", action="store_true", help="음성 인식/합성 테스트 (TestWebSocketFunctionality)")
    parser.add_argument("--test-all", action="store_true", help="모든 테스트 실행 (기본, 음성, 통합 플로우)")
    parser.add_argument("--test-live", action="store_true", help="라이브 오디오 테스트 (서버 별도 실행 필요)")
    parser.add_argument("--test-integration", action="store_true", help="통합 플로우 테스트 (부기->꼬기->멀티미디어, TestChatBotIntegration)")
    args = parser.parse_args()
    
    if args.create_audio:
        create_test_audio()

    try:
        if args.test_basic:
            print("TestBugiFunctionality 실행 중...")
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(TestBugiFunctionality)
            runner = unittest.TextTestRunner()
            runner.run(suite)
        
        if args.test_voice:
            print("TestWebSocketFunctionality 실행 중...")
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(TestWebSocketFunctionality)
            runner = unittest.TextTestRunner()
            runner.run(suite)
        
        if args.test_live:
            # run_live_audio_test는 서버가 외부에서 실행되어야 함을 명시
            asyncio.run(run_live_audio_test())
        
        if args.test_integration:
            print("TestChatBotIntegration 실행 중...")
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(TestChatBotIntegration)
            runner = unittest.TextTestRunner()
            runner.run(suite)
        
        if args.test_all:
            test_combined() 
            print("\nTestChatBotIntegration (as part of --test-all) 실행 중...")
            loader = unittest.TestLoader()
            suite_integration = loader.loadTestsFromTestCase(TestChatBotIntegration)
            runner_integration = unittest.TextTestRunner()
            runner_integration.run(suite_integration)

        # 아무 인자도 지정되지 않은 경우 도움말 표시 또는 기본 테스트 실행
        if not any(vars(args).values()): # 인자가 하나도 True가 아니면
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nMain Kbd Interrupt: 프로그램을 종료합니다.")
    finally:
        print("\n모든 테스트 실행 완료 (main).")

if __name__ == "__main__":
    main() 