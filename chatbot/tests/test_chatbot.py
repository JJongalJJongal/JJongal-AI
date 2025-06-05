import sys
import os
import asyncio
import websockets
import json
import base64
import argparse
import time
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
from chatbot.models.chat_bot_a.chat_bot_a import ChatBotA
from chatbot.models.chat_bot_b.chat_bot_b import ChatBotB
from chatbot.models.chat_bot_b.generators.image_generator import ImageGenerator
from chatbot.models.voice_ws.app import app as voice_ws_app
from chatbot.data.vector_db.core import VectorDB
from shared.utils.file_utils import ensure_directory
import uvicorn

# 테스트용 오디오 파일 경로
SAMPLE_AUDIO_PATH = os.path.join(project_root, "output", "temp", "test_audio.mp3")
# 응답 저장 디렉토리
RESPONSES_DIR = os.path.join(project_root, "output", "temp")

class CCBIntegratedTest(unittest.IsolatedAsyncioTestCase):
    """
    CCB AI 통합 테스트 클래스
    부기(ChatBotA) → 꼬기(ChatBotB) → 웹소켓 → 통합 플로우
    """
    
    server_thread = None
    server_process = None
    
    @classmethod
    def setUpClass(cls):
        """테스트 환경 전체 설정"""
        print("\n" + "="*60)
        print("     CCB AI 통합 테스트 시스템 초기화")
        print("="*60)
        
        # 응답 저장 디렉토리 생성
        ensure_directory(RESPONSES_DIR)
        
        # 테스트용 오디오 파일 생성
        if not os.path.exists(SAMPLE_AUDIO_PATH):
            cls._create_test_audio()
        
        # WebSocket 서버 시작 (voice 테스트용)
        cls._start_websocket_server()
        
        print("✅ 통합 테스트 환경 설정 완료\n")
    
    @classmethod
    def tearDownClass(cls):
        """테스트 환경 정리"""
        print("\n" + "="*60)
        print("     CCB AI 통합 테스트 시스템 종료")
        print("="*60)
        
        cls._stop_websocket_server()
        print("✅ 통합 테스트 환경 정리 완료")
    
    @classmethod
    def _create_test_audio(cls):
        """테스트용 MP3 오디오 파일 생성"""
        print("📁 테스트용 MP3 오디오 파일 생성 중...")
        
        try:
            # OpenAI TTS를 사용해서 실제 테스트용 MP3 생성
            from openai import OpenAI
            import os
            
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                client = OpenAI(api_key=api_key)
                
                # 테스트용 간단한 음성 생성
                response = client.audio.speech.create(
                    model="tts-1",
                    voice="nova",
                    input="안녕하세요. 이것은 테스트용 음성 파일입니다.",
                    response_format="mp3"
                )
                
                # MP3 파일로 저장
                with open(SAMPLE_AUDIO_PATH, "wb") as f:
                    f.write(response.content)
                    
                print(f"   ✅ OpenAI TTS로 MP3 파일 생성: {SAMPLE_AUDIO_PATH}")
                return
                
        except Exception as e:
            print(f"   ⚠️ OpenAI TTS 생성 실패: {e}")
        
        # OpenAI TTS 실패 시 최소한의 MP3 헤더로 더미 파일 생성
        # 간단한 MP3 프레임 헤더 (실제로는 재생되지 않지만 파일 형식은 MP3)
        mp3_header = bytes([
            0xFF, 0xFB, 0x90, 0x00,  # MP3 sync word + header
            0x00, 0x00, 0x00, 0x00,  # 더미 데이터
        ])
        
        # 더미 MP3 데이터 (최소 크기)
        dummy_mp3_data = mp3_header * 100  # 간단한 반복
        
        with open(SAMPLE_AUDIO_PATH, "wb") as f:
            f.write(dummy_mp3_data)
        
        print(f"   ✅ 더미 MP3 파일 생성: {SAMPLE_AUDIO_PATH}")

    @classmethod
    def _start_websocket_server(cls):
        """WebSocket 서버 시작"""
        print("🚀 WebSocket 서버 시작 중...")
        
        # 이미 실행 중인 서버 확인
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', 8000))
            sock.close()
            if result == 0:
                print("   ⚠️ 서버가 이미 실행 중입니다.")
                return
        except:
            pass
        
        cls.server_thread = Thread(target=cls._run_server_in_thread, daemon=True)
        cls.server_thread.start()
        
        # 서버 시작 대기 (최대 15초)
        for i in range(15):
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', 8000))
                sock.close()
                if result == 0:
                    print(f"   ✅ 서버 시작 완료 ({i+1}초 소요)")
                    time.sleep(2)
                    return
            except:
                pass
            time.sleep(1)
        
        print("   ⚠️ 서버 시작 확인 실패, 테스트 계속 진행")

    @classmethod
    def _stop_websocket_server(cls):
        """WebSocket 서버 종료"""
        print("🛑 WebSocket 서버 종료 중...")
        if hasattr(cls, 'server_process') and cls.server_process:
            if hasattr(cls.server_process, 'should_exit'):
                cls.server_process.should_exit = True
            if cls.server_thread:
                cls.server_thread.join(timeout=5)
        print("   ✅ 서버 종료 완료")
    
    @classmethod
    def _run_server_in_thread(cls):
        """별도 스레드에서 서버 실행"""
        try:
            config = uvicorn.Config(voice_ws_app, host="0.0.0.0", port=8000, log_level="warning")
            server = uvicorn.Server(config)
            cls.server_process = server
            server.run()
        except Exception as e:
            print(f"   ❌ 서버 실행 오류: {e}")
        finally:
            print("   🔄 서버 스레드 종료")
    
    # ==========================================
    # 1. 부기(ChatBotA) 기본 기능 테스트
    # ==========================================
    
    async def test_01_bugi_basic_functionality(self):
        """1단계: 부기 챗봇 기본 기능 테스트"""
        print("\n" + "="*50)
        print("🤖 1단계: 부기 챗봇 기본 기능 테스트")
        print("="*50)
        
        # 챗봇 인스턴스 생성
        try:
            vector_db = VectorDB(persist_directory="chatbot/data/vector_db/main")
        except Exception as e:
            print(f"⚠️ VectorDB 초기화 실패, None으로 진행: {e}")
            vector_db = None
            
        chatbot = ChatBotA(vector_db_instance=vector_db)
        
        # 테스트 아이 정보
        child_name = "테스트"
        age = 6
        interests = ["공룡", "우주", "로봇"]
        
        # 챗봇 초기화
        greeting = chatbot.initialize_chat(
            child_name=child_name,
            age=age,
            interests=interests,
            chatbot_name="부기"
        )
        
        self.assertIsNotNone(greeting, "인사말이 생성되지 않았습니다.")
        print(f"✅ 인사말: {greeting}")
        
        # 테스트 대화
        test_inputs = [
            "안녕! 나는 공룡을 좋아해",
            "티라노사우루스가 제일 멋있어",
            "내 이야기에는 용감한 아이가 나올 거야"
        ]
        
        print("\n🗣️ 테스트 대화:")
        for user_input in test_inputs:
            print(f"   사용자: {user_input}")
            response = chatbot.get_response(user_input)
            self.assertIsNotNone(response, f"'{user_input}'에 대한 응답이 없습니다.")
            print(f"   부기: {response[:100]}..." if len(response) > 100 else f"   부기: {response}")
        
        # 이야기 테마 추출
        story = chatbot.suggest_story_theme()
        self.assertIsNotNone(story, "이야기 테마가 생성되지 않았습니다.")
        
        print(f"\n📖 수집된 이야기 테마:")
        print(f"   주제: {story.get('theme', '')}")
        print(f"   줄거리: {story.get('plot_summary', '')}")
        
        self.assertIsNotNone(story.get('plot_summary'), "줄거리가 생성되지 않았습니다.")
        
        # 토큰 사용량 확인
        token_info = chatbot.get_token_usage()
        self.assertIsNotNone(token_info, "토큰 정보가 반환되지 않았습니다.")
        print(f"   사용된 토큰: {token_info.get('total', 0)}")
        
        print("✅ 부기 챗봇 기본 기능 테스트 완료\n")
        
        # 다음 테스트를 위해 story 저장
        self._test_story_data = story
        return story
    
    # ==========================================
    # 2. 꼬기(ChatBotB) 이미지/음성 테스트
    # ==========================================
    
    async def test_02_kogi_multimedia_generation(self):
        """2단계: 꼬기 챗봇 멀티미디어 생성 테스트"""
        print("\n" + "="*50)
        print("🎨 2단계: 꼬기 챗봇 멀티미디어 생성 테스트")
        print("="*50)
        
        # 출력 디렉토리 설정 (output/temp 사용)
        kogi_output_dir = os.path.join(project_root, "output", "temp")
        ensure_directory(kogi_output_dir)
        
        # 꼬기 챗봇 초기화 (RAG 활성화)
        kogi = ChatBotB(
            output_dir=kogi_output_dir,
            vector_db_path="chatbot/data/vector_db/detailed",
            collection_name="fairy_tales"
        )
        
        # 테스트용 스토리 데이터 설정
        child_name_for_test = "병찬"
        age_for_test = 7
        story_outline_data = {
            "title": f"{child_name_for_test}의 자유로운 여행",
            "theme": "여행",
            "plot_summary": (
                f"{child_name_for_test}는 항상 배가 고픈 {age_for_test}살 어린이다. "
                f"어느날 {child_name_for_test}는 일에 너무 지쳐서 여행을 떠나고 싶었다. "
                f"그래서 {child_name_for_test}는 친구들과 함께 여행을 떠났다."
            ),
            "characters": [
                {"name": child_name_for_test, "description": f"{age_for_test}살 아이"},
                {"name": "친구1", "description": f"{age_for_test}살 아이"},
                {"name": "친구2", "description": f"{age_for_test}살 아이"},
            ],
            "setting": "여행지",
            "educational_value": "문제 해결 능력, 협동심",
            "target_age": age_for_test
        }
        
        # 스토리 설정
        kogi.set_story_outline(story_outline_data)
        kogi.set_target_age(age_for_test)
        
        # 음성 클로닝 정보 설정
        kogi.character_voice_mapping = {
            child_name_for_test: "EXAVITQu4vr4xnSDxMaL",  # 아이 목소리
            "엄마": "21m00Tcm4TlvDq8ikWAM",     # 여성 목소리
            "아빠": "VR6AewLTigWG4xSOukaG",     # 남성 목소리
            "요정": "pNInz6obpgDQGcFmaJgB"      # 판타지 목소리
        }
        
        print("🎬 상세 스토리 및 멀티미디어 생성 중... (RAG 활성화)")
        
        # 상세 스토리 생성
        result = await kogi.generate_detailed_story()

        # 결과 검증
        self.assertIsNotNone(result, "스토리 생성 결과가 없습니다.")
        self.assertIn("story_data", result, "결과에 스토리 데이터가 없습니다.")
        
        story_data_result = result.get("story_data")
        self.assertIsNotNone(story_data_result, "상세 스토리가 생성되지 않았습니다.")
        
        print("📚 생성된 스토리:")
        print(f"   제목: {story_data_result.get('title', '제목 없음')}")
        print(f"   생성 상태: {result.get('status', '상태 없음')}")
        
        chapters = story_data_result.get('chapters', [])
        self.assertTrue(len(chapters) > 0, "상세 스토리에 챕터가 없습니다.")
        print(f"   챕터 수: {len(chapters)}")
        
        for i, chapter in enumerate(chapters[:2]):
            title = chapter.get('chapter_title', chapter.get('title', '제목 없음'))
            print(f"     챕터 {i+1}: {title}")

        # 이미지 생성 결과 확인
        image_paths_list = result.get("image_paths", [])
        if image_paths_list:
            print(f"🖼️ 생성된 이미지: {len(image_paths_list)}개")
            generated_image_files_count = 0
            for i, image_path_str in enumerate(image_paths_list[:3]):  # 처음 3개만 확인
                if image_path_str:
                    img_path = Path(image_path_str)
                    if img_path.exists():
                        print(f"     ✅ 이미지 {i+1}: {img_path.stat().st_size} bytes")
                        generated_image_files_count += 1
                    else:
                        print(f"     ❌ 이미지 {i+1}: 파일 없음")
            
            if generated_image_files_count > 0:
                print(f"   총 {generated_image_files_count}개 이미지 생성 성공")
        else:
            print("   ⚠️ 이미지가 생성되지 않았습니다.")

        # 음성 생성 결과 확인
        audio_paths = result.get("audio_paths", [])
        if audio_paths:
            print(f"🔊 생성된 음성: {len(audio_paths)}개")
            for i, audio_path_str in enumerate(audio_paths[:3]):  # 처음 3개만 확인
                audio_path = Path(audio_path_str)
                if audio_path.exists():
                    print(f"     ✅ 음성 {i+1}: {audio_path.stat().st_size} bytes")
                else:
                    print(f"     ❌ 음성 {i+1}: 파일 없음")
        else:
            print("   ⚠️ 음성이 생성되지 않았습니다.")

        self.assertTrue(len(chapters) > 0, "최소한 하나의 챕터는 생성되어야 합니다.")
        
        print("✅ 꼬기 챗봇 멀티미디어 생성 테스트 완료 (RAG 활성화)\n")
        return result
    
    # ==========================================
    # 3. 웹소켓 음성 기능 테스트
    # ==========================================
    
    async def test_03_websocket_voice_functionality(self):
        """3단계: 웹소켓 음성 기능 테스트"""
        print("\n" + "="*50)
        print("🎤 3단계: 웹소켓 음성 기능 테스트")
        print("="*50)
        
        # 필수 파라미터 설정
        child_name = "테스트"
        age = 6
        interests = "공룡,우주,로봇"
        test_auth_token = "development_token"
        
        # WebSocket URI 구성
        base_uri = f"ws://localhost:8000/ws/audio?child_name={child_name}&age={age}&interests={interests}&token={test_auth_token}"
        
        print(f"🔗 WebSocket 연결: {base_uri}")
        
        try:
            async with websockets.connect(base_uri) as websocket:
                print("   ✅ WebSocket 연결 성공")
                
                # 인사말 수신
                greeting_response = await websocket.recv()
                greeting_data = json.loads(greeting_response)
                self.assertIn("text", greeting_data, "인사말에 텍스트가 없습니다.")
                print(f"   🤖 인사말: {greeting_data.get('text', '')}")
                
                # 오디오 파일 확인
                self.assertTrue(os.path.exists(SAMPLE_AUDIO_PATH), f"샘플 오디오 파일 없음: {SAMPLE_AUDIO_PATH}")
                print(f"   📁 오디오 파일 크기: {os.path.getsize(SAMPLE_AUDIO_PATH)} 바이트")
                
                # 오디오 전송
                with open(SAMPLE_AUDIO_PATH, "rb") as audio_file:
                    audio_data = audio_file.read()
                
                print(f"   📤 오디오 전송: {len(audio_data)} 바이트")
                await websocket.send(audio_data)
                
                # 서버 처리 대기
                print("   ⏳ 서버 오디오 처리 대기 중... (3초)")
                await asyncio.sleep(3)
                
                # 응답 수신
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    response_data = json.loads(response)
                    
                    print("   📥 서버 응답 수신:")
                    self.assertIn("type", response_data, "응답에 유형 정보가 없습니다.")
                    print(f"     응답 유형: {response_data.get('type', '')}")
                    print(f"     AI 응답: {response_data.get('text', '')}")
                    print(f"     사용자 음성 인식: {response_data.get('user_text', '')}")
                    print(f"     상태: {response_data.get('status', '')}")
                    
                    if "error_message" in response_data:
                        print(f"     오류: {response_data.get('error_message')}")
                    
                    # 오디오 응답 저장
                    if "audio" in response_data and response_data["audio"]:
                        try:
                            audio_decoded_data = base64.b64decode(response_data["audio"])
                            response_audio_path = os.path.join(project_root, "output", "temp", "ai_ws_response.mp3")
                            ensure_directory(os.path.dirname(response_audio_path))
                            with open(response_audio_path, "wb") as audio_file_out:
                                audio_file_out.write(audio_decoded_data)
                            print(f"     🔊 응답 오디오 저장: {response_audio_path}")
                            self.assertTrue(os.path.exists(response_audio_path), "응답 오디오 파일이 저장되지 않았습니다.")
                        except Exception as audio_error:
                            self.fail(f"오디오 저장 중 오류 발생: {audio_error}")
                
                except asyncio.TimeoutError:
                    self.fail("서버 응답 타임아웃: 30초 동안 응답이 없습니다.")
        
        except Exception as e:
            self.fail(f"웹소켓 연결 중 오류 발생: {e}")
        
        print("✅ 웹소켓 음성 기능 테스트 완료\n")
    
    # ==========================================
    # 4. 부기→꼬기 통합 플로우 테스트
    # ==========================================
    
    async def test_04_bugi_kogi_integration_flow(self):
        """4단계: 부기→꼬기 통합 플로우 테스트"""
        print("\n" + "="*50)
        print("🔄 4단계: 부기→꼬기 통합 플로우 테스트")
        print("="*50)
        
        # 1. 부기 챗봇으로 이야기 수집
        print("🤖 부기 챗봇으로 이야기 수집 중...")
        try:
            vector_db = VectorDB(persist_directory="chatbot/data/vector_db/main")
        except Exception as e:
            print(f"⚠️ VectorDB 초기화 실패, None으로 진행: {e}")
            vector_db = None
            
        chatbot = ChatBotA(vector_db_instance=vector_db)
        chatbot.initialize_chat(
            child_name="민준", # 아이 이름
            age=6, # 아이 나이
            interests=["공룡", "우주", "로봇"], # 아이 관심사
            chatbot_name="부기" # 챗봇 이름
        )
        
        # 테스트 대화
        test_inputs = [
            "우주에서 모험하는 이야기를 만들고 싶어",
            "주인공은 용감한 우주 탐험가야",
            "외계인 친구도 나오면 좋겠어",
            "위험한 소행성 지대를 통과하는 모험이 있으면 좋겠어"
        ]
        
        for user_input in test_inputs:
            print(f"   사용자: {user_input}")
            response = chatbot.get_response(user_input)
            print(f"   부기: {response[:50]}...")
        
        # 이야기 주제 추출
        story_data = chatbot.suggest_story_theme()
        self.assertIsNotNone(story_data, "이야기 주제가 생성되지 않았습니다.")
        
        print("📖 부기가 수집한 이야기 주제:")
        print(f"   제목: {story_data.get('title', story_data.get('theme', '제목 없음'))}")
        print(f"   주제: {story_data.get('theme', '주제 없음')}")
        print(f"   줄거리: {story_data.get('plot_summary', '줄거리 없음')}")
        print(f"   등장인물: {story_data.get('characters', [])}")
        print(f"   배경: {story_data.get('setting', '배경 없음')}")
        
        # 필수 요소 검증
        self.assertIsNotNone(story_data.get('plot_summary'), "줄거리가 수집되지 않았습니다.")
        self.assertTrue(len(story_data.get('characters', [])) > 0, "등장인물이 수집되지 않았습니다.")

        # 2. 꼬기 챗봇으로 상세 이야기 생성
        print("\n🎨 꼬기 챗봇으로 상세 이야기 생성 중...")
        
        kogi_output_dir = os.path.join(project_root, "output", "temp")
        ensure_directory(kogi_output_dir)

        kogi = ChatBotB(
            output_dir=kogi_output_dir,
            vector_db_path="chatbot/data/vector_db/detailed",
            collection_name="fairy_tales"
        )
            
        # 이미지 생성기 확인 (ChatBotB에 이미 설정됨)
        if hasattr(kogi, 'image_generator') and kogi.image_generator:
            print("   ✅ 이미지 생성기가 이미 설정되어 있습니다.")
        elif hasattr(kogi, 'story_engine') and hasattr(kogi.story_engine, 'image_generator'):
            print("   ✅ story_engine에 이미지 생성기가 설정되어 있습니다.")
        else:
            print("   ⚠️ 이미지 생성기를 찾을 수 없습니다. 텍스트만 생성됩니다.")
        
        # 스토리 설정
        kogi.set_story_outline(story_data) 
        kogi.set_target_age(story_data.get('target_age', 6))
            
        # 캐릭터 이름 추출
        characters = story_data.get("characters", [])
        if characters and isinstance(characters[0], dict):
            main_char_name = characters[0].get("name", "테스트주인공")
        elif characters and isinstance(characters[0], str):
            main_char_name = characters[0]
        else:
            main_char_name = "테스트주인공"
            
        # 음성 클로닝 정보 설정
        kogi.set_cloned_voice_info(
            child_voice_id="test_child_voice_id",
            main_character_name=main_char_name
        )
            
        # 상세 스토리 생성
        result = await kogi.generate_detailed_story()
            
        # 결과 검증
        self.assertIsNotNone(result, "스토리 생성 결과가 없습니다.")
        self.assertIn("story_data", result, "스토리 데이터가 없습니다.")
            
        story_data_result = result["story_data"]
        self.assertIsNotNone(story_data_result, "상세 스토리가 생성되지 않았습니다.")
            
        print("📚 생성된 상세 스토리 정보:")
        print(f"   제목: {story_data_result.get('title', '제목 없음')}")
        print(f"   생성 상태: {result.get('status', '상태 없음')}")
            
        chapters = story_data_result.get('chapters', [])
        self.assertTrue(len(chapters) > 0, "상세 스토리에 챕터가 없습니다.")
        print(f"   챕터 수: {len(chapters)}")
        
        # 이미지/음성 정보
        image_paths = result.get("image_paths", [])
        audio_paths = result.get("audio_paths", [])
        
        if image_paths:
            print(f"   🖼️생성된 이미지: {len(image_paths)}개")
        else:
            print("   📝 텍스트만 생성됨 (이미지 없음)")

        if audio_paths:
            print(f"   🔊생성된 음성: {len(audio_paths)}개")
        else:
            print("   🔇 음성 생성 없음")
        
        self.assertTrue(len(chapters) > 0, "최소한 하나의 챕터는 생성되어야 합니다.")
        
        print("✅ 부기→꼬기 통합 플로우 테스트 성공\n")
        
        return result

    async def test_websocket_streaming_voice(self):
        """WebSocket 스트리밍 음성 생성 테스트"""
        print("\n" + "="*60)
        print("🎵 WebSocket 스트리밍 음성 생성 테스트")
        print("="*60)
        
        try:
            # ChatBotB 생성 (RAG 활성화)
            kogi = ChatBotB(
                output_dir=os.path.join(project_root, "output", "temp"),
                vector_db_path="chatbot/data/vector_db/detailed",
                collection_name="fairy_tales"
            )
            
            # 테스트용 스토리 데이터
            test_story_outline = {
                "theme": "우정",
                "child_name": "지우",
                "plot_summary": "작은 토끼와 친구들의 우정 이야기",
                "educational_value": "협력과 배려"
            }
            
            kogi.set_target_age(6)
            kogi.set_story_outline(test_story_outline)
            
            print("✅ ChatBotB 설정 완료 (RAG 활성화)")
            
            # WebSocket 스트리밍 진행 상황 콜백
            async def streaming_progress_callback(data):
                step = data.get("step", "")
                status = data.get("status", "")
                websocket_mode = data.get("websocket_mode", False)
                
                if "websocket" in step or websocket_mode:
                    if status == "chunk_received":
                        chunk_num = data.get("chunk_number", 0)
                        chunk_size = data.get("chunk_size", 0)
                        voice_id = data.get("voice_id", "")
                        print(f"🎵 WebSocket 청크 {chunk_num}: {chunk_size} bytes ({voice_id})")
                    elif status == "starting":
                        print(f"🚀 WebSocket 스트리밍 시작: {step}")
                    elif status == "completed":
                        total_files = data.get("total_audio_files", 0)
                        print(f"✅ WebSocket 스트리밍 완료: {total_files}개 파일")
            
            # WebSocket 스트리밍 동화 생성
            print("\n🎵 WebSocket 스트리밍 동화 생성 시작...")
            result = await kogi.generate_detailed_story(
                use_enhanced=True,
                use_websocket_voice=True,
                progress_callback=streaming_progress_callback
            )
            
            # 결과 분석
            print(f"\n📊 WebSocket 스트리밍 결과:")
            print(f"   - 상태: {result.get('status', 'unknown')}")
            print(f"   - 스토리 ID: {result.get('story_id', 'N/A')}")
            
            # WebSocket 메타데이터 확인
            voice_metadata = result.get("voice_metadata", {})
            if voice_metadata:
                print(f"\n🎵 WebSocket 음성 메타데이터:")
                print(f"   - WebSocket 사용됨: {voice_metadata.get('websocket_used', False)}")
                print(f"   - 생성된 오디오 파일: {voice_metadata.get('total_audio_files', 0)}")
                print(f"   - 사용된 캐릭터: {voice_metadata.get('characters_used', [])}")
                print(f"   - 총 생성 시간: {voice_metadata.get('total_generation_time', 0):.2f}초")
            
            # 오디오 파일 확인
            audio_files = result.get("audio_paths", [])
            if audio_files:
                print(f"\n🎵 생성된 오디오 파일:")
                for i, audio_file in enumerate(audio_files[:3]):  # 처음 3개만 표시
                    if isinstance(audio_file, dict):
                        chapter_num = audio_file.get("chapter_number", i+1)
                        narration_audio = audio_file.get("narration_audio")
                        dialogues = audio_file.get("dialogue_audios", [])
                        streaming_info = audio_file.get("streaming_metadata", {})
                        
                        print(f"   📖 챕터 {chapter_num}:")
                        if narration_audio:
                            print(f"      - 내레이션: {Path(narration_audio).name}")
                        if dialogues:
                            print(f"      - 대사 개수: {len(dialogues)}")
                        if streaming_info.get("websocket_used"):
                            chunks = streaming_info.get("chunks_received", 0)
                            total_bytes = streaming_info.get("total_bytes", 0)
                            print(f"      - WebSocket: {chunks} chunks, {total_bytes} bytes")
                        
                        # 처음 2개 대사만 표시
                        for j, dialogue in enumerate(dialogues[:2]):
                            speaker = dialogue.get("speaker", "unknown")
                            audio_path = dialogue.get("audio_path", "")
                            voice_id = dialogue.get("voice_id", "")
                            print(f"         - {speaker}: {Path(audio_path).name} ({voice_id})")
            
            # 일반 메타데이터
            metadata = result.get("metadata", {})
            if metadata:
                print(f"\n📈 일반 메타데이터:")
                print(f"   - WebSocket 음성: {metadata.get('websocket_voice', False)}")
                print(f"   - Enhanced 모드: {metadata.get('enhanced_mode', False)}")
                print(f"   - 전체 생성 시간: {metadata.get('generation_time', 0):.2f}초")
                print(f"   - 프롬프트 버전: {metadata.get('prompt_version', 'unknown')}")
            
            print("\n✅ WebSocket 스트리밍 음성 테스트 완료!")
            return True

        except Exception as e:
            print(f"\n❌ WebSocket 스트리밍 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def cleanup(self):
        """테스트 리소스 정리"""
        print("🧹 테스트 리소스 정리 중...")
        # 필요한 경우 여기서 리소스 정리
        print("✅ 테스트 리소스 정리 완료")


# 간단한 헬퍼 함수들
def create_test_audio():
    """독립 실행용 테스트 오디오 생성"""
    CCBIntegratedTest._create_test_audio()

async def run_live_audio_test():
    """라이브 오디오 테스트 (서버 별도 실행 필요)"""
    print("\n=== 라이브 오디오 테스트 ===")
    print("⚠️ 이 테스트는 서버가 별도로 실행 중이어야 합니다.")
    
    if not os.path.exists(SAMPLE_AUDIO_PATH):
        create_test_audio()
    
    # 간단한 WebSocket 클라이언트
    uri = "ws://localhost:8000/ws/audio?child_name=테스트&age=5&interests=공룡&token=development_token"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ 서버 연결 성공")
            
            # 인사말 수신
            greeting = await websocket.recv()
            greeting_data = json.loads(greeting)
            print(f"🤖 인사말: {greeting_data.get('text', '')}")
            
            # 오디오 전송
            with open(SAMPLE_AUDIO_PATH, "rb") as f:
                audio_data = f.read()
            
            print(f"📤 오디오 전송: {len(audio_data)/1024:.1f} KB")
            await websocket.send(audio_data)
            
            # 응답 대기
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            response_data = json.loads(response)
            
            print("📥 응답 수신:")
            print(f"   사용자 음성: {response_data.get('user_text', '')}")
            print(f"   AI 응답: {response_data.get('text', '')}")
            
            # 오디오 응답 저장
            if response_data.get('audio'):
                audio_b64 = response_data.get('audio')
                audio_data = base64.b64decode(audio_b64)
                
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"live_response_{timestamp}.mp3"
                filepath = os.path.join(project_root, "output", "temp", filename)
                
                ensure_directory(os.path.join(project_root, "output", "temp"))
                with open(filepath, "wb") as f:
                    f.write(audio_data)
                
                print(f"🔊 오디오 저장: {filepath}")
            
            print("✅ 라이브 테스트 완료")
    
    except Exception as e:
        print(f"❌ 라이브 테스트 실패: {e}")

async def main():
    """메인 테스트 실행 함수"""
    parser = argparse.ArgumentParser(description='CCB AI 통합 테스트')
    parser.add_argument('--test-bugi', action='store_true', help='부기(ChatBotA) 테스트')
    parser.add_argument('--test-kogi', action='store_true', help='꼬기(ChatBotB) 테스트')
    parser.add_argument('--test-voice', action='store_true', help='웹소켓 음성 시스템 테스트')
    parser.add_argument('--test-integration', action='store_true', help='통합 플로우 테스트')
    parser.add_argument('--test-websocket', action='store_true', help='WebSocket 스트리밍 테스트')
    parser.add_argument('--test-all', action='store_true', help='모든 테스트 실행')
    
    args = parser.parse_args()
    
    # 아무 옵션도 주어지지 않으면 도움말 표시
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    tester = CCBIntegratedTest()
    
    try:
        if args.test_bugi or args.test_all:
            await tester.test_01_bugi_basic_functionality()
        
        if args.test_kogi or args.test_all:
            await tester.test_02_kogi_multimedia_generation()
        
        if args.test_voice or args.test_all:
            await tester.test_03_websocket_voice_functionality()
        
        if args.test_websocket or args.test_all:
            await tester.test_websocket_streaming_voice()
        
        if args.test_integration or args.test_all:
            await tester.test_04_bugi_kogi_integration_flow()
        
        print("\n" + "="*80)
        print("🎉 모든 테스트 완료!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 