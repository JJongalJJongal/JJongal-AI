#!/usr/bin/env python3
"""
실시간 마이크 오디오 WebSocket 테스트 클라이언트
(개선된 버전)
"""

import asyncio
import json
import argparse
import pyaudio
import websockets
import wave
import io
import sys
import os
import base64
from datetime import datetime
import signal
import time
import numpy as np

# 오디오 설정 (Whisper 권장 값으로 최적화)
CHUNK = 1024  # 청크 크기 (작게 조정하여 지연 감소)
FORMAT = pyaudio.paInt16  # 오디오 포맷
CHANNELS = 1  # 모노
RATE = 16000  # 샘플레이트 (Whisper 권장)
RECORD_SECONDS = 2  # 보내기 전 녹음 시간
MAX_VOLUME = 32767  # int16 최대값

class AudioStreamClient:
    """실시간 오디오 캡처 및 WebSocket 전송 클라이언트"""

    def __init__(self, server_url, token, child_name, age, interests=None):
        """클라이언트 초기화"""
        self.server_url = server_url
        self.token = token
        self.child_name = child_name
        self.age = age
        self.interests = interests or []
        self.is_streaming = False
        self.websocket = None
        self.ping_task = None
        self.silent_frames = 0  # 무음 프레임 카운터
        
        # 저장된 응답 오디오 관리
        self.responses_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "responses")
        os.makedirs(self.responses_dir, exist_ok=True)
        
        # PyAudio 초기화
        self.p = pyaudio.PyAudio()
        
        # 시그널 핸들러 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, sig, frame):
        """Ctrl+C 처리 함수"""
        print("\n\n프로그램 종료 중...")
        self.is_streaming = False
        
    async def connect(self):
        """WebSocket 연결 생성"""
        # 쿼리 파라미터 구성
        query_params = f"token={self.token}&child_name={self.child_name}&age={self.age}"
        if self.interests:
            interests_str = ",".join(self.interests)
            query_params += f"&interests={interests_str}"
            
        # WebSocket 연결
        uri = f"{self.server_url}?{query_params}"
        try:
            print(f"서버 {self.server_url}에 연결 중...")
            self.websocket = await websockets.connect(uri)
            print(f"서버에 연결됨")
            
            # Ping 작업 시작
            self.ping_task = asyncio.create_task(self._ping_server())
            
            # 인사말 메시지 수신
            greeting = await self.websocket.recv()
            greeting_data = json.loads(greeting)
            print(f"\n챗봇: {greeting_data.get('text', '')}")
            
            # 오디오 응답이 있으면 재생
            if greeting_data.get('audio'):
                await self._save_and_play_audio(greeting_data['audio'], "greeting")
                
            return True
        except Exception as e:
            print(f"연결 실패: {e}")
            return False
        
    async def _ping_server(self):
        """서버 연결 유지를 위한 주기적 ping 전송"""
        while self.is_streaming and self.websocket:
            try:
                await asyncio.sleep(25)  # 25초마다 핑
                if self.websocket and self.is_streaming:
                    await self.websocket.ping()
            except Exception:
                # 오류 무시 (연결 종료 시 자동 중단)
                pass
            
    async def _save_and_play_audio(self, audio_b64, prefix):
        """Base64 인코딩된 오디오를 저장하고 재생"""
        if not audio_b64:
            return False
            
        try:
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.mp3"
            filepath = os.path.join(self.responses_dir, filename)
            
            # 오디오 데이터 디코딩 및 저장
            audio_data = base64.b64decode(audio_b64)
            with open(filepath, "wb") as f:
                f.write(audio_data)
                
            print(f"오디오 응답 저장됨: {filepath}")
            
            # 오디오 재생 (운영체제 확인)
            if sys.platform == "darwin":  # macOS
                os.system(f"afplay {filepath}")
            elif sys.platform == "win32":  # Windows
                os.system(f"start {filepath}")
            else:  # Linux 및 기타
                os.system(f"mpg123 -q {filepath}")
                
            return True
        
        except Exception as e:
            print(f"오디오 처리 오류: {e}")
            return False
            
    def _is_silent(self, data, threshold=0.03):
        """오디오 데이터가 무음인지 확인"""
        as_int = np.frombuffer(data, dtype=np.int16)
        # 최대 볼륨 대비 비율 계산
        if len(as_int) == 0:
            return True
        max_sample = max(abs(int(sample)) for sample in as_int)
        return max_sample / MAX_VOLUME < threshold
            
    async def stream_microphone(self):
        """마이크로부터 오디오 스트리밍"""
        self.is_streaming = True
        
        # 오디오 스트림 열기
        stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print("\n마이크 녹음 시작 - 말씀해주세요. (종료: Ctrl+C)")
        print("(음성이 들리지 않으면 마이크를 확인해주세요)")
        
        # 상태 표시 기호
        status_symbols = ["◐", "◓", "◑", "◒"]
        symbol_idx = 0
        
        try:
            while self.is_streaming:
                # 오디오 데이터 수집
                frames = []
                is_silent = True
                
                # 상태 표시
                print(f"\r{status_symbols[symbol_idx]} 듣는 중...", end="")
                symbol_idx = (symbol_idx + 1) % len(status_symbols)
                
                # 청크별 수집
                for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                    
                    # 무음 여부 확인을 위해 numpy 임포트 (필요한 경우에만)
                    if 'np' not in globals():
                        import numpy as np
                    
                    # 현재 청크가 무음인지 확인
                    if not self._is_silent(data):
                        is_silent = False
                
                # 무음 프레임 처리
                if is_silent:
                    self.silent_frames += 1
                    if self.silent_frames > 5:  # 10초 이상 무음이면 메시지 출력
                        if self.silent_frames % 5 == 0:  # 10초마다 알림
                            print("\r무음이 감지됩니다. 말씀해주세요...      ", end="")
                    continue  # 무음이면 서버로 전송하지 않음
                else:
                    # 소리가 감지되면 카운터 리셋
                    self.silent_frames = 0
                
                # WAV 형식으로 변환
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(self.p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                
                # 서버로 전송
                wav_data = wav_buffer.getvalue()
                print(f"\r오디오 데이터 전송 중... ({len(wav_data)/1024:.1f} KB)", end="")
                
                try:
                    await self.websocket.send(wav_data)
                    
                    # 서버 응답 처리
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
                    response_data = json.loads(response)
                    
                    # 응답 유형에 따른 처리
                    if response_data.get('type') == 'ai_response':
                        print("\n")  # 줄바꿈
                        user_text = response_data.get('user_text', '')
                        bot_text = response_data.get('text', '')
                        
                        if user_text:
                            print(f"인식된 텍스트: 「{user_text}」")
                        print(f"챗봇: {bot_text}")
                        
                        # 오디오 재생
                        if response_data.get('audio'):
                            await self._save_and_play_audio(response_data['audio'], "response")
                    
                    elif response_data.get('type') == 'error':
                        error_msg = response_data.get('error_message', '알 수 없는 오류')
                        error_code = response_data.get('error_code', '')
                        print(f"\n오류 발생: {error_msg} (코드: {error_code})")
                    
                    elif response_data.get('type') == 'ping':
                        print("\r연결 확인 중...", end="")
                
                except asyncio.TimeoutError:
                    print("\n서버 응답 대기 시간 초과. 다시 시도합니다...")
                except websockets.exceptions.ConnectionClosed:
                    print("\n서버와의 연결이 종료되었습니다.")
                    break
                except Exception as e:
                    print(f"\n오류 발생: {e}")
                
        except KeyboardInterrupt:
            print("\n녹음 중단됨")
        finally:
            # 스트리밍 종료 및 리소스 정리
            self.is_streaming = False
            stream.stop_stream()
            stream.close()
            
    async def disconnect(self):
        """연결 종료 및 리소스 정리"""
        self.is_streaming = False
        
        # Ping 작업 중단
        if self.ping_task:
            self.ping_task.cancel()
            try:
                await self.ping_task
            except asyncio.CancelledError:
                pass
        
        # WebSocket 연결 종료
        if self.websocket:
            await self.websocket.close()
            print("WebSocket 연결 종료")
        
        # PyAudio 종료
        if hasattr(self, 'p'):
            self.p.terminate()
            print("오디오 리소스 해제")
        
    async def run(self):
        """메인 실행 함수"""
        try:
            # 서버 연결
            connected = await self.connect()
            if not connected:
                return
            
            # 마이크 스트리밍 시작
            await self.stream_microphone()
            
        except Exception as e:
            print(f"실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 연결 종료
            await self.disconnect()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="실시간 마이크 오디오 WebSocket 테스트 클라이언트")
    parser.add_argument("--server", default="ws://127.0.0.1:8000/ws/audio", help="WebSocket 서버 URL")
    parser.add_argument("--token", default="valid_token", help="인증 토큰")
    parser.add_argument("--name", required=True, help="아이 이름")
    parser.add_argument("--age", type=int, required=True, help="아이 나이 (4-9세)")
    parser.add_argument("--interests", help="관심사 (쉼표로 구분)")
    
    args = parser.parse_args()
    
    # 관심사 처리
    interests = []
    if args.interests:
        interests = [interest.strip() for interest in args.interests.split(",")]
    
    # 비동기 실행
    client = AudioStreamClient(
        server_url=args.server,
        token=args.token,
        child_name=args.name,
        age=args.age,
        interests=interests
    )
    
    # 실행
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\n프로그램이 종료되었습니다.")
    
if __name__ == "__main__":
    main() 