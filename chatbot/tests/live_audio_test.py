#!/usr/bin/env python3
"""
실시간 마이크 오디오 WebSocket 테스트 클라이언트
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

# 오디오 설정
CHUNK = 4096  # 청크 크기
FORMAT = pyaudio.paInt16  # 오디오 포맷
CHANNELS = 1  # 모노
RATE = 16000  # 샘플레이트 (Whisper 권장)
RECORD_SECONDS = 2  # 보내기 전 녹음 시간

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
        
        # 저장된 응답 오디오 관리
        self.responses_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "responses")
        os.makedirs(self.responses_dir, exist_ok=True)
        
        # PyAudio 초기화
        self.p = pyaudio.PyAudio()
        
    async def connect(self):
        """WebSocket 연결 생성"""
        # 쿼리 파라미터 구성
        query_params = f"token={self.token}&child_name={self.child_name}&age={self.age}"
        if self.interests:
            interests_str = ",".join(self.interests)
            query_params += f"&interests={interests_str}"
            
        # WebSocket 연결
        uri = f"{self.server_url}?{query_params}"
        self.websocket = await websockets.connect(uri)
        print(f"서버 {self.server_url}에 연결됨")
        
        # 인사말 메시지 수신
        greeting = await self.websocket.recv()
        greeting_data = json.loads(greeting)
        print(f"\n챗봇: {greeting_data.get('text', '')}")
        
        # 오디오 응답이 있으면 재생
        if greeting_data.get('audio'):
            await self._save_and_play_audio(greeting_data['audio'], "greeting")
            
        return True
        
    async def _save_and_play_audio(self, audio_b64, prefix):
        """Base64 인코딩된 오디오를 저장하고 재생"""
        if not audio_b64:
            return False
            
        try:
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.wav"
            filepath = os.path.join(self.responses_dir, filename)
            
            # 오디오 데이터 디코딩 및 저장
            audio_data = base64.b64decode(audio_b64)
            with open(filepath, "wb") as f:
                f.write(audio_data)
                
            print(f"오디오 응답 저장됨: {filepath}")
            
            # 오디오 재생 (macOS 'afplay' 사용)
            os.system(f"afplay {filepath}")
            return True
        
        except Exception as e:
            print(f"오디오 처리 오류: {e}")
            return False
            
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
        
        try:
            while self.is_streaming:
                # 오디오 데이터 수집
                frames = []
                for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                
                # WAV 형식으로 변환
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(self.p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                
                # 서버로 전송
                wav_data = wav_buffer.getvalue()
                print(f"\r오디오 데이터 전송 중... ({len(wav_data)} 바이트)", end="")
                
                await self.websocket.send(wav_data)
                
                # 서버 응답 처리
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                    response_data = json.loads(response)
                    
                    # 응답 유형에 따른 처리
                    if response_data.get('type') == 'ai_response':
                        print("\n")  # 줄바꿈
                        print(f"인식된 텍스트: {response_data.get('user_text', '')}")
                        print(f"챗봇: {response_data.get('text', '')}")
                        
                        # 오디오 재생
                        if response_data.get('audio'):
                            await self._save_and_play_audio(response_data['audio'], "response")
                    
                    elif response_data.get('type') == 'error':
                        print(f"\n오류: {response_data.get('error_message', '알 수 없는 오류')}")
                    
                    elif response_data.get('type') == 'ping':
                        print("\n서버 연결 확인")
                
                except asyncio.TimeoutError:
                    print("\n서버 응답 대기 중...")
                
        except KeyboardInterrupt:
            print("\n녹음 중단됨")
        finally:
            # 스트리밍 종료 및 리소스 정리
            self.is_streaming = False
            stream.stop_stream()
            stream.close()
            
    async def disconnect(self):
        """연결 종료 및 리소스 정리"""
        if self.websocket:
            await self.websocket.close()
            print("WebSocket 연결 종료")
        
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
    
    # 입력값 검증
    if args.age < 4 or args.age > 9:
        print("나이는 4-9세 사이여야 합니다.")
        return
        
    # 관심사 처리
    interests_list = args.interests.split(',') if args.interests else []
    
    # 출력 디버깅
    print(f"WebSocket 서버에 연결 시도: {args.server}")
    print(f"아이 이름: {args.name}, 나이: {args.age}")
    print(f"관심사: {', '.join(interests_list) if interests_list else '없음'}")
    
    # 비동기 이벤트 루프 생성 및 실행
    client = AudioStreamClient(
        server_url=args.server,
        token=args.token,
        child_name=args.name,
        age=args.age,
        interests=interests_list
    )
    
    asyncio.run(client.run())


if __name__ == "__main__":
    main() 