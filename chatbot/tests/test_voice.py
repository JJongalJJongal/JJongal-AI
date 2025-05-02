import sys
import os
import asyncio
import websockets
import json
import base64
import argparse
from pathlib import Path

# 상위 디렉토리 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 서버 실행 관련 모듈
from models.voice_ws_server import app
import uvicorn
from threading import Thread
import time

# 테스트용 오디오 파일 경로
SAMPLE_AUDIO_PATH = os.path.join(current_dir, "test_audio.wav")

async def test_websocket_connection():
    """WebSocket 연결 테스트"""
    uri = "ws://localhost:8000/ws/audio?token=valid_token&child_name=민준&age=5&interests=공룡,우주"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("WebSocket 연결 성공!")
            
            # 인사말 수신 (서버에서 자동으로 보내는 메시지)
            greeting_response = await websocket.recv()
            greeting_data = json.loads(greeting_response)
            print(f"인사말: {greeting_data.get('text', '')}")
            
            # 샘플 오디오 존재 확인
            if os.path.exists(SAMPLE_AUDIO_PATH):
                # 샘플 오디오 파일 전송
                with open(SAMPLE_AUDIO_PATH, "rb") as audio_file:
                    audio_data = audio_file.read()
                
                # 오디오 데이터 전송
                await websocket.send(audio_data)
                print("샘플 오디오 전송 완료")
                
                # 응답 대기
                response = await websocket.recv()
                response_data = json.loads(response)
                
                # 응답 분석
                print("\n서버 응답:")
                print(f"AI 응답 텍스트: {response_data.get('text', '')}")
                print(f"사용자 음성 인식: {response_data.get('user_text', '')}")
                print(f"상태: {response_data.get('status', '')}")
                
                if "error_message" in response_data:
                    print(f"오류: {response_data.get('error_message')}")
            else:
                print(f"샘플 오디오 파일이 없습니다: {SAMPLE_AUDIO_PATH}")
                print("테스트 오디오 파일을 생성하거나 경로를 수정해주세요.")
    
    except Exception as e:
        print(f"WebSocket 연결 중 오류 발생: {e}")

def run_server():
    """음성 서버 실행"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

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

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="음성 챗봇 테스트")
    parser.add_argument("--create-audio", action="store_true", help="테스트용 오디오 파일 생성")
    parser.add_argument("--run-server", action="store_true", help="서버 실행")
    args = parser.parse_args()
    
    if args.create_audio:
        create_test_audio()
    
    if args.run_server:
        # 서버를 별도 스레드로 실행
        server_thread = Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        print("서버 시작 중... (5초 대기)")
        time.sleep(5)  # 서버가 시작될 때까지 대기
    
    # WebSocket 테스트 실행
    asyncio.run(test_websocket_connection())

if __name__ == "__main__":
    main() 