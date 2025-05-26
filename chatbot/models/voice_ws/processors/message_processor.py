"""
Message 처리 프로세서 - 사용자 메시지 처리 및 챗봇 응답 생성
    
"""
import asyncio # 비동기 작업 대기
from typing import Union

class STTClient:
    async def transcribe(self, audio_data: bytes) -> str:
        print("STTClient -- 음성 -> 텍스트 변환")
        await asyncio.sleep(0.1) # 비동기 작업 대기
        return "사용자가 말한 내용"

class ChatbotClient:
    async def get_response(self, text_input: str) -> str:
        print(f"ChatbotClient -- 텍스트 -> 응답 생성")
        await asyncio.sleep(0.1) # 비동기 작업 대기
        return f"챗봇 응답: {text_input}"

class TTSClient:
    async def synthesize(self, text_input: str) -> bytes:
        print(f"TTSClient -- 텍스트 -> 음성 변환")
        await asyncio.sleep(0.1) # 비동기 작업 대기
        return b"synthesized_audio_data"

class MessageProcessor:
    def __init__(self):
        self.stt_client = STTClient() # 음성 -> 텍스트 변환 클라이언트
        self.chatbot_client = ChatbotClient() # 텍스트 -> 응답 생성 클라이언트
        self.tts_client = TTSClient() # 텍스트 -> 음성 변환 클라이언트
        
    async def process_message(self, message: Union[bytes, str]) -> bytes:
        """
        사용자 메시지를 처리하고 챗봇 응답을 생성.
        
        Args:
            message (Union[bytes, str]): 사용자 메시지 (음성 또는 텍스트)
        """
        if isinstance(message, bytes): # 만약 메시지가 음성 형태라면
            print("MessageProcessor -- 음성 -> 텍스트 변환")
            text_input = await self.stt_client.transcribe(message) # 음성 -> 텍스트 변환
            print(f"MessageProcessor -- 텍스트: '{text_input}'")
        elif isinstance(message, str): # 만약 메시지가 텍스트 형태라면
            print("MessageProcessor -- 텍스트 메시지 수신")
            text_input = message
        else:
            print(f"MessageProcessor -- 지원하지 않는 메시지 타입: {type(message)}. 오류 반환.") # 로깅
            return b"Error: Unsupported message type" # 오류 반환

        print(f"MessageProcessor -- 챗봇 응답 생성: '{text_input}'")
        chatbot_response_text = await self.chatbot_client.get_response(text_input) # 챗봇 응답 생성
        print(f"MessageProcessor -- 챗봇 응답: '{chatbot_response_text}'")

        print(f"MessageProcessor -- 음성 변환: '{chatbot_response_text}'") # 로깅
        audio_response = await self.tts_client.synthesize(chatbot_response_text) # 음성 합성
        print("MessageProcessor -- 음성 변환 완료")

        return audio_response # 음성 합성된 오디오 응답 반환
        