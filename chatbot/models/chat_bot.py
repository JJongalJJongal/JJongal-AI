import openai
from openai import OpenAI
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os
import json
import speech_recognition as sr
import pyttsx3
import tempfile
import wave
import pyaudio
import numpy as np
import random

# 환경 변수 설정
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class ChildChatBot:
    """
    아이들과 음성으로 대화하고 동화 주제를 도출하는 AI 챗봇 클래스
    
    Attributes:
        conversation_history (List[Dict]): 대화 내역을 저장하는 리스트 (role, content)
        age_group (int): 아이의 연령대 (4-9세)
        child_name (str): 아이의 이름 (ex: 철수)
        interests (List[str]): 아이의 관심사 목록 (ex: [친구들, 공룡, 토끼 ...])
        recognizer (sr.Recognizer): 음성 인식기
        engine (pyttsx3.Engine): 음성 합성 엔진
        is_listening (bool): 음성 인식 상태
        chatbot_name (str): 챗봇의 이름
    """
    
    def __init__(self):
        """챗봇 초기화 및 기본 속성 설정"""
        self.conversation_history = []  # 대화 내역 저장 (role, content)
        self.age_group = None          # 아이의 연령대 (ex: 4-9세)
        self.child_name = None         # 아이의 이름
        self.interests = []            # 아이의 관심사
        
        
        # 음성 관련 초기화
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.is_listening = False
        
        # 음성 설정
        self.engine.setProperty('rate', 150)    # 말하기 속도
        self.engine.setProperty('volume', 0.9)  # 볼륨
        
        # 음성 인식 설정
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.energy_threshold = 4000  # 음성 감지 임계값
        
        self.chatbot_name = "꼬꼬"
        
        # Random 하게 대화 시작하는 프롬프트 (간단하고 명확한 질문으로 수정)
        self.greeting_templates = [
            "안녕~! {child_name}야! 나는 {chatbot_name}야! {child_name}{postposition} 놀까?",
            "안녕! {child_name}야! 나는 {chatbot_name}라고 해! {child_name}{postposition} 친구할래?",
            "안녕~! {child_name}야! 나는 {chatbot_name}야! {child_name}{postposition} 이야기할까?",
            "안녕! {child_name}야! 나는 {chatbot_name}라고 해! {child_name}{postposition} 모험을 떠날까?",
            "안녕~! {child_name}야! 나는 {chatbot_name}야! {child_name}{postposition} 여행을 떠날까?",
            "안녕! {child_name}야! 나는 {chatbot_name}라고 해! {child_name}{postposition} 이야기를 만들어볼까?",
            "안녕~! {child_name}야! 나는 {chatbot_name}야! {child_name}{postposition} 모험을 떠날까?",
            "안녕! {child_name}야! 나는 {chatbot_name}라고 해! {child_name}{postposition} 마법 이야기를 할까?",
            "안녕~! {child_name}야! 나는 {chatbot_name}야! {child_name}{postposition} 재미있는 일이 있었어?",
            "안녕! {child_name}야! 나는 {chatbot_name}라고 해! {child_name}{postposition} 오늘 하루는 어땠어?",
            "안녕~! {child_name}야! 나는 {chatbot_name}야! {child_name}{postposition} 오늘은 뭐하고 놀았어?",
            "안녕! {child_name}야! 나는 {chatbot_name}라고 해! {child_name}{postposition} 오늘은 어떤 꿈을 꾸고 일어났어?",
            "안녕~! {child_name}야! 나는 {chatbot_name}야! {child_name}{postposition} 오늘은 어떤 색이 좋아?",
            "안녕! {child_name}야! 나는 {chatbot_name}라고 해! {child_name}{postposition} 오늘은 어떤 동물이 생각나?",
            "안녕~! {child_name}야! 나는 {chatbot_name}야! {child_name}{postposition} 오늘은 어떤 음식을 먹었어?"
        ]
    
    def get_korean_postposition(self, name: str) -> str:
        """
        한국어 이름에 맞는 조사를 반환하는 함수
        
        Args:
            name (str): 이름
            
        Returns:
            str: 이름에 맞는 조사 (이/가, 을/를, 와/과, 의)
            
        Note:
            - 이름의 마지막 글자에 따라 적절한 조사를 선택
            - 받침 유무에 따라 다른 조사 사용
        """
        # 이름이 비어있는 경우 처리
        if not name:
            return "와"
            
        # 이름의 마지막 글자
        last_char = name[-1]
        
        # 한글이 아닌 경우 기본값 반환
        if not ord('가') <= ord(last_char) <= ord('힣'):
            return "와"
            
        # 받침 유무 확인 (Unicode 활용)
        # 한글 유니코드에서 받침은 (글자코드 - 0xAC00) % 28로 계산
        # 0이면 받침 없음, 1 이상이면 받침 있음
        has_jongseong = (ord(last_char) - 0xAC00) % 28 > 0
        
        # 조사 선택
        if has_jongseong:
            return "과"  # 받침 있음
        else:
            return "와"  # 받침 없음
    
    def initialize_chat(self, child_name: str, age: int, interests: List[str] = None, chatbot_name: str = "꼬꼬") -> str:
        """
        챗봇과의 대화를 초기화하는 함수
        
        Args:
            child_name (str): 아이의 이름
            age (int): 아이의 나이
            interests (List[str], optional): 아이의 관심사 목록
            chatbot_name (str, optional): 챗봇의 이름
            
        Returns:
            str: 초기 인사 메시지
        """
        self.child_info = {
            "name": child_name,
            "age": age,
            "interests": interests or []
        }
        self.chatbot_name = chatbot_name
        
        # 이름에 맞는 조사 결정
        postposition = self.get_korean_postposition(child_name)
        
        # 랜덤하게 인사말 선택
        greeting_template = random.choice(self.greeting_templates)
        greeting = greeting_template.format(
            chatbot_name=chatbot_name, 
            child_name=child_name,
            postposition=postposition
        )
        
        self.system_message = f"""당신은 {age}세 아이들을 위한 따뜻하고 공감적인 이야기 친구 {chatbot_name}입니다.
        아이의 감정을 이해하고, 상상력을 키우며, 함께 멋진 이야기를 만들어가는 대화를 이끌어가세요.
        
        대화 지침:
        1. 간단하고 명확한 질문하기
           - 한 번에 하나의 질문만 하기
           - 복잡한 질문 대신 간단한 질문 사용
           - 아이가 이해하기 쉬운 언어 사용
        
        2. 진정성 있는 공감과 반응
           - 아이의 감정을 인정하고 공감하기
           - "정말 신나 보여!"
           - "그때 기분이 어땠어?"
           - 아이의 경험을 자세히 듣기
        
        3. 상상력을 자극하는 질문과 제안
           - "만약 ~한다면 어떻게 될까?"
           - "이런 일이 일어난다면 어떨까?"
           - 아이의 답변을 바탕으로 이야기 확장하기
        
        4. 감각적이고 생생한 묘사
           - 시각, 청각, 촉각 등 감각적 요소 포함
           - "빨간색 꽃이 바람에 살랑살랑 흔들리는 걸 상상해봐"
        
        5. 감정과 경험의 연결
           - 아이의 일상 경험과 이야기 연결
           - "친구랑 놀 때도 이런 기분이 들었지?"
           - 긍정적인 감정 강화
        
        6. 안전하고 건전한 대화 유지
           - 긍정적이고 격려하는 톤 사용
           - 아이의 나이에 맞는 어휘 선택
           - 부정적인 감정은 이해하고 긍정적으로 전환
        """
        
        self.add_message("assistant", greeting)
        return greeting
    
    def add_message(self, role: str, content: str):
        """
        대화 내역에 새로운 메시지를 추가하는 함수
        
        Args:
            role (str): 메시지 발신자 역할 ('user' 또는 'assistant')
            content (str): 메시지 내용
            
        Note:
            - 대화 내역을 저장하여 문맥을 유지
            - GPT-4o-mini가 이전 대화 내용을 참고하여 응답할 수 있도록 함
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def get_response(self, user_input: str) -> str:
        """
        사용자의 입력에 대한 AI 응답을 생성하는 함수
        
        Args:
            user_input (str): 사용자의 입력 메시지
            
        Returns:
            str: AI가 생성한 응답 메시지
            
        Note:
            - GPT-4o-mini를 사용하여 문맥을 고려한 응답 생성
            - 대화 내역을 포함하여 일관된 대화 유지
            - 에러 발생 시 사용자 친화적인 에러 메시지 반환
        """
        try:
            # 사용자 메시지 추가
            self.add_message("user", user_input)
            
            # GPT-4o-mini를 사용하여 응답 생성
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_message},
                    *self.conversation_history[-5:]  # 최근 5개의 메시지만 사용
                ],
                temperature=0.9,  # 더 창의적이고 다양한 응답을 위해 temperature 증가
                max_tokens=500
            )
            
            # 응답 추출 및 저장
            assistant_response = response.choices[0].message.content
            self.add_message("assistant", assistant_response)
            
            return assistant_response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "미안해. 지금은 잠시 후 다시 시도해주세요."
    
    def suggest_story_theme(self) -> Dict[str, str]:
        """
        대화 내용을 바탕으로 동화 주제를 제안하는 함수
        
        Returns:
            Dict[str, str]: 제안된 동화 주제 정보
                - theme: 주제
                - description: 설명
                - educational_value: 교육적 가치
                
        Note:
            - 아이의 연령대와 관심사를 고려한 주제 제안
            - 교육적 가치가 있는 주제 선정
            - JSON 형식으로 구조화된 응답 반환
        """
        try:
            # 대화 내용을 바탕으로 주제 제안 프롬프트 생성
            prompt = f"""
            아이의 이름: {self.child_name}
            아이의 연령: {self.age_group}
            아이의 관심사: {', '.join(self.interests)}
            
            {self.child_name}의 대화 내용을 바탕으로 동화 주제를 제안해주세요.
            
            고려해야 할 것:
            1. 아이의 관심사: {', '.join(self.interests)}
            2. 최근 대화 토픽
            3. 아이의 연령에 맞는 주제
            4. 교육적 가치
            
            Return a JSON with:
            - theme: Main theme of the story
            - description: Brief description of the suggested story
            - educational_value: What the child can learn from this story
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a children's story expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Error suggesting story theme: {str(e)}")
            return {
                "theme": "우정과 용기",
                "description": "친구들과 함께 모험을 하는 이야기",
                "educational_value": "협동과 용기의 가치를 배울 수 있습니다."
            }
    
    def get_conversation_summary(self) -> str:
        """
        대화 내용을 요약하는 함수
        
        Returns:
            str: 대화 내용 요약
            
        Note:
            - 주요 토픽, 관심사, 학습 순간 등을 요약
            - 동화 주제 도출을 위한 인사이트 제공
            - 대화 분석을 통한 교육적 가치 추출
        """
        try:
            # 대화 내용 요약 프롬프트 생성
            prompt = f"""
            Summarize the conversation with {self.child_name} (age {self.age_group}),
            focusing on:
            1. Main topics discussed
            2. Child's interests and preferences
            3. Key learning moments
            4. Potential story themes
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a conversation analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating conversation summary: {str(e)}")
            return "대화 내용을 요약할 수 없습니다."
    
    def save_conversation(self, file_path: str):
        """
        대화 내용을 파일로 저장하는 함수
        
        Args:
            file_path (str): 저장할 파일 경로
            
        Note:
            - 아이 정보, 대화 내역, 요약을 JSON 형식으로 저장
            - 나중에 대화 내용을 불러와서 분석하거나 이어서 대화 가능
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "child_info": {
                        "name": self.child_name,
                        "age": self.age_group,
                        "interests": self.interests
                    },
                    "conversation": self.conversation_history,
                    "summary": self.get_conversation_summary()
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving conversation: {str(e)}")
    
    def load_conversation(self, file_path: str):
        """
        저장된 대화 내용을 불러오는 함수
        
        Args:
            file_path (str): 불러올 파일 경로
            
        Note:
            - 저장된 아이 정보와 대화 내역을 복원
            - 이전 대화를 이어서 진행할 수 있도록 함
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.child_name = data["child_info"]["name"]
                self.age_group = data["child_info"]["age"]
                self.interests = data["child_info"]["interests"]
                self.conversation_history = data["conversation"]
        except Exception as e:
            print(f"Error loading conversation: {str(e)}")
    
    def speak(self, text: str):
        """
        텍스트를 음성으로 변환하여 출력하는 함수
        
        Args:
            text (str): 변환할 텍스트
            
        Note:
            - pyttsx3를 사용하여 텍스트를 음성으로 변환
            - 아이의 연령대에 맞는 속도와 톤으로 출력
        """
        try:
            # 연령대에 따른 음성 속도 조정
            if self.age_group <= 5:
                self.engine.setProperty('rate', 130)  # 더 천천히
            elif self.age_group <= 8:
                self.engine.setProperty('rate', 150)  # 중간 속도
            else:
                self.engine.setProperty('rate', 170)  # 더 빠르게
            
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error in speech synthesis: {str(e)}")
    
    def listen(self) -> Optional[str]:
        """
        마이크로 입력된 음성을 텍스트로 변환하는 함수
        
        Returns:
            Optional[str]: 변환된 텍스트 또는 None
            
        Note:
            - 음성 인식 실패 시 None 반환
            - 배경 소음 조정 및 에러 처리 포함
        """
        try:
            with sr.Microphone() as source:
                print("듣고 있어요...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                # Google Speech Recognition 사용
                text = self.recognizer.recognize_google(audio, language='ko-KR')
                print(f"인식된 텍스트: {text}")
                return text
                
        except sr.WaitTimeoutError:
            print("음성이 감지되지 않았습니다.")
            return None
        except sr.UnknownValueError:
            print("음성을 인식할 수 없습니다.")
            return None
        except sr.RequestError as e:
            print(f"음성 인식 서비스 오류: {str(e)}")
            return None
        except Exception as e:
            print(f"음성 인식 중 오류 발생: {str(e)}")
            return None
    
    def start_voice_chat(self):
        """
        음성 대화를 시작하는 함수
        
        Note:
            - 음성 인식 및 응답을 반복적으로 수행
            - '종료' 또는 '끝내기' 입력 시 대화 종료
        """
        self.is_listening = True
        print(f"{self.chatbot_name}: 안녕하세요! {self.child_name}와(과) 대화를 시작할게요.")
        self.speak(f"안녕하세요! {self.child_name}와(과) 대화를 시작할게요.")
        
        while self.is_listening:
            # 음성 입력 받기
            user_input = self.listen()
            
            if user_input is None:
                continue
                
            # 종료 명령 확인
            if user_input in ['종료', '끝내기', '그만']:
                self.speak("오늘도 즐거운 대화 감사합니다. 다음에 또 만나요!")
                self.is_listening = False
                break
            
            # AI 응답 생성 및 음성 출력
            response = self.get_response(user_input)
            self.speak(response)
    
    def stop_voice_chat(self):
        """
        음성 대화를 종료하는 함수
        
        Note:
            - 음성 인식 상태를 False로 변경하여 대화 종료
        """
        self.is_listening = False
        print("음성 대화가 종료되었습니다.") 