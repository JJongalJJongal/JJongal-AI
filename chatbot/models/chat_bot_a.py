from openai import OpenAI
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os
import json
import random
from pathlib import Path

# 환경 변수 설정
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class StoryCollectionChatBot:
    """
    아이들과 대화하며 동화 줄거리를 수집하는 AI 챗봇 클래스
    
    Attributes:
        conversation_history (List[Dict]): 대화 내역을 저장하는 리스트 (role, content)
        age_group (int): 아이의 연령대 (4-9세)
        child_name (str): 아이의 이름
        interests (List[str]): 아이의 관심사 목록
        chatbot_name (str): 챗봇의 이름
        prompts (Dict): JSON 파일에서 로드한 프롬프트
        story_outline (Dict): 수집된 이야기 줄거리
    """
    
    def __init__(self):
        """챗봇 초기화 및 기본 속성 설정"""
        self.conversation_history = []  # 대화 내역 저장 (role, content)
        self.age_group = None          # 아이의 연령대 (ex: 4-9세)
        self.child_name = None         # 아이의 이름
        self.interests = []            # 아이의 관심사
        
        self.chatbot_name = "부기"
        
        # 프롬프트 로드
        self.prompts = self._load_prompts()   
        
        # 인사말 템플릿
        self.greeting_template = self.prompts.get('greeting_template', '')
        
        # 이야기 줄거리 초기화
        self.story_outline = None
    
    def _load_prompts(self) -> Dict:
        """프롬프트 JSON 파일을 로드하는 메서드"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_path = os.path.join(current_dir, '..', 'data', 'prompts', 'chatbot_prompts.json')
            
            with open(prompts_path, 'r', encoding='utf-8') as f:
                prompts = json.load(f)
                return prompts['chatbot_a']
        except Exception as e:
            print(f"프롬프트 로드 중 오류 발생: {e}")
            return {}
    
    # 아이 정보 업데이트하는 Function
    def update_child_info(self, child_name: str = None, age: int = None, interests: List[str] = None):
        if child_name:
            self.child_name = child_name
        if age:
            self.age_group = age
        if interests:
            self.interests = interests
    
    
    def _has_final_consonant(self, char: str) -> bool:
        """
        한글 문자의 마지막 음절이 받침을 갖는지 확인하는 메서드
        
        Args:
            char (str): 확인할 한글 문자
            
        Returns:
            bool: 받침이 있으면 True, 없으면 False
        """
        if not char:
            return False
            
        # 마지막 문자 추출
        last_char = char[-1]
        
        # 한글이 아닌 경우 기본값 반환
        if not ord('가') <= ord(last_char) <= ord('힣'):
            return False
            
        # 받침 유무 확인
        return (ord(last_char) - 0xAC00) % 28 > 0
    
    def _get_josa(self, word: str, josa_type: str) -> str:
        """
        단어에 맞는 조사를 반환하는 메서드
        
        Args:
            word (str): 조사를 붙일 단어
            josa_type (str): 조사 유형 ('은/는', '이/가', '을/를', '와/과', '으로/로')
            
        Returns:
            str: 선택된 조사
        """
        has_final = self._has_final_consonant(word)
        
        josa_map = {
            '은/는': '은' if has_final else '는',
            '이/가': '이' if has_final else '가',
            '을/를': '을' if has_final else '를',
            '와/과': '과' if has_final else '와',
            '으로/로': '으로' if has_final else '로'
        }
        
        return josa_map.get(josa_type, '')
    
    def format_with_josa(self, word: str, josa_type: str) -> str:
        """
        단어에 조사를 붙여 반환하는 메서드
        
        Args:
            word (str): 조사를 붙일 단어
            josa_type (str): 조사 유형
            
        Returns:
            str: 조사가 붙은 단어
        """
        josa = self._get_josa(word, josa_type)
        return f"{word}{josa}"
    
    def get_system_message(self) -> str:
        """시스템 메시지 반환"""
        return self.prompts.get('system_message_template', '')
    
    def get_greeting(self, child_name: Optional[str] = None) -> str:
        """
        랜덤한 인사말 반환
        
        Args:
            child_name (Optional[str]): 아이의 이름
            
        Returns:
            str: 인사말
        """
        greetings = self.prompts.get('greeting_templates', [])
        greeting = random.choice(greetings) if greetings else "안녕하세요!"
        
        if child_name:
            # 이름에 맞는 조사 선택
            josa = self._get_josa(child_name, '은/는')
            greeting = greeting.replace("{name}", f"{child_name}{josa}")
            
        return greeting
    
    # 후속 질문을 통해 이야기를 이어 나가도록 유도하는 Function
    def get_follow_up_question(self, child_name: Optional[str] = None) -> str:
        """
        랜덤한 후속 질문 반환
        
        Args:
            child_name (Optional[str]): 아이의 이름
            
        Returns:
            str: 후속 질문
        """
        questions = self.prompts.get('follow_up_questions', [])
        question = random.choice(questions) if questions else "더 자세히 이야기해 주세요."
        
        if child_name:
            # 이름에 맞는 조사 선택
            josa = self._get_josa(child_name, '이/가')
            question = question.replace("{name}", f"{child_name}{josa}")
            
        return question
    
    # 아이의 답변에 대해 격려하며 대화를 계속하도록 유도하는 Function
    def get_encouragement(self, child_name: Optional[str] = None) -> str:
        """
        랜덤한 격려 문구 반환
        
        Args:
            child_name (Optional[str]): 아이의 이름
            
        Returns:
            str: 격려 문구
        """
        encouragements = self.prompts.get('encouragement_phrases', [])
        encouragement = random.choice(encouragements) if encouragements else "좋아요!"
        
        if child_name:
            # 이름에 맞는 조사 선택
            josa = self._get_josa(child_name, '이/가')
            encouragement = encouragement.replace("{name}", f"{child_name}{josa}")
            
        return encouragement
    
    # 아이의 연령대에 맞는 언어를 사용하여 대화하는 Function
    def get_age_appropriate_language(self, age: int) -> Dict:
        """연령대에 맞는 언어 설정 반환"""
        age_ranges = self.prompts.get('age_appropriate_language', {})
        
        if 4 <= age <= 5:
            return age_ranges.get('4-5', {})
        elif 6 <= age <= 7:
            return age_ranges.get('6-7', {})
        elif 8 <= age <= 9:
            return age_ranges.get('8-9', {})
        else:
            return {}
    
    def format_story_collection_prompt(self, **kwargs) -> str:
        """스토리 수집 프롬프트 포맷팅"""
        template = self.prompts.get('story_collection_prompt_template', '')
        return template.format(**kwargs)
    
    def add_to_conversation(self, role: str, content: str):
        """대화 내역에 메시지 추가"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def get_conversation_history(self) -> List[Dict]:
        """대화 내역 반환"""
        return self.conversation_history
    
    def get_korean_postposition(self, name: str) -> str:
        """
        한국어 이름에 맞는 조사를 반환하는 함수
        
        Args:
            name (str): 이름
            
        Returns:
            str: 이름에 맞는 조사 (이/가, 을/를, 와/과, 의)
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
        has_jongseong = (ord(last_char) - 0xAC00) % 28 > 0
        
        # 조사 선택
        if has_jongseong:
            return "과"  # 받침 있음
        else:
            return "와"  # 받침 없음
    
    def initialize_chat(self, child_name: str, age: int, interests: List[str] = None, chatbot_name: str = "부기") -> str:
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
        # 입력 검증 및 기본값 설정
        if not child_name or not isinstance(child_name, str):
            raise ValueError("아이의 이름은 필수고, 문자열이어야 합니다.")
        if not isinstance(age, int) or age < 4 or age > 9:
            raise ValueError("아이의 나이는 4-9세 사이 정수이어야 합니다.")
        if interests and not isinstance(interests, list):
            raise ValueError("관심사는 리스트 형태여야 합니다.")
        if chatbot_name and not isinstance(chatbot_name, str):
            raise ValueError("챗봇의 이름은 문자열 형태여야 합니다.")
        
        self.update_child_info(child_name, age, interests)
        
        # 이름에 맞는 조사 결정
        postposition = self.get_korean_postposition(child_name)
        
        # 랜덤하게 인사말 선택
        greeting = self.greeting_template.format(
            chatbot_name=chatbot_name, 
            child_name=child_name,
            postposition=postposition
        )
        
        # 시스템 메시지 설정
        self.system_message = self.prompts["system_message_template"].format(
            age=age,
            chatbot_name=chatbot_name
        )
        
        self.add_to_conversation("assistant", greeting)
        return greeting
    
    def get_response(self, user_input: str) -> str:
        """
        사용자의 입력에 대한 AI 응답을 생성하는 함수
        
        Args:
            user_input (str): 사용자의 입력 메시지
            
        Returns:
            str: AI가 생성한 응답 메시지
        """
        try:
            # 사용자 메시지 추가
            self.add_to_conversation("user", user_input)
            
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
            self.add_to_conversation("assistant", assistant_response)
            
            return assistant_response
            
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {str(e)}")
            return f"미안해 {self.child_name}야. 조금만 후에 다시 이야기할까?"
    
    # 대화 내용을 바탕으로 동화 줄거리 요약 및 Tag 추출 Function
    def collect_story_outline(self) -> Dict:
        """
        대화 내용을 바탕으로 동화 줄거리 요약 및 태그를 추출하는 함수
        
        Returns:
            Dict: 수집된 정보 (summary_text, tags)
        """
        try:
            # 대화 내용을 바탕으로 줄거리 및 태그 추출 프롬프트 생성
            prompt = f"""
            대화 내용:
            {self.conversation_history}
            
            위 정보를 바탕으로 다음 두 가지를 추출해주세요:
            1.  간단한 동화 줄거리 (summary_text)
            2.  이야기와 관련된 핵심 태그 (tags) - 쉼표로 구분된 문자열 형태 (예: 공룡,모험,친구)
            
            출력 형식 (JSON):
            {{
                "summary_text": "[생성된 줄거리]",
                "tags": "[추출된 태그]"
            }}
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "너는 아이와의 대화 내용을 바탕으로 동화의 대략적인 줄거리와 관련된 주제 태그를 추출하는 유치원 선생님이야."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=600,
                response_format={"type": "json_object"} # JSON 형식으로 응답 요청
            )
            
            # 응답에서 JSON 파싱
            result = json.loads(response.choices[0].message.content)
            
            # 관심사를 태그에 추가 (선택 사항)
            # existing_tags = set(result.get("tags", "").split(','))
            # existing_tags.update(self.interests)
            # result["tags"] = ",".join(filter(None, existing_tags))
            
            return result
            
        except Exception as e:
            print(f"동화 줄거리 및 태그 추출 중 오류 발생: {str(e)}")
            # 기본값 반환 시 아이의 관심사를 태그로 사용
            return {
                "summary_text": "주인공이 친구와 함께 모험을 하며 용기와 우정의 가치를 배우는 이야기입니다.",
                "tags": ",".join(self.interests) if self.interests else "모험,우정"
            }
    
    def get_conversation_summary(self) -> str:
        """
        대화 내용을 요약하는 함수
        
        Returns:
            str: 대화 내용 요약
        """
        try:
            # 대화 내용 요약 프롬프트 생성
            prompt = f"""
            {self.child_name}({self.age_group}세)와의 대화를 요약해줘.
            다음 사항에 중점을 두고 요약해줘:
            1. 주요 토픽 ()
            2. 아이의 관심사와 선호도
            3. 주요 학습 순간
            4. 잠재적인 동화 주제
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 대화 분석 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"대화 요약 중 오류 발생: {str(e)}")
            return "대화 내용을 요약할 수 없습니다."
    
    def save_conversation(self, file_path: str):
        """
        대화 내용을 파일로 저장하는 함수
        
        Args:
            file_path (str): 저장할 파일 경로
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "summary": self.get_conversation_summary(),
                    "story_outline": self.story_outline
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"대화 저장 중 오류 발생: {str(e)}")
    
    def load_conversation(self, file_path: str):
        """
        저장된 대화 내용을 불러오는 함수
        
        Args:
            file_path (str): 불러올 파일 경로
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.child_name = data["child_info"]["name"]
                self.age_group = data["child_info"]["age"]
                self.interests = data["child_info"]["interests"]
                self.conversation_history = data["conversation"]
                self.story_outline = data["story_outline"]
        except Exception as e:
            print(f"대화 불러오기 중 오류 발생: {str(e)}") 