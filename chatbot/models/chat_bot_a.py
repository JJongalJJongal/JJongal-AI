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
        story_stage (str): 현재 수집 중인 이야기 요소 (character, setting, problem, resolution)
        token_usage (Dict): 토큰 사용량 추적 (total_prompt, total_completion, total)
        token_limit (int): 전체 대화에서 사용 가능한 최대 토큰 수
    """
    
    def __init__(self, token_limit: int = 10000):
        """
        챗봇 초기화 및 기본 속성 설정
        
        Args:
            token_limit (int, optional): 전체 대화에서 사용 가능한 최대 토큰 수. 기본값은 10000.
        """
        self.conversation_history = []  # 대화 내역 저장 (role, content)
        self.age_group = None          # 아이의 연령대 (ex: 4-9세)
        self.child_name = None         # 아이의 이름
        self.interests = []            # 아이의 관심사
        
        self.chatbot_name = "부기"
        
        # 프롬프트 로드
        self.prompts = self._load_prompts()   
        
        # 이야기 줄거리 초기화
        self.story_outline = None
        
        # 이야기 수집 단계 (character, setting, problem, resolution)
        self.story_stage = "character"
        
        # 이야기 요소 수집 상태
        self.story_elements = {
            "character": {"count": 0, "topics": set()},
            "setting": {"count": 0, "topics": set()},
            "problem": {"count": 0, "topics": set()},
            "resolution": {"count": 0, "topics": set()}
        }
        
        # 토큰 사용량 추적
        self.token_usage = {
            "total_prompt": 0,
            "total_completion": 0,
            "total": 0,
        }
        
        # 토큰 제한
        self.token_limit = token_limit
        
        # 토큰 제한 도달 시 메시지
        self.token_limit_reached_message = "토큰 제한에 걸렸으니 그만 써라 좀..."
        
        # 마지막 단계 전환 대화 턴 수
        self.last_stage_transition = 0
    
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
        """
        아이 정보를 업데이트하는 메서드
        
        Args:
            child_name (str, optional): 아이의 이름
            age (int, optional): 아이의 나이
            interests (List[str], optional): 아이의 관심사 목록
        """
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
        """시스템 메시지 반환 및 포맷팅"""
        system_template = self.prompts.get('system_message_template', '')
        
        # 포맷팅을 위한 변수 준비
        interests_str = ", ".join(self.interests) if self.interests else "다양한 주제"
        
        # system_template이 리스트인 경우 처리
        if isinstance(system_template, list):
            formatted_items = []
            for item in system_template:
                try:
                    # 각 항목에 포맷팅 시도
                    formatted_item = item.format(
                        age=self.age_group,
                        child_name=self.child_name,
                        interests=interests_str,
                        chatbot_name=self.chatbot_name
                    )
                    formatted_items.append(formatted_item)
                except (KeyError, ValueError):
                    # 포맷팅 실패 시 원본 항목 사용
                    formatted_items.append(item)
            
            # 리스트 항목을 줄바꿈으로 합치기
            return "\n".join(formatted_items)
        else:
            # 기존 string 처리 방식
            try:
                return system_template.format(
                    age=self.age_group,
                    child_name=self.child_name,
                    interests=interests_str,
                    chatbot_name=self.chatbot_name
                )
            except (KeyError, ValueError, AttributeError):
                # 포맷팅 실패 시 원본 템플릿 반환
                return system_template
    
    def get_greeting(self) -> str:
        """
        랜덤한 인사말 반환 및 포맷팅
        
        Returns:
            str: 포맷팅된 인사말
        """
        greetings = self.prompts.get('greeting_templates', [])
        greeting = random.choice(greetings) if greetings else "안녕하세요!"
        
        # 관심사 문자열 준비
        interests_str = ""
        if self.interests:
            if len(self.interests) == 1:
                interests_str = self.interests[0]
            else:
                interests_str = ", ".join(self.interests[:-1]) + f" 또는 {self.interests[-1]}"
        
        # 이름과 조사 확인
        if self.child_name:
            has_final = self._has_final_consonant(self.child_name)
            # 아/야 처리
            child_name_with_ya = f"{self.child_name}아" if has_final else f"{self.child_name}야"
            greeting = greeting.replace("{name}아/야", child_name_with_ya)
            
            # 이/가 처리
            child_name_with_ga = f"{self.child_name}이" if has_final else f"{self.child_name}가"
            greeting = greeting.replace("{name}이/가", child_name_with_ga)
            
            # 은/는 처리
            child_name_with_eun = f"{self.child_name}은" if has_final else f"{self.child_name}는"
            greeting = greeting.replace("{name}은/는", child_name_with_eun)
            
            # 을/를 처리
            child_name_with_eul = f"{self.child_name}을" if has_final else f"{self.child_name}를"
            greeting = greeting.replace("{name}을/를", child_name_with_eul)
            
            # 과/와 처리
            child_name_with_gwa = f"{self.child_name}과" if has_final else f"{self.child_name}와"
            greeting = greeting.replace("{name}과/와", child_name_with_gwa)
            
            # 기본 이름 대체
            greeting = greeting.replace("{name}", self.child_name)
        
        # 관심사 대체
        if "{interests}" in greeting and self.interests:
            greeting = greeting.replace("{interests}", interests_str)
        
        return greeting
    
    def get_story_prompting_question(self) -> str:
        """
        현재 이야기 수집 단계에 맞는 질문 반환
        
        Returns:
            str: 이야기 수집 단계에 맞는 질문
        """
        story_questions = self.prompts.get('story_prompting_questions', {})
        current_stage_questions = story_questions.get(self.story_stage, [])
        
        if not current_stage_questions:
            return self.get_follow_up_question()
        
        question = random.choice(current_stage_questions)
        
        # 이름과 조사 처리
        if self.child_name:
            has_final = self._has_final_consonant(self.child_name)
            
            # 아/야 처리
            child_name_with_ya = f"{self.child_name}아" if has_final else f"{self.child_name}야"
            question = question.replace("{name}아/야", child_name_with_ya)
            
            # 이/가 처리
            child_name_with_ga = f"{self.child_name}이" if has_final else f"{self.child_name}가"
            question = question.replace("{name}이/가", child_name_with_ga)
            
            # 은/는 처리
            child_name_with_eun = f"{self.child_name}은" if has_final else f"{self.child_name}는"
            question = question.replace("{name}은/는", child_name_with_eun)
            
            # 을/를 처리
            child_name_with_eul = f"{self.child_name}을" if has_final else f"{self.child_name}를"
            question = question.replace("{name}을/를", child_name_with_eul)
            
            # 과/와 처리
            child_name_with_gwa = f"{self.child_name}과" if has_final else f"{self.child_name}와"
            question = question.replace("{name}과/와", child_name_with_gwa)
            
            # 기본 이름 대체
            question = question.replace("{name}", self.child_name)
        
        # 다음 단계로 진행
        if random.random() < 0.3 and self.conversation_history and len(self.conversation_history) > 4:
            stages = ["character", "setting", "problem", "resolution"]
            current_index = stages.index(self.story_stage)
            if current_index < len(stages) - 1:
                self.story_stage = stages[current_index + 1]
        
        return question
    
    def get_follow_up_question(self) -> str:
        """
        랜덤한 후속 질문 반환 및 포맷팅
        
        Returns:
            str: 포맷팅된 후속 질문
        """
        questions = self.prompts.get('follow_up_questions', [])
        question = random.choice(questions) if questions else "더 자세히 이야기해 주세요."
        
        if self.child_name:
            has_final = self._has_final_consonant(self.child_name)
            # 아/야 처리
            child_name_with_ya = f"{self.child_name}아" if has_final else f"{self.child_name}야"
            question = question.replace("{name}아/야", child_name_with_ya)
            
            # 이/가 처리
            child_name_with_ga = f"{self.child_name}이" if has_final else f"{self.child_name}가"
            question = question.replace("{name}이/가", child_name_with_ga)
            
            # 은/는 처리
            child_name_with_eun = f"{self.child_name}은" if has_final else f"{self.child_name}는"
            question = question.replace("{name}은/는", child_name_with_eun)
            
            # 을/를 처리
            child_name_with_eul = f"{self.child_name}을" if has_final else f"{self.child_name}를"
            question = question.replace("{name}을/를", child_name_with_eul)
            
            # 과/와 처리
            child_name_with_gwa = f"{self.child_name}과" if has_final else f"{self.child_name}와"
            question = question.replace("{name}과/와", child_name_with_gwa)
            
            # 기본 이름 대체
            question = question.replace("{name}", self.child_name)
            
        return question
    
    def get_encouragement(self) -> str:
        """
        랜덤한 격려 문구 반환 및 포맷팅
        
        Returns:
            str: 포맷팅된 격려 문구
        """
        encouragements = self.prompts.get('encouragement_phrases', [])
        encouragement = random.choice(encouragements) if encouragements else "좋아요!"
        
        if self.child_name:
            has_final = self._has_final_consonant(self.child_name)
            # 아/야 처리
            child_name_with_ya = f"{self.child_name}아" if has_final else f"{self.child_name}야"
            encouragement = encouragement.replace("{name}아/야", child_name_with_ya)
            
            # 이/가 처리
            child_name_with_ga = f"{self.child_name}이" if has_final else f"{self.child_name}가"
            encouragement = encouragement.replace("{name}이/가", child_name_with_ga)
            
            # 은/는 처리
            child_name_with_eun = f"{self.child_name}은" if has_final else f"{self.child_name}는"
            encouragement = encouragement.replace("{name}은/는", child_name_with_eun)
            
            # 을/를 처리
            child_name_with_eul = f"{self.child_name}을" if has_final else f"{self.child_name}를"
            encouragement = encouragement.replace("{name}을/를", child_name_with_eul)
            
            # 과/와 처리
            child_name_with_gwa = f"{self.child_name}과" if has_final else f"{self.child_name}와"
            encouragement = encouragement.replace("{name}과/와", child_name_with_gwa)
            
            # 기본 이름 대체
            encouragement = encouragement.replace("{name}", self.child_name)
            
        return encouragement
    
    def get_age_appropriate_language(self) -> Dict:
        """
        아이의 연령대에 맞는 언어 설정 반환
        
        Returns:
            Dict: 연령대에 맞는 언어 설정
        """
        age_ranges = self.prompts.get('age_appropriate_language', {})
        
        if not self.age_group:
            return {}
        
        if 4 <= self.age_group <= 5:
            return age_ranges.get('4-5', {})
        elif 6 <= self.age_group <= 7:
            return age_ranges.get('6-7', {})
        elif 8 <= self.age_group <= 9:
            return age_ranges.get('8-9', {})
        else:
            return {}
    
    def format_story_collection_prompt(self) -> str:
        """
        스토리 수집 프롬프트 포맷팅
        
        Returns:
            str: 포맷팅된 스토리 수집 프롬프트
        """
        template = self.prompts.get('story_collection_prompt_template', '')
        
        # 관심사 문자열 준비
        interests_str = ", ".join(self.interests) if self.interests else "다양한 주제"
        
        # 이름에 맞는 조사 처리
        formatted_template = template
        if self.child_name:
            has_final = self._has_final_consonant(self.child_name)
            # 과/와 조사 처리
            child_name_with_gwa = f"{self.child_name}과" if has_final else f"{self.child_name}와"
            formatted_template = formatted_template.replace("{child_name}와/과", child_name_with_gwa)
        
        # 나머지 변수 포맷팅
        return formatted_template.format(
            child_name=self.child_name,
            age_group=self.age_group,
            interests=interests_str
        )
    
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
        
        # 챗봇 정보 업데이트
        self.update_child_info(child_name, age, interests)
        self.chatbot_name = chatbot_name
        
        # 시스템 메시지 설정
        self.system_message = self.get_system_message()
        
        # 인사말 생성
        greeting = self.get_greeting()
        
        # 대화 시작
        self.add_to_conversation("assistant", greeting)
        
        return greeting
    
    def _analyze_user_response(self, user_input: str) -> None:
        """
        사용자 응답을 분석하여 현재 이야기 단계의 요소들을 추적
        
        Args:
            user_input (str): 사용자 입력
        """
        current_stage = self.story_stage
        
        # 최소 길이 확인 (의미 있는 응답인지)
        if len(user_input) < 10:
            return
            
        # 현재 단계의 카운트 증가
        self.story_elements[current_stage]["count"] += 1
        
        # 주요 키워드 추출 시도
        try:
            # 간단한 키워드 추출 (공백으로 구분된 단어들 중 2글자 이상)
            words = [word for word in user_input.split() if len(word) >= 2]
            # 중요 키워드 추가 (최대 3개)
            important_words = words[:3] if words else []
            self.story_elements[current_stage]["topics"].update(important_words)
        except Exception:
            pass

    def _should_transition_to_next_stage(self) -> bool:
        """
        현재 단계에서 다음 단계로 전환해야 하는지 결정
        
        Returns:
            bool: 다음 단계로 전환해야 하면 True, 아니면 False
        """
        current_stage = self.story_stage
        current_turn = len(self.conversation_history) // 2  # 대화 턴 수 (질문-답변 쌍)
        
        # 마지막 전환 이후 최소 2턴 이상 지났는지 확인
        if current_turn - self.last_stage_transition < 2:
            return False
            
        # 단계별 전환 기준
        transition_criteria = {
            "character": lambda: self.story_elements["character"]["count"] >= 3 or current_turn > 4,
            "setting": lambda: self.story_elements["setting"]["count"] >= 2 or current_turn > 8,
            "problem": lambda: self.story_elements["problem"]["count"] >= 2 or current_turn > 12,
            "resolution": lambda: False  # 마지막 단계는 전환하지 않음
        }
        
        # 현재 단계에 대한 전환 기준 확인
        should_transition = transition_criteria.get(current_stage, lambda: False)()
        
        # 확률적 요소 추가 (단계가 진행될수록 전환 확률 증가)
        stages = ["character", "setting", "problem", "resolution"]
        current_index = stages.index(current_stage)
        
        # 이미 충분한 대화가 이루어졌다면 전환 확률 증가
        if current_index < len(stages) - 1:
            # 단계별 기본 전환 확률
            base_transition_prob = 0.1 * (current_index + 1)
            
            # 대화가 길어질수록 전환 확률 증가
            turn_factor = min(0.5, 0.05 * (current_turn // 2))
            
            # 최종 전환 확률
            transition_prob = base_transition_prob + turn_factor
            
            # 전환 기준을 충족하면 확률 추가 증가
            if should_transition:
                transition_prob += 0.3
                
            return random.random() < transition_prob
            
        return False

    def _transition_to_next_stage(self) -> None:
        """현재 단계에서 다음 단계로 전환"""
        stages = ["character", "setting", "problem", "resolution"]
        current_index = stages.index(self.story_stage)
        
        if current_index < len(stages) - 1:
            self.story_stage = stages[current_index + 1]
            self.last_stage_transition = len(self.conversation_history) // 2

    def get_stage_transition_message(self) -> str:
        """
        단계 전환 시 자연스러운 전환 메시지 생성
        
        Returns:
            str: 단계 전환 메시지
        """
        stage_messages = {
            "setting": [
                "이제 {name}이/가 얘기해준 친구들이 어디에서 살고 있을지 생각해볼까?",
                "재미있는 친구들이네! 이 친구들이 어떤 곳에서 모험을 하면 좋을까?",
                "{name}아/야, 그 친구들이 사는 세계는 어떤 곳인지 상상해볼래?",
                "멋진 캐릭터들이야! 이제 이 친구들이 어디에서 살고 있는지 이야기해줄래?"
            ],
            "problem": [
                "{name}아/야, 그런 멋진 곳에서 어떤 문제가 생길 수 있을까?",
                "그런 신기한 세계에서 우리 친구들에게 어떤 어려움이 찾아올까?",
                "{name}이/가 만든 세계에서 어떤 모험이 시작될 것 같아?",
                "그 곳에서 주인공이 해결해야 할 어떤 문제가 생길 수 있을까?"
            ],
            "resolution": [
                "{name}아/야, 그런 어려운 문제를 어떻게 해결하면 좋을까?",
                "우리 친구들이 그 문제를 해결하려면 어떻게 해야 할까?",
                "{name}이/가 생각하기에 주인공은 어떻게 그 위기를 극복할 수 있을까?",
                "그런 어려움을 어떻게 이겨낼 수 있을지 {name}의 생각이 궁금해!"
            ]
        }
        
        messages = stage_messages.get(self.story_stage, ["다음 이야기도 들려줘!"])
        message = random.choice(messages)
        
        # 이름과 조사 처리
        if self.child_name:
            has_final = self._has_final_consonant(self.child_name)
            # 아/야 처리
            child_name_with_ya = f"{self.child_name}아" if has_final else f"{self.child_name}야"
            message = message.replace("{name}아/야", child_name_with_ya)
            
            # 이/가 처리
            child_name_with_ga = f"{self.child_name}이" if has_final else f"{self.child_name}가"
            message = message.replace("{name}이/가", child_name_with_ga)
            
            # 은/는 처리
            child_name_with_eun = f"{self.child_name}은" if has_final else f"{self.child_name}는"
            message = message.replace("{name}은/는", child_name_with_eun)
            
            # 을/를 처리
            child_name_with_eul = f"{self.child_name}을" if has_final else f"{self.child_name}를"
            message = message.replace("{name}을/를", child_name_with_eul)
            
            # 과/와 처리
            child_name_with_gwa = f"{self.child_name}과" if has_final else f"{self.child_name}와"
            message = message.replace("{name}과/와", child_name_with_gwa)
            
            # 기본 이름 대체
            message = message.replace("{name}", self.child_name)
            
        return message

    def suggest_story_element(self, user_input: str) -> str:
        """
        사용자 입력을 분석하여 이야기 요소 제안
        
        Args:
            user_input (str): 사용자 입력
            
        Returns:
            str: 다음 대화 제안
        """
        # 사용자 입력 분석
        self._analyze_user_response(user_input)
        
        # 단계 전환 여부 확인
        if self._should_transition_to_next_stage():
            self._transition_to_next_stage()
            return self.get_stage_transition_message()
            
        # 현재 수집 단계에 따라 다른 질문 반환
        if random.random() < 0.7:  # 70% 확률로 단계별 질문
            return self.get_story_prompting_question()
        else:  # 30% 확률로 후속 질문
            if random.random() < 0.5:  # 격려
                return self.get_encouragement()
            else:  # 후속 질문
                return self.get_follow_up_question()
    
    def get_response(self, user_input: str) -> str:
        """
        사용자의 입력에 대한 AI 응답을 생성하는 함수
        
        Args:
            user_input (str): 사용자의 입력 메시지
            
        Returns:
            str: AI가 생성한 응답 메시지
        """
        try:
            # 토큰 제한 확인
            if self.token_usage["total"] >= self.token_limit:
                return self.token_limit_reached_message
            
            # 사용자 메시지 추가
            self.add_to_conversation("user", user_input)
            
            # 연령대에 맞는 언어 설정 가져오기
            age_language = self.get_age_appropriate_language()
            
            # 시스템 메시지에 연령대별 언어 정보 추가
            enhanced_system_message = f"{self.system_message}\n\n추가 정보: 이 아이({self.age_group}세)에게 적합한 어휘와 개념:\n"
            enhanced_system_message += f"어휘: {', '.join(age_language.get('vocabulary', []))}\n"
            enhanced_system_message += f"문장 길이: {age_language.get('sentence_length', '')}\n"
            enhanced_system_message += f"개념: {', '.join(age_language.get('concepts', []))}\n"
            
            # 현재 이야기 단계 추가
            enhanced_system_message += f"\n현재 이야기 수집 단계: {self.story_stage}"
            
            # 이야기 제안 힌트 추가
            next_suggestion = self.suggest_story_element(user_input)
            enhanced_system_message += f"\n\n다음 대화 제안: {next_suggestion}"
            
            # GPT-4o-mini를 사용하여 응답 생성
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": enhanced_system_message},
                    *self.conversation_history[-5:]  # 최근 5개의 메시지만 사용
                ],
                temperature=0.9,  # 더 창의적이고 다양한 응답을 위해 temperature 증가
                max_tokens=500
            )
            
            # 토큰 사용량 업데이트
            if hasattr(response, 'usage'):
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                self.token_usage["total_prompt"] += prompt_tokens
                self.token_usage["total_completion"] += completion_tokens
                self.token_usage["total"] += total_tokens
            
            # 응답 추출 및 저장
            assistant_response = response.choices[0].message.content
            self.add_to_conversation("assistant", assistant_response)
            
            return assistant_response
            
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {str(e)}")
            return f"미안해 {self.child_name}야. 조금만 후에 다시 이야기할까?"
    
    def suggest_story_theme(self) -> Dict:
        """
        대화 내용을 바탕으로 동화 줄거리 주제 제안
        
        Returns:
            Dict: 제안된 주제와 간략한 설명
        """
        try:
            # 토큰 제한 확인
            if self.token_usage["total"] >= self.token_limit:
                return {
                    "theme": "토큰 제한",
                    "plot_summary": self.token_limit_reached_message
                }
                
            prompt = self.format_story_collection_prompt()
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 아이의 대화에서 동화 주제를 추출하는 전문가입니다."},
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": f"대화 내용: {json.dumps(self.conversation_history[-10:], ensure_ascii=False)}"}
                ],
                temperature=0.7,
                max_tokens=600,
                response_format={"type": "json_object"}
            )
            
            # 토큰 사용량 업데이트
            if hasattr(response, 'usage'):
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                self.token_usage["total_prompt"] += prompt_tokens
                self.token_usage["total_completion"] += completion_tokens
                self.token_usage["total"] += total_tokens
            
            # 응답에서 JSON 파싱
            result = json.loads(response.choices[0].message.content)
            self.story_outline = result
            
            return result
            
        except Exception as e:
            print(f"동화 주제 제안 중 오류 발생: {str(e)}")
            # 기본값 반환
            fallback_result = {
                "theme": "모험과 우정",
                "characters": ["용감한 주인공", "든든한 친구"],
                "setting": "신비한 숲",
                "plot_summary": "주인공이 친구와 함께 신비한 숲에서 모험을 하며 다양한 어려움을 극복하고 우정의 가치를 배우는 이야기",
                "educational_value": "우정, 협동, 문제 해결",
                "target_age": f"{self.age_group}세"
            }
            
            self.story_outline = fallback_result
            return fallback_result
    
    def collect_story_outline(self) -> Dict:
        """
        대화 내용을 바탕으로 동화 줄거리 요약 및 태그를 추출하는 함수
        
        Returns:
            Dict: 수집된 정보 (summary_text, tags)
        """
        # 토큰 제한 확인
        if self.token_usage["total"] >= self.token_limit:
            return {
                "summary_text": self.token_limit_reached_message,
                "tags": "토큰 제한"
            }
            
        # 이미 수집된 이야기 주제가 있다면 그것을 사용
        if self.story_outline:
            return {
                "summary_text": self.story_outline.get("plot_summary", ""),
                "tags": ",".join([
                    self.story_outline.get("theme", ""),
                    self.story_outline.get("educational_value", "")
                ])
            }
        
        try:
            # 대화 내용을 바탕으로 줄거리 및 태그 추출 프롬프트 생성
            prompt = f"""
            대화 내용:
            {self.conversation_history[-15:]}
            
            위 정보를 바탕으로 다음 두 가지를 추출해주세요:
            1. 간단한 동화 줄거리 (summary_text)
            2. 이야기와 관련된 핵심 태그 (tags) - 쉼표로 구분된 문자열 형태 (예: 공룡,모험,친구)
            
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
            
            # 토큰 사용량 업데이트
            if hasattr(response, 'usage'):
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                self.token_usage["total_prompt"] += prompt_tokens
                self.token_usage["total_completion"] += completion_tokens
                self.token_usage["total"] += total_tokens
            
            # 응답에서 JSON 파싱
            result = json.loads(response.choices[0].message.content)
            
            # 관심사를 태그에 추가 (선택 사항)
            if self.interests:
                existing_tags = set(result.get("tags", "").split(','))
                existing_tags.update(self.interests)
                result["tags"] = ",".join(filter(None, existing_tags))
            
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
            # 토큰 제한 확인
            if self.token_usage["total"] >= self.token_limit:
                return self.token_limit_reached_message
                
            # 대화 내용 요약 프롬프트 생성
            prompt = f"""
            {self.child_name}({self.age_group}세)와의 대화를 요약해줘.
            다음 사항에 중점을 두고 요약해줘:
            1. 주요 토픽과 관심사
            2. 이야기 수집 단계별 내용 (캐릭터, 배경, 문제, 해결책)
            3. 아이의 창의적인 아이디어
            4. 교육적 가치나 교훈
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
            
            # 토큰 사용량 업데이트
            if hasattr(response, 'usage'):
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                self.token_usage["total_prompt"] += prompt_tokens
                self.token_usage["total_completion"] += completion_tokens
                self.token_usage["total"] += total_tokens
            
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
                    "child_info": {
                        "name": self.child_name,
                        "age": self.age_group,
                        "interests": self.interests
                    },
                    "conversation": self.conversation_history,
                    "summary": self.get_conversation_summary(),
                    "story_outline": self.story_outline or self.suggest_story_theme(),
                    "token_usage": self.token_usage
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
    
    def get_token_usage(self) -> Dict:
        """
        현재까지의 토큰 사용량 정보를 반환하는 함수
        
        Returns:
            Dict: 토큰 사용량 정보
        """
        return {
            "token_usage": self.token_usage,
            "token_limit": self.token_limit,
            "remaining_tokens": max(0, self.token_limit - self.token_usage["total"]),
            "percentage_used": min(100, (self.token_usage["total"] / self.token_limit) * 100)
        } 