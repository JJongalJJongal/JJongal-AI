"""
부기 (ChatBot A) LangChain 대화 엔진

기존 ConversationEngine을 확장하여 LangChain 기반 대화 생성 기능을 추가
chat_bot_b와 동일한 LangChain 패턴을 사용
"""
import asyncio
from typing import List, Dict, Any, Optional
import json

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# Project imports
from ..conversation_manager import ConversationManager
from .rag_engine import RAGSystem
from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)

class LangChainConversationEngine(ConversationManager):
    """
    LangChain 기반 대화 엔진
    
    기존 ConversationManager를 확장하여 LangChain의 체인 기능을 활용한
    더 정교한 대화 생성 기능을 제공
    """
    
    def __init__(self, 
                 token_limit: int = 10000,
                 openai_client=None,
                 rag_engine: RAGSystem = None,
                 prompts_file_path: str = "chatbot/data/prompts/chatbot_a_prompts.json",
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.9):
        """
        LangChain 대화 엔진 초기화
        
        Args:
            token_limit: 토큰 제한
            openai_client: OpenAI 클라이언트
            rag_engine: RAG 엔진 인스턴스
            prompts_file_path: 프롬프트 파일 경로
            model_name: 사용할 LLM 모델명
            temperature: 생성 온도
        """
        # 부모 클래스 초기화
        super().__init__(token_limit)
        
        self.openai_client = openai_client
        self.rag_engine = rag_engine
        self.prompts_file_path = prompts_file_path
        self.model_name = model_name
        self.temperature = temperature
        
        # LangChain 구성 요소
        self.conversation_chain = None
        self.story_guidance_chain = None
        self.prompts = None
        
        # 초기화
        self._initialize_langchain_components()
    
    def _initialize_langchain_components(self):
        """LangChain 구성 요소 초기화"""
        try:
            # 1. 프롬프트 로드
            self._load_prompts()
            
            # 2. LangChain 체인 설정
            self._setup_langchain_chains()
            
            logger.info("LangChainConversationEngine 초기화 완료")
            
        except Exception as e:
            logger.error(f"LangChainConversationEngine 초기화 실패: {e}")
            # 초기화 실패 시에도 기본 ConversationEngine 기능은 사용 가능
    
    def _load_prompts(self):
        """프롬프트 파일 로드 - chatbot_a_prompts.json에서 직접 로드"""
        try:
            # shared.configs를 통해 chatbot_a 프롬프트 로드
            from shared.configs import load_chatbot_a_prompts
            self.prompts = load_chatbot_a_prompts()
            
            # 시스템 메시지 구성
            if "system_message_template" in self.prompts:
                system_messages = self.prompts["system_message_template"]
                self.system_message = "\n".join(system_messages) if isinstance(system_messages, list) else system_messages
            else:
                self.system_message = "너는 유치원 선생님 같은 친절하고 따뜻한 성격의 '부기'야."
            
            logger.info(f"ChatBot A 프롬프트 로드 완료")
            
        except Exception as e:
            logger.error(f"프롬프트 파일 로드 실패: {e}")
            # 기본 프롬프트 설정
            self.prompts = {}
            self.system_message = "너는 유치원 선생님 같은 친절하고 따뜻한 성격의 '부기'야."
    
    def _setup_langchain_chains(self):
        """LangChain 체인 설정"""
        try:
            # LLM 모델 설정
            llm = ChatOpenAI(
                temperature=self.temperature,
                model=self.model_name,
                api_key=self.openai_client.api_key if self.openai_client else None
            )
            
            # 1. 일반 대화 체인 - 실제 JSON 프롬프트 사용
            conversation_template = self._build_conversation_template()
            
            conversation_prompt = ChatPromptTemplate.from_template(conversation_template)
            self.conversation_chain = conversation_prompt | llm | StrOutputParser()
            
            # 2. 이야기 안내 체인
            story_guidance_template = self.prompts.get("langchain_templates", {}).get(
                "story_guidance",
                """당신은 아이와 함께 동화를 만드는 전문가입니다.

현재 수집 단계: {current_stage}
수집된 요소들: {collected_elements}

아이의 최근 대화:
{conversation_history}

참고 자료:
{rag_context}

위 정보를 바탕으로 아이가 {current_stage} 단계의 이야기 요소를 자연스럽게 말할 수 있도록 유도하는 질문이나 대화를 생성하세요.
아이의 연령({age_group}세)에 맞는 쉬운 말을 사용하고, 재미있고 상상력을 자극하는 방식으로 접근하세요.

응답:"""
            )
            
            story_guidance_prompt = ChatPromptTemplate.from_template(story_guidance_template)
            self.story_guidance_chain = story_guidance_prompt | llm | StrOutputParser()
            
            logger.info("LangChain 체인 설정 완료")
            
        except Exception as e:
            logger.error(f"LangChain 체인 설정 실패: {e}")
            raise
    
    def _build_conversation_template(self) -> str:
        """실제 JSON 프롬프트를 기반으로 대화 템플릿 구성"""
        # 시스템 메시지를 기본으로 사용
        template_parts = []
        
        if hasattr(self, 'system_message'):
            # 연령별 맞춤화
            system_msg = self.system_message.replace("{age}", "{age_group}")
            template_parts.append(system_msg)
        
        # 추가 지침들
        template_parts.extend([
            "",
            "아이 정보:",
            "- 이름: {child_name}",
            "- 나이: {age_group}세",
            "- 관심사: {interests}",
            "",
            "현재 상황:",
            "- 이전 대화: {conversation_history}",
            "- 아이의 현재 말: {user_input}",
            "",
            "응답 지침:",
            "1. 반드시 한국어로 대답하세요",
            "2. 아이의 연령에 맞는 어휘와 문장 길이를 사용하세요",
            "3. 한 번에 하나의 명확한 질문만 하세요",
            "4. 아이의 상상력과 호기심을 격려하세요",
            "5. 긍정적이고 지지적인 반응을 보이세요",
            "6. 한국어 조사를 올바르게 사용하세요 (받침 없는 이름: ~야, 받침 있는 이름: ~아)",
            "   예: 태영야, 민수야, 지수야 (받침 없음) / 철민아, 은주아 (받침 있음)",
            "",
            "{chatbot_name}의 응답:"
        ])
        
        return "\n".join(template_parts)
    
    def _get_age_appropriate_response_style(self, age_group: int) -> Dict[str, Any]:
        """연령별 응답 스타일 설정"""
        age_language = self.prompts.get("age_appropriate_language", {})
        
        if 4 <= age_group <= 7:
            style_config = age_language.get("4-7", {})
            return {
                "vocabulary_level": "simple",
                "sentence_length": "3-5 단어",
                "concepts": style_config.get("concepts", []),
                "vocabulary": style_config.get("vocabulary", [])
            }
        elif 8 <= age_group <= 9:
            style_config = age_language.get("8-9", {})
            return {
                "vocabulary_level": "advanced",
                "sentence_length": "7-12 단어",
                "concepts": style_config.get("concepts", []),
                "vocabulary": style_config.get("vocabulary", [])
            }
        else:
            return {"vocabulary_level": "adaptive", "sentence_length": "5-8 단어"}
    
    def _select_appropriate_prompt(self, conversation_stage: str, age_group: int) -> str:
        """대화 단계와 연령에 따른 적절한 프롬프트 선택"""
        # story_prompting_questions에서 단계별 질문 선택
        story_questions = self.prompts.get("story_prompting_questions", {})
        
        if conversation_stage in story_questions:
            questions = story_questions[conversation_stage]
            if questions:
                import random
                return random.choice(questions)
        
        # follow_up_questions에서 일반적인 질문 선택
        follow_ups = self.prompts.get("follow_up_questions", [])
        if follow_ups:
            import random
            return random.choice(follow_ups)
        
        return "더 자세히 말해줄 수 있어?"
    
    async def generate_response(self, 
                              user_input: str,
                              child_name: str = "친구",
                              age_group: int = 5,
                              interests: List[str] = None,
                              chatbot_name: str = "부기") -> str:
        """
        LangChain을 사용한 응답 생성
        
        Args:
            user_input: 사용자 입력
            child_name: 아이 이름
            age_group: 연령대
            interests: 관심사 목록
            chatbot_name: 챗봇 이름
            
        Returns:
            str: 생성된 응답
        """
        if not self.conversation_chain:
            # LangChain이 초기화되지 않은 경우 기본 방식 사용
            return f"안녕 {child_name}아! 재미있는 이야기를 함께 만들어보자!"
        
        try:
            # 사용자 입력을 대화 히스토리에 추가
            self.add_message("user", user_input)
            
            # 최근 대화 히스토리 구성
            recent_messages = self.get_recent_messages(5)
            conversation_history = "\n".join([
                f"{'아이' if msg['role'] == 'user' else chatbot_name}: {msg['content']}"
                for msg in recent_messages[:-1]  # 현재 입력 제외
            ])
            
            # 관심사 문자열 구성
            interests_str = ", ".join(interests) if interests else "다양한 주제"
            
            # LangChain 체인으로 응답 생성
            response = await self.conversation_chain.ainvoke({
                "child_name": child_name,
                "age_group": age_group,
                "interests": interests_str,
                "chatbot_name": chatbot_name,
                "conversation_history": conversation_history,
                "user_input": user_input
            })
            
            # 토큰 사용량 추정 및 업데이트
            try:
                # 대략적인 토큰 수 계산 (정확하지 않지만 추정값)
                input_text = f"{conversation_history}\n{user_input}"
                prompt_tokens = len(input_text.split()) * 1.3  # 대략적 추정
                completion_tokens = len(response.split()) * 1.3
                
                self.update_token_usage(int(prompt_tokens), int(completion_tokens))
                logger.info(f"토큰 사용량 업데이트: prompt={int(prompt_tokens)}, completion={int(completion_tokens)}")
            except Exception as e:
                logger.warning(f"토큰 사용량 추정 실패: {e}")
            
            # 응답을 대화 히스토리에 추가
            self.add_message("assistant", response)
            
            logger.info("LangChain 응답 생성 완료")
            return response
            
        except Exception as e:
            logger.error(f"LangChain 응답 생성 실패: {e}")
            # 실패 시 기본 응답
            return f"재미있는 이야기네! {child_name}아, 더 자세히 말해줄 수 있어?"
    
    async def generate_story_guidance(self,
                                    current_stage: str,
                                    collected_elements: Dict[str, Any],
                                    child_name: str = "친구",
                                    age_group: int = 5) -> str:
        """
        LangChain을 사용한 이야기 안내 생성
        
        Args:
            current_stage: 현재 수집 단계
            collected_elements: 수집된 요소들
            child_name: 아이 이름
            age_group: 연령대
            
        Returns:
            str: 이야기 안내 메시지
        """
        if not self.story_guidance_chain:
            return f"이제 {current_stage}에 대해 이야기해볼까?"
        
        try:
            # 최근 대화 히스토리 구성
            recent_messages = self.get_recent_messages(3)
            conversation_history = "\n".join([
                f"{'아이' if msg['role'] == 'user' else '부기'}: {msg['content']}"
                for msg in recent_messages
            ])
            
            # 수집된 요소들 문자열 구성
            elements_str = ""
            for stage, data in collected_elements.items():
                if data.get("count", 0) > 0:
                    topics = ", ".join(list(data.get("topics", [])))
                    elements_str += f"{stage}: {topics}\n"
            
            # RAG 컨텍스트 구성
            rag_context = ""
            if self.rag_engine:
                try:
                    # 현재 단계와 관련된 참고 자료 검색
                    search_query = f"{current_stage} {age_group}세"
                    similar_stories = self.rag_engine.get_similar_stories(search_query, age_group, 2)
                    
                    if similar_stories:
                        rag_context = "참고할 수 있는 이야기들:\n"
                        for story in similar_stories:
                            rag_context += f"- {story.get('title', '제목 없음')}: {story.get('summary', '')[:100]}...\n"
                except Exception as e:
                    logger.warning(f"RAG 검색 실패: {e}")
            
            # LangChain 체인으로 안내 생성
            guidance = await self.story_guidance_chain.ainvoke({
                "current_stage": current_stage,
                "collected_elements": elements_str,
                "conversation_history": conversation_history,
                "rag_context": rag_context,
                "age_group": age_group
            })
            
            logger.info(f"이야기 안내 생성 완료: {current_stage}")
            return guidance
            
        except Exception as e:
            logger.error(f"이야기 안내 생성 실패: {e}")
            # 실패 시 기본 안내
            stage_names = {
                "character": "등장인물",
                "setting": "배경",
                "problem": "문제",
                "resolution": "해결"
            }
            stage_korean = stage_names.get(current_stage, current_stage)
            return f"이제 우리 이야기의 {stage_korean}에 대해 생각해볼까? 어떤 {stage_korean}이 좋을까?"
    
    def generate_response_sync(self, 
                             user_input: str,
                             child_name: str = "친구",
                             age_group: int = 5,
                             interests: List[str] = None,
                             chatbot_name: str = "부기") -> str:
        """
        동기 버전의 응답 생성 (기존 코드 호환성을 위해)
        
        Args:
            user_input: 사용자 입력
            child_name: 아이 이름
            age_group: 연령대
            interests: 관심사 목록
            chatbot_name: 챗봇 이름
            
        Returns:
            str: 생성된 응답
        """
        try:
            # 이벤트 루프 안전 처리 - 먼저 LangChain 체인 존재 확인
            if not self.conversation_chain:
                logger.warning("LangChain 체인이 없어 fallback 모드로 응답 생성")
                return self._generate_response_fallback(user_input, child_name, age_group, interests, chatbot_name)
            
            try:
                # 현재 이벤트 루프 확인
                loop = asyncio.get_running_loop()
                # 이벤트 루프가 실행 중인 경우 새 스레드에서 실행
                import concurrent.futures
                import threading
                
                logger.info("기존 이벤트 루프 감지됨, 새 스레드에서 비동기 처리 실행")
                
                def run_async_in_thread():
                    # 새 이벤트 루프로 실행
                    return asyncio.run(
                        self.generate_response(user_input, child_name, age_group, interests, chatbot_name)
                    )
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async_in_thread)
                    return future.result(timeout=30)  # 30초 타임아웃
                    
            except RuntimeError:
                # 이벤트 루프가 없는 경우 새로 생성
                logger.info("이벤트 루프 없음, 새로 생성하여 비동기 처리 실행")
                return asyncio.run(
                    self.generate_response(user_input, child_name, age_group, interests, chatbot_name)
                )
        except Exception as e:
            logger.error(f"동기 응답 생성 실패: {e}")
            return self._generate_response_fallback(user_input, child_name, age_group, interests, chatbot_name)
    
    def _generate_response_fallback(self, user_input: str, child_name: str, age_group: int, 
                                  interests: List[str], chatbot_name: str) -> str:
        """LangChain 사용 불가 시 프롬프트 기반 응답 생성"""
        try:
            # 올바른 조사를 위한 import
            from shared.utils.korean_utils import format_with_josa
            
            # 사용자 입력을 히스토리에 추가
            self.add_message("user", user_input)
            
            # 연령대에 따른 응답 선택
            if 4 <= age_group <= 7:
                responses = [
                    f"우와! {format_with_josa(child_name, '아/야')}, 정말 재미있는 이야기네!",
                    f"{format_with_josa(child_name, '이/가')} 어떤 기분이었을까?",
                    f"그 다음엔 뭐가 일어났을까?",
                    f"와! {format_with_josa(child_name, '아/야')}, 더 말해줄래?"
                ]
            else:
                responses = [
                    f"정말 흥미로운 이야기구나, {format_with_josa(child_name, '아/야')}!",
                    f"그 상황에서 주인공은 어떤 마음이었을까?",
                    f"다음에 어떤 일이 벌어질 것 같아?",
                    f"이야기가 점점 재미있어지는군! 계속 들려줄래?"
                ]
            
            import random
            response = random.choice(responses)
            
            # 응답을 히스토리에 추가
            self.add_message("assistant", response)
            
            return response
            
        except Exception as e:
            logger.error(f"Fallback 응답 생성 실패: {e}")
            # 예외 상황에서도 올바른 조사 사용
            try:
                from shared.utils.korean_utils import format_with_josa
                return f"재미있는 이야기네! {format_with_josa(child_name, '아/야')}, 더 자세히 말해줄 수 있어?"
            except:
                return f"재미있는 이야기네! {child_name}아, 더 자세히 말해줄 수 있어?"
    
    async def health_check(self) -> bool:
        """LangChain 대화 엔진 상태 확인"""
        try:
            # 기본 ConversationManager 상태 확인
            if self.is_token_limit_reached():
                logger.warning("토큰 제한에 도달함")
                return False
            
            # LangChain 체인 확인
            if not self.conversation_chain:
                logger.warning("LangChain 체인이 초기화되지 않음")
                return False
            
            # 간단한 테스트
            test_response = await self.generate_response("안녕", "테스트", 5, ["테스트"], "부기")
            return len(test_response) > 0
            
        except Exception as e:
            logger.error(f"LangChain 대화 엔진 health check 실패: {e}")
            return False 