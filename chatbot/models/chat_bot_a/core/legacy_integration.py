"""
부기 (ChatBot A) 레거시 통합 모듈

기존 컴포넌트들을 새로운 모듈화 구조로 통합하는 adapter classes
기존 코드의 완전한 호환성을 보장하면서 새로운 LangChain 기능을 제공
"""
from typing import Dict, List, Any, Optional

# Legacy imports
from ..conversation_manager import ConversationManager

# New modular imports
from .conversation_engine import ConversationEngine
from .langchain_conversation_engine import LangChainConversationEngine
from .story_collection_engine import StoryCollectionEngine
from .rag_engine import RAGEngine
from ..processors import MessageProcessor, LanguageProcessor

from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)

class LegacyConversationManagerAdapter:
    """
    ConversationManager를 새로운 ConversationEngine으로 연결하는 어댑터
    """
    
    def __init__(self, token_limit: int = 10000, use_langchain: bool = True, openai_client=None, rag_engine=None):
        """
        adapter initialization
        
        Args:
            token_limit: 토큰 제한
            use_langchain: LangChain 사용 여부
            openai_client: OpenAI 클라이언트
            rag_engine: RAG 엔진
        """
        self.use_langchain = use_langchain
        
        if use_langchain and openai_client:
            # LangChain 기반 엔진 사용
            self.engine = LangChainConversationEngine(
                token_limit=token_limit,
                openai_client=openai_client,
                rag_engine=rag_engine
            )
        else:
            # 기본 엔진 사용
            self.engine = ConversationEngine(token_limit)
        
        # 레거시 호환성을 위한 속성들
        self.conversation_history = []
        self.token_usage = {"total_prompt": 0, "total_completion": 0, "total": 0}
        self.token_limit = token_limit
    
    def add_message(self, role: str, content: str):
        """legacy method: add message"""
        self.engine.add_message(role, content)
        # 레거시 속성 동기화
        self.conversation_history = self.engine.get_conversation_history()
    
    def get_conversation_history(self) -> List[Dict]:
        """legacy method: get conversation history"""
        return self.engine.get_conversation_history()
    
    def get_recent_messages(self, count: int = 5) -> List[Dict]:
        """legacy method: get recent messages"""
        return self.engine.get_recent_messages(count)
    
    def update_token_usage(self, prompt_tokens: int, completion_tokens: int):
        """legacy method: update token usage"""
        self.engine.update_token_usage(prompt_tokens, completion_tokens)
        # 레거시 속성 동기화
        self.token_usage = self.engine.get_token_usage()
    
    def get_token_usage(self) -> Dict[str, int]:
        """legacy method: get token usage"""
        return self.engine.get_token_usage()
    
    def is_token_limit_reached(self) -> bool:
        """legacy method: check token limit"""
        return self.engine.is_token_limit_reached()
    
    def clear_conversation(self):
        """legacy method: clear conversation"""
        self.engine.clear_conversation()
        self.conversation_history = []
    
    def save_conversation(self, file_path: str, additional_data: Optional[Dict] = None) -> bool:
        """legacy method: save conversation"""
        return self.engine.save_conversation(file_path, additional_data)
    
    def load_conversation(self, file_path: str) -> Optional[Dict]:
        """legacy method: load conversation"""
        return self.engine.load_conversation(file_path)
    
    # 새로운 LangChain 기능
    def generate_response_with_langchain(self, user_input: str, child_name: str = "친구", 
                                       age_group: int = 5, interests: List[str] = None, 
                                       chatbot_name: str = "부기") -> str:
        """generate response using LangChain"""
        if isinstance(self.engine, LangChainConversationEngine):
            return self.engine.generate_response_sync(user_input, child_name, age_group, interests, chatbot_name)
        else:
            logger.warning("LangChain 엔진이 아닙니다. 기본 응답을 반환합니다.")
            return f"안녕 {child_name}아! 재미있는 이야기를 함께 만들어보자!"

class LegacyMessageFormatterAdapter:
    """
    adapter that connects MessageFormatter to the new MessageProcessor
    """
    
    def __init__(self, prompts: Dict, child_name: str = None, age_group: int = None, 
                 interests: List[str] = None, chatbot_name: str = "부기"):
        """
        adapter initialization
        
        Args:
            prompts: 프롬프트 딕셔너리
            child_name: 아이 이름
            age_group: 연령대
            interests: 관심사
            chatbot_name: 챗봇 이름
        """
        # 새로운 프로세서 초기화
        self.message_processor = MessageProcessor(prompts, child_name, age_group, interests, chatbot_name)
        self.language_processor = LanguageProcessor()
        
        # 초기화
        self.message_processor.initialize()
        self.language_processor.initialize()
        
        # 레거시 호환성을 위한 속성들
        self.prompts = prompts
        self.child_name = child_name
        self.age_group = age_group
        self.interests = interests or []
        self.chatbot_name = chatbot_name
    
    def update_child_info(self, child_name: str = None, age_group: int = None, 
                         interests: List[str] = None, chatbot_name: str = None):
        """legacy method: 아이 정보 업데이트"""
        self.message_processor.update_child_info(child_name, age_group, interests, chatbot_name)
        
        # 레거시 속성 동기화
        if child_name is not None:
            self.child_name = child_name
        if age_group is not None:
            self.age_group = age_group
        if interests is not None:
            self.interests = interests
        if chatbot_name is not None:
            self.chatbot_name = chatbot_name
    
    def get_system_message(self) -> str:
        """legacy method: 시스템 메시지 반환"""
        return self.message_processor.process({"type": "system_message"})
    
    def get_greeting(self) -> str:
        """legacy method: 인사말 반환"""
        return self.message_processor.process({"type": "greeting"})
    
    def get_story_prompting_question(self, story_stage: str) -> str:
        """레거시 메서드: 이야기 수집 질문 반환"""
        return self.message_processor.process({"type": "story_question", "stage": story_stage})
    
    def get_follow_up_question(self) -> str:
        """레거시 메서드: 후속 질문 반환"""
        return self.message_processor.process({"type": "follow_up"})
    
    def get_encouragement(self) -> str:
        """레거시 메서드: 격려 메시지 반환"""
        return self.message_processor.process({"type": "encouragement"})
    
    def get_stage_transition_message(self, story_stage: str) -> str:
        """레거시 메서드: 단계 전환 메시지 반환"""
        return self.message_processor.process({"type": "stage_transition", "stage": story_stage})
    
    def format_story_collection_prompt(self) -> str:
        """레거시 메서드: 이야기 수집 프롬프트 포맷팅"""
        return self.message_processor.format_story_collection_prompt()
    
    # 새로운 언어 처리 기능
    def check_age_appropriate_vocabulary(self, text: str) -> Dict[str, Any]:
        """연령대 적합 어휘 확인"""
        return self.language_processor.process({
            "type": "vocabulary_check",
            "text": text,
            "age": self.age_group or 5
        })
    
    def simplify_for_age(self, text: str) -> str:
        """연령대에 맞게 텍스트 단순화"""
        return self.language_processor.process({
            "type": "simplify",
            "text": text,
            "age": self.age_group or 5
        })

class LegacyStoryCollectorAdapter:
    """
    StoryCollector를 새로운 StoryCollectionEngine으로 연결하는 어댑터
    """
    
    def __init__(self, openai_client=None, rag_engine=None):
        """
        어댑터 초기화
        
        Args:
            openai_client: OpenAI 클라이언트
            rag_engine: RAG 엔진
        """
        self.engine = StoryCollectionEngine(openai_client, rag_engine)
        
        # 레거시 호환성을 위한 속성들
        self.story_stage = "character"
        self.story_elements = {
            "character": {"count": 0, "topics": set()},
            "setting": {"count": 0, "topics": set()},
            "problem": {"count": 0, "topics": set()},
            "resolution": {"count": 0, "topics": set()}
        }
        self.last_stage_transition = 0
    
    def get_current_stage(self) -> str:
        """레거시 메서드: 현재 단계 반환"""
        stage = self.engine.get_current_stage()
        self.story_stage = stage  # 레거시 속성 동기화
        return stage
    
    def analyze_user_response(self, user_input: str, openai_client=None):
        """레거시 메서드: 사용자 응답 분석"""
        self.engine.analyze_user_response(user_input, openai_client)
        # 레거시 속성 동기화
        self.story_elements = self.engine.get_story_elements()
    
    def should_transition_to_next_stage(self, conversation_length: int) -> bool:
        """레거시 메서드: 단계 전환 여부 확인"""
        return self.engine.should_transition_to_next_stage(conversation_length)
    
    def transition_to_next_stage(self, conversation_length: int) -> bool:
        """레거시 메서드: 다음 단계로 전환"""
        result = self.engine.transition_to_next_stage(conversation_length)
        # 레거시 속성 동기화
        self.story_stage = self.engine.get_current_stage()
        self.last_stage_transition = self.engine.last_stage_transition
        return result
    
    def get_story_elements(self) -> Dict[str, Dict[str, Any]]:
        """레거시 메서드: 수집된 이야기 요소 반환"""
        return self.engine.get_story_elements()

class LegacyStoryAnalyzerAdapter:
    """
    StoryAnalyzer를 새로운 StoryCollectionEngine으로 연결하는 어댑터
    """
    
    def __init__(self, openai_client=None, rag_system=None):
        """
        어댑터 초기화
        
        Args:
            openai_client: OpenAI 클라이언트
            rag_system: RAG 시스템 (레거시 호환성)
        """
        # RAG 시스템을 RAG 엔진으로 변환
        rag_engine = None
        if rag_system:
            try:
                # 기존 RAG 시스템의 설정을 사용하여 새로운 RAG 엔진 생성
                vector_db_path = None
                if hasattr(rag_system, 'persist_directory') and rag_system.persist_directory: # VectorDB 경로 존재 여부 확인
                    vector_db_path = str(rag_system.persist_directory) # VectorDB 경로
                else:
                    # 환경변수에서 경로 가져오기
                    import os
                    vector_db_path = os.getenv("VECTOR_DB_PATH", "/app/chatbot/data/vector_db") # VectorDB 경로
                
                rag_engine = RAGEngine( # RAG 엔진 생성
                    openai_client=openai_client,
                    vector_db_path=vector_db_path # VectorDB 경로
                )
            except Exception as e:
                logger.warning(f"RAG 엔진 변환 실패: {e}")
        
        self.engine = StoryCollectionEngine(openai_client, rag_engine)
        
        # 레거시 호환성을 위한 속성들
        self.openai_client = openai_client
        self.rag_system = rag_system
        self.story_outline = None
    
    def get_conversation_summary(self, conversation_history: List[Dict], 
                               child_name: str = "", age_group: int = 5) -> str:
        """레거시 메서드: 대화 요약"""
        return self.engine.get_conversation_summary(conversation_history, child_name, age_group)
    
    def suggest_story_theme(self, conversation_history: List[Dict], 
                          child_name: str = "", age_group: int = 5,
                          interests: List[str] = None, 
                          story_collection_prompt: str = "") -> Dict:
        """레거시 메서드: 이야기 주제 제안"""
        result = self.engine.suggest_story_theme(
            conversation_history, child_name, age_group, interests, story_collection_prompt
        )
        self.story_outline = result  # 레거시 속성 동기화
        return result
    
    def get_story_outline(self) -> Optional[Dict]:
        """레거시 메서드: 이야기 개요 반환"""
        outline = self.engine.get_story_outline()
        self.story_outline = outline  # 레거시 속성 동기화
        return outline

class LegacyIntegrationManager:
    """
    모든 레거시 컴포넌트를 통합 관리하는 매니저
    """
    
    def __init__(self, token_limit: int = 10000, use_langchain: bool = True, 
                 openai_client=None, rag_system=None, prompts: Dict = None):
        """
        통합 매니저 초기화
        
        Args:
            token_limit: 토큰 제한
            use_langchain: LangChain 사용 여부
            openai_client: OpenAI 클라이언트
            rag_system: RAG 시스템
            prompts: 프롬프트 딕셔너리
        """
        # RAG 엔진 생성
        self.rag_engine = None
        if rag_system:
            try:
                # VectorDB 인스턴스에서 경로 추출
                vector_db_path = None
                if hasattr(rag_system, 'persist_directory') and rag_system.persist_directory: # VectorDB 경로 존재 여부 확인
                    vector_db_path = str(rag_system.persist_directory) # VectorDB 경로
                else:
                    # 환경변수에서 경로 가져오기
                    import os
                    vector_db_path = os.getenv("VECTOR_DB_PATH", "/app/chatbot/data/vector_db") # VectorDB 경로
                
                self.rag_engine = RAGEngine( # RAG 엔진 생성    
                    openai_client=openai_client,
                    vector_db_path=vector_db_path # VectorDB 경로
                )
                logger.info(f"RAG 엔진 생성 완료: {vector_db_path}") # 로그 출력
            except Exception as e:
                logger.warning(f"RAG 엔진 생성 실패: {e}")
        
        # 어댑터들 초기화
        self.conversation = LegacyConversationManagerAdapter(
            token_limit, use_langchain, openai_client, self.rag_engine # RAG 엔진 전달
        )
        
        self.formatter = LegacyMessageFormatterAdapter(
            prompts or {}, None, None, None, "부기" # 프롬프트 딕셔너리 전달
        )
        
        self.collector = LegacyStoryCollectorAdapter(
            openai_client, self.rag_engine # RAG 엔진 전달
        )
        
        self.analyzer = LegacyStoryAnalyzerAdapter(
            openai_client, rag_system # RAG 시스템 전달
        )
        
        logger.info(f"레거시 통합 매니저 초기화 완료 (LangChain: {use_langchain})")
    
    def update_child_info(self, child_name: str = None, age_group: int = None, 
                         interests: List[str] = None, chatbot_name: str = None):
        """모든 컴포넌트의 아이 정보 업데이트"""
        self.formatter.update_child_info(child_name, age_group, interests, chatbot_name)
    
    def get_enhanced_response(self, user_input: str, child_name: str = "친구", 
                            age_group: int = 5, interests: List[str] = None, 
                            chatbot_name: str = "부기") -> str:
        """향상된 응답 생성 (LangChain + RAG)"""
        return self.conversation.generate_response_with_langchain(
            user_input, child_name, age_group, interests, chatbot_name
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 정보 반환"""
        return {
            "use_langchain": isinstance(self.conversation.engine, LangChainConversationEngine),
            "rag_engine_available": self.rag_engine is not None,
            "token_usage": self.conversation.get_token_usage(),
            "current_story_stage": self.collector.get_current_stage(),
            "collected_elements": self.collector.get_story_elements()
        } 