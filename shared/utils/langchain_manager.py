"""
쫑알쫑알 LangChain 통합 관리자

LangChain 공식 Best Practice 기반:
- 중앙집중식 LLM 관리 (Singleton Pattern)
- 정확한 토큰 카운팅 (공식 콜백 사용)
- 표준화된 에러 핸들링
- 메모리 시스템 통합
- 체인 팩토리 패턴
"""

import asyncio
from typing import Dict, Any, Optional, List, Type, Union
from datetime import datetime
import threading
from functools import lru_cache

# LangChain Core
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_community.callbacks.manager import get_openai_callback

# Project imports
from shared.utils.logging_utils import get_module_logger
from shared.configs.app_config import get_env_vars

logger = get_module_logger(__name__)

class LangChainManager:
    """
    LangChain 통합 관리자 (Singleton)
    
    주요 기능:
    - LLM 인스턴스 중앙 관리 및 캐싱
    - 정확한 토큰 사용량 추적
    - 표준화된 체인 생성
    - 통합 에러 핸들링
    - 메모리 시스템 관리
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized') or not self._initialized:
            self._llm_cache = {}
            self._token_usage = {}
            self._memory_store = {}
            self._env_vars = get_env_vars()
            self._default_config = RunnableConfig(
                recursion_limit=10,
                max_concurrency=4
            )
            self._initialized = True
            logger.info("LangChain 통합 관리자 초기화 완료")
    
    @lru_cache(maxsize=10)
    def get_llm(self, 
                model_name: str = "gpt-4o-mini",
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                **kwargs) -> ChatOpenAI:
        """
        LLM 인스턴스 가져오기 (캐시됨)
        
        Args:
            model_name: 모델명
            temperature: 생성 온도
            max_tokens: 최대 토큰 수
            **kwargs: 추가 설정
            
        Returns:
            ChatOpenAI: 캐시된 LLM 인스턴스
        """
        cache_key = f"{model_name}_{temperature}_{max_tokens}_{hash(frozenset(kwargs.items()))}"
        
        if cache_key not in self._llm_cache:
            try:
                llm_config = {
                    "model": model_name,
                    "temperature": temperature,
                    "api_key": self._env_vars.get("openai_api_key")
                }
                
                if max_tokens:
                    llm_config["max_tokens"] = max_tokens
                    
                llm_config.update(kwargs)
                
                self._llm_cache[cache_key] = ChatOpenAI(**llm_config)
                logger.info(f"LLM 인스턴스 생성: {model_name} (temp: {temperature})")
                
            except Exception as e:
                logger.error(f"LLM 인스턴스 생성 실패: {e}")
                raise
        
        return self._llm_cache[cache_key]
    
    def create_conversation_chain(self,
                                 system_prompt: str,
                                 model_name: str = "gpt-4o-mini",
                                 temperature: float = 0.7,
                                 include_history: bool = True):
        """
        표준 대화 체인 생성
        
        Args:
            system_prompt: 시스템 프롬프트
            model_name: 사용할 모델
            temperature: 생성 온도  
            include_history: 대화 히스토리 포함 여부
            
        Returns:
            체인 인스턴스
        """
        try:
            llm = self.get_llm(model_name, temperature)
            
            if include_history:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}")
                ])
            else:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}")
                ])
            
            chain = prompt | llm | StrOutputParser()
            
            logger.info(f"대화 체인 생성 완료: {model_name}")
            return chain
            
        except Exception as e:
            logger.error(f"대화 체인 생성 실패: {e}")
            raise
    
    def create_enhanced_chain(self,
                             template: str,
                             model_name: str = "gpt-4o-mini",
                             temperature: float = 0.7,
                             input_variables: List[str] = None):
        """
        템플릿 기반 향상된 체인 생성
        
        Args:
            template: 프롬프트 템플릿
            model_name: 사용할 모델
            temperature: 생성 온도
            input_variables: 입력 변수 목록
            
        Returns:
            체인 인스턴스
        """
        try:
            llm = self.get_llm(model_name, temperature)
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | llm | StrOutputParser()
            
            logger.info(f"향상된 체인 생성 완료: {model_name}")
            return chain
            
        except Exception as e:
            logger.error(f"향상된 체인 생성 실패: {e}")
            raise
    
    async def invoke_with_callback(self, 
                                  chain,
                                  input_data: Dict[str, Any],
                                  session_id: str = None) -> Dict[str, Any]:
        """
        정확한 토큰 사용량 추적과 함께 체인 실행
        
        Args:
            chain: 실행할 체인
            input_data: 입력 데이터
            session_id: 세션 ID (토큰 사용량 추적용)
            
        Returns:
            Dict: 결과와 토큰 사용량 정보
        """
        try:
            with get_openai_callback() as cb:
                result = await chain.ainvoke(input_data, config=self._default_config)
                
                # 토큰 사용량 업데이트
                token_info = {
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_tokens": cb.total_tokens,
                    "total_cost": cb.total_cost,
                    "timestamp": datetime.now().isoformat()
                }
                
                if session_id:
                    if session_id not in self._token_usage:
                        self._token_usage[session_id] = {
                            "total_prompt_tokens": 0,
                            "total_completion_tokens": 0,
                            "total_tokens": 0,
                            "total_cost": 0.0,
                            "requests": []
                        }
                    
                    session_usage = self._token_usage[session_id]
                    session_usage["total_prompt_tokens"] += cb.prompt_tokens
                    session_usage["total_completion_tokens"] += cb.completion_tokens
                    session_usage["total_tokens"] += cb.total_tokens
                    session_usage["total_cost"] += cb.total_cost
                    session_usage["requests"].append(token_info)
                
                logger.info(f"체인 실행 완료 - 토큰: {cb.total_tokens}, 비용: ${cb.total_cost:.4f}")
                
                return {
                    "result": result,
                    "token_usage": token_info,
                    "success": True
                }
                
        except Exception as e:
            logger.error(f"체인 실행 실패: {e}")
            return {
                "result": None,
                "error": str(e),
                "success": False
            }
    
    def get_token_usage(self, session_id: str = None) -> Dict[str, Any]:
        """토큰 사용량 조회"""
        if session_id:
            return self._token_usage.get(session_id, {})
        return self._token_usage
    
    def create_memory_chain(self,
                           base_chain,
                           get_session_history_func,
                           input_messages_key: str = "input",
                           history_messages_key: str = "history"):
        """
        메모리 기반 체인 생성 (RunnableWithMessageHistory)
        
        Args:
            base_chain: 기본 체인
            get_session_history_func: 세션 히스토리 가져오는 함수
            input_messages_key: 입력 메시지 키
            history_messages_key: 히스토리 메시지 키
            
        Returns:
            메모리 통합 체인
        """
        try:
            memory_chain = RunnableWithMessageHistory(
                base_chain,
                get_session_history_func,
                input_messages_key=input_messages_key,
                history_messages_key=history_messages_key
            )
            
            logger.info("메모리 체인 생성 완료")
            return memory_chain
            
        except Exception as e:
            logger.error(f"메모리 체인 생성 실패: {e}")
            raise
    
    def trim_conversation_history(self,
                                 messages: List,
                                 max_tokens: int = 4000,
                                 model_name: str = "gpt-4o-mini"):
        """
        대화 히스토리 토큰 기반 트리밍
        
        Args:
            messages: 메시지 목록
            max_tokens: 최대 토큰 수
            model_name: 토큰 계산용 모델명
            
        Returns:
            트리밍된 메시지 목록
        """
        try:
            llm = self.get_llm(model_name)
            
            trimmed_messages = trim_messages(
                messages,
                max_tokens=max_tokens,
                strategy="last",
                token_counter=llm,
                include_system=True,
                start_on="human"
            )
            
            if len(trimmed_messages) < len(messages):
                logger.info(f"메시지 트리밍: {len(messages)} -> {len(trimmed_messages)}")
            
            return trimmed_messages
            
        except Exception as e:
            logger.error(f"메시지 트리밍 실패: {e}")
            return messages
    
    def cleanup_session(self, session_id: str):
        """세션 정리"""
        if session_id in self._token_usage:
            del self._token_usage[session_id]
        if session_id in self._memory_store:
            del self._memory_store[session_id]
        logger.info(f"세션 정리 완료: {session_id}")

# 전역 인스턴스
langchain_manager = LangChainManager()

# 편의 함수들
def get_llm(*args, **kwargs):
    """전역 LLM 인스턴스 가져오기"""
    return langchain_manager.get_llm(*args, **kwargs)

def create_conversation_chain(*args, **kwargs):
    """전역 대화 체인 생성"""
    return langchain_manager.create_conversation_chain(*args, **kwargs)

def invoke_with_tracking(*args, **kwargs):
    """토큰 추적과 함께 체인 실행"""
    return langchain_manager.invoke_with_callback(*args, **kwargs) 