"""
부기 (ChatBot A) RAG 엔진

LangChain과 ChromaDB를 활용한 검색 증강 생성 엔진
chat_bot_b와 동일한 LangChain 패턴을 사용하여 일관성 확보
"""
import logging
import json
import uuid
from typing import Dict, List, Any, Optional
from pathlib import Path

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Project imports
from chatbot.data.vector_db.core import VectorDB
from chatbot.data.vector_db.query import query_vector_db, format_query_results
from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)

class RAGEngine:
    """
    LangChain 기반 RAG 엔진
    
    chat_bot_b의 TextGenerator와 동일한 LangChain 패턴을 사용하여
    검색 증강 생성 기능을 제공
    """
    
    def __init__(self,
                 openai_client=None,
                 vector_db_path: str = None,
                 collection_name: str = "fairy_tales",
                 prompts_file_path: str = "chatbot/data/prompts/chatbot_a_prompts.json",
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.7):
        """
        RAG 엔진 초기화
        
        Args:
            openai_client: OpenAI 클라이언트
            vector_db_path: ChromaDB 데이터베이스 경로
            collection_name: ChromaDB 컬렉션 이름
            prompts_file_path: 프롬프트 파일 경로
            model_name: 사용할 LLM 모델명
            temperature: 생성 온도
        """
        self.openai_client = openai_client
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        self.prompts_file_path = prompts_file_path
        self.model_name = model_name
        self.temperature = temperature
        
        # LangChain 구성 요소
        self.vector_store = None
        self.retriever = None
        self.rag_chain = None
        self.theme_enrichment_chain = None
        self.prompts = None
        
        # 초기화
        self._initialize_components()
    
    def _initialize_components(self):
        """LangChain 구성 요소 초기화"""
        try:
            # 1. 프롬프트 로드
            self._load_prompts()
            
            # 2. ChromaDB 초기화
            self._initialize_vector_db()
            
            # 3. LangChain 체인 설정
            self._setup_langchain_chains()
            
            logger.info("RAGEngine 초기화 완료")
            
        except Exception as e:
            logger.error(f"RAGEngine 초기화 실패: {e}")
            raise
    
    def _load_prompts(self):
        """프롬프트 파일 로드"""
        try:
            with open(self.prompts_file_path, 'r', encoding='utf-8') as f:
                self.prompts = json.load(f)
            logger.info(f"프롬프트 파일 로드 완료: {self.prompts_file_path}")
            
        except Exception as e:
            logger.error(f"프롬프트 파일 로드 실패: {e}")
            # 기본 프롬프트 설정
            self.prompts = {
                "rag_templates": {
                    "story_enrichment": "다음 주제를 바탕으로 더 풍부한 동화 주제를 제안해주세요: {theme}",
                    "conversation_enhancement": "다음 대화 내용을 바탕으로 적절한 응답을 생성해주세요: {conversation}"
                }
            }
    
    def _initialize_vector_db(self):
        """ChromaDB 초기화"""
        if not self.vector_db_path:
            logger.warning("ChromaDB 경로가 설정되지 않음. RAG 기능 비활성화")
            return
        
        try:
            self.vector_store = VectorDB(persist_directory=self.vector_db_path)
            
            # 컬렉션 존재 확인
            try:
                collection = self.vector_store.get_collection(self.collection_name)
                logger.info(f"ChromaDB 컬렉션 '{self.collection_name}' 연결 완료")
            except Exception as e:
                logger.warning(f"컬렉션 '{self.collection_name}' 연결 실패: {e}")
                
        except Exception as e:
            logger.error(f"ChromaDB 초기화 실패: {e}")
            raise
    
    def _setup_langchain_chains(self):
        """LangChain 체인 설정"""
        try:
            # LLM 모델 설정
            llm = ChatOpenAI(
                temperature=self.temperature,
                model=self.model_name,
                api_key=self.openai_client.api_key if self.openai_client else None
            )
            
            # 1. 주제 풍부화 체인
            theme_template = self.prompts.get("rag_templates", {}).get(
                "story_enrichment",
                "다음 주제를 바탕으로 {age_group}세 아이에게 적합한 더 풍부한 동화 주제를 제안해주세요.\n\n기본 주제: {theme}\n\n참고 자료:\n{context}"
            )
            
            theme_prompt = ChatPromptTemplate.from_template(theme_template)
            self.theme_enrichment_chain = theme_prompt | llm | StrOutputParser()
            
            # 2. 대화 향상 체인
            conversation_template = self.prompts.get("rag_templates", {}).get(
                "conversation_enhancement",
                "다음 대화 내용과 참고 자료를 바탕으로 아이와의 자연스러운 대화를 이어가는 응답을 생성해주세요.\n\n대화 내용: {conversation}\n\n참고 자료:\n{context}\n\n응답:"
            )
            
            conversation_prompt = ChatPromptTemplate.from_template(conversation_template)
            self.rag_chain = conversation_prompt | llm | StrOutputParser()
            
            logger.info("LangChain 체인 설정 완료")
            
        except Exception as e:
            logger.error(f"LangChain 체인 설정 실패: {e}")
            raise
    
    async def enrich_story_theme(self, theme: str, age_group: int) -> str:
        """
        LangChain을 사용한 스토리 주제 풍부화
        
        Args:
            theme: 기본 주제
            age_group: 연령대
            
        Returns:
            str: 풍부화된 주제
        """
        if not self.theme_enrichment_chain or not self.vector_store:
            return theme
        
        try:
            # 관련 스토리 검색
            reference_stories = await self._retrieve_similar_stories(theme, age_group)
            
            # 컨텍스트 구성
            context = self._format_context(reference_stories)
            
            # LangChain 체인으로 주제 풍부화
            enriched_theme = await self.theme_enrichment_chain.ainvoke({
                "theme": theme,
                "age_group": age_group,
                "context": context
            })
            
            logger.info(f"주제 풍부화 완료: {theme} -> {enriched_theme[:50]}...")
            return enriched_theme
            
        except Exception as e:
            logger.error(f"주제 풍부화 실패: {e}")
            return theme
    
    async def enhance_conversation(self, conversation_text: str, search_query: str = None) -> str:
        """
        LangChain을 사용한 대화 향상
        
        Args:
            conversation_text: 대화 내용
            search_query: 검색 쿼리 (없으면 대화 내용 사용)
            
        Returns:
            str: 향상된 응답
        """
        if not self.rag_chain or not self.vector_store:
            return "RAG 시스템이 초기화되지 않았습니다."
        
        try:
            # 검색 쿼리 설정
            query = search_query or conversation_text
            
            # 관련 스토리 검색
            reference_stories = await self._retrieve_similar_stories(query)
            
            # 컨텍스트 구성
            context = self._format_context(reference_stories)
            
            # LangChain 체인으로 대화 향상
            enhanced_response = await self.rag_chain.ainvoke({
                "conversation": conversation_text,
                "context": context
            })
            
            logger.info("대화 향상 완료")
            return enhanced_response
            
        except Exception as e:
            logger.error(f"대화 향상 실패: {e}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다."
    
    async def _retrieve_similar_stories(self, query_text: str, age_group: int = None) -> List[Dict[str, Any]]:
        """ChromaDB에서 유사 스토리 검색"""
        if not self.vector_store:
            logger.warning("ChromaDB 연결 실패. 빈 참고 스토리 반환")
            return []
        
        try:
            # 검색 쿼리 구성
            search_query = query_text
            if age_group:
                search_query += f" {age_group}세"
            
            # ChromaDB 검색 수행
            search_results = query_vector_db(
                vector_db=self.vector_store,
                collection_name=self.collection_name,
                query_text=search_query,
                n_results=3
            )
            
            # 결과 포맷팅
            formatted_results = format_query_results(search_results)
            
            logger.info(f"RAG 검색 완료: {len(formatted_results)}개의 유사 스토리 반환")
            return formatted_results
            
        except Exception as e:
            logger.warning(f"RAG 검색 실패: {e}. 빈 참고 스토리 반환")
            return []
    
    def _format_context(self, reference_stories: List[Dict[str, Any]]) -> str:
        """참고 스토리를 컨텍스트 문자열로 포맷팅"""
        if not reference_stories:
            return "참고할 수 있는 스토리가 없습니다."
        
        context_parts = []
        for i, story in enumerate(reference_stories, 1):
            context_parts.append(f"참고 스토리 {i}:")
            context_parts.append(f"제목: {story.get('title', '제목 없음')}")
            context_parts.append(f"주제: {story.get('theme', '주제 없음')}")
            context_parts.append(f"내용: {story.get('content', story.get('summary', '내용 없음'))[:200]}...")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def add_story_to_vectordb(self, story_data: Dict[str, Any]) -> bool:
        """
        스토리 데이터를 벡터 데이터베이스에 추가
        
        Args:
            story_data: 스토리 데이터
            
        Returns:
            bool: 성공 여부
        """
        if not self.vector_store:
            logger.warning("ChromaDB가 초기화되지 않음")
            return False
        
        try:
            # 스토리 ID 생성 또는 가져오기
            story_id = story_data.get("story_id", str(uuid.uuid4()))
            
            # 메타데이터 준비
            metadata = {
                "story_id": story_id,
                "title": story_data.get("title", story_data.get("theme", "제목 없음")),
                "theme": story_data.get("theme", ""),
                "age_group": story_data.get("target_age", story_data.get("age_group", 5)),
                "educational_value": story_data.get("educational_value", ""),
                "summary": story_data.get("plot_summary", story_data.get("summary", ""))
            }
            
            # 문서 내용 구성
            content = story_data.get("plot_summary", "")
            if not content:
                content = f"주제: {metadata['theme']}\n교육적 가치: {metadata['educational_value']}"
            
            # 벡터 DB에 추가
            self.vector_store.add_documents(
                documents=[content],
                metadatas=[metadata],
                ids=[story_id]
            )
            
            logger.info(f"벡터 데이터베이스에 스토리 추가 완료: {story_id}")
            return True
            
        except Exception as e:
            logger.error(f"벡터 데이터베이스에 스토리 추가 실패: {e}")
            return False
    
    def get_similar_stories(self, theme: str, age_group: int, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        유사한 스토리 검색 (동기 버전)
        
        Args:
            theme: 주제 또는 검색어
            age_group: 연령대
            n_results: 반환할 결과 수
            
        Returns:
            List[Dict[str, Any]]: 검색 결과 목록
        """
        if not self.vector_store:
            logger.warning("ChromaDB가 초기화되지 않음")
            return []
        
        try:
            # 검색 쿼리 구성
            search_query = f"{theme} {age_group}세"
            
            # ChromaDB 검색 수행
            search_results = query_vector_db(
                vector_db=self.vector_store,
                collection_name=self.collection_name,
                query_text=search_query,
                n_results=n_results
            )
            
            # 결과 포맷팅
            formatted_results = format_query_results(search_results)
            
            logger.info(f"유사 스토리 검색 결과: {len(formatted_results)}개")
            return formatted_results
            
        except Exception as e:
            logger.error(f"유사 스토리 검색 실패: {e}")
            return []
    
    async def health_check(self) -> bool:
        """RAG 엔진 상태 확인"""
        try:
            # 기본 구성 요소 확인
            if not self.theme_enrichment_chain or not self.rag_chain:
                logger.error("LangChain 체인이 초기화되지 않음")
                return False
            
            # 간단한 테스트
            test_theme = "테스트 주제"
            result = await self.enrich_story_theme(test_theme, 5)
            
            return len(result) > 0
            
        except Exception as e:
            logger.error(f"RAG 엔진 health check 실패: {e}")
            return False 