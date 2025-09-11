"""
Text Generator for ChatBot B (아리)
기존 TextGenerator의 문제점을 해결한 안정적인 버전
"""

from src.shared.utils.logging import get_module_logger
from src.shared.configs.prompts import load_prompts_config
import uuid
import time
from typing import Dict, List, Optional, Any
import asyncio

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

logger = get_module_logger(__name__)

class TextGenerator:
    """안정적인 텍스트 생성기 - VectorDB RAG 지원"""
    
    def __init__(self, openai_client, temperature: float = 0.8, model_name: str = "gpt-4o-mini", 
                 vector_db_path: str = None, enable_rag: bool = True):
        self.openai_client = openai_client
        self.temperature = temperature
        self.model_name = model_name
        self.vector_db_path = vector_db_path
        self.enable_rag = enable_rag
        
        # ChatBot B (아리) uses Detailed DB for comprehensive story generation
        if enable_rag and not vector_db_path:
            import os
            base_path = os.getenv("VECTOR_DB_PATH") or os.getenv("CHROMA_DB_PATH") or "data/chroma_db"
            self.vector_db_path = os.path.join(base_path, "detailed")
            logger.info(f"ChatBot B (아리): Detailed DB 경로 설정 - {self.vector_db_path}")
        
        # 프롬프트 로드
        try:
            self.prompts = load_prompts_config("chatbot_b")
            logger.info("프롬프트 로드 성공")
        except Exception as e:
            logger.error(f"프롬프트 로드 실패: {e}")
            self.prompts = self._get_default_prompts()
        
        # LLM 초기화 (API 키가 없을 때는 None으로 설정)
        try:
            self.llm = ChatOpenAI(
                temperature=temperature,
                model=model_name,
                api_key=openai_client.api_key if openai_client else "test-key"
            ) if openai_client else None
        except Exception as e:
            logger.warning(f"LLM 초기화 실패 (API 키 없음): {e}")
            self.llm = None
        
        # VectorDB 설정 (RAG 지원)
        self.vector_db = None
        if enable_rag:
            self._setup_vector_db()
        
        # 기본 체인 생성
        self._setup_chains()
        
        logger.info(f"TextGenerator 초기화 완료 (RAG: {enable_rag})")
    
    def _setup_vector_db(self):
        """VectorDB 설정 (RAG 기능)"""
        try:
            if self.vector_db_path and self.enable_rag:
                import os
                if os.path.exists(self.vector_db_path):
                    from src.data.vector_db.core import LFUHybridVectorDB
                    
                    self.vector_db = LFUHybridVectorDB(
                        persist_directory=self.vector_db_path,
                        collection_name="fairy_tales_detailed",  # ChatBot B uses Detailed collection
                        use_openai=False,  # Use KURE-v1 Korean embeddings
                        enable_bm25=True
                    )
                    logger.info("✅ TextGenerator: Detailed DB 연결 성공")
                else:
                    logger.warning(f"VectorDB 경로가 존재하지 않음: {self.vector_db_path}")
        except Exception as e:
            logger.warning(f"VectorDB 초기화 실패: {e}")
            self.vector_db = None
    
    def _get_default_prompts(self):
        """기본 프롬프트 반환"""
        return {
            "story_generation": {
                "system_prompt": "당신은 창의적이고 따뜻한 아동 동화 작가 '아리'입니다. 4-9세 아이들을 위한 재미있고 교육적인 동화를 만듭니다.\n\n참고 스토리: {reference_stories}",
                "user_prompt": "아이 이름: {child_name}\\n나이: {age}세\\n관심사: {interests}\\n대화 요약: {conversation_summary}\\n스토리 요소: {story_elements}\\n\\n위 정보를 바탕으로 3-4개 챕터로 구성된 동화를 만들어주세요."
            }
        }
    
    def _setup_chains(self):
        """체인 설정"""
        try:
            story_config = self.prompts.get("story_generation", {})
            
            system_prompt = story_config.get(
                "system_prompt", 
                "당신은 창의적인 아동 동화 작가입니다."
            )
            
            user_prompt = story_config.get(
                "user_prompt",
                "아이 이름: {child_name}\\n나이: {age}세\\n\\n재미있는 동화를 만들어주세요."
            )
            
            self.story_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_prompt)
            ])
            
            if self.llm:
                self.story_chain = self.story_prompt | self.llm | StrOutputParser()
                logger.info("체인 설정 완료")
            else:
                self.story_chain = None
                logger.warning("LLM이 없어 체인 설정을 건너뜀")
            
        except Exception as e:
            logger.error(f"체인 설정 실패: {e}")
            self.story_chain = None
    
    async def generate_story(self, story_request: Dict[str, Any]) -> Dict[str, Any]:
        """스토리 생성 - RAG 지원"""
        try:
            # 입력 데이터 준비
            child_name = story_request.get("child_name", "아이")
            age = story_request.get("age", 7)
            interests = ", ".join(story_request.get("interests", []))
            conversation_summary = story_request.get("conversation_summary", "재미있는 모험 이야기")
            story_elements = str(story_request.get("story_elements", {}))
            
            # RAG 검색 (VectorDB 사용)
            reference_stories = ""
            if self.vector_db and self.enable_rag:
                try:
                    # 검색 쿼리 구성
                    search_query = f"{conversation_summary} {interests} {age}세"
                    
                    # VectorDB에서 관련 스토리 검색
                    search_results = self.vector_db.similarity_search(
                        search_query, 
                        k=3,  # 상위 3개 스토리
                        metadata_filter={"age_group": "4-7세" if age <= 7 else "8-9세"}
                    )
                    
                    # 검색 결과 포맷팅
                    if search_results:
                        reference_stories = "\\n\\n".join([
                            f"참고스토리 {i+1}: {result.get('content', '')[:300]}..." 
                            for i, result in enumerate(search_results)
                        ])
                        logger.info(f"RAG 검색 완료: {len(search_results)}개 참고 스토리")
                except Exception as e:
                    logger.warning(f"RAG 검색 실패: {e}")
                    reference_stories = ""
            
            # 스토리 생성
            story_content = await self.story_chain.ainvoke({
                "child_name": child_name,
                "age": age,
                "interests": interests,
                "conversation_summary": conversation_summary,
                "story_elements": story_elements,
                "reference_stories": reference_stories
            })
            
            # 결과 구성
            story_id = str(uuid.uuid4())
            
            # 간단한 챕터 분할 (줄바꿈 기준)
            paragraphs = [p.strip() for p in story_content.split('\\n\\n') if p.strip()]
            
            chapters = []
            for i, paragraph in enumerate(paragraphs[:4]):  # 최대 4챕터
                chapters.append({
                    "chapter_number": i + 1,
                    "title": f"챕터 {i + 1}",
                    "content": paragraph,
                    "word_count": len(paragraph)
                })
            
            return {
                "story_id": story_id,
                "title": f"{child_name}의 모험 이야기",
                "chapters": chapters,
                "status": "completed",
                "generation_time": time.time(),
                "child_name": child_name,
                "target_age": age
            }
            
        except Exception as e:
            logger.error(f"스토리 생성 실패: {e}")
            raise
    
    async def generate_title(self, story_elements: Dict[str, Any], age: int) -> str:
        """제목 생성"""
        try:
            title_config = self.prompts.get("title_generation", {})
            
            if title_config:
                title_prompt = ChatPromptTemplate.from_messages([
                    ("system", title_config.get("system_prompt", "제목을 만드는 전문가입니다.")),
                    ("human", title_config.get("user_prompt", "제목을 만들어주세요: {elements}"))
                ])
                
                title_chain = title_prompt | self.llm | StrOutputParser()
                title = await title_chain.ainvoke({
                    "elements": str(story_elements),
                    "age": age
                })
                return title.strip()
            else:
                # 기본 제목 생성
                main_character = story_elements.get("main_character", "아이")
                setting = story_elements.get("setting", "")
                if setting:
                    return f"{main_character}의 {setting} 모험"
                else:
                    return f"{main_character}의 모험 이야기"
                    
        except Exception as e:
            logger.warning(f"제목 생성 실패, 기본 제목 사용: {e}")
            return f"{story_elements.get('main_character', '아이')}의 모험 이야기"