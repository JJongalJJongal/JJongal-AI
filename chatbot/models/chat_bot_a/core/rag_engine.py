"""
부기 (ChatBot A) RAG 엔진

LangChain과 ChromaDB를 활용한 검색 증강 생성 엔진
chat_bot_b와 동일한 LangChain 패턴을 사용하여 일관성 확보
"""

from typing import Dict, List, Any

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI


# Project imports
from chatbot.data.vector_db.core import VectorDB
from shared.utils.logging_utils import get_module_logger
from chatbot.data.vector_db.query import get_similar_stories

logger = get_module_logger(__name__)

class RAGSystem:
    """
    LangChain 기반 RAG 엔진
    
    chat_bot_b의 TextGenerator와 동일한 LangChain 패턴을 사용하여
    검색 증강 생성 기능을 제공
    """
    
    def __init__(self, vector_db: VectorDB, prompts: Dict[str, Any]):
        self.vector_store = vector_db
        self.prompts = prompts
        self.rag_chain = None
        self.context_enhancement_chain = None

        self._initialize_chains()
    
    def _initialize_chains(self):
        """LLM 체인을 초기화합니다."""
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            
            # 스토리 컨텍스트 보강 체인 (기존 enrich_story_theme 대체)
            enrichment_template = self.prompts.get("rag_templates", {}).get(
                "story_enrichment", 
                "다음 요약/소재를 바탕으로 {age_group}세 아이에게 적합한 더 풍부한 동화 아이디어를 제안해주세요.\n\n기본 아이디어: {query_text}\n\n참고 자료:\n{context}"
            )
            enrichment_prompt = ChatPromptTemplate.from_template(enrichment_template)
            self.context_enhancement_chain = enrichment_prompt | llm | StrOutputParser()

            # 최종 스토리 생성 RAG 체인
            rag_template = self.prompts.get("rag_templates", {}).get(
                "rag_story_generation", 
                "아래 정보를 바탕으로 {age_group}세 어린이에게 적합한 짧고 재미있는 동화의 첫 부분을 생성해 줘. (1~2문단)\n\n[컨텍스트]\n{context}\n\n[요청사항]\n{user_request}"
            )
            rag_prompt = ChatPromptTemplate.from_template(rag_template)
            self.rag_chain = rag_prompt | llm | StrOutputParser()
            
            logger.info("RAGSystem 체인 초기화 완료.")
        except Exception as e:
            logger.error(f"RAGSystem 체인 초기화 중 오류 발생: {e}", exc_info=True)

    async def enhance_story_context(self, query_text: str, age_group: int) -> str:
        """
        주어진 쿼리와 연령대를 바탕으로 관련 스토리를 검색하고,
        이를 통해 더 풍부한 스토리 컨텍스트(아이디어)를 생성합니다.
        (기존 enrich_story_theme 대체)
        """
        logger.info(f"컨텍스트 보강 시작: 쿼리='{query_text}', 연령대={age_group}")
        if not self.context_enhancement_chain or not self.vector_store:
            logger.warning("컨텍스트 보강 체인이 초기화되지 않아 원본 쿼리를 반환합니다.")
            return query_text

        try:
            # 1. 유사 스토리 검색
            reference_stories = await self._retrieve_similar_stories(query_text, age_group)
            
            # 2. 검색된 컨텍스트 포맷팅
            context_str = self._format_retrieved_context(reference_stories)
            if not context_str:
                logger.info("참고할 만한 스토리를 찾지 못해 원본 쿼리를 기반으로 컨텍스트를 생성합니다.")
                context_str = "참고 자료 없음"

            # 3. LLM을 통한 컨텍스트 보강
            enhanced_context = await self.context_enhancement_chain.ainvoke({
                "query_text": query_text,
                "age_group": age_group,
                "context": context_str
            })
            
            logger.info(f"컨텍스트 보강 완료: {query_text} -> {enhanced_context[:50]}...")
            return enhanced_context
        except Exception as e:
            logger.error(f"컨텍스트 보강 중 오류 발생: {e}", exc_info=True)
            return query_text # 오류 발생 시 원본 텍스트 반환

    async def _retrieve_similar_stories(self, query_text: str, age_group: int, n_results: int = 5) -> List[Dict[str, Any]]:
        """유사 스토리를 VectorDB에서 검색"""
        logger.info(f"유사 스토리 검색: 쿼리='{query_text}', 연령대={age_group}")
        age_group_str = f"{age_group}세" # 예: "5세", "6세"
        # 실제 age_group 형식에 맞게 조정 필요 (예: '5-7세')
        # 이 부분은 age_group_utils와 연동하여 더 정교하게 만들 수 있음
        metadata_filter = {"age_group": {"$like": f"%{age_group}%"}}

        try:
            similar_stories = get_similar_stories(
                vector_db=self.vector_store,
                query_text=query_text,
                n_results=n_results,
                metadata_filter=metadata_filter,
                doc_type="summary"  # 요약본 기반으로 유사 스토리 검색
            )
            logger.info(f"{len(similar_stories)}개의 유사한 스토리 검색 완료.")
            return similar_stories
        except Exception as e:
            logger.error(f"유사 스토리 검색 중 오류 발생: {e}", exc_info=True)
            return []

    def _format_retrieved_context(self, stories: List[Dict[str, Any]]) -> str:
        """검색된 스토리 목록을 LLM 프롬프트에 넣기 좋은 형식으로 변환"""
        if not stories:
            return ""
        
        context_parts = []
        for i, story in enumerate(stories):
            metadata = story.get('metadata', {})
            title = metadata.get('title', '제목 없음')
            summary = story.get('document', '') # 'document'에 요약 내용이 저장됨
            
            context_parts.append(f"참고 {i+1}: \"{title}\"\n줄거리: {summary}")

        return "\n\n".join(context_parts)

    def _create_metadata_from_story_data(self, story_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        스토리 데이터에서 메타데이터를 추출 (기존 로직 단순화).
        이제 DB에 직접 저장하는 용도가 아니므로, 필요한 정보만 추출.
        """
        return {
            "title": story_data.get("title", "제목 없음"),
            "summary": story_data.get("summary", ""),
            "characters": story_data.get("characters", []),
            "setting": story_data.get("setting", {}),
            "age_group": story_data.get("age_group"),
            "educational_value": story_data.get("educational_value", "")
        }

    def _format_context_for_generation(self, metadata: Dict[str, Any], additional_request: str) -> str:
        """
        최종 스토리 생성을 위한 컨텍스트 문자열 포맷팅
        """
        context_parts = []
        if metadata.get('title'):
            context_parts.append(f"제목 아이디어: {metadata['title']}")
        if metadata.get('summary'):
            context_parts.append(f"줄거리: {metadata['summary']}")
        if metadata.get('characters'):
            # character가 dict의 list일 경우 이름만 추출
            char_names = [c.get('name', '') for c in metadata['characters'] if isinstance(c, dict)]
            if char_names:
                context_parts.append(f"주요 등장인물: {', '.join(char_names)}")
        if metadata.get('setting'):
            setting_desc = metadata['setting'].get('description', '')
            if setting_desc:
                context_parts.append(f"배경: {setting_desc}")
        if metadata.get('educational_value'):
            context_parts.append(f"교훈: {metadata['educational_value']}")
        
        if additional_request:
            context_parts.append(f"추가 요청사항: {additional_request}")

        return "\n".join(context_parts)

    async def generate_story_part(self, story_context: Dict, user_request: str, age_group: int) -> str:
        """RAG를 사용해 이야기의 일부를 생성합니다."""
        logger.info("RAG 스토리 생성 시작...")
        if not self.rag_chain:
            logger.error("RAG 체인이 초기화되지 않았습니다.")
            return "오류: 스토리 생성 시스템이 준비되지 않았습니다."

        try:
            # 검색된 컨텍스트와 사용자 요청을 결합
            formatted_context = self._format_context_for_generation(story_context, user_request)
            
            response = await self.rag_chain.ainvoke({
                "context": formatted_context,
                "user_request": user_request, # user_request를 프롬프트에 직접 전달
                "age_group": age_group
            })
            
            logger.info("RAG 스토리 생성 완료.")
            return response
        except Exception as e:
            logger.error(f"스토리 생성 중 오류: {e}", exc_info=True)
            return f"오류: 스토리를 생성하는 중 문제가 발생했습니다. ({e})"

    async def get_similar_stories(self, query_text: str, age_group: int, n_results: int = 5) -> list:
        """VectorDB에서 유사 스토리를 검색하는 public 메서드"""
        return await self._retrieve_similar_stories(query_text, age_group, n_results)

async def main_test():
    """테스트용 메인 함수"""
    from dotenv import load_dotenv
    from shared.utils.vector_db_utils import get_db_type_path
    
    load_dotenv()
    
    # DB 경로 설정
    db_path = get_db_type_path(db_type="summary")
    if not db_path.exists():
        logger.error(f"테스트를 위한 DB 경로를 찾을 수 없습니다: {db_path}")
        return

    # 시스템 초기화
    vector_db = VectorDB(persist_directory=str(db_path))
    vector_db.get_collection("fairy_tales")

    # TODO: 프롬프트 로더 구현 후 수정 필요
    prompts = {
        "rag_templates": {
            "story_enrichment": "다음 아이디어를 바탕으로 5세 아이에게 맞는 동화 아이디어를 더 풍부하게 만들어주세요.\n\n기본 아이디어: {query_text}\n\n참고 자료:\n{context}",
            "rag_story_generation": "아래 정보를 바탕으로 5세 어린이를 위한 짧은 동화의 시작 부분을 만들어주세요.\n\n[컨텍스트]\n{context}\n\n[요청사항]\n{user_request}"
        }
    }
    rag_system = RAGSystem(vector_db, prompts)
    
    if not rag_system.context_enhancement_chain or not rag_system.rag_chain:
        logger.error("RAG 시스템 초기화 실패. 테스트를 중단합니다.")
        return
        
    # 1. 컨텍스트 보강 테스트
    test_query = "용감한 토끼"
    enhanced_context = await rag_system.enhance_story_context(test_query, 5)
    print("--- 컨텍스트 보강 결과 ---")
    print(enhanced_context)
    print("-" * 25)

    # 2. 스토리 생성 테스트
    story_idea = {
        "title": "모험을 떠난 아기 토끼",
        "summary": enhanced_context,
        "characters": [{"name": "토토"}],
        "setting": {"description": "신비로운 숲"},
        "educational_value": "용기"
    }
    user_request = "토토가 숲에서 신비한 친구를 만나는 장면으로 시작해줘."
    
    story_part = await rag_system.generate_story_part(story_idea, user_request, 5)
    print("\n--- 스토리 생성 결과 ---")
    print(story_part)
    print("-" * 25)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main_test()) 