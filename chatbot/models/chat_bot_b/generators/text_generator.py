"""
텍스트 생성기

LangChain + ChromaDB RAG System 을 활용한 한국 동화 생성
"""
from shared.utils.logging_utils import get_module_logger
import uuid
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import json

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# Project imports
from .base_generator import BaseGenerator
from chatbot.data.vector_db.core import VectorDB
from chatbot.data.vector_db.query import query_vector_db, format_query_results

# logging 설정
logger = get_module_logger(__name__)

class TextGenerator(BaseGenerator):
    def __init__(self,
                 openai_client = None,
                 vector_db_path: str = None,
                 collection_name: str = "fairy_tales",
                 prompts_file_path: str = "chatbot/data/prompts/chatbot_b_prompts.json",
                 max_retries: int = 3,
                 model_name: str = "gpt-4o",
                 temperature: float = 0.7):
        """
        Args:
            openai_client: OpenAI 클라이언트
            vector_db_path: ChromaDB 데이터베이스 경로
            collection_name: ChromaDB 컬렉션 이름
            prompts_file_path: 프롬프트 파일 경로
            max_retries: 최대 재시도 횟수
            model_name: 사용할 LLM 모델명
            temperature: 생성 온도
        """
        super().__init__(max_retries=max_retries, timeout=120.0)
        
        self.openai_client = openai_client
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        self.prompts_file_path = prompts_file_path
        self.model_name = model_name
        self.temperature = temperature
        
        # LangChain 구성
        self.vector_store = None
        self.retriever = None
        self.text_chain = None
        self.prompts = None
        
        # 초기화
        self._initialize_components()
        
    def _initialize_components(self):
        """LangChain 구성 요소 초기화"""
        try:
            # 1. Prompot load
            self._load_prompts()
            
            # 2. ChromaDB 초기화
            self._initialize_vector_db()
            
            # 3. LangChain 체인 설정
            self._setup_langchain_chain()
            
            logger.info("TextGenerator 초기화 완료")
        
        except Exception as e:
            logger.error(f"TextGenerator 초기화 실패 : {e}")
            raise
    def _load_prompts(self):
        """프롬프트 파일 Load"""
        try:
            with open(self.prompts_file_path, 'r', encoding='utf-8') as f:
                self.prompts = json.load(f) # 프롬프트 파일 로드
            logger.info(f"프롬프트 파일 로드 완료 : {self.prompts_file_path}")
        
        except Exception as e:
            logger.error(f"프롬프트 파일 로드 실패 : {e}")
            raise
    
    def _initialize_vector_db(self):
        """ChromaDB 초기화"""
        if not self.vector_db_path:
            logger.warning("ChromaDB 경로가 설정되지 않음. RAG 기능 비활성화")
            return
        
        try:
            self.vector_store = VectorDB(persist_directory=self.vector_db_path) # ChromaDB 인스턴스 생성
            
            # 컬렉션 존재 확인
            try:
                collection = self.vector_store.get_collection(self.collection_name)
                logger.info(f"ChromaDB 컬렉션 '{self.collection_name}' 연결 완료")
            except Exception as e:
                logger.warning(f"컬렉션 '{self.collection_name}' 연결 실패 : {e}")
        
        except Exception as e:
            logger.error(f"ChromaDB 초기화 실패 : {e}")
            raise
    
    def _setup_langchain_chain(self):
        """LangChain 체인 설정"""
        try:
            # 1. 프롬프트 템플릿 생성
            system_message = self.prompts["story_generation_templates"]["detailed_story_system_message"]
            
            prompt_template = ChatPromptTemplate.from_template(system_message)
            
            # 2. LLM Model 설정
            llm = ChatOpenAI(
                temperature = self.temperature,
                model = self.model_name,
                api_key = self.openai_client.api_key if self.openai_client else None
            )
            
            # 3. 체인 구성
            self.text_chain = prompt_template | llm | StrOutputParser()
            
            logger.info("LangChain 체인 설정 완료")
            
        except Exception as e:
            logger.error(f"LangChain 체인 설정 실패 : {e}")
            raise
    
    async def generate(self,
                       input_data: Dict[str, Any],
                       progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        
        """
        동화 텍스트 생성 (상세한 스토리)
        
        Args:
            input_data: {
                "theme": "동화 주제",
                "child_name": "아이 이름",
                "age_group": "연령대",
                "interests": ["관심사1", "관심사2"...],
                "plot_summary": "요약 줄거리",
                "educational_value": "교육적 가치"
            }
            progress_callback: 진행 상황
        
        Returns:
            {
                "story_id": "생성된 스토리 ID",
                "title": "동화 제목",
                "chapters": [
                    {
                        "chapter_number": 1,
                        "chapter_title": "챕터 제목",
                        "chapter_content": "챕터 내용"
                    }
                ],
                "metadata": {
                    "generation_time": 생성시간,
                    "model_used": "사용된 모델",
                    "rag_sources": ["참고 소스들"]
                }
            } 
        """
        story_id = str(uuid.uuid4()) # 스토리 ID 생성
        self.current_task_id = story_id # 현재 작업 ID 설정
        
        try:
            # 진행 상태 호출
            if progress_callback:
                await progress_callback({
                    "step": "text_generation", # 현재 단계
                    "status": "starting", # 상태
                    "story_id": story_id, # 스토리 ID
                })
            
            # 1. RAG 검색 수행
            reference_stories = await self._retrieve_similar_stories(input_data)
            
            # 진행 상태 호출
            if progress_callback:
                await progress_callback({
                    "step": "text_generation", # 현재 단계
                    "status": "rag_completed", # 상태
                    "reference_count": len(reference_stories), # 참고 스토리 수
                })

            # 2. Prompt 데이터 준비
            prompt_data = self._prepare_prompt_data(input_data, reference_stories)
            
            # 3. LangChain 체인으로 텍스트 생성
            if progress_callback:
                await progress_callback({
                    "step": "text_generation", # 현재 단계
                    "status": "generation_text" # 상태
                })

            generated_text = await self.text_chain.ainvoke(prompt_data)
            
            # 4. 결과 파싱 및 구조화
            story_data = self._parse_generated_story(generated_text)
            
            if progress_callback:
                await progress_callback({
                    "step": "text_generation", # 현재 단계
                    "status": "generation_completed", # 상태
                    "chapters_generated": len(story_data.get("chapters", [])) # 챕터 수
                })
            
            return {
                "story_id": story_id, # 스토리 ID
                **story_data,
                "metadata": {
                    "generation_time": self.total_generation_time, # 생성 시간
                    "model_used": self.model_name, # 사용된 모델
                    "rag_sources": [ref.get("source", "") for ref in reference_stories], # 참고 소스
                    "input_data": input_data # 입력 데이터
                }
            }
        except Exception as e:
            logger.error(f"텍스트 생성 실패 (story_id : {story_id}) : {e}")
            raise
    
    async def _retrieve_similar_stories(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ChromaDB에서 유사 스토리 검색"""
        
        if not self.vector_store: # ChromaDB 연결 확인
            logger.warning("ChromaDB 연결 실패. 빈 참고 스토리 반환")
            return []
        
        try: 
            # 검색 쿼리 구성
            search_query = f"{input_data.get('theme', '')} {input_data.get('plot_summary', '')}"
            
            # ChromaDB 검색 수행
            search_results = query_vector_db(
                vector_db = self.vector_store, # 벡터 DB 인스턴스
                collection_name = self.collection_name, # 컬렉션 이름
                query_text = search_query, # 검색 쿼리
                n_results = 5 # 상위 5개 결과
            )
            
            # 결과 포맷팅
            formatted_results = format_query_results(search_results)
            
            logger.info(f"RAG 검색 완료 : {len(formatted_results)}개의 유사 스토리 반환")
            
            return formatted_results
        
        except Exception as e:
            logger.warning(f"RAG 검색 실패 : {e}. 빈 참고 스토리 반환")
            return []
        
    def _prepare_prompt_data(self, input_data: Dict[str, Any], reference_stories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prompt 데이터 준비"""
        
        # 참고 스토리 텍스트 구성
        reference_text = ""
        if reference_stories:
            reference_text = "\n\n".join([
                f"참고 스토리 {i+1}: {story.get('text', story.get('content', ''))}"
                for i, story in enumerate(reference_stories[:3]) # 상위 3개만 사용
            ])
        else:
            reference_text = "참고 스토리가 없습니다."
        
        # 캐릭터 정보 JSON 형식으로 구성
        characters = input_data.get("characters", [])
        if isinstance(characters, list) and characters:
            characters_json = json.dumps(characters, ensure_ascii=False)
        else:
            characters_json = '[{"name": "주인공", "type": "child"}]'
        
        # 연령대 정보
        age_group = input_data.get("age_group", 5)
        target_age = age_group
        
        return {
            "reference_stories": reference_text, # 참고 스토리 텍스트
            "theme": input_data.get("theme", ""), # 주제
            "child_name": input_data.get("child_name", ""), # 아이 이름
            "age_group": age_group, # 연령대
            "target_age": target_age, # 타겟 연령 (템플릿용)
            "interests": ", ".join(input_data.get("interests", [])), # 관심사
            "plot_summary": input_data.get("plot_summary", ""), # 요약 줄거리
            "educational_value": input_data.get("educational_value", ""), # 교육적 가치
            "characters": ", ".join([c.get("name", c) if isinstance(c, dict) else str(c) for c in characters]), # 캐릭터 이름들
            "characters_json": characters_json, # 캐릭터 JSON
            "setting": input_data.get("setting", ""), # 배경
        }
   
    def _parse_generated_story(self, generated_text: str) -> Dict[str, Any]:
       """생성된 텍스트를 구조화된 스토리로 파싱"""
       
       try:
           # JSON 형태로 파싱 시도
           if generated_text.strip().startswith('{'):
               story_data = json.loads(generated_text) # JSON 파싱
               return story_data

           # JSON 이 아닌 경우 text parsing
           return self._parse_text_story(generated_text)
       
       except json.JSONDecodeError:
           # JSON 파싱 실패 시 text parsing
           return self._parse_text_story(generated_text)
               
    def _parse_text_story(self, text: str) -> Dict[str, Any]:
        """일반 텍스트를 스토리 구조로 parsing"""
        
        lines = text.strip().split("\n") # 줄 단위로 분리
        title = "생성된 동화" # 제목
        chapters = [] # 챕터 리스트
        
        current_chapter = None # 현재 챕터
        current_content = [] # 현재 챕터 내용
        
        for line in lines:
            line = line.strip() # 공백 제거
            
            if not line:
                continue # 빈 줄 건너뛰기
            
            # 제목 추출
            if line.startswith("제목:"):
                title = line.split(":", 1)[1].strip() # 제목 추출
                continue
            
            # 챕터 시작 감지
            if any(keyword in line for keyword in ["챕터", "Chapter", "장", "편"]):
                # 이전 챕터 저장
                if current_chapter:
                    current_chapter["content"] = "\n".join(current_content).strip()
                    chapters.append(current_chapter)
                
                # 새 챕터 시작
                chapter_number = len(chapters) + 1
                current_chapter = {
                    "chapter_number": chapter_number,
                    "chapter_title": line,
                    "content": ""
                }
                current_content = []
            else:
                # 챕터 내용 추가
                if current_chapter:
                    current_content.append(line)
                else:
                    # 첫 번째 챕터가 명시되지 않은 경우
                    if not chapters:
                        current_chapter = {
                            "chapter_number": 1,
                            "chapter_title": "시작",
                            "content": ""
                        }
                        current_content = []
                    current_content.append(line)
        
        # 마지막 챕터 저장
        if current_chapter:
            current_chapter["content"] = "\n".join(current_content).strip()
            chapters.append(current_chapter)
        
        # 챕터가 없는 경우 전체 텍스트를 하나의 챕터로
        if not chapters:
            chapters = [{
                "chapter_number": 1,
                "chapter_title": "동화",
                "content": text.strip()
            }]
        
        return {
            "title": title,
            "chapters": chapters
        }
        
    async def health_check(self) -> bool:
        """TextGenerator 상태 확인"""
        try:
            # 기본 상태 확인
            if not await super().health_check():
                return False
            
            # LangChain 체인 확인
            if not self.text_chain:
                logger.error("LangChain 체인이 초기화되지 않음")
                return False
            
            # 간단한 생성 테스트
            test_data = {
                "theme": "테스트",
                "child_name": "테스트",
                "age_group": 6,
                "interests": ["테스트"],
                "plot_summary": "테스트 스토리",
                "educational_value": "테스트"
            }
            
            # 타임아웃을 짧게 설정하여 빠른 테스트
            original_timeout = self.timeout
            self.timeout = 10.0
            
            try:
                result = await self.generate(test_data)
                return "story_id" in result
            finally:
                self.timeout = original_timeout
                
        except Exception as e:
            logger.error(f"TextGenerator health check 실패: {e}")
            return False