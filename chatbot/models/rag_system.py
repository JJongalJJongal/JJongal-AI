from langchain_community.document_loaders import JSONLoader

import json

from typing import List, Dict, Optional, Any, Union
import os
import logging
import uuid
from pathlib import Path

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 벡터 DB 관련 모듈 임포트
from chatbot.data.vector_db import VectorDB, query_vector_db, format_query_results, get_similar_stories

# 공통 유틸리티 모듈 임포트
from shared.utils.logging_utils import get_module_logger
from shared.utils.file_utils import ensure_directory, save_json, load_json
from shared.utils.openai_utils import initialize_client, generate_chat_completion
from shared.configs.app_config import get_env_vars, get_project_root

# 로거 설정
logger = get_module_logger(__name__)

class RAGSystem:
    """
    LangChain과 ChromaDB를 활용한 향상된 RAG(Retrieval-Augmented Generation) 시스템

    Attributes:
        persist_directory (Path): ChromaDB 저장 디렉토리 경로
        embeddings (BaseEmbeddings): 임베딩 모델
        db_types (List[str]): 사용 가능한 DB 유형 목록
        main_db (Chroma): 메인 벡터 데이터베이스
        detailed_db (Chroma): 상세 정보 벡터 데이터베이스
        summary_db (Chroma): 요약 정보 벡터 데이터베이스
    """

    def __init__(self, 
                 persist_directory: Optional[str] = None, 
                 use_openai_embeddings: bool = False,  # 한국어 모델 우선 사용
                 use_hybrid_mode: bool = True,
                 memory_cache_size: int = 1000,
                 enable_lfu_cache: bool = True):
        """
        RAG 시스템 초기화

        Args:
            persist_directory (Optional[str]): ChromaDB 저장 디렉토리 경로
            use_openai_embeddings (bool): OpenAI 임베딩 사용 여부 (기본값: False, 한국어 모델 우선)
            use_hybrid_mode (bool): 하이브리드 모드(메모리+디스크) 사용 여부
            memory_cache_size (int): 메모리 캐시 크기
            enable_lfu_cache (bool): LFU 캐시 정책 사용 여부
        """
        # 기본 디렉토리 설정
        if persist_directory is None:
            env_vars = get_env_vars()
            data_dir = env_vars.get("data_dir", "data")
            project_root = get_project_root()
            self.persist_directory = project_root / "chatbot" / data_dir / "vector_db"
        else:
            self.persist_directory = Path(persist_directory)
        
        self.db_types = ["main", "detailed", "summary"]
        self.use_hybrid_mode = use_hybrid_mode
        self.memory_cache_size = memory_cache_size
        self.enable_lfu_cache = enable_lfu_cache
        
        # 임베딩 모델 설정 (한국어 모델 우선)
        if use_openai_embeddings:
            # OpenAI API 키 확인
            try:
                # OpenAI 클라이언트 초기화 테스트
                initialize_client()
                self.embeddings_model = "text-embedding-3-small"
                logger.info("OpenAI 임베딩 모델 초기화")
            except Exception as e:
                logger.warning(f"OpenAI 임베딩 초기화 실패: {e}")
                # 한국어 모델로 폴백
                self.embeddings_model = "nlpai-lab/KURE-v1"
                logger.info("한국어 임베딩 모델로 폴백")
        else:
            # 한국어 모델 우선 사용
            self.embeddings_model = "nlpai-lab/KURE-v1"
            logger.info("한국어 임베딩 모델 사용")
        
        # 벡터 데이터베이스 초기화
        self._initialize_vector_dbs()
        
        # OpenAI 클라이언트 초기화
        try:
            self.openai_client = initialize_client()
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            self.openai_client = None
        
        logger.info("RAG 시스템 초기화 완료:")
        logger.info(f"  - 하이브리드 모드: {use_hybrid_mode}")
        logger.info(f"  - 임베딩 모델: {self.embeddings_model}")
        logger.info(f"  - 메모리 캐시 크기: {memory_cache_size}")
        logger.info(f"  - LFU 캐시: {enable_lfu_cache}")
    
    def _initialize_vector_dbs(self) -> None:
        """벡터 데이터베이스 초기화 (하이브리드 모드 지원)"""
        try:
            # 각 DB 유형별 디렉토리 확인 및 생성
            for db_type in self.db_types:
                db_dir = self.persist_directory / db_type
                ensure_directory(db_dir)
            
            # 모듈화된 VectorDB 객체 생성 (하이브리드 모드 적용)
            self.main_vectordb = VectorDB(
                persist_directory=str(self.persist_directory / "main"),
                embedding_model=self.embeddings_model,
                use_hybrid_mode=self.use_hybrid_mode,
                memory_cache_size=self.memory_cache_size,
                enable_lfu_cache=self.enable_lfu_cache
            )
            
            self.detailed_vectordb = VectorDB(
                persist_directory=str(self.persist_directory / "detailed"),
                embedding_model=self.embeddings_model,
                use_hybrid_mode=self.use_hybrid_mode,
                memory_cache_size=self.memory_cache_size,
                enable_lfu_cache=self.enable_lfu_cache
            )
            
            self.summary_vectordb = VectorDB(
                persist_directory=str(self.persist_directory / "summary"),
                embedding_model=self.embeddings_model,
                use_hybrid_mode=self.use_hybrid_mode,
                memory_cache_size=self.memory_cache_size,
                enable_lfu_cache=self.enable_lfu_cache
            )
            
            # 컬렉션 설정 (기존 컬렉션이 있으면 가져오고, 없으면 생성)
            try:
                self.main_collection = self.main_vectordb.get_collection("fairy_tales")
            except:
                logger.info("main 컬렉션이 없습니다. 새로 생성합니다.")
                self.main_collection = self.main_vectordb.create_collection("fairy_tales")
            
            try:
                self.detailed_collection = self.detailed_vectordb.get_collection("fairy_tales")
            except:
                logger.info("detailed 컬렉션이 없습니다. 새로 생성합니다.")
                self.detailed_collection = self.detailed_vectordb.create_collection("fairy_tales")
            
            try:
                self.summary_collection = self.summary_vectordb.get_collection("fairy_tales")
            except:
                logger.info("summary 컬렉션이 없습니다. 새로 생성합니다.")
                self.summary_collection = self.summary_vectordb.create_collection("fairy_tales")
            
            logger.info("벡터 데이터베이스 초기화 완료 (하이브리드 모드)")
        except Exception as e:
            logger.error(f"벡터 데이터베이스 초기화 실패: {e}")
            self.main_vectordb = None
            self.detailed_vectordb = None
            self.summary_vectordb = None
    
    def add_story_to_vectordb(self, story_data: Dict[str, Any]) -> bool:
        """
        스토리 데이터를 벡터 데이터베이스에 추가

        Args:
            story_data: 스토리 데이터

        Returns:
            bool: 성공 여부
        """
        try:
            # 스토리 ID 생성 또는 가져오기
            story_id = story_data.get("story_id", str(uuid.uuid4()))
            
            # 메타데이터 준비
            metadata = {
                "story_id": story_id,
                "title": story_data.get("title", "Untitled Story"),
                "theme": story_data.get("theme", ""),
                "age_group": story_data.get("target_age", 5),
                "educational_value": story_data.get("educational_value", ""),
                "tags": story_data.get("tags", ""),
                "summary": story_data.get("summary", "")
            }
            
            # Main DB에 추가
            main_content = story_data.get("plot_summary", "")
            self.main_vectordb.add_documents(
                documents=[main_content],
                metadatas=[metadata],
                ids=[f"{story_id}_main"]
            )
            
            # Detailed DB에 추가
            detailed_content = ""
            if "scenes" in story_data:
                for scene in story_data["scenes"]:
                    detailed_content += f"{scene.get('title', '')}: {scene.get('description', '')}\n"
                    if "dialogues" in scene:
                        for dialogue in scene["dialogues"]:
                            detailed_content += f"{dialogue.get('character', '')}: {dialogue.get('text', '')}\n"
            else:
                detailed_content = story_data.get("plot_summary", "")
                
            self.detailed_vectordb.add_documents(
                documents=[detailed_content],
                metadatas=[metadata],
                ids=[f"{story_id}_detailed"]
            )
            
            # Summary DB에 추가
            summary_content = f"제목: {metadata['title']}\n주제: {metadata['theme']}\n교육적 가치: {metadata['educational_value']}\n요약: {metadata['summary']}"
            self.summary_vectordb.add_documents(
                documents=[summary_content],
                metadatas=[metadata],
                ids=[f"{story_id}_summary"]
            )
            
            logger.info(f"벡터 데이터베이스에 스토리 추가 완료: {story_id}")
            return True
        except Exception as e:
            logger.error(f"벡터 데이터베이스에 스토리 추가 실패: {e}")
            return False
    
    def get_similar_stories(self, theme: str, age_group: int, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        유사한 스토리 검색

        Args:
            theme: 주제 또는 검색어
            age_group: 연령대
            n_results: 반환할 결과 수

        Returns:
            List[Dict[str, Any]]: 검색 결과 목록 (title, theme, summary 등 포함)
        """
        try:
            # 모듈화된 함수를 사용하여 유사 스토리 검색
            stories = get_similar_stories(
                vector_db=self.summary_vectordb,
                query_text=theme,
                age_group=age_group,
                n_results=n_results,
                collection_name="fairy_tales"
            )
            
            # 결과 형식화
            formatted_results = []
            for story in stories:
                formatted_results.append({
                    "title": story.get("title", "제목 없음"),
                    "theme": story.get("theme", ""),
                    "summary": story.get("summary", ""),
                    "age_group": story.get("age_group", 0),
                    "similarity": story.get("similarity", 0)
                })
                
            logger.info(f"유사 스토리 검색 결과: {len(formatted_results)}개")
            return formatted_results
            
        except Exception as e:
            logger.error(f"유사 스토리 검색 실패: {e}")
            return []
    
    def get_few_shot_prompt(self, age_group: int, theme: str) -> str:
        """
        Few-shot 프롬프트 생성

        Args:
            age_group: 연령대
            theme: 주제

        Returns:
            str: Few-shot 프롬프트
        """
        # 유사 동화 검색
        similar_stories = self.get_similar_stories(theme, age_group, n_results=2)
        similar_stories_text = ""
        for i, story in enumerate(similar_stories, 1):
            similar_stories_text += f"예시 {i}: '{story['title']}'\n"
            similar_stories_text += f"줄거리: {story['summary']}\n"
            similar_stories_text += f"교육적 가치: {story['educational_value']}\n\n"
        
        # Few-shot 프롬프트 생성
        few_shot_prompt = f"""
Few-shot 예시:

1. {age_group}세 아이를 위한 동화 작성 시 고려사항:
   - 문장 길이: {10 if age_group < 6 else 15}단어 이내의 간결한 문장
   - 어휘 수준: 일상적이고 구체적인 단어 사용
   - 구조: 명확한 시작, 중간, 끝 구조
   - 교훈: 명시적이고 이해하기 쉬운 메시지

2. 주제 '{theme}'에 적합한 요소:
   - 캐릭터: 공감할 수 있는 주인공, 1-2명의 조력자, 선명한 성격
   - 갈등: 아이들이 공감할 수 있는 실생활 문제 또는 판타지 요소
   - 해결책: 협력, 창의성, 인내 등의 가치를 통한 해결

3. 유사 동화 분석:
{similar_stories_text}

위 예시들을 참고하여, 다음 구조로 동화를 작성해 주세요:
1. 도입부: 주인공과 배경 소개 (1-2 문단)
2. 전개부: 문제 상황 발생 (1-2 문단)
3. 위기: 문제 심화 (1-2 문단)
4. 해결: 문제 해결 과정 (1-2 문단)
5. 결말: 교훈과 마무리 (1 문단)
"""
        
        return few_shot_prompt
    
    def enrich_story_theme(self, theme: str, age_group: int) -> str:
        """
        스토리 주제 풍부화

        Args:
            theme: 원본 주제
            age_group: 연령대

        Returns:
            str: 풍부화된 주제
        """
        try:
            # 요약 DB에서 주제와 관련된 내용 검색
            query_results = self.summary_vectordb.query(
                query_texts=[f"{theme} for {age_group} year old children"],
                n_results=3,
            )
            
            # 관련 내용 추출
            docs_list = query_results.get("documents", [[]])[0]
            metadatas_list = query_results.get("metadatas", [[]])[0]
            documents = []
            if docs_list and metadatas_list and len(docs_list) == len(metadatas_list):
                for i in range(len(docs_list)):
                    documents.append(Document(page_content=docs_list[i], metadata=metadatas_list[i]))
            
            related_content = "\n\n".join([doc.page_content for doc in documents])
            
            # 주제 풍부화 프롬프트
            prompt_text = """당신은 아동 동화 전문가입니다. 다음 기본 주제를 받아 더 풍부하고 교육적인 주제로 발전시켜주세요.
            
            기본 주제: {theme}
            연령대: {age_group}세
            
            참고 자료:
            {related_content}
            
            위 정보를 바탕으로, 기본 주제를 발전시켜 더 풍부하고 교육적인 주제로 만들어 주세요.
            2-3문장으로 구체적인 주제와 핵심 교육 가치를 포함해 주세요.
            """
            
            # 공통 유틸리티 모듈 활용
            if self.openai_client:
                messages = [
                    {"role": "system", "content": "당신은 아동 동화 전문가입니다."},
                    {"role": "user", "content": prompt_text.format(
                        theme=theme,
                        age_group=age_group,
                        related_content=related_content
                    )}
                ]
                
                content, _ = generate_chat_completion(
                    client=self.openai_client,
                    messages=messages,
                    temperature=0.7
                )
                
                if content:
                    logger.info(f"주제 풍부화 완료: {content[:50]}...")
                    return content
            
            # 실패 시 LangChain 사용 
            prompt = ChatPromptTemplate.from_template(prompt_text)
            model = ChatOpenAI(temperature=0.7)
            chain = (
                {"theme": lambda x: theme, 
                 "age_group": lambda x: age_group, 
                 "related_content": lambda x: related_content}
                | prompt
                | model
                | StrOutputParser()
            )
            
            enriched_theme = chain.invoke({})
            logger.info(f"주제 풍부화 완료 (LangChain): {enriched_theme[:50]}...")
            
            return enriched_theme
        except Exception as e:
            logger.error(f"주제 풍부화 실패: {e}")
            return theme
    
    def import_sample_stories(self) -> int:
        """
        샘플 동화 데이터 가져오기

        Returns:
            int: 가져온 스토리 수
        """
        try:
            # 샘플 스토리 데이터
            sample_stories = [
                {
                    "title": "우주 탐험가 릴리",
                    "theme": "우주 탐험과 용기",
                    "target_age": 5,
                    "plot_summary": "호기심 많은 소녀 릴리는 직접 만든 우주선을 타고 별들 사이로 모험을 떠납니다. 미지의 행성에서 길을 잃었지만, 용기와 지혜로 집으로 돌아가는 길을 찾습니다.",
                    "educational_value": "호기심, 탐험 정신, 문제 해결 능력",
                    "tags": "우주, 모험, 5-6세"
                },
                {
                    "title": "숲속 친구들의 비밀",
                    "theme": "자연 보호와 협력",
                    "target_age": 7,
                    "plot_summary": "숲속 동물들은 숲을 위협하는 오염에 맞서기 위해 힘을 합칩니다. 각자의 특별한 능력을 모아 숲을 되살리는 해결책을 찾아냅니다.",
                    "educational_value": "환경 보호, 팀워크, 문제 해결",
                    "tags": "자연, 동물, 7-8세"
                },
            ]
            
            # 벡터 DB에 추가
            added_count = 0
            for story in sample_stories:
                # 스토리 ID 생성
                story["story_id"] = str(uuid.uuid4())
                
                # 벡터 DB에 추가
                self.add_story_to_vectordb(story)
                
                added_count += 1
            
            logger.info(f"샘플 스토리 {added_count}개 추가 완료")
            return added_count
        except Exception as e:
            logger.error(f"샘플 스토리 가져오기 실패: {e}")
            return 0
    
    def add_story(self, title: str, tags: str, summary: str, content: str) -> str:
        """
        새 스토리 추가 편의 메서드

        Args:
            title: 스토리 제목
            tags: 관련 태그 (콤마로 구분)
            summary: 스토리 요약
            content: 상세 내용

        Returns:
            str: 생성된 스토리 ID
        """
        story_id = str(uuid.uuid4())
        
        # 연령대 추출 (태그에서 파싱)
        age_group = 5  # 기본값
        if "4-6세" in tags:
            age_group = 5
        elif "7-9세" in tags:
            age_group = 8
        
        # 스토리 데이터 구성
        story_data = {
            "story_id": story_id,
            "title": title,
            "tags": tags,
            "summary": summary,
            "plot_summary": content,
            "target_age": age_group,
            "theme": summary.split('.')[0]  # 첫 문장을 주제로 사용
        }
        
        # 벡터 DB에 추가
        self.add_story_to_vectordb(story_data)
        
        return story_id
        
    def query(self, query_text: str, use_summary: bool = False) -> Dict[str, Any]:
        """
        쿼리 수행

        Args:
            query_text: 검색 쿼리
            use_summary: 요약 DB 사용 여부

        Returns:
            Dict[str, Any]: 결과
        """
        try:
            # 검색에 사용할 DB 선택
            db = self.summary_vectordb if use_summary else self.main_vectordb
            
            # 검색 수행
            query_results = db.query(
                query_texts=[query_text],
                n_results=3
            )
            
            # 문서 내용 추출 (Document 객체로 변환)
            docs_list = query_results.get("documents", [[]])[0]
            metadatas_list = query_results.get("metadatas", [[]])[0]
            documents = []
            if docs_list and metadatas_list and len(docs_list) == len(metadatas_list):
                for i in range(len(docs_list)):
                    documents.append(Document(page_content=docs_list[i], metadata=metadatas_list[i]))
            
            # docs_content = "\n\n".join([f"문서 {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)]) # 기존 코드
            # Document 객체로 변환되었으므로 page_content 직접 사용
            docs_content = "\n\n".join([f"문서 {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)])
            
            # 공통 유틸리티 모듈 활용
            if self.openai_client:
                prompt_text = """당신은 동화 전문가입니다. 다음 자료를 바탕으로 질문에 답변해주세요.

                자료:
                {docs}
                
                질문: {question}
                
                위 자료를 바탕으로 질문에 답변해주세요.
                """
                
                messages = [
                    {"role": "system", "content": "당신은 동화 전문가입니다."},
                    {"role": "user", "content": prompt_text.format(
                        docs=docs_content,
                        question=query_text
                    )}
                ]
                
                content, _ = generate_chat_completion(
                    client=self.openai_client,
                    messages=messages,
                    temperature=0.3
                )
                
                if content:
                    return {
                        "question": query_text,
                        "answer": content,
                        "sources": [{"title": doc.metadata.get("title", ""), "content": doc.page_content[:100] + "..."} for doc in documents]
                    }
            
            # 실패 시 LangChain 사용
            prompt = ChatPromptTemplate.from_template(
                """당신은 동화 전문가입니다. 다음 자료를 바탕으로 질문에 답변해주세요.

                자료:
                {docs}
                
                질문: {question}
                
                위 자료를 바탕으로 질문에 답변해주세요.
                """
            )
            
            model = ChatOpenAI(temperature=0.3)
            chain = (
                {"docs": lambda x: docs_content, "question": lambda x: query_text}
                | prompt
                | model
                | StrOutputParser()
            )
            
            answer = chain.invoke({})
            
            return {
                "question": query_text,
                "answer": answer,
                "sources": [{"title": doc.metadata.get("title", ""), "content": doc.page_content[:100] + "..."} for doc in documents]
            }
        except Exception as e:
            logger.error(f"쿼리 수행 실패: {e}")
            return {"question": query_text, "answer": "죄송합니다. 검색 중 오류가 발생했습니다.", "sources": []}
    
    def save_data(self, file_path: Union[str, Path]) -> bool:
        """
        RAG 시스템 상태 데이터 저장
        
        Args:
            file_path (Union[str, Path]): 저장할 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 현재 시스템 상태 데이터 수집 (벡터 DB는 별도로 저장되므로 메타데이터만 저장)
            system_data = {
                "db_types": self.db_types,
                "persist_directory": str(self.persist_directory),
                "status": "active"
            }
            
            # 데이터 저장
            return save_json(system_data, file_path)
        except Exception as e:
            logger.error(f"RAG 시스템 데이터 저장 실패: {e}")
            return False
    
    def load_data(self, file_path: Union[str, Path]) -> bool:
        """
        RAG 시스템 상태 데이터 로드
        
        Args:
            file_path (Union[str, Path]): 로드할 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 데이터 로드
            system_data = load_json(file_path)
            if not system_data:
                logger.error("RAG 시스템 데이터 로드 실패: 파일이 없거나 빈 파일")
                return False
            
            # 시스템 설정 업데이트 (벡터 DB는 초기화 시 이미 로드되므로 불필요)
            if "db_types" in system_data:
                self.db_types = system_data["db_types"]
                
            logger.info("RAG 시스템 데이터 로드 완료")
            return True
        except Exception as e:
            logger.error(f"RAG 시스템 데이터 로드 실패: {e}")
            return False
    
    def close(self) -> None:
        """리소스 정리"""
        logger.info("RAG 시스템 종료")

    def get_system_info(self) -> Dict[str, Any]:
        """
        시스템 정보 반환
        
        Returns:
            Dict: 시스템 상태 정보
        """
        try:
            system_info = {
                "hybrid_mode": self.use_hybrid_mode,
                "embedding_model": self.embeddings_model,
                "memory_cache_size": self.memory_cache_size,
                "lfu_cache_enabled": self.enable_lfu_cache,
                "persist_directory": str(self.persist_directory),
                "db_types": self.db_types
            }
            
            # 각 벡터 DB의 캐시 정보 추가
            if self.main_vectordb:
                system_info["main_db_cache"] = self.main_vectordb.get_cache_info()
            if self.detailed_vectordb:
                system_info["detailed_db_cache"] = self.detailed_vectordb.get_cache_info()
            if self.summary_vectordb:
                system_info["summary_db_cache"] = self.summary_vectordb.get_cache_info()
            
            return system_info
            
        except Exception as e:
            logger.error(f"시스템 정보 조회 실패: {e}")
            return {"error": str(e)}
