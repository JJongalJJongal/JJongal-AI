"""
벡터 데이터베이스 핵심 기능을 제공하는 모듈

ChromaDB를 사용한 벡터 데이터베이스 관리 클래스와 기능을 포함합니다.
"""

import os
import uuid
import logging
from typing import Dict, List, Optional, Union, Any

# 필요한 모듈 확인 및 임포트
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    raise ImportError("chromadb 패키지가 설치되어 있지 않습니다. 'pip install chromadb'를 실행하세요.")
    
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence_transformers 패키지가 설치되어 있지 않습니다. 'pip install sentence-transformers'를 실행하세요.")

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class VectorDB:
    """
    ChromaDB를 사용한 벡터 데이터베이스 클래스
    
    이 클래스는 3가지 유형의 데이터베이스를 지원합니다:
    - main: 일반 검색용 (기본)
    - detailed: 세부 정보 검색용
    - summary: 요약 및 주제 검색용
    """

    def __init__(self, persist_directory: str = None, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        VectorDB 클래스 초기화
        
        Args:
            persist_directory: ChromaDB 저장 디렉토리 경로
            embedding_model: 임베딩 모델 이름 (기본값: "all-MiniLM-L6-v2")
        """
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.collection = None
        
        # ChromaDB 클라이언트 설정
        self._setup_client()
        
        # 임베딩 함수 설정
        self._setup_embedding_function()
        
        logger.info(f"VectorDB 초기화 완료: {persist_directory}")
        
    def _setup_client(self):
        """ChromaDB 클라이언트 설정"""
        try:
            if self.persist_directory:
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
            else:
                self.client = chromadb.Client(Settings(anonymized_telemetry=False))
            logger.info("ChromaDB 클라이언트 설정 완료")
        except Exception as e:
            logger.error(f"ChromaDB 클라이언트 설정 실패: {str(e)}")
            raise
    
    def _setup_embedding_function(self):
        """임베딩 함수 설정"""
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
            logger.info(f"임베딩 함수 설정 완료: {self.embedding_model}")
        except Exception as e:
            logger.error(f"임베딩 함수 설정 실패: {str(e)}")
            raise
    
    def create_collection(self, name: str = "fairy_tales", metadata: Dict[str, str] = None):
        """
        새 컬렉션 생성
        
        Args:
            name: 컬렉션 이름
            metadata: 컬렉션 메타데이터
        """
        try:
            self.collection = self.client.create_collection(
                name=name,
                embedding_function=self.embedding_function,
                metadata=metadata or {"description": "꼬꼬북 프로젝트 동화 데이터"}
            )
            logger.info(f"컬렉션 '{name}' 생성 완료")
            return self.collection
        except Exception as e:
            logger.error(f"컬렉션 생성 실패: {str(e)}")
            raise
    
    def get_collection(self, name: str = "fairy_tales"):
        """
        기존 컬렉션 가져오기
        
        Args:
            name: 컬렉션 이름
        """
        try:
            self.collection = self.client.get_collection(
                name=name,
                embedding_function=self.embedding_function
            )
            logger.info(f"컬렉션 '{name}' 가져오기 완료")
            return self.collection
        except Exception as e:
            logger.error(f"컬렉션 가져오기 실패: {str(e)}")
            raise
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None, ids: List[str] = None):
        """
        문서 추가
        
        Args:
            documents: 임베딩할 문서 목록
            metadatas: 문서 메타데이터 목록 (선택 사항)
            ids: 문서 ID 목록 (선택 사항)
        """
        if not self.collection:
            raise ValueError("컬렉션이 설정되지 않았습니다. create_collection 또는 get_collection을 호출하세요.")
        
        if not ids:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"{len(documents)}개 문서 추가 완료")
        except Exception as e:
            logger.error(f"문서 추가 실패: {str(e)}")
            raise
    
    def query(self, query_texts: List[str], n_results: int = 5, where: Dict[str, Any] = None):
        """
        쿼리 실행
        
        Args:
            query_texts: 쿼리 텍스트 목록
            n_results: 반환할 결과 수
            where: 필터링 조건 (선택 사항)
            
        Returns:
            Dict: 쿼리 결과
        """
        if not self.collection:
            raise ValueError("컬렉션이 설정되지 않았습니다. create_collection 또는 get_collection을 호출하세요.")
        
        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where
            )
            logger.info(f"쿼리 실행 완료: {len(results.get('ids', [[]]))} 결과")
            return results
        except Exception as e:
            logger.error(f"쿼리 실행 실패: {str(e)}")
            raise 