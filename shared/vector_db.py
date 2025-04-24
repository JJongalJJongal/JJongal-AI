import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

class VectorDB:
    """
    Chroma DB를 관리하는 클래스
    
    Attributes:
        client: Chroma DB 클라이언트
        collection: 현재 사용 중인 컬렉션
        embedding_function: 임베딩 생성 함수
    """
    
    def __init__(self, persist_directory: str = "data/vector_db"):
        """
        VectorDB 초기화
        
        Args:
            persist_directory (str): 벡터 DB 데이터를 저장할 디렉토리 경로
        """
        # 저장 디렉토리 생성
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Chroma DB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 기본 임베딩 함수 설정 (sentence-transformers 사용)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = None
    
    def create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        새로운 컬렉션 생성
        
        Args:
            name (str): 컬렉션 이름
            metadata (Dict[str, Any], optional): 컬렉션 메타데이터
        """
        self.collection = self.client.create_collection(
            name=name,
            embedding_function=self.embedding_function,
            metadata=metadata or {}
        )
    
    def get_collection(self, name: str) -> None:
        """
        기존 컬렉션 가져오기
        
        Args:
            name (str): 컬렉션 이름
        """
        self.collection = self.client.get_collection(
            name=name,
            embedding_function=self.embedding_function
        )
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> None:
        """
        문서를 컬렉션에 추가
        
        Args:
            documents (List[str]): 추가할 문서 리스트
            metadatas (List[Dict[str, Any]], optional): 문서 메타데이터 리스트
            ids (List[str], optional): 문서 ID 리스트
        """
        if self.collection is None:
            raise ValueError("컬렉션이 선택되지 않았습니다. 먼저 create_collection() 또는 get_collection()을 호출하세요.")
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_texts: List[str], n_results: int = 5, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        컬렉션에서 유사한 문서 검색
        
        Args:
            query_texts (List[str]): 검색할 쿼리 텍스트 리스트
            n_results (int): 반환할 결과 수
            where (Dict[str, Any], optional): 필터 조건
            
        Returns:
            Dict[str, Any]: 검색 결과
        """
        if self.collection is None:
            raise ValueError("컬렉션이 선택되지 않았습니다. 먼저 create_collection() 또는 get_collection()을 호출하세요.")
        
        return self.collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where
        )
    
    def delete_collection(self, name: str) -> None:
        """
        컬렉션 삭제
        
        Args:
            name (str): 삭제할 컬렉션 이름
        """
        self.client.delete_collection(name=name)
        if self.collection and self.collection.name == name:
            self.collection = None 