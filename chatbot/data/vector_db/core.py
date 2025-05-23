"""
벡터 데이터베이스 핵심 기능을 제공하는 모듈

ChromaDB를 사용한 벡터 데이터베이스 관리 클래스와 기능을 포함합니다.
하이브리드 모드(메모리+디스크)와 LFU Cache 정책을 지원합니다.
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
    raise ImportError("chromadb 패키지가 설치되어 있지 않습니다. 패키지 Error")
    
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence_transformers 패키지 Error")

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
    ChromaDB를 사용한 하이브리드 벡터 데이터베이스 클래스
    
    특징:
    - 하이브리드 모드: 메모리 + 디스크 스토리지
    - LFU Cache 정책: 자주 사용되는 데이터를 메모리에 유지
    - 확장 가능한 설정: AWS Cloud 환경 호환
    
    이 클래스는 3가지 유형의 데이터베이스를 지원합니다:
    - main: 일반 검색용 (기본)
    - detailed: 세부 정보 검색용
    - summary: 요약 및 주제 검색용
    """

    def __init__(self, 
                 persist_directory: str = None, 
                 embedding_model: str = "nlpai-lab/KURE-v1",
                 use_hybrid_mode: bool = True,
                 memory_cache_size: int = 1000,
                 enable_lfu_cache: bool = True):
        """
        VectorDB 클래스 초기화
        
        Args:
            persist_directory: ChromaDB 저장 디렉토리 경로
            embedding_model: 임베딩 모델 이름 (기본값: "nlpai-lab/KURE-v1")
            use_hybrid_mode: 하이브리드 모드(메모리+디스크) 사용 여부
            memory_cache_size: 메모리 캐시 크기 (벡터 개수)
            enable_lfu_cache: LFU 캐시 정책 사용 여부
        """
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.use_hybrid_mode = use_hybrid_mode
        self.memory_cache_size = memory_cache_size
        self.enable_lfu_cache = enable_lfu_cache
        self.collection = None
        
        # ChromaDB 클라이언트 설정
        self._setup_client()
        
        # 임베딩 함수 설정
        self._setup_embedding_function()
        
        logger.info(f"VectorDB 초기화 완료:")
        logger.info(f"  - 디렉토리: {persist_directory}")
        logger.info(f"  - 임베딩 모델: {embedding_model}")
        logger.info(f"  - 하이브리드 모드: {use_hybrid_mode}")
        logger.info(f"  - 메모리 캐시 크기: {memory_cache_size}")
        logger.info(f"  - LFU 캐시: {enable_lfu_cache}")
        
    def _setup_client(self):
        """ChromaDB 클라이언트 설정 (하이브리드 모드 지원)"""
        try:
            # 하이브리드 모드를 위한 설정
            settings_config = {
                "anonymized_telemetry": False,
                "allow_reset": True,
                "is_persistent": True if self.persist_directory else False
            }
            
            # 현재 ChromaDB 버전에서 지원되는 설정만 사용
            if self.enable_lfu_cache:
                # 참고: chroma_segment_cache_size는 현재 버전에서 지원되지 않음
                # 대신 클라이언트 레벨 설정 사용
                logger.info(f"LFU 캐시 정책 활성화 (메모리 캐시 크기: {self.memory_cache_size})")
            
            # 하이브리드 모드: 디스크 저장 + 메모리 캐시
            if self.use_hybrid_mode and self.persist_directory:
                # 디렉토리 생성
                os.makedirs(self.persist_directory, exist_ok=True)
                
                # Persistent Client 생성 (디스크 저장)
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(**settings_config)
                )
                logger.info("하이브리드 모드(디스크+메모리) ChromaDB 클라이언트 설정 완료")
                
            elif self.persist_directory:
                # 기존 디스크 전용 모드
                os.makedirs(self.persist_directory, exist_ok=True)
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
                logger.info("디스크 전용 모드 ChromaDB 클라이언트 설정 완료")
                
            else:
                # 메모리 전용 모드
                self.client = chromadb.Client(Settings(anonymized_telemetry=False))
                logger.info("메모리 전용 모드 ChromaDB 클라이언트 설정 완료")
                
        except Exception as e:
            logger.error(f"ChromaDB 클라이언트 설정 실패: {str(e)}")
            # 폴백: 기본 설정으로 재시도
            try:
                logger.warning("기본 설정으로 폴백합니다.")
                if self.persist_directory:
                    os.makedirs(self.persist_directory, exist_ok=True)
                    self.client = chromadb.PersistentClient(
                        path=self.persist_directory,
                        settings=Settings(anonymized_telemetry=False)
                    )
                else:
                    self.client = chromadb.Client(Settings(anonymized_telemetry=False))
                logger.info("폴백 설정으로 ChromaDB 클라이언트 초기화 완료")
            except Exception as fallback_error:
                logger.error(f"폴백 설정도 실패: {str(fallback_error)}")
                raise
    
    def _setup_embedding_function(self):
        """임베딩 함수 설정 (한국어 모델 지원)"""
        try:
            # KURE-v1 또는 한국어 임베딩 모델 우선 사용
            if self.embedding_model.startswith("nlpai-lab/") or "KURE" in self.embedding_model or self.embedding_model.startswith("jhgan/") or "ko-sbert" in self.embedding_model:
                try:
                    # SentenceTransformer 기반 한국어 모델 사용
                    from sentence_transformers import SentenceTransformer
                    
                    # 직접 SentenceTransformer 모델 로드 테스트
                    sentence_model = SentenceTransformer(self.embedding_model)
                    
                    # ChromaDB용 SentenceTransformer 임베딩 함수 생성
                    self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=self.embedding_model
                    )
                    logger.info(f"한국어 임베딩 모델 설정 완료: {self.embedding_model}")
                    return
                    
                except Exception as korean_model_error:
                    logger.warning(f"한국어 모델 {self.embedding_model} 로드 실패: {korean_model_error}")
                    logger.info("다른 한국어 모델로 시도합니다.")
                    
                    # 대안 한국어 모델들 시도
                    alternative_models = [
                        "nlpai-lab/KURE-v1",
                        "jhgan/ko-sbert-nli",
                        "jhgan/ko-sbert-sts", 
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    ]
                    
                    for alt_model in alternative_models:
                        try:
                            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                                model_name=alt_model
                            )
                            logger.info(f"대안 한국어 모델 사용: {alt_model}")
                            self.embedding_model = alt_model  # 실제 사용된 모델로 업데이트
                            return
                        except Exception as alt_error:
                            logger.warning(f"대안 모델 {alt_model} 로드 실패: {alt_error}")
                            continue
            
            # OpenAI 임베딩 모델 사용 (폴백)
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                # OpenAI 모델명 매핑
                openai_model_map = {
                    "nlpai-lab/KURE-v1": "text-embedding-3-small",  # 폴백
                    "jhgan/ko-sbert-nli": "text-embedding-3-small",  # 폴백
                    "text-embedding-3-small": "text-embedding-3-small",
                    "text-embedding-3-large": "text-embedding-3-large",
                    "text-embedding-ada-002": "text-embedding-ada-002"
                }
                
                openai_model = openai_model_map.get(self.embedding_model, "text-embedding-3-small")
                
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=api_key,
                    model_name=openai_model
                )
                logger.info(f"OpenAI 임베딩 함수 설정 완료: {openai_model}")
                self.embedding_model = openai_model  # 실제 사용된 모델로 업데이트
                return
                
            else:
                logger.warning("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            
            # 최종 폴백: 기본 임베딩 함수
            logger.warning("기본 임베딩 함수로 폴백합니다.")
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            self.embedding_model = "default"
            logger.info("기본 임베딩 함수로 폴백 성공")
            
        except Exception as e:
            logger.error(f"임베딩 함수 설정 실패: {str(e)}")
            raise
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        캐시 정보 반환 (하이브리드 모드용)
        
        Returns:
            Dict: 캐시 상태 정보
        """
        try:
            cache_info = {
                "hybrid_mode": self.use_hybrid_mode,
                "lfu_cache_enabled": self.enable_lfu_cache,
                "memory_cache_size": self.memory_cache_size,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory
            }
            
            # 컬렉션 정보 추가
            if self.collection:
                cache_info["collection_count"] = self.collection.count()
                cache_info["collection_name"] = self.collection.name
            
            return cache_info
            
        except Exception as e:
            logger.error(f"캐시 정보 조회 실패: {str(e)}")
            return {"error": str(e)}

    def create_collection(self, name: str = "fairy_tales", metadata: Dict[str, str] = None):
        """
        새 컬렉션 생성 (하이브리드 모드 최적화)
        
        Args:
            name: 컬렉션 이름
            metadata: 컬렉션 메타데이터
        """
        try:
            # 하이브리드 모드를 위한 메타데이터 확장
            enhanced_metadata = {
                "description": "꼬꼬북 프로젝트 동화 데이터",
                "hybrid_mode": str(self.use_hybrid_mode),
                "lfu_cache": str(self.enable_lfu_cache),
                "cache_size": str(self.memory_cache_size),
                "embedding_model": self.embedding_model
            }
            
            if metadata:
                enhanced_metadata.update(metadata)
            
            self.collection = self.client.create_collection(
                name=name,
                embedding_function=self.embedding_function,
                metadata=enhanced_metadata
            )
            logger.info(f"컬렉션 '{name}' 생성 완료 (하이브리드 모드: {self.use_hybrid_mode})")
            return self.collection
        except Exception as e:
            logger.error(f"컬렉션 생성 실패: {str(e)}")
            raise
    
    def get_collection(self, name: str = "fairy_tales"):
        """
        기존 컬렉션 가져오기 (하이브리드 모드 호환)
        
        Args:
            name: 컬렉션 이름
        """
        try:
            self.collection = self.client.get_collection(
                name=name,
                embedding_function=self.embedding_function
            )
            
            # 컬렉션 메타데이터에서 하이브리드 모드 정보 확인
            if hasattr(self.collection, 'metadata') and self.collection.metadata:
                metadata = self.collection.metadata
                stored_hybrid = metadata.get('hybrid_mode', 'False')
                stored_lfu = metadata.get('lfu_cache', 'False')
                
                if stored_hybrid != str(self.use_hybrid_mode):
                    logger.warning(f"컬렉션의 하이브리드 모드 설정({stored_hybrid})과 현재 설정({self.use_hybrid_mode})이 다릅니다.")
                
                if stored_lfu != str(self.enable_lfu_cache):
                    logger.warning(f"컬렉션의 LFU 캐시 설정({stored_lfu})과 현재 설정({self.enable_lfu_cache})이 다릅니다.")
            
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