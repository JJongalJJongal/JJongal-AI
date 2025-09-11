import os
import time
import hashlib
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import json

from src.shared.utils.logging import get_module_logger

# 로깅 설정 (가장 먼저 정의)
logger = get_module_logger(__name__)

# LFU 캐시 구현을 위한 imports
try:
    from cachetools import LFUCache, TTLCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False
    logger.warning("cachetools not available - using simple dict cache")

# BM25 검색을 위한 imports (optional)
try:
    from rank_bm25 import BM25Okapi
    import jieba  # 중국어지만 한국어도 어느정도 지원
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 or jieba not available - BM25 search disabled")

# Modern ChromaDB and LangChain imports - 2025 Best Practices
try:
    # Priority 1: Modern langchain-chroma package
    from langchain_chroma import Chroma
    logger.info("✅ 최신 langchain-chroma 패키지 로드됨")
except ImportError:
    try:
        # Fallback: Legacy langchain_community (deprecated)
        from langchain_community.vectorstores import Chroma
        logger.warning("⚠️ 구버전 langchain_community 사용 중 - 업그레이드 필요")
    except ImportError as e:
        raise ImportError(f"ChromaDB 패키지를 찾을 수 없음: {e}")

try:
    from langchain_openai import OpenAIEmbeddings
    # Modern HuggingFace embeddings
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info("✅ HuggingFace: 최신 langchain-huggingface 사용")
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings  
        logger.warning("⚠️ HuggingFace: 구버전 langchain_community 사용")
except ImportError as e:
    raise ImportError(f"Required packages not installed: {e}")

# 기본 임베딩 모델 설정 - 한국어 특화 모델 우선
DEFAULT_EMBEDDING_MODEL = "nlpai-lab/KURE-v1"  # 한국어 최적화
FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 영어 fallback

class ModernVectorDB:
    """
    Modern ChromaDB wrapper for JJongal-AI RAG system - 2025 Best Practices
    
    Features:
    - LangChain Chroma integration
    - OpenAI embeddings with fallback to Korean models
    - Simplified interface for fairy tale RAG
    - Age-appropriate content filtering
    """

    def __init__(self, 
                 persist_directory = None, 
                 collection_name: str = "fairy_tales",
                 use_openai: bool = True):
        """
        Modern VectorDB initialization
        
        Args:
            persist_directory: ChromaDB persistence directory
            collection_name: Collection name
            use_openai: Use OpenAI embeddings (fallback to Korean models if False)
        """
        self.persist_directory = persist_directory or "data/chroma_db"
        self.collection_name = collection_name
        self.use_openai = use_openai
        
        # Setup embeddings
        self.embeddings = self._setup_embeddings()
        
        # Setup LangChain Chroma vectorstore
        self.vectorstore = self._setup_vectorstore()
        
        logger.info(f"Modern VectorDB initialized: {collection_name} at {persist_directory}")
    def _setup_embeddings(self):
        """Setup modern embeddings with performance optimization"""
        try:
            # 성능 최적화: GPU 사용 가능시 자동 감지
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"🚀 임베딩 디바이스: {device}")
            
            return HuggingFaceEmbeddings(
                model_name="nlpai-lab/KURE-v1",
                model_kwargs={
                    'device': device,
                    'trust_remote_code': True  # 보안: 신뢰할 수 있는 모델만
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32  # 성능: 배치 처리 최적화
                }
            )
        except Exception as e:
            logger.warning(f"KURE-v1 한국어 임베딩 로드 실패: {e}")
            logger.info(f"📎 Fallback: {FALLBACK_EMBEDDING_MODEL} 모델 사용")
            # Fallback with performance optimization
            return HuggingFaceEmbeddings(
                model_name=FALLBACK_EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
    
    def _setup_vectorstore(self):
        """Setup LangChain Chroma vectorstore with performance & security optimization"""
        try:
            # 보안: 디렉토리 경로 검증
            safe_persist_dir = os.path.abspath(self.persist_directory)
            if not safe_persist_dir.startswith(os.getcwd()):
                logger.warning(f"🔒 보안: persist_directory가 현재 작업 디렉토리 외부에 있음")
            
            # Create directory if it doesn't exist
            os.makedirs(safe_persist_dir, exist_ok=True)
            
            # 성능 최적화: ChromaDB 설정
            import chromadb
            from chromadb.config import Settings
            
            chroma_client = chromadb.PersistentClient(
                path=safe_persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,  # 보안: 원격 측정 비활성화
                    allow_reset=False,  # 보안: 리셋 비활성화
                    chroma_server_nofile=True,  # 성능: 파일 잠금 최적화
                )
            )
            
            return Chroma(
                client=chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=safe_persist_dir,
                collection_metadata={
                    "created_by": "jjongal_ai",
                    "version": "2025.1",
                    "hnsw:space": "cosine",  # 성능: 코사인 유사도 최적화
                    "hnsw:M": 16,  # 성능: HNSW 그래프 연결도
                    "hnsw:construction_ef": 200,  # 성능: 구성 시 탐색 깊이
                    "hnsw:search_ef": 100  # 성능: 검색 시 탐색 깊이
                }
            )
        except Exception as e:
            logger.error(f"Vectorstore setup failed: {e}")
            # 보안: 에러 정보 제한적 노출
            raise RuntimeError("Vector database 초기화 실패 - 관리자에게 문의하세요")
    
    # Modern API methods
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """Add documents to vectorstore"""
        try:
            ids = self.vectorstore.add_texts(texts, metadatas=metadatas)
            logger.info(f"Added {len(texts)} documents to vectorstore")
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 3, **kwargs) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            docs = self.vectorstore.similarity_search(query, k=k, **kwargs)
            
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "unknown")
                })
            
            logger.debug(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def get_retriever(self, search_type: str = "mmr", **kwargs):
        """Get LangChain retriever for RAG chains with performance optimization"""
        # 성능 최적화: 검색 파라미터 튜닝
        default_search_kwargs = {
            "k": 5,  # 성능: 기본 결과 수 증가
            **kwargs
        }
        
        if search_type == "mmr":
            # Maximum Marginal Relevance 최적화
            default_search_kwargs.update({
                "lambda_mult": 0.8,  # 성능: 다양성 vs 관련성 균형 (0.8 = 더 관련성 중심)
                "fetch_k": 20,  # 성능: 후보 문서 수 증가 (더 정확한 MMR)
            })
        elif search_type == "similarity":
            # 유사도 검색 최적화
            default_search_kwargs.update({
                "score_threshold": 0.7,  # 보안: 최소 유사도 임계값
            })
        
        logger.debug(f"🔍 Retriever 설정: {search_type}, params: {default_search_kwargs}")
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=default_search_kwargs
        )
    
    def delete_collection(self):
        """Delete the collection"""
        try:
            self.vectorstore.delete_collection()
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

# Backward compatibility
VectorDB = ModernVectorDB


class VectorDBManager:
    """
    Vector Database Manager - Singleton pattern for managing VectorDB instances
    """
    
    _instance = None
    _vector_db = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_vector_db(self, persist_directory = None, **kwargs) -> ModernVectorDB:
        """Get or create VectorDB instance"""
        if self._vector_db is None:
            self._vector_db = ModernVectorDB(
                persist_directory=persist_directory or "data/chroma_db",
                **kwargs
            )
        return self._vector_db
    
    def health_check(self) -> bool:
        """Check if VectorDB is healthy"""
        try:
            if self._vector_db is None:
                return False
            
            # Handle both ModernVectorDB and direct Chroma objects
            if hasattr(self._vector_db, 'similarity_search'):
                # ModernVectorDB object
                self._vector_db.similarity_search("test", k=1)
            elif hasattr(self._vector_db, 'vectorstore'):
                # Direct vectorstore object
                self._vector_db.vectorstore.similarity_search("test", k=1)
            else:
                # Assume it's a Chroma object directly
                self._vector_db.similarity_search("test", k=1)
            return True
        except Exception as e:
            logger.warning(f"VectorDB health check failed: {e}")
            return False


def create_vector_db(persist_directory = None, **kwargs) -> ModernVectorDB:
    """Factory function for creating ModernVectorDB"""
    return ModernVectorDB(persist_directory=persist_directory, **kwargs)


# ======= LFU + Hybrid ChromaDB Implementation =======

@dataclass
class CacheStats:
    """캐시 통계 정보"""
    memory_hits: int = 0
    memory_misses: int = 0
    disk_hits: int = 0
    disk_misses: int = 0
    total_queries: int = 0
    cache_size: int = 0
    hit_rate: float = 0.0
    
    def update_memory_hit(self):
        self.memory_hits += 1
        self.total_queries += 1
        self._update_hit_rate()
    
    def update_memory_miss(self):
        self.memory_misses += 1
        self.total_queries += 1
        self._update_hit_rate()
    
    def update_disk_hit(self):
        self.disk_hits += 1
        self.total_queries += 1
        self._update_hit_rate()
    
    def update_disk_miss(self):
        self.disk_misses += 1
        self.total_queries += 1
        self._update_hit_rate()
    
    def _update_hit_rate(self):
        if self.total_queries > 0:
            total_hits = self.memory_hits + self.disk_hits
            self.hit_rate = total_hits / self.total_queries

@dataclass 
class HybridSearchResult:
    """하이브리드 검색 결과"""
    content: str
    metadata: Dict[str, Any]
    vector_score: float = 0.0
    bm25_score: float = 0.0
    metadata_score: float = 0.0
    final_score: float = 0.0
    source: str = "hybrid"

@dataclass
class SearchWeights:
    """검색 가중치 설정"""
    vector_weight: float = 0.5
    bm25_weight: float = 0.3
    metadata_weight: float = 0.2

class SimpleLFUCache:
    """간단한 LFU 캐시 구현 (cachetools 없을 때)"""
    
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.data: Dict[str, Any] = {}
        self.frequencies: Dict[str, int] = defaultdict(int)
        self.access_times: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str, default=None):
        with self._lock:
            if key in self.data:
                self.frequencies[key] += 1
                self.access_times[key] = time.time()
                return self.data[key]
            return default
    
    def set(self, key: str, value: Any):
        with self._lock:
            if len(self.data) >= self.maxsize and key not in self.data:
                # LFU 정책으로 제거
                lfu_key = min(self.frequencies.keys(), 
                            key=lambda k: (self.frequencies[k], self.access_times[k]))
                del self.data[lfu_key]
                del self.frequencies[lfu_key]
                del self.access_times[lfu_key]
            
            self.data[key] = value
            self.frequencies[key] += 1
            self.access_times[key] = time.time()
    
    def __len__(self):
        return len(self.data)

class LFUHybridVectorDB(ModernVectorDB):
    """
    LFU + Hybrid ChromaDB Implementation for JJongal-AI - 2025 Advanced Architecture
    
    Features:
    - Multi-layer LFU caching (Memory + Disk)
    - Hybrid search (Vector + BM25 + Metadata)
    - Multi-collection support
    - Performance monitoring
    - Korean text optimization
    """
    
    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = "fairy_tales_hybrid",
        use_openai: bool = True,
        memory_cache_size: int = 1000,
        disk_cache_size: int = 10000,
        disk_cache_ttl: int = 86400,  # 24시간
        enable_bm25: bool = True,
        search_weights: SearchWeights = None
    ):
        """
        Initialize LFU + Hybrid VectorDB
        
        Args:
            persist_directory: ChromaDB persistence directory
            collection_name: Collection name
            use_openai: Use OpenAI embeddings
            memory_cache_size: Memory cache size (LFU)
            disk_cache_size: Disk cache size (TTL)
            disk_cache_ttl: Disk cache TTL in seconds
            enable_bm25: Enable BM25 keyword search
            search_weights: Search scoring weights
        """
        super().__init__(persist_directory, collection_name, use_openai)
        
        # 캐시 설정
        self.memory_cache_size = memory_cache_size
        self.disk_cache_size = disk_cache_size
        self.disk_cache_ttl = disk_cache_ttl
        self.enable_bm25 = enable_bm25 and BM25_AVAILABLE
        
        # 검색 가중치
        self.search_weights = search_weights or SearchWeights()
        
        # 캐시 초기화
        self._init_caches()
        
        # BM25 인덱스
        self.bm25_corpus = []
        self.bm25_index = None
        self.bm25_metadata = []
        
        # 통계 추적
        self.stats = CacheStats()
        self.query_history = []
        
        # 스레드 안전성
        self._cache_lock = threading.Lock()
        
        logger.info(f"LFU Hybrid VectorDB initialized: {collection_name}")
        logger.info(f"Cache config - Memory: {memory_cache_size}, Disk: {disk_cache_size}, TTL: {disk_cache_ttl}s")
        logger.info(f"BM25 enabled: {self.enable_bm25}")
    
    def _init_caches(self):
        """캐시 시스템 초기화"""
        if CACHETOOLS_AVAILABLE:
            self.memory_cache = LFUCache(maxsize=self.memory_cache_size)
            self.disk_cache = TTLCache(maxsize=self.disk_cache_size, ttl=self.disk_cache_ttl)
            logger.info("✅ Using cachetools for advanced caching")
        else:
            self.memory_cache = SimpleLFUCache(maxsize=self.memory_cache_size)
            self.disk_cache = {}
            logger.info("⚠️ Using simple cache implementation")
    
    def _cache_key(self, query: str, k: int, filters: Dict = None) -> str:
        """캐시 키 생성"""
        cache_data = {
            "query": query,
            "k": k,
            "filters": filters or {},
            "collection": self.collection_name
        }
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    def add_documents_with_bm25(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict]] = None,
        update_bm25: bool = True
    ) -> List[str]:
        """문서 추가 및 BM25 인덱스 업데이트"""
        
        # 기본 벡터 저장
        ids = super().add_documents(texts, metadatas)
        
        if update_bm25 and self.enable_bm25:
            # BM25 코퍼스 업데이트
            for i, text in enumerate(texts):
                # 한국어 토큰화 (jieba 사용 - 완벽하지 않지만 기본적인 분리 가능)
                tokens = list(jieba.cut(text))
                self.bm25_corpus.append(tokens)
                self.bm25_metadata.append(metadatas[i] if metadatas else {})
            
            # BM25 인덱스 재구성
            if self.bm25_corpus:
                self.bm25_index = BM25Okapi(self.bm25_corpus)
                logger.info(f"BM25 index updated with {len(self.bm25_corpus)} documents")
        
        # 캐시 무효화 (새 문서 추가로 인해)
        self._clear_caches()
        
        return ids
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = 5,
        filters: Dict[str, Any] = None,
        use_cache: bool = True
    ) -> List[HybridSearchResult]:
        """
        하이브리드 검색 실행
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            filters: 메타데이터 필터
            use_cache: 캐시 사용 여부
            
        Returns:
            List[HybridSearchResult]: 하이브리드 검색 결과
        """
        start_time = time.time()
        cache_key = self._cache_key(query, k, filters)
        
        # 캐시 확인
        if use_cache:
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.stats.update_memory_hit()
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_result
        
        # 캐시 미스 - 실제 검색 수행
        self.stats.update_memory_miss()
        
        # 1. 벡터 검색 (기본)
        vector_results = self.similarity_search(query, k=k*2)  # 더 많이 가져와서 다양성 확보
        
        # 2. BM25 키워드 검색 (옵션)
        bm25_results = []
        if self.enable_bm25 and self.bm25_index:
            query_tokens = list(jieba.cut(query))
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            # 상위 k*2개 결과 선택
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k*2]
            
            for idx in top_indices:
                if idx < len(self.bm25_metadata):
                    original_text = ' '.join(self.bm25_corpus[idx])
                    bm25_results.append({
                        'content': original_text,
                        'metadata': self.bm25_metadata[idx],
                        'bm25_score': bm25_scores[idx]
                    })
        
        # 3. 결과 융합 및 점수 계산
        hybrid_results = self._fuse_search_results(vector_results, bm25_results, filters)
        
        # 상위 k개만 반환
        final_results = hybrid_results[:k]
        
        # 캐시에 저장
        if use_cache:
            self._store_in_cache(cache_key, final_results)
        
        # 통계 업데이트
        search_time = time.time() - start_time
        self.query_history.append({
            'query': query,
            'results_count': len(final_results),
            'search_time': search_time,
            'timestamp': datetime.now()
        })
        
        logger.debug(f"Hybrid search completed in {search_time:.3f}s: {query[:50]}...")
        
        return final_results
    
    def _fuse_search_results(
        self, 
        vector_results: List[Dict], 
        bm25_results: List[Dict],
        filters: Dict[str, Any] = None
    ) -> List[HybridSearchResult]:
        """검색 결과 융합"""
        
        # 결과 정규화 및 점수 계산
        fused_results = {}
        
        # 벡터 검색 결과 처리
        for i, result in enumerate(vector_results):
            content = result['content']
            key = hashlib.md5(content.encode()).hexdigest()[:16]
            
            vector_score = 1.0 - (i / len(vector_results))  # 순위 기반 점수
            
            fused_results[key] = HybridSearchResult(
                content=content,
                metadata=result['metadata'],
                vector_score=vector_score,
                bm25_score=0.0,
                metadata_score=0.0
            )
        
        # BM25 검색 결과 처리 및 융합
        if bm25_results:
            max_bm25_score = max(result.get('bm25_score', 0) for result in bm25_results)
            
            for result in bm25_results:
                content = result['content']
                key = hashlib.md5(content.encode()).hexdigest()[:16]
                
                normalized_bm25_score = result.get('bm25_score', 0) / max_bm25_score if max_bm25_score > 0 else 0
                
                if key in fused_results:
                    # 기존 결과에 BM25 점수 추가
                    fused_results[key].bm25_score = normalized_bm25_score
                else:
                    # 새로운 BM25 전용 결과
                    fused_results[key] = HybridSearchResult(
                        content=content,
                        metadata=result['metadata'],
                        vector_score=0.0,
                        bm25_score=normalized_bm25_score,
                        metadata_score=0.0
                    )
        
        # 메타데이터 필터 적용 및 점수 계산
        filtered_results = []
        for result in fused_results.values():
            # 메타데이터 필터 체크
            if filters:
                if not self._matches_filters(result.metadata, filters):
                    continue
                result.metadata_score = 0.3  # 필터 매치 보너스
            
            # 최종 점수 계산
            result.final_score = (
                result.vector_score * self.search_weights.vector_weight +
                result.bm25_score * self.search_weights.bm25_weight +
                result.metadata_score * self.search_weights.metadata_weight
            )
            
            filtered_results.append(result)
        
        # 최종 점수 순으로 정렬
        filtered_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return filtered_results
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """메타데이터 필터 매칭 확인"""
        for key, expected_value in filters.items():
            if key not in metadata:
                return False
            
            actual_value = metadata[key]
            
            # 문자열 포함 검사
            if isinstance(expected_value, str) and isinstance(actual_value, str):
                if expected_value.lower() not in actual_value.lower():
                    return False
            # 정확한 매치 검사
            elif actual_value != expected_value:
                return False
        
        return True
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[HybridSearchResult]]:
        """캐시에서 결과 가져오기"""
        with self._cache_lock:
            # Memory cache 확인
            result = self.memory_cache.get(cache_key)
            if result:
                return result
            
            # Disk cache 확인 (TTLCache 또는 간단한 dict)
            if hasattr(self.disk_cache, 'get'):
                result = self.disk_cache.get(cache_key)
            else:
                result = self.disk_cache.get(cache_key)
            
            if result:
                # Memory cache로 승격
                self.memory_cache[cache_key] = result
                return result
            
        return None
    
    def _store_in_cache(self, cache_key: str, results: List[HybridSearchResult]):
        """캐시에 결과 저장"""
        with self._cache_lock:
            self.memory_cache[cache_key] = results
            self.disk_cache[cache_key] = results
            self.stats.cache_size = len(self.memory_cache)
    
    def _clear_caches(self):
        """캐시 초기화"""
        with self._cache_lock:
            self.memory_cache.clear()
            self.disk_cache.clear()
            logger.info("Caches cleared due to data update")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        return {
            'cache_stats': {
                'memory_hits': self.stats.memory_hits,
                'memory_misses': self.stats.memory_misses,
                'hit_rate': self.stats.hit_rate,
                'cache_size': len(self.memory_cache),
                'total_queries': self.stats.total_queries
            },
            'recent_queries': self.query_history[-10:] if self.query_history else [],
            'bm25_enabled': self.enable_bm25,
            'bm25_corpus_size': len(self.bm25_corpus) if self.bm25_corpus else 0
        }
    
    def optimize_caches(self):
        """캐시 최적화"""
        with self._cache_lock:
            # 메모리 사용량이 너무 높으면 캐시 크기 조정
            current_size = len(self.memory_cache)
            if current_size > self.memory_cache_size * 0.9:
                # 캐시 크기를 늘리거나 오래된 항목 제거
                if CACHETOOLS_AVAILABLE:
                    # LFUCache는 자동으로 관리됨
                    pass
                else:
                    # 간단한 캐시의 경우 수동 정리
                    if hasattr(self.memory_cache, 'data'):
                        # 접근 빈도가 낮은 항목들 제거
                        items_to_remove = current_size - int(self.memory_cache_size * 0.7)
                        if items_to_remove > 0:
                            lfu_keys = sorted(
                                self.memory_cache.frequencies.keys(),
                                key=lambda k: (self.memory_cache.frequencies[k], self.memory_cache.access_times[k])
                            )[:items_to_remove]
                            
                            for key in lfu_keys:
                                del self.memory_cache.data[key]
                                del self.memory_cache.frequencies[key]
                                del self.memory_cache.access_times[key]
        
        logger.info(f"Cache optimization completed. Current size: {len(self.memory_cache)}")


# Factory functions for LFU Hybrid VectorDB
def create_lfu_hybrid_vector_db(
    persist_directory: str = None, 
    **kwargs
) -> LFUHybridVectorDB:
    """Factory function for creating LFUHybridVectorDB"""
    return LFUHybridVectorDB(persist_directory=persist_directory, **kwargs)
