"""
벡터 데이터베이스 쿼리 관련 기능 모듈

주요 기능:
- 벡터 데이터베이스에서 유사 스토리 검색
- 쿼리 결과 가공 및 형식화
- 다양한 검색 필터 지원
"""

# import logging # get_module_logger 사용
from typing import Dict, List, Any, Optional

from .core import ModernVectorDB as VectorDB # Modern VectorDB 사용
from src.shared.utils.logging import get_module_logger

# 로깅 설정
logger = get_module_logger(__name__)

def query_vector_db(
    vector_db: VectorDB, 
    query_text: str, 
    collection_name: str = "fairy_tales",
    n_results: int = 8,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Modern VectorDB 쿼리 실행
    
    Args:
        vector_db: ModernVectorDB 인스턴스
        query_text: 쿼리 텍스트
        collection_name: 컬렉션 이름 (무시됨 - 이미 설정됨)
        n_results: 반환할 결과 수
        metadata_filter: 메타데이터 필터
        
    Returns:
        List[Dict]: 검색 결과 목록
    """
    logger.info(f"Modern VectorDB에서 '{query_text[:50]}...' 쿼리 실행. 필터: {metadata_filter}")
    try:
        # Modern VectorDB의 similarity_search 사용
        results = vector_db.similarity_search(
            query=query_text,
            k=n_results,
            filter=metadata_filter  # Modern VectorDB는 'filter' 파라미터 사용
        )
        return results
    except Exception as e:
        logger.error(f"Modern VectorDB 쿼리 중 오류 발생: {e}", exc_info=True)
        return []  # 오류 시 빈 결과 반환

def format_query_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Modern VectorDB 검색 결과를 표준 형식으로 변환
    
    Args:
        results: Modern VectorDB의 similarity_search 결과
        
    Returns:
        List[Dict]: 표준화된 결과 리스트
    """
    if not results:
        logger.info("쿼리 결과가 비어있습니다.")
        return []

    formatted = []
    for i, result in enumerate(results):
        entry = {
            "id": f"doc_{i}",  # Modern VectorDB는 별도 ID가 없으므로 생성
            "document": result.get("content", ""),
            "metadata": result.get("metadata", {}),
            "distance": None,  # Modern VectorDB는 거리 점수를 따로 제공하지 않음
            "source": result.get("source", "unknown")
        }
        formatted.append(entry)
    
    logger.debug(f"포맷된 쿼리 결과 {len(formatted)}개")
    return formatted

def get_similar_stories(
    vector_db: VectorDB, 
    query_text: str, 
    n_results: int = 3,
    collection_name: str = "fairy_tales",
    metadata_filter: Optional[Dict[str, Any]] = None, 
    doc_type: Optional[str] = "summary" 
) -> List[Dict[str, Any]]:
    """
    유사한 스토리 검색 (TextGenerator 등 외부 모듈에서 사용하는 주 인터페이스)
    
    Args:
        vector_db: VectorDB 인스턴스
        query_text: 쿼리 텍스트
        n_results: 반환할 결과 수
        collection_name: 컬렉션 이름
        metadata_filter: ChromaDB 'where' 절에 직접 사용될 추가 필터.
                         TextGenerator에서 연령대 필터 등을 전달할 때 사용.
        doc_type: 문서 유형 필터 (기본값: "summary"). 이 필터는 metadata_filter와 결합됨.
        
    Returns:
        List[Dict]: 가공된 유사한 스토리 목록 (format_query_results를 거친 형태)
    """
    
    # doc_type 필터를 metadata_filter에 통합
    final_filter = metadata_filter.copy() if metadata_filter else {}
    if doc_type:
        if "$and" in final_filter: # 이미 $and 조건이 있다면 type 조건을 추가
            # type 조건이 이미 있는지 확인 (중복 방지)
            type_condition_exists = False
            for condition in final_filter["$and"]:
                if isinstance(condition, dict) and "type" in condition:
                    type_condition_exists = True
                    break
            if not type_condition_exists:
                final_filter["$and"].append({"type": doc_type})
        elif final_filter: # 다른 조건이 하나라도 있다면 $and로 묶음
            # 기존 필터가 type 조건인지 확인
            if not (len(final_filter) == 1 and "type" in final_filter):
                 existing_filters = [{k: v} for k, v in final_filter.items()]
                 final_filter = {"$and": existing_filters + [{"type": doc_type}]}
            elif final_filter.get("type") != doc_type: # type 조건만 있는데 값이 다른 경우 (이 경우는 거의 없어야 함)
                 final_filter = {"$and": [{"type": final_filter.get("type")}, {"type": doc_type}] }
            # else: type 조건만 있고 값도 같다면 그대로 사용 (doc_type 추가 불필요)
        else: # 필터가 아예 없었다면 type 조건만 추가
            final_filter = {"type": doc_type}

    logger.info(f"유사 스토리 검색: '{query_text[:50]}...', 컬렉션: {collection_name}, 최종 필터: {final_filter}")
    
    raw_results = query_vector_db(
        vector_db=vector_db,
        query_text=query_text,
        collection_name=collection_name,
        n_results=n_results,
        metadata_filter=final_filter
    )
    
    return format_query_results(raw_results) 