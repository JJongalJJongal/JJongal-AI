"""
벡터 데이터베이스 쿼리 관련 기능 모듈

주요 기능:
- 벡터 데이터베이스에서 유사 스토리 검색
- 쿼리 결과 가공 및 형식화
- 다양한 검색 필터 지원
"""

# import logging # get_module_logger 사용
from typing import Dict, List, Any, Optional

from .core import VectorDB # 상대 경로 유지 (같은 패키지 내)
from shared.utils.logging_utils import get_module_logger

# 로깅 설정
logger = get_module_logger(__name__)

def query_vector_db(
    vector_db: VectorDB, 
    query_text: str, 
    collection_name: str = "fairy_tales",
    n_results: int = 8,
    metadata_filter: Optional[Dict[str, Any]] = None,
    # age_group, theme, doc_type 등은 metadata_filter로 통합됨 (get_similar_stories 참고)
) -> Dict[str, Any]: # VectorDB.query의 반환 타입과 일치 또는 format_query_results를 거친 타입
    """
    벡터 DB에서 쿼리 실행 (내부 사용 함수 또는 get_similar_stories의 일부로 통합 고려)
    이 함수는 get_similar_stories에 의해 거의 대체되었으므로, 직접 사용은 줄어들 것임.
    
    Args:
        vector_db: VectorDB 인스턴스
        query_text: 쿼리 텍스트
        collection_name: 컬렉션 이름
        n_results: 반환할 결과 수
        metadata_filter: ChromaDB 'where' 절에 직접 사용될 필터
        
    Returns:
        Dict: ChromaDB 검색 결과 (또는 VectorDB.query의 결과)
    """
    logger.info(f"컬렉션 '{collection_name}'에서 '{query_text[:50]}...' 쿼리 실행. 필터: {metadata_filter}")
    try:
        results = vector_db.query(
            collection_name=collection_name,
            query_texts=[query_text],
            n_results=n_results,
            where_filter=metadata_filter
        )
        return results
    except Exception as e:
        logger.error(f"벡터 DB 쿼리 중 오류 발생: {e}", exc_info=True)
        return {} # 오류 시 빈 결과 반환

def format_query_results(results: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    ChromaDB 쿼리 결과를 가공하여 반환 (기존 기능 유지, 필요시 수정)
    
    Args:
        results: ChromaDB 쿼리 결과 (또는 VectorDB.query의 결과)
        
    Returns:
        List[Dict]: 가공된 결과 리스트 (문서 내용, 메타데이터, 유사도 등 포함 가능)
    """
    formatted = []
    if not results or not results.get('ids') or not results.get('ids')[0]:
        logger.info("쿼리 결과가 비어있거나 형식이 올바르지 않아 포맷팅할 수 없습니다.")
        return formatted

    ids = results.get('ids')[0]
    documents = results.get('documents', [[]])[0] # 문서 내용이 없을 수 있음
    metadatas = results.get('metadatas', [[]])[0] # 메타데이터가 없을 수 있음
    distances = results.get('distances', [[]])[0] # 유사도(거리)가 없을 수 있음
    
    for i, doc_id in enumerate(ids):
        entry = {
            "id": doc_id,
            "document": documents[i] if documents and i < len(documents) else None,
            "metadata": metadatas[i] if metadatas and i < len(metadatas) else {},
            "distance": distances[i] if distances and i < len(distances) else None
        }
        formatted.append(entry)
    
    # logger.debug(f"포맷된 쿼리 결과: {formatted}") # 너무 길 수 있으므로 주의
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