"""
벡터 데이터베이스 쿼리 관련 기능 모듈

주요 기능:
- 벡터 데이터베이스에서 유사 스토리 검색
- 쿼리 결과 가공 및 형식화
- 다양한 검색 필터 지원
"""

import logging
from typing import Dict, List, Any, Optional

from .core import VectorDB

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def query_vector_db(
    vector_db: VectorDB, 
    query_text: str, 
    collection_name: str = "fairy_tales",
    n_results: int = 8,
    age_group: Optional[int] = None,
    theme: Optional[str] = None,
    doc_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    벡터 DB에서 쿼리 실행
    
    Args:
        vector_db: VectorDB 인스턴스
        query_text: 쿼리 텍스트
        collection_name: 컬렉션 이름
        n_results: 반환할 결과 수
        age_group: 연령대 필터 (선택 사항)
        theme: 주제 필터 (선택 사항)
        doc_type: 문서 유형 필터 (선택 사항)
        
    Returns:
        Dict: 쿼리 결과
    """
    try:
        # 컬렉션 가져오기
        try:
            collection = vector_db.get_collection(collection_name)
        except Exception as e:
            logger.error(f"컬렉션 '{collection_name}' 가져오기 실패: {str(e)}")
            return {"error": f"컬렉션 '{collection_name}' 가져오기 실패: {str(e)}"}
        
        # 필터 설정
        filter_conditions = []
        if age_group is not None:
            filter_conditions.append({"age_group": {"$eq": age_group}})
        if theme:
            filter_conditions.append({"theme": {"$contains": theme}})
        if doc_type:
            filter_conditions.append({"type": {"$eq": doc_type}})

        where_filter = None
        if len(filter_conditions) == 1:
            where_filter = filter_conditions[0]
        elif len(filter_conditions) > 1:
            where_filter = {"$and": filter_conditions}
            
        # 쿼리 실행
        results = vector_db.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        
        logger.info(f"쿼리 실행 완료: '{query_text}' (결과: {len(results.get('ids', [[]]))}개)")
        return results
        
    except Exception as e:
        logger.error(f"쿼리 실행 실패: {str(e)}")
        return {"error": str(e)}

def format_query_results(query_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    쿼리 결과를 가공하여 사용하기 쉬운 형태로 변환
    
    Args:
        query_results: VectorDB의 query 메서드 반환 결과
        
    Returns:
        List[Dict]: 포맷된 결과 목록
    """
    if "error" in query_results:
        return [{"error": query_results["error"]}]
    
    if not query_results or "ids" not in query_results or not query_results["ids"]:
        return []
    
    formatted_results = []
    
    # 첫 번째 쿼리 결과만 처리 (기본적으로 query_texts에 하나만 전달함)
    ids = query_results["ids"][0]
    documents = query_results["documents"][0]
    metadatas = query_results["metadatas"][0]
    distances = query_results["distances"][0]
    
    for i in range(len(ids)):
        result = {
            "id": ids[i],
            "text": documents[i],
            "metadata": metadatas[i],
            "distance": distances[i]
        }
        formatted_results.append(result)
    
    # 거리(유사도)로 정렬
    formatted_results.sort(key=lambda x: x["distance"])
    
    return formatted_results

def get_similar_stories(
    vector_db: VectorDB, 
    query_text: str, 
    age_group: Optional[int] = None, 
    n_results: int = 3,
    collection_name: str = "fairy_tales"
) -> List[Dict[str, Any]]:
    """
    유사한 스토리 검색 (높은 수준의 인터페이스)
    
    Args:
        vector_db: VectorDB 인스턴스
        query_text: 쿼리 텍스트
        age_group: 연령대
        n_results: 반환할 결과 수
        collection_name: 컬렉션 이름
        
    Returns:
        List[Dict]: 유사한 스토리 목록
    """
    # 요약 문서 타입으로 필터링하여 쿼리
    results = query_vector_db(
        vector_db=vector_db,
        query_text=query_text,
        collection_name=collection_name,
        n_results=n_results,
        age_group=age_group,
        doc_type="summary"
    )
    
    formatted_results = format_query_results(results)
    
    # 스토리 형식으로 변환
    stories = []
    for result in formatted_results:
        if "error" in result:
            continue
            
        metadata = result["metadata"]
        stories.append({
            "title": metadata.get("title", "제목 없음"),
            "theme": metadata.get("theme", ""),
            "age_group": metadata.get("age_group", 0),
            "summary": result["text"].split("요약: ")[-1].split("\n")[0] if "요약: " in result["text"] else "",
            "similarity": 1.0 - result["distance"]  # 거리를 유사도로 변환
        })
    
    return stories 