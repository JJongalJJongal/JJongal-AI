#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

"""
쫑알쫑알 프로젝트 벡터 데이터베이스 컬렉션 관리 도구

이 모듈은 세 가지 벡터 데이터베이스를 관리합니다:
1. main DB: 일반적인 검색 목적으로 사용, 전체 스토리와 메타데이터 포함
2. detailed DB: 스토리 전개, 캐릭터 설명, 배경 설정 등 세부 내용 검색에 최적화
3. summary DB: 동화의 주제, 교훈, 키워드, 짧은 요약 등 핵심 정보 검색에 최적화

각 DB는 고유한 임베딩 특성을 가지며, 목적에 맞게 선택하여 사용하세요.
"""

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가 
current_dir_path = Path(__file__).parent # 현재 파일의 디렉토리 (Path 객체)
project_root = current_dir_path.parent.parent.parent # CCB_AI
sys.path.append(str(project_root)) # 프로젝트 루트 디렉토리를 파이썬 경로에 추가

from chatbot.data.vector_db.core import VectorDB
from shared.utils.vector_db_utils import check_collection_info, get_db_type_path

from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__) # 로거 인스턴스 생성

def list_db_collections(vector_db_instance: VectorDB) -> List[Dict[str, Any]]:
    """
    ChromaDB의 컬렉션 목록과 각 컬렉션의 아이템 수를 확인합니다.
    Args:
        vector_db_instance: 초기화된 VectorDB 인스턴스
    Returns:
        List[Dict[str, Any]]: 컬렉션 정보 리스트 (이름, 카운트 등)
    """
    try:
        collections = vector_db_instance.client.list_collections()
        if not collections:
            logger.info("데이터베이스에 컬렉션이 없습니다.")
            return []

        logger.info("\n컬렉션 목록:")
        collection_details = []
        for collection_obj in collections:
            name = collection_obj.name
            count = collection_obj.count()
            # metadata = collection_obj.metadata # 필요시 메타데이터도 가져올 수 있음
            logger.info(f"  - {name} (항목 수: {count})")
            collection_details.append({"name": name, "count": count})
        return collection_details
    except Exception as e:
        logger.error(f"컬렉션 목록 확인 중 오류 발생: {str(e)}")
        return []

def create_db_collection(vector_db_instance: VectorDB, collection_name="fairy_tales") -> bool:
    """
    ChromaDB에 새 컬렉션을 생성합니다.
    Args:
        vector_db_instance: VectorDB 인스턴스
        collection_name: 생성할 컬렉션 이름
    """
    try:
        vector_db_instance.create_collection(
            name=collection_name,
            metadata={"description": "쫑알쫑알 프로젝트 동화 데이터"} # VectorDB.create_collection의 metadata 인자 형식에 맞춤
        )
        logger.info(f"컬렉션 '{collection_name}'이 성공적으로 생성되었습니다.")
        return True
    except Exception as e:
        # 이미 컬렉션이 존재할 경우 chromadb.errors.DuplicateIDError 등이 발생할 수 있음
        logger.warning(f"컬렉션 '{collection_name}' 생성 중 오류 발생 (또는 이미 존재): {str(e)}")
        return False

def delete_db_collection(vector_db_instance: VectorDB, collection_name: str) -> bool:
    """
    ChromaDB에서 지정된 컬렉션을 삭제합니다.
    Args:
        vector_db_instance: VectorDB 인스턴스
        collection_name: 삭제할 컬렉션 이름
    """
    try:
        vector_db_instance.client.delete_collection(name=collection_name)
        logger.info(f"컬렉션 '{collection_name}'이 성공적으로 삭제되었습니다.")
        return True
    except Exception as e:
        logger.error(f"컬렉션 '{collection_name}' 삭제 중 오류 발생: {str(e)}")
        return False

def get_db_collection_info_cli(vector_db_instance: VectorDB, collection_name: str):
    """
    지정된 컬렉션의 상세 정보를 출력합니다. (utils.check_collection_info 사용)
    Args:
        vector_db_instance: VectorDB 인스턴스
        collection_name: 정보를 확인할 컬렉션 이름
    """
    info = check_collection_info(vector_db_instance, collection_name) # 수정된 utils 함수 호출
    if "error" in info:
        logger.error(f"컬렉션 '{info.get('name', collection_name)}' 정보 확인 중 오류: {info['error']}")
    else:
        logger.info(f"\n컬렉션 '{info.get('name')}' 정보:")
        for key, value in info.items():
            if key != 'name': # 이름은 이미 위에서 출력
                logger.info(f"  - {key.replace('_', ' ').capitalize()}: {value}")

def main():
    parser = argparse.ArgumentParser(description='쫑알쫑알 ChromaDB 컬렉션 관리 도구')
    parser.add_argument('action', type=str, choices=['list', 'create', 'delete', 'info'],
                      help='수행할 작업: list (컬렉션 목록), create (컬렉션 생성), delete (컬렉션 삭제), info (컬렉션 정보)')
    parser.add_argument('--db-dir-type', type=str, choices=['main', 'detailed', 'summary'], default='main',
                      help='벡터 DB 저장 위치 유형 (main, detailed, summary)')
    parser.add_argument('--collection', type=str, default='fairy_tales',
                      help='작업 대상 컬렉션 이름')
    
    args = parser.parse_args()
    
    vector_db_persist_path = get_db_type_path(base_directory=str(current_dir_path), db_type=args.db_dir_type)
    
    logger.info(f"대상 벡터 DB 디렉토리: {vector_db_persist_path}")
    
    db_instance = VectorDB(persist_directory=str(vector_db_persist_path)) 
    
    if args.action == 'list':
        list_db_collections(db_instance)
    elif args.action == 'create':
        create_db_collection(db_instance, args.collection)
    elif args.action == 'delete':
        confirm = input(f"정말로 '{args.collection}' 컬렉션을 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다. (y/n): ")
        if confirm.lower() == 'y':
            delete_db_collection(db_instance, args.collection)
    elif args.action == 'info':
        get_db_collection_info_cli(db_instance, args.collection) 
    else:
        logger.error(f"잘못된 작업 유형: {args.action}")
        parser.print_help()

if __name__ == "__main__":
    main()