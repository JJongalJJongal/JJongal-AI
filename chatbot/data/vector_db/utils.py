"""
벡터 데이터베이스 관련 유틸리티 기능 모듈

주요 기능:
- 경로 및 디렉토리 관리
- 데이터 변환 및 정규화
- 디버깅 및 진단 도구
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def ensure_directory(path: str) -> Path:
    """
    디렉토리가 존재하지 않으면 생성
    
    Args:
        path: 경로 문자열
        
    Returns:
        Path: 생성된 디렉토리 경로
    """
    directory = Path(path)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"디렉토리 생성 완료: {directory}")
    return directory

def get_db_type_path(base_directory: str, db_type: str = "main") -> Path:
    """
    지정한 DB 유형의 경로 반환
    
    Args:
        base_directory: 기본 디렉토리 경로
        db_type: DB 유형 ("main", "detailed", "summary" 중 하나)
        
    Returns:
        Path: DB 유형 경로
    """
    allowed_types = ["main", "detailed", "summary"]
    if db_type not in allowed_types:
        logger.warning(f"유효하지 않은 DB 유형: {db_type}, main으로 대체합니다.")
        db_type = "main"
    
    db_path = Path(base_directory) / db_type
    ensure_directory(db_path)
    return db_path

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    JSON 파일 로드
    
    Args:
        file_path: 파일 경로
        
    Returns:
        Dict: 로드된 JSON 데이터
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"JSON 파일 로드 실패: {file_path} - {str(e)}")
        return {}

def save_json_file(data: Dict[str, Any], file_path: str) -> bool:
    """
    JSON 파일 저장
    
    Args:
        data: 저장할 데이터
        file_path: 저장 경로
        
    Returns:
        bool: 성공 여부
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 파일 저장 완료: {file_path}")
        return True
    except Exception as e:
        logger.error(f"JSON 파일 저장 실패: {file_path} - {str(e)}")
        return False

def check_collection_info(collection) -> Dict[str, Any]:
    """
    컬렉션 정보 확인
    
    Args:
        collection: ChromaDB 컬렉션 객체
        
    Returns:
        Dict: 컬렉션 정보
    """
    try:
        count = collection.count()
        name = collection.name
        metadata = collection.get().get("metadatas", [])
        
        # 메타데이터 분석
        age_groups = set()
        themes = set()
        types = set()
        
        for meta in metadata:
            if meta and isinstance(meta, dict):
                if "age_group" in meta:
                    age_groups.add(meta["age_group"])
                if "theme" in meta and meta["theme"]:
                    themes.add(meta["theme"])
                if "type" in meta:
                    types.add(meta["type"])
        
        return {
            "name": name,
            "count": count,
            "age_groups": sorted(list(age_groups)),
            "themes": sorted(list(themes)),
            "types": sorted(list(types))
        }
    except Exception as e:
        logger.error(f"컬렉션 정보 확인 실패: {str(e)}")
        return {"error": str(e)} 