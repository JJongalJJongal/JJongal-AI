"""
꼬꼬북 프로젝트 벡터 데이터베이스 패키지

이 패키지는 이야기 데이터를 벡터화하고 효율적으로 검색하기 위한 기능을 제공합니다.
세 가지 유형의 벡터 데이터베이스를 지원합니다:
- main: 일반 검색용 (기본)
- detailed: 세부 정보 검색용
- summary: 요약 및 주제 검색용
"""

from .core import VectorDB
from .importers import process_story_data
from .query import get_similar_stories
from shared.utils.file_utils import (
    ensure_directory,
    load_json,
    save_json
)

# 순환 import 방지를 위해 함수를 지연 import
def get_db_type_path(base_directory: str, db_type: str) -> str:
    """
    Constructs the path for a specific DB type and ensures the directory exists.
    """
    from pathlib import Path
    db_path = Path(base_directory) / db_type
    ensure_directory(str(db_path))
    return str(db_path)

def check_collection_info(vector_db_instance, collection_name: str):
    """컬렉션 정보 확인 함수를 지연 import로 호출"""
    from shared.utils.vector_db_utils import check_collection_info as _check_collection_info
    return _check_collection_info(vector_db_instance, collection_name)

__all__ = [
    'VectorDB',
    'process_story_data',
    'get_similar_stories',
    'ensure_directory',
    'get_db_type_path',
    'load_json',
    'save_json',
    'check_collection_info'
] 