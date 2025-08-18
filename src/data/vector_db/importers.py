"""
스토리 데이터를 처리하고 벡터 데이터베이스용 표준화된 출력을 생성하는 기능 모듈입니다.

주요 기능:
- 이야기 데이터 처리, 정규화 및 핵심 필드 추출 (age_min, age_max 포함)
"""

import os
import json
# import logging # get_module_logger 사용
import re # 정규표현식 추가
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.shared.utils.logging import get_module_logger

# from .core import VectorDB # VectorDB는 이 모듈에서 직접 사용하지 않음

# 로깅 설정
logger = get_module_logger(__name__)

def parse_age_range(age_input: Any) -> Dict[str, Optional[int]]:
    """Helper function to parse various age inputs into age_min and age_max."""
    age_min, age_max = None, None
    if isinstance(age_input, str):
        match = re.match(r'(\d+)\s*-\s*(\d+)\s*세?', age_input)
        if match:
            age_min = int(match.group(1))
            age_max = int(match.group(2))
        elif age_input.isdigit():
            age_min = int(age_input)
            age_max = int(age_input)
    elif isinstance(age_input, int):
        age_min = age_input
        age_max = age_input
    return {"age_min": age_min, "age_max": age_max}

def create_age_group_string(age_min: Optional[int], age_max: Optional[int]) -> Optional[str]:
    """
    age_min과 age_max를 기반으로 표준화된 'age_group' 문자열을 생성합니다.
    예: '3-5세', '6세', '7세 이상'
    """
    if age_min is None:
        return None
    
    # '6세 이상'과 같은 특수 케이스 처리 (예시)
    if age_max is not None and age_max >= 99: # 99나 특정 큰 수를 '이상'으로 가정
        return f"{age_min}세 이상"
    
    if age_max is None or age_min == age_max:
        return f"{age_min}세"
    
    if age_min > age_max: # 혹시 모를 데이터 오류 방지
        return f"{age_min}세"

    return f"{age_min}-{age_max}세"

def extract_age_group_from_tags(tags: List[str]) -> Optional[str]:
    """
    태그 리스트에서 'X-Y세' 또는 'X세' 형식의 나이 그룹 문자열을 추출합니다.
    """
    if not isinstance(tags, list):
        return None
        
    # '7-9세', '5세', '6세 이상' 등 다양한 형식을 포괄하는 정규표현식
    age_pattern = re.compile(r'(\d+-\d+\s*세|\d+\s*세|\d+\s*세\s*이상)')
    for tag in tags:
        match = age_pattern.search(tag)
        if match:
            # 일치하는 부분을 공백 제거 후 반환
            return match.group(0).strip()
    return None

def process_story_data(story_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    스토리 데이터 전처리 및 정규화. 
    populate_vector_db.py에서 필요한 모든 필드를 추출하여 반환합니다.
    
    Args:
        story_data: 원본 스토리 데이터 (JSON 로드된 딕셔너리)
        
    Returns:
        Dict: 처리 및 표준화된 스토리 데이터. 
              포함 필드 예시: title, summary, content, characters (list of names),
              setting, theme, educational_value, keywords (list), tags (list), 
              age_group (str, from tags).
    """
    processed_data = {}

    # 기본 정보 추출
    processed_data["title"] = story_data.get("title", "제목 없음")
    processed_data["story_id"] = story_data.get("story_id", processed_data["title"].replace(" ", "_"))
    processed_data["summary"] = story_data.get("summary", story_data.get("plot_summary", ""))
    processed_data["setting"] = story_data.get("setting", "")
    processed_data["educational_value"] = story_data.get("educational_value", "")
    
    # 캐릭터 정보 (이름 리스트로 단순화)
    characters_raw = story_data.get("characters", [])
    if isinstance(characters_raw, list):
        processed_data["characters"] = [char.get("name", "") for char in characters_raw if isinstance(char, dict) and char.get("name")]
    else:
        processed_data["characters"] = []

    # 키워드
    processed_data["keywords"] = story_data.get("keywords", []) 
    if isinstance(processed_data["keywords"], str): # 문자열이면 리스트로 변환 시도
        processed_data["keywords"] = [k.strip() for k in processed_data["keywords"].split(',') if k.strip()]
    
    # 태그 처리 및 age_group 추출
    tags_input = story_data.get("tags", [])
    tags_list = []
    if isinstance(tags_input, str):
        tags_list = [t.strip() for t in tags_input.split(',') if t.strip()]
    elif isinstance(tags_input, list):
        tags_list = tags_input
    
    processed_data["tags"] = tags_list
    processed_data["age_group"] = extract_age_group_from_tags(tags_list)
    
    # 콘텐츠 집계 (챕터 내용 포함)
    main_content = story_data.get("content", "")
    if not main_content and "chapters" in story_data and isinstance(story_data["chapters"], list):
        chapter_narrations = []
        for chapter in story_data["chapters"]:
            if isinstance(chapter, dict) and chapter.get("narration"):
                chapter_narrations.append(chapter.get("narration"))
        if chapter_narrations:
            main_content = "\n\n".join(chapter_narrations)
    processed_data["content"] = main_content
    
    # 챕터 원본 구조 (필요시 populate_vector_db.py 에서 활용 가능하도록 유지)
    # if "chapters" in story_data and isinstance(story_data["chapters"], list):
    #     processed_data["chapters_raw"] = story_data["chapters"]

    # 누락된 주요 필드에 대한 로깅 (디버깅 목적)
    # for key in ["title", "summary", "content", "age_min", "age_max"]:
    #     if not processed_data.get(key) and processed_data.get(key) != 0 : # 0은 유효한 age_min일 수 있음
    #         logger.debug(f"Story ID {processed_data['story_id']}: Processed data missing key '{key}'. Original story_data: {story_data.get(key, 'N/A')}")

    return processed_data

# generate_story_documents 함수는 populate_vector_db.py에서 직접 처리하므로 제거됨.
# import_stories 함수는 populate_vector_db.py가 CLI 역할을 하므로 제거됨. 