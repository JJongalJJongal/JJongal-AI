"""
스토리 데이터를 처리하고 벡터 데이터베이스로 가져오는 기능 모듈

주요 기능:
- 이야기 데이터 처리 및 정규화
- 다양한 소스에서 이야기 데이터 가져오기
- 벡터 데이터베이스에 이야기 데이터 저장
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .core import VectorDB

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def process_story_data(story_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    스토리 데이터 전처리 및 정규화
    
    Args:
        story_data: 원본 스토리 데이터
        
    Returns:
        Dict: 처리된 스토리 데이터
    """
    processed_data = {
        "title": story_data.get("title", "제목 없음"),
        "summary": story_data.get("summary", story_data.get("plot_summary", "")),
        "characters": story_data.get("characters", []),
        "setting": story_data.get("setting", ""),
        "theme": story_data.get("theme", ""),
        "age_group": story_data.get("age_group", story_data.get("target_age", 5)),
        "educational_value": story_data.get("educational_value", "")
    }
    
    # 챕터 처리
    if "chapters" in story_data and isinstance(story_data["chapters"], list):
        processed_data["chapters"] = []
        for chapter in story_data["chapters"]:
            processed_chapter = {
                "title": chapter.get("title", ""),
                "narration": chapter.get("narration", ""),
                "image": chapter.get("image", "")
            }
            processed_data["chapters"].append(processed_chapter)
    
    return processed_data

def generate_story_documents(story_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    스토리 데이터를 벡터 DB 문서로 변환
    
    Args:
        story_data: 스토리 데이터
        
    Returns:
        List[Dict]: 문서 텍스트 및 메타데이터 목록
    """
    documents = []
    
    # 기본 문서: 스토리 요약
    base_doc = {
        "text": f"제목: {story_data['title']}\n요약: {story_data['summary']}\n"
                f"주제: {story_data['theme']}\n교육적 가치: {story_data['educational_value']}",
        "metadata": {
            "title": story_data["title"],
            "type": "summary",
            "age_group": story_data["age_group"],
            "theme": story_data["theme"]
        }
    }
    documents.append(base_doc)
    
    # 챕터별 문서 (있는 경우)
    if "chapters" in story_data:
        for i, chapter in enumerate(story_data["chapters"]):
            chapter_doc = {
                "text": f"챕터 {i+1}: {chapter['title']}\n{chapter['narration']}",
                "metadata": {
                    "title": story_data["title"],
                    "type": "chapter",
                    "chapter_number": i + 1,
                    "chapter_title": chapter["title"],
                    "age_group": story_data["age_group"],
                    "theme": story_data["theme"]
                }
            }
            documents.append(chapter_doc)
    
    return documents

def import_stories(stories_dir: str, vector_db: VectorDB, collection_name: str = "fairy_tales"):
    """
    디렉토리에서 스토리 파일들을 읽어 벡터 DB로 가져오기
    
    Args:
        stories_dir: 스토리 파일이 있는 디렉토리 경로
        vector_db: VectorDB 인스턴스
        collection_name: 저장할 컬렉션 이름
    """
    stories_path = Path(stories_dir)
    if not stories_path.exists() or not stories_path.is_dir():
        logger.error(f"디렉토리가 존재하지 않습니다: {stories_dir}")
        return
    
    # 컬렉션 생성 또는 가져오기
    try:
        collection = vector_db.get_collection(collection_name)
        logger.info(f"기존 컬렉션 '{collection_name}' 사용")
    except Exception:
        collection = vector_db.create_collection(collection_name)
        logger.info(f"새 컬렉션 '{collection_name}' 생성")
    
    # 스토리 파일 가져오기
    count = 0
    json_files = list(stories_path.glob("*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                story_data = json.load(f)
            
            # 스토리 데이터 처리
            processed_data = process_story_data(story_data)
            
            # 문서 생성
            documents = generate_story_documents(processed_data)
            
            # 문서 추가
            for doc in documents:
                vector_db.add_documents(
                    documents=[doc["text"]],
                    metadatas=[doc["metadata"]],
                    ids=[f"{processed_data['title']}_{doc['metadata']['type']}_{count}"]
                )
                count += 1
                
            logger.info(f"스토리 가져오기 완료: {processed_data['title']} ({len(documents)} 문서)")
            
        except Exception as e:
            logger.error(f"파일 처리 중 오류 발생: {file_path} - {str(e)}")
    
    logger.info(f"총 {count}개 문서를 {collection_name} 컬렉션에 추가했습니다.") 