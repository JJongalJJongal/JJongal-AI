import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

"""
꼬꼬북 프로젝트 벡터 데이터베이스 데이터 채우기 도구

이 모듈은 전처리된 스토리 데이터를 세 가지 주요 벡터 데이터베이스 유형에 추가합니다:
1. main DB: 일반적인 검색 목적으로 사용, 전체 스토리와 메타데이터 포함
2. detailed DB: 스토리 전개, 캐릭터 설명, 배경 설정 등 세부 내용 검색에 최적화
3. summary DB: 동화의 주제, 교훈, 키워드, 짧은 요약 등 핵심 정보 검색에 최적화

또한, 모든 스토리를 기본 설정으로 한 번에 임포트하는 기능도 제공합니다.
"""

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
current_dir_path = Path(__file__).parent # 현재 파일의 디렉토리 (Path 객체)
project_root = current_dir_path.parent.parent.parent # CCB_AI
sys.path.append(str(project_root))

# 환경 변수 로드
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path=dotenv_path)

from chatbot.data.vector_db.core import VectorDB # .core -> chatbot.data.vector_db.core
from chatbot.data.vector_db.importers import process_story_data # .importers -> chatbot.data.vector_db.importers
from shared.utils.vector_db_utils import get_db_type_path # 공유 유틸리티 사용

from shared.utils.logging_utils import get_module_logger # 공유 로거
from shared.utils.file_utils import load_json as shared_load_json # 공유 JSON 로더

logger = get_module_logger(__name__)

def filter_stories(stories_data: List[Dict[str, Any]], filter_criteria: Dict[str, Any], verbose: bool = False) -> List[Dict[str, Any]]:
    """
    필터링 기준에 맞는 스토리 목록을 반환 (기존 import_stories_with_vector_db.py의 로직)
    """
    filtered_stories = []
    if not filter_criteria: # 필터가 없으면 모든 스토리 반환 (import-all 시 활용)
        if verbose:
            logger.info(f"필터 기준 없음. 전체 {len(stories_data)}개 스토리 반환.")
        return stories_data

    for story in stories_data:
        matches_all_criteria = True
        for key, value in filter_criteria.items():
            story_value = story.get(key)
            # age 필터링은 현재 로직에서 제거됨. 필요 시 재구현.
            if isinstance(story_value, str) and isinstance(value, str):
                if value.lower() not in story_value.lower():
                    matches_all_criteria = False
                    break
            elif story_value != value:
                matches_all_criteria = False
                break
            elif story_value is None: # 스토리에 해당 필드가 없는 경우
                matches_all_criteria = False
                break
        
        if matches_all_criteria:
            filtered_stories.append(story)
    
    if verbose:
        logger.info(f"필터링 결과: 전체 {len(stories_data)}개 중 {len(filtered_stories)}개 스토리가 기준에 부합")
    return filtered_stories

def main():
    parser = argparse.ArgumentParser(description='전처리된 동화 데이터를 ChromaDB의 특정 유형 DB에 추가합니다.')
    parser.add_argument('--db-dir-type', type=str, choices=['main', 'detailed', 'summary'], default='main',
                       help='벡터 DB 저장 위치 유형 (main, detailed, summary). --import-all 시 main으로 고정.')
    parser.add_argument('--collection', type=str, default='fairy_tales',
                       help='ChromaDB 컬렉션 이름. --import-all 시 fairy_tales로 고정.')
    parser.add_argument('--filter-age', type=int, help='연령대로 필터링 (예: 5는 5-6세 관련 태그 검색)')
    parser.add_argument('--import-all', action='store_true', 
                       help='모든 스토리를 기본 설정(main DB, fairy_tales 컬렉션)으로 필터 없이 임포트합니다.')
    parser.add_argument('--verbose', action='store_true', help='상세 정보 출력')
    
    args = parser.parse_args()

    db_type_to_use = args.db_dir_type
    collection_to_use = args.collection
    filter_criteria = {}

    if args.import_all:
        logger.info("'--import-all' 옵션 사용: 모든 스토리를 'main' DB의 'fairy_tales' 컬렉션에 임포트합니다.")
        db_type_to_use = 'main'
        collection_to_use = 'fairy_tales'
        # 필터 기준은 비워둠 (모든 데이터 임포트)
    else:
        if args.filter_age:
            filter_criteria['age'] = args.filter_age # filter_stories 함수와 키 일치
    
    stories_dir = project_root / 'chatbot' / 'data' / 'processed' / 'story_data'
    
    # chatbot.data.vector_db.utils.get_db_type_path 사용
    # current_dir_path는 populate_vector_db.py가 있는 .../vector_db 디렉토리
    vector_db_persist_path = get_db_type_path(base_directory=str(current_dir_path), db_type=db_type_to_use)
        
    # os.makedirs(vector_db_persist_path, exist_ok=True) # get_db_type_path가 ensure_directory 호출

    logger.info(f"스토리 데이터 디렉토리: {stories_dir}")
    logger.info(f"벡터 DB 디렉토리: {vector_db_persist_path} (타입: {db_type_to_use})")
    logger.info(f"대상 컬렉션: {collection_to_use}")
    if filter_criteria:
        logger.info(f"적용된 필터: {filter_criteria}")
    else:
        logger.info("적용된 필터 없음 (모든 데이터 또는 --import-all)")

    vector_db = VectorDB(persist_directory=str(vector_db_persist_path))
    try:
        vector_db.get_collection(collection_to_use)
        logger.info(f"기존 컬렉션 '{collection_to_use}'에 연결.")
    except Exception:
        vector_db.create_collection(
            name=collection_to_use,
            metadata={"description": f"꼬꼬북 {db_type_to_use} 동화 데이터", "type": db_type_to_use}
        )
        logger.info(f"새 컬렉션 '{collection_to_use}' 생성.")
    
    stories_data_list = []
    for json_filename in os.listdir(stories_dir):
        if json_filename.endswith('.json'):
            file_path = stories_dir / json_filename
            story_item = shared_load_json(file_path) # shared.utils.file_utils.load_json 사용
            if story_item is not None: # None 체크 (shared_load_json은 실패 시 None 반환)
                stories_data_list.append(story_item)
                if args.verbose:
                    logger.info(f"파일 로드: {json_filename}")
            # else: shared_load_json 내부에서 오류 로깅됨
    
    logger.info(f"총 {len(stories_data_list)}개 스토리 파일 로드됨.")

    stories_to_process = filter_stories(stories_data_list, filter_criteria, args.verbose)
    
    if not stories_to_process:
        logger.info("임포트할 스토리가 없습니다. (필터 결과 또는 원본 데이터 없음)")
        return

    logger.info(f"\n=== 총 {len(stories_to_process)}개 스토리 임포트 예정 ===")
    if args.verbose:
        for idx, story_p in enumerate(stories_to_process[:3], 1): # 미리보기 3개
            logger.info(f"  미리보기 {idx}: {story_p.get('title', '제목 없음')}")
        if len(stories_to_process) > 3:
            logger.info("  ...")

    confirm = input("ChromaDB에 스토리를 추가하시겠습니까? (y/n): ")
    if confirm.lower() != 'y':
        logger.info("작업이 취소되었습니다.")
        return

    docs_to_add_final = []
    metadatas_to_add_final = []
    ids_to_add_final = []
    imported_story_count = 0

    for story_content_original in stories_to_process: # 변수명 변경 (원래 JSON 의미)
        processed_s_data = process_story_data(story_content_original) 
        
        story_id_base = processed_s_data.get('story_id') # process_story_data에서 story_id를 처리
        if not story_id_base: # 만약 story_id가 없다면 title 기반으로 생성 (백업)
            story_id_base = processed_s_data.get('title', 'untitled').replace(' ', '_') + f"_{imported_story_count}"

        
        # DB 유형에 따라 문서 내용 구성
        doc_text_content = ""
        doc_specific_metadata = {}

        if db_type_to_use == 'summary':
            doc_text_content = processed_s_data.get('summary', '')
            # keywords는 이제 processed_s_data에 list 형태로 존재해야 함
            keywords_list = processed_s_data.get('keywords', [])
            doc_specific_metadata['keywords'] = ", ".join(keywords_list) if keywords_list else ""
        elif db_type_to_use == 'detailed':
            doc_text_content = processed_s_data.get('content', '') 
            # characters는 이제 processed_s_data에 list of names 형태로 존재
            characters_list = processed_s_data.get('characters', [])
            if characters_list:
                doc_specific_metadata['characters'] = ", ".join(characters_list)
        else:  # 'main' 또는 기본
            summary_text = processed_s_data.get('summary', '')
            content_text = processed_s_data.get('content', '')
            doc_text_content = f"{summary_text}\\n\\n{content_text}".strip()

        if not doc_text_content:
            if args.verbose:
                logger.info(f"스토리 ID '{story_id_base}'의 내용이 비어있어 스킵합니다. (DB 유형: {db_type_to_use})")
            continue

        # --- 메타데이터 포맷 통일 (중요) ---
        # DB에 저장될 age_group 형식을 'age_4_7' 등으로 통일
        raw_age_group = processed_s_data.get('age_group')
        final_age_group = None
        if raw_age_group == "4-7세":
            final_age_group = "age_4_7"
        elif raw_age_group == "8-9세":
            final_age_group = "age_8_9"
        # 필요한 경우 다른 조건 추가

        # 공통 메타데이터 (모두 processed_s_data 에서 가져옴)
        common_metadata = {
            'story_id': story_id_base, 
            'title': processed_s_data.get('title', '제목 없음'),
            'tags': ", ".join(processed_s_data.get('tags', [])),
            'age_group': final_age_group, # 통일된 포맷 사용
            'educational_value': processed_s_data.get('educational_value'), 
            'type': db_type_to_use, # 'type' 필드 명시적 추가
            **doc_specific_metadata
        }
        
        final_metadata = {k: v for k, v in common_metadata.items() if v is not None} 

        docs_to_add_final.append(doc_text_content)
        metadatas_to_add_final.append(final_metadata)
        # ID는 story_id_base 와 db_type_to_use 조합으로 고유성 강화 가능성 (중복 방지)
        # 예: f"{story_id_base}_{db_type_to_use}"
        # 현재는 story_id_base (processed_s_data.story_id)를 ID로 사용
        ids_to_add_final.append(story_id_base) 
        imported_story_count += 1

    if docs_to_add_final:
        try:
            vector_db.add_documents(
                documents=docs_to_add_final,
                metadatas=metadatas_to_add_final,
                ids=ids_to_add_final
            )
            logger.info(f"\n총 {len(docs_to_add_final)}개 문서가 ChromaDB 컬렉션 '{collection_to_use}' ({db_type_to_use})에 추가되었습니다.")
        except Exception as e:
            logger.error(f"문서 추가 중 오류 발생: {e}")
    else:
        logger.info("DB에 추가할 문서가 없습니다.")

    # 명시적 persist (VectorDB.core.py에 persist 메서드가 있다면)
    if hasattr(vector_db.client, 'persist') and callable(vector_db.client.persist):
        try:
            vector_db.client.persist()
            logger.info("ChromaDB에 변경사항이 명시적으로 저장되었습니다.")
        except Exception as e:
            logger.error(f"ChromaDB 명시적 저장 실패: {e}. 자동 저장을 기대합니다.")
    else:
        logger.info("ChromaDB는 PersistentClient 사용 시 자동 저장될 수 있습니다.")

    logger.info("데이터 채우기 작업 완료.")

if __name__ == "__main__":
    main() 