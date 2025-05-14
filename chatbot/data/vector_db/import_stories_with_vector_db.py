import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

"""
꼬꼬북 프로젝트 벡터 데이터베이스 데이터 가져오기 도구

이 모듈은 전처리된 스토리 데이터를 세 가지 벡터 데이터베이스에 추가합니다:
1. main DB: 일반적인 검색 목적으로 사용, 전체 스토리와 메타데이터 포함
2. detailed DB: 스토리 전개, 캐릭터 설명, 배경 설정 등 세부 내용 검색에 최적화
3. summary DB: 동화의 주제, 교훈, 키워드, 짧은 요약 등 핵심 정보 검색에 최적화

각 DB는 목적에 맞게 다른 임베딩 특성을 가질 수 있으며, 관련 데이터를 적절히 강조하여 저장합니다.

사용 예시:
- 기본 데이터 추가: import_stories_with_vector_db.py --db-dir main
- 상세 정보용 데이터 추가: import_stories_with_vector_db.py --db-dir detailed
- 요약 정보용 데이터 추가: import_stories_with_vector_db.py --db-dir summary
"""

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)

# 환경 변수 로드
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

# VectorDB 클래스 불러오기
from vector_db import VectorDB

def filter_stories(stories_data: List[Dict[str, Any]], filter_criteria: Dict[str, Any], verbose: bool = False) -> List[Dict[str, Any]]:
    """
    필터링 기준에 맞는 스토리 목록을 반환
    
    Args:
        stories_data: 전처리된 스토리 데이터 리스트
        filter_criteria: 필터링 기준 (예: {"age_group": 5, "theme": "우주"})
        verbose: 상세 정보 출력 여부
        
    Returns:
        List[Dict]: 필터링된 스토리 목록
    """
    filtered_stories = []
    
    for story in stories_data:
        # 모든 필터 기준을 만족하는지 확인
        matches_all_criteria = True
        
        for key, value in filter_criteria.items():
            # 특정 필드에 대한 부분 일치 검색
            if key in story:
                # 문자열의 경우 부분 일치 확인
                if isinstance(story[key], str) and isinstance(value, str):
                    if value.lower() not in story[key].lower():
                        matches_all_criteria = False
                        break
                # 숫자나 기타 타입의 경우 정확히 일치하는지 확인
                elif story[key] != value:
                    matches_all_criteria = False
                    break
            elif key == "age" and "tags" in story:
                # tags 필드에서 연령대 확인 (예: "5-6세")
                age_tag = f"{value}-{value+1}세"
                if age_tag not in story["tags"]:
                    matches_all_criteria = False
                    break
            else:
                # 필드가 없는 경우 기준 불만족
                matches_all_criteria = False
                break
        
        if matches_all_criteria:
            filtered_stories.append(story)
    
    print(f"필터링 결과: 전체 {len(stories_data)}개 중 {len(filtered_stories)}개 스토리가 기준에 부합")
    return filtered_stories

def main():
    """
    전처리된 동화 데이터를 ChromaDB에 추가하는 스크립트
    """
    parser = argparse.ArgumentParser(description='전처리된 동화 데이터를 ChromaDB에 추가')
    parser.add_argument('--filter-age', type=int, help='연령대로 필터링 (예: 5는 5-6세)')
    parser.add_argument('--filter-theme', type=str, help='테마로 필터링 (예: "우주", "모험")')
    parser.add_argument('--collection', type=str, default='fairy_tales', help='ChromaDB 컬렉션 이름')
    parser.add_argument('--verbose', action='store_true', help='상세 정보 출력')
    parser.add_argument('--db-dir', type=str, choices=['root', 'main', 'detailed', 'summary'], default='main',
                       help='벡터 DB 저장 위치 (root, main, detailed, summary)')
    args = parser.parse_args()
    
    # 필터링 기준 설정
    filter_criteria = {}
    if args.filter_age:
        filter_criteria['age'] = args.filter_age
    if args.filter_theme:
        filter_criteria['theme'] = args.filter_theme
    
    # 디렉토리 경로 설정 (절대 경로 사용)
    stories_dir = os.path.join(project_root, 'chatbot', 'data', 'processed', 'story_data')
    
    # DB 디렉토리 설정
    if args.db_dir == 'detailed':
        vector_db_dir = os.path.join(current_dir, 'detailed')
    elif args.db_dir == 'summary':
        vector_db_dir = os.path.join(current_dir, 'summary')
    elif args.db_dir == 'main':
        vector_db_dir = os.path.join(current_dir, 'main')
    else:
        vector_db_dir = current_dir
        
    # DB 유형에 따라 데이터 처리 방식 조정
    db_type = args.db_dir if args.db_dir in ['main', 'detailed', 'summary'] else 'main'
    
    print(f"스토리 데이터 디렉토리: {stories_dir}")
    print(f"벡터 DB 디렉토리: {vector_db_dir}")
    print(f"벡터 DB 타입: {db_type}")
    print(f"필터링 기준: {filter_criteria if filter_criteria else '없음'}")
    
    # VectorDB 초기화
    vector_db = VectorDB(persist_directory=vector_db_dir)
    
    # 컬렉션 생성 또는 가져오기
    try:
        # 기존 컬렉션 가져오기 시도
        vector_db.get_collection(args.collection)
        print(f"기존 컬렉션 '{args.collection}'에 연결되었습니다.")
    except Exception:
        # 컬렉션이 없으면 새로 생성
        vector_db.create_collection(
            name=args.collection,
            metadata={"description": "꼬꼬북 프로젝트 동화 데이터", "type": db_type}
        )
        print(f"새 컬렉션 '{args.collection}'이 생성되었습니다.")
    
    # 스토리 데이터 준비
    stories_data = []
    for json_file in os.listdir(stories_dir):
        if json_file.endswith('.json'):
            file_path = os.path.join(stories_dir, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    story_data = json.load(f)
                    stories_data.append(story_data)
                    if args.verbose:
                        print(f"파일 로드: {json_file}")
            except Exception as e:
                print(f"파일 로드 중 오류 발생: {json_file}, 오류: {str(e)}")
    
    print(f"총 {len(stories_data)}개 스토리 데이터 로드됨")
    
    # 필터링 적용
    if filter_criteria:
        filtered_stories = filter_stories(stories_data, filter_criteria, args.verbose)
        stories_to_import = filtered_stories
    else:
        stories_to_import = stories_data
    
    # 미리보기 출력
    print("\n=== 추가될 스토리 미리보기 ===")
    for idx, story in enumerate(stories_to_import[:5], 1):  # 최대 5개만 표시
        print(f"{idx}. {story.get('title', '제목 없음')} (ID: {story.get('story_id', 'ID 없음')})")
        print(f"   태그: {story.get('tags', '태그 없음')}")
        print(f"   요약: {story.get('summary', '요약 없음')[:100]}...")
        print()
    
    if len(stories_to_import) > 5:
        print(f"...외 {len(stories_to_import) - 5}개 스토리\n")
    
    # 사용자 확인
    confirm = input("ChromaDB에 스토리를 추가하시겠습니까? (y/n): ")
    if confirm.lower() != 'y':
        print("작업이 취소되었습니다.")
        return
    
    # DB 유형에 따라 다른 데이터 준비
    documents = []  # 스토리 내용 (embedding 대상)
    metadatas = []  # 메타데이터
    ids = []        # 문서 ID
    
    for i, story in enumerate(stories_to_import):
        # 스토리 ID 설정
        story_id = story.get('story_id', f'story_{i+1}')
        
        # 기본 메타데이터 설정
        metadata = {
            'story_id': story_id,
            'title': story.get('title', '제목 없음'),
            'tags': story.get('tags', ''),
            'summary': story.get('summary', ''),
            'db_type': db_type  # DB 유형 추가
        }
        
        # 원본 메타데이터가 있으면 추가
        if 'metadata' in story and isinstance(story['metadata'], dict):
            for key, value in story['metadata'].items():
                metadata[f'meta_{key}'] = value
        
        # DB 유형에 따라 다른 내용 사용
        content = story.get('content', '')
        summary = story.get('summary', '')
        
        if db_type == 'summary':
            # 요약 DB: 요약과 핵심 키워드 중심
            doc_text = summary
            metadata['keywords'] = story.get('keywords', '')
        elif db_type == 'detailed':
            # 상세 DB: 전체 내용과 캐릭터 정보, 장면 설명 중심
            doc_text = content
            # 캐릭터 정보가 있으면 추가
            if 'characters' in story and isinstance(story['characters'], list):
                metadata['characters'] = ", ".join([c.get('name', '') for c in story['characters']])
        else:  # 'main' 또는 기본값
            # 일반 DB: 전체 내용 + 요약 (균형 잡힌 정보)
            doc_text = f"{summary}\n\n{content}"
        
        documents.append(doc_text)
        metadatas.append(metadata)
        ids.append(story_id)
    
    # ChromaDB에 스토리 추가
    try:
        vector_db.add_documents(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"\n총 {len(documents)}개 스토리가 ChromaDB ({db_type})에 추가되었습니다.")
    except Exception as e:
        print(f"스토리 추가 중 오류 발생: {str(e)}")
        return
    
    # 간단한 검증 쿼리 실행
    try:
        if len(documents) > 0:
            print("\n=== 검증 쿼리 실행 ===")
            # 첫 번째 스토리의 제목으로 검색
            sample_query = metadatas[0]['title']
            results = vector_db.query(
                query_texts=[sample_query],
                n_results=2
            )
            print(f"쿼리: '{sample_query}'")
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                print(f"결과 {i+1}: {metadata.get('title', '제목 없음')}")
                print(f"   ID: {metadata.get('story_id', 'ID 없음')}")
                print(f"   유형: {metadata.get('db_type', 'unknown')}")
                print(f"   요약: {metadata.get('summary', '요약 없음')[:100]}...\n")
    except Exception as e:
        print(f"검증 쿼리 실행 중 오류 발생: {str(e)}")
    
    print("작업이 완료되었습니다.")

if __name__ == "__main__":
    main() 