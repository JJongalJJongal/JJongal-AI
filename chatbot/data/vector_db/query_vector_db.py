import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

"""
꼬꼬북 프로젝트 벡터 데이터베이스 쿼리 도구

이 모듈은 세 가지 벡터 데이터베이스를 쿼리합니다:
1. main DB: 일반적인 검색 목적으로 사용, 전체 스토리와 메타데이터 포함
2. detailed DB: 스토리 전개, 캐릭터 설명, 배경 설정 등 세부 내용 검색에 최적화
3. summary DB: 동화의 주제, 교훈, 키워드, 짧은 요약 등 핵심 정보 검색에 최적화

각 DB는 고유한 임베딩 특성을 가지며, 목적에 맞게 선택하여 사용하세요.

사용 예시:
- 일반 검색: query_vector_db.py --query "우주 모험" --db-dir main
- 세부 내용 검색: query_vector_db.py --query "주인공이 로켓을 타고 화성으로 여행하는 내용" --db-dir detailed
- 요약/주제 검색: query_vector_db.py --query "용기와 모험심" --db-dir summary
"""

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 환경 변수 로드
dotenv_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

# VectorDB 클래스 불러오기
from vector_db import VectorDB

def main():
    """
    벡터 데이터베이스에 저장된 동화 쿼리 스크립트
    """
    parser = argparse.ArgumentParser(description='ChromaDB에 저장된 동화 데이터 쿼리')
    parser.add_argument('--query', type=str, required=True, help='검색 쿼리')
    parser.add_argument('--collection', type=str, default='fairy_tales', help='ChromaDB 컬렉션 이름')
    parser.add_argument('--filter-tags', type=str, help='태그로 필터링 (예: "5-6세")')
    parser.add_argument('--results', type=int, default=3, help='반환할 결과 수')
    parser.add_argument('--db-dir', type=str, choices=['root', 'main', 'detailed', 'summary'], default='main',
                       help='벡터 DB 저장 위치 (root, main, detailed, summary)')
    args = parser.parse_args()
    
    # 벡터 DB 디렉토리 설정
    if args.db_dir == 'detailed':
        vector_db_dir = os.path.join(current_dir, 'detailed')
    elif args.db_dir == 'summary':
        vector_db_dir = os.path.join(current_dir, 'summary')
    elif args.db_dir == 'main':
        vector_db_dir = os.path.join(current_dir, 'main')
    else:
        vector_db_dir = current_dir
    
    print(f"벡터 DB 디렉토리: {vector_db_dir}")
    print(f"컬렉션: {args.collection}")
    print(f"쿼리: '{args.query}'")
    
    # 필터 설정
    where_filter = None
    if args.filter_tags:
        where_filter = {"tags": {"$contains": args.filter_tags}}
        print(f"필터: 태그에 '{args.filter_tags}' 포함")
    
    # VectorDB 초기화
    vector_db = VectorDB(persist_directory=vector_db_dir)
    
    try:
        # 컬렉션 가져오기
        vector_db.get_collection(args.collection)
        print(f"컬렉션 '{args.collection}'에 연결되었습니다.")
        
        # 쿼리 실행
        results = vector_db.query(
            query_texts=[args.query],
            n_results=args.results,
            where=where_filter
        )
        
        # 결과 출력
        print("\n=== 검색 결과 ===")
        if not results['ids'][0]:
            print("검색 결과가 없습니다.")
            return
            
        for i, (doc_id, doc, metadata) in enumerate(zip(results['ids'][0], results['documents'][0], results['metadatas'][0])):
            print(f"결과 {i+1}: {metadata.get('title', '제목 없음')}")
            print(f"   ID: {doc_id}")
            print(f"   태그: {metadata.get('tags', '태그 없음')}")
            print(f"   요약: {metadata.get('summary', '요약 없음')[:100]}...")
            
            # 스토리 내용 일부 출력
            content = doc.replace(metadata.get('summary', ''), '').strip()
            if content:
                print(f"   내용: {content[:150]}..." if len(content) > 150 else f"   내용: {content}")
            print()
            
    except Exception as e:
        print(f"쿼리 실행 중 오류 발생: {str(e)}")
        
if __name__ == "__main__":
    main() 