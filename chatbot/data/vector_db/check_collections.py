#!/usr/bin/env python3
import os
import sys
import sqlite3
import argparse
from pathlib import Path

"""
꼬꼬북 프로젝트 벡터 데이터베이스 컬렉션 관리 도구

이 모듈은 세 가지 벡터 데이터베이스를 관리합니다:
1. main DB: 일반적인 검색 목적으로 사용, 전체 스토리와 메타데이터 포함
2. detailed DB: 스토리 전개, 캐릭터 설명, 배경 설정 등 세부 내용 검색에 최적화
3. summary DB: 동화의 주제, 교훈, 키워드, 짧은 요약 등 핵심 정보 검색에 최적화

각 DB는 고유한 임베딩 특성을 가지며, 목적에 맞게 선택하여 사용하세요.
"""

# 현재 디렉토리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from vector_db import VectorDB

def check_collections(db_path):
    """
    ChromaDB의 컬렉션 목록을 확인합니다.
    """
    if not os.path.exists(db_path):
        print(f"데이터베이스 파일이 존재하지 않습니다: {db_path}")
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 테이블 목록 확인
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"데이터베이스 테이블 목록: {[table[0] for table in tables]}")
        
        # 컬렉션 정보 확인
        try:
            cursor.execute("SELECT name, id FROM collections")
            collections = cursor.fetchall()
            if collections:
                print("\n컬렉션 목록:")
                for name, uuid in collections:
                    print(f"  - {name} (UUID: {uuid})")
                return collections
            else:
                print("컬렉션이 없습니다.")
                return []
        except sqlite3.OperationalError:
            print("'collections' 테이블이 존재하지 않습니다.")
            return None
        
    except Exception as e:
        print(f"데이터베이스 확인 중 오류 발생: {str(e)}")
        return None
    finally:
        conn.close()

def create_collection(db_dir, collection_name="fairy_tales"):
    """
    ChromaDB에 새 컬렉션을 생성합니다.
    """
    try:
        vector_db = VectorDB(persist_directory=db_dir)
        vector_db.create_collection(
            name=collection_name,
            metadata={"description": "꼬꼬북 프로젝트 동화 데이터"}
        )
        print(f"컬렉션 '{collection_name}'이 생성되었습니다.")
        return True
    except Exception as e:
        print(f"컬렉션 생성 중 오류 발생: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='ChromaDB 컬렉션 확인 및 생성')
    parser.add_argument('--db-dir', type=str, choices=['root', 'main', 'detailed', 'summary'], default='main',
                      help='벡터 DB 저장 위치 (root, main, detailed, summary)')
    parser.add_argument('--create', action='store_true', help='컬렉션이 없으면 생성')
    parser.add_argument('--collection', type=str, default='fairy_tales', help='생성할 컬렉션 이름')
    args = parser.parse_args()
    
    # DB 디렉토리 설정
    if args.db_dir == 'detailed':
        vector_db_dir = os.path.join(current_dir, 'detailed')
    elif args.db_dir == 'summary':
        vector_db_dir = os.path.join(current_dir, 'summary')
    elif args.db_dir == 'main':
        vector_db_dir = os.path.join(current_dir, 'main')
    else:
        vector_db_dir = current_dir
    
    print(f"벡터 DB 디렉토리: {vector_db_dir}")
    
    # ChromaDB 파일 확인
    db_path = os.path.join(vector_db_dir, "chroma.sqlite3")
    collections = check_collections(db_path)
    
    # 컬렉션 생성 (필요한 경우)
    if args.create and (collections is None or not any(c[0] == args.collection for c in (collections or []))):
        print(f"컬렉션 '{args.collection}'이 존재하지 않습니다. 새로 생성합니다...")
        create_collection(vector_db_dir, args.collection)

if __name__ == "__main__":
    main()