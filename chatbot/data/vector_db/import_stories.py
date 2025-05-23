import os
import sys
from pathlib import Path
import json
from dotenv import load_dotenv

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 디렉토리
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..')) # 프로젝트 루트 디렉토리
sys.path.append(project_root)

# 환경 변수 로드
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

# RAG 시스템 불러오기
from chatbot.models.rag_system import RAGSystem  

def main():
    """
    전처리된 동화 데이터를 ChromaDB에 추가하는 스크립트
    
    명령행 인수:
        --filter-age: 연령대로 필터링 (예: 5는 5-6세)
        --filter-theme: 테마로 필터링 (예: "우주", "모험")
        --verbose: 상세 정보 출력
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='전처리된 동화 데이터를 ChromaDB에 추가')
    parser.add_argument('--filter-age', type=int, help='연령대로 필터링 (예: 5는 5-6세)')
    parser.add_argument('--filter-theme', type=str, help='테마로 필터링 (예: "우주", "모험")')
    parser.add_argument('--verbose', action='store_true', help='상세 정보 출력')
    args = parser.parse_args()
    
    # 필터링 기준 설정
    filter_criteria = {}
    if args.filter_age:
        filter_criteria['age_group'] = args.filter_age
    if args.filter_theme:
        filter_criteria['theme'] = args.filter_theme
    
    # 디렉토리 경로 설정
    stories_dir = os.path.join(project_root, 'CCB_AI', 'chatbot', 'data', 'processed', 'story_data')
    
    print(f"스토리 데이터 디렉토리: {stories_dir}")
    print(f"필터링 기준: {filter_criteria if filter_criteria else '없음'}")
    
    # RAG 시스템 초기화
    vector_db_dir = os.path.join(project_root, 'CCB_AI', 'chatbot', 'data', 'vector_db')
    rag_system = RAGSystem(persist_directory=vector_db_dir)
    
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
    
    # 필터링된 스토리 미리보기
    if filter_criteria:
        preview_stories = rag_system.preview_filtered_stories(
            stories_data, 
            filter_criteria,
            verbose=args.verbose
        )
        
        # 미리보기 결과 출력
        print("\n=== 필터링된 스토리 미리보기 ===")
        for idx, story in enumerate(preview_stories[:10], 1):  # 최대 5개만 표시
            print(f"{idx}. {story.get('title', '제목 없음')} (ID: {story.get('story_id', 'ID 없음')})")
            print(f"   태그: {story.get('tags', '태그 없음')}")
            print(f"   요약: {story.get('summary', '요약 없음')[:100]}...")
            print()
        
        if len(preview_stories) > 10:
            print(f"...외 {len(preview_stories) - 10}개 스토리\n")
    
    # 사용자 확인
    confirm = input("ChromaDB에 스토리를 추가하시겠습니까? (y/n): ")
    if confirm.lower() != 'y':
        print("작업이 취소되었습니다.")
        return
    
    # 스토리 추가
    if filter_criteria:
        added_ids = rag_system.filter_and_import_stories(stories_data, filter_criteria)
    else:
        added_ids = rag_system.import_processed_stories(stories_data)
    
    print(f"\n총 {len(added_ids)}개 스토리가 ChromaDB에 추가되었습니다.")
    
    # 벡터 저장소 저장
    rag_system.summary_vectorstore.persist()
    rag_system.detailed_vectorstore.persist()
    print("ChromaDB에 변경사항이 저장되었습니다.")

if __name__ == "__main__":
    main() 