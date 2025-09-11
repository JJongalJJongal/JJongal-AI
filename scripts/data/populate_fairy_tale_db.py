#!/usr/bin/env python3
"""
JJongal-AI Vector Database Population Script
Raw fairy tale data를 ChromaDB에 저장하는 스크립트
"""

import os
import sys
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

# 프로젝트 root path 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.vector_db.core import ModernVectorDB, LFUHybridVectorDB
from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

@dataclass
class FairyTaleData:
    """동화 데이터 구조"""
    title: str
    content: str
    age_group: str
    classification: str
    isbn: str
    author: Optional[str] = None
    illustrator: Optional[str] = None
    publisher: Optional[str] = None
    published_year: Optional[int] = None
    characters: Optional[List[str]] = None
    settings: Optional[List[str]] = None
    actions: Optional[List[str]] = None
    educational_themes: Optional[List[str]] = None
    page_count: Optional[int] = None

class FairyTaleDBPopulator:
    """동화 Vector Database 구성 클래스"""
    
    def __init__(self, db_path: str = None, use_hybrid: bool = True):
        """
        초기화
        
        Args:
            db_path: ChromaDB 저장 경로
            use_hybrid: LFU Hybrid VectorDB 사용 여부
        """
        self.db_path = db_path or os.getenv("VECTOR_DB_PATH", "data/chroma_db")
        self.use_hybrid = use_hybrid
        self.vector_db = None
        
        # 원천데이터와 라벨링데이터 경로
        self.source_data_path = Path("data/raw/Training/01.원천데이터")
        self.labeling_data_path = Path("data/raw/Training/02.라벨링데이터")
        
        # 분류별 폴더 매핑
        self.classification_folders = {
            "의사소통": ["TS_01T_의사소통_01S_유아", "TS_01T_의사소통_02S_초등_저학년"],
            "자연탐구": ["TS_02T_자연탐구_01S_유아", "TS_02T_자연탐구_02S_초등_저학년"],
            "사회관계": ["TS_03T_사회관계_01S_유아", "TS_03T_사회관계_02S_초등_저학년"],
            "예술경험": ["TS_04T_예술경험_01S_유아", "TS_04T_예술경험_02S_초등_저학년"],
            "신체운동·건강": ["TS_05T_신체운동_건강_01S_유아"]
        }
        
    def initialize_db(self):
        """Vector Database 초기화"""
        try:
            logger.info(f"Vector Database 초기화 시작: {self.db_path}")
            
            if self.use_hybrid:
                # LFU Hybrid VectorDB 사용
                self.vector_db = LFUHybridVectorDB(
                    persist_directory=self.db_path,
                    collection_name="fairy_tales",
                    use_openai=False,  # KURE-v1 Korean embeddings 사용
                    memory_cache_size=500,
                    disk_cache_size=2000,
                    enable_bm25=True
                )
                logger.info("LFU Hybrid VectorDB 초기화 완료")
            else:
                # 기본 ModernVectorDB 사용
                self.vector_db = ModernVectorDB(
                    persist_directory=self.db_path,
                    collection_name="fairy_tales"
                )
                logger.info("ModernVectorDB 초기화 완료")
                
        except Exception as e:
            logger.error(f"Vector Database 초기화 실패: {e}")
            raise
    
    def load_fairy_tale_data(self, json_file: Path, labeling_data: Optional[Dict] = None) -> Optional[FairyTaleData]:
        """JSON 파일에서 동화 데이터 로드"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 기본 정보
            title = data.get("title", "제목 없음")
            isbn = data.get("isbn", "")
            read_age = data.get("readAge", "")
            
            # 분류 정보 (파일 경로에서 추출)
            classification = self._extract_classification_from_path(json_file)
            
            # 페이지별 텍스트 결합
            image_info = data.get("imageInfo", [])
            content_parts = []
            characters_set = set()
            settings_set = set()
            actions_set = set()
            
            for page_info in image_info:
                src_text = page_info.get("srcText", "").strip()
                if src_text:
                    content_parts.append(src_text)
                
                # 라벨링 데이터에서 추가 정보 추출
                if labeling_data:
                    for labeled_page in labeling_data.get("imageInfo", []):
                        if labeled_page.get("srcTextID") == page_info.get("srcTextID"):
                            # 등장인물
                            if labeled_page.get("character"):
                                chars = labeled_page["character"].split(",")
                                characters_set.update([c.strip() for c in chars])
                            
                            # 배경
                            if labeled_page.get("setting"):
                                settings_set.add(labeled_page["setting"].strip())
                            
                            # 행동
                            if labeled_page.get("action"):
                                actions = labeled_page["action"].split(",")
                                actions_set.update([a.strip() for a in actions])
            
            # 전체 스토리 텍스트
            full_content = " ".join(content_parts)
            
            if not full_content:
                logger.warning(f"빈 콘텐츠: {json_file}")
                return None
            
            # 연령대 정규화
            age_group = self._normalize_age_group(read_age)
            
            return FairyTaleData(
                title=title,
                content=full_content,
                age_group=age_group,
                classification=classification,
                isbn=isbn,
                author=data.get("author"),
                illustrator=data.get("illustrator"),
                publisher=data.get("publisher"),
                published_year=data.get("publishedYear"),
                characters=list(characters_set) if characters_set else None,
                settings=list(settings_set) if settings_set else None,
                actions=list(actions_set) if actions_set else None,
                page_count=len(image_info)
            )
            
        except Exception as e:
            logger.error(f"데이터 로드 실패 {json_file}: {e}")
            return None
    
    def _extract_classification_from_path(self, file_path: Path) -> str:
        """파일 경로에서 분류 정보 추출"""
        path_str = str(file_path)
        
        if "의사소통" in path_str:
            return "의사소통"
        elif "자연탐구" in path_str:
            return "자연탐구"
        elif "사회관계" in path_str:
            return "사회관계"
        elif "예술경험" in path_str:
            return "예술경험"
        elif "신체운동" in path_str:
            return "신체운동·건강"
        else:
            return "기타"
    
    def _normalize_age_group(self, read_age: str) -> str:
        """연령대 정규화"""
        if "유아" in read_age.lower():
            return "4-7세"
        elif "초등" in read_age.lower() and "저학년" in read_age.lower():
            return "8-9세"
        else:
            return "4-7세"  # 기본값
    
    def load_labeling_data(self, labeling_file: Path) -> Optional[Dict]:
        """라벨링 데이터 로드"""
        try:
            if labeling_file.exists():
                with open(labeling_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"라벨링 데이터 로드 실패 {labeling_file}: {e}")
        return None
    
    def populate_from_raw_data(self, limit_per_category: Optional[int] = None):
        """Raw 데이터에서 Vector DB 구성"""
        if not self.vector_db:
            self.initialize_db()
        
        logger.info("Raw 데이터에서 동화 로딩 시작")
        
        total_processed = 0
        total_added = 0
        
        for classification, folders in self.classification_folders.items():
            logger.info(f"분류 '{classification}' 처리 시작")
            category_count = 0
            
            for folder in folders:
                source_folder = self.source_data_path / folder
                labeling_folder = self.labeling_data_path / folder.replace("TS_", "TL_")
                
                if not source_folder.exists():
                    logger.warning(f"폴더가 존재하지 않음: {source_folder}")
                    continue
                
                # JSON 파일 목록
                json_files = list(source_folder.glob("*.json"))
                
                # limit 적용
                if limit_per_category:
                    json_files = json_files[:limit_per_category]
                
                logger.info(f"폴더 '{folder}': {len(json_files)}개 파일 처리")
                
                batch_documents = []
                batch_metadatas = []
                batch_ids = []
                
                for json_file in json_files:
                    if limit_per_category and category_count >= limit_per_category:
                        break
                    
                    # 라벨링 데이터 로드
                    labeling_file = labeling_folder / json_file.name if labeling_folder.exists() else None
                    labeling_data = self.load_labeling_data(labeling_file) if labeling_file else None
                    
                    # 동화 데이터 로드
                    fairy_tale = self.load_fairy_tale_data(json_file, labeling_data)
                    
                    if fairy_tale and len(fairy_tale.content.strip()) > 50:  # 최소 길이 체크
                        # 문서 텍스트 구성
                        document_text = f"{fairy_tale.title}\n\n{fairy_tale.content}"
                        
                        # 메타데이터 구성
                        metadata = {
                            "title": fairy_tale.title,
                            "age_group": fairy_tale.age_group,
                            "classification": fairy_tale.classification,
                            "isbn": fairy_tale.isbn,
                            "author": fairy_tale.author or "",
                            "publisher": fairy_tale.publisher or "",
                            "published_year": fairy_tale.published_year or 0,
                            "page_count": fairy_tale.page_count or 0,
                            "document_type": "fairy_tale",
                            "data_source": "training_data"
                        }
                        
                        # 추가 정보
                        if fairy_tale.characters:
                            metadata["characters"] = ",".join(fairy_tale.characters)
                        if fairy_tale.settings:
                            metadata["settings"] = ",".join(fairy_tale.settings)
                        if fairy_tale.actions:
                            metadata["actions"] = ",".join(fairy_tale.actions)
                        
                        # 배치에 추가
                        batch_documents.append(document_text)
                        batch_metadatas.append(metadata)
                        batch_ids.append(f"story_{fairy_tale.isbn}_{json_file.stem}")
                        
                        category_count += 1
                        total_processed += 1
                
                # 배치 저장
                if batch_documents:
                    try:
                        if self.use_hybrid:
                            # LFU Hybrid VectorDB 사용
                            self.vector_db.add_documents(
                                texts=batch_documents,
                                metadatas=batch_metadatas,
                                ids=batch_ids
                            )
                        else:
                            # ModernVectorDB 사용
                            self.vector_db.add_documents(
                                documents=batch_documents,
                                metadatas=batch_metadatas,
                                ids=batch_ids
                            )
                        
                        total_added += len(batch_documents)
                        logger.info(f"'{folder}': {len(batch_documents)}개 문서 추가됨")
                        
                    except Exception as e:
                        logger.error(f"배치 저장 실패 '{folder}': {e}")
        
        logger.info(f"데이터 처리 완료: {total_processed}개 처리, {total_added}개 추가")
    
    def test_search_functionality(self):
        """검색 기능 테스트"""
        if not self.vector_db:
            self.initialize_db()
        
        test_queries = [
            "친구와 함께 하는 모험",
            "숲 속의 동물들",
            "마법의 세계",
            "용기있는 아이",
            "자연을 사랑하는 이야기",
            "가족의 소중함"
        ]
        
        logger.info("검색 기능 테스트 시작")
        
        for query in test_queries:
            try:
                if self.use_hybrid:
                    # 하이브리드 검색
                    results = self.vector_db.hybrid_search(query, k=3)
                    logger.info(f"쿼리: '{query}' - {len(results)}개 결과 (하이브리드)")
                    
                    for i, result in enumerate(results):
                        title = result.metadata.get("title", "제목 없음")
                        age_group = result.metadata.get("age_group", "")
                        classification = result.metadata.get("classification", "")
                        score = f"점수: {result.final_score:.3f}" if hasattr(result, 'final_score') else ""
                        logger.info(f"  {i+1}. {title} ({age_group}, {classification}) {score}")
                else:
                    # 일반 검색
                    results = self.vector_db.similarity_search(query, k=3)
                    logger.info(f"쿼리: '{query}' - {len(results)}개 결과")
                    
                    for i, result in enumerate(results):
                        title = result.metadata.get("title", "제목 없음")
                        age_group = result.metadata.get("age_group", "")
                        classification = result.metadata.get("classification", "")
                        logger.info(f"  {i+1}. {title} ({age_group}, {classification})")
                        
            except Exception as e:
                logger.error(f"검색 테스트 실패 (쿼리: {query}): {e}")
    
    def get_database_stats(self):
        """데이터베이스 통계 조회"""
        if not self.vector_db:
            self.initialize_db()
        
        try:
            if self.use_hybrid:
                # LFU Hybrid VectorDB 통계
                stats = self.vector_db.get_cache_stats()
                
                logger.info("=== Vector Database 통계 ===")
                logger.info(f"저장 경로: {self.db_path}")
                logger.info(f"컬렉션: {self.vector_db.collection_name}")
                logger.info(f"캐시 통계: {stats['cache_stats']}")
                logger.info(f"BM25 활성화: {stats['bm25_enabled']}")
                
            else:
                # ModernVectorDB 통계  
                try:
                    collection_count = self.vector_db.vectorstore._collection.count()
                    logger.info("=== Vector Database 통계 ===")
                    logger.info(f"저장 경로: {self.db_path}")
                    logger.info(f"총 문서 수: {collection_count}")
                    
                except Exception as e:
                    logger.warning(f"통계 조회 실패: {e}")
            
            # 분류별 통계 (메타데이터 기반)
            logger.info("\n=== 분류별 통계 ===")
            for classification in self.classification_folders.keys():
                try:
                    if self.use_hybrid:
                        results = self.vector_db.vectorstore.get(
                            where={"classification": classification}
                        )
                    else:
                        results = self.vector_db.vectorstore._collection.get(
                            where={"classification": classification}
                        )
                    
                    count = len(results.get('ids', []))
                    logger.info(f"{classification}: {count}개 문서")
                    
                except Exception as e:
                    logger.warning(f"{classification} 통계 조회 실패: {e}")
                    
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="JJongal-AI Vector Database 구성")
    parser.add_argument("--db-path", help="ChromaDB 저장 경로")
    parser.add_argument("--use-basic", action="store_true", help="기본 VectorDB 사용 (하이브리드 대신)")
    parser.add_argument("--limit", type=int, help="카테고리당 최대 처리 개수")
    parser.add_argument("--test-only", action="store_true", help="검색 테스트만 실행")
    parser.add_argument("--stats", action="store_true", help="통계만 조회")
    
    args = parser.parse_args()
    
    try:
        # 초기화
        use_hybrid = not args.use_basic
        populator = FairyTaleDBPopulator(
            db_path=args.db_path, 
            use_hybrid=use_hybrid
        )
        
        if args.stats:
            populator.get_database_stats()
            return
        
        if args.test_only:
            populator.test_search_functionality()
            return
        
        # 데이터베이스 초기화
        populator.initialize_db()
        
        # 원시 데이터 처리
        logger.info("동화 데이터 처리 시작...")
        start_time = time.time()
        
        populator.populate_from_raw_data(limit_per_category=args.limit)
        
        processing_time = time.time() - start_time
        logger.info(f"데이터 처리 완료 (소요시간: {processing_time:.1f}초)")
        
        # 검색 테스트
        populator.test_search_functionality()
        
        # 통계 출력
        populator.get_database_stats()
        
        logger.info("Vector Database 구성 완료!")
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()