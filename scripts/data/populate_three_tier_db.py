#!/usr/bin/env python3
"""
JJongal-AI 3-Tier Vector Database Population Script
Raw fairy tale data를 3개 ChromaDB로 구성:
1. main DB: 전체 원본 데이터
2. detailed DB: 아리(ChatBot B)용 자세한 이야기 
3. summary DB: 쫑이(ChatBot A)용 요약 정보
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib

# 프로젝트 root path 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.vector_db.core import ModernVectorDB, LFUHybridVectorDB
from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

@dataclass 
class ThreeTierConfig:
    """3-Tier DB 구성"""
    main_db_path: str = "data/chroma_db/main"
    detailed_db_path: str = "data/chroma_db/detailed"  # ChatBot B (아리)
    summary_db_path: str = "data/chroma_db/summary"    # ChatBot A (쫑이)
    use_hybrid: bool = True

@dataclass
class StoryContent:
    """동화 컨텐츠 구조"""
    title: str
    full_text: str          # 전체 원문
    detailed_text: str      # 아리용 상세 텍스트
    summary_text: str       # 쫑이용 요약 텍스트
    metadata: Dict[str, Any]

class ThreeTierDBPopulator:
    """3단계 Vector Database 구성 클래스"""
    
    def __init__(self, config: ThreeTierConfig = None):
        self.config = config or ThreeTierConfig()
        
        # 3개 VectorDB 인스턴스
        self.main_db: Optional[LFUHybridVectorDB] = None
        self.detailed_db: Optional[LFUHybridVectorDB] = None  
        self.summary_db: Optional[LFUHybridVectorDB] = None
        
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
    
    def initialize_databases(self):
        """3개 VectorDB 초기화"""
        logger.info("🚀 3-Tier VectorDB 초기화 시작")
        
        # 디렉토리 생성
        for db_path in [self.config.main_db_path, self.config.detailed_db_path, self.config.summary_db_path]:
            Path(db_path).mkdir(parents=True, exist_ok=True)
        
        if self.config.use_hybrid:
            # 1. Main DB - 원본 데이터 (전체 검색용)
            self.main_db = LFUHybridVectorDB(
                persist_directory=self.config.main_db_path,
                collection_name="fairy_tales_main",
                use_openai=False,
                memory_cache_size=800,
                disk_cache_size=3000, 
                enable_bm25=True
            )
            
            # 2. Detailed DB - ChatBot B (아리)용 상세 이야기
            self.detailed_db = LFUHybridVectorDB(
                persist_directory=self.config.detailed_db_path,
                collection_name="fairy_tales_detailed", 
                use_openai=False,
                memory_cache_size=600,
                disk_cache_size=2000,
                enable_bm25=True  # 상세 검색 지원
            )
            
            # 3. Summary DB - ChatBot A (쫑이)용 요약 정보  
            self.summary_db = LFUHybridVectorDB(
                persist_directory=self.config.summary_db_path,
                collection_name="fairy_tales_summary",
                use_openai=False,
                memory_cache_size=400,
                disk_cache_size=1000,
                enable_bm25=True  # 빠른 검색용
            )
            
            logger.info("✅ LFU Hybrid VectorDB 3개 초기화 완료")
            
        else:
            # 기본 ModernVectorDB 사용
            self.main_db = ModernVectorDB(
                persist_directory=self.config.main_db_path,
                collection_name="fairy_tales_main"
            )
            self.detailed_db = ModernVectorDB(
                persist_directory=self.config.detailed_db_path,
                collection_name="fairy_tales_detailed"
            )
            self.summary_db = ModernVectorDB(
                persist_directory=self.config.summary_db_path,
                collection_name="fairy_tales_summary"
            )
            
            logger.info("✅ ModernVectorDB 3개 초기화 완료")
    
    def process_story_data(self, json_file: Path, labeling_data: Optional[Dict] = None) -> Optional[StoryContent]:
        """JSON 파일에서 3-tier 컨텐츠 생성"""
        try:
            with open(json_file, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            
            # 기본 정보
            title = data.get("title", "제목 없음")
            isbn = data.get("isbn", "")
            read_age = data.get("readAge", "")
            classification = self._extract_classification_from_path(json_file)
            
            # 페이지별 텍스트 수집
            image_info = data.get("imageInfo", [])
            pages_content = []
            characters_set = set()
            settings_set = set()
            actions_set = set()
            
            for page_info in image_info:
                src_text = page_info.get("srcText", "").strip()
                if src_text:
                    pages_content.append(src_text)
                
                # 라벨링 데이터에서 메타데이터 추출
                if labeling_data:
                    for labeled_page in labeling_data.get("imageInfo", []):
                        if labeled_page.get("srcTextID") == page_info.get("srcTextID"):
                            if labeled_page.get("character"):
                                chars = labeled_page["character"].split(",")
                                characters_set.update([c.strip() for c in chars])
                            if labeled_page.get("setting"):
                                settings_set.add(labeled_page["setting"].strip())
                            if labeled_page.get("action"):
                                actions = labeled_page["action"].split(",")
                                actions_set.update([a.strip() for a in actions])
            
            if not pages_content:
                return None
            
            # 1. Full Text - 전체 원문 (Main DB용)
            full_text = " ".join(pages_content)
            
            # 2. Detailed Text - 상세 텍스트 (ChatBot B 아리용)
            detailed_text = self._create_detailed_version(title, pages_content, characters_set, settings_set, actions_set)
            
            # 3. Summary Text - 요약 텍스트 (ChatBot A 쫑이용)
            summary_text = self._create_summary_version(title, pages_content, characters_set, settings_set)
            
            # 공통 메타데이터
            metadata = {
                "title": title,
                "age_group": self._normalize_age_group(read_age),
                "classification": classification,
                "isbn": isbn,
                "author": data.get("author", ""),
                "illustrator": data.get("illustrator", ""),
                "publisher": data.get("publisher", ""),
                "published_year": data.get("publishedYear", 0),
                "page_count": len(image_info),
                "characters": ",".join(characters_set) if characters_set else "",
                "settings": ",".join(settings_set) if settings_set else "",
                "actions": ",".join(actions_set) if actions_set else "",
                "content_length": len(full_text),
                "data_source": "training_data"
            }
            
            return StoryContent(
                title=title,
                full_text=full_text,
                detailed_text=detailed_text,
                summary_text=summary_text,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"스토리 데이터 처리 실패 {json_file}: {e}")
            return None
    
    def _create_detailed_version(self, title: str, pages: List[str], characters: set, settings: set, actions: set) -> str:
        """ChatBot B (아리)용 상세 버전 생성"""
        detailed_parts = [f"제목: {title}"]
        
        # 등장인물 정보
        if characters:
            detailed_parts.append(f"등장인물: {', '.join(characters)}")
        
        # 배경 정보  
        if settings:
            detailed_parts.append(f"배경: {', '.join(settings)}")
        
        # 주요 행동/사건
        if actions:
            detailed_parts.append(f"주요 사건: {', '.join(actions)}")
        
        # 전체 이야기 (페이지별로 구분)
        detailed_parts.append("상세 내용:")
        for i, page in enumerate(pages, 1):
            detailed_parts.append(f"[{i}페이지] {page}")
        
        return "\n\n".join(detailed_parts)
    
    def _create_summary_version(self, title: str, pages: List[str], characters: set, settings: set) -> str:
        """ChatBot A (쫑이)용 요약 버전 생성"""
        # 첫 페이지와 마지막 페이지만 사용하여 간단한 요약 생성
        beginning = pages[0] if pages else ""
        ending = pages[-1] if len(pages) > 1 else ""
        
        summary_parts = [f"제목: {title}"]
        
        # 기본 정보만 포함
        if characters:
            main_chars = list(characters)[:3]  # 주요 등장인물 3명만
            summary_parts.append(f"주인공: {', '.join(main_chars)}")
        
        if settings:
            main_setting = list(settings)[0]  # 주요 배경 1개만
            summary_parts.append(f"배경: {main_setting}")
        
        # 간단한 줄거리 (처음과 끝만)
        if beginning and ending and beginning != ending:
            summary_parts.append(f"이야기: {beginning[:100]}... {ending[:100]}")
        else:
            summary_parts.append(f"이야기: {beginning[:200]}")
        
        return "\n".join(summary_parts)
    
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
                with open(labeling_file, 'r', encoding='utf-8-sig') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"라벨링 데이터 로드 실패 {labeling_file}: {e}")
        return None
    
    def populate_three_tier_dbs(self, limit_per_category: Optional[int] = None):
        """3-Tier DB에 데이터 저장"""
        if not all([self.main_db, self.detailed_db, self.summary_db]):
            self.initialize_databases()
        
        logger.info("🎯 3-Tier DB 데이터 저장 시작")
        
        total_processed = 0
        total_added = 0
        
        for classification, folders in self.classification_folders.items():
            logger.info(f"📂 분류 '{classification}' 처리 시작")
            category_count = 0
            
            # 배치 저장용 리스트들
            main_batch = {"texts": [], "metadatas": [], "ids": []}
            detailed_batch = {"texts": [], "metadatas": [], "ids": []}
            summary_batch = {"texts": [], "metadatas": [], "ids": []}
            
            for folder in folders:
                source_folder = self.source_data_path / folder
                labeling_folder = self.labeling_data_path / folder.replace("TS_", "TL_")
                
                if not source_folder.exists():
                    logger.warning(f"폴더 미존재: {source_folder}")
                    continue
                
                json_files = list(source_folder.glob("*.json"))
                
                if limit_per_category:
                    json_files = json_files[:limit_per_category]
                
                logger.info(f"📁 '{folder}': {len(json_files)}개 파일 처리")
                
                for json_file in json_files:
                    if limit_per_category and category_count >= limit_per_category:
                        break
                    
                    # 라벨링 데이터 로드
                    labeling_file = labeling_folder / json_file.name if labeling_folder.exists() else None
                    labeling_data = self.load_labeling_data(labeling_file) if labeling_file else None
                    
                    # 스토리 컨텐츠 처리
                    story_content = self.process_story_data(json_file, labeling_data)
                    
                    if story_content and len(story_content.full_text.strip()) > 50:
                        # 고유 ID 생성
                        story_hash = hashlib.md5(
                            f"{story_content.title}_{story_content.metadata['isbn']}_{json_file.stem}".encode()
                        ).hexdigest()[:12]
                        
                        # 1. Main DB - 전체 원본 데이터
                        main_metadata = {**story_content.metadata, "db_type": "main", "content_type": "full"}
                        main_batch["texts"].append(f"{story_content.title}\n\n{story_content.full_text}")
                        main_batch["metadatas"].append(main_metadata)
                        main_batch["ids"].append(f"main_{story_hash}")
                        
                        # 2. Detailed DB - 아리용 상세 데이터
                        detailed_metadata = {**story_content.metadata, "db_type": "detailed", "content_type": "detailed"}
                        detailed_batch["texts"].append(story_content.detailed_text)
                        detailed_batch["metadatas"].append(detailed_metadata)
                        detailed_batch["ids"].append(f"detailed_{story_hash}")
                        
                        # 3. Summary DB - 쫑이용 요약 데이터  
                        summary_metadata = {**story_content.metadata, "db_type": "summary", "content_type": "summary"}
                        summary_batch["texts"].append(story_content.summary_text)
                        summary_batch["metadatas"].append(summary_metadata)
                        summary_batch["ids"].append(f"summary_{story_hash}")
                        
                        category_count += 1
                        total_processed += 1
            
            # 배치 저장 (작은 청크로 분할하여 처리)
            if main_batch["texts"]:
                try:
                    batch_size = min(2000, len(main_batch["texts"]))  # 최대 2000개씩 처리
                    
                    for i in range(0, len(main_batch["texts"]), batch_size):
                        end_idx = min(i + batch_size, len(main_batch["texts"]))
                        
                        # 현재 청크 추출
                        main_chunk = {
                            "texts": main_batch["texts"][i:end_idx],
                            "metadatas": main_batch["metadatas"][i:end_idx]
                        }
                        detailed_chunk = {
                            "texts": detailed_batch["texts"][i:end_idx], 
                            "metadatas": detailed_batch["metadatas"][i:end_idx]
                        }
                        summary_chunk = {
                            "texts": summary_batch["texts"][i:end_idx],
                            "metadatas": summary_batch["metadatas"][i:end_idx]
                        }
                        
                        logger.info(f"📦 '{classification}' 청크 {i//batch_size + 1}: {len(main_chunk['texts'])}개 문서 처리중...")
                        
                        # Main DB 저장
                        if self.config.use_hybrid:
                            self.main_db.add_documents_with_bm25(
                                texts=main_chunk["texts"],
                                metadatas=main_chunk["metadatas"],
                                update_bm25=True
                            )
                        else:
                            self.main_db.add_documents(
                                texts=main_chunk["texts"],
                                metadatas=main_chunk["metadatas"]
                            )
                        
                        # Detailed DB 저장
                        if self.config.use_hybrid:
                            self.detailed_db.add_documents_with_bm25(
                                texts=detailed_chunk["texts"],
                                metadatas=detailed_chunk["metadatas"],
                                update_bm25=True
                            )
                        else:
                            self.detailed_db.add_documents(
                                texts=detailed_chunk["texts"],
                                metadatas=detailed_chunk["metadatas"]
                            )
                        
                        # Summary DB 저장
                        if self.config.use_hybrid:
                            self.summary_db.add_documents_with_bm25(
                                texts=summary_chunk["texts"],
                                metadatas=summary_chunk["metadatas"],
                                update_bm25=True
                            )
                        else:
                            self.summary_db.add_documents(
                                texts=summary_chunk["texts"],
                                metadatas=summary_chunk["metadatas"]
                            )
                        
                        logger.info(f"✅ 청크 {i//batch_size + 1} 완료: {len(main_chunk['texts'])}개 문서 저장")
                    
                    total_added += len(main_batch["texts"])
                    logger.info(f"🎯 '{classification}' 완료: 총 {len(main_batch['texts'])}개 문서를 3-Tier DB에 저장")
                    
                except Exception as e:
                    logger.error(f"❌ '{classification}' 배치 저장 실패: {e}")
        
        logger.info(f"🎯 3-Tier DB 구성 완료: {total_processed}개 처리, {total_added}개 저장")
    
    def test_three_tier_search(self):
        """3-Tier DB 검색 기능 테스트"""
        logger.info("🔍 3-Tier DB 검색 테스트 시작")
        
        test_queries = [
            "친구와 우정",
            "자연의 동물들", 
            "용감한 모험",
            "아름다운 음악",
            "건강한 생활"
        ]
        
        for query in test_queries:
            logger.info(f"\n검색어: '{query}'")
            
            # Main DB 검색
            try:
                if self.config.use_hybrid:
                    main_results = self.main_db.hybrid_search(query, k=2)
                else:
                    main_results = self.main_db.similarity_search(query, k=2)
                logger.info(f"  📚 Main DB: {len(main_results)}개 결과")
                for i, result in enumerate(main_results):
                    title = result.metadata.get("title", "제목없음")
                    score = f"(점수: {result.final_score:.3f})" if hasattr(result, 'final_score') else ""
                    logger.info(f"    {i+1}. {title} {score}")
            except Exception as e:
                logger.error(f"  ❌ Main DB 검색 실패: {e}")
            
            # Detailed DB 검색 (아리용)
            try:
                if self.config.use_hybrid:
                    detailed_results = self.detailed_db.hybrid_search(query, k=2)
                else:
                    detailed_results = self.detailed_db.similarity_search(query, k=2)
                logger.info(f"  🎭 Detailed DB (아리용): {len(detailed_results)}개 결과")
                for i, result in enumerate(detailed_results):
                    title = result.metadata.get("title", "제목없음")
                    score = f"(점수: {result.final_score:.3f})" if hasattr(result, 'final_score') else ""
                    logger.info(f"    {i+1}. {title} {score}")
            except Exception as e:
                logger.error(f"  ❌ Detailed DB 검색 실패: {e}")
                
            # Summary DB 검색 (쫑이용)
            try:
                if self.config.use_hybrid:
                    summary_results = self.summary_db.hybrid_search(query, k=2)
                else:
                    summary_results = self.summary_db.similarity_search(query, k=2)
                logger.info(f"  🤖 Summary DB (쫑이용): {len(summary_results)}개 결과")
                for i, result in enumerate(summary_results):
                    title = result.metadata.get("title", "제목없음")
                    score = f"(점수: {result.final_score:.3f})" if hasattr(result, 'final_score') else ""
                    logger.info(f"    {i+1}. {title} {score}")
            except Exception as e:
                logger.error(f"  ❌ Summary DB 검색 실패: {e}")
    
    def get_three_tier_stats(self):
        """3-Tier DB 통계 조회"""
        logger.info("📊 3-Tier DB 통계 조회")
        
        databases = {
            "Main DB (전체 원본)": self.main_db,
            "Detailed DB (아리용)": self.detailed_db,
            "Summary DB (쫑이용)": self.summary_db
        }
        
        for db_name, db in databases.items():
            try:
                if self.config.use_hybrid:
                    stats = db.get_cache_stats()
                    logger.info(f"\n🗄️ {db_name}:")
                    logger.info(f"   캐시 통계: {stats['cache_stats']}")
                    logger.info(f"   BM25 활성화: {stats['bm25_enabled']}")
                    logger.info(f"   경로: {db.persist_directory}")
                else:
                    try:
                        count = db.vectorstore._collection.count()
                        logger.info(f"\n🗄️ {db_name}:")
                        logger.info(f"   문서 수: {count}")
                        logger.info(f"   경로: {db.persist_directory}")
                    except Exception as e:
                        logger.warning(f"   통계 조회 실패: {e}")
            except Exception as e:
                logger.error(f"{db_name} 통계 조회 실패: {e}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="JJongal-AI 3-Tier Vector Database 구성")
    parser.add_argument("--main-db", default="data/chroma_db/main", help="Main DB 경로")
    parser.add_argument("--detailed-db", default="data/chroma_db/detailed", help="Detailed DB 경로 (아리용)")
    parser.add_argument("--summary-db", default="data/chroma_db/summary", help="Summary DB 경로 (쫑이용)")
    parser.add_argument("--use-basic", action="store_true", help="기본 VectorDB 사용 (하이브리드 대신)")
    parser.add_argument("--limit", type=int, help="카테고리당 최대 처리 개수")
    parser.add_argument("--test-only", action="store_true", help="검색 테스트만 실행")
    parser.add_argument("--stats", action="store_true", help="통계만 조회")
    
    args = parser.parse_args()
    
    try:
        # 설정
        config = ThreeTierConfig(
            main_db_path=args.main_db,
            detailed_db_path=args.detailed_db,
            summary_db_path=args.summary_db,
            use_hybrid=not args.use_basic
        )
        
        populator = ThreeTierDBPopulator(config)
        
        if args.stats:
            populator.initialize_databases()
            populator.get_three_tier_stats()
            return
        
        if args.test_only:
            populator.initialize_databases()
            populator.test_three_tier_search()
            return
        
        # 전체 파이프라인 실행
        logger.info("🚀 3-Tier Vector Database 구성 시작")
        start_time = time.time()
        
        populator.populate_three_tier_dbs(limit_per_category=args.limit)
        
        processing_time = time.time() - start_time
        logger.info(f"⏱️ 총 처리 시간: {processing_time:.1f}초")
        
        # 검색 테스트
        populator.test_three_tier_search()
        
        # 통계
        populator.get_three_tier_stats()
        
        logger.info("🎉 3-Tier Vector Database 구성 완료!")
        logger.info("💡 사용법:")
        logger.info("   - Main DB: 전체 원본 데이터 검색")
        logger.info("   - Detailed DB: ChatBot B (아리) 상세 스토리 생성")
        logger.info("   - Summary DB: ChatBot A (쫑이) 빠른 요약 검색")
        
    except Exception as e:
        logger.error(f"💥 실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()