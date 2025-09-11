#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LFU + Hybrid ChromaDB 데이터 파이프라인 - JJongal-AI

680개의 동화 JSON 파일을 효율적으로 처리하여
카테고리별 멀티 컬렉션 LFU Hybrid ChromaDB에 저장하는 파이프라인

Features:
- 배치 처리로 대용량 데이터 효율적 처리
- 5개 누리과정 카테고리별 자동 분류
- 메타데이터 정규화 및 인덱싱
- 진행률 표시 및 오류 복구
- 성능 최적화 및 모니터링
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import argparse

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.vector_db.core import LFUHybridVectorDB, SearchWeights
from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

@dataclass
class ProcessingStats:
    """처리 통계"""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_documents: int = 0
    processing_time: float = 0.0
    categories: Dict[str, int] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = {
                '의사소통': 0,
                '자연탐구': 0,
                '사회관계': 0,
                '예술경험': 0,
                '신체운동_건강': 0,
                '기타': 0
            }

class CategoryClassifier:
    """동화 카테고리 자동 분류기"""
    
    CATEGORY_KEYWORDS = {
        '의사소통': [
            '말하기', '듣기', '읽기', '쓰기', '언어', '대화', '소통', '표현',
            '이야기', '말했어요', '말해', '들었어', '물어봤어', '대답했어'
        ],
        '자연탐구': [
            '자연', '과학', '수학', '탐구', '실험', '관찰', '숫자', '계산',
            '동물', '식물', '곤충', '물', '하늘', '바다', '산', '나무', '꽃'
        ],
        '사회관계': [
            '친구', '가족', '엄마', '아빠', '형', '누나', '동생', '할머니', '할아버지',
            '함께', '도와', '배려', '나누어', '협력', '우정', '사랑', '가족'
        ],
        '예술경험': [
            '음악', '미술', '그림', '노래', '춤', '악기', '색깔', '예쁜', '아름다운',
            '연주', '그렸어', '만들었어', '꾸몄어', '색칠', '만들기'
        ],
        '신체운동_건강': [
            '운동', '건강', '뛰어', '달렸어', '점프', '놀이', '게임', '체조',
            '몸', '힘', '튼튼', '깨끗', '씻기', '양치', '건강한', '활동'
        ]
    }
    
    @classmethod
    def classify_story(cls, story_data: Dict[str, Any]) -> str:
        """동화 내용을 분석하여 카테고리 분류"""
        
        # 1. tags에서 명시적 카테고리 확인
        tags = story_data.get('tags', '').lower()
        for category in cls.CATEGORY_KEYWORDS.keys():
            if category in tags:
                return category
        
        # 2. 내용 기반 분류
        content = f"{story_data.get('title', '')} {story_data.get('content', '')} {story_data.get('summary', '')}"
        content = content.lower()
        
        category_scores = {}
        
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in content)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            # 가장 높은 점수의 카테고리 반환
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return '기타'

class StoryDataProcessor:
    """동화 데이터 처리기"""
    
    def __init__(
        self,
        data_dir: str = "data/processed/story_data",
        base_persist_dir: str = "data/chroma_db_hybrid",
        batch_size: int = 50,
        max_workers: int = 4
    ):
        self.data_dir = Path(data_dir)
        self.base_persist_dir = Path(base_persist_dir)
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # 카테고리별 VectorDB 인스턴스
        self.vector_dbs: Dict[str, LFUHybridVectorDB] = {}
        self.stats = ProcessingStats()
        
        # 초기화
        self._ensure_directories()
        self._initialize_vector_dbs()
    
    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        self.base_persist_dir.mkdir(parents=True, exist_ok=True)
        for category in CategoryClassifier.CATEGORY_KEYWORDS.keys():
            category_dir = self.base_persist_dir / category
            category_dir.mkdir(exist_ok=True)
    
    def _initialize_vector_dbs(self):
        """카테고리별 VectorDB 초기화"""
        search_weights = SearchWeights(
            vector_weight=0.5,    # 벡터 검색
            bm25_weight=0.3,      # 키워드 검색  
            metadata_weight=0.2   # 메타데이터
        )
        
        categories = list(CategoryClassifier.CATEGORY_KEYWORDS.keys()) + ['기타']
        
        # 영문 컬렉션 이름 매핑
        english_names = {
            '의사소통': 'communication',
            '자연탐구': 'nature_exploration', 
            '사회관계': 'social_relations',
            '예술경험': 'art_experience',
            '신체운동_건강': 'physical_health',
            '기타': 'others'
        }
        
        for category in categories:
            persist_dir = str(self.base_persist_dir / category)
            english_name = english_names.get(category, category)
            collection_name = f"fairy_tales_{english_name}"
            
            self.vector_dbs[category] = LFUHybridVectorDB(
                persist_directory=persist_dir,
                collection_name=collection_name,
                use_openai=False,  # KURE-v1 한국어 모델 사용
                memory_cache_size=500,  # 카테고리별로 작은 캐시
                disk_cache_size=2000,
                enable_bm25=True,
                search_weights=search_weights
            )
            
            logger.info(f"✅ VectorDB initialized for category: {category} (collection: {collection_name})")
    
    def load_story_files(self) -> List[Path]:
        """동화 JSON 파일 목록 로드"""
        json_files = list(self.data_dir.glob("*.json"))
        logger.info(f"📁 Found {len(json_files)} story files to process")
        return json_files
    
    def process_single_story(self, file_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
        """단일 동화 파일 처리"""
        try:
            # JSON 파일 로드
            with open(file_path, 'r', encoding='utf-8') as f:
                story_data = json.load(f)
            
            # 필수 필드 확인
            if not story_data.get('content') or not story_data.get('title'):
                return False, f"Missing required fields: {file_path.name}", {}
            
            # 카테고리 분류
            category = CategoryClassifier.classify_story(story_data)
            
            # 문서 텍스트 준비 (제목 + 내용 + 요약)
            document_text = f"{story_data['title']}\n\n{story_data['content']}"
            if story_data.get('summary'):
                document_text += f"\n\n요약: {story_data['summary']}"
            
            # 메타데이터 정규화
            metadata = {
                'story_id': story_data.get('story_id', file_path.stem),
                'title': story_data['title'],
                'category': category,
                'tags': story_data.get('tags', ''),
                'author': story_data.get('metadata', {}).get('author', ''),
                'publisher': story_data.get('metadata', {}).get('publisher', ''),
                'isbn': story_data.get('metadata', {}).get('isbn', ''),
                'published_year': story_data.get('metadata', {}).get('published_year', 0),
                'image_count': len(story_data.get('images', [])),
                'content_length': len(story_data['content']),
                'processed_date': datetime.now().isoformat(),
                'source_file': file_path.name
            }
            
            return True, category, {
                'text': document_text,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
            return False, str(e), {}
    
    def process_batch(self, file_batch: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
        """배치 단위 파일 처리"""
        batch_results = {category: [] for category in self.vector_dbs.keys()}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 병렬 처리 시작
            future_to_file = {
                executor.submit(self.process_single_story, file_path): file_path 
                for file_path in file_batch
            }
            
            # 결과 수집
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    success, result, data = future.result()
                    
                    if success:
                        category = result
                        batch_results[category].append(data)
                        self.stats.processed_files += 1
                        self.stats.categories[category] += 1
                    else:
                        self.stats.failed_files += 1
                        logger.warning(f"⚠️ Failed to process {file_path.name}: {result}")
                        
                except Exception as e:
                    self.stats.failed_files += 1
                    logger.error(f"❌ Exception processing {file_path.name}: {e}")
        
        return batch_results
    
    def store_batch_to_vector_db(self, batch_results: Dict[str, List[Dict[str, Any]]]):
        """배치 결과를 VectorDB에 저장"""
        
        for category, documents in batch_results.items():
            if not documents:
                continue
            
            try:
                vector_db = self.vector_dbs[category]
                
                # 텍스트와 메타데이터 분리
                texts = [doc['text'] for doc in documents]
                metadatas = [doc['metadata'] for doc in documents]
                
                # VectorDB에 추가 (BM25 인덱스도 함께 업데이트)
                start_time = time.time()
                ids = vector_db.add_documents_with_bm25(
                    texts=texts,
                    metadatas=metadatas,
                    update_bm25=True
                )
                store_time = time.time() - start_time
                
                self.stats.total_documents += len(documents)
                
                logger.info(
                    f"✅ Stored {len(documents)} documents in {category} "
                    f"({store_time:.2f}s, {len(documents)/store_time:.1f} docs/s)"
                )
                
            except Exception as e:
                logger.error(f"❌ Failed to store batch for {category}: {e}")
                raise
    
    def run_pipeline(self, resume_from: int = 0) -> ProcessingStats:
        """전체 파이프라인 실행"""
        
        logger.info("🚀 Starting LFU + Hybrid ChromaDB Data Pipeline")
        pipeline_start = time.time()
        
        # 파일 목록 로드
        story_files = self.load_story_files()
        self.stats.total_files = len(story_files)
        
        if resume_from > 0:
            story_files = story_files[resume_from:]
            logger.info(f"📍 Resuming from file #{resume_from}")
        
        # 배치 단위로 처리
        total_batches = (len(story_files) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(story_files), self.batch_size):
            current_batch = batch_idx // self.batch_size + 1
            file_batch = story_files[batch_idx:batch_idx + self.batch_size]
            
            logger.info(f"📦 Processing batch {current_batch}/{total_batches} ({len(file_batch)} files)")
            
            batch_start = time.time()
            
            # 1. 파일 처리
            batch_results = self.process_batch(file_batch)
            
            # 2. VectorDB 저장
            self.store_batch_to_vector_db(batch_results)
            
            batch_time = time.time() - batch_start
            progress = (current_batch / total_batches) * 100
            
            logger.info(
                f"⏱️ Batch {current_batch} completed in {batch_time:.2f}s "
                f"(Progress: {progress:.1f}%)"
            )
            
            # 진행률 표시
            if current_batch % 5 == 0 or current_batch == total_batches:
                self._print_progress_summary(current_batch, total_batches)
        
        # 파이프라인 완료
        self.stats.processing_time = time.time() - pipeline_start
        
        logger.info("✅ Pipeline completed! Generating final report...")
        return self._generate_final_report()
    
    def _print_progress_summary(self, current_batch: int, total_batches: int):
        """진행률 요약 출력"""
        logger.info("=" * 60)
        logger.info(f"📊 Progress Summary (Batch {current_batch}/{total_batches})")
        logger.info(f"   Processed Files: {self.stats.processed_files}")
        logger.info(f"   Failed Files: {self.stats.failed_files}")
        logger.info(f"   Total Documents: {self.stats.total_documents}")
        
        logger.info("   Category Distribution:")
        for category, count in self.stats.categories.items():
            if count > 0:
                logger.info(f"     {category}: {count}")
        logger.info("=" * 60)
    
    def _generate_final_report(self) -> ProcessingStats:
        """최종 처리 보고서 생성"""
        logger.info("🎯 " + "=" * 50)
        logger.info("🎯 LFU + HYBRID CHROMADB PIPELINE COMPLETED")
        logger.info("🎯 " + "=" * 50)
        
        logger.info(f"📈 Total Processing Time: {self.stats.processing_time:.2f} seconds")
        logger.info(f"📁 Total Files: {self.stats.total_files}")
        logger.info(f"✅ Successfully Processed: {self.stats.processed_files}")
        logger.info(f"❌ Failed: {self.stats.failed_files}")
        logger.info(f"📝 Total Documents Created: {self.stats.total_documents}")
        logger.info(f"⚡ Processing Speed: {self.stats.processed_files / (self.stats.processing_time/60):.1f} files/min")
        
        logger.info("\n📊 Category Distribution:")
        for category, count in self.stats.categories.items():
            if count > 0:
                percentage = (count / self.stats.total_documents) * 100
                logger.info(f"   {category}: {count} documents ({percentage:.1f}%)")
        
        # VectorDB 통계
        logger.info("\n🔍 VectorDB Statistics:")
        for category, vector_db in self.vector_dbs.items():
            if self.stats.categories.get(category, 0) > 0:
                cache_stats = vector_db.get_cache_stats()
                logger.info(f"   {category}:")
                logger.info(f"     Documents: {self.stats.categories.get(category, 0)}")
                logger.info(f"     BM25 Corpus: {cache_stats['bm25_corpus_size']}")
                logger.info(f"     Cache Size: {cache_stats['cache_stats']['cache_size']}")
        
        logger.info("🎯 " + "=" * 50)
        
        return self.stats
    
    def test_hybrid_search(self, test_queries: List[str] = None):
        """하이브리드 검색 테스트"""
        if not test_queries:
            test_queries = [
                "친구와 함께하는 이야기",
                "자연을 탐구하는 동화", 
                "음악과 예술 이야기",
                "건강한 몸 만들기",
                "말하기와 듣기"
            ]
        
        logger.info("🔍 Testing Hybrid Search Performance...")
        
        for query in test_queries:
            logger.info(f"\n검색어: '{query}'")
            
            for category, vector_db in self.vector_dbs.items():
                if self.stats.categories.get(category, 0) == 0:
                    continue
                
                try:
                    start_time = time.time()
                    results = vector_db.hybrid_search(query, k=3, use_cache=True)
                    search_time = time.time() - start_time
                    
                    logger.info(f"  {category}: {len(results)} results ({search_time*1000:.1f}ms)")
                    
                    for i, result in enumerate(results[:2], 1):
                        title = result.metadata.get('title', 'No title')
                        score = result.final_score
                        logger.info(f"    {i}. {title} (score: {score:.3f})")
                        
                except Exception as e:
                    logger.error(f"  {category}: Search failed - {e}")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="LFU + Hybrid ChromaDB Data Pipeline")
    parser.add_argument("--data-dir", default="data/processed/story_data", 
                       help="Story data directory")
    parser.add_argument("--persist-dir", default="data/chroma_db_hybrid",
                       help="ChromaDB persistence directory")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for processing")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Max worker threads")
    parser.add_argument("--resume-from", type=int, default=0,
                       help="Resume processing from file index")
    parser.add_argument("--test-search", action="store_true",
                       help="Run search tests after processing")
    
    args = parser.parse_args()
    
    try:
        # 프로세서 초기화
        processor = StoryDataProcessor(
            data_dir=args.data_dir,
            base_persist_dir=args.persist_dir,
            batch_size=args.batch_size,
            max_workers=args.max_workers
        )
        
        # 파이프라인 실행
        stats = processor.run_pipeline(resume_from=args.resume_from)
        
        # 검색 테스트 (옵션)
        if args.test_search:
            processor.test_hybrid_search()
        
        logger.info("✅ Pipeline completed successfully!")
        
        # 성공적으로 처리된 파일 비율
        success_rate = (stats.processed_files / stats.total_files) * 100
        logger.info(f"📊 Success Rate: {success_rate:.1f}%")
        
        if success_rate < 95:
            logger.warning("⚠️ Success rate below 95% - check failed files")
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()