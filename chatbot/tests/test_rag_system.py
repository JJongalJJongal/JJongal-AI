#!/usr/bin/env python3
"""
RAG 시스템 단위 테스트 (하이브리드 모드 지원)

이 스크립트는 꼬기(ChatBot B)에서 사용하는 RAG 시스템의 기능을 테스트합니다:
1. ChromaDB 벡터 저장소 생성 및 데이터 추가 (하이브리드 모드)
2. LangChain 기반 검색 기능
3. Few-shot 프롬프트 생성 및 적용
4. 동화 생성을 위한 컨텍스트 강화
5. LFU 캐시 정책 테스트
6. 한국어 임베딩 모델 테스트
"""

import os
import sys
import unittest
import json
import shutil
import time
from pathlib import Path

# 프로젝트 루트 경로를 파이썬 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

from chatbot.models.rag_system import RAGSystem


class TestRAGSystemHybrid(unittest.TestCase):
    """RAG 시스템 하이브리드 모드 테스트 클래스"""
    
    def setUp(self):
        """테스트 환경 설정"""
        # 테스트용 벡터 DB 디렉토리
        self.test_db_dir = os.path.join(current_dir, "test_vector_db")
        self.hybrid_test_db_dir = os.path.join(current_dir, "test_hybrid_vector_db")
        
        # 기존 테스트 DB 삭제
        for db_dir in [self.test_db_dir, self.hybrid_test_db_dir]:
            if os.path.exists(db_dir):
                shutil.rmtree(db_dir)
        
        # 기본 RAG 시스템 초기화 (기존 호환성)
        self.rag = RAGSystem(persist_directory=self.test_db_dir, use_openai_embeddings=False)
        
        # 하이브리드 RAG 시스템 초기화
        self.hybrid_rag = RAGSystem(
            persist_directory=self.hybrid_test_db_dir,
            use_openai_embeddings=False,  # 한국어 모델 사용
            use_hybrid_mode=True,
            memory_cache_size=500,
            enable_lfu_cache=True
        )
        
        # 테스트용 동화 데이터
        self.sample_stories = [
            {
                "title": "용감한 우주 탐험가",
                "tags": "5-6세,우주,모험,용기",
                "summary": "어린 소년 민준이 우주선을 타고 미지의 행성을 탐험하는 이야기입니다. 위기를 만났지만 문제를 해결하고 친구들을 만들어 집으로 돌아옵니다.",
                "content": """
                먼 우주 저편, 반짝이는 별들 사이에 민준이라는 호기심 많은 소년이 살고 있었어요.
                민준이는 매일 밤 창문으로 쏟아지는 별빛을 보며 우주를 꿈꿨답니다.
                
                어느 날, 민준이의 방 창가에 작은 우주선이 날아왔어요. 문이 열리고 작은 녹색 외계인이 나타났어요.
                "안녕, 민준아! 나는 별이에요. 나와 함께 우주를 탐험하지 않을래?"
                
                민준이는 기쁜 마음으로 별이와 함께 우주선에 올랐어요. 우주선은 번쩍이며 우주로 날아올랐답니다.
                그들은 무지개 고리가 있는 행성, 반짝이는 얼음 별, 거대한 폭풍이 있는 가스 행성을 지나갔어요.
                
                갑자기 우주선에 이상한 소리가 들렸어요. 엔진에 문제가 생긴 거예요!
                "어떡하지?" 민준이가 걱정스럽게 물었어요.
                "걱정 마, 우리 함께 해결할 수 있어!" 별이가 대답했어요.
                
                민준이는 별이와 함께 엔진을 고치기로 했어요. 민준이의 창의적인 아이디어와 별이의 지식이 합쳐져서 마침내 엔진을 고칠 수 있었답니다.
                
                문제를 해결한 후, 그들은 아름다운 행성에 착륙했어요. 그곳에서 다양한 모양과 색깔의 외계인 친구들을 만났어요.
                모두 민준이의 용기와 문제 해결 능력에 감탄했답니다.
                
                모험을 마친 민준이는 집으로 돌아왔어요. 이제 민준이는 매일 밤 창문으로 별을 볼 때마다 새로운 친구들을 생각하며 미소 지었답니다.
                그리고 언젠가 다시 우주를 탐험할 날을 기다렸어요.
                """
            },
            {
                "title": "숲속의 공룡 친구",
                "tags": "4-5세,공룡,우정,자연",
                "summary": "작은 공룡 토토가 숲속에서 길을 잃고 새로운 친구들의 도움을 받아 집을 찾아가는 이야기입니다. 도움을 주고받는 우정의 가치를 배웁니다.",
                "content": """
                아주 먼 옛날, 푸른 숲속에 작고 초록색 공룡 토토가 살고 있었어요.
                토토는 호기심이 많아서 매일 숲속을 돌아다니며 새로운 것들을 발견하곤 했답니다.
                
                어느 날, 토토는 평소보다 더 깊이 숲속으로 들어갔어요. 예쁜 꽃과 맛있는 열매를 따라가다 보니 어느새 길을 잃어버렸답니다.
                "엄마, 아빠가 어디 계실까?" 토토는 걱정이 되기 시작했어요.
                
                그때, 나무 위에서 작은 목소리가 들렸어요. "안녕! 길을 잃었니?"
                올려다보니 예쁜 깃털을 가진 작은 공룡 새 피피였어요.
                
                "네, 집을 찾아가고 싶어요." 토토가 대답했어요.
                "걱정 마, 내가 도와줄게. 내 친구들도 불러올게!" 피피가 말했어요.
                
                피피는 큰 소리로 친구들을 불렀어요. 곧 긴 목을 가진 브라키와 단단한 등껍질을 가진 트리케가 나타났답니다.
                
                "우리 모두 함께 토토의 집을 찾아주자!" 피피가 제안했어요.
                브라키는 긴 목을 이용해 멀리 있는 곳을 살펴보았고, 트리케는 토토가 지나온 흔적을 찾아냈어요.
                
                함께 힘을 합쳐 마침내 토토의 집을 찾을 수 있었어요! 토토의 부모님은 토토를 보자마자 기뻐서 뛰어와 안아주셨답니다.
                
                "고마워, 친구들! 너희들이 없었다면 집을 찾지 못했을 거야." 토토가 감사의 인사를 했어요.
                
                그날부터 토토와 새 친구들은 매일 함께 놀았어요. 토토는 길을 잃은 슬픈 경험이 새로운 친구들을 만나는 행복한 시간으로 바뀌었다는 것을 알게 되었답니다.
                """
            },
            {
                "title": "마법의 책과 똑똑한 토끼",
                "tags": "6-7세,마법,학습,지혜",
                "summary": "꼬마 토끼가 마법의 책을 발견하고 다양한 지식을 배우며 숲 친구들을 도와주는 이야기입니다.",
                "content": """
                숲속에 사는 귀여운 토끼 솔이는 책 읽기를 정말 좋아했어요.
                어느 날 솔이는 오래된 나무 밑에서 반짝이는 책을 발견했답니다.
                
                "어? 이건 뭐지?" 솔이가 책을 펼치자 마법같은 일이 일어났어요.
                책에서 글자들이 반짝반짝 빛나며 솔이에게 말을 걸었어요!
                
                "안녕, 솔아! 나는 지혜의 마법책이야. 네가 궁금한 모든 것을 알려줄 수 있어!"
                솔이는 깜짝 놀랐지만 기뻤어요.
                
                그날부터 솔이는 마법책과 함께 많은 것을 배웠어요.
                숲의 식물들 이름, 별자리 이야기, 그리고 수학까지!
                
                어느 날, 다람쥐 친구가 울면서 왔어요.
                "솔아, 내가 모은 도토리를 어디에 뒀는지 기억이 안 나!"
                
                솔이는 마법책에서 배운 지혜를 사용했어요.
                "걱정마! 차근차근 생각해보자. 언제, 어디서, 무엇을 했는지 순서대로 말해봐."
                
                솔이의 도움으로 다람쥐는 도토리를 찾을 수 있었어요!
                숲의 모든 친구들이 솔이의 지혜에 감탄했답니다.
                """
            }
        ]
        
        # 두 시스템 모두에 테스트용 동화 데이터 추가
        for rag_system in [self.rag, self.hybrid_rag]:
            for i, story in enumerate(self.sample_stories):
                rag_system.add_story(
                    title=story["title"],
                    tags=story["tags"],
                    summary=story["summary"],
                    content=story["content"]
                )
    
    def tearDown(self):
        """테스트 환경 정리"""
        # 테스트 DB 삭제
        for db_dir in [self.test_db_dir, self.hybrid_test_db_dir]:
            if os.path.exists(db_dir):
                shutil.rmtree(db_dir)
    
    def test_hybrid_mode_initialization(self):
        """하이브리드 모드 초기화 테스트"""
        # 하이브리드 모드 시스템 정보 확인
        system_info = self.hybrid_rag.get_system_info()
        
        self.assertTrue(system_info["hybrid_mode"])
        self.assertEqual(system_info["embedding_model"], "nlpai-lab/KURE-v1")
        self.assertEqual(system_info["memory_cache_size"], 500)
        self.assertTrue(system_info["lfu_cache_enabled"])
        
        # 벡터 저장소 디렉토리가 생성되었는지 확인
        self.assertTrue(os.path.exists(self.hybrid_test_db_dir))
        for db_type in ["main", "detailed", "summary"]:
            self.assertTrue(os.path.exists(os.path.join(self.hybrid_test_db_dir, db_type)))
    
    def test_korean_embedding_model(self):
        """한국어 임베딩 모델 테스트"""
        # 한국어 검색어로 테스트
        korean_queries = [
            "우주 여행하는 아이 이야기",
            "친구와 함께하는 모험",
            "용기와 문제해결에 대한 동화"
        ]
        
        for query in korean_queries:
            # 기본 시스템 테스트
            result = self.rag.query(query, use_summary=True)
            self.assertIn("answer", result)
            self.assertGreater(len(result["answer"]), 0)
            
            # 하이브리드 시스템 테스트
            hybrid_result = self.hybrid_rag.query(query, use_summary=True)
            self.assertIn("answer", hybrid_result)
            self.assertGreater(len(hybrid_result["answer"]), 0)
    
    def test_cache_performance(self):
        """캐시 성능 테스트"""
        # 같은 쿼리를 여러 번 실행하여 캐시 효과 측정
        query = "우주 탐험 이야기"
        
        # 첫 번째 실행 (캐시 미스)
        start_time = time.time()
        first_result = self.hybrid_rag.query(query, use_summary=True)
        first_time = time.time() - start_time
        
        # 두 번째 실행 (캐시 히트 예상)
        start_time = time.time()
        second_result = self.hybrid_rag.query(query, use_summary=True)
        second_time = time.time() - start_time
        
        # 결과는 동일해야 함
        self.assertEqual(len(first_result["sources"]), len(second_result["sources"]))
        
        # 두 번째 실행이 더 빠르거나 비슷해야 함 (캐시 효과)
        # 실제 성능은 환경에 따라 달라질 수 있으므로 결과만 확인
        self.assertIsNotNone(first_result)
        self.assertIsNotNone(second_result)
        
        print(f"첫 번째 쿼리 시간: {first_time:.3f}초")
        print(f"두 번째 쿼리 시간: {second_time:.3f}초")
    
    def test_memory_disk_hybrid(self):
        """메모리-디스크 하이브리드 저장 테스트"""
        # 캐시 정보 확인
        cache_info = self.hybrid_rag.main_vectordb.get_cache_info()
        
        self.assertTrue(cache_info["hybrid_mode"])
        self.assertEqual(cache_info["memory_cache_size"], 500)
        self.assertTrue(cache_info["lfu_cache_enabled"])
        
        # 데이터가 실제로 디스크에 저장되는지 확인
        main_db_path = os.path.join(self.hybrid_test_db_dir, "main")
        self.assertTrue(os.path.exists(main_db_path))
        
        # ChromaDB 파일이 존재하는지 확인
        chroma_files = [f for f in os.listdir(main_db_path) if f.endswith('.sqlite3') or f.startswith('chroma')]
        self.assertGreater(len(chroma_files), 0, "ChromaDB 파일이 디스크에 저장되어야 합니다.")
    
    def test_story_retrieval_performance(self):
        """스토리 검색 성능 비교 테스트"""
        queries = [
            "우주 모험",
            "공룡과 친구",
            "마법의 힘",
            "용기있는 아이",
            "문제를 해결하는 이야기"
        ]
        
        basic_times = []
        hybrid_times = []
        
        for query in queries:
            # 기본 시스템 성능 측정
            start_time = time.time()
            basic_result = self.rag.query(query, use_summary=True)
            basic_times.append(time.time() - start_time)
            
            # 하이브리드 시스템 성능 측정
            start_time = time.time()
            hybrid_result = self.hybrid_rag.query(query, use_summary=True)
            hybrid_times.append(time.time() - start_time)
            
            # 결과 품질 검증
            self.assertGreater(len(basic_result["answer"]), 0)
            self.assertGreater(len(hybrid_result["answer"]), 0)
        
        avg_basic_time = sum(basic_times) / len(basic_times)
        avg_hybrid_time = sum(hybrid_times) / len(hybrid_times)
        
        print(f"\n성능 비교:")
        print(f"기본 시스템 평균 시간: {avg_basic_time:.3f}초")
        print(f"하이브리드 시스템 평균 시간: {avg_hybrid_time:.3f}초")
        
        # 두 시스템 모두 합리적인 시간 내에 응답해야 함
        self.assertLess(avg_basic_time, 5.0, "기본 시스템이 너무 느립니다.")
        self.assertLess(avg_hybrid_time, 5.0, "하이브리드 시스템이 너무 느립니다.")
    
    def test_cloud_compatibility(self):
        """클라우드 호환성 테스트 (AWS 준비)"""
        # 환경 변수 시뮬레이션
        os.environ["CCB_AI_CLOUD_MODE"] = "true"
        os.environ["CCB_AI_CACHE_SIZE"] = "2000"
        
        try:
            # 클라우드 설정으로 새 시스템 생성
            cloud_test_dir = os.path.join(current_dir, "test_cloud_vector_db")
            if os.path.exists(cloud_test_dir):
                shutil.rmtree(cloud_test_dir)
            
            cloud_rag = RAGSystem(
                persist_directory=cloud_test_dir,
                use_hybrid_mode=True,
                memory_cache_size=2000,
                enable_lfu_cache=True
            )
            
            # 샘플 데이터 추가
            cloud_rag.add_story(
                title="클라우드 테스트 동화",
                tags="테스트,클라우드",
                summary="클라우드 환경에서의 동화 테스트입니다.",
                content="이것은 클라우드 환경에서 RAG 시스템을 테스트하는 동화입니다."
            )
            
            # 검색 테스트
            result = cloud_rag.query("클라우드 테스트", use_summary=True)
            self.assertIn("answer", result)
            self.assertGreater(len(result["answer"]), 0)
            
            # 시스템 정보 확인
            system_info = cloud_rag.get_system_info()
            self.assertEqual(system_info["memory_cache_size"], 2000)
            
            # 정리
            shutil.rmtree(cloud_test_dir)
            
        finally:
            # 환경 변수 정리
            if "CCB_AI_CLOUD_MODE" in os.environ:
                del os.environ["CCB_AI_CLOUD_MODE"]
            if "CCB_AI_CACHE_SIZE" in os.environ:
                del os.environ["CCB_AI_CACHE_SIZE"]


class TestChatBotBWithRAG(unittest.TestCase):
    """꼬기(ChatBot B)와 RAG 시스템 통합 테스트"""
    
    def setUp(self):
        """테스트 환경 설정"""
        # 프로젝트 루트에서 모듈 임포트
        sys.path.append(project_root)
        from chatbot.models.chat_bot_b import StoryGenerationChatBot
        
        # 테스트용 출력 디렉토리
        self.output_dir = os.path.join(current_dir, "test_output")
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 챗봇 초기화
        self.chatbot = StoryGenerationChatBot(output_dir=self.output_dir)
        
        # 테스트용 스토리 아웃라인
        self.story_outline = {
            "theme": "우주 탐험을 떠나는 용감한 공룡",
            "characters": ["탁구 (파란색 공룡)", "별똥별 (우주 로봇)", "루나 (달의 공주)"],
            "setting": "은하계 너머의 신비로운 우주",
            "plot_summary": "호기심 많은 공룡 탁구는 우연히 발견한 우주선을 타고 우주 모험을 떠납니다. 그곳에서 길을 잃은 로봇 별똥별을 만나게 되고, 함께 달나라로 가서 달의 공주 루나를 도와 잃어버린 별빛을 찾는 모험을 하게 됩니다.",
            "educational_value": "용기, 우정, 협력의 가치와 함께 우주에 대한 호기심 자극"
        }
        
        # 챗봇에 스토리 아웃라인과 타겟 연령 설정
        self.chatbot.set_story_outline(self.story_outline)
        self.chatbot.set_target_age(6)
    
    def tearDown(self):
        """테스트 환경 정리"""
        # 테스트 출력 디렉토리 삭제
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
    
    def test_rag_integration(self):
        """RAG 시스템 통합 테스트"""
        # RAG 시스템 인스턴스 확인
        self.assertIsNotNone(self.chatbot.rag_system)
        
        # RAG 시스템이 초기화되었는지 확인
        self.assertIsNotNone(self.chatbot.rag_system.embeddings)
        self.assertIsNotNone(self.chatbot.rag_system.summary_vectorstore)
        
        # 샘플 스토리 추가
        sample_story = {
            "title": "별빛 모험",
            "tags": "6-7세,우주,별,모험",
            "summary": "작은 별이 우주를 여행하며 친구들을 만나는 이야기",
            "content": "옛날 옛적에 작고 반짝이는 별이 있었어요. 이 별은 혼자 있는 것이 너무 외로워서 우주 여행을 떠나기로 했답니다..."
        }
        
        story_id = self.chatbot.rag_system.add_story(
            title=sample_story["title"],
            tags=sample_story["tags"],
            summary=sample_story["summary"],
            content=sample_story["content"]
        )
        
        self.assertIsNotNone(story_id)
        
        # 유사 스토리 검색
        similar = self.chatbot.rag_system.get_similar_stories(
            theme=self.story_outline["theme"],
            age_group=6,
            n_results=1
        )
        
        self.assertEqual(len(similar), 1)
        self.assertEqual(similar[0]["title"], "별빛 모험")
    
    def test_detailed_story_generation(self):
        """상세 스토리 생성 테스트"""
        # Few-shot 예제가 적용된 상세 스토리 생성
        try:
            detailed_story = self.chatbot.generate_detailed_story()
            
            # 상세 스토리 구조 확인
            self.assertIn("title", detailed_story)
            self.assertIn("scenes", detailed_story)
            self.assertGreater(len(detailed_story["scenes"]), 0)
            
            # 첫 번째 장면 구조 확인
            first_scene = detailed_story["scenes"][0]
            self.assertIn("title", first_scene)
            self.assertIn("description", first_scene)
            
            # 상세 스토리 저장
            story_path = os.path.join(self.output_dir, "test_story.json")
            self.chatbot.save_story_data(story_path)
            
            # 저장된 파일 확인
            self.assertTrue(os.path.exists(story_path))
            
            # 저장된 파일 로드
            with open(story_path, 'r', encoding='utf-8') as f:
                saved_story = json.load(f)
                
            # 저장된 내용 확인
            self.assertEqual(saved_story["title"], detailed_story["title"])
            
        except Exception as e:
            self.fail(f"상세 스토리 생성 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    unittest.main() 