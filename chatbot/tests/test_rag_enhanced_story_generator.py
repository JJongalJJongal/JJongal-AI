#!/usr/bin/env python3
"""
RAG 향상된 스토리 생성기 단위 테스트

이 스크립트는 RAG와 Few-shot 학습을 통합한 향상된 스토리 생성 클래스를 테스트합니다:
1. 샘플 동화 데이터 추가 및 RAG 시스템 연동
2. Few-shot 예제를 활용한 상세 스토리 생성
3. 연령별 맞춤형 프롬프트 적용
4. 파싱 및 저장 기능
"""

import os
import sys
import unittest
import json
import shutil
from pathlib import Path

# 프로젝트 루트 경로를 파이썬 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

from CCB_AI.chatbot.models.rag_system import RAGSystem
from CCB_AI.chatbot.models.rag_enhanced_story_generator import RAGEnhancedStoryGenerator


class TestRAGEnhancedStoryGenerator(unittest.TestCase):
    """RAG 향상된 스토리 생성기 테스트 클래스"""
    
    def setUp(self):
        """테스트 환경 설정"""
        # 테스트용 벡터 DB 디렉토리
        self.test_db_dir = os.path.join(current_dir, "test_rag_db")
        
        # 기존 테스트 DB 삭제
        if os.path.exists(self.test_db_dir):
            shutil.rmtree(self.test_db_dir)
        
        # RAG 시스템 초기화
        self.rag = RAGSystem(persist_directory=self.test_db_dir)
        
        # 테스트용 출력 디렉토리
        self.output_dir = os.path.join(current_dir, "test_stories_output")
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 스토리 생성기 초기화
        self.story_generator = RAGEnhancedStoryGenerator(
            output_dir=self.output_dir,
            rag_system=self.rag
        )
        
        # 테스트용 동화 데이터
        self.sample_stories = [
            {
                "title": "우주 모험가 별이",
                "tags": "5-6세,우주,모험,우정",
                "summary": "호기심 많은 아이 별이가 우주 모험을 떠나 새로운 친구들을 만나는 이야기",
                "content": """
                반짝이는 별들이 가득한 밤하늘을 바라보며 별이는 항상 궁금했어요.
                "저 별들 너머에는 어떤 세계가 있을까?"
                
                어느 날 밤, 별이의 방 창문으로 작은 우주선이 날아왔어요.
                우주선에서는 초록색 피부를 가진 귀여운 외계인 찌찌가 나왔어요.
                
                "안녕, 별이야! 나와 함께 우주 여행을 떠나지 않을래?" 찌찌가 물었어요.
                별이는 너무 신이 나서 바로 찌찌의 우주선에 올랐어요.
                
                별이와 찌찌는 반짝이는 별들 사이를 지나 다양한 행성들을 방문했어요.
                첫 번째로 방문한 행성은 온통 사탕으로 만들어진 달콤한 행성이었어요.
                두 번째 행성은 말하는 꽃들이 가득한 아름다운 정원 행성이었죠.
                
                여행을 하던 중 우주선에 문제가 생겼어요. 별이와 찌찌는 가까운 행성에 착륙했어요.
                그곳에서 로봇 친구 삐삐를 만났어요. 삐삐는 우주선을 고치는 것을 도와주었어요.
                
                "고마워, 삐삐야! 우리와 함께 여행할래?" 별이가 물었어요.
                삐삐는 기쁘게 동의했고, 이제 셋이서 더 즐거운 우주 여행을 계속했어요.
                
                마지막으로 그들은 별이의 집으로 돌아왔어요. 별이는 이제 밤하늘을 볼 때마다
                자신이 방문했던 놀라운 행성들과 만난 새 친구들을 떠올리며 미소지었답니다.
                
                별이는 친구들에게 약속했어요. "다음에 또 우주 여행을 함께 하자!"
                """
            },
            {
                "title": "숲속의 작은 영웅",
                "tags": "4-5세,동물,용기,친구",
                "summary": "작은 토끼 토토가 용기를 내어 숲속 친구들을 위험에서 구하는 모험",
                "content": """
                푸른 숲속, 작은 토끼 토토가 살고 있었어요.
                토토는 다른 토끼들보다 작아서 항상 겁이 많았어요.
                
                어느 날, 숲속에 큰 폭풍이 올 거라는 소식이 전해졌어요.
                모든 동물들이 안전한 곳으로 대피하기 시작했죠.
                
                토토는 대피하던 중 작은 다람쥐 가족이 집에 갇혔다는 소식을 들었어요.
                "누가 다람쥐 가족을 구해줄 수 있을까?" 모두들 걱정했어요.
                
                토토는 무서웠지만 용기를 내기로 했어요.
                "내가 가볼게요!" 토토가 말했어요.
                
                토토는 위험한 길을 지나 다람쥐 가족의 집에 도착했어요.
                "걱정마세요! 제가 안전한 곳으로 안내해 드릴게요."
                
                토토는 다람쥐 가족에게 숨겨진 안전한 길을 알려주었고,
                모두 함께 대피소에 무사히 도착했어요.
                
                폭풍이 지나간 후, 모든 동물들은 토토를 숲속의 영웅으로 칭찬했어요.
                토토는 비록 몸은 작아도 커다란 용기를 가진 토끼라는 것을 알게 되었답니다.
                
                그 후로 토토는 자신감을 갖게 되었고, 숲속의 작은 영웅으로 행복하게 살았답니다.
                """
            }
        ]
        
        # 테스트용 동화 데이터 추가
        for story in self.sample_stories:
            self.story_generator.add_sample_story_to_rag(story)
        
        # 테스트용 스토리 아웃라인
        self.story_outline = {
            "theme": "우주와 공룡 친구들의 모험",
            "characters": ["타키 (파란 공룡)", "별똥별 (반짝이는 별)", "루나 (달의 공주)"],
            "setting": "별들이 빛나는 우주 공간",
            "plot_summary": "호기심 많은 공룡 타키가 별똥별을 따라 우주로 모험을 떠납니다. 그곳에서 달의 공주 루나를 만나 잃어버린 달빛을 찾는 여정을 함께합니다.",
            "educational_value": "우주에 대한 호기심과 친구와의 협력의 중요성"
        }
    
    def tearDown(self):
        """테스트 환경 정리"""
        # 테스트 DB 삭제
        if os.path.exists(self.test_db_dir):
            shutil.rmtree(self.test_db_dir)
        
        # 테스트 출력 디렉토리 삭제
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
    
    def test_initialization(self):
        """초기화 테스트"""
        # 스토리 생성기 인스턴스 확인
        self.assertIsNotNone(self.story_generator)
        self.assertIsInstance(self.story_generator, RAGEnhancedStoryGenerator)
        
        # RAG 시스템 인스턴스 확인
        self.assertIsNotNone(self.story_generator.rag_system)
        self.assertIsInstance(self.story_generator.rag_system, RAGSystem)
        
        # 프롬프트 템플릿 로드 확인
        self.assertIsNotNone(self.story_generator.prompts)
        self.assertIn("system", self.story_generator.prompts)
        self.assertIn("detailed_story_generation", self.story_generator.prompts)
    
    def test_set_story_outline(self):
        """스토리 아웃라인 설정 테스트"""
        # 스토리 아웃라인 설정
        self.story_generator.set_story_outline(self.story_outline)
        
        # 설정된 아웃라인 확인
        self.assertEqual(self.story_generator.story_outline, self.story_outline)
        self.assertEqual(self.story_generator.story_outline["theme"], "우주와 공룡 친구들의 모험")
        self.assertEqual(len(self.story_generator.story_outline["characters"]), 3)
    
    def test_set_target_age(self):
        """타겟 연령 설정 테스트"""
        # 스토리 아웃라인 설정
        self.story_generator.set_story_outline(self.story_outline)
        
        # 타겟 연령 설정
        self.story_generator.set_target_age(5)
        
        # 설정된 타겟 연령 확인
        self.assertEqual(self.story_generator.target_age, 5)
        
        # 스토리 아웃라인에도 추가되었는지 확인
        self.assertEqual(self.story_generator.story_outline["target_age"], 5)
    
    def test_get_age_group(self):
        """연령대 그룹 반환 테스트"""
        # 연령별 그룹 테스트
        test_cases = [
            (4, "4-5"),
            (5, "4-5"),
            (6, "6-7"),
            (7, "6-7"),
            (8, "8-9"),
            (9, "8-9")
        ]
        
        for age, expected_group in test_cases:
            # 타겟 연령 설정
            self.story_generator.set_target_age(age)
            
            # 연령대 그룹 확인
            self.assertEqual(self.story_generator._get_age_group(), expected_group)
    
    def test_get_age_specific_prompt(self):
        """연령대별 프롬프트 반환 테스트"""
        # 프롬프트 리스트 생성
        prompt_list = [
            {"age_group": "4-5", "prompt": "4-5세용 프롬프트"},
            {"age_group": "6-7", "prompt": "6-7세용 프롬프트"},
            {"age_group": "8-9", "prompt": "8-9세용 프롬프트"}
        ]
        
        # 연령별 프롬프트 테스트
        test_cases = [
            (4, "4-5세용 프롬프트"),
            (5, "4-5세용 프롬프트"),
            (6, "6-7세용 프롬프트"),
            (7, "6-7세용 프롬프트"),
            (8, "8-9세용 프롬프트"),
            (9, "8-9세용 프롬프트")
        ]
        
        for age, expected_prompt in test_cases:
            # 타겟 연령 설정
            self.story_generator.set_target_age(age)
            
            # 프롬프트 확인
            self.assertEqual(
                self.story_generator._get_age_specific_prompt(prompt_list),
                expected_prompt
            )
    
    def test_enrich_theme_with_rag(self):
        """RAG를 활용한 주제 강화 테스트"""
        # 스토리 아웃라인 및 타겟 연령 설정
        self.story_generator.set_story_outline(self.story_outline)
        self.story_generator.set_target_age(5)
        
        # 주제 강화
        enriched_theme = self.story_generator._enrich_theme_with_rag()
        
        # 강화된 주제 확인
        self.assertIsNotNone(enriched_theme)
        self.assertGreater(len(enriched_theme), 0)
        
        # 기존 주제 키워드가 포함되어 있는지 확인
        self.assertIn("우주", enriched_theme.lower())
        self.assertIn("공룡", enriched_theme.lower())
    
    def test_get_similar_stories(self):
        """유사 동화 검색 테스트"""
        # 스토리 아웃라인 및 타겟 연령 설정
        self.story_generator.set_story_outline(self.story_outline)
        self.story_generator.set_target_age(5)
        
        # 유사 동화 검색
        similar_stories = self.story_generator._get_similar_stories(n_results=2)
        
        # 결과 확인
        self.assertIsNotNone(similar_stories)
        self.assertLessEqual(len(similar_stories), 2)
        
        # 결과가 있는 경우 첫 번째 결과 확인
        if similar_stories:
            first_story = similar_stories[0]
            self.assertIn("title", first_story)
            self.assertIn("summary", first_story)
            self.assertIn("story_id", first_story)
    
    def test_build_few_shot_prompt(self):
        """Few-shot 프롬프트 생성 테스트"""
        # 스토리 아웃라인 및 타겟 연령 설정
        self.story_generator.set_story_outline(self.story_outline)
        self.story_generator.set_target_age(5)
        
        # Few-shot 프롬프트 생성
        few_shot_prompt = self.story_generator._build_few_shot_prompt()
        
        # 프롬프트 확인
        self.assertIsNotNone(few_shot_prompt)
        self.assertGreater(len(few_shot_prompt), 0)
        
        # 예시라는 단어가 포함되어 있는지 확인
        self.assertIn("예시", few_shot_prompt.lower())
    
    def test_parse_story_response(self):
        """스토리 응답 파싱 테스트"""
        # 테스트용 스토리 텍스트
        story_text = """
        우주 공룡 타키의 별빛 모험
        
        장면 1: 호기심 많은 타키
        
        파란색 공룡 타키는 매일 밤 창밖으로 반짝이는 별들을 바라보며 상상의 나래를 펼쳤어요.
        "저 별들 사이에는 어떤 세계가 있을까?" 타키는 궁금했어요.
        
        타키: 와! 별들이 정말 예쁘게 빛나네!
        엄마: 타키야, 벌써 자야 할 시간이란다.
        타키: 조금만 더 별을 볼래요. 저 별들 사이로 날아가 보고 싶어요!
        
        장면 2: 반짝이는 방문객
        
        그날 밤, 타키의 방에 작고 반짝이는 빛이 날아들었어요. 그것은 바로 별똥별이었어요!
        
        별똥별: 안녕, 타키야! 나는 별똥별이야. 너의 소원을 들었어. 우주 여행을 떠나볼래?
        타키: 정말요? 정말 우주로 갈 수 있어요?
        별똥별: 물론이지! 내 손을 잡으면 우리는 별들 사이로 날아갈 수 있어!
        
        타키는 두근거리는 마음으로 별똥별의 손을 잡았어요. 갑자기 타키의 몸이 빛나기 시작했고, 순식간에 하늘 높이 날아올랐답니다.
        """
        
        # 스토리 텍스트 파싱
        story_data = self.story_generator._parse_story_response(story_text)
        
        # 파싱 결과 확인
        self.assertIn("title", story_data)
        self.assertEqual(story_data["title"], "우주 공룡 타키의 별빛 모험")
        
        # 장면 확인
        self.assertIn("scenes", story_data)
        self.assertEqual(len(story_data["scenes"]), 2)
        
        # 첫 번째 장면 확인
        first_scene = story_data["scenes"][0]
        self.assertEqual(first_scene["title"], "장면 1: 호기심 많은 타키")
        self.assertIn("dialogues", first_scene)
        self.assertEqual(len(first_scene["dialogues"]), 3)
        
        # 두 번째 장면 확인
        second_scene = story_data["scenes"][1]
        self.assertEqual(second_scene["title"], "장면 2: 반짝이는 방문객")
        self.assertIn("dialogues", second_scene)
        self.assertEqual(len(second_scene["dialogues"]), 3)
    
    def test_save_and_load_story(self):
        """스토리 저장 및 로드 테스트"""
        # 테스트용 스토리 데이터
        test_story = {
            "title": "테스트 스토리",
            "scenes": [
                {
                    "title": "장면 1",
                    "description": "테스트 장면 설명",
                    "narration": "내레이션 텍스트",
                    "dialogues": [
                        {"character": "캐릭터1", "text": "대사1"},
                        {"character": "캐릭터2", "text": "대사2"}
                    ]
                }
            ]
        }
        
        # 스토리 데이터 설정
        self.story_generator.detailed_story = test_story
        
        # 저장 경로 설정
        save_path = os.path.join(self.output_dir, "test_story.json")
        
        # 스토리 저장
        self.story_generator.save_story_data(save_path)
        
        # 저장된 파일 확인
        self.assertTrue(os.path.exists(save_path))
        
        # 스토리 데이터 초기화
        self.story_generator.detailed_story = None
        
        # 스토리 로드
        self.story_generator.load_story_data(save_path)
        
        # 로드된 데이터 확인
        self.assertIsNotNone(self.story_generator.detailed_story)
        self.assertEqual(self.story_generator.detailed_story["title"], test_story["title"])
        self.assertEqual(len(self.story_generator.detailed_story["scenes"]), 1)
        self.assertEqual(len(self.story_generator.detailed_story["scenes"][0]["dialogues"]), 2)


if __name__ == "__main__":
    unittest.main() 