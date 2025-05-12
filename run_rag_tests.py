#!/usr/bin/env python3
"""
꼬기(ChatBot B)의 RAG 시스템 및 단위 테스트 실행 스크립트

이 스크립트는 다음 작업을 수행합니다:
1. RAG 시스템 단위 테스트 실행
2. 향상된 스토리 생성기 단위 테스트 실행
3. 샘플 동화 생성 및 RAG 시스템에 추가
"""

import os
import sys
import unittest
import shutil
import json
from pathlib import Path

# 프로젝트 루트 경로를 파이썬 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from chatbot.models.rag_system import RAGSystem
from chatbot.models.rag_enhanced_story_generator import RAGEnhancedStoryGenerator


def run_tests():
    """
    단위 테스트 실행
    
    Returns:
        bool: 모든 테스트가 통과하면 True, 아니면 False
    """
    print("=" * 60)
    print("          꼬기(ChatBot B) RAG 시스템 단위 테스트 실행          ")
    print("=" * 60)
    
    # 테스트 모듈 로드
    from chatbot.tests.test_rag_system import TestRAGSystem
    from chatbot.tests.test_rag_enhanced_story_generator import TestRAGEnhancedStoryGenerator
    
    # 테스트 스위트 구성
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestRAGSystem))
    test_suite.addTest(unittest.makeSuite(TestRAGEnhancedStoryGenerator))
    
    # 테스트 러너 설정
    runner = unittest.TextTestRunner(verbosity=2)
    
    # 테스트 실행
    result = runner.run(test_suite)
    
    # 테스트 결과 확인
    return result.wasSuccessful()


def create_sample_story():
    """
    샘플 동화 생성 및 RAG 시스템에 추가
    
    이 함수는 테스트 목적으로 샘플 동화를 생성하고 RAG 시스템에 추가합니다.
    
    Returns:
        str: 생성된 스토리의 ID
    """
    print("\n" + "=" * 60)
    print("               샘플 동화 생성 및 RAG 시스템 테스트               ")
    print("=" * 60)
    
    # 테스트용 출력 디렉토리
    output_dir = os.path.join(current_dir, "output", "rag_test")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # RAG 시스템 초기화
    rag_system = RAGSystem()
    
    # 스토리 생성기 초기화
    story_generator = RAGEnhancedStoryGenerator(
        output_dir=output_dir,
        rag_system=rag_system
    )
    
    # 샘플 동화 데이터
    sample_story = {
        "title": "별을 찾아 떠나는 여행",
        "tags": "6-7세,우주,모험,별,우정",
        "summary": "작은 소년 태양이가 반짝이는 별들을 찾아 우주 여행을 떠나는 모험 이야기",
        "content": """
        깊고 푸른 밤하늘을 바라보며 태양이는 항상 궁금했어요.
        "별들은 어디서 왔을까? 왜 밤에만 보이는 걸까?"
        
        어느 날 밤, 태양이의 방 창문을 통해 작은 별빛이 스며들었어요.
        그 빛은 점점 커져서 마침내 반짝이는 별 모양의 요정이 되었답니다.
        
        "안녕, 태양아! 나는 별빛 요정 반짝이야. 우주에서 왔어."
        태양이는 놀라면서도 기뻤어요. "와, 정말 별에서 온 거야?"
        
        반짝이는 슬픈 표정을 지었어요. "응, 그런데 내 별 가족들이 사라져버렸어.
        너무 멀리 놀러갔다가 집을 잃어버렸어. 나를 도와줄래?"
        
        태양이는 망설임 없이 대답했어요. "물론이지! 별 가족들을 찾아주자!"
        
        반짝이는 마법의 반짝임으로 태양이를 감쌌어요. 순간 태양이의 몸이 
        가벼워지더니 창문을 통해 밤하늘로 날아올랐답니다.
        
        태양이와 반짝이는 달, 목성, 토성을 지나 더 먼 우주로 여행했어요.
        여행 중에 다양한 별자리 친구들을 만났어요. 북두칠성, 오리온, 카시오페이아...
        모두들 반짝이의 가족을 찾는 것을 도와주었어요.
        
        마침내 그들은 아름다운 빛으로 가득한 성운에 도착했어요.
        그곳에서 반짝이의 별 가족들을 찾을 수 있었답니다!
        
        "고마워, 태양아! 네 덕분에 가족을 다시 만날 수 있었어."
        태양이는 미소 지으며 말했어요. "친구가 어려움에 처하면 도와주는 게 당연한 거야."
        
        반짝이의 가족들은 태양이에게 고마움의 표시로 특별한 선물을 주었어요.
        작은 별 모양 목걸이였는데, 밤에는 실제 별처럼 반짝였답니다.
        
        태양이는 집으로 돌아왔지만, 이제 밤하늘을 볼 때마다 별들이 
        단순한 빛이 아니라 자신의 우주 친구들이라는 것을 알게 되었어요.
        
        그리고 가끔, 아주 맑은 밤에는 반짝이와 그 가족들이 
        태양이에게 반짝이며 인사하는 것을 볼 수 있었답니다.
        """
    }
    
    # RAG 시스템에 샘플 동화 추가
    story_id = story_generator.add_sample_story_to_rag(sample_story)
    print(f"샘플 동화 추가됨: {sample_story['title']} (ID: {story_id})")
    
    # 스토리 아웃라인 설정
    story_outline = {
        "theme": "별빛 요정과 함께하는 우주 모험",
        "characters": ["지수 (호기심 많은 소녀)", "별빛 (반짝이는 요정)", "무지개별 (색색의 빛을 내는 별)"],
        "setting": "무한한 우주와 다양한 별자리들",
        "plot_summary": "호기심 많은 소녀 지수가 어느 날 밤 창가에서 만난 별빛 요정과 함께 우주 여행을 떠납니다. 잃어버린 무지개별을 찾아 다양한 별자리들을 만나고 우정과 용기를 배우는 이야기입니다.",
        "educational_value": "우주와 별자리에 대한 호기심, 친구를 돕는 용기와 협력의 가치"
    }
    
    # 스토리 생성기에 아웃라인 및 타겟 연령 설정
    story_generator.set_story_outline(story_outline)
    story_generator.set_target_age(6)
    
    # Few-shot 프롬프트 확인
    print("\n[Few-shot 프롬프트 샘플]")
    few_shot_prompt = story_generator._build_few_shot_prompt()
    print(few_shot_prompt[:500] + "...(생략)...")
    
    # 유사 스토리 검색
    print("\n[유사 동화 검색 결과]")
    similar_stories = story_generator._get_similar_stories(n_results=1)
    for i, story in enumerate(similar_stories):
        print(f"{i+1}. 제목: {story.get('title')}")
        print(f"   유사도 점수: {story.get('similarity_score'):.4f}")
        print(f"   요약: {story.get('summary')[:100]}...")
    
    # 상세 스토리 생성 (실제 API 호출 방지)
    print("\n[상세 스토리 생성]")
    print("실제 상세 스토리 생성은 API 호출을 방지하기 위해 생략합니다.")
    print("실제 환경에서는 story_generator.generate_detailed_story() 메서드를 호출합니다.")

    # 테스트용 상세 스토리 데이터
    test_story = {
        "title": "지수와 별빛의 우주 탐험",
        "scenes": [
            {
                "title": "반짝이는 만남",
                "description": "호기심 많은 소녀 지수가 창가에서 별빛 요정을 만납니다.",
                "narration": "깊고 푸른 밤하늘을 바라보며 지수는 항상 궁금했어요. '저 별들은 어떤 이야기를 가지고 있을까?'",
                "dialogues": [
                    {"character": "별빛", "text": "안녕, 지수야! 나는 별빛이라고 해. 우주에서 왔어."},
                    {"character": "지수", "text": "와, 정말 별에서 온 거야? 믿을 수 없어!"}
                ]
            }
        ]
    }
    
    # 테스트 스토리 저장
    story_generator.detailed_story = test_story
    story_path = os.path.join(output_dir, "test_story.json")
    story_generator.save_story_data(story_path)
    print(f"\n테스트 스토리가 저장되었습니다: {story_path}")
    
    return story_id


def main():
    """메인 함수"""
    # 단위 테스트 실행
    tests_passed = run_tests()
    
    if tests_passed:
        print("\n✅ 모든 테스트가 통과되었습니다!")
    else:
        print("\n❌ 일부 테스트가 실패했습니다.")
    
    # 샘플 동화 생성 및 테스트
    try:
        story_id = create_sample_story()
        print(f"\n✅ 샘플 동화 테스트가 완료되었습니다. (Story ID: {story_id})")
    except Exception as e:
        print(f"\n❌ 샘플 동화 테스트 중 오류가 발생했습니다: {str(e)}")
    
    print("\n처리가 완료되었습니다.")


if __name__ == "__main__":
    main() 