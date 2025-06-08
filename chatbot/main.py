#!/usr/bin/env python3
"""
부기와 꼬기 - Enhanced 아동 상호작용 챗봇 시스템 v2.0
CLI 인터페이스

부기(Chat-bot A): Enhanced 연령별 특화 대화 및 이야기 주제 추출
꼬기(Chat-bot B): Enhanced 상세 동화, 이미지, 음성 생성 (연령별 최적화)

Features v2.0:
- 연령별 특화 프롬프트 (4-7세, 8-9세)
- 향상된 프롬프트 엔지니어링
- 체인 오브 소트 추론
- 성능 추적 및 최적화
"""

import os
import sys
import time
import argparse
import json
import asyncio
from pathlib import Path

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__)) # chatbot/main.py
parent_dir = os.path.dirname(current_dir) # CCB_AI
sys.path.append(parent_dir) # 경로 설정

# Enhanced 챗봇 시스템
from chatbot.models.chat_bot_a.chat_bot_a import ChatBotA
from chatbot.models.chat_bot_b.chat_bot_b import ChatBotB

# 공통 유틸리티
from shared.utils.logging_utils import get_module_logger

# 워크플로우 모듈
from chatbot.workflow.orchestrator import WorkflowOrchestrator

logger = get_module_logger(__name__)

class EnhancedChatBotSystem:
    """Enhanced 부기와 꼬기 챗봇 시스템 통합 클래스 v2.0"""
    
    def __init__(self, enhanced_mode: bool = True, enable_performance_tracking: bool = True):
        """
        Enhanced 챗봇 시스템 초기화
        
        Args:
            enhanced_mode: Enhanced 프롬프트 시스템 사용 여부
            enable_performance_tracking: 성능 추적 활성화
        """
        self.enhanced_mode = enhanced_mode
        self.enable_performance_tracking = enable_performance_tracking
        
        # Enhanced 챗봇들 초기화
        logger.info(f"Enhanced 챗봇 시스템 초기화 시작 - Mode: {'Enhanced' if enhanced_mode else 'Basic'}")
        
        # VectorDB 인스턴스 생성 (ChatBotA에 필요)
        try:
            from chatbot.data.vector_db.core import VectorDB
            import os
            
            # .env에서 VectorDB 경로 읽기 (통일된 환경변수 사용)
            chroma_base = os.getenv("CHROMA_DB_PATH", "chatbot/data/vector_db")
            vector_db_path = os.path.join(chroma_base, "main")  # main DB 사용
            logger.info(f"VectorDB 경로 환경변수: {vector_db_path}")
            
            # VectorDB 초기화
            vector_db = VectorDB(
                persist_directory=vector_db_path,
                embedding_model="nlpai-lab/KURE-v1",
                use_hybrid_mode=True,
                memory_cache_size=1000,
                enable_lfu_cache=True
            )
            logger.info(f"VectorDB 초기화 완료: {vector_db_path}")
        except Exception as e:
            logger.warning(f"VectorDB 초기화 실패: {e}, None으로 진행")
            vector_db = None

        self.bugi = ChatBotA(
            vector_db_instance=vector_db,
            token_limit=10000,
            use_langchain=True,
            legacy_compatibility=True,
            enhanced_mode=enhanced_mode,
            enable_performance_tracking=enable_performance_tracking
        )  # Enhanced 부기 (Chat-bot A)
        
        self.kkogi = ChatBotB(
            use_enhanced_generators=enhanced_mode,
            enable_performance_tracking=enable_performance_tracking
        )  # Enhanced 꼬기 (Chat-bot B)
        
        self.conversation_history = []
        self.child_info = {
            "name": "",
            "age": 0,
            "interests": []
        }
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # 시스템 메트릭
        self.system_metrics = {
            "session_start_time": time.time(),
            "total_interactions": 0,
            "story_generation_count": 0,
            "enhanced_features_used": 0
        }
        
    def clear_screen(self):
        """터미널 화면 지우기"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_header(self):
        """프로그램 헤더 출력"""
        self.clear_screen()
        print("=" * 60)
        print("        부기와 꼬기 - 아동 상호작용 챗봇 시스템        ")
        print("=" * 60)
        print("부기: 아이와 대화하는 챗봇")
        print("꼬기: 이야기와 삽화를 생성하는 챗봇")
        print("-" * 60)
        
    def get_child_info(self):
        """아이 정보 입력받기"""
        self.print_header()
        print("아이의 정보를 입력해주세요.\n")
        
        # 이름 입력
        self.child_info["name"] = input("아이의 이름: ").strip()
        
        # 나이 입력
        while True:
            try:
                age_input = input("아이의 나이: ").strip()
                self.child_info["age"] = int(age_input)
                if self.child_info["age"] < 3 or self.child_info["age"] > 12:
                    print("3세에서 12세 사이의 나이를 입력해주세요.")
                    continue
                break
            except ValueError:
                print("숫자로 입력해주세요.")
        
        # 관심사 입력
        print("\n아이의 관심사를 쉼표로 구분하여 입력해주세요.")
        print("예: 공룡, 우주, 로봇, 동물")
        interests_input = input("관심사: ").strip()
        if interests_input:
            self.child_info["interests"] = [item.strip() for item in interests_input.split(",")]
        
        return self.child_info
    
    def chat_with_bugi(self):
        """부기(Chat-bot A)와 대화하기"""
        self.print_header()
        print(f"{self.child_info['name']}님, 부기와 대화를 시작합니다!")
        print("대화를 종료하려면 '끝'이라고 입력하세요.\n")
        
        # 대화 초기화
        greeting = self.bugi.initialize_chat(
            child_name=self.child_info["name"],
            age=self.child_info["age"],
            interests=self.child_info["interests"],
            chatbot_name="부기"
        )
        
        print(f"부기: {greeting}")
        print("-" * 60)
        
        # 대화 시작
        while True:
            user_input = input(f"{self.child_info['name']}: ").strip()
            
            if user_input.lower() == "끝":
                print("\n부기: 재미있는 이야기를 만들어볼게요! 잠시만 기다려주세요...")
                break
                
            response = self.bugi.get_response(user_input)
            print(f"부기: {response}")
            print("-" * 60)
            
            # 대화 기록 저장
            self.conversation_history.append({
                "user": user_input,
                "bugi": response
            })
    
    async def generate_story(self):
        """부기의 대화 내용을 바탕으로 꼬기가 이야기 생성 (비동기 처리)"""
        self.print_header()
        print("대화를 바탕으로 이야기를 만들고 있어요...\n")
        
        # 부기가 이야기 주제 추출
        try:
            story_outline = self.bugi.suggest_story_theme()  # 올바른 메서드명 사용
            
            if not story_outline or "plot_summary" not in story_outline:  # 올바른 키 사용
                print("이야기 주제를 추출하지 못했습니다. 더 많은 대화가 필요합니다.")
                return False
                
            print("=== 부기가 추출한 이야기 주제 ===")
            print(f"제목: {story_outline.get('title', '제목 없음')}")
            print(f"줄거리: {story_outline.get('plot_summary', '')}")  # 올바른 키 사용
            print(f"등장인물: {', '.join(story_outline.get('characters', []))}")
            print(f"배경: {story_outline.get('setting', '')}")
            print("-" * 60)
            print("이 주제로 이야기를 만들까요? (y/n)")
            
            if input().lower() != 'y':
                print("이야기 생성을 취소합니다.")
                return False
                
            # 꼬기에게 이야기 주제 전달
            self.kkogi.set_story_outline(story_outline)
            self.kkogi.set_target_age(self.child_info["age"])
            
            print("\n꼬기가 상세 이야기를 만들고 있어요...")
            print("(실제 API 호출이 이루어지며 약간의 시간이 소요될 수 있습니다)")
            
            # 이야기 생성 진행 (비동기 호출)
            detailed_story_result = await self.kkogi.generate_detailed_story()
            
            # 생성된 이야기 정보 출력
            if detailed_story_result:
                story_data = detailed_story_result.get('story_data', {})
                print("\n=== 생성된 이야기 정보 ===")
                print(f"제목: {story_data.get('title', '제목 없음')}")
                print(f"대상 연령: {self.child_info['age']}세")
                print(f"장면 수: {len(story_data.get('chapters', []))}개")
                print(f"상태: {detailed_story_result.get('status', '알 수 없음')}")
                
                # 이야기 저장
                story_file = self.output_dir / f"{self.child_info['name']}의_이야기.json"
                with open(story_file, 'w', encoding='utf-8') as f:
                    json.dump(detailed_story_result, f, ensure_ascii=False, indent=2)
                    
                print(f"\n이야기가 저장되었습니다: {story_file}")
                
                # 삽화 생성 여부 확인
                print("\n이야기에 삽화를 추가할까요? (y/n)")
                if input().lower() == 'y':
                    print("\n삽화를 생성하고 있어요...")
                    try:
                        if hasattr(self.kkogi, 'image_generator'):
                            image_input = {
                                "story_data": story_data,
                                "story_id": detailed_story_result.get('story_id')
                            }
                            await self.kkogi.image_generator.generate(image_input)
                            print("삽화가 생성되었습니다.")
                        else:
                            print("삽화 생성기를 찾을 수 없습니다.")
                    except Exception as e:
                        print(f"삽화 생성 중 오류: {e}")
                
                # 음성 생성 여부 확인
                print("\n이야기에 음성을 추가할까요? (y/n)")
                if input().lower() == 'y':
                    print("\n음성을 생성하고 있어요...")
                    try:
                        if hasattr(self.kkogi, 'voice_generator'):
                            voice_input = {
                                "story_data": story_data,
                                "story_id": detailed_story_result.get('story_id')
                            }
                            await self.kkogi.voice_generator.generate(voice_input)
                            print("음성이 생성되었습니다.")
                        else:
                            print("음성 생성기를 찾을 수 없습니다.")
                    except Exception as e:
                        print(f"음성 생성 중 오류: {e}")
                
                return True
            else:
                print("이야기 생성에 실패했습니다.")
                return False
                
        except Exception as e:
            print(f"이야기 생성 중 오류가 발생했습니다: {str(e)}")
            return False
    
    async def run(self):
        """메인 프로그램 실행 (비동기)"""
        try:
            # 아이 정보 입력
            self.get_child_info()
            
            # 부기와 대화
            self.chat_with_bugi()
            
            # 이야기 생성 (비동기 호출)
            await self.generate_story()
            
            # 종료
            print("\n프로그램을 종료합니다. 생성된 파일은 output 폴더에서 확인할 수 있습니다.")
            
        except KeyboardInterrupt:
            print("\n\n프로그램이 사용자에 의해 종료되었습니다.")
        except Exception as e:
            print(f"\n오류가 발생했습니다: {str(e)}")

def main():
    """프로그램 진입점"""
    parser = argparse.ArgumentParser(description="부기와 꼬기 - 아동 상호작용 챗봇 시스템")
    parser.add_argument('--test', action='store_true', help='테스트 모드 실행')
    args = parser.parse_args()
    
    if args.test: # 테스트 모드 실행
        print("테스트 모드를 실행합니다...")
        import unittest
        from tests.test_chatbot import TestChatBotIntegration
        
        suite = unittest.TestLoader().loadTestsFromTestCase(TestChatBotIntegration)
        unittest.TextTestRunner(verbosity=2).run(suite)
    else: # 실제 챗봇 테스트 실행
        # 출력 디렉토리 설정 (app.py와 일관성 유지)
        output_dir = os.getenv("MULTIMEDIA_OUTPUT_DIR", "output")
        
        fairy_tail = EnhancedChatBotSystem(
            enhanced_mode=True,
            enable_performance_tracking=True
        ) # Enhanced 챗봇 시스템 인스턴스 생성
        asyncio.run(fairy_tail.run())

if __name__ == "__main__":
    main() 