#!/usr/bin/env python3
"""
꼬기(chat_bot_b) WebSocket 클라이언트 테스트 스크립트

이 스크립트는 WebSocket을 통해 꼬기(채팅봇 B)와 통신하여
동화 생성 과정을 테스트합니다.

사용 방법:
python test_story_generator_client.py --server ws://localhost:8000/ws/story_generation --token valid_token --name 민준 --age 6 --interests 공룡,우주
"""

import asyncio
import json
import argparse
import websockets
import sys
import os
from datetime import datetime
from pathlib import Path

class StoryGeneratorClient:
    """꼬기(chat_bot_b) WebSocket 클라이언트 클래스"""
    
    def __init__(self, server_url, token, child_name, age, interests=None, story_outline_file=None):
        """
        클라이언트 초기화
        
        Args:
            server_url (str): WebSocket 서버 URL
            token (str): 인증 토큰
            child_name (str): 아이 이름
            age (int): 아이 나이
            interests (list): 관심사 목록
            story_outline_file (str): 동화 줄거리 JSON 파일 경로
        """
        self.server_url = server_url
        self.token = token
        self.child_name = child_name
        self.age = age
        self.interests = interests or []
        self.story_outline_file = story_outline_file
        self.websocket = None
        self.story_outline = None
        self.detailed_story = None
        self.images = []
        self.voice_data = None
        
        # 파일 저장 경로
        self.output_dir = Path("output") / f"client_{child_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def connect(self):
        """WebSocket 서버에 연결"""
        # 쿼리 파라미터 구성
        query_params = f"token={self.token}&child_name={self.child_name}&age={self.age}"
        if self.interests:
            interests_str = ",".join(self.interests)
            query_params += f"&interests={interests_str}"
        
        # WebSocket 연결
        uri = f"{self.server_url}?{query_params}"
        self.websocket = await websockets.connect(uri)
        print(f"서버 {self.server_url}에 연결됨")
        
        # 인사말 메시지 수신
        greeting = await self.websocket.recv()
        greeting_data = json.loads(greeting)
        if greeting_data.get('type') == 'greeting':
            print(f"\n꼬기: {greeting_data.get('text', '')}")
        else:
            print(f"\n예상치 못한 응답: {greeting_data}")
        
        return True
    
    async def load_story_outline(self):
        """동화 줄거리 로드"""
        if self.story_outline_file:
            try:
                with open(self.story_outline_file, 'r', encoding='utf-8') as f:
                    self.story_outline = json.load(f)
                print(f"동화 줄거리 로드됨: {self.story_outline_file}")
                return True
            except Exception as e:
                print(f"동화 줄거리 로드 실패: {e}")
                return False
        else:
            # 테스트용 예시 스토리 아웃라인
            self.story_outline = {
                "theme": f"우주 탐험을 떠나는 용감한 공룡",
                "characters": ["탁구 (파란색 공룡)", "별똥별 (우주 로봇)", "루나 (달의 공주)"],
                "setting": "은하계 너머의 신비로운 우주",
                "plot_summary": "호기심 많은 공룡 탁구는 우연히 발견한 우주선을 타고 우주 모험을 떠납니다. 그곳에서 길을 잃은 로봇 별똥별을 만나게 되고, 함께 달나라로 가서 달의 공주 루나를 도와 잃어버린 별빛을 찾는 모험을 하게 됩니다.",
                "educational_value": "용기, 우정, 협력의 가치와 함께 우주에 대한 호기심 자극",
                "target_age": self.age
            }
            print("예시 동화 줄거리 생성됨")
            return True
    
    async def send_story_outline(self):
        """동화 줄거리 전송"""
        if not self.story_outline:
            print("전송할 동화 줄거리가 없습니다.")
            return False
        
        try:
            # 동화 줄거리 전송
            message = {
                "type": "story_outline",
                "story_outline": self.story_outline
            }
            await self.websocket.send(json.dumps(message))
            print("동화 줄거리 전송됨")
            
            # 처리 중 메시지 수신
            response = await self.websocket.recv()
            response_data = json.loads(response)
            if response_data.get('type') == 'processing':
                print(f"\n꼬기: {response_data.get('text', '')}")
            
            # 상세 스토리 응답 수신
            story_response = await self.websocket.recv()
            story_data = json.loads(story_response)
            
            if story_data.get('type') == 'detailed_story':
                self.detailed_story = story_data.get('detailed_story')
                print(f"\n꼬기: {story_data.get('text', '')}")
                
                # 상세 스토리 정보 출력
                print(f"상세 스토리 제목: {self.detailed_story['title']}")
                print(f"장면 수: {len(self.detailed_story['scenes'])}")
                
                # 상세 스토리 저장
                story_file = self.output_dir / "detailed_story.json"
                with open(story_file, 'w', encoding='utf-8') as f:
                    json.dump(self.detailed_story, f, ensure_ascii=False, indent=2)
                print(f"상세 스토리 저장됨: {story_file}")
                
                return True
            elif story_data.get('type') == 'error':
                print(f"\n오류: {story_data.get('text')}")
                print(f"오류 메시지: {story_data.get('error_message')}")
                return False
            else:
                print(f"\n예상치 못한 응답: {story_data}")
                return False
        
        except Exception as e:
            print(f"동화 줄거리 전송 중 오류 발생: {e}")
            return False
    
    async def request_illustrations(self):
        """일러스트 생성 요청"""
        if not self.detailed_story:
            print("상세 스토리가 없어 일러스트를 생성할 수 없습니다.")
            return False
        
        try:
            # 일러스트 생성 요청
            message = {
                "type": "generate_illustrations"
            }
            await self.websocket.send(json.dumps(message))
            print("일러스트 생성 요청됨")
            
            # 처리 중 메시지 수신
            response = await self.websocket.recv()
            response_data = json.loads(response)
            if response_data.get('type') == 'processing':
                print(f"\n꼬기: {response_data.get('text', '')}")
            
            # 일러스트 응답 수신
            illust_response = await self.websocket.recv()
            illust_data = json.loads(illust_response)
            
            if illust_data.get('type') == 'illustrations':
                self.images = illust_data.get('images', [])
                print(f"\n꼬기: {illust_data.get('text', '')}")
                
                # 일러스트 정보 출력
                print(f"생성된 일러스트 수: {len(self.images)}")
                for i, img_path in enumerate(self.images):
                    print(f"일러스트 {i+1}: {img_path}")
                
                # 일러스트 정보 저장
                illust_file = self.output_dir / "illustrations.json"
                with open(illust_file, 'w', encoding='utf-8') as f:
                    json.dump(self.images, f, ensure_ascii=False, indent=2)
                print(f"일러스트 정보 저장됨: {illust_file}")
                
                return True
            elif illust_data.get('type') == 'error':
                print(f"\n오류: {illust_data.get('text')}")
                print(f"오류 메시지: {illust_data.get('error_message')}")
                return False
            else:
                print(f"\n예상치 못한 응답: {illust_data}")
                return False
        
        except Exception as e:
            print(f"일러스트 생성 요청 중 오류 발생: {e}")
            return False
    
    async def request_voice(self):
        """내레이션 생성 요청"""
        if not self.detailed_story:
            print("상세 스토리가 없어 내레이션을 생성할 수 없습니다.")
            return False
        
        try:
            # 내레이션 생성 요청
            message = {
                "type": "generate_voice"
            }
            await self.websocket.send(json.dumps(message))
            print("내레이션 생성 요청됨")
            
            # 처리 중 메시지 수신
            response = await self.websocket.recv()
            response_data = json.loads(response)
            if response_data.get('type') == 'processing':
                print(f"\n꼬기: {response_data.get('text', '')}")
            
            # 내레이션 응답 수신
            voice_response = await self.websocket.recv()
            voice_data = json.loads(voice_response)
            
            if voice_data.get('type') == 'voice':
                self.voice_data = voice_data.get('voice_data', {})
                print(f"\n꼬기: {voice_data.get('text', '')}")
                
                # 내레이션 정보 출력
                narr_count = len(self.voice_data.get('narration', {}))
                char_count = len(self.voice_data.get('characters', {}))
                print(f"내레이션 파일 수: {narr_count}")
                print(f"캐릭터 음성 수: {char_count}")
                
                # 내레이션 정보 저장
                voice_file = self.output_dir / "voice_data.json"
                with open(voice_file, 'w', encoding='utf-8') as f:
                    json.dump(self.voice_data, f, ensure_ascii=False, indent=2)
                print(f"내레이션 정보 저장됨: {voice_file}")
                
                return True
            elif voice_data.get('type') == 'error':
                print(f"\n오류: {voice_data.get('text')}")
                print(f"오류 메시지: {voice_data.get('error_message')}")
                return False
            else:
                print(f"\n예상치 못한 응답: {voice_data}")
                return False
        
        except Exception as e:
            print(f"내레이션 생성 요청 중 오류 발생: {e}")
            return False
    
    async def request_preview(self):
        """동화 미리보기 요청"""
        try:
            # 미리보기 요청
            message = {
                "type": "get_preview"
            }
            await self.websocket.send(json.dumps(message))
            print("동화 미리보기 요청됨")
            
            # 미리보기 응답 수신
            preview_response = await self.websocket.recv()
            preview_data = json.loads(preview_response)
            
            if preview_data.get('type') == 'preview':
                preview = preview_data.get('preview', {})
                print(f"\n꼬기: {preview_data.get('text', '')}")
                
                # 미리보기 정보 출력
                print(f"제목: {preview.get('title', '제목 없음')}")
                summary = preview.get('summary', '요약 없음')
                print(f"요약: {summary[:150]}..." if len(summary) > 150 else f"요약: {summary}")
                print(f"이미지 수: {preview.get('image_count', 0)}")
                print(f"예상 재생 시간: {preview.get('duration', '알 수 없음')}")
                
                # 미리보기 정보 저장
                preview_file = self.output_dir / "preview.json"
                with open(preview_file, 'w', encoding='utf-8') as f:
                    json.dump(preview, f, ensure_ascii=False, indent=2)
                print(f"미리보기 정보 저장됨: {preview_file}")
                
                return True
            elif preview_data.get('type') == 'error':
                print(f"\n오류: {preview_data.get('text')}")
                print(f"오류 메시지: {preview_data.get('error_message')}")
                return False
            else:
                print(f"\n예상치 못한 응답: {preview_data}")
                return False
        
        except Exception as e:
            print(f"미리보기 요청 중 오류 발생: {e}")
            return False
    
    async def save_story(self):
        """동화 저장 요청"""
        try:
            # 저장 요청
            story_name = f"{self.child_name}_story_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            message = {
                "type": "save_story",
                "story_name": story_name
            }
            await self.websocket.send(json.dumps(message))
            print(f"동화 저장 요청됨: {story_name}")
            
            # 저장 응답 수신
            save_response = await self.websocket.recv()
            save_data = json.loads(save_response)
            
            if save_data.get('type') == 'save_complete':
                print(f"\n꼬기: {save_data.get('text', '')}")
                print(f"저장 경로: {save_data.get('story_path', '알 수 없음')}")
                return True
            elif save_data.get('type') == 'error':
                print(f"\n오류: {save_data.get('text')}")
                print(f"오류 메시지: {save_data.get('error_message')}")
                return False
            else:
                print(f"\n예상치 못한 응답: {save_data}")
                return False
        
        except Exception as e:
            print(f"동화 저장 요청 중 오류 발생: {e}")
            return False
    
    async def disconnect(self):
        """연결 종료"""
        if self.websocket:
            await self.websocket.close()
            print("WebSocket 연결 종료")
    
    async def run(self):
        """클라이언트 실행"""
        try:
            # 서버 연결
            connected = await self.connect()
            if not connected:
                return False
            
            # 동화 줄거리 로드
            loaded = await self.load_story_outline()
            if not loaded:
                return False
            
            # 메뉴 표시 및 처리
            while True:
                print("\n=== 꼬기(chat_bot_b) 테스트 메뉴 ===")
                print("1. 동화 줄거리 전송")
                print("2. 일러스트 생성 요청")
                print("3. 내레이션 생성 요청")
                print("4. 동화 미리보기 요청")
                print("5. 동화 저장 요청")
                print("0. 종료")
                
                choice = input("\n원하는 작업을 선택하세요: ")
                
                if choice == '1':
                    await self.send_story_outline()
                elif choice == '2':
                    await self.request_illustrations()
                elif choice == '3':
                    await self.request_voice()
                elif choice == '4':
                    await self.request_preview()
                elif choice == '5':
                    await self.save_story()
                elif choice == '0':
                    break
                else:
                    print("잘못된 선택입니다. 다시 선택해주세요.")
            
            return True
        
        except Exception as e:
            print(f"실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # 연결 종료
            await self.disconnect()

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="꼬기(chat_bot_b) WebSocket 클라이언트 테스트")
    parser.add_argument("--server", default="ws://localhost:8000/ws/story_generation", help="WebSocket 서버 URL")
    parser.add_argument("--token", default="valid_token", help="인증 토큰")
    parser.add_argument("--name", required=True, help="아이 이름")
    parser.add_argument("--age", type=int, required=True, help="아이 나이 (4-9세)")
    parser.add_argument("--interests", help="관심사 (쉼표로 구분)")
    parser.add_argument("--story-file", help="동화 줄거리 JSON 파일 경로")
    
    args = parser.parse_args()
    
    # 관심사 처리
    interests = []
    if args.interests:
        interests = args.interests.split(',')
    
    # 비동기 실행
    client = StoryGeneratorClient(
        server_url=args.server,
        token=args.token,
        child_name=args.name,
        age=args.age,
        interests=interests,
        story_outline_file=args.story_file
    )
    
    asyncio.run(client.run())

if __name__ == "__main__":
    main() 