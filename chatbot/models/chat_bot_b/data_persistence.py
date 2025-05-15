"""
스토리 데이터 저장 및 로드를 담당하는 모듈
"""
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import os
import json
import uuid
import time
from datetime import datetime

from shared.utils.logging_utils import get_module_logger
from shared.utils.file_utils import ensure_directory, save_json, load_json

logger = get_module_logger(__name__)

class DataPersistence:
    """
    스토리 데이터 저장 및 로드를 담당하는 클래스
    """
    
    def __init__(self, base_output_dir: Union[str, Path] = None, stories_dir: Union[str, Path] = None):
        """
        데이터 저장소 초기화
        
        Args:
            base_output_dir: 기본 출력 디렉토리
            stories_dir: 스토리 저장 디렉토리
        """
        self.base_output_dir = Path(base_output_dir) if base_output_dir else Path("output")
        
        # 스토리 저장 디렉토리 설정
        self.stories_dir = Path(stories_dir) if stories_dir else self.base_output_dir / "stories"
        ensure_directory(self.stories_dir)
        
        # 회화 저장 디렉토리 설정
        self.conversations_dir = self.base_output_dir / "conversations"
        ensure_directory(self.conversations_dir)
    
    def generate_story_id(self) -> str:
        """
        고유한 스토리 ID 생성
        
        Returns:
            str: 생성된 스토리 ID
        """
        # UUID를 사용하여 고유 ID 생성
        return str(uuid.uuid4())
    
    def save_story_data(self, story_data: Dict, story_id: Optional[str] = None) -> str:
        """
        스토리 데이터 저장
        
        Args:
            story_data: 저장할 스토리 데이터
            story_id: 스토리 ID (없으면 새로 생성)
            
        Returns:
            str: 저장된 스토리 ID
        """
        # 스토리 ID가 없으면 새로 생성
        if not story_id:
            story_id = self.generate_story_id()
            
        # 스토리 저장 디렉토리 설정
        story_dir = self.stories_dir / story_id
        ensure_directory(story_dir)
        
        try:
            # 메타데이터 추가
            story_data["story_id"] = story_id
            if "created_at" not in story_data:
                story_data["created_at"] = datetime.now().isoformat()
            story_data["updated_at"] = datetime.now().isoformat()
            
            # 스토리 데이터 파일 경로
            file_path = story_dir / "story_data.json"
            
            # JSON 파일로 저장
            success = save_json(story_data, file_path)
            
            if success:
                logger.info(f"스토리 데이터 저장 완료: {file_path}")
                return story_id
            else:
                logger.error(f"스토리 데이터 저장 실패: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"스토리 데이터 저장 중 오류 발생: {e}")
            return None
    
    def load_story_data(self, story_id: str) -> Optional[Dict]:
        """
        스토리 데이터 로드
        
        Args:
            story_id: 스토리 ID
            
        Returns:
            Optional[Dict]: 로드된 스토리 데이터 또는 None
        """
        # 스토리 데이터 파일 경로
        file_path = self.stories_dir / story_id / "story_data.json"
        
        try:
            # JSON 파일에서 로드
            story_data = load_json(file_path)
            
            if story_data:
                logger.info(f"스토리 데이터 로드 완료: {file_path}")
                return story_data
            else:
                logger.error(f"스토리 데이터 로드 실패: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"스토리 데이터 로드 중 오류 발생: {e}")
            return None
    
    def list_stories(self) -> List[Dict]:
        """
        저장된 스토리 목록 조회
        
        Returns:
            List[Dict]: 스토리 목록 (ID, 제목, 생성일시 등)
        """
        stories = []
        
        try:
            # 스토리 디렉토리 탐색
            for story_dir in self.stories_dir.iterdir():
                if story_dir.is_dir():
                    story_id = story_dir.name
                    file_path = story_dir / "story_data.json"
                    
                    if file_path.exists():
                        # 기본 정보만 로드
                        story_data = load_json(file_path)
                        if story_data:
                            stories.append({
                                "story_id": story_id,
                                "title": story_data.get("title", "제목 없음"),
                                "created_at": story_data.get("created_at", ""),
                                "updated_at": story_data.get("updated_at", ""),
                                "theme": story_data.get("theme", ""),
                                "age_group": story_data.get("age_group", "")
                            })
            
            # 생성일시 기준으로 정렬
            stories.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            return stories
            
        except Exception as e:
            logger.error(f"스토리 목록 조회 중 오류 발생: {e}")
            return []
    
    def save_conversation(self, conversation_data: Dict, user_id: str = None, 
                          conversation_id: str = None) -> str:
        """
        대화 데이터 저장
        
        Args:
            conversation_data: 저장할 대화 데이터
            user_id: 사용자 ID
            conversation_id: 대화 ID (없으면 새로 생성)
            
        Returns:
            str: 저장된 대화 ID
        """
        # 대화 ID가 없으면, 타임스탬프 + UUID 조합으로 생성
        if not conversation_id:
            timestamp = int(time.time())
            unique_id = str(uuid.uuid4())[:8]
            conversation_id = f"{timestamp}_{unique_id}"
        
        try:
            # 메타데이터 추가
            conversation_data["conversation_id"] = conversation_id
            if user_id:
                conversation_data["user_id"] = user_id
            if "created_at" not in conversation_data:
                conversation_data["created_at"] = datetime.now().isoformat()
            conversation_data["updated_at"] = datetime.now().isoformat()
            
            # 사용자별 대화 저장 디렉토리 설정
            if user_id:
                user_dir = self.conversations_dir / user_id
                ensure_directory(user_dir)
                file_path = user_dir / f"{conversation_id}.json"
            else:
                file_path = self.conversations_dir / f"{conversation_id}.json"
            
            # JSON 파일로 저장
            success = save_json(conversation_data, file_path)
            
            if success:
                logger.info(f"대화 데이터 저장 완료: {file_path}")
                return conversation_id
            else:
                logger.error(f"대화 데이터 저장 실패: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"대화 데이터 저장 중 오류 발생: {e}")
            return None
    
    def load_conversation(self, conversation_id: str, user_id: str = None) -> Optional[Dict]:
        """
        대화 데이터 로드
        
        Args:
            conversation_id: 대화 ID
            user_id: 사용자 ID
            
        Returns:
            Optional[Dict]: 로드된 대화 데이터 또는 None
        """
        try:
            # 대화 데이터 파일 경로 결정
            if user_id:
                file_path = self.conversations_dir / user_id / f"{conversation_id}.json"
            else:
                file_path = self.conversations_dir / f"{conversation_id}.json"
            
            # 파일이 존재하지 않으면 사용자 ID가 없는 경로도 시도
            if not file_path.exists() and user_id:
                file_path = self.conversations_dir / f"{conversation_id}.json"
            
            # JSON 파일에서 로드
            if file_path.exists():
                conversation_data = load_json(file_path)
                
                if conversation_data:
                    logger.info(f"대화 데이터 로드 완료: {file_path}")
                    return conversation_data
            
            logger.error(f"대화 데이터 로드 실패: 파일을 찾을 수 없음 - {conversation_id}")
            return None
                
        except Exception as e:
            logger.error(f"대화 데이터 로드 중 오류 발생: {e}")
            return None
    
    def list_conversations(self, user_id: str = None) -> List[Dict]:
        """
        저장된 대화 목록 조회
        
        Args:
            user_id: 사용자 ID (None이면 모든 대화)
            
        Returns:
            List[Dict]: 대화 목록 (ID, 생성일시 등)
        """
        conversations = []
        
        try:
            # 검색 디렉토리 결정
            if user_id:
                search_dir = self.conversations_dir / user_id
                # 디렉토리가 없으면 빈 목록 반환
                if not search_dir.exists():
                    return []
            else:
                search_dir = self.conversations_dir
            
            # 대화 파일 탐색
            for file_path in search_dir.glob("*.json"):
                if file_path.is_file():
                    # 기본 정보만 로드
                    try:
                        conversation_data = load_json(file_path)
                        if conversation_data:
                            conversations.append({
                                "conversation_id": conversation_data.get("conversation_id", file_path.stem),
                                "user_id": conversation_data.get("user_id", ""),
                                "created_at": conversation_data.get("created_at", ""),
                                "updated_at": conversation_data.get("updated_at", ""),
                                "summary": conversation_data.get("summary", "")
                            })
                    except Exception as e:
                        logger.error(f"대화 파일 로드 중 오류: {file_path} - {e}")
            
            # 생성일시 기준으로 정렬
            conversations.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            return conversations
            
        except Exception as e:
            logger.error(f"대화 목록 조회 중 오류 발생: {e}")
            return []
    
    def save_story_outline(self, outline_data: Dict, story_id: str = None) -> str:
        """
        스토리 개요 저장
        
        Args:
            outline_data: 저장할 스토리 개요 데이터
            story_id: 스토리 ID (없으면 새로 생성)
            
        Returns:
            str: 저장된 스토리 ID
        """
        # 스토리 개요는 스토리 데이터의 일종으로 저장
        outline_data["is_outline"] = True
        
        return self.save_story_data(outline_data, story_id)
    
    def load_story_outline(self, story_id: str) -> Optional[Dict]:
        """
        스토리 개요 로드
        
        Args:
            story_id: 스토리 ID
            
        Returns:
            Optional[Dict]: 로드된 스토리 개요 데이터 또는 None
        """
        # 스토리 데이터 로드
        story_data = self.load_story_data(story_id)
        
        # 스토리 개요인지 확인
        if story_data and story_data.get("is_outline", False):
            return story_data
        
        return None
    
    def convert_outline_to_full_story(self, story_id: str, full_story_data: Dict) -> str:
        """
        스토리 개요를 완전한 스토리로 변환
        
        Args:
            story_id: 개요 스토리 ID
            full_story_data: 완전한 스토리 데이터
            
        Returns:
            str: 저장된 완전한 스토리 ID
        """
        try:
            # 개요 데이터 로드
            outline_data = self.load_story_outline(story_id)
            
            # 개요 데이터가 있으면 병합
            if outline_data:
                # 개요 데이터의 기본 정보 유지
                for key in ["theme", "characters", "setting", "plot_summary", "educational_value", "age_group", "created_at"]:
                    if key in outline_data and key not in full_story_data:
                        full_story_data[key] = outline_data[key]
                
                # 개요 표시 제거
                full_story_data["is_outline"] = False
                
                # 개요에서 변환된 것임을 표시
                full_story_data["converted_from_outline"] = True
                full_story_data["outline_story_id"] = story_id
            
            # 새 스토리로 저장 (같은 ID 사용)
            return self.save_story_data(full_story_data, story_id)
            
        except Exception as e:
            logger.error(f"스토리 개요를 완전한 스토리로 변환 중 오류 발생: {e}")
            return None 