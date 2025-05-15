"""
이미지 및 오디오 파일 관리를 담당하는 모듈
"""
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import os
import shutil
import json
import time
from datetime import datetime

from shared.utils.logging_utils import get_module_logger
from shared.utils.file_utils import ensure_directory, copy_file, save_json, load_json

logger = get_module_logger(__name__)

class MediaManager:
    """
    이미지 및 오디오 파일 관리를 담당하는 클래스
    """
    
    def __init__(self, base_output_dir: Union[str, Path] = None, 
                 image_dir: Union[str, Path] = None, 
                 audio_dir: Union[str, Path] = None):
        """
        미디어 관리자 초기화
        
        Args:
            base_output_dir: 기본 출력 디렉토리
            image_dir: 이미지 저장 디렉토리
            audio_dir: 오디오 저장 디렉토리
        """
        self.base_output_dir = Path(base_output_dir) if base_output_dir else Path("output")
        
        # 이미지 및 오디오 디렉토리 설정
        self.image_dir = Path(image_dir) if image_dir else self.base_output_dir / "images"
        self.audio_dir = Path(audio_dir) if audio_dir else self.base_output_dir / "audio"
        
        # 디렉토리 생성
        ensure_directory(self.image_dir)
        ensure_directory(self.audio_dir)
        
        # 현재 스토리 ID
        self.current_story_id = None
        
        # 미디어 메타데이터
        self.media_metadata = {}
    
    def set_current_story_id(self, story_id: str):
        """
        현재 처리 중인 스토리 ID 설정
        
        Args:
            story_id: 스토리 ID
        """
        self.current_story_id = story_id
        
        # 스토리별 디렉토리 생성
        if story_id:
            self.story_image_dir = self.image_dir / story_id
            self.story_audio_dir = self.audio_dir / story_id
            
            ensure_directory(self.story_image_dir)
            ensure_directory(self.story_audio_dir)
            
            # 메타데이터 초기화
            self.media_metadata = {
                "story_id": story_id,
                "created_at": datetime.now().isoformat(),
                "images": {},
                "audio": {}
            }
    
    def save_image(self, image_data, filename: str = None, chapter_number: int = None) -> Optional[str]:
        """
        이미지 저장
        
        Args:
            image_data: 이미지 데이터 (파일 경로 또는 바이너리 데이터)
            filename: 저장할 파일 이름 (기본값: 자동 생성)
            chapter_number: 관련 챕터 번호
            
        Returns:
            Optional[str]: 저장된 이미지 파일 경로 또는 None
        """
        if not self.current_story_id:
            logger.error("현재 스토리 ID가 설정되지 않았습니다.")
            return None
        
        try:
            # 파일 이름 생성 (제공되지 않은 경우)
            if not filename:
                timestamp = int(time.time())
                if chapter_number:
                    filename = f"chapter_{chapter_number}_{timestamp}.jpg"
                else:
                    filename = f"image_{timestamp}.jpg"
            elif not any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                filename = f"{filename}.jpg"
            
            # 파일 경로 생성
            file_path = self.story_image_dir / filename
            
            # 이미지 데이터 유형에 따라 처리
            if isinstance(image_data, (str, Path)) and os.path.exists(image_data):
                # 파일 경로인 경우 복사
                success = copy_file(image_data, file_path)
            else:
                # 다른 형식은 오류 처리
                logger.error("지원되지 않는 이미지 데이터 형식입니다.")
                return None
                
            if success:
                # 메타데이터 업데이트
                self.media_metadata["images"][filename] = {
                    "file_path": str(file_path),
                    "chapter_number": chapter_number,
                    "created_at": datetime.now().isoformat()
                }
                
                logger.info(f"이미지 저장 완료: {file_path}")
                return str(file_path)
            else:
                logger.error(f"이미지 저장 실패: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"이미지 저장 중 오류 발생: {e}")
            return None
    
    def save_audio(self, audio_data, filename: str = None, chapter_number: int = None, 
                   speaker_type: str = None) -> Optional[str]:
        """
        오디오 저장
        
        Args:
            audio_data: 오디오 데이터 (파일 경로 또는 바이너리 데이터)
            filename: 저장할 파일 이름 (기본값: 자동 생성)
            chapter_number: 관련 챕터 번호
            speaker_type: 화자 유형
            
        Returns:
            Optional[str]: 저장된 오디오 파일 경로 또는 None
        """
        if not self.current_story_id:
            logger.error("현재 스토리 ID가 설정되지 않았습니다.")
            return None
        
        try:
            # 파일 이름 생성 (제공되지 않은 경우)
            if not filename:
                timestamp = int(time.time())
                if chapter_number:
                    if speaker_type:
                        filename = f"chapter_{chapter_number}_{speaker_type}_{timestamp}.mp3"
                    else:
                        filename = f"chapter_{chapter_number}_{timestamp}.mp3"
                else:
                    filename = f"audio_{timestamp}.mp3"
            elif not filename.endswith('.mp3'):
                filename = f"{filename}.mp3"
            
            # 파일 경로 생성
            file_path = self.story_audio_dir / filename
            
            # 오디오 데이터 유형에 따라 처리
            if isinstance(audio_data, (str, Path)) and os.path.exists(audio_data):
                # 파일 경로인 경우 복사
                success = copy_file(audio_data, file_path)
            else:
                # 다른 형식은 오류 처리
                logger.error("지원되지 않는 오디오 데이터 형식입니다.")
                return None
                
            if success:
                # 메타데이터 업데이트
                self.media_metadata["audio"][filename] = {
                    "file_path": str(file_path),
                    "chapter_number": chapter_number,
                    "speaker_type": speaker_type,
                    "created_at": datetime.now().isoformat()
                }
                
                logger.info(f"오디오 저장 완료: {file_path}")
                return str(file_path)
            else:
                logger.error(f"오디오 저장 실패: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"오디오 저장 중 오류 발생: {e}")
            return None
    
    def get_chapter_media(self, chapter_number: int) -> Dict:
        """
        특정 챕터의 미디어 정보 조회
        
        Args:
            chapter_number: 챕터 번호
            
        Returns:
            Dict: 챕터별 미디어 정보 (이미지, 오디오)
        """
        result = {
            "images": [],
            "audio": []
        }
        
        # 이미지 정보 조회
        for filename, metadata in self.media_metadata.get("images", {}).items():
            if metadata.get("chapter_number") == chapter_number:
                result["images"].append({
                    "filename": filename,
                    "file_path": metadata.get("file_path"),
                    "created_at": metadata.get("created_at")
                })
        
        # 오디오 정보 조회
        for filename, metadata in self.media_metadata.get("audio", {}).items():
            if metadata.get("chapter_number") == chapter_number:
                result["audio"].append({
                    "filename": filename,
                    "file_path": metadata.get("file_path"),
                    "speaker_type": metadata.get("speaker_type"),
                    "created_at": metadata.get("created_at")
                })
        
        return result
    
    def get_all_media(self) -> Dict:
        """
        모든 미디어 정보 조회
        
        Returns:
            Dict: 모든 미디어 정보
        """
        return self.media_metadata
    
    def save_metadata(self, file_path: Optional[str] = None) -> bool:
        """
        미디어 메타데이터 저장
        
        Args:
            file_path: 저장할 파일 경로 (기본값: 스토리 ID 기반으로 자동 생성)
            
        Returns:
            bool: 성공 여부
        """
        if not self.current_story_id:
            logger.error("현재 스토리 ID가 설정되지 않았습니다.")
            return False
        
        # 파일 경로 생성 (제공되지 않은 경우)
        if not file_path:
            metadata_dir = self.base_output_dir / "metadata"
            ensure_directory(metadata_dir)
            file_path = metadata_dir / f"{self.current_story_id}_media.json"
        
        try:
            # 현재 시간 추가
            self.media_metadata["updated_at"] = datetime.now().isoformat()
            
            # JSON 파일로 저장
            success = save_json(self.media_metadata, file_path)
            
            if success:
                logger.info(f"미디어 메타데이터 저장 완료: {file_path}")
            else:
                logger.error(f"미디어 메타데이터 저장 실패: {file_path}")
                
            return success
            
        except Exception as e:
            logger.error(f"미디어 메타데이터 저장 중 오류 발생: {e}")
            return False
    
    def load_metadata(self, file_path: str) -> bool:
        """
        미디어 메타데이터 로드
        
        Args:
            file_path: 로드할 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            # JSON 파일에서 로드
            metadata = load_json(file_path)
            
            if metadata:
                self.media_metadata = metadata
                
                # 스토리 ID 설정
                story_id = metadata.get("story_id")
                if story_id:
                    self.set_current_story_id(story_id)
                
                logger.info(f"미디어 메타데이터 로드 완료: {file_path}")
                return True
            else:
                logger.error(f"미디어 메타데이터 로드 실패: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"미디어 메타데이터 로드 중 오류 발생: {e}")
            return False
    
    def export_story_package(self, export_dir: Union[str, Path], story_data: Dict) -> bool:
        """
        이야기 패키지 내보내기 (HTML, 이미지, 오디오 등)
        
        Args:
            export_dir: 내보낼 디렉토리
            story_data: 이야기 데이터
            
        Returns:
            bool: 성공 여부
        """
        if not self.current_story_id:
            logger.error("현재 스토리 ID가 설정되지 않았습니다.")
            return False
        
        try:
            # 내보내기 디렉토리 설정
            export_path = Path(export_dir) / self.current_story_id
            ensure_directory(export_path)
            
            # 하위 디렉토리 생성
            images_export_dir = export_path / "images"
            audio_export_dir = export_path / "audio"
            ensure_directory(images_export_dir)
            ensure_directory(audio_export_dir)
            
            # 이미지 파일 복사
            for filename, metadata in self.media_metadata.get("images", {}).items():
                source_path = metadata.get("file_path")
                if source_path and os.path.exists(source_path):
                    dest_path = images_export_dir / filename
                    copy_file(source_path, dest_path)
            
            # 오디오 파일 복사
            for filename, metadata in self.media_metadata.get("audio", {}).items():
                source_path = metadata.get("file_path")
                if source_path and os.path.exists(source_path):
                    dest_path = audio_export_dir / filename
                    copy_file(source_path, dest_path)
            
            # 메타데이터 저장
            metadata_path = export_path / "media_metadata.json"
            save_json(self.media_metadata, metadata_path)
            
            # 이야기 데이터 저장
            story_path = export_path / "story_data.json"
            save_json(story_data, story_path)
            
            # HTML 템플릿 생성 (추후 구현 또는 StoryParser 활용)
            
            logger.info(f"이야기 패키지 내보내기 완료: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"이야기 패키지 내보내기 중 오류 발생: {e}")
            return False 