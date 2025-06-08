"""
음성 생성기
ElevenLabs API를 사용한 챕터별 음성 생성
등장인물별 다른 음성 지원
WebSocket 실시간 스트리밍 지원
"""

import uuid
import asyncio
import json
import base64
import ssl
import websockets
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import aiohttp

# Project imports
from .base_generator import BaseGenerator
from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)

class VoiceGenerator(BaseGenerator):
    """ElevenLabs API 기반 음성 생성기 (등장인물별 음성 지원)"""
    
    def __init__(self, 
                 elevenlabs_api_key: str = None,
                 voice_id: str = "AW5wrnG1jVizOYY7R1Oo",  # Jiyoung (기본 내레이터 음성)
                 model_id: str = "eleven_multilingual_v2", # 기본 모델 ID (한국어 지원)
                 voice_settings: Dict[str, float] = None, # 음성 설정 (stability, similarity_boost, style, use_speaker_boost)
                 temp_storage_path: str = "output/temp/audio", # 임시 저장 경로
                 max_retries: int = 3, # 최대 재시도 횟수
                 character_voice_mapping: Dict[str, str] = None): # 캐릭터별 음성 ID 매핑
        """
        Args:
            elevenlabs_api_key: ElevenLabs API 키
            voice_id: 기본 내레이터 음성 ID
            model_id: 사용할 모델 ID
            voice_settings: 음성 설정 (stability, similarity_boost, style, use_speaker_boost)
            temp_storage_path: 임시 저장 경로
            max_retries: 최대 재시도 횟수
            character_voice_mapping: 캐릭터별 음성 ID 매핑
        """
        super().__init__(max_retries=max_retries, timeout=120.0)
        
        self.api_key = elevenlabs_api_key
        self.narrator_voice_id = voice_id  # 내레이터 음성 ID
        self.model_id = model_id
        self.temp_storage_path = Path(temp_storage_path)
        
        # 기본 음성 설정
        self.voice_settings = voice_settings or {
            "stability": 0.5, # 안정성 (0.0-1.0)
            "similarity_boost": 0.75, # 유사성 증가 (0.0-1.0)
            "style": 0.2, # 스타일 (0.0-1.0)
            "use_speaker_boost": True # 스피커 부스트 (True/False)
        }
        
        # 캐릭터별 음성 매핑 (캐릭터명 -> 음성 ID)
        self.character_voice_mapping = character_voice_mapping or {}
        
        # 기본 캐릭터 타입별 음성 설정
        self.default_character_voices = {
            "narrator": voice_id,  # 내레이터 # 기본 내레이터 음성 ID (Jiyoung)
            "child": "UvkXHIJzOBYWOI51BDKp",  # Jeong-Ah (아이 목소리) 
            "adult_male": "Ir7oQcBXWiq4oFGROCfj",  # Taemin (어른 남성)
            "adult_female": "21m00Tcm4TlvDq8ikWAM",  # Rachel (어른 여성)
            "grandpa" : "IAETYMYM3nJvjnlkVTKI", # Grandpa (할아버지) - Deok su
            "fantasy": "xi3rF0t7dg7uN2M0WUhr",  # Yuna (판타지 캐릭터)
        }
        
        # 캐릭터 타입별 음성 설정
        self.character_voice_settings = {
            "narrator": {
                "stability": 0.6, # 안정성 (0.0-1.0)
                "similarity_boost": 1.0, # 유사성 증가 (0.0-1.0)
                "style": 0.2, # 스타일 (0.0-1.0)
                "use_speaker_boost": True # 스피커 부스트 (True/False)
            },
            "child": {
                "stability": 0.4, # 안정성 (0.0-1.0)
                "similarity_boost": 0.8, # 유사성 증가 (0.0-1.0)
                "style": 0.7, # 스타일 (0.0-1.0)
                "use_speaker_boost": True # 스피커 부스트 (True/False)
            },
            "adult_male": {
                "stability": 0.7, # 안정성 (0.0-1.0)
                "similarity_boost": 0.8, # 유사성 증가 (0.0-1.0)
                "style": 0.4, # 스타일 (0.0-1.0)
                "use_speaker_boost": True # 스피커 부스트 (True/False)
            },
            "adult_female": {
                "stability": 0.6, # 안정성 (0.0-1.0)
                "similarity_boost": 0.8, # 유사성 증가 (0.0-1.0)
                "style": 0.4, # 스타일 (0.0-1.0)
                "use_speaker_boost": True # 스피커 부스트 (True/False)
            },
            "fantasy": {
                "stability": 0.3, # 안정성 (0.0-1.0)
                "similarity_boost": 0.6, # 유사성 증가 (0.0-1.0)
                "style": 0.8, # 스타일 (0.0-1.0)
                "use_speaker_boost": True # 스피커 부스트 (True/False)
            },
            "grandma": {
                "stability": 0.7, # 안정성 (0.0-1.0)
                "similarity_boost": 0.8, # 유사성 증가 (0.0-1.0)
                "style": 0.4, # 스타일 (0.0-1.0)
                "use_speaker_boost": True # 스피커 부스트 (True/False)
            }
        }
        
        # ElevenLabs API 설정
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "Accept": "audio/mp3",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        # 초기화
        self._initialize_components()
    
    def _initialize_components(self):
        """구성 요소 초기화"""
        try:
            # 1. 임시 저장소 생성 (부모 디렉토리까지 포함)
            self.temp_storage_path.mkdir(parents=True, exist_ok=True)
            
            # 2. API 키 확인
            if not self.api_key:
                logger.warning("ElevenLabs API 키가 설정되지 않음")
            else:
                logger.info(f"ElevenLabs API 키 설정 완료")
            
            logger.info(f"VoiceGenerator 초기화 완료 (임시 저장 경로: {self.temp_storage_path})")
            
        except Exception as e:
            logger.error(f"VoiceGenerator 초기화 실패: {e}")
            raise
    
    def set_character_voice_mapping(self, character_mapping: Dict[str, str]):
        """캐릭터별 음성 매핑 설정"""
        self.character_voice_mapping.update(character_mapping)
        logger.info(f"캐릭터 음성 매핑 업데이트: {character_mapping}")
    
    def get_voice_for_character(self, character_name: str, character_type: str = None) -> str:
        """캐릭터에 맞는 음성 ID 반환 (안전한 폴백 지원)"""
        
        # 1. 직접 매핑된 음성이 있는지 확인
        if character_name in self.character_voice_mapping:
            mapped_voice = self.character_voice_mapping[character_name]
            # 유효하지 않은 음성 ID는 기본 음성으로 대체
            if mapped_voice and not mapped_voice.startswith("test_"):
                return mapped_voice
            else:
                logger.warning(f"유효하지 않은 음성 ID ({mapped_voice})를 기본 음성으로 대체합니다.")
        
        # 2. 캐릭터 타입별 기본 음성 사용
        if character_type and character_type in self.default_character_voices:
            return self.default_character_voices[character_type]
        
        # 3. 캐릭터명으로 타입 추정
        character_lower = character_name.lower()
        if any(keyword in character_lower for keyword in ["아이", "어린이", "꼬마", "소년", "소녀", "민준"]): # 아이, 어린이, 꼬마, 소년, 소녀, 주인공 이름
            return self.default_character_voices["child"]
        elif any(keyword in character_lower for keyword in ["아빠", "엄마", "선생님"]): # 아빠, 엄마, 선생님
            if any(keyword in character_lower for keyword in ["아빠", "할아버지"]): # 아빠, 할아버지
                return self.default_character_voices["adult_male"]
            else: # 엄마
                return self.default_character_voices["adult_female"]
        elif any(keyword in character_lower for keyword in ["할아버지"]): # 할아버지
            return self.default_character_voices["grandpa"]
        elif any(keyword in character_lower for keyword in ["동물", "요정", "마법사", "용", "공주", "외계인", "로봇"]):
            return self.default_character_voices["fantasy"]
        
        # 4. 기본값: 아이 음성 (동화의 주인공은 보통 아이)
        return self.default_character_voices["child"]
    
    def get_voice_settings_for_character(self, character_name: str, character_type: str = None) -> Dict[str, Any]:
        """캐릭터에 맞는 음성 설정 반환"""
        
        # 캐릭터 타입 결정
        if character_type and character_type in self.character_voice_settings:
            return self.character_voice_settings[character_type]
        
        # 캐릭터명으로 타입 추정
        character_lower = character_name.lower()
        if any(keyword in character_lower for keyword in ["아이", "어린이", "꼬마", "소년", "소녀"]):
            return self.character_voice_settings["child"]
        elif any(keyword in character_lower for keyword in ["아빠", "할아버지"]):
            return self.character_voice_settings["adult_male"]
        elif any(keyword in character_lower for keyword in ["엄마", "할머니", "선생님"]):
            return self.character_voice_settings["adult_female"]
        elif any(keyword in character_lower for keyword in ["동물", "요정", "마법사", "용", "공주"]):
            return self.character_voice_settings["fantasy"]
        
        # 기본값: 내레이터 설정
        return self.character_voice_settings["narrator"]

    async def generate(self, 
                      input_data: Dict[str, Any], 
                      progress_callback: Optional[Callable] = None,
                      use_websocket: bool = True) -> Dict[str, Any]:
        """
        챕터별 음성 생성 (등장인물별 다른 음성 지원, WebSocket 스트리밍 지원)
        
        Args:
            input_data: {
                "story_data": {
                    "title": "동화 제목",
                    "chapters": [
                        {
                            "chapter_number": 1,
                            "chapter_title": "챕터 제목",
                            "narration": "내레이션 텍스트",
                            "dialogues": [
                                {"speaker": "캐릭터명", "text": "대사 내용"},
                                ...
                            ]
                        }
                    ],
                    "characters": [
                        {"name": "캐릭터명", "type": "child/adult_male/adult_female/fantasy"}
                    ]
                },
                "story_id": "스토리 ID",
                "voice_settings": {  # 선택적
                    "character_voice_mapping": {"캐릭터명": "음성ID"},
                    "model_id": "모델 ID"
                }
            }
            progress_callback: 진행 상황 콜백
            use_websocket: WebSocket 스트리밍 사용 여부 (기본 True)
            
        Returns:
            {
                "audio_files": [
                    {
                        "chapter_number": 1,
                        "narration_audio": "내레이션 오디오 파일 경로",
                        "dialogue_audios": [
                            {
                                "speaker": "캐릭터명",
                                "audio_path": "대사 오디오 파일 경로",
                                "text": "대사 내용"
                            }
                        ],
                        "combined_audio": "통합 오디오 파일 경로",
                        "generation_time": 생성 시간
                    }
                ],
                "metadata": {
                    "total_audio_files": 생성된 오디오 파일 수,
                    "characters_used": ["사용된 캐릭터 목록"],
                    "voice_mapping": {"캐릭터": "음성ID"},
                    "total_generation_time": 총 생성 시간,
                    "websocket_used": WebSocket 사용 여부
                }
            }
        """
        
        task_id = str(uuid.uuid4())
        self.current_task_id = task_id
        
        # WebSocket 사용 여부 결정
        use_websocket_streaming = use_websocket and bool(self.api_key)
        if use_websocket and not self.api_key:
            logger.warning("ElevenLabs API 키가 없어 WebSocket 스트리밍을 사용할 수 없습니다. HTTP API를 사용합니다.")
        
        try:
            story_data = input_data.get("story_data", {})
            story_id = input_data.get("story_id", task_id)
            chapters = story_data.get("chapters", [])
            characters = story_data.get("characters", [])
            
            # 음성 설정 오버라이드
            voice_settings = input_data.get("voice_settings", {})
            if "character_voice_mapping" in voice_settings:
                self.set_character_voice_mapping(voice_settings["character_voice_mapping"])
            
            current_model_id = voice_settings.get("model_id", self.model_id)
            
            if not chapters:
                raise ValueError("생성할 챕터가 없습니다")
            
            if progress_callback:
                await progress_callback({
                    "step": "voice_generation",
                    "status": "starting",
                    "total_chapters": len(chapters),
                    "task_id": task_id,
                    "websocket_mode": use_websocket_streaming
                })
            
            generated_audio = []
            characters_used = set()
            
            # 각 챕터별로 음성 생성
            for i, chapter in enumerate(chapters):
                chapter_start_time = asyncio.get_event_loop().time()
                
                if progress_callback:
                    await progress_callback({
                        "step": "voice_generation",
                        "status": "processing_chapter",
                        "current_chapter": i + 1,
                        "total_chapters": len(chapters),
                        "chapter_title": chapter.get("chapter_title", ""),
                        "websocket_mode": use_websocket_streaming
                    })
                
                # 챕터별 음성 생성 (WebSocket 또는 HTTP)
                if use_websocket_streaming:
                    chapter_audio = await self._generate_chapter_audio_websocket(
                        chapter=chapter,
                        story_id=story_id,
                        progress_callback=progress_callback
                    )
                else:
                    chapter_audio = await self._generate_chapter_audio_http(
                        chapter=chapter,
                        story_id=story_id,
                        model_id=current_model_id
                    )
                
                # 사용된 캐릭터 추적
                if "dialogue_audios" in chapter_audio:
                    for dialogue in chapter_audio["dialogue_audios"]:
                        characters_used.add(dialogue["speaker"])
                
                chapter_generation_time = asyncio.get_event_loop().time() - chapter_start_time
                chapter_audio["generation_time"] = chapter_generation_time
                
                generated_audio.append(chapter_audio)
                
                if progress_callback:
                    await progress_callback({
                        "step": "voice_generation",
                        "status": "chapter_completed",
                        "current_chapter": i + 1,
                        "total_chapters": len(chapters),
                        "generation_time": chapter_generation_time,
                        "websocket_mode": use_websocket_streaming
                    })
            
            if progress_callback:
                await progress_callback({
                    "step": "voice_generation",
                    "status": "completed",
                    "total_audio_files": len(generated_audio),
                    "websocket_mode": use_websocket_streaming
                })
            
            return {
                "audio_files": generated_audio,
                "metadata": {
                    "total_audio_files": len(generated_audio),
                    "characters_used": list(characters_used),
                    "voice_mapping": self.character_voice_mapping,
                    "total_generation_time": self.total_generation_time,
                    "story_id": story_id,
                    "task_id": task_id,
                    "websocket_used": use_websocket_streaming
                }
            }
            
        except Exception as e:
            logger.error(f"음성 생성 실패 (task_id: {task_id}): {e}")
            # WebSocket 실패 시 HTTP로 폴백
            if use_websocket_streaming:
                logger.info("WebSocket 실패 - HTTP API로 폴백 시도")
                return await self.generate(input_data, progress_callback, use_websocket=False)
            raise
    
    async def _generate_chapter_audio_http(self, 
                                         chapter: Dict[str, Any], 
                                         story_id: str,
                                         model_id: str) -> Dict[str, Any]:
        """HTTP API를 사용한 단일 챕터 음성 생성"""
        
        chapter_number = chapter.get("chapter_number", 1)
        narration = chapter.get("narration", "")
        dialogues = chapter.get("dialogues", [])
        
        result = {
            "chapter_number": chapter_number,
            "narration_audio": None,
            "dialogue_audios": [],
            "combined_audio": None
        }
        
        # 1. 내레이션 음성 생성
        if narration:
            narration_audio = await self._generate_single_audio_http(
                text=narration,
                voice_id=self.narrator_voice_id,
                voice_settings=self.character_voice_settings["narrator"],
                model_id=model_id,
                filename_prefix=f"narration_ch{chapter_number}_{story_id[:8]}"
            )
            result["narration_audio"] = str(narration_audio)
        
        # 2. 대사별 음성 생성
        for i, dialogue in enumerate(dialogues):
            speaker = dialogue.get("speaker", "unknown")
            text = dialogue.get("text", "")
            
            if text:
                # 캐릭터에 맞는 음성 ID와 설정 가져오기
                voice_id = self.get_voice_for_character(speaker)
                voice_settings = self.get_voice_settings_for_character(speaker)
                
                dialogue_audio = await self._generate_single_audio_http(
                    text=text,
                    voice_id=voice_id,
                    voice_settings=voice_settings,
                    model_id=model_id,
                    filename_prefix=f"dialogue_ch{chapter_number}_{i}_{story_id[:8]}"
                )
                
                result["dialogue_audios"].append({
                    "speaker": speaker,
                    "audio_path": str(dialogue_audio),
                    "text": text,
                    "voice_id": voice_id
                })
        
        return result
    
    async def _generate_chapter_audio_websocket(self, 
                                              chapter: Dict[str, Any], 
                                              story_id: str,
                                              progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """WebSocket을 사용한 단일 챕터 스트리밍 음성 생성"""
        
        chapter_number = chapter.get("chapter_number", 1)
        narration = chapter.get("narration", "")
        dialogues = chapter.get("dialogues", [])
        
        result = {
            "chapter_number": chapter_number,
            "narration_audio": None,
            "dialogue_audios": [],
            "combined_audio": None,
            "streaming_metadata": {
                "chunks_received": 0,
                "total_bytes": 0,
                "websocket_used": True
            }
        }
        
        # 1. 내레이션 스트리밍 음성 생성
        if narration:
            narration_audio, streaming_info = await self._generate_single_audio_websocket(
                text=narration,
                voice_id=self.narrator_voice_id,
                voice_settings=self.character_voice_settings["narrator"],
                filename_prefix=f"narration_ch{chapter_number}_{story_id[:8]}",
                progress_callback=progress_callback
            )
            result["narration_audio"] = str(narration_audio)
            result["streaming_metadata"]["narration_chunks"] = streaming_info["chunks_received"]
        
        # 2. 대사별 스트리밍 음성 생성
        for i, dialogue in enumerate(dialogues):
            speaker = dialogue.get("speaker", "unknown")
            text = dialogue.get("text", "")
            
            if text:
                # 캐릭터에 맞는 음성 ID와 설정 가져오기
                voice_id = self.get_voice_for_character(speaker)
                voice_settings = self.get_voice_settings_for_character(speaker)
                
                try:
                    dialogue_audio, streaming_info = await self._generate_single_audio_websocket(
                        text=text,
                        voice_id=voice_id,
                        voice_settings=voice_settings,
                        filename_prefix=f"dialogue_ch{chapter_number}_{i}_{story_id[:8]}",
                        progress_callback=progress_callback
                    )
                    
                    result["dialogue_audios"].append({
                        "speaker": speaker,
                        "audio_path": str(dialogue_audio),
                        "text": text,
                        "voice_id": voice_id,
                        "streaming_info": streaming_info
                    })
                    
                    result["streaming_metadata"]["chunks_received"] += streaming_info["chunks_received"]
                    result["streaming_metadata"]["total_bytes"] += streaming_info["total_bytes"]
                    
                except Exception as e:
                    logger.warning(f"WebSocket 대사 생성 실패 ({speaker}), HTTP로 폴백: {e}")
                    # WebSocket 실패 시 HTTP로 폴백
                    dialogue_audio = await self._generate_single_audio_http(
                        text=text,
                        voice_id=voice_id,
                        voice_settings=voice_settings,
                        model_id=self.model_id,
                        filename_prefix=f"dialogue_ch{chapter_number}_{i}_{story_id[:8]}"
                    )
                    
                    result["dialogue_audios"].append({
                        "speaker": speaker,
                        "audio_path": str(dialogue_audio),
                        "text": text,
                        "voice_id": voice_id,
                        "fallback_used": True
                    })
        
        return result
    
    def _prepare_text_for_speech(self, text: str, speaker_type: str = "narrator") -> str:
        """화자 타입에 맞게 텍스트를 음성 생성용으로 준비"""
        
        # 기본 텍스트 정리
        text = self._clean_text_for_speech(text)
        
        # 화자 타입별 텍스트 조정
        if speaker_type == "child":
            # 아이 캐릭터: 더 활기찬 표현
            text = text.replace(".", "!")
            text = text.replace("요.", "요!")
        elif speaker_type == "fantasy":
            # 판타지 캐릭터: 신비로운 느낌 추가
            text = text.replace("말했습니다", "속삭였습니다")
        
        # ElevenLabs 텍스트 길이 제한 확인
        if len(text) > 5000:
            logger.warning(f"텍스트가 너무 깁니다 ({len(text)}자). 5000자로 자릅니다.")
            text = text[:4997] + "..."
        
        return text
    
    def _clean_text_for_speech(self, text: str) -> str:
        """음성 생성을 위한 텍스트 정리"""
        
        # 기본 정리
        text = text.strip()
        
        # 특수 문자 처리
        replacements = {
            "**": "",  # 마크다운 볼드 제거
            "*": "",   # 마크다운 이탤릭 제거
            "_": "",   # 언더스코어 제거
            "#": "",   # 해시태그 제거
            "`": "",   # 백틱 제거
            "---": ".",  # 구분선을 마침표로
            "...": ".",  # 말줄임표 정리
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # 연속된 공백 정리
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # 연속된 마침표 정리
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()
    
    async def _generate_single_audio_http(self, 
                                         text: str, 
                                         voice_id: str,
                                         voice_settings: Dict[str, Any],
                                         model_id: str,
                                         filename_prefix: str) -> Path:
        """HTTP API를 사용한 단일 오디오 생성"""
        
        try:
            # API 요청 데이터
            data = {
                "text": text,
                "model_id": model_id,
                "voice_settings": voice_settings
            }
            
            # ElevenLabs API 호출
            url = f"{self.base_url}/text-to-speech/{voice_id}"
            
            # SSL 컨텍스트 설정
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            timeout = aiohttp.ClientTimeout(total=120)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=self.headers, json=data, ssl=ssl_context) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"ElevenLabs API 오류 ({response.status}): {error_text}")
                    
                    # 오디오 데이터 읽기
                    audio_data = await response.read()
            
            # 파일 저장
            audio_filename = f"{filename_prefix}.mp3"
            
            # story_id별 폴더 생성 (filename_prefix에서 story_id 추출)
            if "_" in filename_prefix:
                # filename_prefix 예시: "dialogue_ch1_0_fcc21cbb" 또는 "narration_ch1_fcc21cbb"
                story_id_short = filename_prefix.split("_")[-1]  # "fcc21cbb" 추출
                story_folder = self.temp_storage_path / story_id_short
                story_folder.mkdir(parents=True, exist_ok=True)
                audio_path = story_folder / audio_filename
            else:
                # fallback: 기존 방식
                audio_path = self.temp_storage_path / audio_filename
            
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
            
            logger.info(f"HTTP 음성 생성 완료: {audio_path} ({len(audio_data)} bytes)")
            return audio_path
            
        except Exception as e:
            logger.error(f"HTTP 음성 생성 실패 ({filename_prefix}): {e}")
            raise
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """사용 가능한 음성 목록 조회"""
        
        try:
            url = f"{self.base_url}/voices"
            headers = {"xi-api-key": self.api_key}
            
            # SSL 컨텍스트 설정
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            timeout = aiohttp.ClientTimeout(total=60)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, ssl=ssl_context) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"음성 목록 조회 실패 ({response.status}): {error_text}")
                    
                    data = await response.json()
                    return data.get("voices", [])
                    
        except Exception as e:
            logger.error(f"음성 목록 조회 실패: {e}")
            return []
    
    async def health_check(self) -> bool:
        """VoiceGenerator 상태 확인"""
        
        try:
            # 기본 상태 확인
            if not await super().health_check():
                return False
            
            # API 키 확인
            if not self.api_key:
                logger.error("ElevenLabs API 키가 설정되지 않음")
                return False
            
            # 임시 저장소 확인
            if not self.temp_storage_path.exists():
                logger.error(f"임시 저장소가 존재하지 않음: {self.temp_storage_path}")
                return False
            
            # API 연결 테스트 (음성 목록 조회)
            try:
                voices = await self.get_available_voices()
                return len(voices) > 0
                
            except Exception as e:
                logger.error(f"ElevenLabs API 연결 테스트 실패: {e}")
                return False
                
        except Exception as e:
            logger.error(f"VoiceGenerator health check 실패: {e}")
            return False
    
    def get_character_voice_template(self) -> Dict[str, Any]:
        """캐릭터별 음성 설정 템플릿 반환"""
        return {
            "character_voice_mapping": {
                "주인공": "EXAVITQu4vr4xnSDxMaL",  # 아이 목소리
                "엄마": "21m00Tcm4TlvDq8ikWAM",     # 여성 목소리
                "아빠": "VR6AewLTigWG4xSOukaG",     # 남성 목소리
                "요정": "pNInz6obpgDQGcFmaJgB",      # 판타지 목소리
                "내레이션": "xi3rF0t7dg7uN2M0WUhr"    # Yuna (기본 내레이터 음성)
            },
            "character_types": {
                "child": "아이 캐릭터",
                "adult_male": "어른 남성",
                "adult_female": "어른 여성", 
                "fantasy": "판타지 캐릭터",
                "narrator": "내레이터"
            }
        }
    
    def estimate_generation_time(self, text_length: int, num_characters: int = 1) -> float:
        """예상 생성 시간 계산 (초) - 캐릭터 수 고려"""
        # ElevenLabs의 평균 생성 시간을 기반으로 추정
        # 대략 1000자당 10초 정도, 캐릭터 수만큼 배수
        avg_time_per_1000_chars = 10.0
        base_time = (text_length / 1000) * avg_time_per_1000_chars
        return base_time * num_characters

    async def _generate_single_audio_websocket(self, 
                                             text: str, 
                                             voice_id: str,
                                             voice_settings: Dict[str, Any],
                                             filename_prefix: str,
                                             progress_callback: Optional[Callable] = None) -> tuple[Path, Dict[str, Any]]:
        """WebSocket을 사용한 단일 오디오 스트리밍 생성"""
        
        try:
            # WebSocket URI 구성
            model_id = "eleven_flash_v2_5"  # 낮은 지연시간을 위한 모델
            uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}"
            
            # 오디오 파일 경로 설정
            audio_filename = f"{filename_prefix}.mp3"
            
            # story_id별 폴더 생성 (filename_prefix에서 story_id 추출)
            if "_" in filename_prefix:
                # filename_prefix 예시: "dialogue_ch1_0_fcc21cbb" 또는 "narration_ch1_fcc21cbb"
                story_id_short = filename_prefix.split("_")[-1]  # "fcc21cbb" 추출
                story_folder = self.temp_storage_path / story_id_short
                story_folder.mkdir(parents=True, exist_ok=True)
                audio_path = story_folder / audio_filename
            else:
                # fallback: 기존 방식
                audio_path = self.temp_storage_path / audio_filename
            
            # 스트리밍 정보 추적
            streaming_info = {
                "chunks_received": 0,
                "total_bytes": 0,
                "first_chunk_time": None,
                "total_time": None
            }
            
            start_time = asyncio.get_event_loop().time()
            
            # SSL 컨텍스트 설정
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # WebSocket 연결 및 스트리밍
            async with websockets.connect(
                uri, 
                extra_headers={"xi-api-key": self.api_key},
                ssl=ssl_context
            ) as websocket:
                
                # 1. 초기화 메시지 전송 (음성 설정)
                init_message = {
                    "text": " ",  # 공백으로 연결 유지
                    "voice_settings": voice_settings,
                    "generation_config": {
                        "chunk_length_schedule": [120, 160, 250, 290]  # 기본 설정
                    }
                }
                await websocket.send(json.dumps(init_message))
                
                # 2. 텍스트 전송
                text_message = {"text": text}
                await websocket.send(json.dumps(text_message))
                
                # 3. 종료 신호 전송
                end_message = {"text": ""}
                await websocket.send(json.dumps(end_message))
                
                # 4. 오디오 데이터 수신 및 저장
                audio_chunks = []
                
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        if data.get("audio"):
                            # 첫 번째 청크 시간 기록
                            if streaming_info["first_chunk_time"] is None:
                                streaming_info["first_chunk_time"] = asyncio.get_event_loop().time() - start_time
                            
                            # Base64 디코딩 및 저장
                            audio_chunk = base64.b64decode(data["audio"])
                            audio_chunks.append(audio_chunk)
                            
                            streaming_info["chunks_received"] += 1
                            streaming_info["total_bytes"] += len(audio_chunk)
                            
                            # 진행 상황 콜백
                            if progress_callback:
                                await progress_callback({
                                    "step": "websocket_audio_streaming",
                                    "status": "chunk_received",
                                    "chunk_number": streaming_info["chunks_received"],
                                    "chunk_size": len(audio_chunk),
                                    "voice_id": voice_id,
                                    "filename": filename_prefix
                                })
                        
                        elif data.get('isFinal'):
                            logger.info(f"WebSocket 스트리밍 완료: {filename_prefix}")
                            break
                            
                    except websockets.exceptions.ConnectionClosed:
                        logger.info(f"WebSocket 연결 종료: {filename_prefix}")
                        break
                
                # 5. 전체 오디오 파일 저장
                with open(audio_path, 'wb') as f:
                    for chunk in audio_chunks:
                        f.write(chunk)
                
                streaming_info["total_time"] = asyncio.get_event_loop().time() - start_time
                
                # 안전한 로그 출력 (None 값 처리)
                ttfb_str = f"{streaming_info['first_chunk_time']:.2f}s" if streaming_info['first_chunk_time'] is not None else "N/A"
                logger.info(f"WebSocket 음성 생성 완료: {audio_path} "
                          f"({streaming_info['chunks_received']} chunks, "
                          f"{streaming_info['total_bytes']} bytes, "
                          f"TTFB: {ttfb_str})")
                
                return audio_path, streaming_info
                
        except Exception as e:
            logger.error(f"WebSocket 음성 생성 실패 ({filename_prefix}): {e}")
            # WebSocket 실패 시 HTTP API로 폴백
            logger.info(f"WebSocket 실패 - HTTP API로 폴백: {filename_prefix}")
            fallback_path = await self._generate_single_audio_http(
                text=text,
                voice_id=voice_id,
                voice_settings=voice_settings,
                model_id="eleven_multilingual_v2",
                filename_prefix=filename_prefix
            )
            return fallback_path, {"websocket_fallback": True, "chunks_received": 0, "total_bytes": 0}
