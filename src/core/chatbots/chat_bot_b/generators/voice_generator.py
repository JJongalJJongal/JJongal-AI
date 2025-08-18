"""
음성 생성기
ElevenLabs API를 사용한 챕터별 음성 생성
등장인물별 다른 음성 지원
WebSocket 실시간 스트리밍 지원
"""

# VoiceGenerator: 한국어 동화 최적화 ElevenLabs TTS 생성기
# 
# 최적화 설정:
# - Model: eleven_turbo_v2_5 (한국어 지원 + 고품질 + 저지연 250-300ms)
# - Voice Settings: 동화 캐릭터별 맞춤 설정 (stability, similarity_boost, style)
# - Text Processing: 한국어 감정 표현 최적화 + ElevenLabs 프롬프팅 가이드 적용
# - Audio Format: 44.1kHz WAV (고품질) + 스트리밍 최적화
# - Character Mapping: 내레이터, 아이, 어른, 판타지 캐릭터별 전용 음성
# - Audio Chunking: 큰 음성 파일을 문장 단위로 분할하여 순차 재생 지원

import uuid
import asyncio
import json
import base64
import ssl
import websockets
import re
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import aiohttp

# Project imports
from .base_generator import BaseGenerator
from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

class VoiceGenerator(BaseGenerator):
    """ElevenLabs API 기반 음성 생성기 (등장인물별 음성 지원 + 청킹 최적화)"""
    
    def __init__(self, 
                 elevenlabs_api_key: str = None,
                 voice_id: str = "AW5wrnG1jVizOYY7R1Oo",  # Jiyoung (기본 내레이터 음성)
                 model_id: str = "eleven_multilingual_v2", # 한국어 최적화 모델 (품질 우선)
                 voice_settings: Dict[str, float] = None, # 음성 설정 (stability, similarity_boost, style, use_speaker_boost)
                 temp_storage_path: str = "output/temp/audio", # 임시 저장 경로
                 max_retries: int = 3, # 최대 재시도 횟수
                 character_voice_mapping: Dict[str, str] = None, # 캐릭터별 음성 ID 매핑
                 enable_chunking: bool = True, # 텍스트 청킹 활성화
                 max_chunk_length: int = 500): # 청크 최대 길이 (문자 수)
        """
        Args:
            elevenlabs_api_key: ElevenLabs API 키
            voice_id: 기본 내레이터 음성 ID
            model_id: 사용할 모델 ID
            voice_settings: 음성 설정 (stability, similarity_boost, style, use_speaker_boost)
            temp_storage_path: 임시 저장 경로
            max_retries: 최대 재시도 횟수
            character_voice_mapping: 캐릭터별 음성 ID 매핑
            enable_chunking: 텍스트 청킹 활성화 (큰 오디오 파일 방지)
            max_chunk_length: 청크 최대 길이 (문자 수)
        """
        super().__init__(max_retries=max_retries, timeout=120.0)
        
        self.api_key = elevenlabs_api_key
        self.narrator_voice_id = voice_id  # 내레이터 음성 ID
        self.model_id = model_id
        self.temp_storage_path = Path(temp_storage_path)
        
        # 청킹 설정
        self.enable_chunking = enable_chunking
        self.max_chunk_length = max_chunk_length
        
        # 한국어 동화 최적화 음성 설정 (ElevenLabs 공식 가이드 기반)
        self.voice_settings = voice_settings or {
            "stability": 0.60,  # 한국어 동화 내레이션: 약간 높은 안정성 (한국어 특성 고려)
            "similarity_boost": 0.90,  # 한국어 발음 명확성 최대화 (0.85 → 0.90)
            "style": 0.12,  # 한국어 동화 톤: 자연스럽게 조정 (0.15 → 0.12)
            "use_speaker_boost": True  # 화자 특성 강화
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
        
        # 한국어 동화 캐릭터별 최적화 음성 설정
        self.character_voice_settings = {
            "narrator": {
                "stability": 0.55,  # 한국어 내레이터: 안정성과 감정 표현 균형 (0.50 → 0.55)
                "similarity_boost": 0.90,  # 한국어 발음 명확성 최대화 (0.85 → 0.90)
                "style": 0.18,  # 한국어 동화책 읽기 스타일 (0.20 → 0.18)
                "use_speaker_boost": True
            },
            "child": {
                "stability": 0.40,  # 한국어 아이: 활발함과 이해도 균형 (0.35 → 0.40)
                "similarity_boost": 0.85,  # 한국어 아이 발음 명확성 향상 (0.75 → 0.85)
                "style": 0.22,  # 한국어 천진난만함 조정 (0.25 → 0.22)
                "use_speaker_boost": True
            },
            "adult_male": {
                "stability": 0.65,  # 한국어 어른 남성: 신뢰감 강화 (0.60 → 0.65)
                "similarity_boost": 0.88,  # 한국어 남성 발음 명확성 (0.80 → 0.88)
                "style": 0.08,  # 한국어 자연스러운 스타일 (0.10 → 0.08)
                "use_speaker_boost": True
            },
            "adult_female": {
                "stability": 0.55,  # 한국어 어른 여성: 따뜻함과 안정성 (0.50 → 0.55)
                "similarity_boost": 0.88,  # 한국어 여성 발음 명확성 (0.80 → 0.88)
                "style": 0.12,  # 한국어 따뜻한 스타일 (0.15 → 0.12)
                "use_speaker_boost": True
            },
            "fantasy": {
                "stability": 0.35,  # 한국어 판타지: 창의성과 이해도 균형 (0.30 → 0.35)
                "similarity_boost": 0.80,  # 한국어 판타지 명확성 향상 (0.70 → 0.80)
                "style": 0.30,  # 한국어 극적 스타일 조정 (0.35 → 0.30)
                "use_speaker_boost": True
            },
            "grandpa": {
                "stability": 0.70,  # 한국어 할아버지: 존경스러운 안정감 (0.65 → 0.70)
                "similarity_boost": 0.92,  # 한국어 노인 발음 최대 명확성 (0.85 → 0.92)
                "style": 0.08,  # 한국어 자상한 스타일 (0.10 → 0.08)
                "use_speaker_boost": True
            },
            "magical": {  # 한국어 마법 캐릭터
                "stability": 0.30,  # 한국어 마법: 표현력과 이해도 균형 (0.25 → 0.30)
                "similarity_boost": 0.80,  # 한국어 마법적 명확성 (0.75 → 0.80)
                "style": 0.35,  # 한국어 마법적 스타일 (0.40 → 0.35)
                "use_speaker_boost": True
            }
        }
        
        # ElevenLabs API 설정
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "Accept": "audio/wav",
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

    def _get_speaker_type_for_character(self, character_name: str) -> str:
        """캐릭터명을 기반으로 speaker_type 결정"""
        
        # 캐릭터명을 소문자로 변환하여 분석
        char_lower = character_name.lower()
        
        # 특정 키워드 기반 매핑
        if any(keyword in char_lower for keyword in ["아이", "child", "소년", "소녀", "어린이", "kids"]):
            return "child"
        elif any(keyword in char_lower for keyword in ["할아버지", "할머니", "grandpa", "grandma", "어르신"]):
            return "grandpa"
        elif any(keyword in char_lower for keyword in ["마법사", "요정", "magic", "fairy", "fantasy", "마법", "요술"]):
            return "fantasy"
        elif any(keyword in char_lower for keyword in ["아빠", "아버지", "father", "dad", "남자", "man", "형", "오빠"]):
            return "adult_male"
        elif any(keyword in char_lower for keyword in ["엄마", "어머니", "mother", "mom", "여자", "woman", "누나", "언니"]):
            return "adult_female"
        elif "내레이" in char_lower or "narrator" in char_lower:
            return "narrator"
        else:
            # 기본값은 narrator
            return "narrator"

    async def generate(self, 
                      input_data: Dict[str, Any], 
                      progress_callback: Optional[Callable] = None,
                      use_websocket: bool = True) -> Dict[str, Any]:
        """
        한국어 동화 음성 생성 메인 함수
        
        Args:
            input_data: 동화 데이터 (chapters 포함)
            progress_callback: 진행 상황 콜백 함수
            use_websocket: WebSocket 스트리밍 사용 여부
            
        Returns:
            생성된 음성 파일 정보와 메타데이터
        """
        
        try:
            task_id = str(uuid.uuid4())
            story_id = input_data.get("story_id", task_id)
            chapters = input_data.get("chapters", [])
            
            if not chapters:
                raise ValueError("생성할 챕터가 없습니다.")
            
            if progress_callback:
                await progress_callback({
                    "step": "voice_generation_start",
                    "status": "started",
                    "task_id": task_id,
                    "total_chapters": len(chapters),
                    "chunking_enabled": self.enable_chunking,
                    "model_id": self.model_id
                })
            
            # 모델 ID 최적화 (한국어 동화 전용)
            use_websocket_streaming = use_websocket and self.api_key
            model_id = self.model_id
            
            # ElevenLabs v3 Turbo 사용 가능 시 업그레이드
            if "eleven_turbo_v2_5" in model_id:
                logger.info("ElevenLabs Turbo v2.5 모델 사용 (한국어 최적화)")
            
            all_audio_files = []
            
            # 챕터별 음성 생성
            for chapter_idx, chapter in enumerate(chapters):
                
                if progress_callback:
                    await progress_callback({
                        "step": "chapter_processing",
                        "status": "processing",
                        "current_chapter": chapter_idx + 1,
                        "total_chapters": len(chapters),
                        "chapter_data": {
                            "chapter_number": chapter.get("chapter_number", chapter_idx + 1),
                            "has_narration": bool(chapter.get("narration")),
                            "dialogue_count": len(chapter.get("dialogues", []))
                        }
                    })
                
                # 챕터별 음성 생성 (청킹 지원)
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
                        model_id=model_id
                    )
                
                all_audio_files.append(chapter_audio)
                
                if progress_callback:
                    await progress_callback({
                        "step": "chapter_completed",
                        "status": "completed",
                        "current_chapter": chapter_idx + 1,
                        "total_chapters": len(chapters),
                        "chapter_result": {
                            "chapter_number": chapter_audio["chapter_number"],
                            "narration_chunks": len(chapter_audio.get("narration_audio_chunks", [])),
                            "dialogue_chunks": len(chapter_audio.get("dialogue_audio_chunks", [])),
                            "total_audio_files": (
                                len(chapter_audio.get("narration_audio_chunks", [])) + 
                                sum(len(d.get("chunks", [])) for d in chapter_audio.get("dialogue_audio_chunks", []))
                            )
                        }
                    })
            
            # 오디오 매니페스트 생성 (순서 보장 및 프론트엔드 지원)
            audio_manifest = self._create_audio_manifest(all_audio_files, story_id)
            
            # 매니페스트 파일 저장
            story_folder = self.temp_storage_path / story_id[:8]
            story_folder.mkdir(parents=True, exist_ok=True)
            manifest_path = story_folder / "audio_manifest.json"
            
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(audio_manifest, f, ensure_ascii=False, indent=2)
            
            # 성공 결과 반환
            result = {
                "task_id": task_id,
                "story_id": story_id,
                "status": "completed",
                "audio_files": all_audio_files,
                "audio_manifest": audio_manifest,
                "manifest_path": str(manifest_path),
                "total_chapters": len(chapters),
                "chunking_enabled": self.enable_chunking,
                "model_used": model_id,
                "generation_method": "websocket" if use_websocket_streaming else "http",
                "summary": {
                    "total_narration_chunks": sum(len(ch.get("narration_audio_chunks", [])) for ch in all_audio_files),
                    "total_dialogue_chunks": sum(len(ch.get("dialogue_audio_chunks", [])) for ch in all_audio_files),
                    "total_audio_files": audio_manifest["total_files"],
                    "estimated_duration": audio_manifest["total_duration_estimate"]
                }
            }
            
            if progress_callback:
                await progress_callback({
                    "step": "voice_generation_completed",
                    "status": "completed",
                    "task_id": task_id,
                    "result": result
                })
            
            logger.info(f"동화 음성 생성 완료 (Task: {task_id}): {result['summary']}")
            
            return result
            
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
        """HTTP API를 사용한 단일 챕터 음성 생성 (청킹 지원)"""
        
        chapter_number = chapter.get("chapter_number", 1)
        narration = chapter.get("narration", "")
        dialogues = chapter.get("dialogues", [])
        
        result = {
            "chapter_number": chapter_number,
            "narration_audio": None,
            "narration_audio_chunks": [],
            "dialogue_audios": [],
            "dialogue_audio_chunks": [],
            "combined_audio": None
        }
        
        # 1. 내레이션 음성 생성 (청킹 지원)
        if narration:
            if self.enable_chunking:
                # 청킹된 내레이션 생성
                narration_chunks = await self._generate_chunked_audio(
                    text=narration,
                    voice_id=self.narrator_voice_id,
                    voice_settings=self.character_voice_settings["narrator"],
                    filename_prefix=f"narration_ch{chapter_number}_{story_id[:8]}",
                    speaker_type="narrator",
                    use_websocket=False,
                    progress_callback=None
                )
                result["narration_audio_chunks"] = narration_chunks
                # 첫 번째 청크를 기본 narration_audio로 설정 (하위 호환성)
                if narration_chunks:
                    result["narration_audio"] = narration_chunks[0]["audio_path"]
            else:
                # 기존 방식 (청킹 비활성화)
                narration_audio = await self._generate_single_audio_http(
                    text=narration,
                    voice_id=self.narrator_voice_id,
                    voice_settings=self.character_voice_settings["narrator"],
                    model_id=model_id,
                    filename_prefix=f"narration_ch{chapter_number}_{story_id[:8]}"
                )
                result["narration_audio"] = str(narration_audio)
        
        # 2. 대사별 음성 생성 (청킹 지원)
        for i, dialogue in enumerate(dialogues):
            speaker = dialogue.get("speaker", "unknown")
            text = dialogue.get("text", "")
            
            if text:
                # 캐릭터에 맞는 음성 ID와 설정 가져오기
                voice_id = self.get_voice_for_character(speaker)
                voice_settings = self.get_voice_settings_for_character(speaker)
                
                # speaker_type 결정
                speaker_type = self._get_speaker_type_for_character(speaker)
                
                if self.enable_chunking:
                    # 청킹된 대사 생성
                    dialogue_chunks = await self._generate_chunked_audio(
                        text=text,
                        voice_id=voice_id,
                        voice_settings=voice_settings,
                        filename_prefix=f"dialogue_ch{chapter_number}_{i}_{story_id[:8]}",
                        speaker_type=speaker_type,
                        use_websocket=False,
                        progress_callback=None
                    )
                    
                    result["dialogue_audio_chunks"].append({
                        "speaker": speaker,
                        "voice_id": voice_id,
                        "chunks": dialogue_chunks
                    })
                    
                    # 첫 번째 청크를 기본 dialogue_audios에도 추가 (하위 호환성)
                    if dialogue_chunks:
                        result["dialogue_audios"].append({
                            "speaker": speaker,
                            "audio_path": dialogue_chunks[0]["audio_path"],
                            "text": text,
                            "voice_id": voice_id
                        })
                else:
                    # 기존 방식 (청킹 비활성화)
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
        """WebSocket을 사용한 단일 챕터 스트리밍 음성 생성 (청킹 지원)"""
        
        chapter_number = chapter.get("chapter_number", 1)
        narration = chapter.get("narration", "")
        dialogues = chapter.get("dialogues", [])
        
        result = {
            "chapter_number": chapter_number,
            "narration_audio": None,
            "narration_audio_chunks": [],
            "dialogue_audios": [],
            "dialogue_audio_chunks": [],
            "combined_audio": None,
            "streaming_metadata": {
                "chunks_received": 0,
                "total_bytes": 0,
                "websocket_used": True
            }
        }
        
        # 1. 내레이션 스트리밍 음성 생성 (청킹 지원)
        if narration:
            if self.enable_chunking:
                # 청킹된 내레이션 생성
                narration_chunks = await self._generate_chunked_audio(
                    text=narration,
                    voice_id=self.narrator_voice_id,
                    voice_settings=self.character_voice_settings["narrator"],
                    filename_prefix=f"narration_ch{chapter_number}_{story_id[:8]}",
                    speaker_type="narrator",
                    use_websocket=True,
                    progress_callback=progress_callback
                )
                result["narration_audio_chunks"] = narration_chunks
                # 첫 번째 청크를 기본 narration_audio로 설정 (하위 호환성)
                if narration_chunks:
                    result["narration_audio"] = narration_chunks[0]["audio_path"]
                    # 스트리밍 메타데이터 합산
                    for chunk in narration_chunks:
                        if "streaming_info" in chunk:
                            result["streaming_metadata"]["chunks_received"] += chunk["streaming_info"].get("chunks_received", 0)
                            result["streaming_metadata"]["total_bytes"] += chunk["streaming_info"].get("total_bytes", 0)
            else:
                # 기존 방식 (청킹 비활성화)
                narration_audio, streaming_info = await self._generate_single_audio_websocket(
                    text=narration,
                    voice_id=self.narrator_voice_id,
                    voice_settings=self.character_voice_settings["narrator"],
                    filename_prefix=f"narration_ch{chapter_number}_{story_id[:8]}",
                    progress_callback=progress_callback
                )
                result["narration_audio"] = str(narration_audio)
                result["streaming_metadata"]["narration_chunks"] = streaming_info["chunks_received"]
        
        # 2. 대사별 스트리밍 음성 생성 (청킹 지원)
        for i, dialogue in enumerate(dialogues):
            speaker = dialogue.get("speaker", "unknown")
            text = dialogue.get("text", "")
            
            if text:
                # 캐릭터에 맞는 음성 ID와 설정 가져오기
                voice_id = self.get_voice_for_character(speaker)
                voice_settings = self.get_voice_settings_for_character(speaker)
                
                # speaker_type 결정
                speaker_type = self._get_speaker_type_for_character(speaker)
                
                try:
                    if self.enable_chunking:
                        # 청킹된 대사 생성
                        dialogue_chunks = await self._generate_chunked_audio(
                            text=text,
                            voice_id=voice_id,
                            voice_settings=voice_settings,
                            filename_prefix=f"dialogue_ch{chapter_number}_{i}_{story_id[:8]}",
                            speaker_type=speaker_type,
                            use_websocket=True,
                            progress_callback=progress_callback
                        )
                        
                        result["dialogue_audio_chunks"].append({
                            "speaker": speaker,
                            "voice_id": voice_id,
                            "chunks": dialogue_chunks
                        })
                        
                        # 첫 번째 청크를 기본 dialogue_audios에도 추가 (하위 호환성)
                        if dialogue_chunks:
                            result["dialogue_audios"].append({
                                "speaker": speaker,
                                "audio_path": dialogue_chunks[0]["audio_path"],
                                "text": text,
                                "voice_id": voice_id,
                                "streaming_info": dialogue_chunks[0].get("streaming_info")
                            })
                            
                            # 스트리밍 메타데이터 합산
                            for chunk in dialogue_chunks:
                                if "streaming_info" in chunk:
                                    result["streaming_metadata"]["chunks_received"] += chunk["streaming_info"].get("chunks_received", 0)
                                    result["streaming_metadata"]["total_bytes"] += chunk["streaming_info"].get("total_bytes", 0)
                    else:
                        # 기존 방식 (청킹 비활성화)
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
        """한국어 동화 최적화 텍스트 전처리 (ElevenLabs 감정 인식 활용)"""
        
        # 기본 텍스트 정리
        text = self._clean_text_for_speech(text)
        
        # 동화 특화 감정 태그 적용 (ElevenLabs v3 스타일)
        if speaker_type == "child":
            # 아이: 호기심 많고 순수한 톤
            if "?" in text:
                text = f"[curious] {text}"
            elif "!" in text:
                text = f"[excited] {text}"
            elif "무서" in text or "두려" in text:
                text = f"[whispers] {text}"
            else:
                text = f"{text} (밝은 목소리로)"
                
        elif speaker_type == "adult_male":
            # 어른 남성: 든든하고 지혜로운 톤
            if "?" in text:
                text = f"{text} (부드럽게 물으며)"
            elif "조심" in text or "위험" in text:
                text = f"[serious] {text}"
            else:
                text = f"{text} (차분하고 든든하게)"
                
        elif speaker_type == "adult_female":
            # 어른 여성: 따뜻하고 포근한 톤
            if "?" in text:
                text = f"{text} (다정하게 물으며)"
            elif "사랑" in text or "소중" in text:
                text = f"[warmly] {text}"
            else:
                text = f"{text} (따뜻하고 부드럽게)"
                
        elif speaker_type == "fantasy" or speaker_type == "magical":
            # 판타지/마법: 신비롭고 매력적인 톤
            if "마법" in text or "요술" in text:
                text = f"[mischievously] {text}"
            elif "!" in text:
                text = f"[excited] {text}"
            elif "사라지다" in text or "나타나다" in text:
                text = f"[whispers] {text} [dramatic pause]"
            else:
                text = f"{text} (신비한 목소리로)"
                
        elif speaker_type == "grandpa":
            # 할아버지: 지혜롭고 자상한 톤
            if "옛날" in text:
                text = f"{text} (옛 이야기를 들려주듯이)"
            else:
                text = f"{text} (자상하고 지혜롭게)"
            
        elif speaker_type == "narrator":
            # 내레이터: 동화책을 읽어주는 따뜻하고 표현력 있는 톤
            if "옛날" in text or "예전" in text or "아주 먼 곳" in text:
                text = f"[warmly] {text} (이야기를 시작하며)"
            elif "그래서" in text or "그러나" in text or "하지만" in text:
                text = f"{text} (이야기를 이어가며)"
            elif "끝" in text or "마지막" in text or "행복하게" in text:
                text = f"[satisfied] {text} (이야기를 마무리하며)"
            elif "!" in text and ("와" in text or "어머" in text):
                text = f"[excited] {text}"
            elif "?" in text:
                text = f"[curious] {text}"
            elif "..." in text:
                text = f"{text} [pause]"
            else:
                text = f"{text} (동화책을 읽어주듯이)"
        
        # 감정 강화를 위한 추가 처리
        # 대화문에 따옴표가 있으면 더 생동감 있게 처리
        if '"' in text or "'" in text:
            if speaker_type != "narrator":
                # 캐릭터 대사는 더 감정적으로
                if "기뻐" in text or "좋아" in text:
                    text = f"[happy] {text}"
                elif "슬퍼" in text or "울어" in text:
                    text = f"[sad] {text}"
                elif "화나" in text or "짜증" in text:
                    text = f"[annoyed] {text}"
        
        # ElevenLabs Turbo v2.5 텍스트 길이 제한 (40,000자)
        if len(text) > 39000:
            logger.warning(f"텍스트가 너무 깁니다 ({len(text)}자). 39000자로 자릅니다.")
            text = text[:38997] + "..."
        
        return text
    
    def _clean_text_for_speech(self, text: str) -> str:
        """한국어 동화 음성 생성을 위한 텍스트 정리"""
        
        # 기본 정리
        text = text.strip()
        
        # 마크다운 및 특수 문자 처리
        replacements = {
            "**": "",  # 마크다운 볼드 제거
            "*": "",   # 마크다운 이탤릭 제거
            "_": "",   # 언더스코어 제거
            "#": "",   # 해시태그 제거
            "`": "",   # 백틱 제거
            "---": ".",  # 구분선을 마침표로
            "…": ".",   # 말줄임표 정리
            "...": ".",  # 영문 말줄임표 정리
            "~": "",    # 물결표 제거
            "^": "",    # 캐럿 제거
            "[": "",    # 대괄호 제거
            "]": "",    
            "{": "",    # 중괄호 제거
            "}": "",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # 한국어 동화 특화 정리
        import re
        
        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        # 연속된 마침표 정리
        text = re.sub(r'\.{2,}', '.', text)
        
        # 한국어 문장 부호 정리
        text = re.sub(r'[，、]', ',', text)  # 쉼표 통일
        text = re.sub(r'[。．]', '.', text)  # 마침표 통일
        text = re.sub(r'[？]', '?', text)   # 물음표 통일
        text = re.sub(r'[！]', '!', text)   # 느낌표 통일
        
        # 연속된 문장 부호 정리
        text = re.sub(r'[,]{2,}', ',', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # 문장 끝 정리 (마침표 없이 끝나는 문장에 마침표 추가)
        if text and not text.endswith(('.', '!', '?', ',')):
            text += '.'
        
        return text.strip()
    
    async def _generate_single_audio_http(self, 
                                         text: str, 
                                         voice_id: str,
                                         voice_settings: Dict[str, Any],
                                         model_id: str,
                                         filename_prefix: str) -> Path:
        """HTTP API를 사용한 단일 오디오 생성 (동화 최적화)"""
        
        try:
            # API 요청 데이터 (한국어 동화 최적화)
            data = {
                "text": text,
                "model_id": model_id,
                "voice_settings": voice_settings,
                "output_format": "wav_44100",  # 고품질 44.1kHz WAV 포맷
                "optimize_streaming_latency": 1,  # 동화 생성: 품질 우선 (2 → 1)
                "apply_text_normalization": True,  # 한국어 텍스트 정규화 활성화
                "pronunciation_dictionary_locators": []  # 발음 사전 (필요시 추가)
            }
            
            # ElevenLabs API 호출
            url = f"{self.base_url}/text-to-speech/{voice_id}"
            
            # SSL 컨텍스트 설정
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            timeout = aiohttp.ClientTimeout(total=180)  # 동화 생성: 더 긴 타임아웃 (120 → 180)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=self.headers, json=data, ssl=ssl_context) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"ElevenLabs API 오류 ({response.status}): {error_text}")
                    
                    # 오디오 데이터 읽기
                    audio_data = await response.read()
            
            # 파일 저장
            audio_filename = f"{filename_prefix}.wav"
            
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
            model_id = "eleven_turbo_v2_5"  # 한국어 지원 + 고품질 + 저지연 (250-300ms)
            uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}"
            
            # 오디오 파일 경로 설정
            audio_filename = f"{filename_prefix}.wav"
            
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
                    },
                    "output_format": {
                        "container": "wav",
                        "encoding": "pcm_44100"
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
                model_id="eleven_turbo_v2_5",
                filename_prefix=filename_prefix
            )
            return fallback_path, {"websocket_fallback": True, "chunks_received": 0, "total_bytes": 0}

    def _chunk_text_by_sentences(self, text: str, max_chunk_length: int = None) -> List[str]:
        """텍스트를 문장 단위로 청킹하여 작은 오디오 파일들로 분할"""
        
        if not self.enable_chunking:
            return [text]
        
        max_length = max_chunk_length or self.max_chunk_length
        
        # 기본 정리
        text = self._clean_text_for_speech(text)
        
        if len(text) <= max_length:
            return [text]
        
        # 한국어 문장 분리 패턴
        sentence_patterns = [
            r'[.!?]\s+',  # 마침표, 느낌표, 물음표 + 공백
            r'[。！？]\s*',  # 한국어 문장부호
            r'\n\n+',  # 단락 구분
            r'\n',  # 줄바꿈
        ]
        
        chunks = []
        current_chunk = ""
        
        # 문장 단위로 분리
        sentences = []
        remaining_text = text
        
        for pattern in sentence_patterns:
            if not remaining_text:
                break
                
            parts = re.split(f'({pattern})', remaining_text)
            if len(parts) > 1:
                sentences = []
                i = 0
                while i < len(parts):
                    sentence = parts[i]
                    if i + 1 < len(parts) and re.match(pattern, parts[i + 1]):
                        sentence += parts[i + 1]
                        i += 2
                    else:
                        i += 1
                    
                    if sentence.strip():
                        sentences.append(sentence.strip())
                break
        
        # 문장이 분리되지 않은 경우 단어 단위로 분리
        if not sentences:
            words = text.split()
            current_sentence = ""
            for word in words:
                if len(current_sentence + " " + word) <= max_length:
                    current_sentence += " " + word if current_sentence else word
                else:
                    if current_sentence:
                        sentences.append(current_sentence)
                    current_sentence = word
            if current_sentence:
                sentences.append(current_sentence)
        
        # 청크로 그룹화
        for sentence in sentences:
            # 한 문장이 max_length를 초과하는 경우 강제 분할
            if len(sentence) > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # 긴 문장을 단어 단위로 분할
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk + " " + word) <= max_length:
                        temp_chunk += " " + word if temp_chunk else word
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
                continue
            
            # 현재 청크에 문장을 추가할 수 있는지 확인
            if len(current_chunk + " " + sentence) <= max_length:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                # 현재 청크 저장하고 새 청크 시작
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 빈 청크 제거
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        logger.info(f"텍스트 청킹 완료: {len(text)}자 → {len(chunks)}개 청크 (최대 {max_length}자)")
        
        return chunks
    
    def _create_audio_manifest(self, audio_files: List[Dict[str, Any]], story_id: str) -> Dict[str, Any]:
        """오디오 파일들의 재생 순서와 메타데이터를 담은 매니페스트 생성"""
        
        manifest = {
            "story_id": story_id,
            "version": "1.0",
            "created_at": asyncio.get_event_loop().time(),
            "chunking_enabled": self.enable_chunking,
            "audio_sequence": [],
            "total_files": 0,
            "total_duration_estimate": 0,
            "playback_instructions": {
                "sequential": True,
                "auto_next": True,
                "fade_between": True,
                "fade_duration_ms": 200
            }
        }
        
        sequence_number = 1
        
        for chapter_data in audio_files:
            chapter_number = chapter_data.get("chapter_number", 1)
            
            # 내레이션 청크들 추가
            if "narration_audio_chunks" in chapter_data:
                for i, chunk_info in enumerate(chapter_data["narration_audio_chunks"]):
                    manifest["audio_sequence"].append({
                        "sequence": sequence_number,
                        "type": "narration",
                        "chapter": chapter_number,
                        "chunk_index": i,
                        "file_path": chunk_info["audio_path"],
                        "text": chunk_info["text"],
                        "voice_id": chunk_info.get("voice_id", self.narrator_voice_id),
                        "speaker": "narrator",
                        "estimated_duration": len(chunk_info["text"]) * 0.05  # 대략 50ms per character
                    })
                    sequence_number += 1
            elif "narration_audio" in chapter_data and chapter_data["narration_audio"]:
                # 청킹되지 않은 내레이션
                manifest["audio_sequence"].append({
                    "sequence": sequence_number,
                    "type": "narration",
                    "chapter": chapter_number,
                    "chunk_index": 0,
                    "file_path": chapter_data["narration_audio"],
                    "voice_id": self.narrator_voice_id,
                    "speaker": "narrator",
                    "estimated_duration": 0
                })
                sequence_number += 1
            
            # 대사 청크들 추가
            if "dialogue_audio_chunks" in chapter_data:
                for dialogue_data in chapter_data["dialogue_audio_chunks"]:
                    for i, chunk_info in enumerate(dialogue_data["chunks"]):
                        manifest["audio_sequence"].append({
                            "sequence": sequence_number,
                            "type": "dialogue",
                            "chapter": chapter_number,
                            "chunk_index": i,
                            "file_path": chunk_info["audio_path"],
                            "text": chunk_info["text"],
                            "voice_id": chunk_info.get("voice_id"),
                            "speaker": dialogue_data["speaker"],
                            "estimated_duration": len(chunk_info["text"]) * 0.05
                        })
                        sequence_number += 1
            elif "dialogue_audios" in chapter_data:
                # 청킹되지 않은 대사들
                for dialogue in chapter_data["dialogue_audios"]:
                    manifest["audio_sequence"].append({
                        "sequence": sequence_number,
                        "type": "dialogue",
                        "chapter": chapter_number,
                        "chunk_index": 0,
                        "file_path": dialogue["audio_path"],
                        "text": dialogue.get("text", ""),
                        "voice_id": dialogue.get("voice_id"),
                        "speaker": dialogue["speaker"],
                        "estimated_duration": 0
                    })
                    sequence_number += 1
        
        manifest["total_files"] = len(manifest["audio_sequence"])
        manifest["total_duration_estimate"] = sum(item["estimated_duration"] for item in manifest["audio_sequence"])
        
        return manifest
    
    async def _generate_chunked_audio(self, 
                                    text: str, 
                                    voice_id: str,
                                    voice_settings: Dict[str, Any],
                                    filename_prefix: str,
                                    speaker_type: str = "narrator",
                                    use_websocket: bool = True,
                                    progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """텍스트를 청킹하여 순차적으로 작은 오디오 파일들 생성"""
        
        # 텍스트 청킹
        text_chunks = self._chunk_text_by_sentences(text, self.max_chunk_length)
        
        if len(text_chunks) == 1 and not self.enable_chunking:
            # 청킹이 비활성화되었거나 청킹이 필요없는 경우 기존 방식 사용
            if use_websocket:
                audio_path, streaming_info = await self._generate_single_audio_websocket(
                    text=text_chunks[0],
                    voice_id=voice_id,
                    voice_settings=voice_settings,
                    filename_prefix=filename_prefix,
                    progress_callback=progress_callback
                )
                return [{
                    "chunk_index": 0,
                    "text": text_chunks[0],
                    "audio_path": str(audio_path),
                    "voice_id": voice_id,
                    "streaming_info": streaming_info
                }]
            else:
                audio_path = await self._generate_single_audio_http(
                    text=text_chunks[0],
                    voice_id=voice_id,
                    voice_settings=voice_settings,
                    model_id=self.model_id,
                    filename_prefix=filename_prefix
                )
                return [{
                    "chunk_index": 0,
                    "text": text_chunks[0],
                    "audio_path": str(audio_path),
                    "voice_id": voice_id
                }]
        
        # 청킹된 오디오 생성
        chunk_results = []
        
        for i, chunk_text in enumerate(text_chunks):
            chunk_filename = f"{filename_prefix}_chunk{i+1:03d}"
            
            # 텍스트 전처리 (감정 태그 추가)
            processed_text = self._prepare_text_for_speech(chunk_text, speaker_type)
            
            if progress_callback:
                await progress_callback({
                    "step": "chunked_audio_generation",
                    "status": "processing_chunk",
                    "current_chunk": i + 1,
                    "total_chunks": len(text_chunks),
                    "chunk_text": chunk_text[:50] + "..." if len(chunk_text) > 50 else chunk_text,
                    "speaker_type": speaker_type
                })
            
            try:
                if use_websocket:
                    audio_path, streaming_info = await self._generate_single_audio_websocket(
                        text=processed_text,
                        voice_id=voice_id,
                        voice_settings=voice_settings,
                        filename_prefix=chunk_filename,
                        progress_callback=progress_callback
                    )
                    chunk_results.append({
                        "chunk_index": i,
                        "text": chunk_text,
                        "processed_text": processed_text,
                        "audio_path": str(audio_path),
                        "voice_id": voice_id,
                        "streaming_info": streaming_info
                    })
                else:
                    audio_path = await self._generate_single_audio_http(
                        text=processed_text,
                        voice_id=voice_id,
                        voice_settings=voice_settings,
                        model_id=self.model_id,
                        filename_prefix=chunk_filename
                    )
                    chunk_results.append({
                        "chunk_index": i,
                        "text": chunk_text,
                        "processed_text": processed_text,
                        "audio_path": str(audio_path),
                        "voice_id": voice_id
                    })
                
                logger.info(f"청크 오디오 생성 완료: {chunk_filename} ({i+1}/{len(text_chunks)})")
                
            except Exception as e:
                logger.error(f"청크 오디오 생성 실패 ({chunk_filename}): {e}")
                # 실패한 청크는 건너뛰고 계속 진행
                continue
        
        logger.info(f"청킹된 오디오 생성 완료: {len(chunk_results)}/{len(text_chunks)} 청크 성공")
        
        return chunk_results
