from typing import List, Dict, Any
import os
import json
import torch
from transformers import AutoProcessor, AutoModel
import soundfile as sf
import numpy as np
from dotenv import load_dotenv
import openai

# 환경 변수 설정
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

class VoiceGenerator:
    """
    Swivid TTS와 OpenAI TTS를 활용한 음성 생성 클래스
    
    Attributes:
        model_path (str): Swivid TTS 모델 경로
        fine_tuned_model (str): Fine-tuning된 모델 ID
        processor (AutoProcessor): Swivid TTS 프로세서
        model (AutoModel): Swivid TTS 모델
        device (str): 연산 디바이스 (CPU/GPU)
    """
    
    def __init__(self):
        """VoiceGenerator 초기화"""
        self.model_path = "path/to/swivid/model"
        self.fine_tuned_model = None
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def prepare_training_data(self, audio_samples: List[str], text_samples: List[str]) -> Dict[str, Any]:
        """
        TTS Fine-tuning을 위한 데이터 준비
        
        Args:
            audio_samples (List[str]): 오디오 샘플 파일 경로 목록
            text_samples (List[str]): 텍스트 샘플 목록
            
        Returns:
            Dict[str, Any]: 전처리된 학습 데이터
        """
        try:
            processed_data = []
            for audio_path, text in zip(audio_samples, text_samples):
                # 오디오 데이터 로드 및 전처리
                audio_data, sample_rate = sf.read(audio_path)
                
                # 텍스트 정규화
                normalized_text = self._normalize_text(text)
                
                processed_data.append({
                    "audio": audio_data,
                    "text": normalized_text,
                    "sample_rate": sample_rate
                })
            
            return {
                "processed_data": processed_data,
                "metadata": {
                    "total_samples": len(processed_data),
                    "sample_rate": sample_rate
                }
            }
        except Exception as e:
            print(f"Error preparing training data: {str(e)}")
            return {}
    
    def _normalize_text(self, text: str) -> str:
        """
        텍스트 정규화
        
        Args:
            text (str): 정규화할 텍스트
            
        Returns:
            str: 정규화된 텍스트
        """
        # TODO: 실제 텍스트 정규화 로직 구현
        return text
    
    def fine_tune_voice(self, audio_samples: List[str], text_samples: List[str]):
        """
        사용자 음성으로 모델을 파인튜닝하는 함수
        
        Args:
            audio_samples (List[str]): 오디오 샘플 파일 경로 목록
            text_samples (List[str]): 텍스트 샘플 목록
        """
        try:
            # 데이터 준비
            training_data = self.prepare_training_data(audio_samples, text_samples)
            
            # 모델 및 프로세서 로드
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
            
            # Fine-tuning 설정
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
            
            # Fine-tuning 실행
            self.model.train()
            for epoch in range(3):  # 3 에포크
                for item in training_data["processed_data"]:
                    # 입력 데이터 준비
                    inputs = this.processor(
                        text=item["text"],
                        audio=item["audio"],
                        return_tensors="pt"
                    ).to(this.device)
                    
                    # Forward pass
                    outputs = this.model(**inputs)
                    loss = outputs.loss
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            
            # 모델 저장
            this.model.save_pretrained("models/fine_tuned_voice")
            this.processor.save_pretrained("models/fine_tuned_voice")
            
        except Exception as e:
            print(f"Error during voice fine-tuning: {str(e)}")
            raise
    
    def generate_voice(self, text: str, use_openai: bool = False) -> str:
        """
        텍스트를 음성으로 변환하는 함수
        
        Args:
            text (str): 변환할 텍스트
            use_openai (bool): OpenAI TTS 사용 여부
            
        Returns:
            str: 생성된 음성 파일 경로
        """
        try:
            if use_openai:
                # OpenAI TTS 사용
                response = openai.Audio.create(
                    model="tts-1",
                    voice="alloy",
                    input=text
                )
                
                # 오디오 파일 저장
                output_path = "output/openai_audio.mp3"
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                return output_path
            else:
                # Fine-tuning된 모델이 있으면 사용
                if this.fine_tuned_model:
                    model_path = "models/fine_tuned_voice"
                else:
                    model_path = this.model_path
                
                # 모델 및 프로세서 로드
                processor = AutoProcessor.from_pretrained(model_path)
                model = AutoModel.from_pretrained(model_path).to(this.device)
                
                # 텍스트 정규화
                normalized_text = this._normalize_text(text)
                
                # 음성 생성
                inputs = processor(
                    text=normalized_text,
                    return_tensors="pt"
                ).to(this.device)
                
                with torch.no_grad():
                    outputs = model.generate(**inputs)
                
                # 오디오 저장
                output_path = "output/audio.wav"
                sf.write(output_path, outputs.cpu().numpy(), samplerate=22050)
                
                return output_path
                
        except Exception as e:
            print(f"Error generating voice: {str(e)}")
            return "" 