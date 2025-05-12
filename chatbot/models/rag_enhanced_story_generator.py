#!/usr/bin/env python3
"""
RAG 기반 향상된 스토리 생성기

이 모듈은 LangChain과 ChromaDB 기반 RAG 시스템을 활용하여
꼬기(ChatBot B)의 동화 생성 능력을 향상시킵니다.

주요 기능:
1. Few-shot 학습을 통한 연령별 맞춤형 동화 생성
2. 유사 동화 검색 및 참조를 통한 창의성 향상
3. 컨텍스트 강화로 일관되고 교육적인 동화 생성
"""

import os
import json
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from .rag_system import RAGSystem

# 환경 변수 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


class RAGEnhancedStoryGenerator:
    """
    RAG와 Few-shot 학습을 통합한 향상된 스토리 생성 클래스
    
    Attributes:
        rag_system: RAG 시스템 인스턴스
        prompts: 프롬프트 템플릿
        output_dir: 출력 파일 저장 경로
    """
    
    def __init__(self, output_dir: str = "output", rag_system: Optional[RAGSystem] = None):
        """
        생성기 초기화
        
        Args:
            output_dir: 출력 디렉터리
            rag_system: 사용할 RAG 시스템 인스턴스 (없으면 새로 생성)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # RAG 시스템 초기화
        self.rag_system = rag_system if rag_system else RAGSystem()
        
        # 프롬프트 템플릿 로드
        self.load_prompts()
        
        # 현재 스토리 정보
        self.story_outline = None
        self.target_age = None
        self.detailed_story = None
    
    def load_prompts(self):
        """프롬프트 템플릿 로드"""
        try:
            prompts_path = os.path.join(current_dir, "..", "data", "prompts", "chatbot_b_prompts.json")
            with open(prompts_path, 'r', encoding='utf-8') as f:
                self.prompts = json.load(f)
        except Exception as e:
            print(f"프롬프트 로드 중 오류 발생: {str(e)}")
            # 기본 프롬프트 설정 (최소한의 템플릿)
            self.prompts = {
                "system": {
                    "role": [
                        "너는 꼬기라는 이름을 가진 챗봇이야. 너는 부기 (chatbot_a) 와 대화를 하면서 동화를 만들어줄거야.",
                        "너는 부기 (chatbot_a) 가 만들어 준 대략적인 동화 스토리를 통해 상세한 스토리를 만들어줄거야."
                    ],
                    "instructions": [
                        "아이들의 연령대와 관심사를 반영하여 이야기를 더욱 흥미롭게 만들어야 해."
                    ]
                },
                "detailed_story_generation": [
                    {
                        "age_group": "4-5",
                        "prompt": "다음 동화 줄거리를 바탕으로 4-5세 아이들을 위한 상세한 이야기를 만들어주세요."
                    },
                    {
                        "age_group": "6-7",
                        "prompt": "다음 동화 줄거리를 바탕으로 6-7세 아이들을 위한 상세한 이야기를 만들어주세요."
                    },
                    {
                        "age_group": "8-9",
                        "prompt": "다음 동화 줄거리를 바탕으로 8-9세 아이들을 위한 상세한 이야기를 만들어주세요."
                    }
                ]
            }
    
    def set_story_outline(self, story_outline: Dict[str, Any]):
        """
        스토리 아웃라인 설정
        
        Args:
            story_outline: 아웃라인 정보 (주제, 등장인물, 배경, 줄거리 등)
        """
        self.story_outline = story_outline
    
    def set_target_age(self, age: int):
        """
        대상 연령 설정
        
        Args:
            age: 아이 연령 (4-9세)
        """
        self.target_age = age
        
        # 스토리 아웃라인에도 추가
        if self.story_outline:
            self.story_outline["target_age"] = age
    
    def _get_age_group(self) -> str:
        """연령대 문자열 반환"""
        if self.target_age <= 5:
            return "4-5"
        elif self.target_age <= 7:
            return "6-7"
        else:
            return "8-9"
    
    def _get_age_specific_prompt(self, prompt_list: List[Dict[str, str]]) -> str:
        """연령대에 맞는 프롬프트 반환"""
        age_group = self._get_age_group()
        for item in prompt_list:
            if item.get("age_group") == age_group:
                return item.get("prompt", "")
        return prompt_list[0].get("prompt", "") if prompt_list else ""
    
    def _enrich_theme_with_rag(self) -> str:
        """RAG 시스템을 활용하여 주제 강화"""
        if not self.story_outline:
            return ""
        
        # 주제 추출
        theme = self.story_outline.get("theme", "")
        
        # RAG 시스템으로 주제 강화
        return self.rag_system.enrich_story_theme(
            theme=theme,
            age_group=self.target_age
        )
    
    def _get_similar_stories(self, n_results: int = 2) -> List[Dict[str, Any]]:
        """유사 동화 검색"""
        if not self.story_outline:
            return []
        
        # 주제 추출
        theme = self.story_outline.get("theme", "")
        
        # 유사 동화 검색
        return self.rag_system.get_similar_stories(
            theme=theme,
            age_group=self.target_age,
            n_results=n_results
        )
    
    def _build_few_shot_prompt(self) -> str:
        """Few-shot 프롬프트 생성"""
        if not self.story_outline:
            return ""
        
        # 주제 추출
        theme = self.story_outline.get("theme", "")
        
        # Few-shot 프롬프트 생성
        return self.rag_system.get_few_shot_prompt(
            age_group=self.target_age,
            theme=theme,
            n_examples=2
        )
    
    def generate_detailed_story(self) -> Dict[str, Any]:
        """
        RAG와 Few-shot 학습을 활용한 상세 스토리 생성
        
        Returns:
            Dict: 상세 스토리 정보
        """
        if not self.story_outline:
            raise ValueError("스토리 아웃라인이 설정되지 않았습니다.")
        
        if not self.target_age:
            raise ValueError("대상 연령이 설정되지 않았습니다.")
        
        # 1. 연령대에 맞는 프롬프트 가져오기
        base_prompt = self._get_age_specific_prompt(self.prompts.get("detailed_story_generation", []))
        
        # 2. RAG로 스토리 주제 강화
        enriched_theme = self._enrich_theme_with_rag()
        
        # 3. Few-shot 예시 생성
        few_shot_examples = self._build_few_shot_prompt()
        
        # 4. 유사 스토리 검색 및 참조 정보 추출
        similar_stories = self._get_similar_stories(n_results=2)
        reference_info = ""
        if similar_stories:
            reference_info = "참고할 만한 유사 동화 정보:\n"
            for i, story in enumerate(similar_stories):
                reference_info += f"{i+1}. 제목: {story.get('title')}\n"
                reference_info += f"   요약: {story.get('summary')}\n\n"
        
        # 5. 시스템 프롬프트 구성
        system_content = "\n".join(self.prompts.get("system", {}).get("role", []))
        system_content += "\n" + "\n".join(self.prompts.get("system", {}).get("instructions", []))
        
        # 6. 최종 프롬프트 구성
        user_content = f"""
        {base_prompt}
        
        {few_shot_examples}
        
        스토리 정보:
        주제: {enriched_theme}
        등장인물: {', '.join(self.story_outline.get('characters', []))}
        배경: {self.story_outline.get('setting', '')}
        줄거리: {self.story_outline.get('plot_summary', '')}
        교육적 가치: {self.story_outline.get('educational_value', '')}
        
        {reference_info}
        
        아이 연령: {self.target_age}세
        
        다음 형식으로 상세 스토리를 생성해주세요:
        1. 제목
        2. 장면별 내용 (각 장면은 제목, 설명, 내레이션, 대사를 포함)
        3. 전체 이야기는 시작, 중간, 끝 구조를 가져야 함
        4. {self.target_age}세 아이에게 적합한 언어와 내용
        """
        
        # 7. GPT-4o로 상세 스토리 생성
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                max_tokens=3000
            )
            
            # 응답 추출
            story_text = response.choices[0].message.content
            
            # 응답을 구조화된 형식으로 변환
            story_data = self._parse_story_response(story_text)
            
            # 상세 스토리 저장
            self.detailed_story = story_data
            
            return story_data
            
        except Exception as e:
            raise Exception(f"상세 스토리 생성 중 오류 발생: {str(e)}")
    
    def _parse_story_response(self, story_text: str) -> Dict[str, Any]:
        """
        생성된 스토리 텍스트 파싱
        
        Args:
            story_text: 생성된 스토리 텍스트
            
        Returns:
            Dict: 구조화된 스토리 데이터
        """
        # 기본 구조 설정
        story_data = {
            "title": "",
            "scenes": []
        }
        
        try:
            # 제목 추출 (첫 번째 줄 또는 '제목:' 포함된 줄)
            lines = story_text.strip().split('\n')
            for line in lines:
                if line.strip() and not line.startswith('#') and not line.startswith('장면'):
                    story_data["title"] = line.replace('제목:', '').strip()
                    break
            
            # 장면 분리
            # '장면' 또는 '# '으로 시작하는 줄을 기준으로 분리
            scenes_text = []
            current_scene = []
            
            in_scenes = False
            for line in lines:
                if line.strip().startswith(('장면', '# ', '## ')):
                    if current_scene:
                        scenes_text.append('\n'.join(current_scene))
                        current_scene = []
                    in_scenes = True
                    current_scene.append(line)
                elif in_scenes:
                    current_scene.append(line)
            
            if current_scene:
                scenes_text.append('\n'.join(current_scene))
            
            # 장면이 분리되지 않았다면 전체 내용을 단일 장면으로 처리
            if not scenes_text:
                scenes_text = [story_text]
            
            # 각 장면 파싱
            for i, scene_text in enumerate(scenes_text):
                scene = {
                    "title": f"장면 {i+1}",
                    "description": scene_text.strip(),
                    "narration": "",
                    "dialogues": []
                }
                
                # 장면 제목 추출
                lines = scene_text.strip().split('\n')
                for line in lines:
                    if line.strip().startswith(('장면', '# ', '## ')):
                        scene["title"] = line.replace('장면', '').replace('#', '').strip()
                        break
                
                # 내레이션과 대사 분리 시도
                narration_parts = []
                dialogues = []
                
                in_dialogue = False
                current_character = ""
                current_dialogue = ""
                
                for line in lines[1:]:  # 첫 줄(제목)은 건너뜀
                    # 대화 형식 (캐릭터: 대사)
                    if ':' in line and not line.strip().startswith('내레이션:'):
                        parts = line.split(':', 1)
                        if len(parts) == 2 and parts[0].strip() and not parts[0].strip().lower() in ['http', 'https']:
                            if in_dialogue and current_character and current_dialogue:
                                dialogues.append({
                                    "character": current_character,
                                    "text": current_dialogue.strip()
                                })
                            
                            current_character = parts[0].strip()
                            current_dialogue = parts[1].strip()
                            in_dialogue = True
                        else:
                            if in_dialogue:
                                current_dialogue += " " + line.strip()
                            else:
                                narration_parts.append(line)
                    else:
                        if in_dialogue:
                            current_dialogue += " " + line.strip()
                        else:
                            narration_parts.append(line)
                
                # 마지막 대화 처리
                if in_dialogue and current_character and current_dialogue:
                    dialogues.append({
                        "character": current_character,
                        "text": current_dialogue.strip()
                    })
                
                # 내레이션 설정
                scene["narration"] = '\n'.join(narration_parts).strip()
                scene["dialogues"] = dialogues
                
                # 장면 추가
                story_data["scenes"].append(scene)
            
            return story_data
            
        except Exception as e:
            print(f"스토리 텍스트 파싱 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본 형식으로 반환
            return {
                "title": "스토리 제목",
                "scenes": [
                    {
                        "title": "장면 1",
                        "description": story_text,
                        "narration": "",
                        "dialogues": []
                    }
                ]
            }
    
    def save_story_data(self, file_path: str):
        """
        생성된 스토리 데이터 저장
        
        Args:
            file_path: 저장할 파일 경로
        """
        if not self.detailed_story:
            raise ValueError("저장할 스토리 데이터가 없습니다.")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.detailed_story, f, ensure_ascii=False, indent=2)
    
    def load_story_data(self, file_path: str):
        """
        저장된 스토리 데이터 로드
        
        Args:
            file_path: 로드할 파일 경로
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            self.detailed_story = json.load(f)
    
    def add_sample_story_to_rag(self, story_data: Dict[str, Any]) -> str:
        """
        샘플 스토리를 RAG 시스템에 추가
        
        Args:
            story_data: 스토리 데이터
            
        Returns:
            str: 생성된 스토리 ID
        """
        # 필수 필드 검증
        required_fields = ["title", "tags", "summary", "content"]
        for field in required_fields:
            if field not in story_data:
                raise ValueError(f"필수 필드가 누락되었습니다: {field}")
        
        # RAG 시스템에 스토리 추가
        return self.rag_system.add_story(
            title=story_data["title"],
            tags=story_data["tags"],
            summary=story_data["summary"],
            content=story_data["content"]
        ) 