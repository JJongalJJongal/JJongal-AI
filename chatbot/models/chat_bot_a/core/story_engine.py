"""
ChatBot A 통합 이야기 엔진

이야기 수집, 분석, 구조화를 담당하는 통합 엔진
기존 StoryCollector, StoryAnalyzer, StoryCollectionEngine의 기능을 하나로 통합
"""
from typing import Dict, List, Any, Optional, Set, Tuple
import random
import json
import os
import re
from pathlib import Path

from shared.utils.logging_utils import get_module_logger
from shared.utils.vector_db_utils import get_db_type_path
from chatbot.data.vector_db.core import VectorDB
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from .rag_engine import RAGSystem

logger = get_module_logger(__name__)

class StoryEngine:
    """
    이야기 수집, 분석 및 구조화를 담당하는 통합 엔진
    
    기존의 분산된 이야기 관련 기능들을 하나의 엔진으로 통합하여
    더 효율적이고 일관된 이야기 처리를 제공
    
    주요 기능:
    - 이야기 요소 수집 및 단계 관리
    - 사용자 응답 분석 및 키워드 추출
    - 대화 요약 및 이야기 주제 제안
    - 이야기 구조화 및 개요 생성
    """
    
    def __init__(self, user_data: Dict, story_data: Optional[Dict] = None, openai_client=None, **kwargs):
        self.user_data = user_data
        self.story_data = story_data if story_data else self._initialize_story_data()
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
        
        # RAG 시스템 초기화
        self.rag_system = self._initialize_rag_system()
        self.openai_client = openai_client
        
        # === 이야기 수집 상태 ===
        self.story_stage = "character"  # 현재 수집 단계
        self.story_elements = {
            "character": {"count": 0, "topics": set()},
            "setting": {"count": 0, "topics": set()},
            "problem": {"count": 0, "topics": set()},
            "resolution": {"count": 0, "topics": set()}
        }
        self.last_stage_transition = 0  # 마지막 단계 전환 턴
        
        # === 분석 및 생성 ===
        self.openai_client = None
        self.conversation_manager = None
        self.story_outline = None  # 생성된 이야기 개요
        
        # === 수집 통계 ===
        self.total_interactions = 0
        self.quality_scores = []
        
        logger.info("통합 이야기 엔진 초기화 완료")
    
    def _initialize_rag_system(self) -> Optional[RAGSystem]:
        """RAG 시스템을 초기화하고 인스턴스를 반환합니다."""
        try:
            # # 임시로 RAG 시스템 비활성화 (ChromaDB 충돌 방지)
            # logger.warning("RAG 시스템이 임시로 비활성화됨 (ChromaDB 충돌 방지)")
            # return None
            
            # 이 파일의 위치를 기준으로 vector_db 폴더의 기본 경로를 계산합니다.
            # story_engine.py -> core -> chat_bot_a -> models -> chatbot -> data/vector_db
            base_dir = Path(__file__).resolve().parent.parent.parent.parent / 'data' / 'vector_db'
            
            # base_directory 인자를 추가하여 DB 경로를 가져옵니다.
            db_path_str = get_db_type_path(db_type="summary", base_directory=str(base_dir))
            db_path = Path(db_path_str) 

            if not db_path.exists():
                logger.warning(f"RAG를 위한 VectorDB 경로를 찾을 수 없습니다: {db_path}")
                return None
            
            vector_db = VectorDB(persist_directory=str(db_path))
            vector_db.get_collection("fairy_tales")
            
            prompts = {
                "rag_templates": {
                    "story_enrichment": "다음 아이디어를 바탕으로 {age_group}세 아이에게 맞는 동화 아이디어를 더 풍부하게 만들어주세요.\n\n기본 아이디어: {query_text}\n\n참고 자료:\n{context}",
                    "rag_story_generation": "아래 정보를 바탕으로 {age_group}세 어린이를 위한 짧은 동화의 시작 부분을 만들어주세요.\n\n[컨텍스트]\n{context}\n\n[요청사항]\n{user_request}"
                }
            }
            return RAGSystem(vector_db, prompts)
        except Exception as e:
            logger.error(f"RAG 시스템 초기화 실패: {e}", exc_info=True)
            return None

    def _initialize_story_data(self) -> Dict:
        story_data = {
            "title": "",
            "summary": "",
            "characters": [],
            "setting": {},
            "content": "",
            "chapters": []
        }
        logger.info(f"스토리 데이터 초기화 완료: {story_data}")
        return story_data

    async def suggest_story_idea(self, conversation_history: List[Dict]) -> Dict:
        """
        대화 기록을 바탕으로 RAG를 활용하여 동화 아이디어를 제안합니다.
        """
        if not self.rag_system:
            logger.warning("RAG 시스템이 없어 기본 아이디어를 반환합니다.")
            return {"title": "모험과 우정", "summary": "동물 친구들의 신나는 모험 이야기"}

        try:
            age_group, interests = self._extract_user_info_from_conversation(conversation_history)
            base_idea = f"{interests[0]}에 대한 이야기" if interests else "친구들과의 신나는 모험"
            
            enhanced_idea = await self.rag_system.enhance_story_context(base_idea, age_group)

            prompt_template = ChatPromptTemplate.from_template(
                "다음 아이디어를 바탕으로 {age_group}세 아이가 좋아할 만한 동화의 제목과 줄거리를 만들어줘.\\n\\n아이디어: {enhanced_idea}\\n\\n출력 형식:\\n제목: [여기에 제목]\\n줄거리: [여기에 한두 문단 요약]"
            )
            chain = prompt_template | self.llm
            response = await chain.ainvoke({
                "enhanced_idea": enhanced_idea,
                "age_group": age_group
            })
            
            parsed_idea = self._parse_llm_response_for_story_idea(response.content)
            return parsed_idea
        except Exception as e:
            logger.error(f"동화 아이디어 제안 중 오류 발생: {e}", exc_info=True)
            return {"title": "오류", "summary": "아이디어를 생성하는 데 문제가 발생했습니다."}
            
    def _parse_llm_response_for_story_idea(self, response_text: str) -> Dict:
        """LLM의 응답에서 제목과 줄거리를 추출하는 함수"""
        try:
            title_match = re.search(r"제목:\\s*(.*)", response_text)
            summary_match = re.search(r"줄거리:\\s*(.*)", response_text, re.DOTALL)

            title = title_match.group(1).strip() if title_match else "알 수 없는 제목"
            summary = summary_match.group(1).strip() if summary_match else "줄거리를 생성하지 못했습니다."
            
            return {"title": title, "summary": summary}
        except Exception as e:
            logger.error(f"LLM 응답 파싱 실패: {e}\n원본 응답: {response_text}")
            return {"title": "파싱 오류", "summary": response_text}

    def _extract_user_info_from_conversation(self, conversation_history: List[Dict]) -> Tuple[int, List[str]]:
        """대화 기록에서 사용자 정보(나이, 관심사) 추출"""
        age_group = self.user_data.get("age_group", 5) 
        interests = self.user_data.get("interests", [])

        for message in conversation_history:
            if message.get("role") == "user":
                text = message.get("content", "").lower()
                if "공룡" in text and "공룡" not in interests:
                    interests.append("공룡")
                if "우주" in text and "우주" not in interests:
                    interests.append("우주")
        
        return age_group, interests

    def update_story_element(self, element_type: str, new_value: Any) -> Dict:
        if element_type in self.story_data:
            self.story_data[element_type] = new_value
            logger.info(f"스토리 요소 업데이트: {element_type} = {new_value}")
        else:
            logger.warning(f"알 수 없는 스토리 요소 타입: {element_type}")
        return self.story_data

    async def generate_full_story(self, initial_idea: Dict) -> Dict:
        """초기 아이디어를 바탕으로 전체 동화 줄거리를 생성합니다."""
        logger.info(f"전체 동화 생성 시작: {initial_idea}")
        self.story_data.update(initial_idea)
        
        prompt_template = ChatPromptTemplate.from_template(
            """
            다음 정보를 바탕으로 {age_group}세 아이를 위한 완전한 동화 줄거리를 생성해 주세요.

            제목: {title}
            기본 아이디어: {summary}

            생성할 내용:
            - 주요 등장인물 (2-3명)
            - 이야기의 배경 설정
            - 기-승-전-결 구조의 전체 줄거리

            출력 형식:
            등장인물: [이름1], [이름2], ...
            배경: [배경 설명]
            줄거리: [전체 줄거리]
            """
        )
        
        chain = prompt_template | self.llm
        
        try:
            response = await chain.ainvoke({
                "age_group": self.user_data.get("age_group", 5),
                "title": self.story_data.get("title"),
                "summary": self.story_data.get("summary")
            })
            parsed_story = self._parse_full_story_response(response.content)
            self.story_data.update(parsed_story)
            logger.info("전체 동화 생성 완료.")
            return self.story_data
        except Exception as e:
            logger.error(f"전체 동화 생성 중 오류 발생: {e}", exc_info=True)
            return {"error": str(e)}

    def _parse_full_story_response(self, response_text: str) -> Dict:
        """LLM의 전체 줄거리 응답을 파싱하여 구조화된 데이터로 변환합니다."""
        parsed_data = {}
        try:
            characters_match = re.search(r"등장인물:\\s*(.*)", response_text, re.DOTALL)
            if characters_match:
                parsed_data["characters"] = [{"name": c.strip()} for c in characters_match.group(1).split(',')]

            setting_match = re.search(r"배경:\\s*(.*)", response_text, re.DOTALL)
            if setting_match:
                parsed_data["setting"] = {"description": setting_match.group(1).strip()}

            parsed_data["content"] = response_text 
            logger.info("전체 스토리 응답 파싱 완료.")
        except Exception as e:
            logger.error(f"전체 스토리 응답 파싱 실패: {e}", exc_info=True)
            parsed_data["error"] = "Failed to parse the story structure."
            parsed_data["content"] = response_text
            
        return parsed_data
        
    def get_story_data(self) -> Dict:
        return self.story_data

    def reset_story(self):
        self.story_data = self._initialize_story_data()
        logger.info("스토리 데이터가 초기화되었습니다.")

    # ==========================================
    # 이야기 수집 관련 메서드
    # ==========================================
    
    def get_current_stage(self) -> str:
        """현재 이야기 수집 단계 반환"""
        return self.story_stage
    
    def analyze_user_response(self, user_input: str, openai_client=None) -> Dict[str, Any]:
        """
        사용자 응답을 분석하여 이야기 요소를 추출
        
        Args:
            user_input: 사용자 입력 텍스트
            openai_client: OpenAI API 클라이언트 (선택적)
            
        Returns:
            Dict: 분석 결과 (키워드, 품질 점수 등)
        """
        client = openai_client or self.openai_client
        self.total_interactions += 1
        
        analysis_result = {
            "keywords": [],
            "quality_score": 0.5,
            "stage": self.story_stage,
            "suggestions": []
        }
        
        if not client:
            # 클라이언트가 없는 경우 기본 처리
            self.story_elements[self.story_stage]["count"] += 1
            return analysis_result
        
        try:
            # GPT를 통한 고급 분석
            system_message = f"""
            당신은 {self.story_stage} 단계에서 아이의 응답을 분석하는 전문가입니다.
            다음 작업을 수행하세요:
            1. 주요 키워드 3-5개 추출
            2. 응답의 창의성 점수 (0-1)
            3. 다음 질문을 위한 제안사항
            
            JSON 형식으로 응답하세요:
            {{
                "keywords": ["키워드1", "키워드2", ...],
                "quality_score": 0.8,
                "suggestions": ["제안1", "제안2", ...]
            }}
            """
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"분석할 텍스트: '{user_input}'"}
            ]
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,
                temperature=0.3
            )
            
            # 토큰 사용량 업데이트
            if response.usage and self.conversation_manager:
                self.conversation_manager.update_token_usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens
                )

            # JSON 응답 파싱 시도
            response_text = response.choices[0].message.content
            try:
                parsed_result = json.loads(response_text)
                analysis_result.update(parsed_result)
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 키워드만 추출
                keywords = self._extract_keywords_fallback(response_text)
                analysis_result["keywords"] = keywords
            
            # 현재 단계에 결과 반영
            if analysis_result["keywords"]:
                self.story_elements[self.story_stage]["topics"].update(analysis_result["keywords"])
                self.story_elements[self.story_stage]["count"] += 1
                
            # 품질 점수 기록
            self.quality_scores.append(analysis_result["quality_score"])
            
        except Exception as e:
            logger.error(f"사용자 응답 분석 중 오류: {e}")
            # 오류 시 기본 처리
            self.story_elements[self.story_stage]["count"] += 1
        
        return analysis_result
    
    def _extract_keywords_fallback(self, text: str) -> List[str]:
        """JSON 파싱 실패 시 키워드 추출 폴백"""
        keywords = []
        # 간단한 키워드 추출 로직
        words = text.replace(',', ' ').replace('.', ' ').split()
        for word in words:
            clean_word = word.strip().strip('"\'')
            if len(clean_word) > 1 and clean_word not in keywords:
                keywords.append(clean_word)
                if len(keywords) >= 5:
                    break
        return keywords
    
    def should_transition_to_next_stage(self, conversation_length: int) -> bool:
        """
        다음 단계로 전환해야 하는지 결정
        
        Args:
            conversation_length: 현재까지의 전체 대화 메시지 수
            
        Returns:
            bool: 전환 여부
        """
        current_turn = conversation_length // 2
        
        # 최소 대화 턴 수 확인
        if current_turn - self.last_stage_transition < 2:
            return False
        
        # 단계별 전환 기준
        transition_criteria = {
            "character": lambda: (
                self.story_elements["character"]["count"] >= 3 or 
                current_turn > 4 or
                len(self.story_elements["character"]["topics"]) >= 5
            ),
            "setting": lambda: (
                self.story_elements["setting"]["count"] >= 2 or 
                current_turn > 8 or
                len(self.story_elements["setting"]["topics"]) >= 3
            ),
            "problem": lambda: (
                self.story_elements["problem"]["count"] >= 2 or 
                current_turn > 12 or
                len(self.story_elements["problem"]["topics"]) >= 3
            ),
            "resolution": lambda: False  # 마지막 단계
        }
        
        # 기본 전환 조건 확인
        should_transition = transition_criteria.get(self.story_stage, lambda: False)()
        
        # 확률적 전환 로직
        stages = ["character", "setting", "problem", "resolution"]
        current_index = stages.index(self.story_stage)
        
        if current_index < len(stages) - 1:
            # 기본 전환 확률 (단계가 진행될수록 증가)
            base_prob = 0.1 * (current_index + 1)
            
            # 대화 길이에 따른 추가 확률
            turn_factor = min(0.4, 0.05 * (current_turn // 2))
            
            # 품질 점수에 따른 조정
            avg_quality = sum(self.quality_scores[-3:]) / len(self.quality_scores[-3:]) if self.quality_scores else 0.5
            quality_factor = 0.2 * avg_quality
            
            # 최종 전환 확률
            final_prob = base_prob + turn_factor + quality_factor
            
            # 기본 조건을 만족하면 확률 증가
            if should_transition:
                final_prob += 0.3
            
            return random.random() < final_prob
        
        return False
    
    def transition_to_next_stage(self, conversation_length: int) -> bool:
        """다음 단계로 전환"""
        stages = ["character", "setting", "problem", "resolution"]
        current_index = stages.index(self.story_stage)
        
        if current_index < len(stages) - 1:
            old_stage = self.story_stage
            self.story_stage = stages[current_index + 1]
            self.last_stage_transition = conversation_length // 2
            
            logger.info(f"이야기 수집 단계 전환: {old_stage} → {self.story_stage}")
            return True
        
        return False
    
    def get_story_elements(self) -> Dict[str, Dict[str, Any]]:
        """수집된 이야기 요소 반환 (set을 list로 변환)"""
        result = {}
        for stage, data in self.story_elements.items():
            result[stage] = {
                "count": data["count"],
                "topics": list(data["topics"])
            }
        return result
    
    # ==========================================
    # 이야기 분석 및 생성 메서드
    # ==========================================
    
    def _get_enhanced_prompt(self, prompt: str) -> str:
        """Enhanced 모드에서 프롬프트 처리"""
        if not self.enhanced_mode:
            return prompt
        
    
    def get_conversation_summary(self, conversation_history: List[Dict], 
                                 child_name: str = "", age_group: int = 5) -> str:
        """
        대화 내용 요약 생성
        
        Args:
            conversation_history: 대화 내역
            child_name: 아이 이름
            age_group: 아이 연령대
            
        Returns:
            str: 대화 요약
        """
        if not self.openai_client:
            return "OpenAI 클라이언트가 초기화되지 않아 요약을 생성할 수 없습니다."
        
        if len(conversation_history) < 3:
            return f"{child_name}와의 대화가 아직 충분하지 않습니다."
        
        # 최근 대화 내용 추출
        recent_messages = conversation_history[-10:]
        conversation_text = ""
        
        for message in recent_messages:
            role = "아이" if message["role"] == "user" else "부기"
            conversation_text += f"{role}: {message['content']}\n"
        
        # 수집된 요소 정보 추가
        elements_summary = self._get_elements_summary()
        
        system_message = f"""
        당신은 {age_group}세 아이 {child_name}와의 대화를 요약하는 전문가입니다.
        
        다음 정보를 포함하여 요약해주세요:
        1. 아이가 보인 주요 관심사
        2. 언급된 이야기 요소들
        3. 대화의 전반적인 흐름과 분위기
        4. 수집된 이야기 단계별 진행 상황
        
        현재 수집 상황: {elements_summary}
        """
        
        try:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"다음 대화를 요약해주세요:\n\n{conversation_text}"}
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=400,
                temperature=0.5
            )
            
            # 토큰 사용량 업데이트
            if response.usage and self.conversation_manager:
                self.conversation_manager.update_token_usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens
                )

            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"대화 요약 생성 중 오류: {e}")
            return f"대화 요약 생성 중 오류가 발생했습니다: {str(e)}"
    
    def suggest_story_theme(self, conversation_history: List[Dict], 
                          child_name: str = "", age_group: int = 5,
                          interests: List[str] = None, 
                          story_collection_prompt: str = "") -> Dict:
        """
        수집된 대화를 바탕으로 이야기 주제 제안
        
        Args:
            conversation_history: 대화 내역
            child_name: 아이 이름
            age_group: 아이 연령대
            interests: 아이의 관심사
            story_collection_prompt: 이야기 수집 프롬프트
            
        Returns:
            Dict: 이야기 주제 및 구조
        """
        if len(conversation_history) < 4:
            return self._get_default_story_structure(child_name, age_group)
        
        # RAG 시스템을 통한 주제 풍부화
        enriched_context = ""
        if self.rag_system and interests:
            base_theme = f"{interests[0]}와 관련된 모험" if interests else "모험과 우정"
            try:
                enriched_context = self.rag_system.enrich_story_theme(base_theme, age_group)
            except Exception as e:
                logger.warning(f"RAG 시스템 사용 중 오류: {e}")
        
        # 대화 요약 및 요소 정보
        conversation_summary = self.get_conversation_summary(conversation_history, child_name, age_group)
        elements_info = self._get_detailed_elements_info()
        
        system_message = f"""
        당신은 {age_group}세 아이를 위한 동화 구성 전문가입니다.
        
        다음 정보를 바탕으로 완전한 이야기 구조를 제안해주세요:
        - 아이 이름: {child_name}
        - 관심사: {', '.join(interests) if interests else '다양한 주제'}
        - 수집된 이야기 요소: {elements_info}
        
        JSON 형식으로 응답해주세요:
        {{
            "title": "이야기 제목",
            "theme": "주요 주제",
            "characters": ["등장인물1", "등장인물2"],
            "setting": "배경 설명",
            "plot_summary": "줄거리 요약",
            "educational_value": "교육적 가치",
            "target_age": {age_group},
            "estimated_length": "예상 길이",
            "key_scenes": ["주요 장면1", "주요 장면2"]
        }}
        """
        
        try:
            prompt_content = story_collection_prompt
            if enriched_context:
                prompt_content += f"\n\n참고 정보:\n{enriched_context}"
            prompt_content += f"\n\n대화 요약:\n{conversation_summary}"
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_content}
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            # 토큰 사용량 업데이트
            if response.usage and self.conversation_manager:
                self.conversation_manager.update_token_usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens
                )

            # JSON 응답 파싱
            response_text = response.choices[0].message.content
            story_data = self._parse_story_response(response_text)
            
            # 이야기 개요 저장
            self.story_outline = story_data
            
            # RAG 시스템에 추가
            if self.rag_system:
                try:
                    self.rag_system.add_story_to_vectordb(story_data)
                except Exception as e:
                    logger.error(f"RAG 시스템에 스토리 추가 실패: {e}")
            
            return story_data
            
        except Exception as e:
            logger.error(f"이야기 주제 생성 중 오류: {e}")
            return self._get_error_story_structure(str(e), age_group)
    
    def _parse_story_response(self, response_text: str) -> Dict:
        """GPT 응답에서 이야기 구조 파싱"""
        try:
            # JSON 부분 추출
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                # JSON이 없는 경우 수동 파싱
                return self._manual_parse_story(response_text)
                
        except json.JSONDecodeError:
            return self._manual_parse_story(response_text)
    
    def _manual_parse_story(self, response_text: str) -> Dict:
        """텍스트에서 이야기 요소 수동 추출"""
        story_data = {
            "title": "추출된 이야기",
            "theme": "",
            "characters": [],
            "setting": "",
            "plot_summary": "",
            "educational_value": "우정과 협력",
            "target_age": 5,
            "estimated_length": "중간",
            "key_scenes": []
        }
        
        # 주제 추출
        for pattern in ["주제:", "테마:", "theme:"]:
            if pattern in response_text.lower():
                start = response_text.lower().find(pattern) + len(pattern)
                end = response_text.find("\n", start)
                if end == -1:
                    end = start + 100
                story_data["theme"] = response_text[start:end].strip()
                break
        
        # 전체 텍스트를 줄거리로 사용 (길이 제한)
        if not story_data["plot_summary"]:
            story_data["plot_summary"] = response_text[:300] + "..."
        
        return story_data
    
    # ==========================================
    # 유틸리티 및 상태 관리 메서드
    # ==========================================
    
    def _get_elements_summary(self) -> str:
        """수집된 요소들의 간단한 요약"""
        summary_parts = []
        for stage, data in self.story_elements.items():
            if data["count"] > 0:
                summary_parts.append(f"{stage}: {data['count']}개 수집")
        return ", ".join(summary_parts) if summary_parts else "아직 수집된 요소 없음"
    
    def _get_detailed_elements_info(self) -> str:
        """수집된 요소들의 상세 정보"""
        info_parts = []
        for stage, data in self.story_elements.items():
            if data["topics"]:
                topics_str = ", ".join(list(data["topics"])[:3])  # 최대 3개만
                info_parts.append(f"{stage}: {topics_str}")
        return " | ".join(info_parts) if info_parts else "수집된 세부 정보 없음"
    
    def _get_default_story_structure(self, child_name: str, age_group: int) -> Dict:
        """기본 이야기 구조 반환"""
        return {
            "title": f"{child_name}의 모험",
            "theme": "우정과 모험",
            "characters": [child_name, "친구들"],
            "setting": "마법의 숲",
            "plot_summary": "더 많은 이야기를 들려줘!",
            "educational_value": "우정과 협력의 중요성",
            "target_age": age_group,
            "estimated_length": "짧음",
            "key_scenes": ["만남", "모험", "해결"]
        }
    
    def _get_error_story_structure(self, error_msg: str, age_group: int) -> Dict:
        """오류 시 기본 구조 반환"""
        return {
            "title": "이야기 생성 오류",
            "theme": f"오류 발생: {error_msg}",
            "characters": ["미정"],
            "setting": "미정",
            "plot_summary": "이야기 생성 중 오류가 발생했습니다.",
            "educational_value": "미정",
            "target_age": age_group,
            "estimated_length": "미정",
            "key_scenes": []
        }
    
    def get_story_outline(self) -> Optional[Dict]:
        """생성된 이야기 개요 반환"""
        return self.story_outline
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """수집 통계 반환"""
        avg_quality = sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0
        
        return {
            "current_stage": self.story_stage,
            "total_interactions": self.total_interactions,
            "elements_collected": sum(data["count"] for data in self.story_elements.values()),
            "average_quality": round(avg_quality, 2),
            "stage_progress": {
                stage: {
                    "count": data["count"],
                    "topics_count": len(data["topics"])
                }
                for stage, data in self.story_elements.items()
            }
        }
    
    def update_from_saved_data(self, saved_data: Dict) -> None:
        """저장된 데이터로부터 상태 복원"""
        if "story_stage" in saved_data:
            self.story_stage = saved_data["story_stage"]
        
        if "story_elements" in saved_data:
            for stage, data in saved_data["story_elements"].items():
                if stage in self.story_elements:
                    self.story_elements[stage]["count"] = data.get("count", 0)
                    self.story_elements[stage]["topics"] = set(data.get("topics", []))
        
        if "last_stage_transition" in saved_data:
            self.last_stage_transition = saved_data["last_stage_transition"]
        
        if "story_outline" in saved_data:
            self.story_outline = saved_data["story_outline"]
        
        if "total_interactions" in saved_data:
            self.total_interactions = saved_data["total_interactions"]
        
        if "quality_scores" in saved_data:
            self.quality_scores = saved_data["quality_scores"]
        
        logger.info("이야기 엔진 상태 복원 완료")
    
    def reset(self) -> None:
        """엔진 상태 초기화"""
        self.story_stage = "character"
        self.story_elements = {
            "character": {"count": 0, "topics": set()},
            "setting": {"count": 0, "topics": set()},
            "problem": {"count": 0, "topics": set()},
            "resolution": {"count": 0, "topics": set()}
        }
        self.last_stage_transition = 0
        self.story_outline = None
        self.total_interactions = 0
        self.quality_scores = []
        
        logger.info("이야기 엔진 상태 초기화 완료")
    
    # ==========================================
    # Enhanced 모드 지원 메서드들
    # ==========================================
    
    def set_age_specific_mode(self, age_group: int):
        """연령별 특화 모드 설정"""
        self.age_group = age_group
        logger.info(f"연령별 특화 모드 설정: {age_group}세")
    
    def analyze_input(self, user_input: str, enhanced_mode: bool = False, age_group: int = None) -> Dict[str, Any]:
        """
        사용자 입력 분석 (Enhanced 모드 지원)
        
        Args:
            user_input: 사용자 입력
            enhanced_mode: Enhanced 모드 사용 여부
            age_group: 연령대
            
        Returns:
            Dict: 분석 결과
        """
        if enhanced_mode and age_group:
            # Enhanced 모드에서는 연령별 특화 분석 - 기본 분석 방식과 동일하게 처리
            logger.info(f"Enhanced 모드로 입력 분석: 연령대 {age_group}세")
            return self.analyze_user_response(user_input)
        else:
            # 기본 분석 방식 사용
            return self.analyze_user_response(user_input)

    def generate_enhanced_response(self, response_context: Dict) -> str:
        """
        Enhanced 모드에서 사용하는 GPT-4o-mini 기반 응답 생성 (ChatBot A 호환)
        실제 AI가 사용자 입력을 이해하고 맥락적으로 응답
        
        Args:
            response_context: 응답 생성을 위한 컨텍스트
                - user_input: 사용자 입력
                - analysis: 분석 결과
                - conversation_history: 대화 기록
                - child_age: 아이 나이
                - child_interests: 아이 관심사
                - enhanced_mode: Enhanced 모드 여부
                
        Returns:
            str: GPT가 생성한 자연스러운 응답
        """
        try:
            user_input = response_context.get("user_input", "")
            analysis = response_context.get("analysis", {})
            conversation_history = response_context.get("conversation_history", [])
            child_age = response_context.get("child_age", 5)
            child_interests = response_context.get("child_interests", [])
            child_name = response_context.get("child_name", "친구")
            
            logger.info(f"GPT 기반 Enhanced 응답 생성: {user_input[:50]}...")
            
            # OpenAI 클라이언트가 없으면 기본 응답
            if not self.openai_client:
                return "재미있는 이야기야! 더 말해줄래?"
            
            # 분석 결과에서 정보 추출
            stage = analysis.get("stage", self.story_stage)
            quality_score = analysis.get("quality_score", 0.5)
            keywords = analysis.get("keywords", [])
            
            # 최근 대화 히스토리 구성
            history_text = ""
            if conversation_history:
                recent_messages = conversation_history[-3:]  # 최근 3개 메시지
                for msg in recent_messages:
                    role = "아이" if msg.get("role") == "user" else "부기"
                    history_text += f"{role}: {msg.get('content', '')}\n"
            
            # 관심사 문자열 구성
            interests_str = ", ".join(child_interests) if child_interests else "다양한 주제"
            
            # 수집 단계별 가이드
            stage_guides = {
                "character": "등장인물(주인공, 친구들, 동물 등)에 대해 더 자세히 알고 싶어",
                "setting": "이야기가 일어나는 장소나 배경에 대해 궁금해",
                "problem": "어떤 문제나 모험이 생기는지 알고 싶어",
                "resolution": "문제를 어떻게 해결하는지 듣고 싶어"
            }
            current_guide = stage_guides.get(stage, "이야기에 대해 더 알고 싶어")
            
            # GPT-4o-mini에게 전달할 시스템 메시지
            system_message = f"""당신은 {child_age}세 아이 {child_name}와 대화하는 친근한 AI 친구 '부기'입니다.

역할과 목적:
- 아이의 상상력을 격려하고 이야기 요소를 자연스럽게 수집
- 현재 {stage} 단계에서 {current_guide}
- 아이의 연령({child_age}세)에 맞는 쉬운 말과 재미있는 표현 사용

아이 정보:
- 이름: {child_name}
- 나이: {child_age}세  
- 관심사: {interests_str}

대화 규칙:
1. 따뜻하고 격려적인 태도 유지
2. 아이의 말에 구체적으로 반응하기
3. 호기심을 자극하는 후속 질문하기
4. 상상력을 칭찬하고 더 이끌어내기
5. 안전하고 긍정적인 내용만 다루기

현재 상황:
- 수집 단계: {stage}
- 사용자 입력 품질: {quality_score:.1f}/1.0
- 추출된 키워드: {', '.join(keywords) if keywords else '없음'}"""

            # 사용자 메시지 구성
            user_message = f"""최근 대화:
{history_text}

아이의 새로운 말: "{user_input}"

위 대화를 바탕으로 아이에게 자연스럽고 격려적인 응답을 해주세요. 아이의 말에 구체적으로 반응하고, {stage} 단계와 관련된 재미있는 후속 질문을 포함해주세요."""

            # GPT-4o-mini로 응답 생성
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=150,
                temperature=0.8
            )
            
            # 토큰 사용량 업데이트
            if response.usage and self.conversation_manager:
                self.conversation_manager.update_token_usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens
                )

            generated_response = response.choices[0].message.content.strip()
            logger.info(f"GPT 응답 생성 완료: {len(generated_response)}글자")
            
            return generated_response
                
        except Exception as e:
            logger.error(f"GPT Enhanced 응답 생성 중 오류: {e}")
            # 오류 시 기본 응답
            return f"와! {child_name}의 이야기 정말 재미있어! 더 자세히 말해줄래?"

    def generate_contextual_response(self, user_input: str, analysis_result: Dict, conversation_history: List[Dict]) -> str:
        """
        기본 모드에서 사용하는 GPT-4o-mini 기반 맥락적 응답 생성 (ChatBot A 호환)
        실제 AI가 대화 맥락을 이해하고 적절한 응답 생성
        
        Args:
            user_input: 사용자 입력
            analysis_result: 분석 결과
            conversation_history: 대화 기록
            
        Returns:
            str: GPT가 생성한 맥락적 응답
        """
        try:
            logger.info(f"GPT 기반 맥락적 응답 생성: {user_input[:50]}...")
            
            # OpenAI 클라이언트가 없으면 기본 응답
            if not self.openai_client:
                return "계속 이야기해줘! 더 듣고 싶어!"
            
            # 분석 결과에서 정보 추출
            keywords = analysis_result.get("keywords", [])
            quality_score = analysis_result.get("quality_score", 0.5)
            stage = analysis_result.get("stage", self.story_stage)
            
            # 최근 대화 히스토리 구성
            history_text = ""
            if conversation_history:
                recent_messages = conversation_history[-2:]  # 최근 2개 메시지
                for msg in recent_messages:
                    role = "아이" if msg.get("role") == "user" else "부기"
                    history_text += f"{role}: {msg.get('content', '')}\n"
            
            # GPT-4o-mini에게 전달할 시스템 메시지
            system_message = f"""당신은 아이와 대화하는 친근한 AI 친구 '부기'입니다.

역할:
- 아이의 이야기에 관심을 보이고 격려하기
- 자연스럽고 친근한 반응 보이기
- 이야기를 계속 이끌어나가기

대화 원칙:
1. 아이의 말에 구체적으로 반응
2. 긍정적이고 격려적인 태도
3. 호기심을 보이며 더 듣고 싶어하는 모습
4. 간단하고 이해하기 쉬운 말 사용
5. 2-3문장으로 간결하게 응답

현재 상황:
- 수집 단계: {stage}
- 입력 품질: {quality_score:.1f}/1.0
- 키워드: {', '.join(keywords) if keywords else '없음'}"""

            user_message = f"""최근 대화:
{history_text}

아이의 말: "{user_input}"

위 내용에 자연스럽게 반응하며, 아이가 계속 이야기하고 싶어지는 응답을 해주세요."""

            # GPT-4o-mini로 응답 생성
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100,
                temperature=0.7
            )
            
            # 토큰 사용량 업데이트
            if response.usage and self.conversation_manager:
                self.conversation_manager.update_token_usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens
                )

            generated_response = response.choices[0].message.content.strip()
            logger.info(f"GPT 맥락적 응답 생성 완료: {len(generated_response)}글자")
            
            return generated_response
                
        except Exception as e:
            logger.error(f"GPT 맥락적 응답 생성 중 오류: {e}")
            # 오류 시 기본 응답
            return "정말 흥미로운 이야기야! 더 말해줄래?"

    def create_story_summary(self, conversation_history: List[Dict], 
                            child_name: str = "", age_group: int = 5,
                            interests: List[str] = None) -> Dict:
        """
        충분한 대화가 수집되었을 때 이야기 요약을 생성 (ChatBot B 전달용)
        
        Args:
            conversation_history: 대화 기록
            child_name: 아이 이름
            age_group: 연령대
            interests: 관심사 목록
            
        Returns:
            Dict: 이야기 요약 데이터
        """
        try:
            # 충분한 대화가 있는지 확인
            if len(conversation_history) >= 5:
                logger.info("충분한 대화가 수집되어 이야기 요약을 생성합니다")
                return self.suggest_story_theme(
                    conversation_history=conversation_history,
                    child_name=child_name,
                    age_group=age_group,
                    interests=interests,
                    story_collection_prompt="수집된 대화를 바탕으로 이야기 요약을 생성해주세요"
                )
            else:
                return {
                    "title": f"{child_name}의 이야기",
                    "theme": "아직 수집 중",
                    "characters": [],
                    "setting": "",
                    "plot_summary": "더 많은 대화가 필요해!",
                    "educational_value": "상상력과 창의성",
                    "target_age": age_group,
                    "estimated_length": "짧음",
                    "key_scenes": [],
                    "collection_status": "진행 중"
                }
        except Exception as e:
            logger.error(f"이야기 요약 생성 중 오류: {e}")
            return {
                "title": "이야기 생성 오류",
                "theme": "오류",
                "plot_summary": "이야기 요약 생성 중 오류가 발생했습니다",
                "collection_status": "오류"
            }