"""
ChatBot A 통합 이야기 엔진

이야기 수집, 분석, 구조화를 담당하는 통합 엔진
기존 StoryCollector, StoryAnalyzer, StoryCollectionEngine의 기능을 하나로 통합
"""
from typing import Dict, List, Any, Optional, Set
import random
import json

from shared.utils.logging_utils import get_module_logger
from shared.utils.openai_utils import generate_chat_completion

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
    
    def __init__(self, openai_client=None, rag_system=None, conversation_manager=None, 
                 enhanced_mode=False, performance_tracking=False):
        """
        이야기 엔진 초기화
        
        Args:
            openai_client: OpenAI API 클라이언트
            rag_system: RAG 시스템 인스턴스
            conversation_manager: ConversationManager 인스턴스
            enhanced_mode: Enhanced 모드 사용 여부
            performance_tracking: 성능 추적 활성화 여부
        """
        # === Enhanced 모드 설정 ===
        self.enhanced_mode = enhanced_mode
        self.performance_tracking = performance_tracking
        
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
        self.openai_client = openai_client
        self.rag_system = rag_system
        self.conversation_manager = conversation_manager
        self.story_outline = None  # 생성된 이야기 개요
        
        # === 수집 통계 ===
        self.total_interactions = 0
        self.quality_scores = []
        
        logger.info(f"통합 이야기 엔진 초기화 완료 - Enhanced: {enhanced_mode}, Performance Tracking: {performance_tracking}")
    
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
        if len(conversation_history) < 5:
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
            "plot_summary": "아직 충분한 대화가 수집되지 않았습니다. 더 많은 이야기를 들려주세요!",
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
            # Enhanced 모드에서는 연령별 특화 분석
            return self._analyze_input_enhanced(user_input, age_group)
        else:
            # 기본 분석 방식 사용
            return self.analyze_user_response(user_input)
    
    def _analyze_input_enhanced(self, user_input: str, age_group: int) -> Dict[str, Any]:
        """Enhanced 모드 사용자 입력 분석"""
        analysis_result = {
            "keywords": [],
            "quality_score": 0.5,
            "stage": self.story_stage,
            "suggestions": [],
            "age_appropriateness": True,
            "creativity_level": "medium"
        }
        
        if not self.openai_client:
            return analysis_result
        
        try:
            system_message = f"""
            당신은 {age_group}세 아이의 응답을 분석하는 전문가입니다.
            현재 이야기 수집 단계: {self.story_stage}
            
            다음을 분석하여 JSON으로 응답하세요:
            1. 주요 키워드 추출
            2. 창의성 점수 (0-1)
            3. 연령 적절성 평가
            4. 다음 질문 제안사항
            
            JSON 형식:
            {{
                "keywords": ["키워드1", "키워드2"],
                "quality_score": 0.8,
                "age_appropriateness": true,
                "creativity_level": "high",
                "suggestions": ["제안1", "제안2"]
            }}
            """
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"분석할 응답: '{user_input}'"}
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content
            try:
                parsed_result = json.loads(response_text)
                analysis_result.update(parsed_result)
            except json.JSONDecodeError:
                keywords = self._extract_keywords_fallback(response_text)
                analysis_result["keywords"] = keywords
            
            # 현재 단계에 결과 반영
            if analysis_result["keywords"]:
                self.story_elements[self.story_stage]["topics"].update(analysis_result["keywords"])
                self.story_elements[self.story_stage]["count"] += 1
                
        except Exception as e:
            logger.error(f"Enhanced 입력 분석 중 오류: {e}")
        
        return analysis_result
    
    def generate_enhanced_response(self, response_context: Dict) -> str:
        """Enhanced 응답 생성"""
        if not self.openai_client:
            return "죄송해요, 지금은 응답을 생성할 수 없어요."
        
        try:
            user_input = response_context.get("user_input", "")
            child_age = response_context.get("child_age", 5)
            child_interests = response_context.get("child_interests", [])
            analysis = response_context.get("analysis", {})
            
            system_message = f"""
            당신은 {child_age}세 아이와 대화하는 친근한 AI 부기입니다.
            현재 이야기 수집 단계: {self.story_stage}
            
            아이의 관심사: {', '.join(child_interests) if child_interests else '다양한 주제'}
            
            Enhanced 모드로 연령에 맞는 친근하고 격려적인 응답을 해주세요.
            이야기 요소를 자연스럽게 수집하면서 아이의 상상력을 자극해주세요.
            """
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"아이의 말: '{user_input}'"}
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,
                temperature=0.8
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Enhanced 응답 생성 중 오류: {e}")
            return "와! 정말 재미있는 이야기네요. 더 들려주세요!"
    
    def generate_contextual_response(self, user_input: str, analysis_result: Dict, conversation_history: List[Dict]) -> str:
        """맥락적 응답 생성"""
        if not self.openai_client:
            return "그래요! 더 재미있는 이야기를 들려주세요!"
        
        try:
            # 최근 대화 맥락 추출
            recent_context = ""
            if conversation_history:
                recent_messages = conversation_history[-4:]
                for msg in recent_messages:
                    role = "아이" if msg["role"] == "user" else "부기"
                    recent_context += f"{role}: {msg['content']}\n"
            
            system_message = f"""
            당신은 아이와 이야기를 만들어가는 친근한 AI 부기입니다.
            현재 단계: {self.story_stage}
            
            최근 대화:
            {recent_context}
            
            자연스럽고 격려적인 응답으로 이야기를 이어가세요.
            """
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"맥락적 응답 생성 중 오류: {e}")
            return "정말 흥미로운 이야기예요! 계속 들려주세요."
    
    def suggest_enhanced_theme(self, age_group: int, interests: List[str]) -> Dict:
        """Enhanced 테마 제안"""
        base_themes = {
            4: ["동물 친구들의 모험", "마법의 숲 탐험", "용감한 꼬마 영웅"],
            5: ["우주 여행", "바닷속 모험", "로봇 친구와의 하루"],
            6: ["시간 여행", "비밀 정원 발견", "마법사 수련"],
            7: ["드래곤과의 우정", "잃어버린 보물 찾기", "신비한 섬 모험"],
            8: ["미래 도시 탐험", "고대 문명 발견", "마법 학교 이야기"],
            9: ["차원 이동 모험", "신화 속 영웅 이야기", "과학자가 되는 꿈"]
        }
        
        age_themes = base_themes.get(age_group, base_themes[5])
        
        # 관심사 기반 테마 조합
        if interests:
            interest_based_themes = []
            for interest in interests[:2]:  # 최대 2개 관심사
                for theme in age_themes:
                    interest_based_themes.append(f"{interest}를 좋아하는 {theme}")
            
            if interest_based_themes:
                age_themes.extend(interest_based_themes)
        
        selected_theme = random.choice(age_themes)
        
        # 관심사 기반 캐릭터 및 설정 생성
        main_interest = interests[0] if interests else "모험"
        characters = ["주인공"]
        if "공룡" in interests:
            characters.extend(["티라노사우루스 친구", "트리케라톱스"])
        elif "우주" in interests:
            characters.extend(["외계인 친구", "로봇 조종사"])
        elif "로봇" in interests:
            characters.extend(["도우미 로봇", "메카닉 친구"])
        else:
            characters.extend(["동물 친구", "마법사"])
        
        # Enhanced 줄거리 생성
        plot_summary = f"{age_group}세 주인공이 {main_interest}와 관련된 모험을 떠나는 이야기입니다. "
        plot_summary += f"용감한 주인공은 친구들과 함께 어려움을 극복하고 소중한 것을 배우게 됩니다."
        
        return {
            # 기존 suggest_story_theme 호환 필드들
            "title": f"{main_interest} 모험 이야기",
            "theme": selected_theme,
            "characters": characters,
            "setting": f"{main_interest}가 펼쳐지는 신비한 세계",
            "plot_summary": plot_summary,
            "educational_value": "용기와 우정, 협력의 중요성",
            "target_age": age_group,
            "estimated_length": "중간",
            "key_scenes": [
                "모험의 시작",
                f"{main_interest}와의 만남",
                "어려움 극복",
                "성공과 성장"
            ],
            # Enhanced 전용 필드들
            "suggested_theme": selected_theme,
            "age_group": age_group,
            "interests_used": interests[:2] if interests else [],
            "alternative_themes": random.sample(age_themes, min(3, len(age_themes)))
        }
    
    def create_enhanced_story_outline(self, conversation_history: List[Dict], child_age: int, child_interests: List[str], child_name: str) -> Dict:
        """Enhanced 스토리 개요 생성"""
        try:
            # 대화에서 수집된 요소들 분석
            elements_info = self._get_detailed_elements_info()
            conversation_summary = self.get_conversation_summary(conversation_history, child_name, child_age)
            
            # Enhanced 프롬프트로 구조화된 스토리 생성
            story_prompt = f"""
            {child_age}세 {child_name}와의 대화를 바탕으로 완전한 동화를 구성해주세요.
            
            아이 정보:
            - 이름: {child_name}
            - 나이: {child_age}세  
            - 관심사: {', '.join(child_interests) if child_interests else '다양한 주제'}
            
            수집된 이야기 요소: {elements_info}
            대화 요약: {conversation_summary}
            
            다음 JSON 구조로 응답해주세요:
            {{
                "title": "동화 제목",
                "main_character": "주인공",
                "setting": "배경",
                "problem": "해결할 문제",
                "solution": "해결 방법",
                "message": "전달하고자 하는 메시지",
                "scenes": ["장면1", "장면2", "장면3"],
                "enhanced_features": {{
                    "age_appropriate_language": true,
                    "educational_value": "배울 수 있는 점",
                    "emotional_elements": ["감정 요소들"]
                }}
            }}
            """
            
            if self.openai_client:
                messages = [
                    {"role": "system", "content": "당신은 아동 문학 전문가입니다."},
                    {"role": "user", "content": story_prompt}
                ]
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=1500,
                    temperature=0.7
                )
                
                response_text = response.choices[0].message.content
                try:
                    story_outline = json.loads(response_text)
                    self.story_outline = story_outline
                    return story_outline
                except json.JSONDecodeError:
                    pass
            
            # 폴백: 기본 구조 반환
            return self._get_default_enhanced_outline(child_name, child_age, child_interests)
            
        except Exception as e:
            logger.error(f"Enhanced 스토리 개요 생성 중 오류: {e}")
            return self._get_default_enhanced_outline(child_name, child_age, child_interests)
    
    def _get_default_enhanced_outline(self, child_name: str, child_age: int, interests: List[str]) -> Dict:
        """기본 Enhanced 스토리 개요"""
        main_interest = interests[0] if interests else "모험"
        
        return {
            "title": f"{child_name}의 {main_interest} 모험",
            "main_character": child_name,
            "setting": "마법의 세계",
            "problem": "친구를 도와야 하는 상황",
            "solution": "용기와 지혜로 문제 해결",
            "message": "우정과 용기의 소중함",
            "scenes": [
                f"{child_name}가 모험을 시작하는 장면",
                "어려움에 부딪히는 장면", 
                "친구들과 함께 문제를 해결하는 장면"
            ],
            "enhanced_features": {
                "age_appropriate_language": True,
                "educational_value": "협동과 문제해결 능력",
                "emotional_elements": ["우정", "용기", "성취감"]
            }
        }
    
    def create_story_outline(self) -> Dict:
        """기본 스토리 개요 생성"""
        return self.get_story_outline() or self._get_default_story_structure("주인공", 5)
    
    def create_enhanced_summary(self, conversation_history: List[Dict], age_group: int) -> str:
        """Enhanced 대화 요약"""
        if not conversation_history:
            return "아직 대화가 시작되지 않았습니다."
        
        # Enhanced 모드에서는 더 상세한 분석 포함
        summary = self.get_conversation_summary(conversation_history, "", age_group)
        
        if self.enhanced_mode:
            # 추가 분석 정보 포함
            elements_summary = self._get_elements_summary()
            summary += f"\n\n수집 진행 상황: {elements_summary}"
            
            if self.quality_scores:
                avg_quality = sum(self.quality_scores) / len(self.quality_scores)
                summary += f"\n평균 응답 품질: {avg_quality:.2f}/1.0"
        
        return summary
    
    def create_conversation_summary(self) -> str:
        """기본 대화 요약 (레거시 호환)"""
        if self.conversation_manager:
            history = self.conversation_manager.get_all_messages()
            return self.get_conversation_summary(history)
        return "대화 기록이 없습니다."
    
    def suggest_enhanced_element(self, user_input: str, age_group: int, interests: List[str]) -> str:
        """Enhanced 스토리 요소 제안"""
        if not self.openai_client:
            return "그 아이디어 정말 좋아요! 더 자세히 말해주세요."
        
        try:
            system_message = f"""
            당신은 {age_group}세 아이의 창의성을 자극하는 전문가입니다.
            현재 단계: {self.story_stage}
            아이 관심사: {', '.join(interests) if interests else '다양한 주제'}
            
            아이의 아이디어를 바탕으로 더 흥미로운 요소를 제안해주세요.
            연령에 맞고 교육적 가치가 있는 제안을 해주세요.
            """
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"아이의 아이디어: '{user_input}'"}
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=150,
                temperature=0.8
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Enhanced 요소 제안 중 오류: {e}")
            return "와! 그런 아이디어는 어떨까요? 정말 재미있을 것 같아요!"
    
    def suggest_story_element(self, user_input: str) -> str:
        """기본 스토리 요소 제안"""
        # 간단한 키워드 기반 제안
        suggestions = {
            "character": ["용감한 기사", "마법사", "귀여운 동물 친구"],
            "setting": ["마법의 숲", "구름 위의 성", "바닷속 궁전"],
            "problem": ["잃어버린 보물", "마법에 걸린 친구", "악한 마녀"],
            "resolution": ["용기로 극복", "친구들과 협력", "지혜로운 해결"]
        }
        
        stage_suggestions = suggestions.get(self.story_stage, suggestions["character"])
        suggestion = random.choice(stage_suggestions)
        
        return f"그런 아이디어는 어때요? '{suggestion}' 같은 건 어떨까요?" 