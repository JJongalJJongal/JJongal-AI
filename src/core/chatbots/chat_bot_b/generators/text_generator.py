"""
텍스트 생성기 (Enhanced for Advanced Prompt System)

LangChain + ChromaDB RAG System과 개선된 프롬프트 엔지니어링을 활용한 한국 동화 생성
- 구조화된 프롬프트 접근법 (Role → Objective → Instructions → Reasoning → Output → Examples)
- 연령별 특화 프롬프트 (4-7세, 8-9세)
- 체인 오브 소트 추론 통합
- 성능 추적 및 최적화
"""
from src.shared.utils.logging import get_module_logger
import uuid
import time
from typing import Dict, List, Optional, Callable, Any
import json
import re
import asyncio

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Project imports
from .base_generator import BaseGenerator
from src.data.vector_db.core import ModernVectorDB as VectorDB
from src.data.vector_db.query import get_similar_stories

# logging 설정
logger = get_module_logger(__name__)

class TextGenerator(BaseGenerator):
    """
    개선된 텍스트 생성기
    
    Features:
    - 구조화된 프롬프트 시스템 (OpenAI 권장 형식)
    - 연령별 맞춤 생성 (4-7세, 8-9세)
    - 체인 오브 소트 추론 통합
    - 성능 추적 및 메트릭
    - A/B 테스팅 지원 준비
    """
    
    def __init__(self,
                 openai_client = None,
                 vector_db_path = None,
                 collection_name: str = "fairy_tales",
                 prompts_file_path: str = "chatbot/data/prompts/chatbot_b_prompts.json",
                 max_retries: int = 3,
                 model_name: str = "gpt-4o",
                 temperature: float = 0.7,
                 enable_performance_tracking: bool = True,
                 model_kwargs = None):
        """
        Args:
            openai_client: OpenAI 클라이언트
            vector_db_path: ChromaDB 데이터베이스 경로
            collection_name: ChromaDB 컬렉션 이름
            prompts_file_path: 개선된 프롬프트 파일 경로
            max_retries: 최대 재시도 횟수
            model_name: 사용할 LLM 모델명
            temperature: 생성 온도
            enable_performance_tracking: 성능 추적 활성화
            model_kwargs: LLM 모델 키워드 인수 (ex: {"max_tokens": 1000})
        """
        super().__init__(max_retries=max_retries, timeout=180.0)
        
        self.openai_client = openai_client # OpenAI 클라이언트
        
        # 통일된 벡터DB 경로 설정
        if vector_db_path is None:
            import os
            chroma_base = os.getenv("CHROMA_DB_PATH", "/app/chatbot/data/vector_db")
            vector_db_path = os.path.join(chroma_base, "main")  # 기본값: main DB 사용
            logger.info(f"TextGenerator: 벡터DB 경로가 지정되지 않음. 환경변수에서 설정: {vector_db_path}")
        
        self.vector_db_path = vector_db_path # ChromaDB 경로
        self.collection_name = collection_name # ChromaDB 컬렉션 이름
        self.prompts_file_path = prompts_file_path # 개선된 프롬프트 파일 경로
        self.model_name = model_name # 사용할 LLM 모델명
        self.temperature = temperature # 생성 온도
        self.enable_performance_tracking = enable_performance_tracking # 성능 추적 활성화
        self.model_kwargs = model_kwargs or {}
        
        # Enhanced LangChain 구성
        self.vector_store = None
        self.retriever = None
        self.text_chains = {}  # 연령별 체인
        self.prompts
        
        # 성능 추적
        self.performance_metrics = {
            "generation_times": [],
            "token_usage": [],
            "success_rate": 0,
            "error_count": 0,
            "age_group_usage": {}
            }
        
        # 초기화
        self._initialize_components()
        
    def _initialize_components(self):
        """Enhanced LangChain 구성 요소 초기화"""
        try:
            # 1. Enhanced Prompts 로드
            self._load_enhanced_prompts()
            
            # 2. ChromaDB 초기화
            self._initialize_vector_db()
            
            # 3. Enhanced LangChain 체인 설정 (연령별)
            self._setup_enhanced_chains()
            
            logger.info("Enhanced TextGenerator 초기화 완료")
        
        except Exception as e:
            logger.error(f"Enhanced TextGenerator 초기화 실패: {e}")
            raise
            
    def _load_enhanced_prompts(self):
        """Enhanced 프롬프트 시스템 로드"""
        try:
            with open(self.prompts_file_path, 'r', encoding='utf-8') as f:
                self.prompts = json.load(f)
            
            # 새로운 프롬프트 구조 검증
            required_sections = ["enhanced_story_generation", "chain_of_thought_templates"]
            for section in required_sections:
                if section not in self.prompts:
                    logger.warning(f"프롬프트 섹션 '{section}' 없음. 기본값 사용")
                    
            logger.info(f"Enhanced 프롬프트 파일 로드 완료: {self.prompts_file_path}")
        
        except Exception as e:
            logger.error(f"Enhanced 프롬프트 파일 로드 실패: {e}")
            raise
    
    def _initialize_vector_db(self):
        """ChromaDB 초기화"""
        logger.info(f"VectorDB 초기화 시작 - 경로: {self.vector_db_path}")
        
        if not self.vector_db_path:
            logger.warning("ChromaDB 경로가 설정되지 않음. RAG 기능 비활성화")
            return
        
        # 경로 존재 확인
        import os
        if not os.path.exists(self.vector_db_path):
            logger.error(f"VectorDB 경로가 존재하지 않음: {self.vector_db_path}")
            self.vector_store = None
            return
            
        logger.info(f"VectorDB 경로 확인됨: {self.vector_db_path}")
        
        try:
            self.vector_store = VectorDB(
                persist_directory=self.vector_db_path
                )
            logger.info(f"VectorDB 객체 생성 완료")
            
            # vectorstore 
            if hasattr(self.vector_store, 'vectorstore') and self.vector_store.vectorstore:
                logger.info(f"VectorDB '{self.collection_name}' complete")

                try:
                    test_results = self.vector_store.similarity_search("test", k=1)
                    logger.info(f"VectorDB test complete")
                except Exception as test_e:
                    logger.warning(f"VectorDB Test failed: {test_e}")
            else:
                logger.warning(f"VectorDB link state not found")

        except Exception as e:
            logger.error(f"ChromaDB 초기화 실패: {e}")
            self.vector_store = None
            logger.warning("VectorDB 없이 기본 생성 모드로 진행")
    
    def _setup_enhanced_chains(self):
        """Enhanced LangChain 체인 설정 (연령별)"""
        try:
            # 연령별 체인 생성
            age_groups = ["age_4_7", "age_8_9"]
            
            for age_group in age_groups:
                self._create_age_specific_chain(age_group)
            
            logger.info("Enhanced LangChain 체인 설정 완료")
            
        except Exception as e:
            logger.error(f"Enhanced LangChain 체인 설정 실패: {e}")
            raise
    
    def _create_age_specific_chain(self, age_group: str):
        """연령별 특화 체인 생성"""
        try:
            if self.prompts is None:
                logger.warning("Prompts not loaded, using fallback values")
                enhanced_prompts = {}
            else:
                enhanced_prompts = self.prompts.get("enhanced_story_generation", {})

            age_config = enhanced_prompts.get(age_group, {})
            structured_prompt = age_config.get("structured_prompt", {})
            
            prompt_template = ChatPromptTemplate.from_template(structured_prompt)
            
            # LLM 모델 설정
            llm = ChatOpenAI(
                temperature=self.temperature,
                model=self.model_name,
                api_key=self.openai_client.api_key if self.openai_client else None
            )
            
            # 체인 구성
            self.text_chains[age_group] = prompt_template | llm | StrOutputParser()
            
            logger.info(f"연령별 체인 생성 완료: {age_group}")
            
        except Exception as e:
            logger.error(f"연령별 체인 생성 실패 ({age_group}): {e}")
            raise
    
    def _build_structured_prompt(self, role: str, objective: str, 
                                instructions: List[str], reasoning_steps: List[str],
                                age_group: str) -> str:
        """구조화된 프롬프트 생성 (OpenAI 권장 형식)"""
        
        # Chain-of-Thought 추론 단계 통합
        cot_templates = self.prompts("chain_of_thought_templates")
        # reasoning_template = cot_templates.get("story_development_reasoning", {}) # 현재 미사용
        
        prompt_parts = [
            f"## ROLE\n{role}",
            f"\n## OBJECTIVE\n{objective}",
            "\n## INSTRUCTIONS"
        ]
        
        # 지시사항 추가
        for i, instruction in enumerate(instructions, 1):
            prompt_parts.append(f"{i}. {instruction}")
        
        # 추론 단계 추가 (Chain-of-Thought)
        prompt_parts.append("\n## REASONING PROCESS")
        prompt_parts.append("다음 단계를 순서대로 수행하세요:")
        
        for step in reasoning_steps:
            prompt_parts.append(f"- {step}")
            
        # 출력 형식 지정
        # JSON 예시 부분을 일반 여러 줄 문자열로 변경하고, 내부 중괄호는 이중으로 이스케이프.
        output_format_json_example = """
```json
{{
  "title": "동화 제목",
  "chapters": [
    {{
      "chapter_number": 1,
      "chapter_title": "챕터 제목",
      "narration": "이곳은 조용한 숲 속... (대사가 아닌 서술 부분)",
      "dialogues": [
          {{"speaker": "토끼", "text": "안녕, 거북아!"}},
          {{"speaker": "거북이", "text": "안녕, 토끼야. 어디 가니?"}}
      ],
      "educational_point": "교육적 포인트",
      "interaction_question": "상호작용 질문"
    }}
  ],
  "reasoning_process": "추론 과정 설명"
}}
```"""

        prompt_parts.extend([
            "\n## OUTPUT FORMAT",
            "다음 JSON 형식으로 응답해주세요:",
            output_format_json_example,
            "\n## INPUT DATA",
            "동화 정보: {story_outline}",
            "참고 스토리: {reference_stories}",
            "아이 정보: {child_info}"
        ])
        
        return "\n".join(prompt_parts)
    
    def _determine_age_group(self, target_age: int) -> str:
        """연령대에 따른 체인 선택"""
        if 4 <= target_age <= 7:
            return "age_4_7"
        elif 8 <= target_age <= 9:
            return "age_8_9"
        else:
            # 기본값
            return "age_4_7" if target_age < 8 else "age_8_9"

    async def generate(self,
                       input_data: Dict[str, Any],
                       progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Enhanced 동화 텍스트 생성
        
        Args:
            input_data: {
                "theme": "동화 주제",
                "child_name": "아이 이름", 
                "age_group": "연령대",
                "target_age": 구체적 나이,
                "interests": ["관심사1", "관심사2"...],
                "plot_summary": "요약 줄거리",
                "educational_value": "교육적 가치"
            }
            progress_callback: 진행 상황 콜백
        
        Returns:
            Enhanced 스토리 데이터 with 성능 메트릭
        """
        start_time = time.time()
        story_id = str(uuid.uuid4())
        self.current_task_id = story_id
        
        try:
            # OpenAI 클라이언트 상태 확인
            logger.info(f"OpenAI 클라이언트 상태: {self.openai_client is not None}")
            logger.info(f"VectorDB 상태: {self.vector_store is not None}")
            logger.info(f"Text chains 상태: {len(self.text_chains) if self.text_chains else 0}개")
            
            # 연령대 결정
            target_age = input_data.get("target_age", input_data.get("age_group", 7))
            age_group_key = self._determine_age_group(target_age)
            logger.info(f"결정된 연령대: {target_age} -> {age_group_key}")
            
            # 진행 상황 업데이트
            if progress_callback:
                await progress_callback({
                    "step": "enhanced_text_generation",
                    "status": "starting",
                    "story_id": story_id,
                    "age_group": age_group_key,
                    "prompt_version": "2.0_enhanced"
                })
            
            # 1. RAG 검색 수행 (Enhanced)
            reference_stories = await self._retrieve_similar_stories(input_data)
            
            # 진행 상황 업데이트
            if progress_callback:
                await progress_callback({
                    "step": "rag_retrieval",
                    "status": "completed",
                    "retrieved_count": len(reference_stories)
                })
            
            # 2. Enhanced 프롬프트 데이터 준비
            prompt_data = self._prepare_enhanced_prompt_data(
                input_data, reference_stories, age_group_key
            )
            
            # 3. 연령별 체인으로 생성
            chain = self.text_chains.get(age_group_key)
            if not chain:
                raise ValueError(f"연령별 체인을 찾을 수 없음: {age_group_key}")
                
            # 진행 상황 업데이트
            if progress_callback:
                await progress_callback({
                    "step": "story_generation",
                    "status": "processing",
                    "chain_type": age_group_key
                })
            
            # 4. 텍스트 생성 with 체인 오브 소트
            logger.info(f"텍스트 생성 시작 - Chain: {age_group_key}")
            logger.info(f"Prompt 데이터 keys: {list(prompt_data.keys())}")
            
            generated_text = await chain.ainvoke(prompt_data)
            
            logger.info(f"생성된 텍스트 길이: {len(generated_text) if generated_text else 0}")
            logger.info(f"생성된 텍스트 미리보기: {generated_text[:200] if generated_text else 'None'}...")
            
            # 5. Enhanced 파싱
            story_data = self._parse_enhanced_story(generated_text)
            logger.info(f"파싱된 스토리 데이터 keys: {list(story_data.keys()) if story_data else 'None'}")
            logger.info(f"파싱된 chapters 개수: {len(story_data.get('chapters', [])) if story_data else 0}")
            
            # 6. 성능 메트릭 수집
            generation_time = time.time() - start_time
            self._update_performance_metrics(generation_time, True, age_group_key)
            
            # 7. 최종 결과 구성
            result = {
                "story_id": story_id,
                "title": story_data.get("title", "생성된 동화"),
                "chapters": story_data.get("chapters", []),
                "metadata": {
                    "generation_time": generation_time,
                    "model_used": self.model_name,
                    "age_group": age_group_key,
                    "rag_sources": [story.get("title", "Unknown") for story in reference_stories],
                    "reasoning_process": story_data.get("reasoning_process", ""),
                    "prompt_version": "2.0_enhanced",
                    "educational_integration": story_data.get("educational_integration", ""),
                    "chain_of_thought": True
                }
            }
            
            # 진행 상황 업데이트
            if progress_callback:
                await progress_callback({
                    "step": "enhanced_text_generation",
                    "status": "completed",
                    "story_id": story_id,
                    "generation_time": generation_time,
                    "chapters_count": len(story_data.get("chapters", []))
                })
            
            return result
            
        except Exception as e:
            age_group = locals().get('age_group_key', 'unknown')
            self._update_performance_metrics(0, False, age_group=age_group)
            logger.error(f"Enhanced 텍스트 생성 실패: {e}")
            raise

    async def _retrieve_similar_stories(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ChromaDB에서 유사 스토리 검색 (Enhanced)"""
        if not self.vector_store:
            logger.warning("VectorDB가 초기화되지 않음. RAG 검색 생략")
            return []
        
        try:
            # 부기의 분석 데이터를 우선 활용한 검색 쿼리 구성
            extracted_keywords = input_data.get("extracted_keywords", [])
            conversation_summary = input_data.get("conversation_summary", "")
            conversation_topics = input_data.get("conversation_analysis", {}).get("conversation_topics", [])
            
            # 기본 정보
            theme = input_data.get("theme", "")
            educational_value = input_data.get("educational_value", "")
            interests = input_data.get("interests", [])
            target_age = input_data.get("target_age", 7)
            
            # 부기 데이터가 있으면 우선 사용, 없으면 기본 정보 사용
            if extracted_keywords or conversation_summary:
                # 실제 대화 기반 검색 쿼리
                query_parts = []
                if conversation_summary:
                    query_parts.append(conversation_summary[:100])  # 요약의 처음 100자
                if extracted_keywords:
                    query_parts.extend(extracted_keywords[:5])  # 상위 5개 키워드
                if conversation_topics:
                    query_parts.extend(conversation_topics[:3])  # 상위 3개 토픽
                
                query_text = " ".join(query_parts)
                logger.info(f"부기 대화 분석 기반 RAG 검색: '{query_text[:50]}...'")
            else:
                # 기본 정보 기반 검색 쿼리 (폴백)
                query_text = f"{theme} {educational_value} {' '.join(interests)}"
                logger.info(f"기본 정보 기반 RAG 검색: '{query_text[:50]}...'")
            
            # 검색 쿼리가 비어있으면 기본값 사용
            if not query_text.strip():
                query_text = f"아이 동화 {target_age}세"
            
            # 연령대에 따른 age_group 문자열 생성
            def _get_age_group_for_filter(age: int) -> str:
                if 4 <= age <= 7:
                    return "4-7세"
                elif 8 <= age <= 9:
                    return "8-9세"
                else: # DB에 '4-7세' 데이터가 더 많을 수 있으므로 기본값 설정
                    return "4-7세"
            
            age_group_str = _get_age_group_for_filter(target_age)
            metadata_filter = {"age_group": age_group_str}
            
            logger.info(f"RAG 검색 필터 생성: target_age={target_age} -> age_group='{age_group_str}'")

            # 벡터 검색 수행 (get_similar_stories 사용)
            # get_similar_stories는 동기 함수이므로 asyncio.to_thread 사용
            results = await asyncio.to_thread(
                get_similar_stories,
                vector_db=self.vector_store,
                query_text=query_text,
                n_results=5,
                metadata_filter=metadata_filter,
                collection_name=self.collection_name,
                doc_type=None # DB에 'type' 필드가 없음
            )
            
            logger.info(f"RAG 검색 완료: {len(results)}개의 유사 스토리 반환")
            return results
            
        except Exception as e:
            logger.warning(f"Enhanced RAG 검색 실패: {e}. 빈 참고 스토리 반환")
            return []

    def _prepare_enhanced_prompt_data(self, input_data: Dict[str, Any], 
                                    reference_stories: List[Dict[str, Any]], 
                                    age_group: str) -> Dict[str, Any]:
        """Enhanced 프롬프트 데이터 준비 - 부기의 대화 분석 데이터 활용"""
        
        # 부기에서 받은 실제 대화 분석 데이터 추출
        conversation_summary = input_data.get("conversation_summary", "")
        extracted_keywords = input_data.get("extracted_keywords", [])
        conversation_analysis = input_data.get("conversation_analysis", {})
        
        logger.info(f"부기 대화 분석 데이터: keywords={len(extracted_keywords)}, summary={len(conversation_summary)}")
        
        # 기본 스토리 정보 (부기 데이터 우선 활용)
        story_outline = {
            "theme": input_data.get("theme", ""),
            "plot_summary": input_data.get("plot_summary", ""),
            "educational_value": input_data.get("educational_value", ""),
            "target_age": input_data.get("target_age", 7),
            "setting": input_data.get("setting", ""),
            "characters": input_data.get("characters", []),
            # 부기 분석 데이터 추가
            "conversation_keywords": extracted_keywords,
            "conversation_topics": conversation_analysis.get("conversation_topics", []),
            "actual_conversation_length": input_data.get("actual_conversation_length", 0),
            "story_generation_method": input_data.get("story_generation_method", "default")
        }
        
        # 아이 정보 (child_profile 우선 사용)
        child_profile = input_data.get("child_profile", {})
        child_info = {
            "name": child_profile.get("name") or input_data.get("child_name", "친구"),
            "age": child_profile.get("age") or input_data.get("target_age", 7),
            "interests": child_profile.get("interests") or input_data.get("interests", []),
            "learning_preferences": input_data.get("learning_preferences", [])
        }
        
        # 실제 대화 내용 기반 개인화
        personalization_data = {
            "conversation_summary": conversation_summary,
            "user_mentioned_topics": extracted_keywords[:5],  # 상위 5개 키워드
            "conversation_style": self._analyze_conversation_style(conversation_analysis),
            "suggested_story_elements": self._extract_story_elements_from_conversation(
                conversation_summary, extracted_keywords
            )
        }
        
        # 참고 스토리 포맷팅 (대화 키워드 기반으로 더 관련성 높게)
        formatted_references = []
        for story in reference_stories[:3]:  # 상위 3개만 사용
            formatted_references.append({
                "title": story.get("title", ""),
                "summary": story.get("content", "")[:300] + "...",
                "educational_theme": story.get("educational_theme", ""),
                "age_group": story.get("age_group", ""),
                "key_lessons": story.get("key_lessons", [])
            })
        
        return {
            "story_outline": json.dumps(story_outline, ensure_ascii=False, indent=2),
            "reference_stories": json.dumps(formatted_references, ensure_ascii=False, indent=2),
            "child_info": json.dumps(child_info, ensure_ascii=False, indent=2),
            "personalization_data": json.dumps(personalization_data, ensure_ascii=False, indent=2)
        }
    
    def _analyze_conversation_style(self, conversation_analysis: Dict[str, Any]) -> str:
        """대화 스타일 분석"""
        user_messages = conversation_analysis.get("user_messages", 0)
        total_words = conversation_analysis.get("total_words", 0)
        
        if user_messages == 0:
            return "간단한 대화"
        
        avg_words_per_message = total_words / max(user_messages, 1)
        
        if avg_words_per_message > 15:
            return "자세하고 표현력이 풍부한 대화"
        elif avg_words_per_message > 8:
            return "적당히 활발한 대화"
        else:
            return "간단하고 간결한 대화"
    
    def _extract_story_elements_from_conversation(self, conversation_summary: str, 
                                                extracted_keywords: List[str]) -> Dict[str, List[str]]:
        """대화 내용에서 스토리 요소 추출"""
        
        # 키워드 기반 카테고리 분류
        character_keywords = []
        setting_keywords = []
        object_keywords = []
        emotion_keywords = []
        
        # 간단한 키워드 분류 (실제로는 더 정교한 NLP 모델 사용 가능)
        character_words = ["친구", "엄마", "아빠", "동물", "공주", "왕자", "마법사", "요정", "강아지", "고양이"]
        setting_words = ["집", "학교", "숲", "바다", "산", "공원", "마을", "성", "하늘", "우주"]
        object_words = ["장난감", "책", "공", "자동차", "꽃", "나무", "음식", "케이크", "선물"]
        emotion_words = ["행복", "슬픔", "무서워", "신나", "재미", "즐거", "기쁨", "사랑"]
        
        for keyword in extracted_keywords:
            keyword_lower = keyword.lower()
            if any(char in keyword_lower for char in character_words):
                character_keywords.append(keyword)
            elif any(place in keyword_lower for place in setting_words):
                setting_keywords.append(keyword)
            elif any(obj in keyword_lower for obj in object_words):
                object_keywords.append(keyword)
            elif any(emotion in keyword_lower for emotion in emotion_words):
                emotion_keywords.append(keyword)
        
        return {
            "characters": character_keywords[:3],
            "settings": setting_keywords[:3], 
            "objects": object_keywords[:3],
            "emotions": emotion_keywords[:3]
        }
    
    def _parse_enhanced_story(self, generated_text: str) -> Dict[str, Any]:
        """Enhanced 스토리 파싱 (JSON 형식 지원 + 추론 과정 추출)"""
        try:
            # JSON 블록 추출 시도
            json_match = re.search(r'```json\s*(.*?)\s*```', generated_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                parsed_data = json.loads(json_str)
                logger.info("JSON 블록 파싱 성공. 데이터 구조를 검증합니다.")

                # 데이터 구조 검증 및 보강
                for chapter in parsed_data.get("chapters", []):
                    if "narration" not in chapter:
                        chapter["narration"] = chapter.get("chapter_content", "")
                    if "dialogues" not in chapter:
                        chapter["dialogues"] = []
                    if "chapter_content" in chapter:
                        del chapter["chapter_content"] # 오래된 필드 제거
                
                # 추론 과정이 포함되어 있는지 확인
                if "reasoning_process" not in parsed_data:
                    # 텍스트에서 추론 과정 추출 시도
                    reasoning_match = re.search(r'추론\s*과정[:\s]*(.*?)(?=\n\n|\n#|$)', generated_text, re.DOTALL | re.IGNORECASE)
                    if reasoning_match:
                        parsed_data["reasoning_process"] = reasoning_match.group(1).strip()
                
                return parsed_data
            else:
                # JSON 블록이 없으면 텍스트 파싱
                logger.warning("JSON 블록을 찾을 수 없음. 일반 텍스트 파싱으로 전환합니다.")
                return self._parse_text_story_enhanced(generated_text)
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 실패, Enhanced 텍스트 파싱으로 전환: {e}")
            return self._parse_text_story_enhanced(generated_text)
    
    def _parse_text_story_enhanced(self, text: str) -> Dict[str, Any]:
        """Enhanced 텍스트 스토리 파싱 (내레이션/대사 분리 기능 추가)"""
        try:
            title_match = re.search(r'제목[:\s]*(.*?)(?=\n|$)', text, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else "생성된 동화"
            
            chapters_text = re.split(r'챕터\s*\d+|장\s*\d+', text)[1:]
            if not chapters_text:
                chapters_text = [text]

            chapters = []
            for i, content in enumerate(chapters_text, 1):
                chapter_title_match = re.search(r'[:\s]*(.*?)(?=\n|$)', content)
                chapter_title = chapter_title_match.group(1).strip() if chapter_title_match else f"챕터 {i}"

                dialogues = []
                narration_parts = []
                
                # 대사 패턴: "화자: 대사" 또는 화자: "대사"
                dialogue_pattern = re.compile(r'^\s*([가-힣\w\s]+?)\s*:\s*["“](.+?)["”]', re.MULTILINE)
                
                last_pos = 0
                for match in dialogue_pattern.finditer(content):
                    # 대사 앞부분을 내레이션으로 추가
                    narration_parts.append(content[last_pos:match.start()].strip())
                    
                    # 대사 추가
                    dialogues.append({"speaker": match.group(1).strip(), "text": match.group(2).strip()})
                    last_pos = match.end()

                # 나머지 텍스트를 내레이션으로 추가
                narration_parts.append(content[last_pos:].strip())

                # 비어있지 않은 내레이션만 합치기
                full_narration = "\n".join(part for part in narration_parts if part)

                chapters.append({
                    "chapter_number": i,
                    "chapter_title": chapter_title,
                    "narration": full_narration,
                    "dialogues": dialogues,
                    "educational_point": self._extract_educational_point(content),
                    "interaction_question": self._extract_interaction_question(content)
                })

            reasoning_match = re.search(r'추론\s*과정[:\s]*(.*?)(?=\n\n|\n#|$)', text, re.DOTALL | re.IGNORECASE)
            reasoning_process = reasoning_match.group(1).strip() if reasoning_match else ""
            
            return {
                "title": title,
                "chapters": chapters if chapters else [{"chapter_number": 1, "chapter_title": "동화", "narration": text, "dialogues": []}],
                "reasoning_process": reasoning_process
            }
            
        except Exception as e:
            logger.error(f"Enhanced 텍스트 파싱 실패: {e}")
            return {
                "title": "생성된 동화",
                "chapters": [{"chapter_number": 1, "chapter_title": "동화", "narration": text, "dialogues": []}],
                "reasoning_process": ""
            }
    
    def _extract_educational_point(self, text: str) -> str:
        """교육적 포인트 추출"""
        educational_patterns = [
            r'교육[적인]*\s*[포인트|내용|가치][:\s]*(.*?)(?=\n|$)',
            r'배울\s*점[:\s]*(.*?)(?=\n|$)',
            r'교훈[:\s]*(.*?)(?=\n|$)'
        ]
        
        for pattern in educational_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_interaction_question(self, text: str) -> str:
        """상호작용 질문 추출"""
        question_patterns = [
            r'질문[:\s]*(.*?\?)',
            r'(.*?는\s*어떻게\s*생각해\?)',
            r'(.*?라면\s*어떻게\s*할까\?)'
        ]
        
        for pattern in question_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _update_performance_metrics(self, generation_time: float, success: bool, age_group: str):
        """성능 메트릭 업데이트 (Enhanced)"""
        if not self.enable_performance_tracking:
            return
            
        if success:
            self.performance_metrics["generation_times"].append(generation_time)
            
            # 연령대별 사용량 추적
            if age_group not in self.performance_metrics["age_group_usage"]:
                self.performance_metrics["age_group_usage"][age_group] = 0
            self.performance_metrics["age_group_usage"][age_group] += 1
            
            # 성공률 계산
            total_attempts = len(self.performance_metrics["generation_times"]) + self.performance_metrics["error_count"]
            self.performance_metrics["success_rate"] = len(self.performance_metrics["generation_times"]) / total_attempts
        else:
            self.performance_metrics["error_count"] += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 조회 (Enhanced)"""
        if not self.performance_metrics["generation_times"]:
            return self.performance_metrics
            
        times = self.performance_metrics["generation_times"]
        return {
            **self.performance_metrics,
            "avg_generation_time": sum(times) / len(times),
            "min_generation_time": min(times),
            "max_generation_time": max(times),
            "total_generations": len(times),
            "most_used_age_group": max(self.performance_metrics["age_group_usage"], 
                                     key=self.performance_metrics["age_group_usage"].get, 
                                     default="unknown") if self.performance_metrics["age_group_usage"] else "unknown"
        }
