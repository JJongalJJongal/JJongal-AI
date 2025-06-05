"""
텍스트 생성기 (Enhanced for Advanced Prompt System)

LangChain + ChromaDB RAG System과 개선된 프롬프트 엔지니어링을 활용한 한국 동화 생성
- 구조화된 프롬프트 접근법 (Role → Objective → Instructions → Reasoning → Output → Examples)
- 연령별 특화 프롬프트 (4-7세, 8-9세)
- 체인 오브 소트 추론 통합
- 성능 추적 및 최적화
"""
from shared.utils.logging_utils import get_module_logger
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
from chatbot.data.vector_db.core import VectorDB
from chatbot.data.vector_db.query import get_similar_stories

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
                 vector_db_path: str = None,
                 collection_name: str = "fairy_tales",
                 prompts_file_path: str = "chatbot/data/prompts/chatbot_b_prompts.json",
                 max_retries: int = 3,
                 model_name: str = "gpt-4o",
                 temperature: float = 0.7,
                 enable_performance_tracking: bool = True,
                 model_kwargs: Dict[str, Any] = None):
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
        self.vector_db_path = vector_db_path # ChromaDB 경로
        self.collection_name = collection_name # ChromaDB 컬렉션 이름
        self.prompts_file_path = prompts_file_path # 개선된 프롬프트 파일 경로
        self.model_name = model_name # 사용할 LLM 모델명
        self.temperature = temperature # 생성 온도
        self.enable_performance_tracking = enable_performance_tracking # 성능 추적 활성화
        self.model_kwargs = model_kwargs or {} # LLM 모델 키워드 인수 (ex: {"max_tokens": 1000})
        
        # Enhanced LangChain 구성
        self.vector_store = None
        self.retriever = None
        self.text_chains = {}  # 연령별 체인
        self.prompts = None
        
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
        if not self.vector_db_path:
            logger.warning("ChromaDB 경로가 설정되지 않음. RAG 기능 비활성화")
            return
        
        try:
            self.vector_store = VectorDB(persist_directory=self.vector_db_path)
            
            # 컬렉션 존재 확인
            try:
                collection = self.vector_store.get_collection(self.collection_name)
                logger.info(f"ChromaDB 컬렉션 '{self.collection_name}' 연결 완료")
            except Exception as e:
                logger.warning(f"컬렉션 '{self.collection_name}' 연결 실패: {e}")
        
        except Exception as e:
            logger.error(f"ChromaDB 초기화 실패: {e}")
            raise
    
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
            # Enhanced 프롬프트 구조에서 연령별 프롬프트 가져오기
            enhanced_prompts = self.prompts.get("enhanced_story_generation", {})
            age_config = enhanced_prompts.get(age_group, {})
            structured_prompt = age_config.get("structured_prompt", {})
            
            # 구조화된 프롬프트 구성
            role = structured_prompt.get("role", "전문 동화 작가로서")
            objective = structured_prompt.get("objective", "몰입감 있는 동화를 제작해주세요.")
            instructions = structured_prompt.get("instructions", [])
            reasoning_steps = structured_prompt.get("reasoning_steps", [])
            
            # 프롬프트 템플릿 생성
            system_template = self._build_structured_prompt(
                role=role,
                objective=objective,
                instructions=instructions,
                reasoning_steps=reasoning_steps,
                age_group=age_group
            )
            
            prompt_template = ChatPromptTemplate.from_template(system_template)
            
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
        cot_templates = self.prompts.get("chain_of_thought_templates", {})
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
  \"title\": \"동화 제목\",
  \"chapters\": [
    {{
      \"chapter_number\": 1,
      \"chapter_title\": \"챕터 제목\",
      \"chapter_content\": \"챕터 내용\",
      \"educational_point\": \"교육적 포인트\",
      \"interaction_question\": \"상호작용 질문\"
    }}
  ],
  \"reasoning_process\": \"추론 과정 설명\"
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
            # 연령대 결정
            target_age = input_data.get("target_age", input_data.get("age_group", 7))
            age_group_key = self._determine_age_group(target_age)
            
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
            generated_text = await chain.ainvoke(prompt_data)
            
            # 5. Enhanced 파싱
            story_data = self._parse_enhanced_story(generated_text)
            
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
            self._update_performance_metrics(0, False, age_group_key if 'age_group_key' in locals() else "unknown")
            logger.error(f"Enhanced 텍스트 생성 실패: {e}")
            raise

    async def _retrieve_similar_stories(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ChromaDB에서 유사 스토리 검색 (Enhanced)"""
        if not self.vector_store:
            logger.warning("VectorDB가 초기화되지 않음. RAG 검색 생략")
            return []
        
        try:
            # 쿼리 구성 (더 정교한 검색)
            theme = input_data.get("theme", "")
            educational_value = input_data.get("educational_value", "")
            interests = input_data.get("interests", [])
            age_group = input_data.get("target_age", input_data.get("age_group", 7))
            
            query_text = f"{theme} {educational_value} {' '.join(interests)}"
            
            # 연령대에 따른 필터링
            metadata_filter = {
                "age_min": {"$lte": age_group},
                "age_max": {"$gte": age_group}
            } if isinstance(age_group, int) else {}
            
            # 벡터 검색 수행 (get_similar_stories 사용)
            # get_similar_stories는 동기 함수이므로 asyncio.to_thread 사용
            results = await asyncio.to_thread(
                get_similar_stories,
                vector_db=self.vector_store,
                query_text=query_text,
                n_results=5,
                metadata_filter=metadata_filter,
                collection_name=self.collection_name,
                doc_type="summary" # 필요시 다른 doc_type 지정 가능
            )
            
            logger.info(f"RAG 검색 완료: {len(results)}개의 유사 스토리 반환")
            return results
            
        except Exception as e:
            logger.warning(f"Enhanced RAG 검색 실패: {e}. 빈 참고 스토리 반환")
            return []

    def _prepare_enhanced_prompt_data(self, input_data: Dict[str, Any], 
                                    reference_stories: List[Dict[str, Any]], 
                                    age_group: str) -> Dict[str, Any]:
        """Enhanced 프롬프트 데이터 준비"""
        
        # 기본 스토리 정보
        story_outline = {
            "theme": input_data.get("theme", ""),
            "plot_summary": input_data.get("plot_summary", ""),
            "educational_value": input_data.get("educational_value", ""),
            "target_age": input_data.get("target_age", 7),
            "setting": input_data.get("setting", ""),
            "characters": input_data.get("characters", [])
        }
        
        # 아이 정보
        child_info = {
            "name": input_data.get("child_name", "친구"),
            "age": input_data.get("target_age", 7),
            "interests": input_data.get("interests", []),
            "learning_preferences": input_data.get("learning_preferences", [])
        }
        
        # 참고 스토리 포맷팅 (더 정교한 구조)
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
            "child_info": json.dumps(child_info, ensure_ascii=False, indent=2)
        }
    
    def _parse_enhanced_story(self, generated_text: str) -> Dict[str, Any]:
        """Enhanced 스토리 파싱 (JSON 형식 지원 + 추론 과정 추출)"""
        try:
            # JSON 블록 추출 시도
            json_match = re.search(r'```json\s*(.*?)\s*```', generated_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                parsed_data = json.loads(json_str)
                
                # 추론 과정이 포함되어 있는지 확인
                if "reasoning_process" not in parsed_data:
                    # 텍스트에서 추론 과정 추출 시도
                    reasoning_match = re.search(r'추론\s*과정[:\s]*(.*?)(?=\n\n|\n#|$)', generated_text, re.DOTALL | re.IGNORECASE)
                    if reasoning_match:
                        parsed_data["reasoning_process"] = reasoning_match.group(1).strip()
                
                return parsed_data
            else:
                # JSON 블록이 없으면 텍스트 파싱
                return self._parse_text_story_enhanced(generated_text)
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 실패, Enhanced 텍스트 파싱으로 전환: {e}")
            return self._parse_text_story_enhanced(generated_text)
    
    def _parse_text_story_enhanced(self, text: str) -> Dict[str, Any]:
        """Enhanced 텍스트 스토리 파싱"""
        try:
            # 기본 제목 추출
            title_match = re.search(r'제목[:\s]*(.*?)(?=\n|$)', text, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else "생성된 동화"
            
            # 챕터 추출 (더 정교한 패턴)
            chapter_patterns = [
                r'챕터\s*(\d+)[:\s]*(.*?)(?=챕터\s*\d+|$)',
                r'장\s*(\d+)[:\s]*(.*?)(?=장\s*\d+|$)',
                r'(\d+)\.\s*(.*?)(?=\d+\.|$)'
            ]
            
            chapters = []
            for pattern in chapter_patterns:
                matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
                if matches:
                    for match in matches:
                        chapter_num = int(match.group(1))
                        chapter_content = match.group(2).strip()
                        
                        # 챕터 제목과 내용 분리
                        lines = chapter_content.split('\n', 1)
                        chapter_title = lines[0].strip()
                        chapter_text = lines[1].strip() if len(lines) > 1 else chapter_content
                        
                        chapters.append({
                            "chapter_number": chapter_num,
                            "chapter_title": chapter_title,
                            "chapter_content": chapter_text,
                            "educational_point": self._extract_educational_point(chapter_text),
                            "interaction_question": self._extract_interaction_question(chapter_text)
                        })
                    break
            
            # 추론 과정 추출
            reasoning_match = re.search(r'추론\s*과정[:\s]*(.*?)(?=\n\n|\n#|$)', text, re.DOTALL | re.IGNORECASE)
            reasoning_process = reasoning_match.group(1).strip() if reasoning_match else ""
            
            return {
                "title": title,
                "chapters": chapters if chapters else [{"chapter_number": 1, "chapter_title": "동화", "chapter_content": text}],
                "reasoning_process": reasoning_process
            }
            
        except Exception as e:
            logger.error(f"Enhanced 텍스트 파싱 실패: {e}")
            return {
                "title": "생성된 동화",
                "chapters": [{"chapter_number": 1, "chapter_title": "동화", "chapter_content": text}],
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
            r'(.*?는\s*어떻게\s*생각하나요\?)',
            r'(.*?라면\s*어떻게\s*할까요\?)'
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
    
    async def health_check(self) -> Dict[str, bool]:
        """Enhanced 상태 확인"""
        health_status = {
            "enhanced_prompts_loaded": bool(self.prompts),
            "vector_db_connected": bool(self.vector_store),
            "age_4_7_chain_ready": "age_4_7" in self.text_chains,
            "age_8_9_chain_ready": "age_8_9" in self.text_chains,
            "performance_tracking": self.enable_performance_tracking
        }
        
        # 전체 상태
        health_status["overall_healthy"] = all(health_status.values())
        
        return health_status 