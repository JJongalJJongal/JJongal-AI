"""
ChatBot A (Jjongi)

This is the main chatbot for the Jjongi project.
Langchain + Langsmith 
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import time

from src.shared.utils.logging import get_module_logger
from src.shared.utils.age_group_utils import AgeGroupManager

from langchain_core.messages import AIMessage, HumanMessage

from src.shared.utils.korean import format_with_josa, process_template_with_particles

from .chains.conversation_chain import ConversationChain
from .chains.rag_chain import ModernRAGChain
from .memory.conversation_memory import ConversationMemoryManager
from .monitoring.langsmith_config import setup_langsmith_tracing
from .tools.story_analysis_tool import StoryAnalysisTool
from .voice_options import JjongiVoiceManager, JjongiVoiceType

logger = get_module_logger(__name__)


class ChatBotA:
    """
    ChatBot A (Jjongi) - story collection chatbot
    
    Features:
    - story collection for children having fun with the chatbot
    - conversation (4-7 age, 8-9 age)
    - real-time voice chat with multiple voice options
    - 3 different voice personalities (cheerful, gentle, playful)
    - Langsmith monitoring
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.8,
        enable_monitoring: bool = True   
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.enable_monitoring = enable_monitoring
        
        # Age group manager
        self.age_group_manager = AgeGroupManager()
        
        # Voice options manager
        self.voice_manager = JjongiVoiceManager()
        
        # LangChain components initialize
        self.memory_manager = ConversationMemoryManager()
        self.conversation_chain = ConversationChain(
            model_name=model_name,
            temperature=temperature,
            memory_manager=self.memory_manager
        )
        
        # Modern RAG chain (2025 best practices) - ChatBot A uses Summary DB
        import os
        
        # ChatBot A (쫑이) uses Summary DB for quick story element retrieval
        summary_db_path = os.getenv("SUMMARY_DB_PATH") or os.getenv("CHATBOT_A_DB_PATH")
        if not summary_db_path:
            # Use 3-Tier structure: Summary DB for ChatBot A
            base_path = os.getenv("VECTOR_DB_PATH") or os.getenv("CHROMA_DB_PATH") or "data/chroma_db"
            summary_db_path = os.path.join(base_path, "summary")
            logger.info(f"ChatBot A (쫑이): Summary DB 경로 설정 - {summary_db_path}")
        
        # Check if path exists and create if in development
        if not os.path.exists(summary_db_path):
            if "data/chroma_db" in summary_db_path:
                os.makedirs(summary_db_path, exist_ok=True)
                logger.info(f"Created Summary DB directory for ChatBot A: {summary_db_path}")
            else:
                logger.warning(f"Summary DB path doesn't exist: {summary_db_path}")
                summary_db_path = None
        
        vector_db_path = summary_db_path
                
        self.rag_chain = ModernRAGChain(
            model_name=model_name,
            temperature=temperature,
            vector_db_path=vector_db_path
        )
        
        self.story_tool = StoryAnalysisTool()
        
        # LangSmith monitoring setup
        self.tracer = None
        if enable_monitoring:
            self.tracer = setup_langsmith_tracing("ChatBot-A-Jjongi")
        
        
        # session manage
        self.active_sessions = {}
        
        logger.info("ChatBot A (Jjongi) initialized with LangChain + LangSmith")
        
    async def health_check(self) -> bool:
        """
        Health check for ChatBot A
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Check if core components are available
            checks = [
                self.memory_manager is not None,
                self.conversation_chain is not None,
                self.rag_chain is not None,
                self.story_tool is not None
            ]
            
            if not all(checks):
                logger.warning("ChatBot A health check failed: missing core components")
                return False
            
            # Check if LLM is responsive (development mode - skip if no API key)
            if hasattr(self.conversation_chain, 'llm'):
                try:
                    # Check if OpenAI API key is available
                    import os
                    openai_key = os.getenv("OPENAI_API_KEY")
                    if not openai_key or openai_key.startswith("sk-proj-REPLACE"):
                        logger.info("ChatBot A health check: 개발 모드 (API 키 없음) - LLM 테스트 건너뛰기")
                        return True
                    
                    # Quick test with minimal token usage only if API key is available
                    test_result = await self.conversation_chain.llm.ainvoke("테스트")
                    if not test_result:
                        logger.warning("ChatBot A health check failed: LLM not responsive")
                        return False
                except Exception as e:
                    logger.info(f"ChatBot A health check: 개발 모드에서 LLM 오류 무시 - {e}")
                    return True
            
            # Check database connection
            try:
                if hasattr(self.memory_manager, '_get_optimized_connection'):
                    conn = self.memory_manager._get_optimized_connection()
                    conn.close()
            except Exception as e:
                logger.warning(f"ChatBot A health check failed: DB error - {e}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"ChatBot A health check error: {e}")
            return False
        
    async def initialize_chat(
        self,
        child_name: str,
        child_age: int,
        child_interests: List[str] = [],
        voice_sample_path: str = ""
    ) -> str:
        """
        Conversation initialization
        
        Args:
            child_name: str
            child_age: int
            child_interests: List[str]
            voice_sample_path: str
            
        Returns:
            str: Session ID
        """
        
        try:
            # session create
            session_id = self.memory_manager.create_session(
                child_name=child_name,
                child_age=child_age,
                child_interests=child_interests,
            )
            
            # LangSmith session start
            if self.tracer:
                self.tracer.start_conversation_session(
                    child_name=child_name,
                    age=child_age,
                    interests=child_interests or []
                )
            
            # Get age group configuration
            age_group = self.age_group_manager.classify_age_group(child_age)
            age_group_config = self.age_group_manager.get_settings(age_group)
            
            # sessin state init
            self.active_sessions[session_id] = {
                "child_name": child_name,
                "child_age": child_age,
                "child_interests": child_interests or [],
                "age_group_config": age_group_config,
                "story_elements": {
                    "character": [],
                    "setting": [],
                    "problem": [],
                    "resolution": [],
                },
                "conversation_stage": "greeting", # greeting -> collection -> completion
                "turn_count": 0,
                "voice_sample_path": voice_sample_path,
                "start_time": datetime.now()
            }
            
            # first hello
            greeting = await self._generate_greeting(child_name, child_age, child_interests)
            
            # AI Message add to memory
            ai_message = AIMessage(content=greeting)
            self.memory_manager.add_message(session_id, ai_message)
            
            logger.info(f"Chat session initialized for {child_name} (age {child_age}) : {session_id}")
            return session_id
        
        except Exception as e:
            logger.error(f"Failed to initialize chat session: {e}")
            raise
    
    async def get_response(
        self, 
        user_input: str,
        session_id: Optional[str] = None,
        audio_data: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Jiongi's response generation for user input
        
        Args:
            user_input(str): User input(text)
            session_id(str): Conversation session ID
            audio_data(bytes): Audio data if available
            
        Returns:
            Dict[str, Any]: Response data
                - response(str) : Jiongi's response
                - story_elements_found(str): Found story elements
                - conversation_stage(str): Current conversation stage
                - completion_percentage(float): Estimated completion percentage
        """
        
        # If no session_id provided, use the first available session (for backward compatibility)
        if session_id is None:
            if not self.active_sessions:
                raise ValueError("No active sessions found")
            session_id = list(self.active_sessions.keys())[0]
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        start_time = time.time()
        
        try:
            # user message add to memory
            human_message = HumanMessage(content=user_input)
            self.memory_manager.add_message(session_id, human_message)
            
            session["turn_count"] += 1
            
            # story element collection
            story_elements_found = await self._extract_story_elements(
                user_input, session_id
            )
            
            # session add story elements
            await self._update_story_elements(session_id, story_elements_found)
            
            # update conversation stage
            self._update_conversation_stage(session_id)
            
            # get response 
            response = await self.conversation_chain.generate_response(
                user_input=user_input,
                session_id=session_id,
                enhance_response=True
            )
            
            # AI Message add to memory
            ai_message = AIMessage(content=response)
            self.memory_manager.add_message(session_id, ai_message)
            
            # completion percentage 
            completion_percentage = self._calculate_completion_percentage(session_id)
            
            # LamgSmith tracing
            processing_time = time.time() - start_time
            if self.tracer:
                self.tracer.trace_conversation_turn(
                    user_input=user_input,
                    ai_response=response,
                    story_elements_found=story_elements_found,
                    intent=self._detect_intent(user_input),
                    processing_time=processing_time,
                    turn_number=session["turn_count"]
                )
            
            results = {
                "response": response,
                "story_elements_found": story_elements_found,
                "conversation_stage": session["conversation_stage"],
                "completion_percentage": completion_percentage,
                "session_id": session_id,
                "turn_count": session["turn_count"]
            }
        
            logger.debug(f"Generated response for session {session_id} : {len(response)} chars")
            return results        
        except Exception as e:
            logger.error(f"Error generating response for session {session_id}: {e}")
            
            # 에러 시 기본 응답
            fallback_response = f"앗, {session['child_name']}아! 다시 말해줄래?"
            return {
                "response": fallback_response,
                "story_elements_found": [],
                "conversation_stage": session["conversation_stage"],
                "completion_percentage": self._calculate_completion_percentage_safe(session_id),
                "session_id": session_id,
                "turn_count": session["turn_count"]
            }
    
    async def get_story_outline_for_chatbot_b(self, session_id: str) -> Dict[str, Any]:
        """
        Get story outline for ChatBot B
        
        Args:
            session_id(str): Conversation session ID
            
        Returns:
            Dict[str, Any]: Story outline
        """
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        try:
            # conversation summary
            conversation_summary = self.conversation_chain.create_conversation_summary(session_id)
            
            # story elements summary
            story_elements = session["story_elements"]
            story_outline = {
                "session_id": session_id,
                "child_profile": {
                    "name": session["child_name"],
                    "age": session["child_age"],
                    "interests": session["child_interests"]
                },
                "story_elements": {
                    "main_characters": story_elements["character"],
                    "settings": story_elements["setting"],
                    "central_problems": story_elements["problem"],
                    "proposed_resolutions": story_elements["resolution"]
                },
                "conversation_summary": conversation_summary,
                "story_theme": await self._generate_story_theme(session_id),
                "target_age_group": "4-7세" if session["child_age"] <= 7 else "8-9세",
                "estimated_story_length": "short", # short, medium, long
                "special_requirements": await self._analyze_special_requirements(session_id),
                "collection_metadata": {
                    "total_turns": session["turn_count"],
                    "collection_duration_minutes": (
                        datetime.now() - session["start_time"]
                    ).total_seconds() / 60,
                    "completion_percentage": self._calculate_completion_percentage(session_id)
                }
            }
            
            # LangSmith tracing
            if self.tracer:
                self.tracer.trace_story_progress(
                    collected_elements=story_outline["story_elements"],
                    completion_percentage=story_outline["collection_metadata"]["completion_percentage"],
                    current_stage="handoff_to_chatbot_b"
                )
                
            logger.info(f"Story outline prepared for ChatBot B : {session_id}")
            return story_outline
        except Exception as e:
            logger.error(f"Error creating story outline for session {session_id}: {e}")
            raise
    
    
    async def end_conversation(self, session_id: str) -> Dict[str, Any]:
        """
        Conversation session end 

        Args:
            session_id(str): Conversation session ID
            
        Returns:
            Dict[str, Any]: End conversation result
        """
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        try:
            # ChatBot B story line create
            story_outline = await self.get_story_outline_for_chatbot_b(session_id)
            
            # session end at memory manager
            session_summary = self.memory_manager.close_session(session_id)
            
            # LangSmith session end
            if self.tracer:
                self.tracer.end_conversation_session(story_outline)
            
            completion_status = "completed" if self._calculate_completion_percentage(session_id) >= 0.80 else "partial"
            session_data = self.active_sessions.pop(session_id)
            
            final_summary = {
                "session_id": session_id,
                "child_name": session_data["child_name"],
                "total_turns": session_data["turn_count"],
                "story_outline": story_outline,
                "session_duration": (
                    datetime.now() - session_data["start_time"]
                ).total_seconds() / 60,
                "completion_status": completion_status
            }
            
            logger.info(f"Conversation session ended: {session_id}") 
            return final_summary
        
        except Exception as e:
            logger.error(f"Error ending conversation for session {session_id}: {e}")
            raise
        
    # Inner helper functions
    async def _generate_greeting(self, child_name: str, child_age: int, interests: Optional[List[str]] = None) -> str:
        """ first greeting for chatbot A """
        
        import random
        from src.shared.configs.prompts import load_chatbot_a_prompts
        
        try:
            # load prompts
            prompts = load_chatbot_a_prompts()
            templates = prompts.get("greeting_templates", [])
            
            if not templates:
                return f"안녕 {child_name}아! 난 쫑이야! 오늘 재미있는 이야기 만들어보자!"
            
            greeting_template = random.choice(templates)
            
            # Use the centralized template processing with particle adjustment
            greeting = process_template_with_particles(
                greeting_template,
                {
                    "name": child_name,
                    "interests": ", ".join(interests) if interests else "여러 가지"
                }
            )
            
            logger.debug(f"Generated greeting for {child_name} : {greeting[:50]}...")
            return greeting

        except Exception as e:
            logger.error(f"Error generating greeting: {e}")
            # 에러 시 기본 인사말
            return f"안녕 {format_with_josa(child_name, '아/야')}! 난 쫑이야! 재미있는 이야기 만들어보자!"
    
    async def _extract_story_elements(self, user_input: str, session_id: str) -> List[Dict[str, Any]]:
        """사용자 입력에서 이야기 요소 추출"""
        session = self.active_sessions[session_id]
        
        # check current missing elements
        current_elements = session["story_elements"]
        needed_elements = []
        
        if len(current_elements["character"]) < 2:
            needed_elements.append("character")
        if len(current_elements["setting"]) < 1:
            needed_elements.append("setting")
        if len(current_elements["problem"]) < 1:
            needed_elements.append("problem")
        if len(current_elements["resolution"]) < 1:
            needed_elements.append("resolution")
        
        # extract elements with highest priority
        if needed_elements:
            target_element = needed_elements[0]
            result = await self.conversation_chain.collect_story_elements(
                user_input=user_input,
                target_element=target_element,
                child_age=session["child_age"]
            )
            return result.get("elements_found", [])
        
        return []
    
    async def _update_story_elements(self, session_id: str, found_elements: List[Dict[str, Any]]):
        """update found story elements to session"""
        session = self.active_sessions[session_id]
        
        for element in found_elements:
            element_type = element.get("type")
            content = element.get("content")
            confidence = element.get("confidence", 0.8)
            
            if element_type and content and element_type in session["story_elements"]:
                session["story_elements"][element_type].append({
                    "content": content,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                })
                
                # 데이터베이스에도 저장
                self.memory_manager.add_story_element(
                    session_id=session_id,
                    element_type=element_type,
                    content=content,
                    confidence_score=confidence
                )
    
    def _update_conversation_stage(self, session_id: str):
        """update conversation stage"""
        completion = self._calculate_completion_percentage(session_id)
        session = self.active_sessions[session_id]
        
        if completion < 30:
            session["conversation_stage"] = "greeting"
        elif completion < 80:
            session["conversation_stage"] = "collection"
        else:
            session["conversation_stage"] = "completion"
    
    def _calculate_completion_percentage(self, session_id: str) -> float:
        """이야기 수집 완성도 계산"""
        session = self.active_sessions[session_id]
        elements = session["story_elements"]
        
        # 필수 요소별 가중치
        weights = {
            "character": 30,   # 캐릭터 2개 이상
            "setting": 20,     # 배경 1개 이상
            "problem": 25,     # 문제 1개 이상
            "resolution": 25   # 해결책 1개 이상
        }
        
        total_score = 0
        for element_type, weight in weights.items():
            element_list = elements.get(element_type, [])
            
            if element_type == "character":
                # 캐릭터는 2개 이상 필요
                score = min(len(element_list) / 2.0, 1.0) * weight
            else:
                # 다른 요소들은 1개 이상
                score = min(len(element_list), 1.0) * weight
            
            total_score += score
        
        return round(total_score, 1)
    
    def _calculate_completion_percentage_safe(self, session_id: str) -> float:
        """안전한 이야기 수집 완성도 계산 (세션이 존재하지 않을 경우 0 반환)"""
        if session_id not in self.active_sessions:
            return 0.0
        return self._calculate_completion_percentage(session_id)
    
    def _detect_intent(self, user_input: str) -> str:
        """사용자 입력의 의도 파악"""
        # 기본적인 의도 분류
        if any(word in user_input for word in ["누구", "이름", "캐릭터"]):
            return "character_inquiry"
        elif any(word in user_input for word in ["어디", "장소", "배경"]):
            return "setting_inquiry"
        elif any(word in user_input for word in ["문제", "곤란", "어려움"]):
            return "problem_inquiry"
        elif any(word in user_input for word in ["해결", "방법", "도움"]):
            return "resolution_inquiry"
        else:
            return "general_conversation"
    
    async def _generate_story_theme(self, session_id: str) -> str:
        """generate story theme based on collected elements"""
        session = self.active_sessions[session_id]
        elements = session["story_elements"]
        
        # keyword based theme classification
        all_content = " ".join([
            " ".join([elem["content"] for elem in elem_list])
            for elem_list in elements.values()
        ])
        
        if any(word in all_content for word in ["마법", "요정", "마술"]):
            return "fantasy_magic"
        elif any(word in all_content for word in ["동물", "숲", "자연"]):
            return "nature_animals"
        elif any(word in all_content for word in ["모험", "여행", "탐험"]):
            return "adventure"
        elif any(word in all_content for word in ["친구", "우정", "도움"]):
            return "friendship"
        else:
            return "general_story"
    
    async def _analyze_special_requirements(self, session_id: str) -> List[str]:
        """special requirements analysis (age group manager)"""
        session = self.active_sessions[session_id]
        requirements = []
        
        # requirements from age group manager
        age_config = session["age_group_config"]
        requirements.extend(age_config.educational_focus)
        
        # requirements from child interests
        interests = session["child_interests"]
        if "동물" in interests:
            requirements.append("include_animals")
        if "공주" in interests or "프린세스" in interests:
            requirements.append("include_princess_theme")
        
        return requirements
    
    # === Voice Option Methods ===
    
    def get_voice_options(self) -> List[Dict[str, Any]]:
        """Get available voice options for 쫑이"""
        return self.voice_manager.get_voice_options()
    
    def set_voice_option(self, voice_type: str) -> bool:
        """Change 쫑이's voice option"""
        success = self.voice_manager.set_voice_option(voice_type)
        if success:
            logger.info(f"Voice option changed to: {voice_type}")
        return success
    
    def get_current_voice_config(self) -> Dict[str, Any]:
        """Get current voice configuration for TTS generation"""
        return self.voice_manager.get_current_voice_config()
    
    def update_voice_ids(self, voice_id_mapping: Dict[str, str]) -> None:
        """Update ElevenLabs Voice IDs (for admin use)"""
        self.voice_manager.update_voice_ids(voice_id_mapping)
        logger.info("Voice IDs updated for ChatBot A")
    
    def get_voice_config_for_conversation(self, session_id: Optional[str] = None, preference: Optional[str] = None) -> Dict[str, Any]:
        """Get optimized voice configuration for current conversation context"""
        
        # If session exists, consider age and context
        if session_id and session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            child_age = session["child_age"]
            conversation_stage = session["conversation_stage"]
            
            # Age-based voice optimization
            if child_age <= 5:
                # Younger children prefer gentler voices
                if not preference:
                    preference = "gentle"
            elif child_age >= 8:
                # Older children can handle more playful interaction
                if not preference:
                    preference = "playful"
        
        return self.voice_manager.get_voice_config_for_conversation(preference)
    
    def validate_voice_configuration(self) -> Dict[str, Any]:
        """Validate current voice configuration status"""
        validation_results = self.voice_manager.validate_voice_ids()
        
        # Check if any voices need configuration
        needs_setup = any(not is_valid for is_valid in validation_results.values())
        
        return {
            "validation_results": validation_results,
            "needs_setup": needs_setup,
            "current_voice": self.voice_manager.current_voice.value,
            "total_voices": len(validation_results)
        }
    
    