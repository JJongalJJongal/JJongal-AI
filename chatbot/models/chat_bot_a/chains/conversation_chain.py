"""
Enhanced Conversation Chain for ChatBot A (쫑이/Jjongi)

LangChain-based conversation management with advanced prompting,
age-appropriate responses, and story collection capabilities.
"""

from typing import Dict, Any, List, Optional
import asyncio

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from ..memory.conversation_memory import ConversationMemoryManager
from ..monitoring.langsmith_config import trace_chatbot_method
from shared.utils.logging_utils import get_module_logger
from shared.configs.prompts_config import load_chatbot_a_prompts

logger = get_module_logger(__name__)


class ConversationChain:
    """
    Advanced conversation chain with LangChain integration.
    
    Features:
    - Age-appropriate conversation management
    - Story element collection
    - Context-aware responses
    - LangSmith monitoring integration
    - RAG-enhanced responses
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.8,
        memory_manager: ConversationMemoryManager = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.memory_manager = memory_manager
        
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=temperature,
            model_kwargs={"top_p": 0.9}
        )
        
        self.prompts = load_chatbot_a_prompts()
        self._setup_chains()
    
    def _setup_chains(self):
        """Setup LangChain conversation chains"""
        
        # Main conversation chain
        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="conversation_history"),
            ("human", "{user_input}")
        ])
        
        self.conversation_chain = (
            RunnablePassthrough.assign(
                formatted_context=RunnableLambda(self._format_conversation_context)
            )
            | self.conversation_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Story collection chain
        self.story_collection_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_story_collection_prompt()),
            ("human", """
아래 아이의 답변에서 동화 요소를 추출해주세요.

아이의 말: {user_input}
찾는 요소: {target_element}
아이 나이: {child_age}세

JSON 형식으로 답변해주세요:
{{
    "elements_found": [
        {{
            "type": "character|setting|problem|resolution",
            "content": "추출된 내용",
            "confidence": 0.0-1.0
        }}
    ],
    "continue_collection": true/false,
    "suggested_question": "다음 질문 예시"
}}
            """)
        ])
        
        self.story_collection_chain = (
            self.story_collection_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Response enhancement chain
        self.enhancement_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that enhances the response to be more engaging and interesting."),
            ("human", """
            Original Response: {original_response}
            Child Name: {child_name}
            Child Age: {child_age}
            Child Interests: {child_interests}

            Enhanced Response: {enhanced_response}
            """)
            ])
        
        self.enhancement_chain = (
            self.enhancement_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for conversation"""
        base_prompt = self.prompts.get("system_message_template", [])
        if isinstance(base_prompt, list):
            return "\n".join(base_prompt)
        return str(base_prompt)
    
    def _get_story_collection_prompt(self) -> str:
        """Get system prompt for story collection"""
        return """당신은 아이들로부터 동화 이야기 요소를 수집하는 전문가입니다.

주요 역할:
1. 아이의 말에서 다음 4가지 동화 요소를 찾아내기
   - character (등장인물): 주인공, 조연, 동물, 마법 존재 등
   - setting (배경): 장소, 시간, 환경, 분위기
   - problem (문제): 갈등, 위기상황, 문제, 도전
   - resolution (해결): 문제 해결 방법, 결말, 교훈

2. 아이의 연령대에 맞는 이해도 고려
3. 창의적인 상상력을 존중하며 격려
4. 자연스럽고 즐거운 대화 유도

응답 시 고려사항:
- 아이다운 순수한 상상력 존중
- 아이의 관심사와 흥미 파악
- 긍정적이고 따뜻한 피드백 제공
- 연령대에 적합한 어휘 사용"""
    
    def _format_conversation_context(self, inputs: Dict[str, Any]) -> str:
        """Format conversation context for the prompt"""
        session_id = inputs.get("session_id")
        if not session_id or not self.memory_manager:
            return ""
        
        try:
            context = self.memory_manager.get_conversation_context(session_id)
            
            formatted_parts = []
            
            # Child profile
            child_profile = context.get("child_profile", {})
            formatted_parts.append(f"아이 정보: {child_profile.get('name')} ({child_profile.get('age')}세)")
            
            if child_profile.get('interests'):
                formatted_parts.append(f"관심사: {', '.join(child_profile['interests'])}")
            
            # Story collection progress
            story_elements = self.memory_manager.get_story_elements(session_id)
            if story_elements:
                element_summary = {}
                for element in story_elements:
                    element_type = element["element_type"]
                    if element_type not in element_summary:
                        element_summary[element_type] = []
                    element_summary[element_type].append(element["content"])
                
                formatted_parts.append("수집된 이야기 요소:")
                for element_type, contents in element_summary.items():
                    formatted_parts.append(f"- {element_type}: {', '.join(contents)}")
            
            return "\n".join(formatted_parts)
            
        except Exception as e:
            logger.error(f"Error formatting conversation context: {e}")
            return ""
    
    @trace_chatbot_method("generate_response")
    async def generate_response(
        self,
        user_input: str,
        session_id: str,
        enhance_response: bool = True
    ) -> str:
        """Generate conversational response using LangChain"""
        try:
            # Get conversation context
            if self.memory_manager and session_id in self.memory_manager.active_sessions:
                context = self.memory_manager.get_conversation_context(session_id)
                conversation_history = context.get("recent_messages", [])
                child_profile = context.get("child_profile", {})
            else:
                conversation_history = []
                child_profile = {}
            
            # Convert to LangChain messages
            lc_messages = []
            for msg in conversation_history:
                if msg["type"] == "HumanMessage":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["type"] == "AIMessage":
                    lc_messages.append(AIMessage(content=msg["content"]))
            
            # Generate response
            response = await self.conversation_chain.ainvoke({
                "user_input": user_input,
                "session_id": session_id,
                "conversation_history": lc_messages
            })
            
            # Enhance response if requested
            if enhance_response and child_profile:
                response = await self._enhance_response(response, child_profile)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "앗, 미안해! 잠깐 생각이 복잡해졌어. 다시 말해줄 수 있을까?"
    
    @trace_chatbot_method("collect_story_elements")
    async def collect_story_elements(
        self,
        user_input: str,
        target_element: str,
        child_age: int
    ) -> Dict[str, Any]:
        """Extract story elements from user input"""
        try:
            response = await self.story_collection_chain.ainvoke({
                "user_input": user_input,
                "target_element": target_element,
                "child_age": child_age
            })
            
            # Parse JSON response
            import json
            try:
                result = json.loads(response)
                return result
            except json.JSONDecodeError:
                logger.warning("Failed to parse story collection JSON response")
                return {
                    "elements_found": [],
                    "continue_collection": True,
                    "suggested_question": "더 자세히 말해줄 수 있을까?"
                }
                
        except Exception as e:
            logger.error(f"Error collecting story elements: {e}")
            return {
                "elements_found": [],
                "continue_collection": True,
                "suggested_question": "정말 재미있는 이야기구나! 더 들려줄래?"
            }
    
    async def _enhance_response(self, response: str, child_profile: Dict[str, Any]) -> str:
        """Enhance response with child-specific personalization"""
        try:
            enhanced = await self.enhancement_chain.ainvoke({
                "original_response": response,
                "child_name": child_profile.get("name", "친구"),
                "child_age": child_profile.get("age", 5),
                "child_interests": ", ".join(child_profile.get("interests", []))
            })
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing response: {e}")
            return response  # Return original if enhancement fails
    
    @trace_chatbot_method("generate_story_guidance")
    async def generate_story_guidance(
        self,
        session_id: str,
        target_element: str,
        rag_context: str = ""
    ) -> str:
        """Generate guidance to help collect specific story elements"""
        
        guidance_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 아이들로부터 이야기 요소를 수집하는 도우미 전문가입니다.
            
목표 요소: {target_element}
아이가 {target_element}에 대해 이야기할 수 있도록 자연스럽게 유도하는 질문을 만들어주세요.

지침:
1. 아이의 상상력을 자극하는 재미있는 질문
2. 연령대에 맞는 쉽고 친근한 표현
3. 창의적 사고를 격려하는 분위기
4. 재미있고 흥미로운 접근 방식
5. 자연스러운 대화 유도
            """),
            ("human", """
찾고자 하는 요소: {target_element}
참고 정보: {rag_context}

{target_element}에 대한 이야기를 자연스럽게 유도하는 질문을 만들어주세요.
            """)
        ])
        
        guidance_chain = guidance_prompt | self.llm | StrOutputParser()
        
        try:
            guidance = await guidance_chain.ainvoke({
                "target_element": target_element,
                "rag_context": rag_context
            })
            
            return guidance
            
        except Exception as e:
            logger.error(f"Error generating story guidance: {e}")
            
            # Fallback guidance
            element_names = {
                "character": "등장인물",
                "setting": "배경", 
                "problem": "문제",
                "resolution": "해결방법"
            }
            
            element_korean = element_names.get(target_element, target_element)
            return f"네 이야기의 {element_korean}에 대해 더 말해줄래? 어떤 {element_korean}인지 궁금해!"
    
    def create_conversation_summary(self, session_id: str) -> str:
        """Create a summary of the conversation for handoff to ChatBot B"""
        if not self.memory_manager:
            return "대화 기록을 찾을 수 없어요."
        
        try:
            context = self.memory_manager.get_conversation_context(session_id)
            story_elements = self.memory_manager.get_story_elements(session_id)
            
            summary_parts = []
            
            # Child profile
            child_profile = context.get("child_profile", {})
            summary_parts.append(f"아이: {child_profile.get('name')} ({child_profile.get('age')}세)")
            
            # Story elements
            if story_elements:
                summary_parts.append("\n수집된 이야기 요소:")
                element_groups = {}
                for element in story_elements:
                    element_type = element["element_type"]
                    if element_type not in element_groups:
                        element_groups[element_type] = []
                    element_groups[element_type].append(element["content"])
                
                for element_type, contents in element_groups.items():
                    summary_parts.append(f"- {element_type}: {'; '.join(contents)}")
            
            # Memory summary
            if context.get("memory_summary"):
                summary_parts.append(f"\n대화 요약: {context['memory_summary']}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error creating conversation summary: {e}")
            return "대화 요약을 생성하는 중 오류가 발생했습니다."