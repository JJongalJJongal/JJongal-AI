"""
OpenAI API 관련 유틸리티 모듈
"""
import os
import logging
from typing import Dict, Optional, Tuple, Any
from openai import OpenAI
from ..configs.app_config import get_env_vars

logger = logging.getLogger(__name__)


def initialize_client() -> OpenAI:
    """
    OpenAI 클라이언트 초기화
    
    Returns:
        OpenAI: 초기화된 OpenAI 클라이언트
        
    Raises:
        ValueError: API 키가 없을 경우
    """
    env_vars = get_env_vars()
    api_key = env_vars.get("openai_api_key")
    
    if not api_key:
        logger.warning("OPENAI_API_KEY가 설정되지 않았습니다.")
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
    
    try:
        client = OpenAI(api_key=api_key)
        logger.info("OpenAI 클라이언트 초기화 성공")
        return client
    except Exception as e:
        logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
        raise


def track_token_usage(current_usage: Dict[str, int], new_usage: Dict[str, int], token_limit: int) -> Tuple[Dict[str, int], bool]:
    """
    토큰 사용량 추적 및 제한 체크
    
    Args:
        current_usage (Dict[str, int]): 현재까지의 토큰 사용량
        new_usage (Dict[str, int]): 새로운, 추가된 토큰 사용량
        token_limit (int): 전체 토큰 제한
        
    Returns:
        Tuple[Dict[str, int], bool]: 업데이트된 토큰 사용량과 제한 내 여부(True/False)
    """
    # 토큰 사용량 업데이트
    updated_usage = {
        "total_prompt": current_usage.get("total_prompt", 0) + new_usage.get("prompt_tokens", 0),
        "total_completion": current_usage.get("total_completion", 0) + new_usage.get("completion_tokens", 0),
        "total": 0
    }
    
    # 전체 합계 계산
    updated_usage["total"] = updated_usage["total_prompt"] + updated_usage["total_completion"]
    
    # 한도 체크
    within_limit = updated_usage["total"] < token_limit
    
    # 85% 이상 사용 시 경고
    if updated_usage["total"] >= token_limit * 0.85:
        logger.warning(f"토큰 사용량 85% 이상 도달: {updated_usage['total']}/{token_limit}")
    
    return updated_usage, within_limit


def generate_chat_completion(
    client: OpenAI, 
    messages: list, 
    model: str = "gpt-4o", 
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    OpenAI API를 통해 채팅 완료 응답 생성
    
    Args:
        client (OpenAI): OpenAI 클라이언트
        messages (list): 메시지 목록
        model (str): 사용할 모델
        temperature (float): 온도 (창의성 조절)
        max_tokens (int, optional): 최대 토큰 수
        
    Returns:
        Tuple[Optional[str], Optional[Dict[str, Any]]]: 
            생성된 텍스트와 사용량 정보 (오류 시 None, None)
    """
    try:
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        response = client.chat.completions.create(**params)
        
        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        return content, usage
    except Exception as e:
        logger.error(f"Chat completion 생성 중 오류 발생: {e}")
        return None, None 