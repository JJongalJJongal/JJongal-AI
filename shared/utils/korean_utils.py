"""
한국어 처리 관련 유틸리티 모듈
"""
from typing import Dict


def has_final_consonant(word: str) -> bool:
    """
    한글 문자의 마지막 음절이 받침을 갖는지 확인하는 메서드
    
    Args:
        word (str): 확인할 한글 문자
        
    Returns:
        bool: 받침이 있으면 True, 없으면 False
    """
    if not word:
        return False
        
    # 마지막 문자 추출
    last_char = word[-1]
    
    # 한글이 아닌 경우 기본값 반환
    if not ord('가') <= ord(last_char) <= ord('힣'):
        return False
    
    # 실제 받침이 없는 자음들을 명시적으로 확인
    # 한글 유니코드에서 종성(받침) 계산
    char_code = ord(last_char) - ord('가')
    jong_code = char_code % 28  # 종성(받침) 코드
    
    # 종성 코드가 0이면 받침이 없음
    # 하지만 일부 특수한 경우를 추가로 확인
    if jong_code == 0:
        return False
    
    # 예외적으로 받침이 없다고 처리해야 할 문자들 (실제로는 거의 없음)
    # 대부분의 경우 유니코드 계산이 정확함
    no_final_exceptions = []  # '영'은 실제로 받침이 있음
    if last_char in no_final_exceptions:
        return False
        
    return True


def get_josa(word: str, josa_type: str) -> str:
    """
    단어에 맞는 조사를 반환하는 메서드
    
    Args:
        word (str): 조사를 붙일 단어
        josa_type (str): 조사 유형 ('은/는', '이/가', '을/를', '와/과', '으로/로', '아/야')
        
    Returns:
        str: 선택된 조사
    """
    has_final = has_final_consonant(word)
    
    josa_map: Dict[str, str] = {
        '은/는': '은' if has_final else '는',
        '이/가': '이' if has_final else '가',
        '을/를': '을' if has_final else '를',
        '와/과': '과' if has_final else '와',
        '으로/로': '으로' if has_final else '로',
        '아/야': '아' if has_final else '야'
    }
    
    return josa_map.get(josa_type, '')


def format_with_josa(word: str, josa_type: str) -> str:
    """
    단어에 조사를 붙여 반환하는 메서드
    
    Args:
        word (str): 조사를 붙일 단어
        josa_type (str): 조사 유형
        
    Returns:
        str: 조사가 붙은 단어
    """
    josa = get_josa(word, josa_type)
    return f"{word}{josa}" 