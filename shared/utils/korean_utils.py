"""
한국어 처리 관련 유틸리티 모듈
"""
import re
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


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
    
    # 특별한 받침 처리 (ㄹ 받침은 을/를에서 예외 처리됨)
    # 대부분의 경우 유니코드 계산이 정확함
    no_final_exceptions = []
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
    
    # Special handling for ㄹ final consonant with 을/를
    last_char = word[-1] if word else ''
    if josa_type == '을/를' and last_char and ord('가') <= ord(last_char) <= ord('힣'):
        char_code = ord(last_char) - ord('가')
        jong_code = char_code % 28
        # ㄹ (8) 받침이면 '를' 사용
        if jong_code == 8:  # ㄹ
            return '를'
    
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


def adjust_korean_particles_in_text(text: str, name: str) -> str:
    """
    텍스트 내의 한국어 조사 패턴을 아이의 이름에 맞게 조정하는 함수
    
    Args:
        text (str): 조정할 텍스트
        name (str): 아이의 이름 (조사 조정의 기준)
        
    Returns:
        str: 조사가 조정된 텍스트
    """
    try:
        adjusted_text = text
        
        # Common particle patterns that need adjustment for names
        particle_patterns = [
            ("{name}아/야", "아/야"),
            ("{name}이/가", "이/가"),
            ("{name}을/를", "을/를"),
            ("{name}은/는", "은/는"),
            ("{name}와/과", "와/과"),
            ("{name}으로/로", "으로/로")
        ]
        
        for pattern, josa_type in particle_patterns:
            if pattern in adjusted_text:
                adjusted_text = adjusted_text.replace(pattern, format_with_josa(name, josa_type))
        
        return adjusted_text
        
    except Exception as e:
        logger.warning(f"Failed to adjust Korean particles: {e}")
        return text  # Return original text if adjustment fails


def process_template_with_particles(template: str, replacements: Dict[str, str]) -> str:
    """
    템플릿 문자열에서 변수를 치환하고 한국어 조사를 적절히 조정하는 함수
    
    Args:
        template (str): 처리할 템플릿 문자열
        replacements (Dict[str, str]): 치환할 변수들 {"variable_name": "value"}
        
    Returns:
        str: 처리된 문자열
    """
    try:
        result = template
        
        # Process each variable with potential particle patterns
        for var_name, value in replacements.items():
            # Handle particle patterns: {var_name}조사/조사
            particle_patterns = [
                (f"{{{var_name}}}아/야", "아/야"),
                (f"{{{var_name}}}이/가", "이/가"),
                (f"{{{var_name}}}을/를", "을/를"),
                (f"{{{var_name}}}은/는", "은/는"),
                (f"{{{var_name}}}와/과", "와/과"),
                (f"{{{var_name}}}으로/로", "으로/로")
            ]
            
            for pattern, josa_type in particle_patterns:
                if pattern in result:
                    result = result.replace(pattern, format_with_josa(value, josa_type))
            
            # Replace any remaining plain variable patterns
            result = result.replace(f"{{{var_name}}}", value)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing template with particles: {e}")
        return template  # Return original template if processing fails


def extract_particles_from_templates(templates: List[str]) -> Dict[str, List[str]]:
    """
    템플릿 리스트에서 한국어 조사 패턴을 추출하는 함수
    
    Args:
        templates (List[str]): 분석할 템플릿 문자열들
        
    Returns:
        Dict[str, List[str]]: 발견된 조사 패턴들 {"pattern": ["template1", "template2"]}
    """
    particle_pattern = re.compile(r'\{[^}]+\}[가-힣]+/[가-힣]+')
    patterns_found = {}
    
    for template in templates:
        matches = particle_pattern.findall(template)
        for match in matches:
            if match not in patterns_found:
                patterns_found[match] = []
            patterns_found[match].append(template)
    
    return patterns_found


def validate_korean_particles(text: str) -> List[str]:
    """
    텍스트에서 잘못된 한국어 조사 사용을 검증하는 함수
    
    Args:
        text (str): 검증할 텍스트
        
    Returns:
        List[str]: 발견된 문제점들의 리스트
    """
    issues = []
    
    # Check for unresolved particle patterns
    unresolved_patterns = re.findall(r'\{[^}]+\}[가-힣]+/[가-힣]+', text)
    if unresolved_patterns:
        issues.append(f"Unresolved particle patterns found: {unresolved_patterns}")
    
    # Check for common incorrect patterns
    incorrect_patterns = [
        (r'[가-힣]+이가', "Possible incorrect '이가' usage"),
        (r'[가-힣]+을를', "Possible incorrect '을를' usage"),
        (r'[가-힣]+은는', "Possible incorrect '은는' usage")
    ]
    
    for pattern, description in incorrect_patterns:
        if re.search(pattern, text):
            issues.append(description)
    
    return issues 