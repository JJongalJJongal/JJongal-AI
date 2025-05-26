"""
부기 (ChatBot A) 언어 처리기

한국어 특화 언어 처리, 문법 검사, 어휘 분석을 담당하는 프로세서
"""
from typing import Dict, List, Any, Optional, Tuple
import re

from .base_processor import BaseProcessor
from shared.utils.logging_utils import get_module_logger
from shared.utils.korean_utils import has_final_consonant, format_with_josa

logger = get_module_logger(__name__)

class LanguageProcessor(BaseProcessor):
    """
    한국어 특화 언어 처리를 담당하는 프로세서
    
    BaseProcessor를 상속받아 한국어 문법, 어휘, 표현 처리 기능을 제공
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        언어 프로세서 초기화
        
        Args:
            config: 언어 처리 설정
        """
        super().__init__(config)
        
        # 연령대별 어휘 사전
        self.age_vocabulary = {
            4: {
                "simple_words": ["엄마", "아빠", "집", "학교", "친구", "놀이", "음식"],
                "animals": ["강아지", "고양이", "토끼", "곰", "새"],
                "emotions": ["기쁘다", "슬프다", "화나다", "무섭다"],
                "actions": ["가다", "오다", "먹다", "자다", "놀다"]
            },
            5: {
                "simple_words": ["가족", "선생님", "교실", "공원", "장난감", "책"],
                "animals": ["사자", "코끼리", "기린", "원숭이", "물고기"],
                "emotions": ["행복하다", "걱정하다", "신나다", "부끄럽다"],
                "actions": ["달리다", "점프하다", "그리다", "노래하다", "춤추다"]
            },
            6: {
                "simple_words": ["모험", "여행", "꿈", "마법", "보물", "성"],
                "animals": ["용", "유니콘", "펭귄", "돌고래", "나비"],
                "emotions": ["용감하다", "호기심", "놀라다", "감동하다"],
                "actions": ["탐험하다", "발견하다", "도와주다", "구하다", "만들다"]
            },
            7: {
                "simple_words": ["우주", "로봇", "과학", "실험", "발명", "미래"],
                "animals": ["공룡", "외계인", "로봇동물", "상상의동물"],
                "emotions": ["자신감", "도전", "성취감", "실망", "희망"],
                "actions": ["연구하다", "실험하다", "창조하다", "해결하다", "협력하다"]
            },
            8: {
                "simple_words": ["환경", "자연", "지구", "우정", "팀워크", "리더십"],
                "animals": ["멸종동물", "바다동물", "정글동물", "극지동물"],
                "emotions": ["책임감", "공감", "배려", "인내", "감사"],
                "actions": ["보호하다", "지키다", "이해하다", "배우다", "가르치다"]
            },
            9: {
                "simple_words": ["역사", "문화", "전통", "예술", "음악", "스포츠"],
                "animals": ["신화동물", "전설동물", "고대동물"],
                "emotions": ["존경", "자부심", "겸손", "열정", "의지"],
                "actions": ["계획하다", "준비하다", "도전하다", "극복하다", "성장하다"]
            }
        }
        
        # 한국어 조사 패턴
        self.josa_patterns = {
            "이/가": ("이", "가"),
            "을/를": ("을", "를"),
            "은/는": ("은", "는"),
            "과/와": ("과", "와"),
            "아/야": ("아", "야"),
            "으로/로": ("으로", "로")
        }
        
    def initialize(self) -> bool:
        """
        프로세서 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            # 기본 설정 확인
            self.default_age = self.config.get('default_age', 5)
            self.enable_grammar_check = self.config.get('enable_grammar_check', True)
            
            self.is_initialized = True
            logger.info("LanguageProcessor 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"LanguageProcessor 초기화 실패: {e}")
            return False
    
    def process(self, input_data: Any) -> Any:
        """
        언어 처리
        
        Args:
            input_data: 처리할 언어 데이터
            
        Returns:
            Any: 처리된 언어 데이터
        """
        if not self.validate_input(input_data):
            return None
            
        if isinstance(input_data, dict):
            process_type = input_data.get('type')
            text = input_data.get('text', '')
            
            if process_type == 'vocabulary_check':
                age = input_data.get('age', self.default_age)
                return self.check_age_appropriate_vocabulary(text, age)
            elif process_type == 'josa_format':
                name = input_data.get('name', '')
                josa_type = input_data.get('josa_type', '이/가')
                return self.format_josa(name, josa_type)
            elif process_type == 'simplify':
                age = input_data.get('age', self.default_age)
                return self.simplify_for_age(text, age)
            elif process_type == 'extract_keywords':
                return self.extract_korean_keywords(text)
        
        return str(input_data)
    
    def get_age_appropriate_vocabulary(self, age: int) -> Dict[str, List[str]]:
        """
        연령대에 맞는 어휘 반환
        
        Args:
            age: 아이의 나이
            
        Returns:
            Dict[str, List[str]]: 연령대별 어휘 사전
        """
        # 해당 연령과 그 이하 연령의 어휘를 모두 포함
        vocabulary = {}
        
        for vocab_age in range(4, min(age + 1, 10)):
            if vocab_age in self.age_vocabulary:
                age_vocab = self.age_vocabulary[vocab_age]
                for category, words in age_vocab.items():
                    if category not in vocabulary:
                        vocabulary[category] = []
                    vocabulary[category].extend(words)
        
        # 중복 제거
        for category in vocabulary:
            vocabulary[category] = list(set(vocabulary[category]))
        
        return vocabulary
    
    def check_age_appropriate_vocabulary(self, text: str, age: int) -> Dict[str, Any]:
        """
        텍스트가 연령대에 적합한 어휘를 사용하는지 확인
        
        Args:
            text: 확인할 텍스트
            age: 아이의 나이
            
        Returns:
            Dict[str, Any]: 어휘 적합성 분석 결과
        """
        appropriate_vocab = self.get_age_appropriate_vocabulary(age)
        all_appropriate_words = []
        
        for words in appropriate_vocab.values():
            all_appropriate_words.extend(words)
        
        # 텍스트에서 단어 추출 (간단한 공백 기준)
        words_in_text = text.split()
        
        appropriate_count = 0
        inappropriate_words = []
        
        for word in words_in_text:
            # 구두점 제거
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in all_appropriate_words:
                appropriate_count += 1
            elif len(clean_word) > 1:  # 한 글자 단어는 제외
                inappropriate_words.append(clean_word)
        
        total_words = len([w for w in words_in_text if len(re.sub(r'[^\w]', '', w)) > 1])
        appropriateness_score = appropriate_count / total_words if total_words > 0 else 0
        
        return {
            "age": age,
            "total_words": total_words,
            "appropriate_words": appropriate_count,
            "inappropriate_words": inappropriate_words,
            "appropriateness_score": appropriateness_score,
            "is_appropriate": appropriateness_score >= 0.3  # 30% 이상이면 적합
        }
    
    def format_josa(self, name: str, josa_type: str) -> str:
        """
        이름에 맞는 조사 형태 반환
        
        Args:
            name: 이름
            josa_type: 조사 타입 (예: "이/가", "을/를")
            
        Returns:
            str: 이름 + 적절한 조사
        """
        if not name or josa_type not in self.josa_patterns:
            return name
        
        has_final = has_final_consonant(name)
        final_josa, no_final_josa = self.josa_patterns[josa_type]
        
        josa = final_josa if has_final else no_final_josa
        return f"{name}{josa}"
    
    def simplify_for_age(self, text: str, age: int) -> str:
        """
        연령대에 맞게 텍스트 단순화
        
        Args:
            text: 단순화할 텍스트
            age: 아이의 나이
            
        Returns:
            str: 단순화된 텍스트
        """
        # 연령대별 복잡한 표현을 간단한 표현으로 변환
        simplification_rules = {
            4: {
                "모험을 떠나다": "놀러 가다",
                "탐험하다": "찾아보다",
                "발견하다": "찾다",
                "해결하다": "고치다"
            },
            5: {
                "협력하다": "함께 하다",
                "도전하다": "해보다",
                "성취하다": "해내다",
                "극복하다": "이기다"
            },
            6: {
                "창조하다": "만들다",
                "발명하다": "새로 만들다",
                "연구하다": "알아보다",
                "실험하다": "해보다"
            }
        }
        
        # 해당 연령 이하의 모든 규칙 적용
        simplified_text = text
        for rule_age in range(4, min(age + 1, 7)):
            if rule_age in simplification_rules:
                for complex_expr, simple_expr in simplification_rules[rule_age].items():
                    simplified_text = simplified_text.replace(complex_expr, simple_expr)
        
        return simplified_text
    
    def extract_korean_keywords(self, text: str) -> List[str]:
        """
        한국어 텍스트에서 키워드 추출
        
        Args:
            text: 키워드를 추출할 텍스트
            
        Returns:
            List[str]: 추출된 키워드 목록
        """
        # 간단한 키워드 추출 (명사 위주)
        # 실제로는 더 정교한 형태소 분석이 필요하지만, 기본적인 패턴 매칭 사용
        
        # 일반적인 한국어 명사 패턴 (2-4글자)
        noun_pattern = r'[가-힣]{2,4}(?=[을를이가은는과와에서]|$|\s)'
        keywords = re.findall(noun_pattern, text)
        
        # 중복 제거 및 필터링
        keywords = list(set(keywords))
        
        # 불용어 제거
        stop_words = ['그것', '이것', '저것', '여기', '거기', '저기', '때문', '경우', '정도']
        keywords = [kw for kw in keywords if kw not in stop_words and len(kw) >= 2]
        
        return keywords
    
    def analyze_sentence_complexity(self, text: str) -> Dict[str, Any]:
        """
        문장 복잡도 분석
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            Dict[str, Any]: 복잡도 분석 결과
        """
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        total_sentences = len(sentences)
        total_chars = len(text.replace(' ', ''))
        avg_sentence_length = total_chars / total_sentences if total_sentences > 0 else 0
        
        # 복잡한 문법 구조 확인
        complex_patterns = [
            r'[가-힣]+(?:하면서|하지만|그러나|따라서|그래서)',  # 접속어
            r'[가-힣]+(?:ㄴ다면|다면|라면)',  # 가정법
            r'[가-힣]+(?:기 때문에|때문에)',  # 이유
            r'[가-힣]+(?:에 의해|에 따라)'  # 수동/의존
        ]
        
        complex_count = 0
        for pattern in complex_patterns:
            complex_count += len(re.findall(pattern, text))
        
        complexity_score = (avg_sentence_length / 20) + (complex_count / total_sentences) if total_sentences > 0 else 0
        
        return {
            "total_sentences": total_sentences,
            "avg_sentence_length": avg_sentence_length,
            "complex_structures": complex_count,
            "complexity_score": min(complexity_score, 1.0),  # 0-1 범위로 정규화
            "is_simple": complexity_score < 0.3
        }
    
    def suggest_age_appropriate_alternatives(self, text: str, target_age: int) -> List[str]:
        """
        연령대에 맞는 대안 표현 제안
        
        Args:
            text: 원본 텍스트
            target_age: 목표 연령
            
        Returns:
            List[str]: 대안 표현 목록
        """
        alternatives = []
        
        # 복잡도 분석
        complexity = self.analyze_sentence_complexity(text)
        
        if not complexity["is_simple"]:
            # 문장을 더 간단하게 나누기
            simplified = self.simplify_for_age(text, target_age)
            if simplified != text:
                alternatives.append(simplified)
            
            # 짧은 문장으로 나누기
            if complexity["avg_sentence_length"] > 15:
                sentences = text.split('.')
                short_sentences = []
                for sentence in sentences:
                    if len(sentence.strip()) > 15:
                        # 간단한 문장 분할 (접속어 기준)
                        parts = re.split(r'(그리고|그래서|하지만|그러나)', sentence)
                        short_sentences.extend([p.strip() for p in parts if p.strip() and p not in ['그리고', '그래서', '하지만', '그러나']])
                    else:
                        short_sentences.append(sentence.strip())
                
                if len(short_sentences) > 1:
                    alternatives.append('. '.join(short_sentences))
        
        return alternatives 