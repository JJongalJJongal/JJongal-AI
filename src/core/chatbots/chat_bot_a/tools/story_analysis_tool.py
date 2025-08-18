"""
Story Analysis Tool for ChatBot A (쫑이/Jjongi)

Basic story analysis functionality for extracting and analyzing
story elements from children's conversations.
"""

from typing import Dict, Any, List, Optional
from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

class StoryAnalysisTool:
    """
    Basic story analysis tool for ChatBot A
    
    Features:
    - Story element validation
    - Content analysis
    - Theme detection
    """
    
    def __init__(self):
        """Initialize the story analysis tool"""
        logger.info("StoryAnalysisTool initialized")
        
    def analyze_story_elements(self, elements: Dict[str, List]) -> Dict[str, Any]:
        """
        Analyze collected story elements

        Args:
            elements (Dict[str, List]): Dictionary containing story elements

        Returns:
            Dict[str, Any]: results
        """
        
        try:
            analysis = {
                "completeness_score": self._calculate_completeness(elements),
                "theme_suggestions": self._suggest_themes(elements),
                "missing_elements": self._identify_missing_elements(elements),
                "quality_score": self._assess_quality(elements)
            }
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing story elements: {e}")
            return {
                "completeness_score": 0.0,
                "theme_suggestions": [],
                "missing_elements": ["character", "setting", "problem", "resolution"],
                "quality_score": 0.0
            }
    
    def _calculate_completeness(self, elements: Dict[str, List]) -> float:
        """Calculate story completeness percentage"""
        
        total_required = 4 # character, setting, problem, resolution
        completed = 0
        
        if len(elements.get("character", [])) >= 1:
            completed += 1
        if len(elements.get("setting", [])) >= 1:
            completed += 1
        if len(elements.get("problem", [])) >= 1:
            completed += 1
        if len(elements.get("resolution", [])) >= 1:
            completed += 1
            
        return completed / total_required
    
    def _suggest_themes(self, elements: Dict[str, List]) -> List[str]:
        """Suggest story themes based on elements"""
        themes = []
        
        # Combine all element content for analysis
        all_content = " ".join([
            " ".join([elem.get("content", "") if isinstance(elem, dict) else str(elem)
                      for elem in elem_list])
            for elem_list in elements.values()
        ]).lower()
        
        # Basic theme detection
        if any(word in all_content for word in ["마법", "요정", "마술"]):
            themes.append("fantasy_magic")
        if any(word in all_content for word in ["동물", "숲", "자연"]):
            themes.append("nature_animals")
        if any(word in all_content for word in ["모험", "여행", "탐험"]):
            themes.append("adventure")
        if any(word in all_content for word in ["친구", "우정", "도움"]):
            themes.append("friendship")    
            
        return themes if themes else ["general_story"]
    
    def _identify_missing_elements(self, elements: Dict[str, List]) -> List[str]:
        """Identify missing story elements"""
        missing = []
        
        if len(elements.get("character", [])) < 1:
            missing.append("character")
        if len(elements.get("setting", [])) < 1:
            missing.append("setting")
        if len(elements.get("problem", [])) < 1:
            missing.append("problem")
        if len(elements.get("resolution", [])) < 1:
            missing.append("resolution")
            
        return missing
    
    def _assess_quality(self, elements: Dict[str, List]) -> float:
        """Assess the quality of story elements"""
        # Basic quality assessment based on completeness and diversity
        completeness = self._calculate_completeness(elements)
        
        # Additional quality factors
        diversity_bonus = 0.0
        if len(elements.get("character", [])) >= 2:
            diversity_bonus += 0.1
            
        total_elements = sum(len(elem_list) for elem_list in elements.values())
        richness_bonus = min(total_elements * 0.05, 0.2)
        
        quality_score = min(completeness + diversity_bonus + richness_bonus, 1.0)
        return quality_score
    