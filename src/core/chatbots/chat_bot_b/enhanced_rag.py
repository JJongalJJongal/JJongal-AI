"""
Enhanced RAG System for Advanced Story Generation

Advanced Retrieval-Augmented Generation with:
- Multi-source knowledge retrieval
- Cultural context-aware search
- Semantic similarity scoring
- Knowledge graph integration
- Adaptive filtering and ranking
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import asyncio
from pathlib import Path
import numpy as np
from src.shared.utils.logging import get_module_logger
from shared.utils.langchain_manager import langchain_manager

logger = get_module_logger(__name__)

@dataclass
class RAGSource:
    """RAG knowledge source configuration"""
    name: str
    vector_db_path: str
    collection_name: str
    weight: float
    cultural_tags: List[str]
    content_type: str  # "fairy_tales", "educational", "cultural", "modern"
    age_range: Tuple[int, int]
    language: str

@dataclass
class RAGResult:
    """RAG retrieval result"""
    content: str
    source: str
    similarity_score: float
    cultural_relevance: float
    age_appropriateness: float
    metadata: Dict[str, Any]

class EnhancedRAGManager:
    """
    Enhanced RAG Manager for multi-source knowledge retrieval
    
    Features:
    - Multiple knowledge sources integration
    - Cultural context-aware retrieval
    - Age-specific content filtering
    - Semantic similarity ranking
    - Adaptive source weighting
    """
    
    def __init__(self):
        self.rag_sources = self._initialize_rag_sources()
        self.vector_stores = {}
        self.cultural_embeddings = {}
        self.performance_metrics = {
            "total_queries": 0,
            "successful_retrievals": 0,
            "average_retrieval_time": 0,
            "source_usage": {},
            "cultural_accuracy": 0
        }
        
    def _initialize_rag_sources(self) -> List[RAGSource]:
        """Initialize multiple RAG knowledge sources"""
        sources = []
        
        # Traditional Korean Fairy Tales
        sources.append(RAGSource(
            name="korean_traditional",
            vector_db_path="/app/chatbot/data/vector_db/korean_traditional",
            collection_name="traditional_tales",
            weight=1.0,
            cultural_tags=["korean", "traditional", "folklore"],
            content_type="fairy_tales",
            age_range=(4, 9),
            language="ko"
        ))
        
        # Modern Educational Stories
        sources.append(RAGSource(
            name="educational_modern",
            vector_db_path="/app/chatbot/data/vector_db/educational",
            collection_name="modern_educational",
            weight=0.8,
            cultural_tags=["modern", "educational", "values"],
            content_type="educational",
            age_range=(4, 9),
            language="ko"
        ))
        
        # Cultural Values Database
        sources.append(RAGSource(
            name="cultural_values",
            vector_db_path="/app/chatbot/data/vector_db/cultural",
            collection_name="korean_values",
            weight=0.6,
            cultural_tags=["values", "morals", "social"],
            content_type="cultural",
            age_range=(4, 9),
            language="ko"
        ))
        
        # International Stories (for comparison and diversity)
        sources.append(RAGSource(
            name="international",
            vector_db_path="/app/chatbot/data/vector_db/international",
            collection_name="world_tales",
            weight=0.4,
            cultural_tags=["international", "diverse", "universal"],
            content_type="fairy_tales",
            age_range=(4, 9),
            language="multi"
        ))
        
        # Age-specific Content Database
        sources.append(RAGSource(
            name="age_specific_4_7",
            vector_db_path="/app/chatbot/data/vector_db/age_specific",
            collection_name="stories_4_7",
            weight=0.9,
            cultural_tags=["age_appropriate", "simple"],
            content_type="fairy_tales",
            age_range=(4, 7),
            language="ko"
        ))
        
        sources.append(RAGSource(
            name="age_specific_8_9",
            vector_db_path="/app/chatbot/data/vector_db/age_specific",
            collection_name="stories_8_9",
            weight=0.9,
            cultural_tags=["age_appropriate", "complex"],
            content_type="fairy_tales",
            age_range=(8, 9),
            language="ko"
        ))
        
        return sources
    
    async def initialize_vector_stores(self):
        """Initialize vector stores for all RAG sources"""
        try:
            from chatbot.data.vector_db.core import VectorDB
            
            for source in self.rag_sources:
                if Path(source.vector_db_path).exists():
                    try:
                        self.vector_stores[source.name] = VectorDB(
                            persist_directory=source.vector_db_path,
                            embedding_model='nlpai-lab/KURE-v1',
                            use_hybrid_mode=True
                        )
                        
                        # Test collection existence
                        collection = self.vector_stores[source.name].get_collection(source.collection_name)
                        logger.info(f"Initialized RAG source: {source.name} ({collection.count()} documents)")
                        
                    except Exception as e:
                        logger.warning(f"Failed to initialize RAG source {source.name}: {e}")
                else:
                    logger.warning(f"RAG source path not found: {source.vector_db_path}")
                    
            logger.info(f"Enhanced RAG Manager initialized with {len(self.vector_stores)} sources")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced RAG Manager: {e}")
            raise
    
    async def enhanced_retrieval(self,
                                query: str,
                                target_age: int,
                                cultural_context: Dict[str, Any],
                                language: str = "ko",
                                max_results: int = 10) -> List[RAGResult]:
        """
        Enhanced retrieval with multi-source integration
        
        Args:
            query: Search query
            target_age: Target age for content filtering
            cultural_context: Cultural context for relevance scoring
            language: Target language
            max_results: Maximum number of results
            
        Returns:
            List of ranked RAG results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Track performance
            self.performance_metrics["total_queries"] += 1
            
            # Get relevant sources for the query
            relevant_sources = self._select_relevant_sources(
                target_age, cultural_context, language
            )
            
            # Perform parallel retrieval from multiple sources
            retrieval_tasks = []
            for source in relevant_sources:
                if source.name in self.vector_stores:
                    task = self._retrieve_from_source(
                        source, query, target_age, cultural_context, max_results // len(relevant_sources) + 1
                    )
                    retrieval_tasks.append(task)
            
            # Execute parallel retrievals
            source_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
            
            # Combine and rank results
            all_results = []
            for i, results in enumerate(source_results):
                if isinstance(results, Exception):
                    logger.warning(f"Retrieval failed for source {relevant_sources[i].name}: {results}")
                    continue
                all_results.extend(results)
            
            # Advanced ranking and filtering
            ranked_results = await self._advanced_ranking(
                all_results, query, target_age, cultural_context
            )
            
            # Update performance metrics
            retrieval_time = asyncio.get_event_loop().time() - start_time
            self._update_performance_metrics(retrieval_time, len(ranked_results))
            
            return ranked_results[:max_results]
            
        except Exception as e:
            logger.error(f"Enhanced retrieval failed: {e}")
            return []
    
    def _select_relevant_sources(self,
                                target_age: int,
                                cultural_context: Dict[str, Any],
                                language: str) -> List[RAGSource]:
        """Select relevant RAG sources based on context"""
        relevant_sources = []
        
        for source in self.rag_sources:
            # Age range filtering
            if not (source.age_range[0] <= target_age <= source.age_range[1]):
                continue
                
            # Language filtering
            if source.language not in [language, "multi"]:
                continue
                
            # Cultural relevance scoring
            cultural_relevance = self._calculate_cultural_relevance(
                source, cultural_context
            )
            
            if cultural_relevance > 0.3:  # Minimum relevance threshold
                relevant_sources.append(source)
        
        # Sort by weight and relevance
        relevant_sources.sort(key=lambda s: s.weight, reverse=True)
        
        logger.info(f"Selected {len(relevant_sources)} relevant RAG sources for age {target_age}")
        return relevant_sources
    
    async def _retrieve_from_source(self,
                                   source: RAGSource,
                                   query: str,
                                   target_age: int,
                                   cultural_context: Dict[str, Any],
                                   max_results: int) -> List[RAGResult]:
        """Retrieve from a specific RAG source"""
        try:
            from chatbot.data.vector_db.query import get_similar_stories
            
            vector_store = self.vector_stores[source.name]
            
            # Create age-specific metadata filter
            age_group = "4-7세" if 4 <= target_age <= 7 else "8-9세"
            metadata_filter = {"age_group": age_group}
            
            # Perform vector search
            raw_results = await asyncio.to_thread(
                get_similar_stories,
                vector_db=vector_store,
                query_text=query,
                n_results=max_results,
                metadata_filter=metadata_filter,
                collection_name=source.collection_name
            )
            
            # Convert to RAGResult objects with enhanced scoring
            rag_results = []
            for result in raw_results:
                cultural_relevance = self._calculate_cultural_relevance_for_content(
                    result, cultural_context
                )
                
                age_appropriateness = self._calculate_age_appropriateness(
                    result, target_age
                )
                
                rag_result = RAGResult(
                    content=result.get("content", ""),
                    source=source.name,
                    similarity_score=result.get("similarity_score", 0.0),
                    cultural_relevance=cultural_relevance,
                    age_appropriateness=age_appropriateness,
                    metadata={
                        **result,
                        "source_weight": source.weight,
                        "source_cultural_tags": source.cultural_tags
                    }
                )
                
                rag_results.append(rag_result)
            
            # Update source usage metrics
            if source.name not in self.performance_metrics["source_usage"]:
                self.performance_metrics["source_usage"][source.name] = 0
            self.performance_metrics["source_usage"][source.name] += len(rag_results)
            
            return rag_results
            
        except Exception as e:
            logger.error(f"Failed to retrieve from source {source.name}: {e}")
            return []
    
    def _calculate_cultural_relevance(self,
                                    source: RAGSource,
                                    cultural_context: Dict[str, Any]) -> float:
        """Calculate cultural relevance score for a source"""
        relevance_score = 0.0
        
        # Check cultural tag overlap
        context_values = cultural_context.get("traditional_values", [])
        context_themes = cultural_context.get("moral_themes", [])
        
        cultural_keywords = context_values + context_themes
        cultural_keywords = [kw.lower() for kw in cultural_keywords]
        
        # Score based on tag overlap
        for tag in source.cultural_tags:
            for keyword in cultural_keywords:
                if tag.lower() in keyword or keyword in tag.lower():
                    relevance_score += 0.2
        
        # Bonus for exact cultural matches
        if "korean" in source.cultural_tags and cultural_context.get("language") == "ko":
            relevance_score += 0.3
            
        if "traditional" in source.cultural_tags and "전통" in str(cultural_context):
            relevance_score += 0.2
        
        return min(relevance_score, 1.0)
    
    def _calculate_cultural_relevance_for_content(self,
                                                content: Dict[str, Any],
                                                cultural_context: Dict[str, Any]) -> float:
        """Calculate cultural relevance for specific content"""
        relevance_score = 0.0
        
        content_text = content.get("content", "").lower()
        title = content.get("title", "").lower()
        
        # Check for cultural value alignment
        context_values = cultural_context.get("traditional_values", [])
        for value in context_values:
            if value.lower() in content_text or value.lower() in title:
                relevance_score += 0.15
        
        # Check for moral theme alignment
        context_themes = cultural_context.get("moral_themes", [])
        for theme in context_themes:
            if theme.lower() in content_text or theme.lower() in title:
                relevance_score += 0.15
        
        # Check for character archetype alignment
        context_archetypes = cultural_context.get("character_archetypes", [])
        for archetype in context_archetypes:
            if archetype.lower() in content_text or archetype.lower() in title:
                relevance_score += 0.1
        
        return min(relevance_score, 1.0)
    
    def _calculate_age_appropriateness(self,
                                     content: Dict[str, Any],
                                     target_age: int) -> float:
        """Calculate age appropriateness score"""
        appropriateness_score = 0.5  # Base score
        
        content_age_group = content.get("age_group", "")
        
        if target_age <= 7:
            if "4-7" in content_age_group:
                appropriateness_score = 1.0
            elif "8-9" in content_age_group:
                appropriateness_score = 0.3
        else:
            if "8-9" in content_age_group:
                appropriateness_score = 1.0
            elif "4-7" in content_age_group:
                appropriateness_score = 0.7
        
        # Additional scoring based on content complexity
        content_text = content.get("content", "")
        if len(content_text) > 1000 and target_age <= 6:
            appropriateness_score *= 0.8  # Penalty for long content for young children
        
        return appropriateness_score
    
    async def _advanced_ranking(self,
                              results: List[RAGResult],
                              query: str,
                              target_age: int,
                              cultural_context: Dict[str, Any]) -> List[RAGResult]:
        """Advanced ranking algorithm for RAG results"""
        
        # Calculate composite scores
        for result in results:
            # Weighted composite score
            composite_score = (
                result.similarity_score * 0.3 +
                result.cultural_relevance * 0.4 +
                result.age_appropriateness * 0.2 +
                result.metadata.get("source_weight", 0.5) * 0.1
            )
            
            # Bonus for diverse sources
            if len(set(r.source for r in results)) > 1:
                source_diversity_bonus = 0.05
                composite_score += source_diversity_bonus
            
            result.metadata["composite_score"] = composite_score
        
        # Sort by composite score
        ranked_results = sorted(
            results,
            key=lambda r: r.metadata["composite_score"],
            reverse=True
        )
        
        # Apply diversity filtering to avoid too many results from the same source
        diverse_results = self._apply_diversity_filtering(ranked_results)
        
        return diverse_results
    
    def _apply_diversity_filtering(self, results: List[RAGResult]) -> List[RAGResult]:
        """Apply diversity filtering to ensure source variety"""
        diverse_results = []
        source_counts = {}
        max_per_source = 3  # Maximum results per source
        
        for result in results:
            source = result.source
            if source not in source_counts:
                source_counts[source] = 0
            
            if source_counts[source] < max_per_source:
                diverse_results.append(result)
                source_counts[source] += 1
        
        return diverse_results
    
    def _update_performance_metrics(self, retrieval_time: float, result_count: int):
        """Update performance metrics"""
        if result_count > 0:
            self.performance_metrics["successful_retrievals"] += 1
        
        # Update average retrieval time
        total_queries = self.performance_metrics["total_queries"]
        current_avg = self.performance_metrics["average_retrieval_time"]
        
        self.performance_metrics["average_retrieval_time"] = (
            (current_avg * (total_queries - 1) + retrieval_time) / total_queries
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced RAG performance metrics"""
        return {
            "enhanced_rag_metrics": self.performance_metrics,
            "initialized_sources": len(self.vector_stores),
            "total_sources": len(self.rag_sources),
            "source_health": {
                source_name: bool(source_name in self.vector_stores)
                for source_name in [s.name for s in self.rag_sources]
            }
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Health check for RAG system"""
        health_status = {
            "enhanced_rag_ready": True,
            "sources_initialized": len(self.vector_stores) > 0,
            "all_sources_healthy": len(self.vector_stores) == len(self.rag_sources)
        }
        
        # Test each source
        for source in self.rag_sources:
            source_healthy = source.name in self.vector_stores
            health_status[f"source_{source.name}"] = source_healthy
        
        health_status["overall_healthy"] = all(health_status.values())
        return health_status

# Global enhanced RAG manager instance
enhanced_rag_manager = EnhancedRAGManager()