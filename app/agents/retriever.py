# app/agents/enhanced_retriever.py
"""
Enhanced policy retriever with improved query building and relevance scoring.
Focuses on retrieving the top 3 most relevant policy documents.
"""

from typing import List, Dict, Set
import re
from collections import Counter

from app.agents.base import BaseAgent
from app.services.qdrant_client import search_policies
from app.models.schemas import RetrievalResult, PolicyDocument, ClassificationResult
from app.utils.exceptions import RetrievalError


class EnhancedPolicyRetriever(BaseAgent):
    """
    Enhanced retriever that builds better queries and provides detailed explanations
    for why specific policies were retrieved.
    """
    
    def __init__(self):
        super().__init__("enhanced_policy_retriever")
        self.hate_speech_keywords = self._load_hate_speech_keywords()
        self.policy_type_keywords = self._load_policy_type_keywords()
    
    def _load_hate_speech_keywords(self) -> Dict[str, List[str]]:
        """Load categorized keywords for hate speech detection."""
        return {
            "identity_targets": [
                "race", "religion", "gender", "sexuality", "disability", 
                "ethnicity", "nationality", "caste", "age"
            ],
            "harmful_actions": [
                "hate", "harassment", "violence", "discrimination", 
                "threat", "intimidation", "dehumanize", "attack"
            ],
            "content_types": [
                "slur", "epithet", "insult", "stereotype", "degrading",
                "offensive", "toxic", "abusive"
            ]
        }
    
    def _load_policy_type_keywords(self) -> Dict[str, List[str]]:
        """Load keywords that indicate different policy types."""
        return {
            "community_guidelines": ["community", "guidelines", "standards", "rules"],
            "legal_framework": ["legal", "law", "penal", "constitutional", "criminal"],
            "platform_policy": ["platform", "terms", "service", "prohibited", "ads"],
            "enforcement": ["enforcement", "violation", "ban", "removal", "suspension"]
        }
    
    async def _execute(self, original_text: str, classification: ClassificationResult) -> RetrievalResult:
        """
        Execute enhanced policy retrieval with improved query building.
        
        Args:
            original_text: The text to analyze for policy retrieval.
            classification: The classification result from the detection agent.
            
        Returns:
            RetrievalResult with top 3 relevant policies and explanations.
        """
        try:
            # Build enhanced query
            enhanced_query = self._build_enhanced_query(original_text, classification)
            
            # Search with higher limit to get more candidates
            raw_results = search_policies(enhanced_query, limit=10)
            
            if not raw_results:
                return self._create_empty_result(enhanced_query)
            
            # Process and rank results
            processed_results = self._process_and_rank_results(
                raw_results, original_text, classification
            )
            
            # Select top 3 with diversity
            top_policies = self._select_diverse_top_policies(processed_results)
            
            # Convert to PolicyDocument objects
            policy_documents = [
                self._create_policy_document(result, original_text, classification)
                for result in top_policies
            ]
            
            return RetrievalResult(
                policies=policy_documents,
                query_used=enhanced_query,
                total_candidates=len(raw_results)
            )
            
        except Exception as e:
            raise RetrievalError(f"Enhanced retrieval failed: {str(e)}")
    
    def _build_enhanced_query(self, text: str, classification: ClassificationResult) -> str:
        """
        Build an enhanced query using multiple strategies.
        
        Args:
            text: Original text to analyze.
            classification: Classification result.
            
        Returns:
            Enhanced query string optimized for policy retrieval.
        """
        query_parts = []
        
        # 1. Add original text (truncated if too long)
        original_part = text[:200] if len(text) > 200 else text
        query_parts.append(original_part)
        
        # 2. Add classification-specific terms
        classification_terms = self._get_classification_terms(classification.label)
        query_parts.extend(classification_terms)
        
        # 3. Extract and add key phrases from the text
        key_phrases = self._extract_key_phrases(text)
        query_parts.extend(key_phrases[:5])  # Top 5 key phrases
        
        # 4. Add reasoning keywords if available
        if classification.reasoning:
            reasoning_keywords = self._extract_reasoning_keywords(classification.reasoning)
            query_parts.extend(reasoning_keywords[:3])
        
        # Join and clean the query
        query = " ".join(query_parts)
        return self._clean_query(query)
    
    def _get_classification_terms(self, label: str) -> List[str]:
        """Get relevant terms based on classification label."""
        term_mapping = {
            "hate_speech": ["hate speech", "harassment", "discrimination", "violation"],
            "toxic": ["toxic content", "harmful", "abusive", "moderation"],
            "offensive": ["offensive content", "inappropriate", "community standards"],
            "borderline": ["borderline content", "review", "guidelines"],
            "neutral": ["content policy", "community guidelines"]
        }
        return term_mapping.get(label, ["content policy"])
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases that might match policy content."""
        # Simple approach: find words that match our keyword categories
        text_lower = text.lower()
        key_phrases = []
        
        for category, keywords in self.hate_speech_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    key_phrases.append(keyword)
        
        return list(set(key_phrases))  # Remove duplicates
    
    def _extract_reasoning_keywords(self, reasoning: str) -> List[str]:
        """Extract important keywords from classification reasoning."""
        # Extract words that are likely to be policy-relevant
        words = re.findall(r'\b\w+\b', reasoning.lower())
        
        # Filter for meaningful words (not common stop words)
        stop_words = {"the", "is", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        meaningful_words = [w for w in words if len(w) > 3 and w not in stop_words]
        
        # Return most frequent meaningful words
        word_counts = Counter(meaningful_words)
        return [word for word, count in word_counts.most_common(5)]
    
    def _clean_query(self, query: str) -> str:
        """Clean and optimize the query string."""
        # Remove excessive whitespace and duplicates
        words = query.split()
        unique_words = []
        seen = set()
        
        for word in words:
            word_clean = word.lower().strip()
            if word_clean and word_clean not in seen:
                unique_words.append(word)
                seen.add(word_clean)
        
        return " ".join(unique_words[:50])  # Limit query length
    
    def _process_and_rank_results(
        self, 
        raw_results: List[Dict], 
        original_text: str, 
        classification: ClassificationResult
    ) -> List[Dict]:
        """
        Process raw results and add enhanced ranking scores.
        
        Args:
            raw_results: Raw results from vector search.
            original_text: Original text for context matching.
            classification: Classification result for relevance.
            
        Returns:
            Processed results with enhanced scores and explanations.
        """
        processed = []
        
        for result in raw_results:
            # Calculate additional relevance factors
            content = result["data"].get("content", "")
            title = result["data"].get("title", "")
            
            # Text similarity bonus
            text_similarity = self._calculate_text_similarity(original_text, content)
            
            # Classification relevance bonus
            classification_relevance = self._calculate_classification_relevance(
                classification, content, title
            )
            
            # Policy type relevance
            policy_type_score = self._calculate_policy_type_relevance(
                result["data"].get("type", ""), classification.label
            )
            
            # Calculate final score
            base_score = result["score"]
            enhanced_score = (
                base_score * 0.6 +
                text_similarity * 0.2 +
                classification_relevance * 0.15 +
                policy_type_score * 0.05
            )
            
            # Add explanation
            explanation = self._generate_relevance_explanation(
                result, original_text, classification, enhanced_score
            )
            
            processed_result = result.copy()
            processed_result["enhanced_score"] = enhanced_score
            processed_result["explanation"] = explanation
            processed.append(processed_result)
        
        # Sort by enhanced score
        return sorted(processed, key=lambda x: x["enhanced_score"], reverse=True)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on common words."""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_classification_relevance(
        self, 
        classification: ClassificationResult, 
        content: str, 
        title: str
    ) -> float:
        """Calculate how relevant the policy is to the classification."""
        combined_text = f"{title} {content}".lower()
        relevance_score = 0.0
        
        # Check for classification-specific terms
        if classification.label == "hate":
            hate_terms = ["hate", "speech", "harassment", "discrimination", "violence"]
            relevance_score += sum(0.2 for term in hate_terms if term in combined_text)
        
        elif classification.label == "toxic":
            toxic_terms = ["toxic", "harmful", "abusive", "offensive"]
            relevance_score += sum(0.25 for term in toxic_terms if term in combined_text)
        
        # Check for reasoning-related terms
        if classification.reasoning:
            reasoning_words = classification.reasoning.lower().split()
            common_words = set(reasoning_words).intersection(set(combined_text.split()))
            relevance_score += len(common_words) * 0.1
        
        return min(relevance_score, 1.0)  # Cap at 1.0
    
    def _calculate_policy_type_relevance(self, policy_type: str, classification_label: str) -> float:
        """Calculate relevance based on policy type and classification."""
        type_relevance_map = {
            "hate_speech": {
                "community_guidelines": 0.8,
                "legal_framework": 0.9,
                "platform_policy": 0.7
            },
            "toxic": {
                "community_guidelines": 0.9,
                "platform_policy": 0.8,
                "legal_framework": 0.6
            }
        }
        
        return type_relevance_map.get(classification_label, {}).get(policy_type, 0.5)
    
    def _select_diverse_top_policies(self, processed_results: List[Dict]) -> List[Dict]:
        """Select top 3 policies with provider diversity."""
        if len(processed_results) <= 3:
            return processed_results
        
        selected = []
        used_providers = set()
        
        # First pass: select highest scoring from different providers
        for result in processed_results:
            provider = result["data"].get("provider", "Unknown")
            if provider not in used_providers:
                selected.append(result)
                used_providers.add(provider)
                if len(selected) == 3:
                    break
        
        # Fill remaining slots with highest scores
        while len(selected) < 3 and len(selected) < len(processed_results):
            for result in processed_results:
                if result not in selected:
                    selected.append(result)
                    break
        
        return selected[:3]
    
    def _generate_relevance_explanation(
        self, 
        result: Dict, 
        original_text: str, 
        classification: ClassificationResult,
        score: float
    ) -> str:
        """Generate human-readable explanation for why this policy was retrieved."""
        policy_title = result["data"].get("title", "Unknown Policy")
        provider = result["data"].get("provider", "Unknown")
        
        # Find matching keywords
        content = result["data"].get("content", "").lower()
        text_lower = original_text.lower()
        
        matching_keywords = []
        for category, keywords in self.hate_speech_keywords.items():
            for keyword in keywords:
                if keyword in text_lower and keyword in content:
                    matching_keywords.append(keyword)
        
        explanation_parts = [
            f"Retrieved '{policy_title}' from {provider}"
        ]
        
        if matching_keywords:
            explanation_parts.append(f"due to matching terms: {', '.join(matching_keywords[:3])}")
        
        if classification.label in ["hate_speech", "toxic"]:
            explanation_parts.append(f"relevant for {classification.label} classification")
        
        explanation_parts.append(f"(relevance: {score:.2f})")
        
        return " ".join(explanation_parts)
    
    def _create_policy_document(
        self, 
        result: Dict, 
        original_text: str, 
        classification: ClassificationResult
    ) -> PolicyDocument:
        """Create a PolicyDocument from the search result."""
        data = result["data"]
        
        return PolicyDocument(
            id=result["id"],
            title=data.get("title", "Untitled Policy"),
            content=data.get("content", ""),
            category=self._determine_category(data),
            relevance_score=result["enhanced_score"],
            source=data.get("provider", "Unknown"),
            policy_type=data.get("type", "general"),
            explanation=result.get("explanation", "")
        )
    
    def _determine_category(self, policy_data: Dict) -> str:
        """Determine policy category from metadata."""
        policy_type = policy_data.get("type", "general")
        provider = policy_data.get("provider", "").lower()
        
        if "legal" in policy_type or "law" in policy_type:
            return "legal"
        elif any(platform in provider for platform in ["reddit", "meta", "youtube", "google"]):
            return "platform"
        else:
            return "community"
    
    def _create_empty_result(self, query: str) -> RetrievalResult:
        """Create an empty result when no policies are found."""
        return RetrievalResult(
            policies=[],
            query_used=query,
            total_candidates=0
        )