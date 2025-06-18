# app/agents/reasoner.py

from app.agents.base import BaseAgent
from app.services.llm_services import DIALService
from app.models.schemas import PolicyDocument, ClassificationResult
from app.utils.exceptions import AgentExecutionError


class PolicyReasoner(BaseAgent):
    def __init__(self, llm_service: DIALService):
        super().__init__("policy_reasoner")
        self.llm_service = llm_service

    async def _execute(
        self,
        text: str,
        policies: list[PolicyDocument],
        classification: ClassificationResult
    ) -> dict:
        """
        Returns:
        {
            "explanation": "<overall explanation>",
            "policy_summaries": {
                "<policy_id>": "<1-2 sentence explanation>",
                ...
            }
        }
        """
        try:
            prompt = self._build_prompt(text, policies, classification)
            result = await self.llm_service.reason_with_context(prompt)
            if not isinstance(result, dict) or "explanation" not in result:
                raise ValueError("Missing 'explanation' in LLM response.")
            return result
        except Exception as e:
            raise AgentExecutionError(f"Failed to generate policy reasoning: {e}")

    def _build_prompt(self, text: str, policies: list[PolicyDocument], classification: ClassificationResult) -> str:
        policy_section = "\n\n".join(
            f"ID: {p.id}\nTitle: {p.title}\nContent: {p.content.strip()[:1000]}"  # truncate long docs
            for p in policies
        )

        return f"""
You are a senior content policy analyst. Your task is to help explain a classification decision using real platform or legal policies.

User Input:
\"\"\"{text}\"\"\"

Classification:
- Label: {classification.label.upper()}
- Confidence: {classification.confidence:.2f}
- Reasoning: {classification.reasoning}

Relevant Policies:
{policy_section}

Instructions:
- First, summarize overall why these policies justify the classification.
- Then, write 1–2 sentence explanations for each policy ID showing its specific relevance.
- Do NOT hallucinate IDs. Use only the ones provided.
- Respond ONLY in this JSON format:

{{
  "explanation": "High-level summary of why these policies support the decision.",
  "policy_summaries": {{
    "policy_id_1": "This policy addresses...",
    "policy_id_2": "Relevant because...",
    ...
  }}
}}
"""
