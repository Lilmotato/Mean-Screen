# app/agents/reasoner.py
from app.agents.base import BaseAgent
from app.services.llm_services import DIALService
from app.models.schemas import PolicyDocument, ClassificationResult
from app.utils.exceptions import AgentExecutionError

class PolicyReasoner(BaseAgent):
    def __init__(self, llm_service: DIALService):
        super().__init__("policy_reasoner")
        self.llm_service = llm_service

    async def _execute(self, text: str, policies: list[PolicyDocument], classification: ClassificationResult) -> str:
        try:
            prompt = self._build_prompt(text, policies, classification)
            result = await self.llm_service.reason_with_context(prompt)
            return result["explanation"]
        except Exception as e:
            raise AgentExecutionError(f"Failed to generate policy reasoning: {e}")

    def _build_prompt(self, text: str, policies: list[PolicyDocument], classification: ClassificationResult) -> str:
        policy_snippets = "\n\n".join(
            f"[{p.id}] {p.title}\n{p.content}" for p in policies
        )

        return f"""
You are an expert policy analyst tasked with explaining content moderation classifications.

Text:
\"\"\"
{text}
\"\"\"

Classification:
- Label: {classification.label}
- Confidence: {classification.confidence:.2f}
- Reasoning: {classification.reasoning}

Relevant Policies:
{policy_snippets}

Respond ONLY in the following JSON format:
{{
  "explanation": "Your explanation goes here, based on the policies cited."
}}

Instructions:
- Use the policies provided to justify the classification.
- Mention any specific policies (by title or ID).
- Be factual and concise.
"""
