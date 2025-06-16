#llm_service.py
import os
import json
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from app.utils.exceptions import LLMServiceError

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DIALService:
    """
    DIALService wraps LangChain's AzureChatOpenAI for classifying text via the DIAL API.
    """

    def __init__(self):
        api_key = os.getenv("DIAL_API_KEY")
        deployment = os.getenv("DIAL_DEPLOYMENT_NAME", "gpt-4")
        endpoint = os.getenv("DIAL_API_URL", "https://ai-proxy.lab.epam.com")
        api_version = os.getenv("DIAL_API_VERSION", "2023-12-01-preview")

        if not api_key:
            raise ValueError("Missing required environment variable: DIAL_API_KEY")

        try:
            self.client = AzureChatOpenAI(
                openai_api_version=api_version,
                azure_deployment=deployment,
                azure_endpoint=endpoint,
                api_key=api_key,
            )
            logger.info(f"DIALService initialized with deployment: {deployment}")
        except Exception as e:
            logger.error(f"Failed to initialize AzureChatOpenAI: {e}")
            raise LLMServiceError("DIALService initialization failed")


    async def classify_text(self, text: str) -> Dict[str, Any]:
        """
        Sends a classification prompt to the DIAL LLM and parses the response as JSON.
        """

        system_prompt = (
            "You are a hate speech classifier. Analyze the text and return ONLY a JSON response like:\n"
            '{\n'
            '  "label": "hate|toxic|offensive|neutral|ambiguous",\n'
            '  "confidence": 0.85,\n'
            '  "reasoning": "Brief explanation"\n'
            '}'
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Classify this text: {text}")
        ]

        try:
            logger.info(f"Sending classification request to DIAL for: {text}")
            response = await self.client.ainvoke(messages)
            content = response.content.strip()
            logger.info(f"Raw LLM response: {content}")

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON returned: {content}")
            raise LLMServiceError(f"Invalid JSON response: {content}")

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            raise LLMServiceError(f"LLM classification failed: {e}")

    async def reason_with_context(self, prompt: str) -> Dict[str, Any]:
        """
        Uses the DIAL LLM to explain a classification decision based on provided policy context.
        Expects a JSON response like:
        {
          "explanation": "..."
        }
        """
        messages = [
            SystemMessage(content="You are a content policy analyst. Return only JSON."),
            HumanMessage(content=prompt)
        ]

        try:
            logger.info("Sending reasoning prompt to DIAL")
            response = await self.client.ainvoke(messages)
            content = response.content.strip()
            logger.info(f"Raw reasoning response: {content}")

            return json.loads(content)

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in reasoning response: {content}")
            raise LLMServiceError(f"Invalid JSON response: {content}")

        except Exception as e:
            logger.error(f"LLM reasoning failed: {e}")
            raise LLMServiceError(f"LLM reasoning failed: {e}")
