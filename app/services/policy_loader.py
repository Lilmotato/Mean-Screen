import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from app.services.embed_service import get_embedding_service
from app.services.qdrant_client import add_policy, init_collection
from app.utils.exceptions import PolicyLoadError

logger = logging.getLogger(__name__)


@dataclass
class PolicyMetadata:
    """Metadata structure for a policy document."""
    filename: str
    provider: str
    policy_type: str
    title: str
    content: str
    word_count: int


class PolicyDocumentLoader:
    """
    Loads and processes policy documents from the data/policy_docs directory.
    Extracts metadata and prepares documents for vector storage.
    """

    def __init__(self, docs_path: str = "data/policy_docs"):
        self.docs_path = Path(docs_path)
        self.embedding_service = get_embedding_service()
        self._provider_mapping = self._build_provider_mapping()

        if not self.docs_path.exists():
            raise PolicyLoadError(f"Policy docs path not found: {self.docs_path}")

    def _build_provider_mapping(self) -> Dict[str, Dict[str, str]]:
        """Maps known policy filenames to metadata (provider, type, title)."""
        return {
            "reddit_policy.txt": {
                "provider": "Reddit",
                "type": "community_guidelines",
                "title": "Reddit Content Policy - Hate Speech and Harassment"
            },
            "meta_community_standards.txt": {
                "provider": "Meta",
                "type": "community_standards",
                "title": "Meta Community Standards - Hate Speech Policy"
            },
            "indian_legal_framework.txt": {
                "provider": "India",
                "type": "legal_framework",
                "title": "Indian Legal Framework - Hate Speech and Online Content"
            },
            "youtube_community_guidelines.txt": {
                "provider": "YouTube",
                "type": "community_guidelines",
                "title": "YouTube Community Guidelines - Hate Speech Policy"
            },
            "google_prohibited_content.txt": {
                "provider": "Google",
                "type": "platform_policy",
                "title": "Google Ads and Search - Prohibited Content Policy"
            }
        }

    def load_all_policies(self) -> List[PolicyMetadata]:
        """
        Load all .txt policy files from the directory.

        Returns:
            List of parsed and validated PolicyMetadata objects.
        """
        txt_files = list(self.docs_path.glob("*.txt"))
        if not txt_files:
            raise PolicyLoadError(f"No .txt files found in {self.docs_path}")

        policies = []
        logger.info(f"Loading {len(txt_files)} policy files from {self.docs_path}")

        for file_path in txt_files:
            try:
                policy = self._load_single_policy(file_path)
                if policy:
                    policies.append(policy)
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")

        logger.info(f"Successfully loaded {len(policies)} policy documents.")
        return policies

    def _load_single_policy(self, file_path: Path) -> Optional[PolicyMetadata]:
        """
        Load and parse a single .txt policy document.

        Returns:
            A PolicyMetadata object or None.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                logger.warning(f"Empty file: {file_path.name}")
                return None

            info = self._provider_mapping.get(
                file_path.name,
                {
                    "provider": file_path.stem.replace('_', ' ').title(),
                    "type": "general_policy",
                    "title": f"{file_path.stem.replace('_', ' ').title()} Policy"
                }
            )

            return PolicyMetadata(
                filename=file_path.name,
                provider=info["provider"],
                policy_type=info["type"],
                title=info["title"],
                content=content,
                word_count=len(content.split())
            )

        except UnicodeDecodeError:
            logger.error(f"File encoding error: {file_path.name}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error with {file_path.name}: {e}")
            return None

    def store_policies_in_vector_db(self, policies: List[PolicyMetadata]) -> Dict[str, str]:
        """
        Stores policies as vectors into the Qdrant vector database.

        Returns:
            Dict mapping filenames to Qdrant vector IDs.
        """
        try:
            init_collection()
            stored = {}

            for policy in policies:
                vector = self.embedding_service.embed_text(policy.content)
                metadata = {
                    "provider": policy.provider,
                    "type": policy.policy_type,
                    "title": policy.title,
                    "content": policy.content,
                    "filename": policy.filename,
                    "word_count": policy.word_count,
                }
                vector_id = add_policy(vector, metadata)
                stored[policy.filename] = vector_id
                logger.info(f"Stored: {policy.title} (ID: {vector_id})")

            return stored
        except Exception as e:
            raise PolicyLoadError(f"Failed to store policies in vector DB: {e}")


def initialize_policy_database(docs_path: str = "data/policy_docs") -> int:
    """
    Load and store all .txt policies from a given directory.

    Returns:
        Number of successfully stored policies.
    """
    loader = PolicyDocumentLoader(docs_path)
    policies = loader.load_all_policies()

    if not policies:
        raise PolicyLoadError("No valid policy documents found.")

    stored = loader.store_policies_in_vector_db(policies)
    return len(stored)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        count = initialize_policy_database()
        print(f"✅ Successfully stored {count} policies into Qdrant.")
    except PolicyLoadError as e:
        print(f"❌ Policy loading failed: {e}")
