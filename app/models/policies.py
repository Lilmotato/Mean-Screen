from typing import Literal

from pydantic import BaseModel, Field


class PolicyInput(BaseModel):
    """Schema for inputting a moderation policy"""

    text: str
    provider: str
    type: str
