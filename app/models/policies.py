from pydantic import BaseModel, Field
from typing import Literal

class PolicyInput(BaseModel):
    """Schema for inputting a moderation policy"""
    text: str 
    provider: str
    type: str
