from datetime import datetime

from pydantic import BaseModel


class ClassificationLog(BaseModel):
    input_text: str
    classification: str
    policy_match: str
    moderation_action: str
    timestamp: datetime


from datetime import datetime

from pydantic import BaseModel


class ClassificationLog(BaseModel):
    input_text: str
    classification: str
    policy_match: str
    moderation_action: str
    timestamp: datetime
