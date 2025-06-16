from pydantic import BaseModel
from datetime import datetime

class ClassificationLog(BaseModel):
    input_text: str
    classification: str
    policy_match: str
    moderation_action: str
    timestamp: datetime
from pydantic import BaseModel
from datetime import datetime

class ClassificationLog(BaseModel):
    input_text: str
    classification: str
    policy_match: str
    moderation_action: str
    timestamp: datetime
