from pydantic import BaseModel, Field, HttpUrl 
from typing import Any


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl] # role -> agent URL
    config: dict[str, Any]


class QuestionPair(BaseModel):
    """A pair of questions to evaluate for duplication."""
    question1: str
    question2: str


class DeduplicationResponse(BaseModel):
    """Response from purple agent with prediction and optional justification."""
    justification: str | None = Field(None, description="Optional explanation of the prediction")
    prediction: int = Field(..., ge=0, le=1, description="Binary prediction: 1 for duplicate, 0 for not duplicate")
