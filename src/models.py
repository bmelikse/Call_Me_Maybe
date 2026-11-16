from typing import Any
from pydantic import BaseModel


class ParameterSpec(BaseModel):
    """Type description of one parameter or return value."""
    # e.g.: number
    type: str


class FunctionDefinition(BaseModel):
    """One callable function the LLM is allowed to choose."""

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec


class TestPrompt(BaseModel):
    """One natural-language request from the input file."""

    __test__ = False
    prompt: str


class FunctionCall(BaseModel):
    """One entry of the output file."""

    prompt: str
    name: str
    parameters: dict[str, Any]


class Candidate(BaseModel):
    """A candidate token considered by the model."""

    text: str
    prob: float
    allowed: bool


class Step(BaseModel):
    """Recorded state of one generation step."""

    request: str = ""
    phase: str = ""
    typed: str = ""
    candidates: list[Candidate] = []
    chosen: str = ""