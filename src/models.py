from pydantic import BaseModel
from typing import Any

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

