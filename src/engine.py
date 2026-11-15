"""The generation loop."""

from typing import Any

import numpy as np
from llm_sdk import Small_LLM_Model

from .constraints import (
    build_arg_prompt,
    build_prompt,
    pick_function_name,
    pick_number,
    pick_string,
)
from .models import FunctionCall, FunctionDefinition


def generate_function_call(
    model: Small_LLM_Model,
    request: str,
    functions: list[FunctionDefinition],
    token_texts: list[str],
    number_ids: list[int],
    newline_ids: list[int],
    string_mask: np.ndarray,
) -> FunctionCall:
    """Turn one request into a function call."""
    names = [f.name for f in functions]
    prompt = build_prompt(request, functions)
    name = pick_function_name(model, prompt, names, token_texts)
    function = next(f for f in functions if f.name == name)
    prompt = build_arg_prompt(request, function)

    parameters: dict[str, Any] = {}
    for param, spec in function.parameters.items():
        prompt += f"{param} = "
        if spec.type in ("number", "integer"):
            whole = spec.type == "integer"
            raw = pick_number(
                model, prompt, token_texts, number_ids, newline_ids, whole
            )
            parameters[param] = int(raw) if whole else float(raw)
            prompt += raw + "\n"
        elif spec.type == "boolean":
            choice = pick_function_name(
                model, prompt, ["true", "false"], token_texts
            )
            parameters[param] = choice == "true"
            prompt += f"{choice}\n"
        elif spec.type == "string":
            prompt += '"'
            text = pick_string(model, prompt, token_texts, string_mask)
            parameters[param] = text
            prompt += f'{text}"\n'
        else:
            raise ValueError(f"Unsupported parameter type: {spec.type}")
    return FunctionCall(prompt=request, name=name, parameters=parameters)
