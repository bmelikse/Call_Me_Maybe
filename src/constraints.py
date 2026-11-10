import numpy as np
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition
import re

PARTIAL = re.compile(r"-?(\d+(\.d*)?)?")  # the building phase, anything that could become a number
COMPLETE = re.compile(r"-?\d+(\.\d+)?")  # fully finished numbers

def build_token_texts(model: Small_LLM_Model, size: int) -> list[str]:
    """Return the real text of every token ID."""
    return [model.decode([i]) for i in range(size)]

def build_prompt(request: str, functions: list[FunctionDefinition]) -> str:
    """Describe the functions and the request to the model."""
    lines = ["Available functions:"]
    for f in functions:
        params = ", ".join(f"{n}: {p.type}" for n, p in f.parameters.items())
        lines.append(f"- {f.name}({params}): {f.description}")
    lines.append("")
    lines.append("Pick the function that fits the request, then copy its")
    lines.append("argument values from the request, in order.")
    lines.append(f"Request: {request}")
    lines.append("Function to call:\n")
    return "\n".join(lines)

def pick_function_name(
    model: Small_LLM_Model,
    prompt: str,
    names: list[str],
    token_texts: list[str],
) -> str:
    """Let the model spell one valid function name."""
    ids: list[int] = model.encode(prompt).tolist()[0]
    typed = ""
    while typed not in names:
        logits = np.array(model.get_logits_from_input_ids(ids), dtype=np.float32)
        mask = np.full(len(logits), -np.inf)
        for i, text in enumerate(token_texts):
            if text and any(n.startswith(typed + text) for n in names):
                mask[i] = 0.0
        next_id = int(np.argmax(logits + mask))
        ids.append(next_id)
        typed += token_texts[next_id]
    return typed

NUMBER_CHARS = set("0123456789.-")

def build_number_ids(token_texts: list[str]) -> list[int]:
    """IDs of tokens made only of digits, '.' or '-'."""
    return [
        i for i, t in enumerate(token_texts) if t and set(t) <= NUMBER_CHARS
    ]

def build_newline_ids(token_texts: list[str]) -> list[int]:
    """IDs of tokens made only of newline characters."""
    return [i for i, t in enumerate(token_texts) if t and set(t) <= {"\n"}]

def pick_number(
        model: Small_LLM_Model,
        prompt: str,
        token_texts: list[str],
        number_ids: list[int],
        newline_ids: list[int],
) -> float:
    """Let the model write one number, then stop at a newline."""
    ids: list[int] = model.encode(prompt).tolist()[0]
    typed = ""
    for _ in range(30):
        logits = np.array(
            model.get_logits_from_input_ids(ids), dtype=np.float32
        )
        mask = np.full(len(logits), -np.inf)  # new np list
        for i in number_ids:
            if PARTIAL.fullmatch(typed + token_texts[i]):
                mask[i] = 0.0
        if COMPLETE.fullmatch(typed):
            for i in newline_ids:
                mask[i] = 0.0
        next_id = int(np.argmax(logits + mask))
        if next_id in newline_ids:
            break
        ids.append(next_id)
        typed += token_texts[next_id]
    return float(typed) if COMPLETE.fullmatch(typed) else 0.0

# which tokens can appear inside the string:
def build_string_mask(token_texts: list[str]) -> np.ndarray:
    """0 for tokens allowed inside a string, -inf for the rest."""
    mask = np.full(len(token_texts), -np.inf, dtype=np.float32)

    for i, text in enumerate(token_texts):
        if not text or "\n" in text or "\ufffd" in text:  # no empty tokens, newlines with quote and half-finished multi-byte characters
            continue
        if '"' not in text or text == '"':
            mask[i] = 0.0
    return mask

def pick_string(
        model: Small_LLM_Model,
        prompt: str,
        token_texts: list[str],
        string_mask: np.ndarray,
) -> str:
    """Let the model write one string, ending at a closing quote."""
    quote_id = token_texts.index('"')
    ids: list[int] = model.encode(prompt).tolist()[0]
    typed = ""
    for _ in range(100):
        logits = np.array(
            model.get_logits_from_input_ids(ids), dtype=np.float32
        )
        next_id = int(np.argmax(logits + string_mask))
        if next_id == quote_id:
            break
        ids.append(next_id)
        typed += token_texts[next_id]
    return typed