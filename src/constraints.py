import numpy as np
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition
import re

PARTIAL = re.compile(r"-?(\d+(\.\d*)?)?")  # the building phase, anything that could become a number
COMPLETE = re.compile(r"-?\d+(\.\d+)?")  # fully finished numbers!
INT_PARTIAL = re.compile(r"-?\d*")
INT_COMPLETE = re.compile(r"-?\d+")

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
    lines.append("Pick the function that fits the request, then write")
    lines.append("the argument values it needs, like in the example.")
    lines.append("")
    lines.append("Example:")
    lines.append('Request: Replace all digits in "Test 123" with hashes')
    lines.append("fn_substitute_string_with_regex")
    lines.append('source_string = "Test 123"')
    lines.append(r'regex = "\d+"')
    lines.append('replacement = "#"')
    lines.append("")
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
    candidates = [
    i for i, t in enumerate(token_texts)
    if t and any(t in n for n in names)
    ]
    while typed not in names:
        logits = np.array(model.get_logits_from_input_ids(ids), dtype=np.float32)
        mask = np.full(len(logits), -np.inf)
        for i in candidates:
            text = token_texts[i]
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
        whole: bool = False
) -> float:
    """Let the model write one number, then stop at a newline."""
    ids: list[int] = model.encode(prompt).tolist()[0]
    typed = ""
    partial = INT_PARTIAL if whole else PARTIAL
    complete = INT_COMPLETE if whole else COMPLETE
    for _ in range(30):
        logits = np.array(
            model.get_logits_from_input_ids(ids), dtype=np.float32
        )
        mask = np.full(len(logits), -np.inf)  # new np list
        for i in number_ids:
            if partial.fullmatch(typed + token_texts[i]):
                mask[i] = 0.0
        if complete.fullmatch(typed):
            for i in newline_ids:
                mask[i] = 0.0
        next_id = int(np.argmax(logits + mask))
        if next_id in newline_ids:
            break
        ids.append(next_id)
        typed += token_texts[next_id]
    return float(typed) if complete.fullmatch(typed) else 0.0

# which tokens can appear inside the string:
def build_string_mask(token_texts: list[str]) -> np.ndarray:
    """0 for tokens allowed inside a string, -inf for the rest."""
    mask = np.full(len(token_texts), -np.inf, dtype=np.float32)

    for i, text in enumerate(token_texts):
        if not text or "\ufffd" in text:  # no empty tokens, newlines with quote and half-finished multi-byte characters
            continue
        if len(text) > 1 and len(set(text)) == 1 and not text[0].isalnum():
            continue  # skip repeat tokens like '****'
        if '"' in text or "\n" not in text:
            mask[i] = 0.0  # keep ur original score
    return mask

def pick_string(
        model: Small_LLM_Model,
        prompt: str,
        token_texts: list[str],
        string_mask: np.ndarray,
) -> str:
    """Let the model write one string, ending at a closing quote."""
    ids: list[int] = model.encode(prompt).tolist()[0]
    typed = ""
    for _ in range(60):
        logits = np.array(
            model.get_logits_from_input_ids(ids), dtype=np.float32)
        next_id = int(np.argmax(logits + string_mask))
        text = token_texts[next_id]
        if '"' in text:
            before = text.split('"')[0]
            if not typed or before.strip():
                typed += before
            break
        ids.append(next_id)
        if not typed and text.strip():  # no space right after the quote
            text = text.lstrip()
        typed += text
    return typed

def build_arg_prompt(request: str, function: FunctionDefinition) -> str:
    """Short prompt for the argument step: only the chosen function."""
    params = ", ".join(
        f"{n}: {p.type}" for n, p in function.parameters.items()
    )
    return (
        "Example:\n"
        "Function: fn_example(text: string, pattern: string, "
        "new: string)\n"
        'Request: Replace all digits in "Test 123" with hashes\n'
        "Arguments:\n"
        'text = "Test 123"\n'
        r'pattern = "\d+"' "\n"
        'new = "#"\n'
        "\n"
        f"Function: {function.name}({params})\n"
        f"Description: {function.description}\n"
        f"Request: {request}\n"
        "Arguments:\n"
    )