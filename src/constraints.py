"""Regex patterns and token constraints."""

from .trace import Tracer
import numpy as np
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition
import re

# v the building phase, anything that could become a number
PARTIAL = re.compile(r"-?(\d+(\.\d*)?)?")
COMPLETE = re.compile(r"-?\d+(\.\d+)?")  # fully finished numbers!
INT_PARTIAL = re.compile(r"-?\d*")
INT_COMPLETE = re.compile(r"-?\d+")
NUMBER_CHARS = set("0123456789.-")


def closes_string(text: str) -> bool:
    """True if the token holds a quote followed by a newline."""
    _, quote, tail = text.partition('"')
    return bool(quote) and "\n" in tail


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
    tracer: Tracer | None = None,
) -> str:
    """Let the model spell one valid function name."""
    ids: list[int] = model.encode(prompt).tolist()[0]
    typed = ""
    candidates = [
        i for i, t in enumerate(token_texts)
        if t and any(t in n for n in names)
    ]
    # the set of newline-only tokens v
    stops = {i for i, t in enumerate(token_texts) if t and set(t) <= {"\n"}}
    while True:
        longer = [n for n in names if n.startswith(typed) and n != typed]
        if typed in names and not longer:
            break
        logits = np.array(
            model.get_logits_from_input_ids(ids),
            dtype=np.float32)
        mask = np.full(len(logits), -np.inf)
        for i in candidates:
            text = token_texts[i]
            if text and any(n.startswith(typed + text) for n in names):
                mask[i] = 0.0
        if typed in names:
            for i in stops:
                mask[i] = 0.0
        if not np.isfinite(mask).any():
            raise ValueError(f"Cannot spell any of {names}")
        next_id = int(np.argmax(logits + mask))
        if tracer is not None:
            tracer.log_step(typed, logits, mask, next_id, token_texts)
        if next_id in stops:
            break
        ids.append(next_id)
        typed += token_texts[next_id]
    return typed


def build_number_ids(token_texts: list[str]) -> list[int]:
    """IDs of tokens made of digits, '.' or '-' (maybe a leading space)."""
    return [
        i for i, t in enumerate(token_texts)
        if t.lstrip(" ") and set(t.lstrip(" ")) <= NUMBER_CHARS
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
        whole: bool = False,
        tracer: Tracer | None = None,
) -> str:
    """Let the model write one number and return its text."""
    ids: list[int] = model.encode(prompt).tolist()[0]
    typed = ""
    partial = INT_PARTIAL if whole else PARTIAL
    complete = INT_COMPLETE if whole else COMPLETE
    for _ in range(30):
        logits = np.array(
            model.get_logits_from_input_ids(ids), dtype=np.float32
        )
        wanted = int(np.argmax(logits))  # the model's favourite
        mask = np.full(len(logits), -np.inf)
        for i in number_ids:
            if partial.fullmatch(typed + token_texts[i].lstrip(" ")):
                mask[i] = 0.0
        if complete.fullmatch(typed):
            for i in newline_ids:
                mask[i] = 0.0
            if mask[wanted] == -np.inf:
                break  # wants to stop, so we stop it
        next_id = int(np.argmax(logits + mask))
        if next_id in newline_ids:
            break
        ids.append(next_id)
        typed += token_texts[next_id].lstrip(" ")
    return typed if complete.fullmatch(typed) else "0"


# which tokens can appear inside the string:
def build_string_mask(token_texts: list[str]) -> np.ndarray:
    """0 for tokens allowed inside a string, -inf for the rest."""
    mask = np.full(len(token_texts), -np.inf, dtype=np.float32)

    for i, text in enumerate(token_texts):
        # no empty tokens, \n with quote and half-done multi-byte chars
        if not text or "\ufffd" in text:
            continue
        if len(text) > 1 and len(set(text)) == 1 and not text[0].isalnum():
            continue  # skip repeat tokens like '****'
        if "\n" not in text or closes_string(text):
            mask[i] = 0.0  # keep ur original score
    return mask


def pick_string(
        model: Small_LLM_Model,
        prompt: str,
        token_texts: list[str],
        string_mask: np.ndarray,
        tracer: Tracer | None = None,
) -> str:
    """Let the model write one string, ending at a closing quote."""
    ids: list[int] = model.encode(prompt).tolist()[0]
    typed = ""
    for _ in range(60):
        logits = np.array(
            model.get_logits_from_input_ids(ids), dtype=np.float32)
        next_id = int(np.argmax(logits + string_mask))
        if tracer is not None:
            tracer.log_step(typed, logits, string_mask, next_id, token_texts)
        text = token_texts[next_id]
        if closes_string(text):
            before = text.partition('"')[0]
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
