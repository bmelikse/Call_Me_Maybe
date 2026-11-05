"""Constraint checking and logit masking for valid JSON generation."""

import numpy as np
from .models import FunctionDefinition

def is_token_allowed(current_text: str, token_text: str, valid_names: list[str]) -> bool:
    """Determine if appending token_text to current_text is allowed."""
    potential_text = current_text + token_text

    if '{"name": "' in current_text:
        name_is_fully_typed = any(current_text.endswith(name) for name in valid_names)
        
        if not name_is_fully_typed:
            name_typed_so_far = potential_text.split('{"name": "')[-1]
            for name in valid_names:
                if name.startswith(name_typed_so_far):
                    return True
            return False

    for name in valid_names:
        if current_text.endswith(name):
            return token_text == '"'

    return True

def mask_logits_for_json(
        logits: np.ndarray,
        current_text: str,
        vocab_map: dict[int, str],
        functions: list[FunctionDefinition]
) -> np.ndarray:
    """
    Optimized mask function: Skips the 152,000-token loop entirely 
    if the function name has already been fully typed.
    """
    valid_function_names = [f.name for f in functions]

    if any(current_text.endswith(name) for name in valid_function_names):
        return logits

    if '{"name": "' in current_text:
        for token_id, token_text in vocab_map.items():
            if not is_token_allowed(current_text, token_text, valid_function_names):
                logits[token_id] = -float("inf")

    return logits
