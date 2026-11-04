"""Constraint checking and logit masking for valid JSON generation."""

import numpy as np
from .models import FunctionDefinition

def is_token_allowed(current_text: str, token_text: str, valid_names: list[str]) -> bool:
    """Determine if appending token_text to current_text is allowed."""
    potential_text = current_text + token_text

    if '{"name":' in current_text:
        name_typed_so_far = potential_text.split('{"name": "')[-1]

        for name in valid_names:
            if name.startswith(name_typed_so_far):
                return True
        return False
    return True


def mask_logits_for_json(
        logits: np.ndarray,
        current_text: str,
        vocab_map: dict[int, str],
        functions: list[FunctionDefinition]
) -> np.ndarray:
    """
    Examines every token in the vocabulary. If a token's text would violate 
    our JSON structure or function names, we punish its logit score to -inf.
    """

    valid_function_names = [f.name for f in functions]

    for token_id, token_text in vocab_map.items():
        if not is_token_allowed(current_text, token_text, valid_function_names):
            logits[token_id] = -float("inf")

    return logits
