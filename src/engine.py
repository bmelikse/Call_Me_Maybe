'''The generation loop.'''

import numpy as np
import torch
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition

def generate_function_call(
        model: Small_LLM_Model,
        prompt: str,
        functions: list[FunctionDefinition]
) -> str:
    """Run the token generation loop for a given prompt."""

    full_prompt = f"Prompt: {prompt}\nJSON Result: "

    tensor_ids = model.encode(full_prompt)
    ids: list[int] = tensor_ids.tolist()[0]

    for _ in range(100):
        # raw logits
        logits = model.get_logits_from_input_ids(ids)

        # this is where constrained decoding mask will be later
        # doing greedy decoding for now

        next_id = int(np.argmax(logits))
        ids.append(next_id)

        decoded_token = model.decode([next_id])
        if decoded_token == "}":
            break

    result_text = model.decode(ids)
    return result_text
