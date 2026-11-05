'''The generation loop.'''

import numpy as np
import torch
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition
from .constraints import mask_logits_for_json

def generate_function_call(
        model: Small_LLM_Model,
        prompt: str,
        functions: list[FunctionDefinition],
        vocab_map: dict[int, str]
) -> str:
    """Run the token generation loop for a given prompt."""

    full_prompt = f'Prompt: {prompt}\nJSON Result: {{"prompt": "{prompt}", "name": "'

    tensor_ids = model.encode(full_prompt)
    ids: list[int] = tensor_ids.tolist()[0]

    valid_names = [f.name for f in functions]

    for _ in range(150):
        # raw logits
        logits = model.get_logits_from_input_ids(ids)

        logits_np = np.array(logits, dtype=np.float32)
        current_text = model.decode(ids)

        masked_logits = mask_logits_for_json(logits_np, current_text, vocab_map, functions)
        
        next_id = int(np.argmax(masked_logits))
        ids.append(next_id)

        decoded_token = model.decode([next_id])
        if decoded_token == "}" and current_text.endswith("}"):
            break

    return model.decode(ids)
