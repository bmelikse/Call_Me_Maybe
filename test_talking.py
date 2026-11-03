import numpy as np
from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

prompt: str = "what is an algorithm?"

ids: list[int] = model.encode(prompt).tolist()[0]

for _ in range(500):
    logits = model.get_logits_from_input_ids(ids)
    next_id = int(np.argmax(logits))
    ids.append(next_id)

result = model.decode(ids)
print(result)