import numpy as np
from llm_sdk import Small_LLM_Model

# 1. Initialize the model
model = Small_LLM_Model()

# 2. Define the starting text
prompt = "what is an algorithm?"

# 3. Turn the prompt text into a list of numbers (token IDs)
ids: list[int] = model.encode(prompt).tolist()[0]

# 4. Generate new tokens one-by-one in a loop
for _ in range(500):
    # Ask the model for the score (logit) of every single token in its vocabulary
    logits = model.get_logits_from_input_ids(ids)

    # Pick the token ID that has the highest score (greedy decoding)
    next_id = int(np.argmax(logits))

    # Add that new token ID to our list so the model can see it on the next loop
    ids.append(next_id)

# 5. Turn all the token IDs back into English text and print it
result = model.decode(ids)
print(result)