from pathlib import Path
from src.loader import load_list, InputError
from src.models import TestPrompt

# Define the path to the test prompts file
path = Path("data/input/function_calling_tests.json")

try:
    # Try loading the prompts using our loader
    prompts = load_list(path, TestPrompt)
    print("Success! Loaded prompts:")
    for p in prompts:
        print(f"- {p.prompt}")
        
except InputError as err:
    # If anything goes wrong, it will print our clean error message
    print(f"Error caught: {err}")