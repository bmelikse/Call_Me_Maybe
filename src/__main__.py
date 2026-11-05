import argparse
from pathlib import Path
import json
import os
from src.loader import load_list, InputError
from src.models import FunctionDefinition, TestPrompt
from src.engine import generate_function_call
from llm_sdk import Small_LLM_Model

def main() -> None:
    parser = argparse.ArgumentParser(description="Function calling with small LLM.")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file."
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the test prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="Path where the output JSON file will be written.",
    )
    args = parser.parse_args()

    functions = load_list(Path(args.functions_definition), FunctionDefinition)
    prompts = load_list(Path(args.input), TestPrompt)
    
    model = Small_LLM_Model()
    vocab_map = {token_id: token_str for token_str, token_id in model._tokenizer.get_vocab().items()}

    results = []

    for prompt_data in prompts:
        prompt_text = prompt_data.prompt 
        output_text = generate_function_call(model, prompt_text, functions, vocab_map)
        results.append(output_text)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()