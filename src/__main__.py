import argparse
from pathlib import Path
import json

from .loader import load_list, InputError, check_functions
from .models import FunctionDefinition, TestPrompt, FunctionCall
from llm_sdk import Small_LLM_Model

def main() -> None:
    # argument parser object
    parser = argparse.ArgumentParser(description="Function calling with small LLM.")

    parser.add_argument(
        "--functions_definition",
        # if no flags, it does default
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

    try:
        # load and validate function definitions using loader.py
        print(f"Loading functions from: {args.functions_definition}")
        functions = load_list(Path(args.functions_definition), FunctionDefinition)
        check_functions(functions)

        # load and validate test prompts!
        print(f"Loading test prompts from: {args.input}")
        tests = load_list(Path(args.input), TestPrompt)

        # next, initialize the LLM model
        print("Initializing LLM model...")
        model = Small_LLM_Model()

    except InputError as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()