import argparse
import json
import sys
from pathlib import Path
from llm_sdk import Small_LLM_Model
from src.constraints import build_token_texts, build_number_ids, build_newline_ids
from src.engine import generate_function_call
from src.loader import InputError, check_functions, load_list
from src.models import FunctionDefinition, TestPrompt


def main() -> int:
    """Run the program and return the exit code."""
    parser = argparse.ArgumentParser(
        description="Function calling with a small LLM."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file.",
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
        functions = load_list(
            Path(args.functions_definition), FunctionDefinition
        )
        check_functions(functions)
        prompts = load_list(Path(args.input), TestPrompt)
    except InputError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    model = Small_LLM_Model()
    first = model.encode("hi").tolist()[0]
    size = len(model.get_logits_from_input_ids(first))
    token_texts = build_token_texts(model, size)
    number_ids = build_number_ids(token_texts)
    newline_ids = build_newline_ids(token_texts)

    calls = [
        generate_function_call(model, p.prompt, functions, token_texts, number_ids, newline_ids)
        for p in prompts
    ]

    output = Path(args.output)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump([c.model_dump() for c in calls], file, indent=2)
    except OSError as err:
        print(f"Error: cannot write {output}: {err.strerror}",
              file=sys.stderr)
        return 1

    print(f"Saved {len(calls)} results to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())