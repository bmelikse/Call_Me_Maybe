from src.constraints import is_token_allowed
from src.models import FunctionDefinition

# 1. Include the required 'returns' field in mock definitions
mock_functions = [
    FunctionDefinition(
        name="fn_add_numbers", 
        description="Add two numbers", 
        parameters={},
        returns={"type": "integer"}
    ),
    FunctionDefinition(
        name="fn_greet", 
        description="Greet a user", 
        parameters={},
        returns={"type": "string"}
    )
]

valid_names = [f.name for f in mock_functions]

print("--- Testing Function Name Spelling ---")
allowed_1 = is_token_allowed('{"name": "fn_ad', 'd', valid_names)
print(f"Adding 'd' to '{{\"name\": \"fn_ad': {allowed_1}")  # Expected: True

allowed_2 = is_token_allowed('{"name": "fn_ad', 'banana', valid_names)
print(f"Adding 'banana' to '{{\"name\": \"fn_ad': {allowed_2}")  # Expected: False

print("\n--- Testing Name Completion & Quote Rule ---")
current_text_finished = '{"name": "fn_add_numbers'

allowed_3 = is_token_allowed(current_text_finished, 'a', valid_names)
print(f"Adding 'a' after finished name: {allowed_3}")  # Expected: False

allowed_4 = is_token_allowed(current_text_finished, '"', valid_names)
print(f"Adding '\"' after finished name: {allowed_4}")  # Expected: True