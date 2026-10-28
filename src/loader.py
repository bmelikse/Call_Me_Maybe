"""Loading and validating the JSON input files."""

import json
from pathlib import Path
from typing import TypeVar
from pydantic import BaseModel, ValidationError
from .models import FunctionDefinition

T = TypeVar("T", bound=BaseModel)

def _describe(error: ValidationError) ->  str:
    """Turn a pydantic error into one short readable line."""
    parts = []
    for err in error.errors():
        where = ".".join(str(p) for p in err["loc"]) or "value"
        parts.append(f"{where}: {err['msg']}")
    return ";".join(parts)

class InputError(Exception):
    """Raised when an input file is missing, unreadable or malformed."""

def load_list(path: Path, model: type[T]) -> list[T]:
    """Read a JSON array from ``path`` and validate it against a Pydantic model."""
    try:
        with path.open(encoding="utf-8") as file:
            raw = json.load(file)
    except FileNotFoundError:
        raise InputError(f"File not found: {path}") from None
    except OSError as err:
        raise InputError(f"Cannot read {path}: {err.strerror}") from err
    except UnicodeDecodeError:
        raise InputError(f"{path} is not valid UTF-8 text") from None
    except json.JSONDecodeError as err:
        raise InputError(
            f"Invalid JSON in {path} (line {err.lineno}, "
            f"column {err.colno}): {err.msg}"
        ) from err

    if not isinstance(raw, list):
        raise InputError(f"{path} must contain a JSON array at top level")

    items: list[T] = []
    for index, entry in enumerate(raw):
        try:
            items.append(model.model_validate(entry))
        except ValidationError as err:
            raise InputError(
                f"{path}, item {index}: {_describe(err)}"
            ) from err
    return items


def check_functions(functions: list[FunctionDefinition]) -> None:
    """Raise InputError if the list is empty or has duplicate names."""
    if not functions:
        raise InputError("The functions definition file is empty")
    seen: set[str] = set()
    for function in functions:
        if function.name in seen:
            raise InputError(f"Duplicate function name: {function.name}")
        seen.add(function.name)
