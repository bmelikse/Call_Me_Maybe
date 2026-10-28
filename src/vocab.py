import json
from pathlib import Path

def load_vocabulary(vocab_path: str) -> dict[int, str]:
    """Load the vocabulary file and return a mapping of token ID -> string."""
    path = Path(vocab_path)

    with path.open(encoding="utf-8") as file:
        raw_vocab = json.load(file)

    vocab_map: dict[int, str] = {
    int(token_id): token for token, token_id in raw_vocab.items()
    }

    return vocab_map

