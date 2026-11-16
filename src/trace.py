"""Optional recording of the generation steps (used by the visualizer)."""

import numpy as np
from pydantic import BaseModel

from .models import Candidate, Step


class Tracer(BaseModel):
    """Collects what happens at each generation step."""

    steps: list[Step] = []
    request: str = ""
    phase: str = ""

    def log_step(
        self,
        typed: str,
        logits: np.ndarray,
        mask: np.ndarray,
        chosen_id: int,
        token_texts: list[str],
    ) -> None:
        """Record the model's favorite tokens and the one we picked."""
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        ids = [int(i) for i in np.argsort(-logits)[:8]]
        if chosen_id not in ids:
            ids.append(chosen_id)
        candidates = [
            Candidate(
                text=token_texts[i],
                prob=float(probs[i]),
                allowed=bool(np.isfinite(mask[i])),
            )
            for i in ids
        ]
        self.steps.append(
            Step(
                request=self.request,
                phase=self.phase,
                typed=typed,
                candidates=candidates,
                chosen=token_texts[chosen_id],
            )
        )