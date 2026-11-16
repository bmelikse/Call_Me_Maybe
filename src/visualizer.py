"""Matplotlib animation of the generation steps (bonus feature)."""

import logging
import warnings
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.widgets import Button

from .models import Step

GREEN, RED, BLUE = "#2e9e5b", "#d9534f", "#4aa3df"


def make_button(fig: Figure, x: float, label: str) -> Button:
    """A light-gray button with a black label, at horizontal position x."""
    button = Button(
        fig.add_axes((x, 0.04, 0.15, 0.08)), label,
        color="#dddddd", hovercolor="#c4c4c4",
    )
    button.label.set_color("black")
    return button


def show(steps: list[Step], gif_path: str | None = None) -> None:
    """Animate the recorded steps in a window, or save them as a GIF."""
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*Glyph.*missing from font")
    plt.rcParams.update({
        "font.family": [
            "Arial", "Liberation Sans", "DejaVu Sans", "Noto Sans CJK JP",
        ],
        "font.weight": "bold",
        "font.size": 13,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
    })
    fig, (bars, info) = plt.subplots(
        1, 2, figsize=(19.5, 9), gridspec_kw={"width_ratios": [3, 2]}
    )
    fig.subplots_adjust(bottom=0.2)
    if fig.canvas.manager:
        fig.canvas.manager.set_window_title("Call Me Maybe")

    def draw(i: int) -> list[Any]:
        """Draw step number i."""
        step = steps[i]
        cands = step.candidates
        bars.clear()
        bars.barh(
            [repr(c.text) for c in cands],
            [c.prob for c in cands],
            color=[GREEN if c.allowed else RED for c in cands],
            edgecolor=["black" if c.text == step.chosen else "none"
                       for c in cands],
            linewidth=3,
        )
        bars.invert_yaxis()
        bars.set_xlim(0, 1)
        bars.set_title("green = allowed, red = blocked, outline = chosen")

        wanted = cands[0]  # the model's own favorite
        verdict = f"Model wanted {wanted.text!r}: allowed"
        if not wanted.allowed:
            verdict = (f"Model wanted {wanted.text!r}: BLOCKED\n"
                       f"Forced to write {step.chosen!r}")
        written = (step.typed + step.chosen).replace("\n", "\\n")

        info.clear()
        info.axis("off")
        for y, text, extra in [
            (0.97, f"Request:\n{step.request}", {}),
            (0.68, f"Generating: {step.phase}", {}),
            (0.56, f"Written so far:\n{written}", {"family": "monospace"}),
            (0.25, verdict, {"color": GREEN if wanted.allowed else RED}),
            (0.10, f"Step {i + 1} / {len(steps)}", {}),
        ]:
            info.text(0, y, text, va="top", wrap=True, **extra)
        info.plot([0, 1], [0.03, 0.03], color="#dddddd", linewidth=8)
        info.plot([0, (i + 1) / len(steps)], [0.03, 0.03],
                  color=BLUE, linewidth=8)
        info.set(xlim=(0, 1), ylim=(0, 1))
        return []

    if gif_path:
        FuncAnimation(
            fig, draw, frames=len(steps), interval=800
        ).save(gif_path, writer="pillow")
        return

    state: dict[str, Any] = {"i": 0, "playing": True}

    def go(i: int) -> None:
        """Show step i (kept inside the valid range)."""
        state["i"] = max(0, min(i, len(steps) - 1))
        draw(state["i"])
        fig.canvas.draw_idle()

    def toggle(_event: Any) -> None:
        """Pause or resume the automatic playback."""
        state["playing"] = not state["playing"]
        play.label.set_text("Pause" if state["playing"] else "Play")
        fig.canvas.draw_idle()

    def step_by(delta: int) -> None:
        """Pause, then move one step backward or forward."""
        state["playing"] = False
        play.label.set_text("Play")
        go(state["i"] + delta)

    def tick() -> None:
        """Called by the timer: move on while playing."""
        if state["playing"] and state["i"] < len(steps) - 1:
            go(state["i"] + 1)

    prev = make_button(fig, 0.25, "Previous")
    play = make_button(fig, 0.425, "Pause")
    nxt = make_button(fig, 0.6, "Next")
    prev.on_clicked(lambda _e: step_by(-1))
    play.on_clicked(toggle)
    nxt.on_clicked(lambda _e: step_by(1))

    timer = fig.canvas.new_timer(interval=800)
    timer.add_callback(tick)
    timer.start()
    draw(0)
    plt.show()
