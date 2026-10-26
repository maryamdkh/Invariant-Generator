"""Record when a running PySR search first reaches a loss level.

PySR writes hall_of_fame.csv only when the search ends, but it evaluates its
early-stop condition throughout the search. The condition is extended so
that, the first time the loss falls below `threshold`, Julia creates a marker
file; the stop rule itself is unchanged. The marker's timestamp gives the
time to recovery, separately from when the run was stopped (by hand or by
the time limit).
"""

from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import Any


class LossWatcher:
    def __init__(self, directory: Path, threshold: float) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.threshold = float(threshold)
        self.marker = self.directory / "reached_recovery_loss.marker"
        self.marker.unlink(missing_ok=True)
        self.time_to_threshold: float | None = None
        self.final_best_loss: float | None = None
        self._started = 0.0

    def julia_stop_condition(self, original: str | None) -> str:
        """`original` with a side effect: touch the marker at the threshold."""
        rule = "false"
        if original and "=" in original:
            rule = original.split("=", 1)[1].strip()
        path = str(self.marker).replace("\\", "\\\\").replace('"', '\\"')
        return (
            "stop_if(loss, complexity) = begin; "
            f'if loss < {self.threshold:.6e} && !isfile("{path}"); touch("{path}"); end; '
            f"{rule}; end"
        )

    def __enter__(self) -> "LossWatcher":
        self._started = time.time()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.marker.exists():
            self.time_to_threshold = max(0.0, self.marker.stat().st_mtime - self._started)
        for path in self.directory.rglob("hall_of_fame.csv"):
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    losses = [float(row["Loss"]) for row in csv.DictReader(handle)]
            except (OSError, ValueError, KeyError, csv.Error):
                continue
            if losses:
                best = min(losses)
                self.final_best_loss = best if self.final_best_loss is None else min(best, self.final_best_loss)

    def report(self) -> dict[str, Any]:
        return {
            "loss_threshold": self.threshold,
            "pysr_time_to_threshold": self.time_to_threshold,
            "final_best_loss": self.final_best_loss,
        }


def recovery_loss_threshold(weight_mode: str, y: Any, relative_l2: float = 1e-6) -> float:
    """PySR training loss that corresponds to relative L2 error `relative_l2`:
    relative-squared weights (inverse_target_squared) give loss ~ error^2;
    plain MSE gives loss ~ error^2 * mean(y^2)."""
    import numpy as np

    if weight_mode == "inverse_target_squared":
        return relative_l2**2
    return relative_l2**2 * float(np.mean(np.square(y)))
