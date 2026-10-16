"""Feynman Symbolic Regression Database problems from PMLB metadata files.

PMLB (used by SRBench) stores each AI Feynman problem with its formula and
sampling ranges in `metadata.yaml`. This module parses those files and samples
fresh noiseless data uniformly in the published ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

_RANGE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+in\s+\[\s*([^,\]]+)\s*,\s*([^\]]+)\]\s*$")
_FORMULA = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")


@dataclass(slots=True)
class FeynmanProblem:
    name: str
    target: str
    formula: str
    variables: list[str]
    ranges: dict[str, tuple[float, float]]

    def symbols(self) -> dict[str, object]:
        """SymPy symbols for the inputs; all published ranges are positive."""
        import sympy as sp

        positive = all(low > 0 for low, _ in self.ranges.values())
        return {name: sp.Symbol(name, positive=positive or None) for name in self.variables}

    def expression(self):
        import sympy as sp

        # Formulas use numpy-style names, which SymPy would read as unknown functions.
        functions = {"arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan, "ln": sp.log}
        return sp.sympify(self.formula, locals={**functions, **self.symbols()})

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        import sympy as sp

        symbols = self.symbols()
        function = sp.lambdify([symbols[name] for name in self.variables], self.expression(), "numpy")
        values = function(*[X[:, idx] for idx in range(X.shape[1])])
        return np.broadcast_to(np.asarray(values, dtype=np.float64), (X.shape[0],)).copy()

    def sample(self, n_samples: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        X = np.column_stack([rng.uniform(*self.ranges[name], size=n_samples) for name in self.variables])
        y = self.evaluate(X)
        if not np.isfinite(y).all():
            raise ValueError(f"{self.name}: formula is not finite on the sampled range.")
        return X, y


def log_reducibility(problem: FeynmanProblem, *, n_points: int = 400, seed: int = 1) -> int | None:
    """Smallest n with f = g(B log x), B of shape (n, d); None if an input is non-positive.

    If f depends on x only through n monomials, every gradient of f with
    respect to log x lies in the row space of B, so the numerical rank of the
    stacked, row-normalized gradients is n. This uses the known formula and is
    for analysis only; the pipeline never sees it.
    """
    import sympy as sp

    if any(low <= 0 for low, _ in problem.ranges.values()):
        return None
    symbols = problem.symbols()
    ordered = [symbols[name] for name in problem.variables]
    expression = problem.expression()
    X, _ = problem.sample(n_points, seed=seed)
    columns = [X[:, idx] for idx in range(X.shape[1])]
    gradients = np.column_stack([
        np.broadcast_to(
            np.asarray(sp.lambdify(ordered, sp.diff(expression, symbol) * symbol, "numpy")(*columns), float),
            (n_points,),
        )
        for symbol in ordered
    ])
    gradients = gradients / np.linalg.norm(gradients, axis=1, keepdims=True).clip(1e-300)
    singular_values = np.linalg.svd(gradients, compute_uv=False)
    return int((singular_values > 1e-6 * singular_values[0]).sum())


def select_pilot_problems(problems: list[FeynmanProblem]) -> list[FeynmanProblem]:
    """Main-set equations with >= 6 inputs and bonus equations with >= 4 inputs,
    restricted to strictly positive input ranges (required by log features)."""
    selected = []
    for problem in problems:
        bonus = problem.name.startswith("feynman_test_")
        if len(problem.variables) < (4 if bonus else 6):
            continue
        if any(low <= 0 for low, _ in problem.ranges.values()):
            continue
        selected.append(problem)
    return selected


def parse_pmlb_metadata(path: str | Path) -> FeynmanProblem:
    """Read the formula, variables and ranges from a PMLB Feynman metadata.yaml."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    name = re.search(r"^dataset:\s*(\S+)", text, re.MULTILINE)
    formula: tuple[str, str] | None = None
    variables: list[str] = []
    ranges: dict[str, tuple[float, float]] = {}
    in_description = False
    for line in text.splitlines():
        if line.startswith("description:"):
            in_description = True
            continue
        if in_description and line and not line.startswith(" "):
            break
        if not in_description:
            continue
        range_match = _RANGE.match(line)
        if range_match:
            variable = range_match.group(1)
            variables.append(variable)
            ranges[variable] = (float(range_match.group(2)), float(range_match.group(3)))
            continue
        formula_match = _FORMULA.match(line)
        if formula_match and formula is None and " in " not in line:
            formula = (formula_match.group(1), formula_match.group(2))
    if name is None or formula is None or not variables:
        raise ValueError(f"Could not parse a Feynman problem from {path}")
    return FeynmanProblem(
        name=name.group(1),
        target=formula[0],
        formula=formula[1],
        variables=variables,
        ranges=ranges,
    )
