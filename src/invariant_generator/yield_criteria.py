"""Published anisotropic yield criteria used as ground truths.

Stress vectors use the repository's Voigt order [s11, s22, s33, s23, s13, s12]
= [xx, yy, zz, yz, zx, xy] with x = rolling, y = transverse, z = normal
direction. Every criterion returns an equivalent stress that is homogeneous
of degree 1 and equals 1 for unit uniaxial tension along x.

- CB2001: Cazacu & Barlat (2001), Math. Mech. Solids 6:613-630, Eqs. (4), (5),
  (8). Coefficients from the paper's 2090-T3 and 6016-T4 fits (plane stress);
  coefficients that plane-stress data cannot identify (a5, a6, b6-b9, b11) are
  set to their isotropic value 1.
- Yld2004-18p: Barlat et al. (2005), Int. J. Plasticity 21:1009-1039,
  Eqs. (5), (14), (15); coefficients from Table 2 (exponent a = 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _components(X: np.ndarray) -> tuple[np.ndarray, ...]:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 6:
        raise ValueError("Stress must have shape (N, 6) in Voigt order [xx, yy, zz, yz, zx, xy].")
    return tuple(X[:, k] for k in range(6))


# ---------------------------------------------------------------- CB2001


@dataclass(frozen=True, slots=True)
class CB2001:
    a: tuple[float, ...]  # a1..a6
    b: tuple[float, ...]  # b1..b11
    c: float
    source: str = ""

    def invariants(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generalized invariants J2^0 (Eq. 5) and J3^0 (Eq. 4)."""
        sx, sy, sz, syz, sxz, sxy = _components(X)
        a1, a2, a3, a4, a5, a6 = self.a
        b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11 = self.b
        J2 = (
            a1 / 6 * (sx - sy) ** 2 + a2 / 6 * (sy - sz) ** 2 + a3 / 6 * (sx - sz) ** 2
            + a4 * sxy**2 + a5 * sxz**2 + a6 * syz**2
        )
        J3 = (
            (b1 + b2) / 27 * sx**3 + (b3 + b4) / 27 * sy**3
            + (2 * (b1 + b4) - b2 - b3) / 27 * sz**3
            - (b1 * sy + b2 * sz) * sx**2 / 9
            - (b3 * sz + b4 * sx) * sy**2 / 9
            - ((b1 - b2 + b4) * sx + (b1 - b3 + b4) * sy) * sz**2 / 9
            + 2 / 9 * (b1 + b4) * sx * sy * sz
            - sxz**2 / 3 * (2 * b9 * sy - b8 * sz - (2 * b9 - b8) * sx)
            - sxy**2 / 3 * (2 * b10 * sz - b5 * sy - (2 * b10 - b5) * sx)
            - syz**2 / 3 * ((b6 + b7) * sx - b6 * sy - b7 * sz)
            + 2 * b11 * sxy * sxz * syz
        )
        return J2, J3

    def _f(self, X: np.ndarray) -> np.ndarray:
        J2, J3 = self.invariants(X)
        return J2**3 - self.c * J3**2  # Eq. (8), homogeneous of degree 6

    def equivalent_stress(self, X: np.ndarray) -> np.ndarray:
        reference = self._f(np.array([[1.0, 0, 0, 0, 0, 0]]))[0]
        value = self._f(X) / reference
        with np.errstate(invalid="ignore"):
            return np.where(value > 0, np.abs(value) ** (1 / 6), np.nan)


def _cb2001(a1, a2, a3, a4, b1, b2, b3, b4, b5, b10, c, source):
    return CB2001(
        a=(a1, a2, a3, a4, 1.0, 1.0),
        b=(b1, b2, b3, b4, b5, 1.0, 1.0, 1.0, 1.0, b10, 1.0),
        c=c,
        source=source,
    )


# ------------------------------------------------------------ Yld2004-18p


@dataclass(frozen=True, slots=True)
class Yld2004_18p:
    c1: dict[str, float]  # c'_ij
    c2: dict[str, float]  # c''_ij
    exponent: float = 8.0
    source: str = field(default="")

    @staticmethod
    def _matrix(c: dict[str, float]) -> np.ndarray:
        """Eq. (15) acting on [xx, yy, zz, yz, zx, xy]."""
        C = np.zeros((6, 6))
        C[0, 1], C[0, 2] = -c["12"], -c["13"]
        C[1, 0], C[1, 2] = -c["21"], -c["23"]
        C[2, 0], C[2, 1] = -c["31"], -c["32"]
        C[3, 3], C[4, 4], C[5, 5] = c["44"], c["55"], c["66"]
        return C

    @staticmethod
    def _principal(v: np.ndarray) -> np.ndarray:
        xx, yy, zz, yz, zx, xy = v.T
        T = np.stack([
            np.stack([xx, xy, zx], axis=-1),
            np.stack([xy, yy, yz], axis=-1),
            np.stack([zx, yz, zz], axis=-1),
        ], axis=-2)
        return np.linalg.eigvalsh(T)

    def _phi(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        s = X.copy()
        s[:, :3] -= X[:, :3].mean(axis=1, keepdims=True)  # deviator, T in Eq. (5)
        S1 = self._principal(s @ self._matrix(self.c1).T)
        S2 = self._principal(s @ self._matrix(self.c2).T)
        diff = np.abs(S1[:, :, None] - S2[:, None, :])
        return (diff**self.exponent).sum(axis=(1, 2))  # Eq. (14) = 4 sigma_bar^a

    def equivalent_stress(self, X: np.ndarray) -> np.ndarray:
        reference = self._phi(np.array([[1.0, 0, 0, 0, 0, 0]]))[0]
        return (self._phi(X) / reference) ** (1 / self.exponent)


def _yld2004(values: list[float], source: str) -> Yld2004_18p:
    keys = ["12", "13", "21", "23", "31", "32", "44", "55", "66"]
    return Yld2004_18p(
        c1=dict(zip(keys, values[:9])), c2=dict(zip(keys, values[9:])), source=source,
    )


CRITERIA = {
    # Cazacu & Barlat (2001), Section 3.2 (2090-T3) and 3.1 (6016-T4).
    "cb2001_2090T3": _cb2001(
        0.586, 1.05, 0.823, 0.96, 1.44, 0.061, -1.302, -0.281, -0.375, 0.445, 1.285,
        "Cazacu & Barlat (2001), 2090-T3",
    ),
    "cb2001_6016T4": _cb2001(
        0.334, 0.815, 0.815, 0.42, 0.04, -1.205, -0.958, 0.306, 0.153, -0.02, 1.4,
        "Cazacu & Barlat (2001), 6016-T4",
    ),
    # Barlat et al. (2005), Table 2, Yld2004-18p (c'12..c'66, c''12..c''66).
    "yld2004_2090T3": _yld2004([
        -0.069888, 0.936408, 0.079143, 1.003060, 0.524741, 1.363180, 1.023770, 1.069060, 0.954322,
        0.981171, 0.476741, 0.575316, 0.866827, 1.145010, -0.079294, 1.051660, 1.147100, 1.404620,
    ], "Barlat et al. (2005), Table 2, 2090-T3"),
    "yld2004_6111T4": _yld2004([
        1.241024, 1.078271, 1.216463, 1.223867, 1.093105, 0.889161, 0.501909, 0.557173, 1.349094,
        0.775366, 0.922743, 0.765487, 0.793356, 0.918689, 1.027625, 1.115833, 1.112273, 0.589787,
    ], "Barlat et al. (2005), Table 2, 6111-T4"),
}
