"""Simple CCD baseline for limb chains."""

from __future__ import annotations

import math

import numpy as np


class CCDSolver:
    """Planar CCD baseline used only for benchmark comparisons."""

    def solve(
        self,
        lengths: list[float],
        target: tuple[float, float],
        max_iters: int = 64,
        tolerance: float = 1e-3,
    ) -> list[float]:
        n = len(lengths)
        angles = [0.0] * n
        tx, ty = target
        for _ in range(max_iters):
            pts = self._forward(lengths, angles)
            ex, ey = pts[-1]
            if math.hypot(tx - ex, ty - ey) < tolerance:
                break
            for i in reversed(range(n)):
                bx, by = pts[i]
                ex, ey = pts[-1]
                v1 = np.array([ex - bx, ey - by], dtype=float)
                v2 = np.array([tx - bx, ty - by], dtype=float)
                n1 = np.linalg.norm(v1) + 1e-9
                n2 = np.linalg.norm(v2) + 1e-9
                cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                delta = math.acos(cos_a)
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                if cross < 0:
                    delta = -delta
                angles[i] += delta
                pts = self._forward(lengths, angles)
        return angles

    @staticmethod
    def _forward(lengths: list[float], angles: list[float]) -> list[tuple[float, float]]:
        out = [(0.0, 0.0)]
        a = 0.0
        x, y = 0.0, 0.0
        for l, da in zip(lengths, angles):
            a += da
            x += l * math.cos(a)
            y += l * math.sin(a)
            out.append((x, y))
        return out
