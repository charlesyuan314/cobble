from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import override

TOL = 1e-12


@dataclass(frozen=True)
class Polynomial:
    coeffs: Sequence[float]

    def __post_init__(self):
        # Ensure coeffs is a tuple for hashability (required for lru_cache)
        if not isinstance(self.coeffs, tuple):
            object.__setattr__(self, "coeffs", tuple(self.coeffs))

    def trim(self) -> Polynomial:
        i = len(self.coeffs) - 1
        while i > 0 and abs(self.coeffs[i]) < TOL:
            i -= 1
        return Polynomial(self.coeffs[: i + 1])

    def add(self, other: Polynomial, trim: bool = True) -> Polynomial:
        n = max(len(self.coeffs), len(other.coeffs))
        out = [0.0] * n
        for i in range(n):
            out[i] = (self.coeffs[i] if i < len(self.coeffs) else 0.0) + (
                other.coeffs[i] if i < len(other.coeffs) else 0.0
            )
        result = Polynomial(out)
        return result.trim() if trim else result

    def mul(self, other: Polynomial) -> Polynomial:
        out = [0.0] * (len(self.coeffs) + len(other.coeffs) - 1)
        for i, ai in enumerate(self.coeffs):
            if abs(ai) < TOL:
                continue
            for j, bj in enumerate(other.coeffs):
                if abs(bj) < TOL:
                    continue
                out[i + j] += ai * bj
        return Polynomial(out).trim()

    @classmethod
    def pow_x(cls, power: int) -> Polynomial:
        if power < 0:
            raise ValueError("Negative power not supported")
        return cls([0.0] * power + [1.0])

    def compose(self, other: Polynomial) -> Polynomial:
        # Horner's method for composition: self(other(x))
        out = Polynomial([0.0])
        for coeff in reversed(self.coeffs):
            out = out.mul(other)
            out = out.add(Polynomial([coeff]))
        return out.trim()

    def scale(self, c: float) -> Polynomial:
        return Polynomial([coeff * c for coeff in self.coeffs])

    def degree(self) -> int:
        return len(self.coeffs) - 1

    def is_constant(self) -> bool:
        return self.degree() == 0

    class Parity(Enum):
        EVEN = "even"
        ODD = "odd"
        MIXED = "mixed"

    def parity(self) -> Parity:
        """Check if polynomial has definite parity (all odd or all even)."""
        has_even = abs(self.coeffs[0]) > TOL or self.even_component().degree() >= 2
        has_odd = self.odd_component().degree() >= 1

        if has_even and has_odd:
            return self.Parity.MIXED
        elif has_odd:
            return self.Parity.ODD
        else:
            return self.Parity.EVEN

    def even_component(self) -> Polynomial:
        """Extract the even parity component of the polynomial."""
        even_coeffs = [
            coeff if power % 2 == 0 else 0.0 for power, coeff in enumerate(self.coeffs)
        ]
        return Polynomial(even_coeffs).trim()

    def odd_component(self) -> Polynomial:
        """Extract the odd parity component of the polynomial."""
        odd_coeffs = [
            coeff if power % 2 == 1 else 0.0 for power, coeff in enumerate(self.coeffs)
        ]
        return Polynomial(odd_coeffs).trim()

    def eval(self, x: float) -> float:
        acc = 0.0
        for coeff in reversed(self.coeffs):
            acc = acc * x + coeff
        return acc

    def derivative(self) -> Polynomial:
        """Compute the derivative of polynomial."""
        if len(self.coeffs) <= 1:
            return Polynomial([0.0])
        return Polynomial(
            [i * self.coeffs[i] for i in range(1, len(self.coeffs))]
        ).trim()

    def find_roots_in_interval(
        self, a: float, b: float, max_iter: int = 100
    ) -> list[float]:
        """Find roots of polynomial in interval [a, b] using a combination of methods."""
        if a > b:
            a, b = b, a

        deg = self.degree()
        if deg == 0:
            return []
        if deg == 1:
            # Linear case: ax + b = 0 => x = -b/a
            root = (
                -self.coeffs[0] / self.coeffs[1] if abs(self.coeffs[1]) > TOL else None
            )
            return [root] if root is not None and a <= root <= b else []

        # For higher degree polynomials, use numerical methods
        roots: list[float] = []

        # Sample points to look for sign changes (Intermediate Value Theorem)
        num_samples = min(1000, max(100, deg * 20))
        step = (b - a) / num_samples

        prev_x = a
        prev_val = self.eval(a)

        for i in range(1, num_samples + 1):
            x = a + i * step
            val = self.eval(x)

            # Check for sign change
            if prev_val * val < 0:
                # Use bisection method to refine the root
                root = self._bisection_root(prev_x, x, max_iter)
                if root is not None:
                    roots.append(root)
            elif abs(val) < TOL:
                # Direct hit on a root
                roots.append(x)

            prev_x = x
            prev_val = val

        # Remove duplicates (roots found multiple times)
        unique_roots: list[float] = []
        for root in roots:
            is_duplicate = False
            for existing in unique_roots:
                if abs(root - existing) < TOL:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_roots.append(root)

        return unique_roots

    def _bisection_root(self, a: float, b: float, max_iter: int) -> float | None:
        """Find a root in [a, b] using bisection method."""
        fa = self.eval(a)
        fb = self.eval(b)

        if fa * fb > 0:
            return None  # No root in interval

        for _ in range(max_iter):
            c = (a + b) / 2
            fc = self.eval(c)

            if abs(fc) < TOL or abs(b - a) < TOL:
                return c

            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc

        return (a + b) / 2

    def gqsp_chebyshev(self) -> Polynomial:
        """Return a Chebyshev polynomial with the same coefficients."""

        from numpy.polynomial.chebyshev import cheb2poly
        return Polynomial(cheb2poly(self.coeffs))  # type: ignore

    @cache
    def sup_abs_on_circle(self, radius: float) -> float:
        """
        Calculate sup(|f(z)|) on circle |z| = radius in the complex plane.

        For a polynomial with real coefficients, one can sample points on the circle
        and find the maximum absolute value.

        Results are cached for performance.
        """
        import numpy as np

        if radius < TOL:
            # At origin, just return the constant term
            return abs(self.coeffs[0]) if self.coeffs else 0.0

        # Sample many points on the circle |z| = radius
        num_samples = max(1000, self.degree() * 50)
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)

        max_abs = 0.0
        for theta in angles:
            z = radius * np.exp(1j * theta)
            # Evaluate polynomial at z using Horner's method
            val = 0.0 + 0.0j
            for coeff in reversed(self.coeffs):
                val = val * z + coeff
            abs_val = abs(val)
            if abs_val > max_abs:
                max_abs = abs_val

        return float(max_abs)

    @cache
    def sup_abs_on_interval(self, radius: float) -> float:
        """
        Calculate sup(|f(x)|) on interval [-radius, radius] by evaluating f at:
        1. Boundary points
        2. Critical points where f'(x) = 0
        3. Zeros of f(x) = 0

        Results are cached for performance.
        """
        deg = self.degree()
        if deg == 0:
            return abs(self.coeffs[0]) if self.coeffs else 0.0

        candidate_points = [-radius, radius]  # Always check boundaries

        # Find critical points where f'(x) = 0
        derivative = self.derivative()
        critical_points = derivative.find_roots_in_interval(-radius, radius)
        candidate_points.extend(critical_points)

        # Find zeros of f(x) = 0 (these can be local extrema of |f(x)|)
        zeros = self.find_roots_in_interval(-radius, radius)
        candidate_points.extend(zeros)

        # Evaluate |f(x)| at all candidate points
        max_abs_value = 0.0
        for x in candidate_points:
            if -radius <= x <= radius:  # Ensure point is in interval
                abs_value = abs(self.eval(x))
                if abs_value > max_abs_value:
                    max_abs_value = abs_value

        return max_abs_value

    def approx_eq(self, other: Polynomial, tol: float = TOL) -> bool:
        """Check if two polynomials are approximately equal within tolerance."""
        return len(self.coeffs) == len(other.coeffs) and all(
            abs(x - y) < tol for x, y in zip(self.coeffs, other.coeffs)
        )

    @override
    def __str__(self) -> str:
        terms: list[str] = []
        for power, coeff in enumerate(self.coeffs):
            if abs(coeff) < TOL:
                continue
            if power == 0:
                terms.append(f"{coeff:.4g}")
            elif power == 1:
                if abs(coeff - 1.0) < TOL:
                    terms.append("x")
                elif abs(coeff + 1.0) < TOL:
                    terms.append("-x")
                else:
                    terms.append(f"{coeff:.4g}x")
            else:
                if abs(coeff - 1.0) < TOL:
                    terms.append(f"x^{power}")
                elif abs(coeff + 1.0) < TOL:
                    terms.append(f"-x^{power}")
                else:
                    terms.append(f"{coeff:.4g}x^{power}")
        if not terms:
            return "0"
        return " + ".join(terms)
