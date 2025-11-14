from __future__ import annotations
from collections.abc import Sequence
from typing import Callable, TYPE_CHECKING

from cobble.polynomial import Polynomial, TOL

if TYPE_CHECKING:
    from cobble.expr import Division, Expr, Sum


def _flatten_and_combine_terms(
    terms: Sequence[tuple[float, Expr]],
) -> list[tuple[float, Expr]]:
    """Flatten nested sums, combine like terms, and drop zeros.

    This function normalizes most expression types, but preserves Tensor structure
    to enable bilinear factorization patterns like (A+B)⊗(C+D).
    """
    from cobble.expr import Poly, Sum, Tensor

    flat: list[tuple[float, Expr]] = []
    for coef, term in terms:
        if isinstance(term, Tensor):
            flat.append((coef, term))
        elif isinstance(term, Sum):
            for c2, t2 in _flatten_and_combine_terms(term.terms):
                flat.append((coef * c2, t2))
        else:
            flat.append((coef, term.optimize()))

    combined: list[tuple[float, Expr]] = []
    key_to_idx: dict[str, int] = {}
    for coef, term in flat:
        key = repr(term)
        idx = key_to_idx.get(key)
        if idx is not None:
            c2, t2 = combined[idx]
            combined[idx] = (c2 + coef, t2)
        else:
            key_to_idx[key] = len(combined)
            combined.append((coef, term))

    # Don't drop Poly terms based on scalar coefficient alone, because
    # the polynomial might have large coefficients when scaled
    return [(c, t) for c, t in combined if abs(c) > TOL or isinstance(t, Poly)]


def _merge_scalars_into_expressions(
    terms: list[tuple[float, Expr]],
) -> list[tuple[float, Expr]]:
    """Merge scalar coefficients into expressions: c·poly(A,f) -> poly(A, c·f), c·I -> Const(c), etc."""
    from cobble.expr import Const, Poly

    merged: list[tuple[float, Expr]] = []
    for c, t in terms:
        if isinstance(t, Poly):
            # c·poly(A,f) -> poly(A, c·f)
            merged.append((1.0, Poly(t.expr, t.p.scale(c)).optimize()))
        elif isinstance(t, Const):
            # c·Const(v) -> Const(c·v)
            merged.append((1.0, Const(c * t.value)))
        else:
            merged.append((c, t))
    return merged


def _merge_const_with_poly(terms: list[tuple[float, Expr]]) -> Expr:
    """Merge constant terms with polynomials: c·I + poly(A,f) => poly(A, f + c)."""
    from cobble.expr import Const, Poly, Sum

    const_val = 0.0
    others: list[tuple[float, Expr]] = []

    for c, t in terms:
        if isinstance(t, Const):
            const_val += c * t.value
        else:
            others.append((c, t))

    if abs(const_val) > TOL and const_val.imag == 0:
        # Check if single other is Poly; if so, fold constant
        if (
            len(others) == 1
            and isinstance(others[0][1], Poly)
            and abs(others[0][0] - 1.0) < TOL
        ):
            poly_node: Poly = others[0][1]
            new_p = poly_node.p.add(Polynomial([const_val.real]), trim=False)
            return Poly(poly_node.expr, new_p).optimize()
        # Otherwise keep explicit Const term
        others.insert(0, (1.0, Const(const_val)))

    if len(others) == 1:
        c, t = others[0]
        if abs(c - 1.0) < TOL:
            return t

    return Sum(others) if others else Const(0.0)


def _combine_polys_with_same_base(
    terms: list[tuple[float, Expr]],
) -> list[tuple[float, Expr]]:
    """Combine polynomials with same base: poly(A,f) + poly(A,g) => poly(A, f+g)."""
    from cobble.expr import Expr, Poly

    poly_groups: list[tuple[Expr, list[Polynomial]]] = []
    others: list[tuple[float, Expr]] = []

    for c, t in terms:
        if isinstance(t, Poly) and abs(c - 1.0) < TOL:
            placed = False
            for base, plist in poly_groups:
                if Expr.structural_eq(base, t.expr):
                    plist.append(t.p)
                    placed = True
                    break
            if not placed:
                poly_groups.append((t.expr, [t.p]))
        else:
            others.append((c, t))

    combined: list[tuple[float, Expr]] = []
    for base, plist in poly_groups:
        acc = Polynomial([0.0])
        for p in plist:
            acc = acc.add(p, trim=False)
        combined.append((1.0, Poly(base, acc).optimize()))
    combined.extend(others)

    return combined


def _combine_base_expr_with_poly(
    terms: list[tuple[float, Expr]],
) -> list[tuple[float, Expr]]:
    """Combine base expression with polynomial: (A + B) + poly(A + B, f) => poly(A + B, [1] + f)."""
    from cobble.expr import Expr, Poly, Sum

    # Separate polynomials from other terms
    polys: list[tuple[float, Poly]] = []
    non_polys: list[tuple[float, Expr]] = []

    for c, t in terms:
        if isinstance(t, Poly):
            polys.append((c, t))
        else:
            non_polys.append((c, t))

    # Try to match non-poly terms with poly bases
    if non_polys and polys:
        # First, try to match individual scalar multiples of base expressions
        # e.g., (c, Base('X')) with Poly(Base('X'), p)
        for i, (non_poly_c, non_poly_term) in enumerate(non_polys):
            for j, (poly_c, poly) in enumerate(polys):
                if abs(poly_c - 1.0) < TOL and Expr.structural_eq(
                    non_poly_term, poly.expr
                ):
                    # Found a match: c * Base matches Poly(Base, p)
                    # Combine as: Poly(Base, c*[0,1] + p)
                    identity_poly = Polynomial([0.0, 1.0]).scale(non_poly_c)
                    new_p = identity_poly.add(poly.p, trim=False)
                    combined_poly = Poly(poly.expr, new_p).optimize()

                    # Build result without the matched terms
                    result: list[tuple[float, Expr]] = [(1.0, combined_poly)]
                    for k, (c, t) in enumerate(non_polys):
                        if k != i:
                            result.append((c, t))
                    for k, (c, p) in enumerate(polys):
                        if k != j:
                            result.append((c, p))
                    return result

        # If no individual match, try forming a sum from non-poly terms
        if len(non_polys) > 1:
            non_poly_sum = Sum(non_polys).optimize()
        else:
            c, t = non_polys[0]
            if abs(c - 1.0) < TOL:
                non_poly_sum = t
            else:
                non_poly_sum = Sum([(c, t)]).optimize()

        # Check if this sum matches any polynomial base
        matched_poly_idx = None
        for idx, (poly_c, poly) in enumerate(polys):
            if abs(poly_c - 1.0) < TOL and Expr.structural_eq(non_poly_sum, poly.expr):
                matched_poly_idx = idx
                break

        if matched_poly_idx is not None:
            # Combine the non-poly sum with the polynomial
            _, matched_poly = polys[matched_poly_idx]
            # Treat non_poly_sum as poly(non_poly_sum, x) = [0, 1]
            new_p = Polynomial([0.0, 1.0]).add(matched_poly.p, trim=False)
            combined_poly = Poly(non_poly_sum, new_p).optimize()

            # Build the final result
            result = [(1.0, combined_poly)]
            # Add remaining polynomials
            for idx, (poly_c, poly) in enumerate(polys):
                if idx != matched_poly_idx:
                    result.append((poly_c, poly))
            return result

    # If no match, keep everything as is
    return terms


def _try_poly_product_fusion(factors: list[Expr]) -> Expr | None:
    """Fuse product of Consts, a single expression E, and/or Poly(E,·) into Poly(E,·).

    Allowed factors: Const, Expr(E), Poly(E, ·). Any other factor aborts.
    Const factors scale the polynomial. Expr(E) multiplies by x.
    All non-const factors must have the same base expression E.
    """

    from cobble.expr import Expr, Const, Poly, Sum

    base: Expr | None = None
    acc_poly: Polynomial | None = None
    scale: float = 1.0
    for f in factors:
        if isinstance(f, Const) and f.value.imag == 0:
            scale *= f.value.real
            continue
        if isinstance(f, Poly):
            if base is None:
                base = f.expr
                acc_poly = f.p
            elif Expr.structural_eq(f.expr, base):
                assert acc_poly is not None
                acc_poly = acc_poly.mul(f.p)
            else:
                return None
            continue
        # Check if it's a scalar multiple Sum like (c·E)
        if isinstance(f, Sum) and len(f.terms) == 1:
            coef, term = f.terms[0]
            if base is None:
                base = term
                acc_poly = Polynomial([0.0, coef])  # c*x
            elif Expr.structural_eq(term, base):
                assert acc_poly is not None
                acc_poly = acc_poly.mul(Polynomial([0.0, coef]))
            else:
                return None
            continue
        # Any other expression (Base, etc.)
        if base is None:
            base = f
            acc_poly = Polynomial([0.0, 1.0])  # x
        elif Expr.structural_eq(f, base):
            assert acc_poly is not None
            acc_poly = acc_poly.mul(Polynomial([0.0, 1.0]))
        else:
            return None
        continue
    if base is None or acc_poly is None:
        return None
    if abs(scale - 1.0) > TOL:
        acc_poly = acc_poly.scale(scale)
    return Poly(base, acc_poly)


def _are_scalar_multiples(e1: Expr, e2: Expr) -> tuple[bool, float]:
    """Check if two expressions are scalar multiples. Returns (is_multiple, scale_factor)."""
    from cobble.expr import Sum

    if not (isinstance(e1, Sum) and isinstance(e2, Sum)):
        return False, 0.0

    if len(e1.terms) != len(e2.terms):
        return False, 0.0

    # Build maps from term repr to coefficient
    e1_map = {repr(t): c for c, t in e1.terms}
    e2_map = {repr(t): c for c, t in e2.terms}

    if set(e1_map.keys()) != set(e2_map.keys()):
        return False, 0.0

    # Find the scale factor from the first term
    first_key = next(iter(e1_map.keys()))
    if abs(e1_map[first_key]) < TOL:
        return False, 0.0

    scale = e2_map[first_key] / e1_map[first_key]

    # Check if all other terms match this scale
    for key in e1_map:
        expected = e1_map[key] * scale
        if abs(e2_map[key] - expected) > TOL:
            return False, 0.0

    return True, scale


def _try_factor_by_common(
    terms: list[tuple[float, Expr]],
    extract_fn: Callable[[Expr], tuple[Expr, Expr] | None],
    build_fn: Callable[[Expr, Sum], Expr],
    check_scalars: bool = False,
) -> tuple[bool, list[tuple[float, Expr]]]:
    """Generic factoring: group terms by common factor and build factored result.

    Args:
        terms: List of (coeff, expr) tuples
        extract_fn: Function that takes an expr and returns (common_factor, rest) or None
        build_fn: Function that takes (common_factor, Sum(rest_terms)) and builds result
        check_scalars: If True, group factors that are scalar multiples
    """
    from cobble.expr import Sum

    groups: dict[str, list[tuple[float, Expr]]] = {}
    factor_map: dict[str, Expr] = {}
    original_terms_map: dict[str, list[tuple[float, Expr]]] = {}  # Track originals

    for c, t in terms:
        result = extract_fn(t)
        if result:
            factor, rest = result
            factor_repr = repr(factor)

            if check_scalars:
                # Look for scalar multiples
                found = False
                for canon_repr, canon_factor in factor_map.items():
                    is_mult, scale = _are_scalar_multiples(canon_factor, factor)
                    if is_mult:
                        groups[canon_repr].append((c * scale, rest))
                        original_terms_map.setdefault(canon_repr, []).append((c, t))
                        found = True
                        break
                if not found:
                    groups.setdefault(factor_repr, []).append((c, rest))
                    factor_map[factor_repr] = factor
                    original_terms_map.setdefault(factor_repr, []).append((c, t))
            else:
                groups.setdefault(factor_repr, []).append((c, rest))
                factor_map[factor_repr] = factor
                original_terms_map.setdefault(factor_repr, []).append((c, t))
        else:
            key = repr(("", t))
            groups.setdefault(key, []).append((c, t))
            original_terms_map.setdefault(key, []).append((c, t))

    new_terms: list[tuple[float, Expr]] = []
    did_factor = False
    for key, group in groups.items():
        if (
            key in factor_map
            and len(group) > 1
            and all(not isinstance(t, Sum) for _, t in group)
        ):
            new_terms.append((1.0, build_fn(factor_map[key], Sum(group))))
            did_factor = True
        else:
            # Don't factor - use original terms
            new_terms.extend(original_terms_map[key])

    return did_factor, new_terms


def _factor_sum(terms: list[tuple[float, Expr]]) -> Expr:
    """Factor a Sum by common product or tensor factors."""
    from cobble.expr import Sum, Prod, Tensor

    if len(terms) == 1 and abs(terms[0][0] - 1.0) < TOL:
        return terms[0][1]

    current_terms = terms
    seen: set[str] = set()

    while True:
        snapshot = repr(Sum(current_terms))
        if snapshot in seen:
            break
        seen.add(snapshot)

        def _extract_prod_left(t: Expr) -> tuple[Expr, Expr] | None:
            if isinstance(t, Prod) and len(t.factors) >= 2:
                return (
                    t.factors[0],
                    Prod(list(t.factors[1:])) if len(t.factors) > 2 else t.factors[1],
                )
            return None

        def _extract_prod_right(t: Expr) -> tuple[Expr, Expr] | None:
            if isinstance(t, Prod) and len(t.factors) >= 2:
                return (
                    t.factors[-1],
                    Prod(list(t.factors[:-1])) if len(t.factors) > 2 else t.factors[0],
                )
            return None

        def _extract_tensor_left(t: Expr) -> tuple[Expr, Expr] | None:
            if isinstance(t, Tensor) and len(t.factors) == 2:
                return t.factors[0], t.factors[1]
            return None

        def _extract_tensor_right(t: Expr) -> tuple[Expr, Expr] | None:
            if isinstance(t, Tensor) and len(t.factors) == 2:
                return t.factors[1], t.factors[0]
            return None

        # Try all factorization strategies
        strategies: list[
            tuple[
                Callable[[Expr], tuple[Expr, Expr] | None],
                Callable[[Expr, Sum], Expr],
                bool,
            ]
        ] = [
            (_extract_prod_left, lambda f, s: Prod([f, s]), False),
            (_extract_prod_right, lambda f, s: Prod([s, f]), False),
            (_extract_tensor_left, lambda f, s: Tensor([f, s]), True),
            (_extract_tensor_right, lambda f, s: Tensor([s, f]), True),
        ]

        did_any_factor = False
        for extract_fn, build_fn, check_scalars in strategies:
            did_factor, current_terms = _try_factor_by_common(
                current_terms, extract_fn, build_fn, check_scalars
            )
            if did_factor:
                did_any_factor = True
                break

        if not did_any_factor:
            break

    if len(current_terms) == 1 and abs(current_terms[0][0] - 1.0) < TOL:
        return current_terms[0][1]
    return Sum(current_terms)


def _try_combine_divisions_in_sum(terms: list[tuple[float, Expr]]) -> Expr | None:
    """Try to combine divisions in a sum by finding common denominators.

    For example: a/x + b/x => (a+b)/x, or 1/x - p(x)/(c*x) => (c - p(x))/(c*x)
    """
    from cobble.expr import Basic, Const, Division, Expr, Poly, Prod, Sum

    # Collect division terms and non-division terms
    divisions: list[tuple[float, Division]] = []
    other_terms: list[tuple[float, Expr]] = []

    for coef, term in terms:
        if isinstance(term, Division):
            divisions.append((coef, term))
        else:
            other_terms.append((coef, term))

    # If at least 2 divisions, try to combine them
    if len(divisions) < 2:
        return None

    # Group by base expression, extracting scalar multiples
    # e.g., A and Poly(A, [0, c]) where the latter is c*A
    def extract_base_and_scale(denom: Expr) -> tuple[Expr | None, float]:
        """Extract (base, scale) such that denom = scale * base."""
        if isinstance(denom, Poly):
            # Check if it's of the form c*x (i.e., [0, c])
            if len(denom.p.coeffs) == 2 and abs(denom.p.coeffs[0]) < TOL:
                return (denom.expr, denom.p.coeffs[1])
        elif isinstance(denom, (Basic, Const)):
            return (denom, 1.0)
        return (None, 1.0)

    base_groups: dict[str, list[tuple[float, Division, float]]] = {}
    ungrouped: list[tuple[float, Division]] = []

    for coef, div in divisions:
        base, scale = extract_base_and_scale(div.denominator)
        if base is not None:
            base_key = repr(base)
            base_groups.setdefault(base_key, []).append((coef, div, scale))
        else:
            ungrouped.append((coef, div))

    combined_divisions: list[tuple[float, Expr]] = []

    for base_key, group in base_groups.items():
        if len(group) == 1:
            combined_divisions.append((group[0][0], group[0][1]))
        else:
            # Multiple divisions sharing the same base
            # Convert to common denominator (use lcm of scales, but for simplicity use product)
            # a/(s1*b) + c/(s2*b) = (a*s2 + c*s1)/(s1*s2*b)
            base_expr = None
            numerator_terms: list[tuple[float, Expr]] = []
            scale_product = 1.0

            for coef, div, scale in group:
                if base_expr is None:
                    # Extract base from first item
                    extracted_base, _ = extract_base_and_scale(div.denominator)
                    base_expr = extracted_base

                # Adjust numerator: a/(s*b) contributes a*(product/s) to numerator
                # But compute the full product first
                scale_product *= scale
            assert base_expr is not None

            for coef, div, scale in group:
                # Multiply this numerator by (scale_product / scale)
                adjustment = scale_product / scale if abs(scale) > TOL else 1.0
                if abs(adjustment - 1.0) < TOL:
                    numerator_terms.append((coef, div.numerator))
                else:
                    numerator_terms.append((coef * adjustment, div.numerator))

            combined_numerator = Sum(numerator_terms).optimize()

            # Combined denominator is scale_product * base_expr
            if abs(scale_product - 1.0) < TOL:
                combined_denom = base_expr
            else:
                combined_denom = Poly(
                    base_expr, Polynomial([0.0, scale_product])
                ).optimize()

            combined_divisions.append(
                (1.0, Division(combined_numerator, combined_denom))
            )

    combined_divisions.extend(ungrouped)

    if len(combined_divisions) < len(divisions):
        all_terms = other_terms + combined_divisions
        if len(all_terms) == 1 and abs(all_terms[0][0] - 1.0) < TOL:
            return all_terms[0][1]
        return Sum(all_terms)

    return None

