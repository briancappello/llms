import math
import random
import time

# (formula, equivalent Python function) pairs, as a spreadsheet column would use them
FORMULAS = [
    ("price * qty * (1 - discount)", lambda r: r["price"] * r["qty"] * (1 - r["discount"])),
    ("if(qty >= 10, price * 0.9, price) * qty + shipping",
     lambda r: (r["price"] * 0.9 if r["qty"] >= 10 else r["price"]) * r["qty"] + r["shipping"]),
    ("round(price * (1 + tax) , 2)", lambda r: float(round(r["price"] * (1 + r["tax"]), 2))),
    ("max(0, revenue - cost) / max(1, units)", lambda r: max(0, r["revenue"] - r["cost"]) / max(1, r["units"])),
    ("sqrt(abs(revenue - cost)) + min(price, shipping, 20)",
     lambda r: math.sqrt(abs(r["revenue"] - r["cost"])) + min(r["price"], r["shipping"], 20)),
    ("(1 + rate / 12) ^ 12 - 1", lambda r: (1 + r["rate"] / 12) ** 12 - 1),
    ("units % 7 + -qty ^ 2 / 100", lambda r: r["units"] % 7 + -(r["qty"] ** 2) / 100),
    ("if(revenue > cost, (revenue - cost) / revenue, 0) * 100",
     lambda r: ((r["revenue"] - r["cost"]) / r["revenue"] if r["revenue"] > r["cost"] else 0) * 100),
    ("(price < 50) + (qty != 3) * 2 + (tax == 0.2) * 4",
     lambda r: (r["price"] < 50) + (r["qty"] != 3) * 2 + (r["tax"] == 0.2) * 4),
    ("abs(price - 50) ^ 0.5 * 2 ^ -1", lambda r: abs(r["price"] - 50) ** 0.5 * 2 ** -1),
]


def _rows(rng, n):
    rows = []
    for _ in range(n):
        rows.append({
            "price": round(rng.uniform(1, 120), 2), "qty": rng.randint(1, 30), "discount": rng.choice([0, 0.05, 0.1]),
            "shipping": rng.choice([0, 4.99, 9.99]), "tax": rng.choice([0.0, 0.07, 0.2]),
            "revenue": rng.uniform(1, 1e5), "cost": rng.uniform(1, 1e5), "units": rng.randint(0, 500),
            "rate": rng.uniform(0.0, 0.2),
        })
    return rows


def _generated_sources(rng, n):
    """Larger machine-written formulas (e.g. imported sheets) that must also compile quickly."""
    names = ["price", "qty", "tax", "units"]
    out = []
    for _ in range(n):
        terms = []
        for _ in range(rng.randint(20, 60)):
            a, b = rng.choice(names), rng.choice(names)
            terms.append(rng.choice([f"{a} * {rng.randint(1, 9)}", f"({a} + {b}) / 2", f"max({a}, {b}, 1)",
                                     f"if({a} > {b}, {a}, -{b})", f"{a} ^ 2 % 11"]))
        out.append(" + ".join(terms))
    return out


def bench(mod) -> float:
    rng = random.Random(4242)
    rows = _rows(rng, 24000)
    gen_src = _generated_sources(rng, 400)

    t0 = time.perf_counter()
    compiled = [mod.compile_expr(src) for src, _ in FORMULAS]
    big = [mod.compile_expr(s) for s in gen_src]
    sums = [0.0] * len(compiled)
    for r in rows:
        for k, e in enumerate(compiled):
            sums[k] += e.evaluate(r)
    big_vals = [e.evaluate(rows[i]) for i, e in enumerate(big)]
    elapsed = time.perf_counter() - t0

    for k, (src, fn) in enumerate(FORMULAS):
        want = sum(float(fn(r)) for r in rows)
        assert math.isclose(sums[k], want, rel_tol=1e-9, abs_tol=1e-6), (src, sums[k], want)
    assert big[0].variables <= {"price", "qty", "tax", "units"}
    assert all(isinstance(v, float) for v in big_vals)
    check = mod.evaluate(gen_src[3], rows[3])
    assert check == big_vals[3]
    return elapsed
