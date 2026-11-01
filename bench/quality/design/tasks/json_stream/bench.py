import json
import random
import time


def _records(rng, n):
    levels = ["DEBUG", "INFO", "WARN", "ERROR"]
    words = ["user", "login", "café", "timeout", "retry", "naïve", "path=/v1/items", 'quote"d', "tab\there", "😀"]
    out = []
    for i in range(n):
        rec = {
            "ts": 1_700_000_000.0 + i * 0.137,
            "seq": i,
            "level": rng.choice(levels),
            "msg": " ".join(rng.choice(words) for _ in range(rng.randint(3, 12))),
            "host": f"web-{rng.randint(1, 40):02d}",
            "latency_ms": round(rng.uniform(0.1, 900.0), 3),
            "tags": [rng.choice(words) for _ in range(rng.randint(0, 4))],
            "ok": rng.random() < 0.9,
            "extra": None if rng.random() < 0.5 else {"retries": rng.randint(0, 5), "trace": "%032x" % rng.getrandbits(128)},
        }
        out.append(rec)
    return out


def bench(mod) -> float:
    rng = random.Random(20261004)
    records = _records(rng, 8000)
    parts = []
    for r in records:
        parts.append(json.dumps(r, ensure_ascii=rng.random() < 0.5))
        parts.append(rng.choice(["\n", "\r\n", "", " "]))
    # a few bare scalars between objects, as some producers emit heartbeats
    text = "".join(parts) + "\n7 true\n"
    chunks, i = [], 0
    while i < len(text):
        k = rng.choice([1, 7, 64, 512, 1500, 4096, rng.randint(1, 8192)])
        chunks.append(text[i:i + k])
        i += k

    t0 = time.perf_counter()
    total = []
    for _ in range(3):
        d = mod.JSONStreamDecoder(max_buffer=1_000_000)
        got = []
        for c in chunks:
            got.extend(d.feed(c))
        got.extend(d.close())
        total.append(got)
    elapsed = time.perf_counter() - t0

    for got in total:
        assert len(got) == len(records) + 2, len(got)
        assert got[-2:] == [7, True]
        for k in (0, 1, 1234, 4999, len(records) - 1):
            assert got[k] == records[k], k
        assert sum(g["seq"] for g in got[:-2]) == sum(range(len(records)))
    return elapsed
