#!/usr/bin/env python3
"""Parse a /v1/chat/completions response JSON file and emit tab-separated stats."""
import json, sys

with open(sys.argv[1]) as f:
    d = json.load(f)

t = d.get("timings", {})
u = d.get("usage", {})

pp = t.get("prompt_per_second", 0) or 0
tg = t.get("predicted_per_second", 0) or 0
ptok = u.get("prompt_tokens", "")
ctok = u.get("completion_tokens", "")
prompt_ms = t.get("prompt_ms", 0) or 0
pred_ms = t.get("predicted_ms", 0) or 0
print(f"{pp:.2f}\t{tg:.2f}\t{ptok}\t{ctok}\t{prompt_ms:.0f}\t{pred_ms:.0f}")
