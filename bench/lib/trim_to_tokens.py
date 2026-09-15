#!/usr/bin/env python3
"""Use llama-server's /tokenize endpoint to find the byte-prefix of a corpus
that produces exactly `target` tokens (within a small margin), then emit
JSON suitable for /v1/chat/completions on stdout with that text + a coding
question + a max_tokens setting."""
import json, sys, urllib.request, urllib.error

URL = sys.argv[1]                  # e.g. http://127.0.0.1:8015/tokenize
CORPUS_PATH = sys.argv[2]
TARGET_TOKENS = int(sys.argv[3])
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 128

QUESTION = (
    "\n\n// =============================================================\n"
    "// QUESTION:\n"
    "// You have just read a substantial slice of the llama.cpp codebase.\n"
    "// Identify what you believe is the single most error-prone area in\n"
    "// the code above, name a specific file and function, and explain in\n"
    "// 3-5 sentences why it's brittle. Be concrete and brief.\n"
)

def tokenize(text):
    body = json.dumps({"content": text, "add_special": False}).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return d.get("tokens", [])

with open(CORPUS_PATH, "r", encoding="utf-8", errors="replace") as f:
    corpus = f.read()

# Binary search on byte length to land within ~1% of target tokens for the
# corpus portion only (the question adds a fixed ~120 tokens).
lo, hi = 0, len(corpus)
best_text = ""
best_diff = 10**9
for _ in range(20):
    mid = (lo + hi) // 2
    chunk = corpus[:mid]
    n = len(tokenize(chunk))
    diff = abs(n - TARGET_TOKENS)
    if diff < best_diff:
        best_diff = diff
        best_text = chunk
    if n < TARGET_TOKENS:
        lo = mid + 1
    elif n > TARGET_TOKENS:
        hi = mid - 1
    else:
        break

# Construct the chat request body.
full_text = best_text + QUESTION
payload = {
    "messages": [{"role": "user", "content": full_text}],
    "max_tokens": MAX_TOKENS,
    "stream": False,
    "temperature": 0.0,
}
print(json.dumps(payload))
